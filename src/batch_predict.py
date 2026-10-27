"""
CreatorPulse — Stage 1: batch prediction on collected YouTube comments.

Loads the saved TF-IDF + LinearSVC pipeline and predicts one of the six
intent categories for every usable comment in the raw comment CSVs.

The pipeline is used as saved: its TfidfVectorizer already contains
src.preprocess.normalize + src.preprocess.tokenize, so new comments get
exactly the training-time preprocessing. Only .predict / .decision_function
are called here; no vectoriser or model is ever fitted on prediction data.

Inputs (read-only)
    data/raw/*.csv                 every file with comment_id + text columns
                                   (educational_comments.csv, comments_<id>.csv)
    data/raw/channel_videos_*.csv  used only to fill missing video titles/dates
    data/labelled/comments_labelled.csv          to flag already-labelled comments
    data/processed/splits_v1/video_train.csv     to flag comments the model trained on

Outputs
    data/processed/comments_predicted.csv   one row per predicted comment
    data/processed/comments_excluded.csv    rows that could not be predicted, with reason
    data/processed/category_summary.csv     counts and percentages per category
    data/processed/batch_predict_stats.json run statistics

Exclusions: missing comment_id, missing/empty text, text with no usable tokens
after normalisation (e.g. only punctuation), and repeated comment_ids (the
first occurrence is kept).

Model choice: models/creatorpulse_svm_synthetic.joblib is used when present
(real + synthetic training data; better on the real held-out comments, see
reports/synthetic_comparison.md), otherwise models/creatorpulse_svm.joblib.
Override with --model.

Usage:
    python -m src.batch_predict
    python -m src.batch_predict --model models/creatorpulse_svm.joblib
    python -m src.batch_predict --inputs data/raw/comments_WD6R-2xQv-E.csv
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELLED_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from src.predict import ModelLoadError, load_model  # noqa: E402
from src.preprocess import normalize, tokenize  # noqa: E402

CLASSES = ["Question", "Request", "Confusion/Problem", "Praise", "Spam", "Other"]
CSV_ENCODING = "utf-8-sig"
MODELS_DIR = PROJECT_ROOT / "models"
SYNTHETIC_MODEL = MODELS_DIR / "creatorpulse_svm_synthetic.joblib"
BASELINE_MODEL = MODELS_DIR / "creatorpulse_svm.joblib"
LABELLED = LABELLED_DATA_DIR / "comments_labelled.csv"
TRAIN_SPLIT = PROCESSED_DATA_DIR / "splits_v1" / "video_train.csv"

OUT_PREDICTED = PROCESSED_DATA_DIR / "comments_predicted.csv"
OUT_EXCLUDED = PROCESSED_DATA_DIR / "comments_excluded.csv"
OUT_SUMMARY = PROCESSED_DATA_DIR / "category_summary.csv"
OUT_STATS = PROCESSED_DATA_DIR / "batch_predict_stats.json"

# Columns carried through when present. Anything missing becomes empty and is
# listed in `metadata_missing`.
META_COLUMNS = ["video_id", "channel_id", "channel_title", "video_title", "video_published_at",
                "published_at", "author_channel_id", "like_count", "reply_count", "creator_replied"]
REQUIRED_META = ["video_id", "channel_title", "video_title", "published_at", "author_channel_id"]
LOW_CONFIDENCE_MARGIN = 0.25   # SVM score gap between best and runner-up


def default_model_path() -> Path:
    return SYNTHETIC_MODEL if SYNTHETIC_MODEL.exists() else BASELINE_MODEL


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)


def find_comment_files(raw_dir: Path = RAW_DATA_DIR) -> List[Path]:
    """Raw CSVs that hold comments (have comment_id and text), sorted by name."""
    files = []
    for p in sorted(raw_dir.glob("*.csv")):
        try:
            cols = set(pd.read_csv(p, nrows=0, encoding=CSV_ENCODING).columns)
        except Exception:
            continue
        if {"comment_id", "text"} <= cols:
            files.append(p)
    return files


def load_video_catalogue(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """video_id -> title, published_at from channel_videos_*.csv (may be empty)."""
    frames = []
    for p in sorted(raw_dir.glob("channel_videos_*.csv")):
        df = read_csv(p)
        if {"video_id", "title"} <= set(df.columns):
            frames.append(df.reindex(columns=["video_id", "title", "published_at"]).fillna(""))
    if not frames:
        return pd.DataFrame(columns=["video_id", "title", "published_at"])
    return pd.concat(frames).drop_duplicates("video_id")


def load_comments(files: List[Path], catalogue: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Union of all comment files with uniform columns. Raw files are only read."""
    frames = []
    for p in files:
        df = read_csv(p)
        df["source_file"] = p.name
        df["source_row"] = np.arange(len(df)) + 2   # 1-based, after header
        frames.append(df)
    if not frames:
        raise SystemExit("[ERROR] no comment CSV files found (need comment_id and text columns)")
    df = pd.concat(frames, ignore_index=True)
    for col in ["comment_id", "text"] + META_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    for col in ["comment_id", "video_id", "channel_id", "author_channel_id"]:
        df[col] = df[col].astype(str).str.strip()

    if catalogue is not None and len(catalogue):
        cat = catalogue.set_index("video_id")
        need_title = (df["video_title"] == "") & df["video_id"].isin(cat.index)
        df.loc[need_title, "video_title"] = df.loc[need_title, "video_id"].map(cat["title"])
        need_date = (df["video_published_at"] == "") & df["video_id"].isin(cat.index)
        df.loc[need_date, "video_published_at"] = df.loc[need_date, "video_id"].map(cat["published_at"])

    df["metadata_missing"] = df[REQUIRED_META].apply(
        lambda r: ";".join(c for c in REQUIRED_META if str(r[c]).strip() == ""), axis=1)
    df.loc[df["channel_title"] == "", "channel_title"] = "unknown"
    df.loc[df["video_title"] == "", "video_title"] = "unknown"
    ts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    bad_ts = ts.isna() & (df["published_at"] != "")
    df.loc[bad_ts, "metadata_missing"] = df.loc[bad_ts, "metadata_missing"].map(
        lambda s: (s + ";" if s else "") + "published_at_unparseable")
    return df


def split_usable(df: pd.DataFrame):
    """Return (usable, excluded) with an `excluded_reason` on excluded rows."""
    reason = pd.Series("", index=df.index, dtype=object)
    reason[df["comment_id"] == ""] = "missing_comment_id"
    empty = df["text"].astype(str).str.strip() == ""
    reason[(reason == "") & empty] = "empty_text"
    no_tokens = df["text"].map(lambda t: not tokenize(normalize(t)) if str(t).strip() else False)
    reason[(reason == "") & no_tokens] = "no_usable_tokens"
    dup = df["comment_id"].duplicated(keep="first") & (df["comment_id"] != "")
    reason[(reason == "") & dup] = "duplicate_comment_id"
    # The channel's own comments (e.g. a pinned note by the creator) are not audience feedback.
    if {"author_channel_id", "channel_id"} <= set(df.columns):
        own = (df["author_channel_id"].astype(str) != "") & (df["author_channel_id"] == df["channel_id"])
        reason[(reason == "") & own] = "creator_comment"
    excluded = df[reason != ""].assign(excluded_reason=reason[reason != ""])
    return df[reason == ""].copy(), excluded


def predict_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    """Predict with the saved pipeline (transform only, never fit)."""
    texts = df["text"].tolist()
    out = df.copy()
    if not texts:
        out["predicted_category"] = []
        return out
    pred = model.predict(texts)
    scores = model.decision_function(texts)
    classes = [str(c) for c in model.classes_]
    order = np.sort(scores, axis=1)
    out["predicted_category"] = pred.astype(str)
    out["prediction_margin"] = np.round(order[:, -1] - order[:, -2], 3)
    out["low_confidence"] = out["prediction_margin"] < LOW_CONFIDENCE_MARGIN
    for i, c in enumerate(classes):
        out[f"score_{c}"] = np.round(scores[:, i], 3)
    tfidf = model.named_steps["tfidf"] if hasattr(model, "named_steps") else None
    if tfidf is not None:
        out["known_features"] = tfidf.transform(texts).getnnz(axis=1)
    return out


def training_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Mark comments already labelled / used to train the model (predictions there are optimistic)."""
    out = df.copy()
    out["dataset_label"] = ""
    out["in_labelled_set"] = False
    out["seen_in_training"] = False
    if LABELLED.exists():
        lab = read_csv(LABELLED)
        col = "label_6class" if "label_6class" in lab.columns else "label"
        m = lab.set_index("comment_id")[col]
        out["in_labelled_set"] = out["comment_id"].isin(m.index)
        out.loc[out["in_labelled_set"], "dataset_label"] = out.loc[out["in_labelled_set"], "comment_id"].map(m)
    if TRAIN_SPLIT.exists():
        out["seen_in_training"] = out["comment_id"].isin(set(read_csv(TRAIN_SPLIT)["comment_id"]))
    return out


def category_summary(pred: pd.DataFrame) -> pd.DataFrame:
    n = len(pred)
    unseen = pred[~pred["seen_in_training"]]
    rows = []
    for c in CLASSES:
        k = int((pred["predicted_category"] == c).sum())
        k2 = int((unseen["predicted_category"] == c).sum())
        rows.append({"category": c, "comments": k, "percent": round(100 * k / n, 2) if n else 0.0,
                     "comments_not_seen_in_training": k2,
                     "percent_not_seen_in_training": round(100 * k2 / len(unseen), 2) if len(unseen) else 0.0})
    return pd.DataFrame(rows)


ORDERED_COLUMNS = (["comment_id", "video_id", "channel_id", "channel_title", "video_title",
                    "video_published_at", "published_at", "author_channel_id", "like_count",
                    "reply_count", "creator_replied", "text", "predicted_category",
                    "prediction_margin", "low_confidence", "known_features"]
                   + [f"score_{c}" for c in CLASSES]
                   + ["dataset_label", "in_labelled_set", "seen_in_training", "metadata_missing",
                      "source_file", "source_row"])


def run(model_path=None, inputs=None, out_dir: Path = PROCESSED_DATA_DIR, raw_dir: Path = RAW_DATA_DIR) -> dict:
    model_path = Path(model_path) if model_path else default_model_path()
    model = load_model(model_path)
    missing_classes = set(CLASSES) - {str(c) for c in model.classes_}
    if missing_classes:
        raise ModelLoadError(f"model is missing classes {sorted(missing_classes)}")
    files = [Path(p) for p in inputs] if inputs else find_comment_files(raw_dir)
    df = load_comments(files, load_video_catalogue(raw_dir))
    usable, excluded = split_usable(df)
    pred = training_flags(predict_frame(model, usable))
    pred = pred[[c for c in ORDERED_COLUMNS if c in pred.columns]]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / OUT_PREDICTED.name, index=False, encoding=CSV_ENCODING)
    excluded[["comment_id", "video_id", "text", "excluded_reason", "source_file", "source_row"]].to_csv(
        out_dir / OUT_EXCLUDED.name, index=False, encoding=CSV_ENCODING)
    summary = category_summary(pred)
    summary.to_csv(out_dir / OUT_SUMMARY.name, index=False, encoding=CSV_ENCODING)

    labelled = pred[pred["in_labelled_set"]]
    held_out = labelled[~labelled["seen_in_training"]]
    stats = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"path": str(model_path.relative_to(PROJECT_ROOT)) if model_path.is_relative_to(PROJECT_ROOT)
                  else str(model_path), "sha256": sha256(model_path)},
        "input_files": [{"file": p.name, "rows": int((df["source_file"] == p.name).sum())} for p in files],
        "input_rows": int(len(df)),
        "predicted": int(len(pred)),
        "excluded": int(len(excluded)),
        "excluded_by_reason": excluded["excluded_reason"].value_counts().to_dict(),
        "low_confidence_predictions": int(pred["low_confidence"].sum()),
        "comments_with_missing_metadata": int((pred["metadata_missing"] != "").sum()),
        "missing_metadata_fields": pd.Series(
            [f for s in pred["metadata_missing"] for f in s.split(";") if f]).value_counts().to_dict(),
        "in_labelled_set": int(len(labelled)),
        "seen_in_training": int(pred["seen_in_training"].sum()),
        "agreement_with_dataset_labels": {
            "note": ("Dataset labels are AI-generated. Comments seen in training agree almost by "
                     "construction; only the not-seen ones are a fair check."),
            "seen_in_training": round(float((labelled[labelled["seen_in_training"]]["predicted_category"]
                                             == labelled[labelled["seen_in_training"]]["dataset_label"]).mean()), 3)
            if labelled["seen_in_training"].any() else None,
            "labelled_not_seen_in_training": {
                "n": int(len(held_out)),
                "accuracy": round(float((held_out["predicted_category"] == held_out["dataset_label"]).mean()), 3)
                if len(held_out) else None},
        },
        "category_counts": dict(zip(summary["category"], summary["comments"])),
    }
    (out_dir / OUT_STATS.name).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Predict intent categories for collected comments.")
    parser.add_argument("--model", default=None, help="saved pipeline (default: synthetic model if present)")
    parser.add_argument("--inputs", nargs="*", default=None, help="comment CSVs (default: all in data/raw)")
    parser.add_argument("--out-dir", default=str(PROCESSED_DATA_DIR))
    args = parser.parse_args(argv)
    try:
        stats = run(args.model, args.inputs, Path(args.out_dir))
    except ModelLoadError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
