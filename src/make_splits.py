"""
CreatorPulse — Reproducible, leakage-safe train/test splits.

Input (read-only):
    data/processed/training_comments.csv   deduplicated, labelled comments
    data/raw/educational_comments.csv      author / channel metadata

Output (a NEW folder; refuses to overwrite unless --force):
    data/processed/splits_v1/
        assignments.csv     one row per comment: groups, split and CV fold columns
        random_train.csv    \\  split 1: stratified group split
        random_test.csv     /
        video_train.csv     \\  split 2: held-out videos (one per channel)
        video_test.csv      /
        excluded.csv        rows left out of both splits, with the reason
        manifest.json       seed, thresholds, input SHA-256, versions, counts,
                            class balance and leakage checks

Preprocessed columns (text_norm, tokens, lemmas) are recomputed with
src.preprocess and added; `text` is the untouched original.

Leakage control
---------------
A *leakage group* joins comments that must stay on the same side:
  - near-duplicates at LEAK_THRESHOLD (0.70, stricter than the 0.80 used for
    deduplication), using the same normalisation as src.prepare_training_data;
  - comments by the same author (style and copy-paste leak across posts).

Split 1 — "random": StratifiedGroupKFold(5, shuffle, seed) over leakage groups.
  Fold 0 is the test set (~20%). `cv_fold` (0-4) is exported for
  cross-validation with the same guarantees.

Split 2 — "video": whole videos are held out — one per channel, chosen from
  N_DRAWS seeded random draws to best match the overall class balance with
  every class present on both sides. Joining videos into leakage groups would
  chain almost everything into one component (authors comment on several
  videos), so instead any training comment that shares a leakage group with a
  test comment is *purged* (dropped from that split's training set).

Fitting rule: anything learned from data (TF-IDF vocabulary, embeddings,
scalers) must be fitted on the train file only.

Usage:
    python -m src.make_splits
    python -m src.make_splits --seed 7 --out data/processed/splits_seed7
"""

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.prepare_training_data import dedup_key, near_duplicate_groups
from src.preprocess import preprocess_frame

SEED = 42
LEAK_THRESHOLD = 0.70
N_FOLDS = 5
N_DRAWS = 5000

TRAINING_INPUT = PROCESSED_DATA_DIR / "training_comments.csv"
RAW_INPUT = RAW_DATA_DIR / "educational_comments.csv"
DEFAULT_OUT = PROCESSED_DATA_DIR / "splits_v1"
CSV_ENCODING = "utf-8-sig"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _union_components(n: int, keys: list) -> np.ndarray:
    """Connected components where rows sharing any key value are joined."""
    rows, cols = list(range(n)), list(range(n))
    for col in keys:
        first = {}
        for i, v in enumerate(col):
            if v == "" or v is None:
                continue
            if v in first:
                rows.append(first[v]); cols.append(i)
            else:
                first[v] = i
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return connected_components(adj, directed=False)[1]


def leakage_groups(df: pd.DataFrame, threshold: float = LEAK_THRESHOLD) -> np.ndarray:
    near = near_duplicate_groups(df["text"].map(dedup_key), threshold)
    return _union_components(len(df), [list(near), list(df["author_channel_id"])])


def random_group_split(df, groups, seed=SEED, n_folds=N_FOLDS):
    from sklearn.model_selection import StratifiedGroupKFold

    folds = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(df, df["label"], groups)):
        folds[test_idx] = k
    return folds


def _distribution(labels: pd.Series, classes) -> np.ndarray:
    counts = labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float)
    return counts / max(counts.sum(), 1)


def choose_test_videos(df, seed=SEED, n_draws=N_DRAWS, target_frac=0.20):
    """One video per channel; best class-balance match over seeded draws."""
    rng = np.random.default_rng(seed)
    classes = sorted(df["label"].unique())
    overall = _distribution(df["label"], classes)
    by_channel = {c: sorted(v.unique()) for c, v in df.groupby("channel_id")["video_id"]}
    best, best_score = None, np.inf
    for _ in range(n_draws):
        pick = [rng.choice(v) for _, v in sorted(by_channel.items())]
        test = df["video_id"].isin(pick)
        test_labels, train_labels = df.loc[test, "label"], df.loc[~test, "label"]
        if set(test_labels) != set(classes) or set(train_labels) != set(classes):
            continue
        frac = test.mean()
        score = np.abs(_distribution(test_labels, classes) - overall).sum() + 2 * abs(frac - target_frac)
        if score < best_score:
            best, best_score = sorted(pick), score
    if best is None:
        raise RuntimeError("No video draw put every class on both sides; increase --draws.")
    return best


def cross_split_similarity(train_text: pd.Series, test_text: pd.Series):
    """Max char-n-gram cosine between any test and train comment (normalised)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    tr = train_text.map(dedup_key)
    te = test_text.map(dedup_key)
    exact = int(te[te != ""].isin(set(tr[tr != ""])).sum())
    trk = tr[(tr != "") & ~tr.str.startswith("emoji:")]
    tek = te[(te != "") & ~te.str.startswith("emoji:")]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(pd.concat([trk, tek]))
    sim = (vec.transform(tek) @ vec.transform(trk).T).toarray()
    best = sim.max(axis=1) if sim.size else np.array([0.0])
    return {
        "identical_normalised_text_across_split": exact,
        "max_similarity": round(float(best.max()), 3),
        "test_rows_with_train_neighbour_>=0.70": int((best >= 0.70).sum()),
        "test_rows_with_train_neighbour_>=0.60": int((best >= 0.60).sum()),
    }


def describe(name, train, test):
    classes = sorted(set(train["label"]) | set(test["label"]))
    return {
        "train_rows": len(train), "test_rows": len(test),
        "train_labels": train["label"].value_counts().reindex(classes, fill_value=0).to_dict(),
        "test_labels": test["label"].value_counts().reindex(classes, fill_value=0).to_dict(),
        "shared_authors": len(set(train["author_channel_id"]) & set(test["author_channel_id"])),
        "shared_videos": len(set(train["video_id"]) & set(test["video_id"])),
        "shared_leakage_groups": len(set(train["leakage_group"]) & set(test["leakage_group"])),
        "similarity": cross_split_similarity(train["text"], test["text"]),
    }


def build_splits(train_df: pd.DataFrame, raw: pd.DataFrame, seed: int = SEED,
                 threshold: float = LEAK_THRESHOLD, n_draws: int = N_DRAWS):
    meta = raw.set_index("comment_id")[["author_channel_id", "channel_id", "channel_title"]]
    df = train_df.merge(meta, left_on="comment_id", right_index=True, how="left")
    df = preprocess_frame(df)

    invalid = df["tokens"].str.strip() == ""
    excluded = df.loc[invalid, ["comment_id", "video_id", "text", "label"]].assign(
        reason="no text content (empty after preprocessing)")
    df = df.loc[~invalid].reset_index(drop=True)

    df["leakage_group"] = leakage_groups(df, threshold)
    df["cv_fold"] = random_group_split(df, df["leakage_group"], seed)
    df["split_random"] = np.where(df["cv_fold"] == 0, "test", "train")

    test_videos = choose_test_videos(df, seed, n_draws)
    is_test = df["video_id"].isin(test_videos)
    test_groups = set(df.loc[is_test, "leakage_group"])
    purged = ~is_test & df["leakage_group"].isin(test_groups)
    df["split_video"] = np.select([is_test, purged], ["test", "purged"], "train")
    return df, excluded, test_videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe train/test splits.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threshold", type=float, default=LEAK_THRESHOLD)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true", help="Replace the output folder if it exists.")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        if not args.force:
            raise SystemExit(f"[ERROR] {out} already exists. Use --out NEW_DIR or --force.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    train_df = pd.read_csv(TRAINING_INPUT, dtype=str, keep_default_na=False)
    raw = pd.read_csv(RAW_INPUT, dtype=str, keep_default_na=False)
    df, excluded, test_videos = build_splits(train_df, raw, args.seed, args.threshold, args.draws)

    keep = ["comment_id", "video_id", "channel_title", "text", "label", "duplicate_count",
            "text_norm", "tokens", "lemmas", "leakage_group", "cv_fold"]
    write = lambda d, name: d[keep].to_csv(out / name, index=False, encoding=CSV_ENCODING)
    df[["comment_id", "video_id", "channel_title", "label", "leakage_group", "cv_fold",
        "split_random", "split_video"]].to_csv(out / "assignments.csv", index=False, encoding=CSV_ENCODING)
    write(df[df.split_random == "train"], "random_train.csv")
    write(df[df.split_random == "test"], "random_test.csv")
    write(df[df.split_video == "train"], "video_train.csv")
    write(df[df.split_video == "test"], "video_test.csv")
    excluded.to_csv(out / "excluded.csv", index=False, encoding=CSV_ENCODING)

    import nltk, scipy, sklearn
    rand = describe("random", df[df.split_random == "train"], df[df.split_random == "test"])
    vid = describe("video", df[df.split_video == "train"], df[df.split_video == "test"])
    vid["purged_from_train"] = int((df.split_video == "purged").sum())
    vid["test_videos"] = {v: df.loc[df.video_id == v, "channel_title"].iloc[0] for v in test_videos}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": f"python -m src.make_splits --seed {args.seed} --threshold {args.threshold} --draws {args.draws}",
        "seed": args.seed, "leakage_threshold": args.threshold, "n_folds": N_FOLDS, "video_draws": args.draws,
        "inputs_sha256": {str(TRAINING_INPUT.relative_to(TRAINING_INPUT.parents[2])): sha256(TRAINING_INPUT),
                          str(RAW_INPUT.relative_to(RAW_INPUT.parents[2])): sha256(RAW_INPUT)},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "nltk": nltk.__version__},
        "rows": {"input": len(train_df), "excluded": len(excluded), "used": len(df)},
        "leakage_groups": {"count": int(df.leakage_group.nunique()),
                           "largest": int(df.leakage_group.value_counts().iloc[0])},
        "split_random": rand,
        "split_video": vid,
        "cv_fold_sizes": df.cv_fold.value_counts().sort_index().to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=int), encoding="utf-8")

    print(f"Splits written to {out}")
    for name, s in (("random", rand), ("video", vid)):
        print(f"  {name:<6} train={s['train_rows']:>4} test={s['test_rows']:>4} "
              f"shared authors={s['shared_authors']} shared groups={s['shared_leakage_groups']} "
              f"max sim={s['similarity']['max_similarity']}")
    print(f"  video split purged {vid['purged_from_train']} training rows linked to test comments")


if __name__ == "__main__":
    main()
