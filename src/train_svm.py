"""
CreatorPulse — baseline comment-intent classifier: TF-IDF + Linear SVM.

Six classes (the `label_6class` column of data/labelled/comments_labelled.csv):
    Question, Request, Confusion/Problem, Praise, Spam, Other

IMPORTANT — label provenance
----------------------------
The labels were generated automatically by Claude (an AI model) and have NOT
been fully verified by a human. They may contain errors. Every metric this
script reports measures *agreement with those AI-generated labels*, not
accuracy against ground truth. Treat results as provisional until a
human-verified test set exists.

Inputs (read-only; nothing here is modified)
--------------------------------------------
    data/labelled/comments_labelled.csv          label_6class (+ label, text)
    data/processed/splits_v1/video_train.csv     leakage-safe split (primary)
    data/processed/splits_v1/video_test.csv
    data/processed/splits_v1/random_train.csv    leakage-safe split (secondary)
    data/processed/splits_v1/random_test.csv
    data/processed/splits_v1/assignments.csv     cv_fold for grouped 5-fold CV

The splits come from src.make_splits: near-duplicates (char TF-IDF cosine
>= 0.70) and comments by the same author share a `leakage_group`, which never
crosses train/test. The *video* split holds out one whole video per channel
(training comments linked to a test comment are purged), so it is the primary
evaluation. The random split and grouped CV are reported for comparison; CV
also lets every Confusion/Problem example be scored once (out of fold).

Text features
-------------
The pipeline takes the ORIGINAL `text` and applies src.preprocess.normalize
inside TfidfVectorizer, so training and prediction use identical
normalisation (equal to the `text_norm` column; checked at run time):
lowercase, <url>/<user>/<time> placeholders, contractions expanded
(didn't -> did not), Hinglish spelling variants unified (nhi -> nahi), emoji
kept. Tokens come from src.preprocess.tokenize: Devanagari stays whole, "?"
and "!" are tokens, each emoji is a token. No stopword removal and no
lemmatisation, so negations (not, nahi, नहीं), question words and Hinglish
words all survive. The `lemmas` column is not used.

Model: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
       -> LinearSVC(class_weight="balanced").
No oversampling or synthetic duplication; `duplicate_count` is reported but
not used as a sample weight (one flooding account would get 14x weight).

Outputs
-------
    models/creatorpulse_svm.joblib       pipeline fitted on video_train
    reports/classification_report.txt
    reports/confusion_matrix.png         video-split test set
    reports/evaluation_summary.json
    reports/test_predictions.csv         per-comment predictions (video test)

Synthetic training data (optional)
----------------------------------
    python -m src.train_svm --synthetic data/synthetic/synthetic_comments.csv

adds the synthetic rows (src.generate_synthetic) to the TRAINING data of every
evaluation (video split, random split and each CV training fold). Test sets
stay 100% real. Outputs go to models/creatorpulse_svm_synthetic.joblib and
reports/synthetic/ so the real-only baseline is never overwritten.

Usage:
    python -m src.train_svm
    python -m src.train_svm --synthetic data/synthetic/synthetic_comments.csv
"""

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import LABELLED_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.preprocess import normalize, tokenize

CLASSES = ["Question", "Request", "Confusion/Problem", "Praise", "Spam", "Other"]
LABEL_COLUMN = "label_6class"
SEED = 42

LABELLED = LABELLED_DATA_DIR / "comments_labelled.csv"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits_v1"
MODEL_PATH = PROJECT_ROOT / "models" / "creatorpulse_svm.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
CSV_ENCODING = "utf-8-sig"

# Per-class test support below these counts is flagged in every report.
INSUFFICIENT_SUPPORT = 10   # metrics essentially uninformative
LOW_SUPPORT = 30            # metrics unstable (wide confidence interval)

LABEL_PROVENANCE = (
    "Labels were generated automatically by Claude (an AI model) and have not been "
    "fully human-verified; they may contain errors. All metrics measure agreement "
    "with these AI-generated labels, not accuracy against ground truth. A "
    "human-verified test set is needed before claiming real-world accuracy."
)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_pipeline(ngram_range=(1, 2), min_df=2, C=1.0) -> Pipeline:
    """TF-IDF (fitted on whatever data .fit receives) + LinearSVC.

    Input is raw comment text; normalisation and tokenisation are part of the
    pipeline so the saved model needs no separate preprocessing step.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize,   # same as the text_norm column
            tokenizer=tokenize,       # keeps Devanagari, ?, !, emoji, negations
            token_pattern=None,
            lowercase=False,          # normalize() already lower-cases
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )),
        ("svm", LinearSVC(class_weight="balanced", C=C, random_state=SEED, max_iter=10000)),
    ])


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)


def load_labels(path: Path = LABELLED) -> pd.DataFrame:
    df = read_csv(path)
    missing = [c for c in ("comment_id", "text", "label", LABEL_COLUMN) if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] {path} is missing columns: {missing}")
    return df


def audit_labels(labels: pd.DataFrame) -> dict:
    """Missing labels, duplicate IDs, unknown classes and class balance."""
    lab = labels[LABEL_COLUMN].str.strip()
    unknown = sorted(set(lab) - set(CLASSES) - {""})
    return {
        "rows": len(labels),
        "missing_label_6class": int((lab == "").sum()),
        "duplicate_comment_ids": int(labels["comment_id"].duplicated().sum()),
        "unknown_labels": unknown,
        "class_distribution": lab.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
    }


def attach_labels(split: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Add label_6class to a split file and verify it matches the labelled file."""
    lab = labels.set_index("comment_id")
    out = split.copy()
    missing = ~out["comment_id"].isin(lab.index)
    if missing.any():
        raise SystemExit(f"[ERROR] {int(missing.sum())} split rows are not in {LABELLED.name}")
    out[LABEL_COLUMN] = out["comment_id"].map(lab[LABEL_COLUMN])
    if (out["text"] != out["comment_id"].map(lab["text"])).any():
        raise SystemExit("[ERROR] split text differs from the labelled file")
    if (out["label"] != out["comment_id"].map(lab["label"])).any():
        raise SystemExit("[ERROR] split 5-class label differs from the labelled file")
    return out


def load_split(name: str, labels: pd.DataFrame):
    train = attach_labels(read_csv(SPLITS_DIR / f"{name}_train.csv"), labels)
    test = attach_labels(read_csv(SPLITS_DIR / f"{name}_test.csv"), labels)
    return train, test


def leakage_checks(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    tr_norm = set(train["text"].map(normalize)) - {""}
    checks = {
        "shared_comment_ids": len(set(train.comment_id) & set(test.comment_id)),
        "shared_leakage_groups": len(set(train.leakage_group) & set(test.leakage_group)),
        "identical_normalised_text": int(test["text"].map(normalize).isin(tr_norm).sum()),
        "shared_videos": len(set(train.video_id) & set(test.video_id)),
    }
    if checks["shared_comment_ids"] or checks["shared_leakage_groups"]:
        raise SystemExit(f"[ERROR] train/test leakage detected: {checks}")
    return checks


def distribution(y) -> dict:
    return pd.Series(y).value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict()


def duplicate_count_summary(df: pd.DataFrame) -> dict:
    dc = df["duplicate_count"].astype(int)
    by = df.assign(dc=dc).groupby(LABEL_COLUMN)["dc"]
    return {
        "note": ("Number of original comments each kept row stands for after near-duplicate "
                 "collapsing. Descriptive only - NOT used as a sample weight, because it mixes "
                 "'many people said this' with 'one account flooded this'."),
        "rows_with_duplicates": int((dc > 1).sum()),
        "total_comments_represented": int(dc.sum()),
        "max": int(dc.max()),
        "per_class_rows": by.size().reindex(CLASSES, fill_value=0).astype(int).to_dict(),
        "per_class_comments_represented": by.sum().reindex(CLASSES, fill_value=0).astype(int).to_dict(),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n (None when n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def bootstrap_ci(y_true, y_pred, metric, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(metric(y_true[idx], y_pred[idx]))
    return [round(float(np.percentile(stats, 2.5)), 3), round(float(np.percentile(stats, 97.5)), 3)]


def reliability(support: int) -> str:
    if support < INSUFFICIENT_SUPPORT:
        return "insufficient"
    if support < LOW_SUPPORT:
        return "low"
    return "adequate"


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)


def evaluate(y_true, y_pred, with_ci: bool = True) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = sum(t == c and q == c for t, q in zip(y_true, y_pred))
        per_class[c] = {
            "precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
            "f1": round(float(f[i]), 3), "support": int(s[i]),
            "predicted_count": int(sum(q == c for q in y_pred)),
            "true_positives": tp,
            "recall_95ci_wilson": wilson_interval(tp, int(s[i])),
            "reliability": reliability(int(s[i])),
        }
    adequate = [c for c in CLASSES if per_class[c]["support"] >= INSUFFICIENT_SUPPORT]
    out = {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(macro_f1(y_true, y_pred), 3),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=CLASSES, average="weighted",
                                      zero_division=0), 3),
        "macro_f1_classes_with_support_ge_10": {
            "classes": adequate,
            "value": round(f1_score(y_true, y_pred, labels=adequate, average="macro",
                                    zero_division=0), 3),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": CLASSES,
                             "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()},
    }
    if with_ci:
        out["accuracy_95ci_bootstrap"] = bootstrap_ci(y_true, y_pred, accuracy_score)
        out["macro_f1_95ci_bootstrap"] = bootstrap_ci(y_true, y_pred, macro_f1)
    return out


def majority_baseline(y_train, y_test) -> dict:
    majority = pd.Series(y_train).value_counts().idxmax()
    pred = [majority] * len(y_test)
    return {"predicts": majority,
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "macro_f1": round(macro_f1(list(y_test), pred), 3)}


def load_synthetic(path) -> pd.DataFrame:
    """Synthetic rows (text + label_6class) for training only."""
    df = read_csv(Path(path))
    missing = [c for c in ("comment_id", "text", LABEL_COLUMN) if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] {path} is missing columns: {missing}")
    unknown = sorted(set(df[LABEL_COLUMN]) - set(CLASSES))
    if unknown:
        raise SystemExit(f"[ERROR] {path} has unknown labels: {unknown}")
    if not df["comment_id"].str.startswith("syn_").all():
        raise SystemExit("[ERROR] synthetic comment_id values must start with 'syn_'")
    return df


def with_extra(train: pd.DataFrame, extra) -> tuple:
    """(texts, labels) for fitting: real training rows plus optional synthetic rows."""
    if extra is None or len(extra) == 0:
        return train["text"], train[LABEL_COLUMN]
    return (pd.concat([train["text"], extra["text"]], ignore_index=True),
            pd.concat([train[LABEL_COLUMN], extra[LABEL_COLUMN]], ignore_index=True))


def grouped_cv(labels: pd.DataFrame, extra=None) -> dict:
    """5-fold CV on splits_v1 cv_fold (leakage groups never cross folds).

    Every comment is predicted once by a model that never saw it or its
    near-duplicates/same-author comments, so all 19 Confusion/Problem examples
    get scored. Folds share videos, so this is closer to the random split.
    """
    rows = pd.concat([read_csv(SPLITS_DIR / "random_train.csv"),
                      read_csv(SPLITS_DIR / "random_test.csv")], ignore_index=True)
    rows = attach_labels(rows, labels)
    oof = pd.Series(index=rows.index, dtype=object)
    fold_scores = []
    for k in sorted(rows["cv_fold"].unique()):
        te = rows["cv_fold"] == k
        model = build_pipeline().fit(*with_extra(rows.loc[~te], extra))
        oof[te] = model.predict(rows.loc[te, "text"])
        fold_scores.append({
            "fold": int(k), "n_test": int(te.sum()),
            "accuracy": round(accuracy_score(rows.loc[te, LABEL_COLUMN], oof[te]), 3),
            "macro_f1": round(macro_f1(list(rows.loc[te, LABEL_COLUMN]), list(oof[te])), 3),
        })
    pooled = evaluate(rows[LABEL_COLUMN], oof, with_ci=False)
    return {
        "description": "Pooled out-of-fold predictions, 5 folds from splits_v1 cv_fold",
        "folds": fold_scores,
        "macro_f1_mean_sd": [round(float(np.mean([f["macro_f1"] for f in fold_scores])), 3),
                             round(float(np.std([f["macro_f1"] for f in fold_scores])), 3)],
        "pooled": pooled,
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # One-hue sequential ramp (light = near zero, dark = most of the row).
    cmap = LinearSegmentedColormap.from_list(
        "blue", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    row_totals = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0)

    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=150)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if share[i, j] > 0.55 else "#1f2328"
            pct = f"{share[i, j]:.0%}" if row_totals[i, 0] else "–"
            ax.text(j, i, f"{cm[i, j]}\n{pct}", ha="center", va="center",
                    fontsize=9.5, color=colour, fontweight="bold" if i == j else "normal")
    ticks = [f"{c}\n(n={int(row_totals[k, 0])})" for k, c in enumerate(CLASSES)]
    ax.set_xticks(range(len(CLASSES)), [c.replace("/", "/\n") for c in CLASSES], fontsize=9)
    ax.set_yticks(range(len(CLASSES)), ticks, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10, color="#57606a")
    ax.set_ylabel("Label in dataset (AI-generated)", fontsize=10, color="#57606a")
    ax.set_title(title, fontsize=11.5, loc="left", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(CLASSES)), minor=True)
    ax.set_yticks(np.arange(-.5, len(CLASSES)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    fig.text(0.01, 0.01, "Cell: count and share of the true-label row. Colour = row share. "
             "Labels are AI-generated; rows with n < 10 are not reliable.",
             fontsize=8, color="#57606a")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fmt_metrics_block(name: str, ev: dict, y_true, y_pred) -> str:
    lines = [f"=== {name} ===", f"Test comments: {ev['n']}"]
    if "accuracy_95ci_bootstrap" in ev:
        lines += [
            f"Accuracy:     {ev['accuracy']:.3f}  (95% bootstrap CI {ev['accuracy_95ci_bootstrap']})",
            f"Macro F1:     {ev['macro_f1']:.3f}  (95% bootstrap CI {ev['macro_f1_95ci_bootstrap']})",
        ]
    else:
        lines += [f"Accuracy:     {ev['accuracy']:.3f}", f"Macro F1:     {ev['macro_f1']:.3f}"]
    m = ev["macro_f1_classes_with_support_ge_10"]
    lines += [f"Weighted F1:  {ev['weighted_f1']:.3f}",
              f"Macro F1 over classes with >= 10 test examples ({', '.join(m['classes'])}): {m['value']:.3f}",
              "",
              classification_report(list(y_true), list(y_pred), labels=CLASSES, digits=3, zero_division=0),
              "Reliability (by test support; recall 95% Wilson CI):"]
    for c in CLASSES:
        pc = ev["per_class"][c]
        lines.append(f"  {c:<18} support={pc['support']:>3}  recall CI={pc['recall_95ci_wilson']}  -> {pc['reliability']}")
    cm = ev["confusion_matrix"]["rows_true_cols_pred"]
    short = ["Quest", "Reqst", "Conf", "Praise", "Spam", "Other"]
    lines += ["", "Confusion matrix (rows = dataset label, cols = predicted):",
              " " * 19 + "".join(f"{s:>7}" for s in short)]
    for c, row in zip(CLASSES, cm):
        lines.append(f"  {c:<17}" + "".join(f"{v:>7}" for v in row))
    return "\n".join(lines)


def class_notes(primary: dict, cv: dict) -> list:
    notes = []
    for c in CLASSES:
        n_test = primary["per_class"][c]["support"]
        if n_test < INSUFFICIENT_SUPPORT:
            pooled = cv["pooled"]["per_class"][c]
            notes.append(
                f"{c}: only {n_test} test example(s) on the video split, so its test precision/"
                f"recall/F1 cannot be evaluated reliably (recall 95% CI "
                f"{primary['per_class'][c]['recall_95ci_wilson']}). Pooled over 5-fold grouped CV "
                f"({pooled['support']} examples): precision {pooled['precision']}, recall "
                f"{pooled['recall']}, F1 {pooled['f1']} - still a small sample.")
    return notes


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + LinearSVC baseline.")
    parser.add_argument("--model-out", default=str(MODEL_PATH))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--synthetic", default=None,
                        help="CSV of synthetic comments to add to the TRAINING data only")
    args = parser.parse_args()
    synthetic = load_synthetic(args.synthetic) if args.synthetic else None
    if synthetic is not None:
        if args.model_out == str(MODEL_PATH):
            args.model_out = str(MODEL_PATH.with_name("creatorpulse_svm_synthetic.joblib"))
        if args.reports_dir == str(REPORTS_DIR):
            args.reports_dir = str(REPORTS_DIR / "synthetic")
    model_out, reports = Path(args.model_out), Path(args.reports_dir)

    labels = load_labels()
    audit = audit_labels(labels)
    if audit["missing_label_6class"] or audit["duplicate_comment_ids"] or audit["unknown_labels"]:
        raise SystemExit(f"[ERROR] label audit failed: {audit}")

    # Primary split: held-out videos.
    train, test = load_split("video", labels)
    leak = leakage_checks(train, test)
    norm_ok = bool((train["text"].map(normalize) == train["text_norm"]).all()
                   and (test["text"].map(normalize) == test["text_norm"]).all())

    model = build_pipeline().fit(*with_extra(train, synthetic))
    pred = model.predict(test["text"])
    primary = evaluate(test[LABEL_COLUMN], pred)
    baseline = majority_baseline(train[LABEL_COLUMN], test[LABEL_COLUMN])

    # Secondary: random stratified-group split and grouped 5-fold CV.
    r_train, r_test = load_split("random", labels)
    r_leak = leakage_checks(r_train, r_test)
    r_pred = build_pipeline().fit(*with_extra(r_train, synthetic)).predict(r_test["text"])
    secondary = evaluate(r_test[LABEL_COLUMN], r_pred)
    cv = grouped_cv(labels, synthetic)

    # Save model and reports.
    model_out.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    plot_confusion_matrix(np.array(primary["confusion_matrix"]["rows_true_cols_pred"]),
                          reports / "confusion_matrix.png",
                          f"TF-IDF + LinearSVC — held-out videos (n={len(test)})")

    test.assign(predicted=pred, correct=pred == test[LABEL_COLUMN].to_numpy())[
        ["comment_id", "video_id", "channel_title", "text", LABEL_COLUMN, "predicted", "correct"]
    ].to_csv(reports / "test_predictions.csv", index=False, encoding=CSV_ENCODING)

    vocab = len(model.named_steps["tfidf"].vocabulary_)
    notes = class_notes(primary, cv)
    import sklearn
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "path": str(model_out.relative_to(PROJECT_ROOT)) if model_out.is_relative_to(PROJECT_ROOT) else str(model_out),
            "pipeline": "TfidfVectorizer -> LinearSVC",
            "trained_on": "data/processed/splits_v1/video_train.csv"
                          + (f" + {len(synthetic)} synthetic rows ({args.synthetic})" if synthetic is not None else ""),
            "synthetic_training_rows": (distribution(synthetic[LABEL_COLUMN]) if synthetic is not None else None),
            "input": "raw comment text (normalised inside the pipeline)",
            "tfidf": {"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True,
                      "preprocessor": "src.preprocess.normalize (== text_norm column)",
                      "tokenizer": "src.preprocess.tokenize", "stopwords": None,
                      "lemmatisation": False, "vocabulary_size": vocab},
            "svm": {"class_weight": "balanced", "C": 1.0, "random_state": SEED},
            "resampling": ("synthetic template-generated rows added to training data only; "
                           "all test sets are real comments" if synthetic is not None
                           else "none (no oversampling or synthetic duplication)"),
            "sample_weights": "none (duplicate_count not used)",
        },
        "label_provenance": LABEL_PROVENANCE,
        "label_column": LABEL_COLUMN,
        "classes": CLASSES,
        "dataset_audit": {
            "labelled_file": audit,
            "training_rows_after_dedup_and_exclusion": int(len(r_train) + len(r_test)),
            "duplicate_count": duplicate_count_summary(pd.concat([r_train, r_test])),
            "text_norm_matches_pipeline_normalisation": norm_ok,
        },
        "primary_evaluation_video_split": {
            "description": ("One whole video per channel held out (5 videos); training comments "
                            "sharing a leakage group with any test comment were purged."),
            "train_distribution": distribution(train[LABEL_COLUMN]),
            "test_distribution": distribution(test[LABEL_COLUMN]),
            "leakage_checks": leak,
            "majority_class_baseline": baseline,
            **primary,
        },
        "secondary_evaluation_random_split": {
            "description": "StratifiedGroupKFold fold 0 (videos shared, leakage groups not shared).",
            "train_distribution": distribution(r_train[LABEL_COLUMN]),
            "test_distribution": distribution(r_test[LABEL_COLUMN]),
            "leakage_checks": r_leak,
            **secondary,
        },
        "grouped_5fold_cv": cv,
        "class_reliability_notes": notes,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    (reports / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    header = [
        "CreatorPulse — TF-IDF + LinearSVC baseline (6 classes, label_6class)",
        f"Generated: {summary['created_utc']}",
        "",
        "LABEL PROVENANCE: " + LABEL_PROVENANCE,
        "",
        "Model trained on splits_v1/video_train.csv"
        + (f" PLUS {len(synthetic)} synthetic rows (training only; every test set is real)"
           if synthetic is not None else "") + "; features fitted on training data only.",
        f"TF-IDF word 1-2 grams, min_df=2, sublinear_tf, vocabulary={vocab}; "
        "LinearSVC class_weight='balanced'.",
        "",
        "Class distribution (train / test, video split):",
    ] + [f"  {c:<18} {summary['primary_evaluation_video_split']['train_distribution'][c]:>4} / "
         f"{summary['primary_evaluation_video_split']['test_distribution'][c]:>3}" for c in CLASSES] + [
        "",
        f"Majority-class baseline (always '{baseline['predicts']}'): accuracy {baseline['accuracy']:.3f}, "
        f"macro F1 {baseline['macro_f1']:.3f}",
        "",
    ]
    cv_text = ("=== Grouped 5-fold CV (pooled out-of-fold, all 899 comments) ===\n"
               f"Per-fold macro F1 mean {cv['macro_f1_mean_sd'][0]:.3f} (sd {cv['macro_f1_mean_sd'][1]:.3f}); "
               f"pooled accuracy {cv['pooled']['accuracy']:.3f}, pooled macro F1 {cv['pooled']['macro_f1']:.3f}, "
               f"pooled weighted F1 {cv['pooled']['weighted_f1']:.3f}\n"
               + "\n".join(f"  {c:<18} P={cv['pooled']['per_class'][c]['precision']:.3f} "
                           f"R={cv['pooled']['per_class'][c]['recall']:.3f} F1={cv['pooled']['per_class'][c]['f1']:.3f} "
                           f"n={cv['pooled']['per_class'][c]['support']}" for c in CLASSES))
    text = "\n".join(header) + "\n".join([
        fmt_metrics_block("PRIMARY: held-out videos (splits_v1 video split)", primary,
                          test[LABEL_COLUMN], pred),
        "", fmt_metrics_block("SECONDARY: random stratified-group split", secondary,
                              r_test[LABEL_COLUMN], r_pred),
        "", cv_text, "", "Classes that cannot be evaluated reliably:",
        *[f"  - {n}" for n in notes], "",
    ])
    (reports / "classification_report.txt").write_text(text, encoding="utf-8")

    print(text)
    print(f"Model saved to {model_out}")
    print(f"Reports written to {reports}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
