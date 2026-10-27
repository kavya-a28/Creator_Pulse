"""
CreatorPulse — does synthetic training data help on REAL comments?

Trains the same TF-IDF + LinearSVC pipeline (src.train_svm.build_pipeline)
under several training-data settings and scores every one on the same real
test data. Synthetic rows only ever enter TRAINING data.

Settings
    real only                  splits_v1 training rows (the current baseline)
    real + N synthetic/class   N in 250, 500, 1000, all (1667)
    synthetic only             no real training rows at all

Evaluations (all real comments)
    video split   183 comments from 5 held-out videos (primary)
    random split  180 comments
    grouped CV    all 899 comments, pooled out-of-fold (every Confusion/Problem
                  and Spam example scored once)
    CV, unseen    the grouped-CV comments NOT listed in
                  data/synthetic/viewed_while_writing_templates.csv. The
                  person writing the templates read those 92 comments, so
                  templates may echo them; the unseen subset is the cleaner
                  estimate of what synthetic data adds.

Also reports a paired bootstrap of the macro-F1 difference (real+all synthetic
minus real only) on the video test set, and what happens when a model trained
and tested on synthetic data is scored (to show why that number is
meaningless).

Outputs
    reports/synthetic_comparison.json
    reports/synthetic_comparison.md

Usage:
    python -m src.compare_synthetic
"""

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROJECT_ROOT
from src.train_svm import (CLASSES, LABEL_COLUMN, SEED, SPLITS_DIR, accuracy_score, attach_labels,
                           build_pipeline, evaluate, load_labels, load_split, load_synthetic,
                           macro_f1, read_csv, with_extra)

SYNTHETIC = DATA_DIR / "synthetic" / "synthetic_comments.csv"
VIEWED = DATA_DIR / "synthetic" / "viewed_while_writing_templates.csv"
REPORTS = PROJECT_ROOT / "reports"
SIZES = [250, 500, 1000, None]   # per class; None = all rows


def subsample(syn: pd.DataFrame, per_class):
    if per_class is None:
        return syn
    parts = [g.sample(min(per_class, len(g)), random_state=SEED)
             for _, g in syn.groupby(LABEL_COLUMN, sort=True)]
    return pd.concat(parts).sort_values("comment_id")


def paired_bootstrap_diff(y, pred_a, pred_b, n_boot=2000, seed=SEED):
    """Macro-F1(b) - macro-F1(a) on the same resampled test rows."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(macro_f1(y[i], b[i]) - macro_f1(y[i], a[i]))
    diffs = np.array(diffs)
    return {"mean_diff": round(float(diffs.mean()), 3),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 3),
                     round(float(np.percentile(diffs, 97.5)), 3)],
            "share_of_resamples_where_synthetic_is_better": round(float((diffs > 0).mean()), 3)}


def summarise(ev: dict) -> dict:
    return {"accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"], "weighted_f1": ev["weighted_f1"],
            "per_class_f1": {c: ev["per_class"][c]["f1"] for c in CLASSES},
            "per_class_recall": {c: f'{ev["per_class"][c]["true_positives"]}/{ev["per_class"][c]["support"]}'
                                 for c in CLASSES},
            **({"macro_f1_95ci": ev["macro_f1_95ci_bootstrap"]} if "macro_f1_95ci_bootstrap" in ev else {})}


def run_setting(name, extra, real_on, labels, video, random_, cv_rows, viewed_ids):
    def fit(train):
        if real_on:
            return build_pipeline().fit(*with_extra(train, extra))
        return build_pipeline().fit(extra["text"], extra[LABEL_COLUMN])

    v_train, v_test = video
    v_pred = fit(v_train).predict(v_test["text"])
    r_train, r_test = random_
    r_pred = fit(r_train).predict(r_test["text"])
    oof = pd.Series(index=cv_rows.index, dtype=object)
    for k in sorted(cv_rows["cv_fold"].unique()):
        te = cv_rows["cv_fold"] == k
        oof[te] = fit(cv_rows.loc[~te]).predict(cv_rows.loc[te, "text"])
    unseen = ~cv_rows["comment_id"].isin(viewed_ids)
    return {
        "name": name,
        "synthetic_rows": 0 if extra is None else len(extra),
        "real_training_rows": len(v_train) if real_on else 0,
        "video": summarise(evaluate(v_test[LABEL_COLUMN], v_pred)),
        "random": summarise(evaluate(r_test[LABEL_COLUMN], r_pred, with_ci=False)),
        "cv_pooled": summarise(evaluate(cv_rows[LABEL_COLUMN], oof, with_ci=False)),
        "cv_unseen": summarise(evaluate(cv_rows.loc[unseen, LABEL_COLUMN], oof[unseen], with_ci=False)),
        "_video_pred": list(v_pred),
    }


def main() -> int:
    labels = load_labels()
    syn = load_synthetic(SYNTHETIC)
    viewed_ids = set(read_csv(VIEWED)["comment_id"]) if VIEWED.exists() else set()
    video = load_split("video", labels)
    random_ = load_split("random", labels)
    cv_rows = attach_labels(pd.concat([read_csv(SPLITS_DIR / "random_train.csv"),
                                       read_csv(SPLITS_DIR / "random_test.csv")], ignore_index=True), labels)
    n_unseen = int((~cv_rows["comment_id"].isin(viewed_ids)).sum())

    results = [run_setting("real only", None, True, labels, video, random_, cv_rows, viewed_ids)]
    for n in SIZES:
        extra = subsample(syn, n)
        label = f"real + {n}/class synthetic" if n else f"real + all synthetic ({len(syn)})"
        results.append(run_setting(label, extra, True, labels, video, random_, cv_rows, viewed_ids))
    results.append(run_setting("synthetic only", syn, False, labels, video, random_, cv_rows, viewed_ids))

    y_video = list(video[1][LABEL_COLUMN])
    full = next(r for r in results if r["synthetic_rows"] == len(syn) and r["real_training_rows"])
    boot = paired_bootstrap_diff(y_video, results[0]["_video_pred"], full["_video_pred"])

    # Synthetic-on-synthetic (why it must never be the test set).
    s = syn.sample(frac=1.0, random_state=SEED)
    cut = int(len(s) * 0.8)
    ss_pred = build_pipeline().fit(s["text"][:cut], s[LABEL_COLUMN][:cut]).predict(s["text"][cut:])
    syn_on_syn = {"accuracy": round(accuracy_score(s[LABEL_COLUMN][cut:], ss_pred), 3),
                  "macro_f1": round(macro_f1(list(s[LABEL_COLUMN][cut:]), list(ss_pred)), 3)}

    for r in results:
        r.pop("_video_pred")
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("All test comments are real. Labels on real comments are AI-generated (Claude) and "
                 "not fully human-verified. Synthetic rows are training-only."),
        "synthetic_file": str(SYNTHETIC.relative_to(PROJECT_ROOT)),
        "viewed_comments_excluded_from_cv_unseen": len(viewed_ids),
        "cv_unseen_rows": n_unseen,
        "results": results,
        "video_macro_f1_paired_bootstrap_all_synthetic_minus_real": boot,
        "synthetic_test_on_synthetic_train_DO_NOT_USE": syn_on_syn,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "synthetic_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")

    md = ["# Synthetic training data: effect on real comments", "",
          f"Generated {out['created_utc']}. {out['note']}", "",
          "| Training data | Video acc | Video macro F1 | Random macro F1 | CV macro F1 (899) "
          f"| CV macro F1, unseen ({n_unseen}) |",
          "|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['name']} | {r['video']['accuracy']:.3f} | {r['video']['macro_f1']:.3f} "
                  f"{r['video']['macro_f1_95ci']} | {r['random']['macro_f1']:.3f} | "
                  f"{r['cv_pooled']['macro_f1']:.3f} | {r['cv_unseen']['macro_f1']:.3f} |")
    md += ["", "Per-class recall, grouped CV (all 899 real comments):", "",
           "| Training data | " + " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 1) + "|"]
    for r in results:
        md.append(f"| {r['name']} | " + " | ".join(r["cv_pooled"]["per_class_recall"][c] for c in CLASSES) + " |")
    md += ["", "Per-class recall, grouped CV, unseen comments only:", "",
           "| Training data | " + " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 1) + "|"]
    for r in results:
        md.append(f"| {r['name']} | " + " | ".join(r["cv_unseen"]["per_class_recall"][c] for c in CLASSES) + " |")
    md += ["", f"Paired bootstrap, video test macro F1 (all synthetic minus real only): "
               f"{boot['mean_diff']:+.3f}, 95% CI {boot['ci95']}, synthetic better in "
               f"{boot['share_of_resamples_where_synthetic_is_better']:.0%} of resamples.", "",
           f"Trained and tested on synthetic data: accuracy {syn_on_syn['accuracy']:.3f}, macro F1 "
           f"{syn_on_syn['macro_f1']:.3f}. This says nothing about real comments and must not be "
           "reported as model performance.", ""]
    (REPORTS / "synthetic_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
