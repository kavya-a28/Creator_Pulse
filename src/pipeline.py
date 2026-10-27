"""
CreatorPulse — run the audience-intelligence pipeline end to end.

Stages (in order)
    predict    src.batch_predict  -> comments_predicted.csv, category_summary.csv
    topics     src.topics         -> topic_clusters.csv, topic_summary.csv
    insights   src.insights       -> audience_insights.json, topic_trends.csv, video_topic_summary.csv
    recommend  src.recommend      -> content_recommendations.csv / .json

After the stages run, consistency checks confirm each stage read the previous
stage's output correctly (row counts reconcile, every topic comment exists in
the predictions, every recommendation's evidence exists in the topics) and
that raw and labelled data files are byte-identical before and after.
Results go to data/processed/pipeline_report.json.

The model is only loaded, never retrained, so the held-out evaluation in
reports/ is untouched.

Usage:
    python -m src.pipeline
    python -m src.pipeline --stages topics insights recommend
    python -m src.pipeline --model models/creatorpulse_svm.joblib
    python -m src.pipeline --include Praise --min-evidence 2
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import batch_predict, insights, recommend, topics  # noqa: E402
from src.config import LABELLED_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402

STAGES = ["predict", "topics", "insights", "recommend"]
REPORT = PROCESSED_DATA_DIR / "pipeline_report.json"


def _hashes(dirs) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.csv")):
            out[str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)] = \
                hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def consistency_checks(out_dir: Path, categories, allow_spam: bool = False) -> dict:
    """Cross-stage checks; each value is True or a description of the problem."""
    c = {}
    stats = json.loads((out_dir / "batch_predict_stats.json").read_text(encoding="utf-8"))
    pred = _read(out_dir / "comments_predicted.csv")
    excl = _read(out_dir / "comments_excluded.csv")
    c["predicted_plus_excluded_equals_input"] = (len(pred) + len(excl) == stats["input_rows"]) or \
        f"{len(pred)} + {len(excl)} != {stats['input_rows']}"
    c["predicted_comment_ids_unique"] = bool(pred["comment_id"].is_unique)
    summary = _read(out_dir / "category_summary.csv")
    c["category_summary_matches_predictions"] = bool(summary["comments"].astype(int).sum() == len(pred))
    clusters = _read(out_dir / "topic_clusters.csv")
    want = pred["predicted_category"].isin([x for x in categories if x != "Spam" or allow_spam]).sum()
    c["topic_input_matches_selected_predictions"] = (len(clusters) == want) or f"{len(clusters)} vs {want}"
    c["topic_comments_exist_in_predictions"] = bool(set(clusters["comment_id"]) <= set(pred["comment_id"]))
    c["no_spam_in_topics"] = True if allow_spam else bool((clusters["predicted_category"] != "Spam").all())
    ts = _read(out_dir / "topic_summary.csv")
    c["topic_summary_counts_match_clusters"] = bool(ts["n_comments_raw"].astype(int).sum() == len(clusters))
    c["every_topic_has_a_representative"] = bool(
        (ts.loc[ts["topic_id"].astype(int) != -1, "representative_comment_id"] != "").all())
    vts = _read(out_dir / "video_topic_summary.csv")
    for level in ("video", "channel"):
        c[f"video_topic_summary_{level}_counts_match"] = bool(
            vts.loc[vts["level"] == level, "n_comments_raw"].astype(int).sum() == len(clusters))
    tr = _read(out_dir / "topic_trends.csv")
    c["trends_cover_every_topic"] = bool(set(tr["topic_id"]) == set(ts["topic_id"]))
    rec = json.loads((out_dir / "content_recommendations.json").read_text(encoding="utf-8"))
    ids = set(clusters["comment_id"])
    text = dict(zip(clusters["comment_id"], clusters["text"]))
    c["recommendation_evidence_traceable"] = all(
        set(r["evidence_comment_ids"]) <= ids and all(text[e["comment_id"]] == e["text"]
                                                      for e in r["representative_comments"])
        for r in rec["recommendations"])
    c["recommendations_have_3_to_5_examples"] = all(3 <= len(r["representative_comments"]) <= 5
                                                    for r in rec["recommendations"]) or \
        "some recommendations have fewer than 3 distinct example comments"
    c["no_duplicate_recommendation_topics"] = len({t for r in rec["recommendations"] for t in r["topic_ids"]}) == \
        sum(len(r["topic_ids"]) for r in rec["recommendations"])
    return {k: (bool(v) if not isinstance(v, str) else v) for k, v in c.items()}


def run(stages=None, model=None, include=None, min_evidence=recommend.MIN_EVIDENCE,
        out_dir: Path = PROCESSED_DATA_DIR, raw_dir: Path = RAW_DATA_DIR) -> dict:
    stages = stages or STAGES
    out_dir = Path(out_dir)
    protected = [Path(raw_dir), LABELLED_DATA_DIR]
    before = _hashes(protected)
    categories = topics.DEFAULT_CATEGORIES + list(include or [])
    report = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stages": {}}
    for st in STAGES:
        if st not in stages:
            continue
        t0 = time.time()
        if st == "predict":
            res = batch_predict.run(model, out_dir=out_dir, raw_dir=Path(raw_dir))
            res = {k: res[k] for k in ("input_rows", "predicted", "excluded", "excluded_by_reason",
                                       "category_counts", "model", "seen_in_training")}
        elif st == "topics":
            res = topics.run(out_dir / "comments_predicted.csv", out_dir, categories)
        elif st == "insights":
            res = insights.run(out_dir, out_dir)
        else:
            res = recommend.run(out_dir, out_dir, min_evidence)
        res["seconds"] = round(time.time() - t0, 1)
        report["stages"][st] = res
        print(f"[ok] {st:<9} {json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, ensure_ascii=False)}")

    after = _hashes(protected)
    report["raw_and_labelled_files_unchanged"] = before == after
    if set(STAGES) <= set(stages) or all((out_dir / f).exists() for f in (
            "comments_predicted.csv", "topic_clusters.csv", "video_topic_summary.csv", "content_recommendations.json")):
        report["consistency_checks"] = consistency_checks(out_dir, categories)
    stats = json.loads((out_dir / "batch_predict_stats.json").read_text(encoding="utf-8"))
    ts = _read(out_dir / "topic_summary.csv")
    rec = json.loads((out_dir / "content_recommendations.json").read_text(encoding="utf-8"))
    report["totals"] = {
        "input_comments": stats["input_rows"], "processed_comments": stats["predicted"],
        "excluded_comments": stats["excluded"],
        "discovered_topics": int((ts["topic_id"].astype(int) != -1).sum()),
        "generated_recommendations": rec["n_recommendations"],
    }
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / REPORT.name).write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                  default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return report


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Run the CreatorPulse audience pipeline.")
    p.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    p.add_argument("--model", default=None)
    p.add_argument("--include", nargs="*", default=[], choices=topics.OPTIONAL_CATEGORIES)
    p.add_argument("--min-evidence", type=int, default=recommend.MIN_EVIDENCE)
    p.add_argument("--out-dir", default=str(PROCESSED_DATA_DIR))
    p.add_argument("--raw-dir", default=str(RAW_DATA_DIR))
    a = p.parse_args(argv)
    rep = run(a.stages, a.model, a.include, a.min_evidence, Path(a.out_dir), Path(a.raw_dir))
    print("\nTotals:", json.dumps(rep["totals"]))
    checks = rep.get("consistency_checks", {})
    failed = {k: v for k, v in checks.items() if v is not True}
    print("Raw/labelled files unchanged:", rep["raw_and_labelled_files_unchanged"])
    print(f"Consistency checks: {len(checks) - len(failed)}/{len(checks)} passed")
    for k, v in failed.items():
        print(f"  [check] {k}: {v}")
    return 0 if rep["raw_and_labelled_files_unchanged"] and not failed else 3


if __name__ == "__main__":
    sys.exit(main())
