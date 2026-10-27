"""
CreatorPulse — URL-to-insights analysis (used by the Streamlit app).

run_analysis(url, settings) runs eight stages in order and records each one in
the run's metadata.json as it starts and finishes:

    1 validate    parse the URL, check the API key and the saved model
    2 collect     src.collect (existing YouTube collectors), raw CSV saved per run
    3 preprocess  src.preprocess (normalise + tokenise, as in training)
    4 classify    src.batch_predict with the saved TF-IDF + LinearSVC pipeline
    5 topics      src.topics
    6 insights    src.insights (trend rules use the collector's capture notes)
    7 recommend   src.recommend
    8 dashboard   cross-stage consistency checks + dashboard summary

A failing stage stops the run, marks it failed with a plain-language error,
and keeps every file the earlier stages wrote. The model is only loaded,
never refitted or retrained.
"""

import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src import batch_predict, insights, recommend, topics
from src import collect as collect_mod
from src.collect import CollectionError, collect, parse_target, plan_quota
from src.predict import ModelLoadError, load_model
from src.preprocess import preprocess_frame
from src.runs import Run

CSV_ENCODING = "utf-8-sig"


@dataclass
class AnalysisSettings:
    max_videos: int = 5            # channel URLs only
    max_comments: int = 300        # per video
    exclude_spam: bool = True      # from topic discovery

    def as_dict(self) -> dict:
        return asdict(self)


class AnalysisFailed(Exception):
    def __init__(self, run: Run):
        super().__init__(run.meta.get("error", {}).get("message", "analysis failed"))
        self.run = run


def target_key(url: str) -> str:
    """Stable identity of what is being analyzed (for duplicate detection)."""
    t = parse_target(url)
    return f"{t.kind}:{t.value.strip().lower().rstrip('/')}"


def _error(code, message, fix="", detail="") -> dict:
    return {"code": code, "message": message, "fix": fix, "detail": detail}


def run_analysis(url: str, settings: Optional[AnalysisSettings] = None,
                 progress: Optional[Callable[[Run], None]] = None, youtube=None,
                 runs_dir: Optional[Path] = None, raw_dir: Optional[Path] = None, model_path=None,
                 run: Optional[Run] = None) -> Run:
    """Run the full analysis. Returns the Run; raises AnalysisFailed (with .run) on failure.

    Pass `run` to fill a run that was already created (the app does this so the
    page knows the run id before the work starts).
    """
    settings = settings or AnalysisSettings()
    notify = progress or (lambda r: None)
    run = run or Run.create(url.strip(), settings.as_dict(), runs_dir, raw_dir)
    notify(run)
    model_path = Path(model_path) if model_path else batch_predict.default_model_path()
    current = "validate"

    def step(key, detail=""):
        nonlocal current
        current = key
        run.start_stage(key, detail)
        notify(run)

    def done(key, detail=""):
        run.finish_stage(key, detail)
        notify(run)

    try:
        # 1. Validate --------------------------------------------------------
        step("validate", "Checking the link")
        target = parse_target(url)
        run.meta["target"] = {"kind": target.kind, "value": target.value, "key": target_key(url),
                              "note": target.note}
        if target.note:
            run.meta["warnings"].append(target.note)
        try:
            load_model(model_path)
        except ModelLoadError as exc:
            raise CollectionError("model_missing", "The trained model could not be loaded.",
                                  "Train it once with: python -m src.train_svm", str(exc)) from exc
        client = youtube or collect_mod.build_client()      # looked up at call time
        quota = plan_quota(target.kind, settings.max_videos, settings.max_comments)
        run.meta["quota_estimate_units"] = quota
        done("validate", f"{'Video' if target.kind == 'video' else 'Channel'} link OK; "
                         f"up to ~{quota} API units needed")

        # 2. Collect ---------------------------------------------------------
        step("collect", "Contacting YouTube")
        res = collect(target, settings.max_videos, settings.max_comments, client,
                      on_progress=lambda msg: (run.update_stage("collect", msg), notify(run)))
        run.raw_dir.mkdir(parents=True, exist_ok=True)
        res.comments.to_csv(run.raw_path, index=False, encoding=CSV_ENCODING)
        run.add_output("raw_comments", run.raw_path)
        run.meta["collection"] = {
            "collected_at": res.collected_at, "channel_id": res.channel_id, "channel_title": res.channel_title,
            "channel_thumbnail": res.channel_thumbnail,
            "videos": res.videos, "skipped_videos": res.skipped_videos, "api_units_used": res.api_units_used,
            "comments": int(len(res.comments)),
        }
        run.meta["warnings"].extend(res.warnings)
        if res.skipped_videos:
            run.meta["warnings"].append(f"{len(res.skipped_videos)} video(s) skipped: " + "; ".join(
                f"{s['title'][:40]} ({s['reason']})" for s in res.skipped_videos[:5]))
        done("collect", f"{len(res.comments)} comments from {len(res.videos)} video(s)")

        # 3. Preprocess -------------------------------------------------------
        step("preprocess", "Normalising text")
        pre = preprocess_frame(res.comments.assign(text=res.comments["text"].fillna("").astype(str)),
                               do_lemmatize=False)
        pre_path = run.path("preprocessed_comments.csv")
        pre.to_csv(pre_path, index=False, encoding=CSV_ENCODING)
        run.add_output("preprocessed_comments", pre_path)
        empty = int((pre["tokens"].fillna("").str.strip() == "").sum())
        done("preprocess", f"{len(pre)} comments normalised" + (f"; {empty} have no usable text" if empty else ""))

        # 4. Classify --------------------------------------------------------
        step("classify", "Loading the saved model")
        stats = batch_predict.run(model_path, inputs=[run.raw_path], out_dir=run.processed, raw_dir=run.dir)
        for f in ("comments_predicted.csv", "comments_excluded.csv", "category_summary.csv",
                  "batch_predict_stats.json"):
            run.add_output(f.rsplit(".", 1)[0], run.path(f))
        if stats["predicted"] == 0:
            raise CollectionError("no_usable_comments", "None of the collected comments had usable text.",
                                  "Try a video with more written comments.")
        done("classify", f"{stats['predicted']} classified, {stats['excluded']} excluded")

        # 5. Topics ----------------------------------------------------------
        step("topics", "Grouping similar comments")
        cats = topics.DEFAULT_CATEGORIES + ([] if settings.exclude_spam else ["Spam"])
        tstats = topics.run(run.path("comments_predicted.csv"), run.processed, cats,
                            allow_spam=not settings.exclude_spam)
        for f in ("topic_clusters.csv", "topic_summary.csv"):
            run.add_output(f.rsplit(".", 1)[0], run.path(f))
        done("topics", f"{tstats['topics']} topics from {tstats['selected_comments']} comments "
                       f"({tstats['unclustered_comments']} unclustered)")

        # 6. Insights --------------------------------------------------------
        step("insights", "Counting topics, videos and replies")
        capture = {v["video_id"]: v for v in res.videos}
        istats = insights.run(run.processed, run.processed, video_capture=capture, data_end=res.collected_at,
                              sampled_sources={})
        for f in ("audience_insights.json", "topic_trends.csv", "video_topic_summary.csv"):
            run.add_output(f.rsplit(".", 1)[0], run.path(f))
        done("insights", "Trends: " + ("available" if istats["trend_status"] == "ok" else "not enough history yet"))

        # 7. Recommend -------------------------------------------------------
        step("recommend", "Scoring topics")
        rstats = recommend.run(run.processed, run.processed)
        for f in ("content_recommendations.csv", "content_recommendations.json"):
            run.add_output(f.rsplit(".", 1)[0], run.path(f))
        done("recommend", f"{rstats['recommendations']} recommendation(s)")

        # 8. Dashboard -------------------------------------------------------
        step("dashboard", "Checking that every stage's output lines up")
        from src.pipeline import consistency_checks
        checks = consistency_checks(run.processed, cats, allow_spam=not settings.exclude_spam)
        failed = {k: v for k, v in checks.items() if v is not True}
        run.meta["consistency_checks"] = checks
        hard = {k: v for k, v in failed.items() if k != "recommendations_have_3_to_5_examples"}
        if hard:
            run.meta["warnings"].append("Some consistency checks failed: " + ", ".join(hard))
        ts = pd.read_csv(run.path("topic_summary.csv"), dtype=str, keep_default_na=False, encoding=CSV_ENCODING)
        run.meta["totals"] = {
            "comments_collected": int(len(res.comments)), "comments_analyzed": int(stats["predicted"]),
            "comments_excluded": int(stats["excluded"]), "videos_analyzed": int(len(res.videos)),
            "videos_skipped": int(len(res.skipped_videos)),
            "topics": int((ts["topic_id"].astype(int) != -1).sum()) if len(ts) else 0,
            "recommendations": int(rstats["recommendations"]),
            "category_counts": stats["category_counts"],
        }
        done("dashboard", f"{len(checks) - len(failed)}/{len(checks)} checks passed")
        run.complete()
        notify(run)
        return run

    except CollectionError as err:
        run.fail(current, err.to_dict())
    except Exception as exc:   # unexpected processing error: keep earlier outputs, log details
        (run.dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        from src.runs import STAGE_LABEL
        run.fail(current, _error("processing_error", f"{STAGE_LABEL[current]} failed unexpectedly.",
                                 "The outputs of the earlier stages were kept. Technical details are in "
                                 f"data/runs/{run.run_id}/error.log.", f"{type(exc).__name__}: {str(exc)[:300]}"))
    notify(run)
    raise AnalysisFailed(run)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
