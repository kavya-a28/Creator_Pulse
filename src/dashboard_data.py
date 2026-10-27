"""
CreatorPulse — load one analysis run's outputs for the dashboard.

Pure pandas, no Streamlit, so it can be tested and cached. Every file is
optional: a run that failed part-way returns whatever its finished stages
wrote, and the dashboard shows only those parts.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.runs import Run

CSV_ENCODING = "utf-8-sig"
CLASSES = ["Question", "Request", "Confusion/Problem", "Praise", "Spam", "Other"]
ACTIONABLE = ["Question", "Request", "Confusion/Problem"]


def _csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RunData:
    meta: dict
    comments: Optional[pd.DataFrame] = None        # predictions + topic columns
    excluded: Optional[pd.DataFrame] = None
    categories: Optional[pd.DataFrame] = None
    clusters: Optional[pd.DataFrame] = None
    topics: Optional[pd.DataFrame] = None
    insights: Optional[dict] = None
    trends: Optional[pd.DataFrame] = None
    video_topics: Optional[pd.DataFrame] = None
    recs: Optional[dict] = None
    recs_table: Optional[pd.DataFrame] = None
    available: dict = field(default_factory=dict)


def load_run_data(run_id: str, runs_dir: Optional[Path] = None) -> RunData:
    run = Run.load(run_id, runs_dir)
    p = run.processed
    d = RunData(meta=run.meta)
    pred = _csv(p / "comments_predicted.csv")
    d.excluded = _csv(p / "comments_excluded.csv")
    d.categories = _csv(p / "category_summary.csv")
    d.clusters = _csv(p / "topic_clusters.csv")
    d.topics = _csv(p / "topic_summary.csv")
    d.insights = _json(p / "audience_insights.json")
    d.trends = _csv(p / "topic_trends.csv")
    d.video_topics = _csv(p / "video_topic_summary.csv")
    d.recs = _json(p / "content_recommendations.json")
    d.recs_table = _csv(p / "content_recommendations.csv")
    if pred is not None:
        d.comments = comments_table(pred, d.clusters)
    if d.topics is not None:
        for c in ("topic_id", "n_comments_raw", "n_unique_comments", "n_videos", "n_question", "n_request",
                  "n_confusion_problem"):
            if c in d.topics:
                d.topics[c] = pd.to_numeric(d.topics[c], errors="coerce").fillna(0).astype(int)
    d.available = {k: getattr(d, k) is not None for k in
                   ("comments", "clusters", "topics", "insights", "trends", "recs")}
    return d


def comments_table(pred: pd.DataFrame, clusters: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per analyzed comment with its topic (blank when it was not clustered)."""
    out = pred.copy()
    out["topic_name"] = ""
    out["topic_id"] = ""
    if clusters is not None and len(clusters):
        m = clusters.set_index("comment_id")
        out["topic_name"] = out["comment_id"].map(m["topic_name"]).fillna("")
        out["topic_id"] = out["comment_id"].map(m["topic_id"]).fillna("")
    out["published_dt"] = pd.to_datetime(out["published_at"], utc=True, errors="coerce")
    return out


def filter_comments(df: pd.DataFrame, categories=None, topic=None, video=None, query: str = "") -> pd.DataFrame:
    out = df
    if categories:
        out = out[out["predicted_category"].isin(categories)]
    if topic:
        out = out[out["topic_name"] == topic]
    if video:
        out = out[out["video_title"] == video]
    q = (query or "").strip().lower()
    if q:
        out = out[out["text"].str.lower().str.contains(q, regex=False)]
    return out


def export_comments(df: pd.DataFrame) -> bytes:
    cols = ["comment_id", "text", "predicted_category", "topic_name", "video_title", "video_id",
            "channel_title", "published_at", "author_channel_id", "like_count", "reply_count",
            "creator_replied", "prediction_margin", "low_confidence"]
    return df[[c for c in cols if c in df.columns]].to_csv(index=False).encode(CSV_ENCODING)


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    counts = df["predicted_category"].value_counts().reindex(CLASSES, fill_value=0)
    return pd.DataFrame({"category": CLASSES, "comments": counts.values.astype(int),
                         "percent": [round(100 * v / n, 1) if n else 0.0 for v in counts.values]})


def topic_comments(d: RunData, topic_id: int) -> pd.DataFrame:
    c = d.clusters
    g = c[c["topic_id"].astype(int) == int(topic_id)].copy()
    g["similarity_to_topic"] = pd.to_numeric(g["similarity_to_topic"], errors="coerce")
    return g.sort_values(["is_representative", "similarity_to_topic"], ascending=[False, False])
