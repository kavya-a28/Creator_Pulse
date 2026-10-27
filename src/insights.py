"""
CreatorPulse — Stage 3: audience insights and trend analysis.

Reads the outputs of the earlier stages and summarises what the audience is
asking for, which topics recur across videos and channels, what reply data
shows, and whether any topic is growing.

Inputs
    data/processed/comments_predicted.csv   (src.batch_predict)
    data/processed/batch_predict_stats.json (src.batch_predict)
    data/processed/topic_clusters.csv       (src.topics)
    data/processed/topic_summary.csv        (src.topics)

Outputs
    data/processed/audience_insights.json
    data/processed/topic_trends.csv
    data/processed/video_topic_summary.csv

Counting rules
    raw      every comment
    unique   one per near-duplicate group (copy-pasted / flooded comments count once)
    commenters  distinct author_channel_id, when available

Trend rules
-----------
Topic growth is the change in a topic's SHARE of actionable comments between
the latest 7 days and the 7 days before (ending at the newest comment in the
data), so a busier week does not make every topic look like it is growing.

It is only computed on videos whose comment capture is COMPLETE for both
periods. A capture is:
    complete  every top-level comment up to collection time was kept
              (fetch_comments.py files; build_labelling_dataset captures
              smaller than the per-video cap and not cut by the channel target)
    sampled   only some comments were kept (hit the per-video cap of 50, or the
              channel target of 200). Which comments were kept depends on the
              API's ordering, so their timestamps do not measure volume.
An eligible video must also be complete back to the start of the earlier
period and have data up to the end of the later period.

Status per topic: growing / declining / stable / insufficient_data. With fewer
than MIN_ELIGIBLE_VIDEOS eligible videos or fewer than MIN_PERIOD_COMMENTS
actionable comments in either period, every topic is insufficient_data and the
report says trends cannot yet be established. A topic is never called
trending because of its overall size.

Creator replies: creator_replied True = a creator reply was found;
False = the thread had no replies, or every reply was returned and none was by
the creator (as of collection time); empty = reply data incomplete. Only False
supports "no creator reply at collection time"; empty is never treated as
unanswered.

Usage:
    python -m src.insights
    python -m src.insights --window-days 14
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR  # noqa: E402
from src.topics import UNCLUSTERED_ID, near_duplicate_groups  # noqa: E402

CSV_ENCODING = "utf-8-sig"
CLASSES = ["Question", "Request", "Confusion/Problem", "Praise", "Spam", "Other"]
ACTIONABLE = ["Question", "Request", "Confusion/Problem"]

IN_PREDICTED = PROCESSED_DATA_DIR / "comments_predicted.csv"
IN_STATS = PROCESSED_DATA_DIR / "batch_predict_stats.json"
IN_CLUSTERS = PROCESSED_DATA_DIR / "topic_clusters.csv"
IN_TOPICS = PROCESSED_DATA_DIR / "topic_summary.csv"
OUT_INSIGHTS = PROCESSED_DATA_DIR / "audience_insights.json"
OUT_TRENDS = PROCESSED_DATA_DIR / "topic_trends.csv"
OUT_VIDEO = PROCESSED_DATA_DIR / "video_topic_summary.csv"

WINDOW_DAYS = 7
MIN_ELIGIBLE_VIDEOS = 3
MIN_PERIOD_COMMENTS = 30       # actionable unique comments per period (eligible videos)
MIN_TOPIC_COMMENTS = 5         # topic unique comments across both periods
GROWTH_RATIO = 1.5             # share ratio to call growing / declining
MIN_ABS_CHANGE = 2             # and at least this many more / fewer comments
# Collection settings of src.build_labelling_dataset (defaults used for the data).
SAMPLED_SOURCES = {"educational_comments.csv": {"per_video_cap": 50, "channel_target": 200}}
DEDICATED_VIDEO_MIN_COMMENTERS = 3


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

def capture_coverage(pred: pd.DataFrame, sampled_sources=None, excluded=None, video_capture=None) -> pd.DataFrame:
    """One row per video: comment count, time span and whether the capture is complete.

    Counts include comments excluded from prediction (empty text etc.), because
    the cap applied to everything that was collected.
    """
    sampled_sources = SAMPLED_SOURCES if sampled_sources is None else sampled_sources
    df = pred.copy()
    extra = {}
    if excluded is not None and len(excluded) and {"source_file", "video_id"} <= set(excluded.columns):
        extra = excluded.groupby(["source_file", "video_id"]).size().to_dict()
    df["_ts"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    rows = []
    for (src, vid), g in df.groupby(["source_file", "video_id"], sort=True):
        rows.append({"source_file": src, "video_id": vid, "channel_title": g["channel_title"].iloc[0],
                     "video_title": g["video_title"].iloc[0],
                     "video_published_at": g["video_published_at"].iloc[0] if "video_published_at" in g else "",
                     "n_comments": len(g) + int(extra.get((src, vid), 0)),
                     "n_excluded_from_prediction": int(extra.get((src, vid), 0)), "first_comment_at": g["_ts"].min(), "last_comment_at": g["_ts"].max(),
                     "n_missing_timestamp": int(g["_ts"].isna().sum())})
    cov = pd.DataFrame(rows)
    if cov.empty:
        return cov
    cov["capture"] = "complete"
    cov["capture_reason"] = "all top-level comments fetched (fetch_comments)"
    for src, cfg in sampled_sources.items():
        m = cov["source_file"] == src
        if not m.any():
            continue
        cap, target = cfg["per_video_cap"], cfg["channel_target"]
        hit_cap = m & (cov["n_comments"] >= cap)
        cov.loc[hit_cap, ["capture", "capture_reason"]] = ["sampled", f"hit per-video cap of {cap}"]
        # The last video taken for a channel that reached its target may have been cut short.
        sub = cov[m].copy()
        sub["_vp"] = pd.to_datetime(sub["video_published_at"], utc=True, errors="coerce")
        for ch, g in sub.groupby("channel_title"):
            if g["n_comments"].sum() >= target:
                last = g.sort_values("_vp").index[0]          # videos were fetched newest first
                if cov.loc[last, "capture"] == "complete":
                    cov.loc[last, ["capture", "capture_reason"]] = [
                        "sampled", f"last video before channel target of {target} was reached"]
        rest = m & (cov["capture"] == "complete")
        cov.loc[rest, "capture_reason"] = f"fewer than the per-video cap ({cap}) were available"
    # Captures described by the collector (app runs): complete / newest_n / sampled.
    cov["complete_since"] = pd.to_datetime(cov["video_published_at"], utc=True, errors="coerce") \
        .fillna(cov["first_comment_at"])
    cov.loc[cov["capture"] != "complete", "complete_since"] = pd.NaT
    for vid, note in (video_capture or {}).items():
        m = cov["video_id"] == vid
        if not m.any():
            continue
        kind = note.get("capture", "sampled")
        if kind == "complete":
            cov.loc[m, ["capture", "capture_reason"]] = ["complete", "every comment up to collection time"]
        elif kind == "newest_n":
            cov.loc[m, ["capture", "capture_reason"]] = [
                "newest_n", f"newest {note.get('n_comments')} comments (newest-first order verified); "
                "complete only since the oldest one"]
            cov.loc[m, "complete_since"] = pd.Timestamp(note["complete_since"]) if note.get("complete_since") \
                else pd.NaT
        else:
            cov.loc[m, ["capture", "capture_reason"]] = ["sampled", "capped and API order not verified"]
            cov.loc[m, "complete_since"] = pd.NaT
    return cov


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def topic_trends(clusters: pd.DataFrame, topics: pd.DataFrame, coverage: pd.DataFrame,
                 window_days: int = WINDOW_DAYS, data_end=None):
    """Return (trends_df, trend_meta)."""
    c = clusters.copy()
    c["_ts"] = pd.to_datetime(c["published_at"], utc=True, errors="coerce")
    meta = {"window_days": window_days, "method": "topic share of actionable unique comments, "
            "eligible (complete-capture) videos only"}
    if c["_ts"].notna().sum() == 0 or coverage.empty:
        meta.update(status="insufficient_data", reason="no usable timestamps")
        end = None
    else:
        end = pd.Timestamp(data_end) if data_end is not None else c["_ts"].max()
        meta["data_end_source"] = "collection time" if data_end is not None else "newest comment"
    if end is not None:
        cur_start, prev_start = end - timedelta(days=window_days), end - timedelta(days=2 * window_days)
        cov = coverage.copy()
        if "complete_since" not in cov:
            cov["complete_since"] = pd.NaT
        reaches_end = (cov["last_comment_at"] >= end - timedelta(days=1)) if data_end is None \
            else pd.Series(True, index=cov.index)     # known collection time: silence is real
        cov["eligible"] = (cov["capture"].isin(["complete", "newest_n"]) & cov["complete_since"].notna()
                           & (cov["complete_since"] <= prev_start) & reaches_end)
        eligible = set(cov.loc[cov["eligible"], "video_id"])
        meta.update({
            "data_end": end.isoformat(),
            "previous_period": [prev_start.isoformat(), cur_start.isoformat()],
            "current_period": [cur_start.isoformat(), end.isoformat()],
            "videos_total": int(len(cov)),
            "videos_complete_capture": int((cov["capture"] == "complete").sum()),
            "videos_newest_n_capture": int((cov["capture"] == "newest_n").sum()),
            "videos_sampled_capture": int((cov["capture"] == "sampled").sum()),
            "eligible_videos": sorted(eligible),
        })
    else:
        eligible = set()
        cur_start = prev_start = None

    def period_counts(frame):
        if end is None:
            return frame.iloc[0:0], frame.iloc[0:0]
        prev = frame[(frame["_ts"] > prev_start) & (frame["_ts"] <= cur_start)]
        cur = frame[(frame["_ts"] > cur_start) & (frame["_ts"] <= end)]
        return prev, cur

    elig = c[c["video_id"].isin(eligible)]
    e_prev, e_cur = period_counts(elig)
    a_prev, a_cur = period_counts(c)
    n_prev_u, n_cur_u = int(e_prev["is_canonical"].sum()), int(e_cur["is_canonical"].sum())
    global_ok = (end is not None and len(eligible) >= MIN_ELIGIBLE_VIDEOS
                 and n_prev_u >= MIN_PERIOD_COMMENTS and n_cur_u >= MIN_PERIOD_COMMENTS)
    if end is not None:
        meta.update({"eligible_actionable_unique_previous": n_prev_u,
                     "eligible_actionable_unique_current": n_cur_u})
        if global_ok:
            meta.update(status="ok", reason="")
        else:
            why = []
            if len(eligible) < MIN_ELIGIBLE_VIDEOS:
                why.append(f"only {len(eligible)} video(s) have a complete comment capture covering both "
                           f"periods (need {MIN_ELIGIBLE_VIDEOS})")
            if n_prev_u < MIN_PERIOD_COMMENTS or n_cur_u < MIN_PERIOD_COMMENTS:
                why.append(f"eligible videos have {n_prev_u} / {n_cur_u} actionable unique comments in the "
                           f"previous / current period (need {MIN_PERIOD_COMMENTS} each)")
            meta.update(status="insufficient_data",
                        reason="Trends cannot yet be established: " + "; ".join(why) + ". "
                        + ("Most videos were sampled (capped), so their timestamps reflect which comments the "
                           "API returned, not how many people commented."
                           if (cov["capture"] == "sampled").mean() > 0.5 else
                           "A single snapshot of recent comments does not cover two full comparison periods."))

    rows = []
    for _, t in topics.iterrows():
        tid = int(t["topic_id"])
        def cnt(frame, unique):
            f = frame[frame["topic_id"].astype(int) == tid]
            return int(f["is_canonical"].sum()) if unique else int(len(f))
        pu, cu = cnt(e_prev, True), cnt(e_cur, True)
        ps = pu / n_prev_u if n_prev_u else np.nan
        cs = cu / n_cur_u if n_cur_u else np.nan
        if tid == UNCLUSTERED_ID:
            status, reason = "not_applicable", "Unclustered is not a topic"
        elif not global_ok:
            status, reason = "insufficient_data", "global coverage too low (see audience_insights.json)"
        elif pu + cu < MIN_TOPIC_COMMENTS:
            status, reason = "insufficient_data", f"fewer than {MIN_TOPIC_COMMENTS} unique comments in the two periods"
        elif ps == 0 and cu >= MIN_ABS_CHANGE + 1:
            status, reason = "growing", "absent in the previous period"
        elif ps > 0 and cs / ps >= GROWTH_RATIO and cu - pu >= MIN_ABS_CHANGE:
            status, reason = "growing", f"share x{cs / ps:.2f}"
        elif ps > 0 and cs / ps <= 1 / GROWTH_RATIO and pu - cu >= MIN_ABS_CHANGE:
            status, reason = "declining", f"share x{cs / ps:.2f}"
        else:
            status, reason = "stable", "change below thresholds"
        rows.append({
            "topic_id": tid, "topic_name": t["topic_name"],
            "n_unique_comments_total": int(t["n_unique_comments"]),
            "eligible_previous_unique": pu, "eligible_current_unique": cu,
            "eligible_previous_raw": cnt(e_prev, False), "eligible_current_raw": cnt(e_cur, False),
            "eligible_previous_share": round(ps, 3) if not np.isnan(ps) else "",
            "eligible_current_share": round(cs, 3) if not np.isnan(cs) else "",
            "all_videos_previous_unique_NOT_COMPARABLE": cnt(a_prev, True),
            "all_videos_current_unique_NOT_COMPARABLE": cnt(a_cur, True),
            "status": status, "reason": reason,
        })
    meta["growing_topics"] = [r["topic_name"] for r in rows if r["status"] == "growing"]
    return pd.DataFrame(rows), meta


# --------------------------------------------------------------------------
# Video / channel comparisons
# --------------------------------------------------------------------------

def video_topic_summary(clusters: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    has_author = (clusters["author_channel_id"] != "").any()
    for level, keys in (("video", ["video_id"]), ("channel", ["channel_title"])):
        totals = pred.groupby(keys).size()
        act_tot = clusters[clusters["is_canonical"]].groupby(keys).size()
        for key, g in clusters.groupby(keys + ["topic_id"], sort=True):
            k = key[0]
            u = g[g["is_canonical"]]
            rows.append({
                "level": level,
                "video_id": k if level == "video" else "",
                "video_title": g["video_title"].iloc[0] if level == "video" else "",
                "channel_title": g["channel_title"].iloc[0],
                "topic_id": int(key[1]), "topic_name": g["topic_name"].iloc[0],
                "n_comments_raw": int(len(g)), "n_unique_comments": int(len(u)),
                "n_unique_commenters": int(g.loc[g["author_channel_id"] != "", "author_channel_id"].nunique())
                if has_author else "",
                "share_of_actionable_unique": round(len(u) / act_tot.get(k, 1), 3),
                "dominant_intent": u["predicted_category"].mode().iloc[0] if len(u) else "",
                "total_comments_in_" + "scope": int(totals.get(k, 0)),
                "actionable_unique_in_scope": int(act_tot.get(k, 0)),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_unc"] = out["topic_id"] == UNCLUSTERED_ID
    out = out.sort_values(["level", "channel_title", "video_id", "_unc", "n_unique_comments"],
                          ascending=[False, True, True, True, False]).drop(columns="_unc")
    return out


# --------------------------------------------------------------------------
# Insights
# --------------------------------------------------------------------------

def _topic_brief(t: pd.Series) -> dict:
    return {"topic_id": int(t["topic_id"]), "topic_name": t["topic_name"],
            "n_comments_raw": int(t["n_comments_raw"]), "n_unique_comments": int(t["n_unique_comments"]),
            "n_unique_commenters": int(t["n_unique_commenters"]) if str(t["n_unique_commenters"]) != "" else None,
            "n_videos": int(t["n_videos"]), "dominant_intent": t["dominant_intent"],
            "representative_comment_id": t["representative_comment_id"],
            "representative_comment": t["representative_comment"]}


def reply_indicators(frame: pd.DataFrame, by: str) -> list:
    if "creator_replied" not in frame.columns:
        return []
    f = frame.copy()
    f["_cr"] = _bool(f["creator_replied"])
    f["_rc"] = pd.to_numeric(f.get("reply_count", ""), errors="coerce")
    out = []
    for k, g in f.groupby(by, sort=True):
        out.append({by: k if not isinstance(k, (np.integer,)) else int(k), "comments": int(len(g)),
                    "creator_reply_found": int((g["_cr"] == True).sum()),  # noqa: E712
                    "no_creator_reply_at_collection": int((g["_cr"] == False).sum()),  # noqa: E712
                    "reply_data_incomplete": int(g["_cr"].isna().sum()),
                    "threads_with_any_reply": int((g["_rc"] > 0).sum())})
    return out


def build_insights(pred, stats, clusters, topics, coverage, trends, trend_meta) -> dict:
    # Category counts, raw and near-duplicate-collapsed (over all predicted comments).
    groups = near_duplicate_groups(pred["text"].tolist())
    p = pred.assign(_g=groups)
    p["_canon"] = ~p.duplicated("_g")
    cats = []
    n, nu = len(p), int(p["_canon"].sum())
    for cname in CLASSES:
        m = p["predicted_category"] == cname
        cats.append({"category": cname, "comments": int(m.sum()), "percent": round(100 * m.sum() / n, 2) if n else 0,
                     "unique_comments": int((m & p["_canon"]).sum()),
                     "unique_percent": round(100 * (m & p["_canon"]).sum() / nu, 2) if nu else 0,
                     "not_seen_in_training": int((m & ~_bool(p["seen_in_training"]).fillna(False)).sum())})

    real = topics[topics["topic_id"].astype(int) != UNCLUSTERED_ID].copy()
    for col in ["n_unique_comments", "n_comments_raw", "n_videos", "n_question", "n_request", "n_confusion_problem"]:
        real[col] = real[col].astype(int)
    real["_commenters"] = pd.to_numeric(real["n_unique_commenters"], errors="coerce").fillna(real["n_unique_comments"])
    ranked = real.sort_values(["n_unique_comments", "n_videos", "topic_id"], ascending=[False, False, True])

    def by_intent(col, intent):
        r = real[real[col] > 0].sort_values([col, "n_videos", "topic_id"], ascending=[False, False, True])
        out = []
        for _, t in r.head(8).iterrows():
            b = _topic_brief(t)
            b["n_unique_with_this_intent"] = int(t[col])
            # Quote a comment of THIS intent (the topic's representative may be another intent).
            ex = clusters[(clusters["topic_id"].astype(int) == int(t["topic_id"]))
                          & (clusters["predicted_category"] == intent) & clusters["is_canonical"]].copy()
            ex["_s"] = pd.to_numeric(ex["similarity_to_topic"], errors="coerce").fillna(0)
            ex = ex.sort_values(["_s", "comment_id"], ascending=[False, True])
            b["example_comment_id"] = ex["comment_id"].iloc[0] if len(ex) else ""
            b["example_comment"] = ex["text"].iloc[0] if len(ex) else ""
            out.append(b)
        # Exact/near-duplicate repeats of the same comment (also outside topics).
        reps = (clusters[clusters["predicted_category"] == intent]
                .groupby("near_dup_group").agg(size=("comment_id", "size"), text=("text", "first"),
                                               comment_id=("comment_id", "first"),
                                               commenters=("author_channel_id", "nunique")))
        reps = reps[reps["size"] > 1].sort_values(["size", "comment_id"], ascending=[False, True]).head(5)
        return {"top_topics": out,
                "most_repeated_comments": [{"comment_id": r.comment_id, "text": r.text, "copies": int(r.size),
                                            "distinct_commenters": int(r.commenters)} for r in reps.itertuples()]}

    cross = real[real["n_videos"] >= 2].sort_values(["n_videos", "n_unique_comments"], ascending=False)
    dedicated = real[(real["dominant_intent"].isin(["Question", "Confusion/Problem"]))
                     & (real["_commenters"] >= DEDICATED_VIDEO_MIN_COMMENTERS)]
    dedicated = dedicated.sort_values(["_commenters", "n_videos"], ascending=False)

    channels = []
    for ch, g in pred.groupby("channel_title", sort=True):
        cg = clusters[clusters["channel_title"] == ch]
        top = (cg[(cg["topic_id"].astype(int) != UNCLUSTERED_ID) & cg["is_canonical"]]
               .groupby("topic_name").size().sort_values(ascending=False).head(3))
        channels.append({"channel_title": ch, "videos": int(g["video_id"].nunique()), "comments": int(len(g)),
                         "actionable_comments": int(g["predicted_category"].isin(ACTIONABLE).sum()),
                         "actionable_percent": round(100 * g["predicted_category"].isin(ACTIONABLE).mean(), 1),
                         "category_counts": g["predicted_category"].value_counts().reindex(CLASSES, fill_value=0)
                         .astype(int).to_dict(),
                         "top_topics_by_unique_comments": {k: int(v) for k, v in top.items()}})

    clusters_t = clusters.assign(topic=clusters["topic_name"])
    top_names = set(ranked.head(10)["topic_name"])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": [
            "Categories are model predictions (TF-IDF + LinearSVC trained on AI-generated labels).",
            "raw = every comment; unique = one per near-duplicate group; commenters = distinct author IDs.",
            f"{stats.get('seen_in_training', 0)} of {stats.get('predicted', n)} comments were in the model's "
            "training data, so their predictions are optimistic.",
        ],
        "run_summary": {
            "input_comments": stats.get("input_rows"), "processed_comments": stats.get("predicted", n),
            "excluded_comments": stats.get("excluded"), "excluded_by_reason": stats.get("excluded_by_reason"),
            "input_files": stats.get("input_files"), "model": stats.get("model"),
            "videos": int(pred["video_id"].nunique()),
            "channels_known": int(pred.loc[pred["channel_title"] != "unknown", "channel_title"].nunique()),
            "comments_with_unknown_channel": int((pred["channel_title"] == "unknown").sum()),
            "actionable_comments_for_topics": int(len(clusters)),
            "actionable_unique_comments": int(clusters["is_canonical"].sum()),
            "topics_discovered": int(len(real)),
            "comments_in_topics": int((clusters["topic_id"].astype(int) != UNCLUSTERED_ID).sum()),
            "comments_unclustered": int((clusters["topic_id"].astype(int) == UNCLUSTERED_ID).sum()),
            "comment_time_range": [str(pd.to_datetime(pred["published_at"], utc=True, errors="coerce").min()),
                                   str(pd.to_datetime(pred["published_at"], utc=True, errors="coerce").max())],
        },
        "categories": {"total_comments": n, "total_unique_comments": nu, "per_category": cats},
        "most_discussed_topics": [_topic_brief(t) for _, t in ranked.head(10).iterrows()],
        "most_common": {"Question": by_intent("n_question", "Question"),
                        "Request": by_intent("n_request", "Request"),
                        "Confusion/Problem": by_intent("n_confusion_problem", "Confusion/Problem")},
        "topics_across_multiple_videos": [dict(_topic_brief(t), video_titles=t["video_titles"],
                                               channels=t["channels"]) for _, t in cross.iterrows()],
        "trends": dict(trend_meta, topics=trends[trends["status"].isin(["growing", "declining"])]
                       [["topic_name", "status", "reason"]].to_dict("records")),
        "capture_coverage": [{k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}
                             for r in coverage.to_dict("records")],
        "channel_comparison": channels,
        "creator_replies": {
            "meaning": ("creator_reply_found: a reply by the channel was seen. no_creator_reply_at_collection: no "
                        "replies, or all replies were returned and none was by the channel. reply_data_incomplete: "
                        "cannot tell; never counted as unanswered."),
            "by_category": reply_indicators(pred, "predicted_category"),
            "top_topics": reply_indicators(clusters_t[clusters_t["topic"].isin(top_names)], "topic"),
        },
        "repeated_questions_needing_a_dedicated_video": [
            dict(_topic_brief(t), reason=f"{int(t['_commenters'])} distinct commenters across {int(t['n_videos'])} "
                 f"video(s) asked or reported problems about this") for _, t in dedicated.iterrows()],
        "limitations": [
            (f"{int((pred['channel_title'] == 'unknown').sum())} comments from "
             f"{pred.loc[pred['channel_title'] == 'unknown', 'video_id'].nunique()} video(s) have no channel/title "
             "metadata in the raw file; they are grouped as channel 'unknown'.")
            if (pred["channel_title"] == "unknown").any() else "",
            (f"Creator replies were found on {int((_bool(pred['creator_replied']) == True).sum())} of {n} "  # noqa: E712
             "comments, so reply-based insights are very limited.") if "creator_replied" in pred else "",
            trend_meta.get("reason", ""),
            "Topic discovery uses word and character overlap, so synonyms (e.g. 'stock market' / 'trading') and "
            "Hindi vs English wording of the same topic may form separate topics.",
        ],
    }


def run(out_dir: Path = PROCESSED_DATA_DIR, in_dir: Path = PROCESSED_DATA_DIR,
        window_days: int = WINDOW_DAYS, video_capture=None, data_end=None, sampled_sources=None) -> dict:
    """video_capture / data_end come from the app collector (src.collect); the CLI leaves them None."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    for f in (IN_PREDICTED, IN_CLUSTERS, IN_TOPICS):
        if not (in_dir / f.name).exists():
            raise SystemExit(f"[ERROR] {in_dir / f.name} not found; run the earlier stages first")
    pred = read_csv(in_dir / IN_PREDICTED.name)
    stats = json.loads((in_dir / IN_STATS.name).read_text(encoding="utf-8")) if (in_dir / IN_STATS.name).exists() else {}
    clusters = read_csv(in_dir / IN_CLUSTERS.name)
    clusters["topic_id"] = clusters["topic_id"].astype(int)
    clusters["is_canonical"] = _bool(clusters["is_canonical"]).fillna(True)
    if "source_file" not in pred.columns:
        pred["source_file"] = ""
    topics = read_csv(in_dir / IN_TOPICS.name)
    missing = set(clusters["comment_id"]) - set(pred["comment_id"])
    if missing:
        raise SystemExit(f"[ERROR] {len(missing)} topic comments are not in comments_predicted.csv "
                         "(stage outputs are out of sync; re-run src.topics)")
    excl_path = in_dir / "comments_excluded.csv"
    coverage = capture_coverage(pred, sampled_sources=sampled_sources,
                                excluded=read_csv(excl_path) if excl_path.exists() else None,
                                video_capture=video_capture)
    trends, meta = topic_trends(clusters, topics, coverage, window_days, data_end=data_end)
    video = video_topic_summary(clusters, pred)
    insights = build_insights(pred, stats, clusters, topics, coverage, trends, meta)
    insights["limitations"] = [x for x in insights["limitations"] if x]
    out_dir.mkdir(parents=True, exist_ok=True)
    trends.to_csv(out_dir / OUT_TRENDS.name, index=False, encoding=CSV_ENCODING)
    video.to_csv(out_dir / OUT_VIDEO.name, index=False, encoding=CSV_ENCODING)
    (out_dir / OUT_INSIGHTS.name).write_text(json.dumps(insights, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")
    return {"trend_status": meta.get("status"), "trend_reason": meta.get("reason"),
            "topics": insights["run_summary"]["topics_discovered"],
            "cross_video_topics": len(insights["topics_across_multiple_videos"]),
            "dedicated_video_candidates": len(insights["repeated_questions_needing_a_dedicated_video"])}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Audience insights and topic trends.")
    p.add_argument("--in-dir", default=str(PROCESSED_DATA_DIR))
    p.add_argument("--out-dir", default=str(PROCESSED_DATA_DIR))
    p.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    a = p.parse_args(argv)
    print(json.dumps(run(Path(a.out_dir), Path(a.in_dir), a.window_days), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
