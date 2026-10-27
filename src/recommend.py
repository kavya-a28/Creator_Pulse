"""
CreatorPulse — Stage 4: content recommendations.

Turns recurring audience questions, requests and problems (the topics from
src.topics) into video ideas. Every number and every quoted comment comes
from the stage outputs; nothing is invented. No LLM is used: titles and
formats come from transparent rules, so the pipeline needs no paid API.

Inputs
    data/processed/topic_summary.csv, topic_clusters.csv (src.topics)
    data/processed/topic_trends.csv, audience_insights.json (src.insights)

Outputs
    data/processed/content_recommendations.csv
    data/processed/content_recommendations.json

Evidence counting
    * Spam never reaches this stage (excluded in src.topics).
    * Near-duplicate comments count once (unique comments).
    * Several comments by the same author_channel_id count once (commenters).
      When author IDs are missing, unique comments are used instead.
    A topic needs >= MIN_EVIDENCE distinct commenters (default 3) to be
    recommended; smaller topics are listed in the JSON as "not enough evidence".

Priority score (0-100) = 100 x weighted sum of four 0-1 components:
    frequency   0.40  log(1 + commenters) / log(1 + 20)   (20+ commenters = 1)
    recurrence  0.25  (videos - 1) / 4                    (5+ videos = 1)
    recency     0.15  exp(-days since the topic's latest comment / 14),
                      days counted back from the newest comment in the data
    intent      0.20  mean over unique comments of Confusion/Problem 1.0,
                      Request 0.9, Question 0.8 (people stuck first)
    A topic whose trend status is "growing" gets recency = 1. With the current
    data no trend could be established, so this never applies yet.

Overlap control: candidate topics whose combined comment texts are similar
(cosine >= MERGE_SIMILARITY with the topic features) are merged into one
recommendation, so the same idea is not suggested twice.

Usage:
    python -m src.recommend
    python -m src.recommend --min-evidence 2 --top 15
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR  # noqa: E402
from src.preprocess import normalize  # noqa: E402
from src.topics import UNCLUSTERED_ID, build_features  # noqa: E402

CSV_ENCODING = "utf-8-sig"
ACTIONABLE = ["Question", "Request", "Confusion/Problem"]
INTENT_WEIGHT = {"Confusion/Problem": 1.0, "Request": 0.9, "Question": 0.8}
WEIGHTS = {"frequency": 0.40, "recurrence": 0.25, "recency": 0.15, "intent": 0.20}
FREQ_SATURATION = 20
RECURRENCE_SATURATION_VIDEOS = 5
RECENCY_HALF_LIFE_DAYS = 14          # exp(-days / 14)
MIN_EVIDENCE = 3
MERGE_SIMILARITY = 0.35
MAX_EXAMPLES = 5

OUT_CSV = PROCESSED_DATA_DIR / "content_recommendations.csv"
OUT_JSON = PROCESSED_DATA_DIR / "content_recommendations.json"

# Keyword signals (matched on normalised text) used to pick a content format.
SIGNALS = {
    "problem": r"\b(error|errors|issue|issues|problem|problems|fix|bug|crash|stuck|not working|nahi chal|"
               r"kaam nahi|samajh nahi|nahi aa raha|nahi ho raha|failed|fail)\b",
    "comparison": r"\b(vs|versus|difference|better|best|compare|comparison|which one|konsa|kaunsa)\b",
    "series": r"\b(course|courses|series|playlist|full|complete|basic to advanced|zero to hero|bootcamp|batch)\b",
    "practice": r"\b(practice|questions|exercise|exercises|pyq|problems to solve|mock|quiz)\b",
    "career": r"\b(job|jobs|placement|placements|career|fresher|freshers|degree|college|salary|internship|"
              r"graduated|graduate|class|11th|12th|bca|btech|mca|student|students|roadmap|start)\b",
}
FORMATS = {
    "debugging walkthrough": "{t}: Common Problems and How to Fix Them",
    "comparison": "{t}: Which One Should You Choose?",
    "tutorial series": "{t} Full Course for Beginners",
    "practice session": "{t} Practice Session: Solving Your Questions",
    "roadmap / career Q&A": "{t}: Roadmap and Your Questions Answered",
    "concept explanation": "{t} Explained: Answering Your Questions",
    "tutorial": "{t}: A Complete Guide",
}


# Topics named after a shared need (see topics.CONCEPTS) get a title written for that need.
CONCEPT_TITLES = {
    "Study materials": "{x}Study Materials: Notes, Books and Resources",
    "Is the course still relevant?": "Is This Course Still Relevant? What Has Changed Since It Was Recorded",
    "Study partners and classmates": "Find Study Partners: Learning This Course Together",
    "Note-taking app": "My Note-Taking Setup: The App I Use and Why",
}
_QUESTION_START = re.compile(r"^(is|are|can|could|do|does|did|how|what|why|which|when|where|should|will|who|"
                             r"i am|i'm|i have|for)\b", re.I)


def make_title(name: str, fmt: str) -> str:
    """Video title for a topic: need-specific titles, the viewers' own question, or a format template."""
    for label, tmpl in CONCEPT_TITLES.items():
        if name == label or name.startswith(label + ":"):
            other = name[len(label) + 1:].strip()
            return tmpl.format(x=f"{other} " if other else "")
    if name.endswith("?") or _QUESTION_START.match(name):
        q = name if name.endswith("?") else name + "?"
        return f"{q[:1].upper() + q[1:]} Your Questions Answered"
    return FORMATS[fmt].format(t=name.replace(" / ", " & "))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def signal_counts(texts: List[str]) -> Dict[str, int]:
    norm = [normalize(t) for t in texts]
    return {k: sum(bool(re.search(p, t)) for t in norm) for k, p in SIGNALS.items()}


def choose_format(intent_counts: Dict[str, int], sig: Dict[str, int], n: int) -> str:
    """Pick a format from intent mix and keyword signals (share of unique comments)."""
    share = {k: v / n for k, v in sig.items()} if n else {}
    problems = intent_counts.get("Confusion/Problem", 0)
    if share.get("comparison", 0) >= 0.5:
        return "comparison"
    if problems / max(n, 1) >= 0.4 or share.get("problem", 0) >= 0.4:
        return "debugging walkthrough"
    main = max(intent_counts, key=lambda k: (intent_counts[k], k)) if intent_counts else "Question"
    if share.get("practice", 0) >= 0.4:
        return "practice session"
    if main == "Question" and share.get("career", 0) >= 0.5:
        return "roadmap / career Q&A"
    if share.get("comparison", 0) >= 0.5:
        return "comparison"
    if share.get("career", 0) >= 0.5:
        return "roadmap / career Q&A"
    if main == "Request":
        return "tutorial series" if share.get("series", 0) >= 0.3 else "tutorial"
    return "concept explanation"


def evidence_rows(clusters: pd.DataFrame, topic_ids: List[int]) -> pd.DataFrame:
    g = clusters[clusters["topic_id"].isin(topic_ids) & clusters["predicted_category"].isin(ACTIONABLE)]
    return g


def evidence_counts(g: pd.DataFrame) -> dict:
    u = g[g["is_canonical"]]
    has_author = (g["author_channel_id"] != "").any()
    commenters = g.loc[g["author_channel_id"] != "", "author_channel_id"].nunique() if has_author else None
    return {"n_comments_raw": int(len(g)), "n_unique_comments": int(len(u)),
            "n_unique_commenters": int(commenters) if commenters is not None else None,
            "effective_count": int(commenters) if commenters is not None else int(len(u)),
            "counted_by": "distinct commenter IDs" if commenters is not None else "unique comments (no commenter IDs)",
            "n_videos": int(g["video_id"].nunique()),
            "intent_counts": {k: int(v) for k, v in u["predicted_category"].value_counts().items()}}


def score(ev: dict, latest, data_end, trend_status: str) -> dict:
    freq = min(1.0, math.log1p(ev["effective_count"]) / math.log1p(FREQ_SATURATION))
    rec = min(1.0, (ev["n_videos"] - 1) / (RECURRENCE_SATURATION_VIDEOS - 1))
    if trend_status == "growing":
        recency, days = 1.0, None
    elif latest is None or pd.isna(latest) or data_end is None:
        recency, days = 0.0, None
    else:
        days = max(0.0, (data_end - latest).total_seconds() / 86400)
        recency = math.exp(-days / RECENCY_HALF_LIFE_DAYS)
    n = sum(ev["intent_counts"].values())
    intent = sum(INTENT_WEIGHT.get(k, 0) * v for k, v in ev["intent_counts"].items()) / n if n else 0.0
    comps = {"frequency": freq, "recurrence": rec, "recency": recency, "intent": intent}
    total = 100 * sum(WEIGHTS[k] * v for k, v in comps.items())
    return {"priority_score": round(total, 1), "components": {k: round(v, 3) for k, v in comps.items()},
            "contributions": {k: round(100 * WEIGHTS[k] * v, 1) for k, v in comps.items()},
            "days_since_latest_comment": round(days, 1) if days is not None else None}


def key_terms(topic_row) -> set:
    """The words a topic is about: its top two terms (single words of them)."""
    terms = [t.strip() for t in str(topic_row.get("top_terms", "") or "").split(";") if t.strip()][:2]
    return {w for t in terms for w in t.split()}


def supporting_commenters(g: pd.DataFrame, topic_row) -> int:
    """Distinct people whose comment in this topic actually contains one of its key terms.
    A recommendation needs this many supporters, so it never rests on comments that were grouped
    together by an incidental word."""
    keys = key_terms(topic_row)
    if not keys or "topic_text" not in g:
        return evidence_counts(g)["effective_count"]
    ok = g[g["topic_text"].fillna("").map(lambda s: bool(set(s.split()) & keys))]     # copies by others count too
    authors = ok.loc[ok["author_channel_id"] != "", "author_channel_id"]
    return int(authors.nunique()) if len(authors) else int(len(ok))


def pick_examples(g: pd.DataFrame, k: int = MAX_EXAMPLES) -> pd.DataFrame:
    """Up to k real comments: one per duplicate group and per author, most central first."""
    u = g[g["is_canonical"]].copy()
    u["_sim"] = pd.to_numeric(u["similarity_to_topic"], errors="coerce").fillna(0)
    u["_rep"] = _bool(u["is_representative"]).fillna(False)
    # Prefer informative comments (>= 2 topic words) over one-word ones like "python".
    u["_informative"] = u["topic_text"].fillna("").str.split().str.len() >= 2 if "topic_text" in u else True
    u = u.sort_values(["_rep", "_informative", "_sim", "comment_id"], ascending=[False, False, False, True])
    picked, authors = [], set()
    for _, r in u.iterrows():
        a = r["author_channel_id"]
        if a and a in authors:
            continue
        picked.append(r)
        if a:
            authors.add(a)
        if len(picked) == k:
            break
    return pd.DataFrame(picked)


def merge_groups(cands: pd.DataFrame, clusters: pd.DataFrame, threshold: float = MERGE_SIMILARITY) -> List[List[int]]:
    """Group candidate topics whose combined texts are similar (union-find on topic pairs)."""
    ids = cands["topic_id"].tolist()
    if len(ids) < 2:
        return [[i] for i in ids]
    docs = [" ".join(clusters.loc[clusters["topic_id"] == t, "text"]) for t in ids]
    X, _ = build_features(docs, "word+char")
    S = (X @ X.T).toarray()
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] >= threshold:
                a, b = find(i), find(j)
                parent[max(a, b)] = min(a, b)
    groups: Dict[int, List[int]] = {}
    for i, t in enumerate(ids):
        groups.setdefault(find(i), []).append(t)
    return list(groups.values())


def recommend(topics: pd.DataFrame, clusters: pd.DataFrame, trends: pd.DataFrame, data_end,
              min_evidence: int = MIN_EVIDENCE, top: int = 20):
    topics = topics.copy()
    topics["topic_id"] = topics["topic_id"].astype(int)
    real = topics[topics["topic_id"] != UNCLUSTERED_ID]
    status = dict(zip(trends["topic_id"].astype(int), trends["status"])) if len(trends) else {}

    # Candidates: topics whose unique comments are mostly actionable and have enough distinct people.
    cands, rejected = [], []
    for _, t in real.iterrows():
        g = evidence_rows(clusters, [t["topic_id"]])
        ev = evidence_counts(g)
        allu = clusters[(clusters["topic_id"] == t["topic_id"]) & clusters["is_canonical"]]
        actionable_share = ev["n_unique_comments"] / max(len(allu), 1)
        support = supporting_commenters(g, t)
        if str(t.get("topic_kind", "")) == "check-in":
            rejected.append({"topic_id": int(t["topic_id"]), "topic_name": t["topic_name"],
                             "reason": "viewer check-ins (e.g. 'who is starting in 2026?'), not a content need"})
        elif actionable_share < 0.5:
            rejected.append({"topic_id": int(t["topic_id"]), "topic_name": t["topic_name"],
                             "reason": "mostly non-actionable comments"})
        elif ev["effective_count"] < min_evidence:
            rejected.append({"topic_id": int(t["topic_id"]), "topic_name": t["topic_name"],
                             "reason": f"not enough evidence: {ev['effective_count']} {ev['counted_by']} "
                                       f"(need {min_evidence})"})
        elif support < min_evidence:
            rejected.append({"topic_id": int(t["topic_id"]), "topic_name": t["topic_name"],
                             "reason": f"comments don't consistently mention the topic: only {support} different "
                                       f"people's comments contain its key terms (need {min_evidence})"})
        else:
            cands.append(t)
    cands = pd.DataFrame(cands)
    if cands.empty:
        return [], rejected

    recs = []
    for group in merge_groups(cands, clusters):
        g = evidence_rows(clusters, group)
        ev = evidence_counts(g)
        sub = cands[cands["topic_id"].isin(group)].sort_values("n_unique_comments", ascending=False,
                                                                 key=lambda s: s.astype(int))
        lead = sub.iloc[0]
        ts = pd.to_datetime(g["published_at"], utc=True, errors="coerce")
        latest = ts.max() if ts.notna().any() else None
        tstat = "growing" if any(status.get(t) == "growing" for t in group) else (
            status.get(int(lead["topic_id"]), "unknown"))
        sc = score(ev, latest, data_end, tstat)
        u_texts = g.loc[g["is_canonical"], "text"].tolist()
        sig = signal_counts(u_texts)
        fmt = choose_format(ev["intent_counts"], sig, ev["n_unique_comments"])
        name = lead["topic_name"]
        related = [n for n in sub["topic_name"].tolist()[1:]]
        main_intent = max(ev["intent_counts"], key=lambda k: (ev["intent_counts"][k], INTENT_WEIGHT[k]))
        title = make_title(name, fmt)
        ex = pick_examples(g)
        channels = [c for c in dict.fromkeys(g["channel_title"]) if c]
        verb = {"Request": "asked for content on", "Question": "asked questions about",
                "Confusion/Problem": "reported problems with"}[main_intent]
        mix = ", ".join(f"{v} {k}" for k, v in sorted(ev["intent_counts"].items(), key=lambda kv: -kv[1]))
        top2 = sorted(sc["contributions"].items(), key=lambda kv: -kv[1])[:2]
        reason = (f"{ev['effective_count']} different commenters {verb} this ({ev['n_unique_comments']} unique "
                  f"comments, {ev['n_comments_raw']} in total) across {ev['n_videos']} video(s)"
                  f"{' on ' + ', '.join(channels) if channels else ''}. Intent mix: {mix}. "
                  + (f"Latest comment {sc['days_since_latest_comment']:.0f} day(s) before the end of the data. "
                     if sc["days_since_latest_comment"] is not None else "")
                  + f"Trend: {tstat.replace('_', ' ')}. Score driven mostly by "
                  + " and ".join(f"{k} (+{v})" for k, v in top2) + "."
                  + (f" Merged with related topic(s): {', '.join(related)}." if related else ""))
        recs.append({
            "suggested_title": title, "topic_name": name, "topic_ids": [int(t) for t in sub["topic_id"]],
            "related_topics_merged": related, "main_intent": main_intent, "suggested_format": fmt,
            **sc, **{k: ev[k] for k in ("n_comments_raw", "n_unique_comments", "n_unique_commenters",
                                         "counted_by", "n_videos", "intent_counts")},
            "channels": channels,
            "video_titles": [v for v in dict.fromkeys(g["video_title"]) if v],
            "latest_comment_at": latest.isoformat() if latest is not None else None,
            "trend_status": tstat, "format_signals": sig, "reason": reason,
            "representative_comments": [{"comment_id": r["comment_id"], "video_id": r["video_id"],
                                         "predicted_category": r["predicted_category"], "text": r["text"]}
                                        for _, r in ex.iterrows()],
            "evidence_comment_ids": sorted(g["comment_id"].tolist()),
        })
    recs.sort(key=lambda r: (-r["priority_score"], r["topic_name"]))
    recs = recs[:top]
    for i, r in enumerate(recs, 1):
        r["rank"] = i
        r["recommendation_id"] = f"rec_{i:02d}"
    return recs, rejected


CSV_COLUMNS = ["rank", "recommendation_id", "suggested_title", "topic_name", "topic_ids", "related_topics_merged",
               "main_intent", "suggested_format", "priority_score", "score_frequency", "score_recurrence",
               "score_recency", "score_intent", "n_comments_raw", "n_unique_comments", "n_unique_commenters",
               "n_videos", "channels", "latest_comment_at", "trend_status", "reason", "representative_comments",
               "representative_comment_ids", "evidence_comment_ids"]


def to_frame(recs: List[dict]) -> pd.DataFrame:
    if not recs:
        return pd.DataFrame(columns=CSV_COLUMNS)    # header-only file, still readable
    rows = []
    for r in recs:
        rows.append({
            "rank": r["rank"], "recommendation_id": r["recommendation_id"], "suggested_title": r["suggested_title"],
            "topic_name": r["topic_name"], "topic_ids": ";".join(map(str, r["topic_ids"])),
            "related_topics_merged": "; ".join(r["related_topics_merged"]),
            "main_intent": r["main_intent"], "suggested_format": r["suggested_format"],
            "priority_score": r["priority_score"],
            **{f"score_{k}": v for k, v in r["contributions"].items()},
            "n_comments_raw": r["n_comments_raw"], "n_unique_comments": r["n_unique_comments"],
            "n_unique_commenters": r["n_unique_commenters"] if r["n_unique_commenters"] is not None else "",
            "n_videos": r["n_videos"], "channels": " | ".join(r["channels"]),
            "latest_comment_at": r["latest_comment_at"], "trend_status": r["trend_status"],
            "reason": r["reason"],
            "representative_comments": " || ".join(c["text"].replace("\n", " ") for c in r["representative_comments"]),
            "representative_comment_ids": ";".join(c["comment_id"] for c in r["representative_comments"]),
            "evidence_comment_ids": ";".join(r["evidence_comment_ids"]),
        })
    return pd.DataFrame(rows)


def run(in_dir: Path = PROCESSED_DATA_DIR, out_dir: Path = PROCESSED_DATA_DIR,
        min_evidence: int = MIN_EVIDENCE, top: int = 20) -> dict:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    for f in ("topic_summary.csv", "topic_clusters.csv"):
        if not (in_dir / f).exists():
            raise SystemExit(f"[ERROR] {in_dir / f} not found; run src.topics first")
    topics = read_csv(in_dir / "topic_summary.csv")
    clusters = read_csv(in_dir / "topic_clusters.csv")
    clusters["topic_id"] = clusters["topic_id"].astype(int)
    clusters["is_canonical"] = _bool(clusters["is_canonical"]).fillna(True)
    trends = read_csv(in_dir / "topic_trends.csv") if (in_dir / "topic_trends.csv").exists() else pd.DataFrame()
    data_end = None
    ins_path = in_dir / "audience_insights.json"
    if ins_path.exists():
        end = json.loads(ins_path.read_text(encoding="utf-8")).get("trends", {}).get("data_end")
        data_end = pd.Timestamp(end) if end else None
    if data_end is None:
        ts = pd.to_datetime(clusters["published_at"], utc=True, errors="coerce")
        data_end = ts.max() if ts.notna().any() else None
    recs, rejected = recommend(topics, clusters, trends, data_end, min_evidence, top)

    # Traceability check: every quoted comment and evidence id exists in the topic file.
    known = set(clusters["comment_id"])
    for r in recs:
        ids = [c["comment_id"] for c in r["representative_comments"]] + r["evidence_comment_ids"]
        bad = [i for i in ids if i not in known]
        if bad:
            raise SystemExit(f"[ERROR] {r['recommendation_id']} cites unknown comments {bad[:3]}")
        texts = dict(zip(clusters["comment_id"], clusters["text"]))
        if any(texts[c["comment_id"]] != c["text"] for c in r["representative_comments"]):
            raise SystemExit(f"[ERROR] {r['recommendation_id']} quotes text that differs from the source")

    out_dir.mkdir(parents=True, exist_ok=True)
    to_frame(recs).to_csv(out_dir / OUT_CSV.name, index=False, encoding=CSV_ENCODING)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {"weights": WEIGHTS, "intent_weights": INTENT_WEIGHT, "min_evidence": min_evidence,
                   "frequency_saturation": FREQ_SATURATION,
                   "recurrence_saturation_videos": RECURRENCE_SATURATION_VIDEOS,
                   "recency_scale_days": RECENCY_HALF_LIFE_DAYS, "merge_similarity": MERGE_SIMILARITY,
                   "llm_used": False,
                   "titles": "rule-based templates per format; edit freely - they are suggestions"},
        "data_end": data_end.isoformat() if data_end is not None else None,
        "n_recommendations": len(recs),
        "recommendations": recs,
        "topics_not_recommended": rejected,
    }
    (out_dir / OUT_JSON.name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    return {"recommendations": len(recs), "topics_not_recommended": len(rejected)}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Generate content recommendations from audience topics.")
    p.add_argument("--in-dir", default=str(PROCESSED_DATA_DIR))
    p.add_argument("--out-dir", default=str(PROCESSED_DATA_DIR))
    p.add_argument("--min-evidence", type=int, default=MIN_EVIDENCE)
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args(argv)
    print(json.dumps(run(Path(a.in_dir), Path(a.out_dir), a.min_evidence, a.top), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
