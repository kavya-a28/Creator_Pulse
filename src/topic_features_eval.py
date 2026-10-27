"""
CreatorPulse — which text representation should topic discovery use?

Compares word TF-IDF, character n-gram TF-IDF, their combination (the
default in src.topics) and, when installed, multilingual sentence embeddings.

Two kinds of evidence:

1. Probe pairs (hand-written for this diagnostic, NOT audience comments and
   never used as evidence anywhere else). Same-topic pairs that differ in
   language, script, spelling or shorthand, and different-topic pairs that
   share a keyword. Reported: mean similarity of each kind, ROC AUC (how well
   similarity separates them) and how many pairs land on the right side of
   the clustering threshold.
2. The real comments: number of topics, share of comments clustered and mean
   similarity of members to their topic, when src.topics runs with each
   representation on data/processed/comments_predicted.csv.

Output: reports/topic_feature_evaluation.md (and .json)

Usage:
    python -m src.topic_features_eval
    python -m src.topic_features_eval --embeddings   # also try multilingual embeddings
                                                     # (downloads ~470 MB the first time)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR  # noqa: E402
from src.topics import CLUSTER_THRESHOLD, build_features, discover, read_csv  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"

# (text_a, text_b, kind) — kind is what the pair is meant to test.
SAME_TOPIC = [
    ("Python indentation error aa raha hai", "How can I fix indentation error in Python?", "hinglish vs english"),
    ("Python indentation error aa raha hai", "Indentation issue in my Python code", "paraphrase"),
    ("How can I fix indentation error in Python?", "Indentation issue in my Python code", "paraphrase"),
    ("pyhton indentation eror", "python indentation error", "spelling"),
    ("python ka course bana do please", "please make a python course", "hinglish vs english"),
    ("java me dsa series laao", "DSA in Java please", "hinglish vs english"),
    ("stock market course launch karo", "please make a trading course", "synonym"),
    ("machine learning ka video", "ML course please", "abbreviation"),
    ("notes ka link kaam nahi kar raha", "notes link is not working", "hinglish vs english"),
    ("मुझे पायथन सीखनी है", "I want to learn python", "devanagari vs english"),
    ("react hooks samajh nahi aaya", "confused about react hooks", "hinglish vs english"),
    ("kali linux ka video dalo", "kali linux tutorial please", "hinglish vs english"),
    ("system design course please", "please make a system design playlist", "paraphrase"),
    ("cyber security course", "cybersecurity batch kab aayega", "spacing variant"),
    ("devops roadmap please", "dev ops kaise start kare", "spacing variant"),
    ("AI ML course series", "artificial intelligence and machine learning course", "abbreviation"),
]
DIFFERENT_TOPIC = [
    ("Python indentation error aa raha hai", "please make a python course", "shared keyword"),
    ("java course please", "java or python which is better for placements?", "shared keyword"),
    ("system design course please", "game design course please", "shared keyword"),
    ("AI course launch karo", "will AI replace software jobs?", "shared keyword"),
    ("stock market course", "market research job kaise milegi", "shared keyword"),
    ("data structures course", "data science roadmap", "shared keyword"),
    ("react course please", "react native app crash ho raha hai", "shared keyword"),
    ("how did you get my phone number?", "coding ke liye kaun sa phone lu?", "shared keyword"),
    ("machine learning course", "washing machine repair video", "shared keyword"),
    ("notes ka link do", "linkedin profile kaise banaye", "unrelated"),
    ("dynamic programming lectures", "hackathon preparation tips", "unrelated"),
    ("cloud and devops course", "BCA ke baad kya karu", "unrelated"),
]


def _auc(pos, neg):
    """Probability a random same-topic pair scores higher than a different-topic pair."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def probe_scores(kind: str) -> dict:
    texts = [t for a, b, _ in SAME_TOPIC + DIFFERENT_TOPIC for t in (a, b)]
    X, _ = build_features(texts, kind)
    sims = [float((X[2 * i] @ X[2 * i + 1].T).toarray()[0, 0]) for i in range(len(texts) // 2)]
    pos, neg = sims[:len(SAME_TOPIC)], sims[len(SAME_TOPIC):]
    return {
        "mean_similarity_same_topic": round(float(np.mean(pos)), 3),
        "mean_similarity_different_topic": round(float(np.mean(neg)), 3),
        "roc_auc": round(_auc(pos, neg), 3),
        "same_topic_pairs_above_threshold": f"{sum(p >= CLUSTER_THRESHOLD for p in pos)}/{len(pos)}",
        "different_topic_pairs_below_threshold": f"{sum(n < CLUSTER_THRESHOLD for n in neg)}/{len(neg)}",
        "pairs": [{"a": a, "b": b, "kind": k, "same_topic": i < len(SAME_TOPIC), "similarity": round(s, 3)}
                  for i, ((a, b, k), s) in enumerate(zip(SAME_TOPIC + DIFFERENT_TOPIC, sims))],
    }


def real_data_scores(kind: str) -> dict:
    path = PROCESSED_DATA_DIR / "comments_predicted.csv"
    if not path.exists():
        return {"skipped": "run src.batch_predict first"}
    clusters, summary, stats = discover(read_csv(path), features=kind)
    topics = summary[summary["topic_id"] != -1]
    return {"topics": stats["topics"], "clustered_comments": stats["clustered_comments"],
            "unclustered_comments": stats["unclustered_comments"],
            "clustered_share": round(stats["clustered_comments"] / max(stats["selected_comments"], 1), 3),
            "mean_member_similarity": round(float(topics["mean_similarity_to_topic"].astype(float).mean()), 3)
            if len(topics) else None,
            "largest_topics": topics.nlargest(5, "n_unique_comments")["topic_name"].tolist()}


def embeddings_available() -> str:
    try:
        from src.topics import embed
        embed(["test"])
        return ""
    except Exception as exc:  # package or model missing / no network
        return f"{type(exc).__name__}: {str(exc)[:160]}"


def main(argv=None) -> int:
    import argparse
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Compare topic feature representations.")
    ap.add_argument("--embeddings", action="store_true",
                    help="also evaluate multilingual sentence embeddings (needs sentence-transformers)")
    args = ap.parse_args(argv)
    kinds = ["word", "char", "word+char"]
    why_not = embeddings_available() if args.embeddings else "not requested (run with --embeddings)"
    if not why_not:
        kinds.append("embeddings")
    out = {"threshold": CLUSTER_THRESHOLD, "embeddings_unavailable_reason": why_not or None, "results": {}}
    for k in kinds:
        out["results"][k] = {"probes": probe_scores(k), "real_data": real_data_scores(k)}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "topic_feature_evaluation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")
    md = ["# Topic features: comparison", "",
          f"Probe pairs are hand-written diagnostics ({len(SAME_TOPIC)} same-topic, {len(DIFFERENT_TOPIC)} "
          "different-topic sharing a keyword), not audience comments. Clustering threshold "
          f"{CLUSTER_THRESHOLD}.", ""]
    if why_not:
        md += [f"Multilingual embeddings were not evaluated: {why_not}", ""]
    md += ["| Features | Same-topic sim | Diff-topic sim | ROC AUC | Same >= thr | Diff < thr "
           "| Real: topics | Real: clustered | Real: member sim |", "|---" * 9 + "|"]
    for k, r in out["results"].items():
        p, d = r["probes"], r["real_data"]
        md.append(f"| {k} | {p['mean_similarity_same_topic']} | {p['mean_similarity_different_topic']} | "
                  f"{p['roc_auc']} | {p['same_topic_pairs_above_threshold']} | "
                  f"{p['different_topic_pairs_below_threshold']} | {d.get('topics')} | "
                  f"{d.get('clustered_share')} | {d.get('mean_member_similarity')} |")
    md += ["", "Per-pair similarity:", "", "| Pair | Kind | Same topic | " + " | ".join(out["results"]) + " |",
           "|---" * (3 + len(out["results"])) + "|"]
    for i, pair in enumerate(out["results"]["word"]["probes"]["pairs"]):
        md.append(f"| {pair['a']} / {pair['b']} | {pair['kind']} | {'yes' if pair['same_topic'] else 'no'} | "
                  + " | ".join(str(out['results'][k]['probes']['pairs'][i]['similarity']) for k in out["results"])
                  + " |")
    (REPORTS / "topic_feature_evaluation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
