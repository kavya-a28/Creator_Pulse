"""Tests for the audience-intelligence stages (batch prediction, topics,
insights/trends, recommendations) and the pipeline runner.

Everything runs on small hand-made fixtures in a temporary folder; the real
data, model and reports are never touched.
"""

import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import pytest

from src import batch_predict, insights, pipeline, recommend, topics
from src.train_svm import build_pipeline

CLASSES = ["Question", "Request", "Confusion/Problem", "Praise", "Spam", "Other"]
TRAIN = [
    ("what is the difference between list and tuple ?", "Question"),
    ("which language should i learn first ?", "Question"),
    ("how does recursion work ?", "Question"),
    ("please make a video on django", "Request"),
    ("please make a python course", "Request"),
    ("sir upload notes please", "Request"),
    ("indentation error aa raha hai", "Confusion/Problem"),
    ("my code is not working error", "Confusion/Problem"),
    ("samajh nahi aaya stuck", "Confusion/Problem"),
    ("best video thank you ❤", "Praise"),
    ("thank you bhaiya great explanation", "Praise"),
    ("earn money daily dm me on telegram", "Spam"),
    ("check my channel subscribe link", "Spam"),
    ("day 5 done", "Other"),
    ("who is watching in 2026", "Other"),
]


@pytest.fixture()
def model_path(tmp_path):
    texts, labels = zip(*TRAIN)
    m = build_pipeline(min_df=1).fit(list(texts), list(labels))
    p = tmp_path / "model.joblib"
    joblib.dump(m, p)
    return p


@pytest.fixture(autouse=True)
def isolate_labels(tmp_path, monkeypatch):
    # Never read the real labelled file / training split in these tests.
    monkeypatch.setattr(batch_predict, "LABELLED", tmp_path / "none_labelled.csv")
    monkeypatch.setattr(batch_predict, "TRAIN_SPLIT", tmp_path / "none_train.csv")


def _ts(days_ago, end=datetime(2026, 10, 24, tzinfo=timezone.utc)):
    return (end - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_raw(raw_dir, rows, name="comments_test.csv", columns=None):
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if columns:
        df = df[columns]
    df.to_csv(raw_dir / name, index=False, encoding="utf-8-sig")
    return raw_dir / name


def basic_rows():
    rows = []
    texts = [
        "Python indentation error aa raha hai", "How can I fix indentation error in Python?",
        "Indentation issue in my Python code", "please make a django course", "django course please sir",
        "django ka course bana do", "best video thank you ❤", "earn money daily dm me on telegram",
        "day 5 done", "what is a tuple?",
    ]
    for i, t in enumerate(texts):
        rows.append({"comment_id": f"c{i}", "video_id": f"v{i % 3}", "text": t,
                     "author_channel_id": f"a{i}", "published_at": _ts(i), "like_count": 0,
                     "reply_count": 0, "creator_replied": "False", "channel_id": "ch",
                     "channel_title": "Chan", "video_title": f"Video {i % 3}",
                     "video_published_at": _ts(30)})
    return rows


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------

def test_batch_predict_handles_missing_empty_and_duplicates(tmp_path, model_path):
    rows = basic_rows()
    rows += [
        {**rows[0], "comment_id": "c0"},                  # duplicate id
        {**rows[1], "comment_id": "e1", "text": ""},      # empty
        {**rows[1], "comment_id": "e2", "text": "   "},   # whitespace
        {**rows[1], "comment_id": "e3", "text": "...,,"},  # no usable tokens
        {**rows[1], "comment_id": "", "text": "no id"},   # missing id
        {**rows[1], "comment_id": "t1", "published_at": "not a date"},
    ]
    raw = tmp_path / "raw"
    write_raw(raw, rows)
    # A second file with no channel/title columns at all.
    write_raw(raw, [{"comment_id": "x1", "video_id": "vx", "text": "please make a react course",
                     "author_channel_id": "ax", "published_at": _ts(1)}], name="comments_other.csv")
    write_raw(raw, [{"video_id": "vx", "title": "React video", "published_at": _ts(9)}],
              name="channel_videos_abc.csv")
    out = tmp_path / "out"
    stats = batch_predict.run(model_path, out_dir=out, raw_dir=raw)
    pred = pd.read_csv(out / "comments_predicted.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    excl = pd.read_csv(out / "comments_excluded.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert stats["input_rows"] == len(rows) + 1
    assert stats["predicted"] + stats["excluded"] == stats["input_rows"]
    assert sorted(excl["excluded_reason"]) == sorted(
        ["duplicate_comment_id", "empty_text", "empty_text", "no_usable_tokens", "missing_comment_id"])
    assert pred["comment_id"].is_unique
    assert set(pred["predicted_category"]) <= set(CLASSES)
    x1 = pred[pred["comment_id"] == "x1"].iloc[0]
    assert x1["channel_title"] == "unknown" and x1["video_title"] == "React video"   # filled from catalogue
    assert "channel_title" in x1["metadata_missing"]
    assert "published_at_unparseable" in pred.loc[pred["comment_id"] == "t1", "metadata_missing"].iloc[0]
    summ = pd.read_csv(out / "category_summary.csv")
    assert summ["comments"].sum() == len(pred) and abs(summ["percent"].sum() - 100) < 0.1


def test_batch_predict_never_refits_the_vectoriser(tmp_path, model_path):
    before = joblib.load(model_path)
    vocab, idf = dict(before.named_steps["tfidf"].vocabulary_), before.named_steps["tfidf"].idf_.copy()
    raw = tmp_path / "raw"
    write_raw(raw, basic_rows())
    batch_predict.run(model_path, out_dir=tmp_path / "out", raw_dir=raw)
    after = joblib.load(model_path)
    assert after.named_steps["tfidf"].vocabulary_ == vocab
    assert np.array_equal(after.named_steps["tfidf"].idf_, idf)


def test_batch_predict_matches_direct_model_predictions(tmp_path, model_path):
    raw = tmp_path / "raw"
    write_raw(raw, basic_rows())
    batch_predict.run(model_path, out_dir=tmp_path / "out", raw_dir=raw)
    pred = pd.read_csv(tmp_path / "out" / "comments_predicted.csv", dtype=str, keep_default_na=False,
                       encoding="utf-8-sig")
    direct = joblib.load(model_path).predict(pred["text"].tolist())
    assert list(pred["predicted_category"]) == list(direct)


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------

def frame(texts, cats, authors=None, videos=None, days=None):
    n = len(texts)
    return pd.DataFrame({
        "comment_id": [f"k{i}" for i in range(n)], "text": texts, "predicted_category": cats,
        "author_channel_id": authors or [f"a{i}" for i in range(n)],
        "video_id": videos or [f"v{i % 2}" for i in range(n)],
        "video_title": "t", "channel_title": "Chan", "channel_id": "ch",
        "published_at": [_ts(d) for d in (days or range(n))],
        "creator_replied": "False", "reply_count": "0", "like_count": "0",
        "seen_in_training": "False", "low_confidence": "False",
    })


def test_near_duplicates_grouped():
    g = topics.near_duplicate_groups([
        "Please make a video on Django REST framework", "please make a video on django rest framework!!",
        "Please make a video on Django REST framework 🙏", "How do I install numpy?"])
    assert g[0] == g[1] == g[2] and g[3] != g[0]


def test_indentation_example_forms_one_topic_and_requests_stay_separate():
    df = frame(["Python indentation error aa raha hai", "How can I fix indentation error in Python?",
                "Indentation issue in my Python code", "please make a python course",
                "python ka full course bana do", "python course chahiye sir", "what is the capital of peru"],
               ["Confusion/Problem", "Question", "Confusion/Problem", "Request", "Request", "Request", "Question"])
    cl, summ, _ = topics.discover(df)
    t = dict(zip(cl["comment_id"], cl["topic_id"]))
    assert t["k0"] == t["k1"] == t["k2"] != topics.UNCLUSTERED_ID
    assert t["k3"] == t["k4"] == t["k5"] != topics.UNCLUSTERED_ID
    assert t["k0"] != t["k3"]
    assert t["k6"] == topics.UNCLUSTERED_ID
    name = summ.loc[summ["topic_id"] == t["k0"], "topic_name"].iloc[0]
    assert "Indentation" in name


def test_shared_keyword_alone_does_not_make_a_topic():
    df = frame(["python indentation error", "please make a python course"], ["Confusion/Problem", "Request"])
    cl, _, _ = topics.discover(df)
    # Only "python" is shared and the intents differ -> no topic of size >= 2.
    assert (cl["topic_id"] == topics.UNCLUSTERED_ID).all()


def test_dominant_word_split():
    df = frame(["dynamic programming lectures please", "more dynamic programming videos",
                "which programming language is best", "best programming language for dsa"],
               ["Request", "Request", "Question", "Question"])
    cl, summ, _ = topics.discover(df)
    t = dict(zip(cl["comment_id"], cl["topic_id"]))
    assert t["k0"] == t["k1"] and t["k2"] == t["k3"] and t["k0"] != t["k2"]


def test_duplicates_not_double_counted_and_spam_excluded():
    texts = ["please make a video on kubernetes"] * 3 + ["kubernetes tutorial please sir", "earn money dm telegram"]
    df = frame(texts, ["Request"] * 4 + ["Spam"])
    cl, summ, stats = topics.discover(df)
    assert "Spam" not in set(cl["predicted_category"])
    row = summ[summ["topic_id"] != topics.UNCLUSTERED_ID].iloc[0]
    assert row["n_comments_raw"] == 4 and row["n_unique_comments"] == 2
    assert stats["comments_in_duplicate_groups"] == 3


def test_empty_selection_is_handled():
    df = frame(["best video", "day 5 done"], ["Praise", "Other"])
    cl, summ, stats = topics.discover(df)
    assert stats["selected_comments"] == 0 and len(cl) == 0


# --------------------------------------------------------------------------
# Stage 3
# --------------------------------------------------------------------------

def _clusters_for_trends(n_videos=6, growing=True):
    rows = []
    i = 0
    for v in range(n_videos):
        for d in range(14):
            # one 'other' comment per day per video, plus topic A comments that grow in the last week
            rows.append((f"x{i}", f"v{v}", 0 if (growing and d < 7 and d % 2 == 0) else 1, d))
            i += 1
    df = pd.DataFrame(rows, columns=["comment_id", "video_id", "topic_id", "day"])
    df["published_at"] = df["day"].map(lambda d: _ts(d + 0.1))
    df["is_canonical"] = True
    df["topic_name"] = df["topic_id"].map({0: "A", 1: "B"})
    df["source_file"] = "comments_x.csv"
    df["channel_title"] = "Chan"
    df["video_title"] = df["video_id"]
    df["video_published_at"] = _ts(40)
    return df


def test_trends_growing_with_complete_capture():
    cl = _clusters_for_trends()
    cov = insights.capture_coverage(cl, sampled_sources={})
    summ = pd.DataFrame({"topic_id": [0, 1], "topic_name": ["A", "B"], "n_unique_comments": [16, 40]})
    tr, meta = insights.topic_trends(cl, summ, cov)
    assert meta["status"] == "ok"
    assert tr.set_index("topic_name").loc["A", "status"] == "growing"
    assert tr.set_index("topic_name").loc["B", "status"] in ("declining", "stable")


def test_trends_insufficient_when_capture_is_sampled():
    cl = _clusters_for_trends()
    cl["source_file"] = "educational_comments.csv"     # cap 50; each video has 14 comments ...
    cov = insights.capture_coverage(cl, sampled_sources={"educational_comments.csv":
                                                         {"per_video_cap": 10, "channel_target": 1000}})
    summ = pd.DataFrame({"topic_id": [0, 1], "topic_name": ["A", "B"], "n_unique_comments": [16, 40]})
    tr, meta = insights.topic_trends(cl, summ, cov)
    assert meta["status"] == "insufficient_data"
    assert set(tr["status"]) == {"insufficient_data"}
    assert "cannot yet be established" in meta["reason"]


def test_trends_without_timestamps():
    cl = _clusters_for_trends()
    cl["published_at"] = ""
    cov = insights.capture_coverage(cl, sampled_sources={})
    summ = pd.DataFrame({"topic_id": [0, 1], "topic_name": ["A", "B"], "n_unique_comments": [1, 1]})
    tr, meta = insights.topic_trends(cl, summ, cov)
    assert meta["status"] == "insufficient_data"


def test_reply_indicators_do_not_treat_unknown_as_unanswered():
    df = pd.DataFrame({"cat": ["Q", "Q", "Q"], "creator_replied": ["True", "False", ""], "reply_count": ["1", "0", "3"]})
    r = insights.reply_indicators(df, "cat")[0]
    assert (r["creator_reply_found"], r["no_creator_reply_at_collection"], r["reply_data_incomplete"]) == (1, 1, 1)


# --------------------------------------------------------------------------
# Stage 4
# --------------------------------------------------------------------------

def _rec_inputs(texts, cats, authors, videos):
    df = frame(texts, cats, authors=authors, videos=videos)
    cl, summ, _ = topics.discover(df)
    cl["topic_id"] = cl["topic_id"].astype(int)
    return cl, summ


def test_recommendation_counts_repeat_commenters_once_and_is_traceable():
    texts = ["please make a kubernetes course", "kubernetes course please", "kubernetes full course chahiye",
             "kubernetes ka course bana do", "kubernetes course please sir!"]
    # Two people only (a0 posted three times) -> below the 3-commenter minimum.
    cl, summ = _rec_inputs(texts, ["Request"] * 5, ["a0", "a0", "a0", "a1", "a1"], ["v0", "v1", "v2", "v0", "v1"])
    recs, rejected = recommend.recommend(summ, cl, pd.DataFrame(), pd.Timestamp(_ts(0)))
    assert recs == [] and "not enough evidence: 2" in rejected[0]["reason"]
    # Five different people -> recommended, with real comments as evidence.
    cl, summ = _rec_inputs(texts, ["Request"] * 5, ["a0", "a1", "a2", "a3", "a4"], ["v0", "v1", "v2", "v0", "v1"])
    recs, _ = recommend.recommend(summ, cl, pd.DataFrame(), pd.Timestamp(_ts(0)))
    assert len(recs) == 1
    r = recs[0]
    assert r["n_unique_commenters"] == 5 and r["n_videos"] == 3 and r["main_intent"] == "Request"
    assert 3 <= len(r["representative_comments"]) <= 5
    assert {c["text"] for c in r["representative_comments"]} <= set(texts)
    assert set(r["evidence_comment_ids"]) <= set(cl["comment_id"])
    assert 0 <= r["priority_score"] <= 100
    assert abs(sum(r["contributions"].values()) - r["priority_score"]) < 0.3


def test_problem_topics_get_debugging_format_and_outrank_equal_requests():
    probs = ["docker container crash error", "docker container error aa raha hai", "docker container not working error"]
    reqs = ["please make a flutter app course", "flutter app course please", "flutter app ka course bana do"]
    cl, summ = _rec_inputs(probs + reqs, ["Confusion/Problem"] * 3 + ["Request"] * 3,
                           [f"a{i}" for i in range(6)], ["v0", "v1", "v2"] * 2)
    recs, _ = recommend.recommend(summ, cl, pd.DataFrame(), pd.Timestamp(_ts(0)))
    by = {r["main_intent"]: r for r in recs}
    assert by["Confusion/Problem"]["suggested_format"] == "debugging walkthrough"
    assert by["Confusion/Problem"]["priority_score"] >= by["Request"]["priority_score"]


def test_similar_topics_merged_into_one_recommendation():
    cl = pd.DataFrame({
        "comment_id": [f"m{i}" for i in range(6)],
        "text": ["react hooks course please", "react hooks video please", "react hooks tutorial",
                 "react hooks course chahiye", "react hooks explained please", "react hooks playlist"],
        "predicted_category": "Request", "author_channel_id": [f"a{i}" for i in range(6)],
        "video_id": ["v0", "v1", "v2"] * 2, "video_title": "t", "channel_title": "C",
        "published_at": _ts(1), "is_canonical": True, "similarity_to_topic": "0.9",
        "is_representative": [True, False, False, True, False, False], "topic_id": [0, 0, 0, 1, 1, 1],
        "topic_text": "react hooks"})
    summ = pd.DataFrame({"topic_id": [0, 1], "topic_name": ["React Hooks", "React Hooks (2)"],
                         "n_unique_comments": [3, 3]})
    recs, _ = recommend.recommend(summ, cl, pd.DataFrame(), pd.Timestamp(_ts(0)))
    assert len(recs) == 1 and sorted(recs[0]["topic_ids"]) == [0, 1]


# --------------------------------------------------------------------------
# Whole pipeline
# --------------------------------------------------------------------------

def test_pipeline_end_to_end_reads_each_stage_output(tmp_path, model_path):
    rows = []
    texts = (["Python indentation error aa raha hai", "How can I fix indentation error in Python?",
              "Indentation issue in my Python code", "python indentation error again"]
             + ["please make a django course", "django course please sir", "django ka course bana do"]
             + ["best video thank you ❤", "day 5 done", "earn money daily dm me on telegram"])
    for i, t in enumerate(texts):
        rows.append({"comment_id": f"p{i}", "video_id": f"v{i % 3}", "text": t, "author_channel_id": f"a{i}",
                     "published_at": _ts(i), "creator_replied": "False", "reply_count": 0,
                     "channel_title": "Chan", "video_title": f"V{i % 3}", "video_published_at": _ts(30)})
    raw = tmp_path / "raw"
    write_raw(raw, rows)
    out = tmp_path / "out"
    rep = pipeline.run(model=model_path, out_dir=out, raw_dir=raw)
    assert rep["raw_and_labelled_files_unchanged"]
    failed = {k: v for k, v in rep["consistency_checks"].items()
              if v is not True and k != "recommendations_have_3_to_5_examples"}
    assert not failed, failed
    for f in ["comments_predicted.csv", "category_summary.csv", "topic_clusters.csv", "topic_summary.csv",
              "audience_insights.json", "topic_trends.csv", "video_topic_summary.csv",
              "content_recommendations.csv", "content_recommendations.json", "pipeline_report.json"]:
        assert (out / f).exists(), f
    ins = json.loads((out / "audience_insights.json").read_text(encoding="utf-8"))
    assert ins["run_summary"]["input_comments"] == len(rows)
    assert rep["totals"]["input_comments"] == len(rows)


# --------------------------------------------------------------------------
# Fixes found on a real run (Apna College Git tutorial)
# --------------------------------------------------------------------------

def test_creator_comments_are_excluded(tmp_path, model_path):
    rows = basic_rows()
    rows[0]["author_channel_id"] = "ch"            # same as channel_id -> the creator's own comment
    raw = tmp_path / "raw"
    write_raw(raw, rows)
    stats = batch_predict.run(model_path, out_dir=tmp_path / "out", raw_dir=raw)
    assert stats["excluded_by_reason"].get("creator_comment") == 1
    pred = pd.read_csv(tmp_path / "out" / "comments_predicted.csv", dtype=str, keep_default_na=False,
                       encoding="utf-8-sig")
    assert "c0" not in set(pred["comment_id"])


def test_same_problem_from_two_sides_is_one_topic_named_by_its_command():
    texts = ['27:40 in the VS code terminal "ls -a" is not working, this is the fix',
             'Try "dir -force" in vs code if "ls -a" is not working',
             'If ls -a is not working then use ls -h to show hidden files',
             'Ls -force use kro ls -a', 'You can use in windows ls -Force', 'ls -Force ( for power shell )',
             'what is git stash??', 'how can i get .git folder', 'how do we get here from git bash']
    # Unrelated comments around them, as in a real comment section (topic features need a realistic corpus).
    filler = ["please make a video on docker", "react course please", "how to deploy on aws?", "what is a pull request",
              "pls make video on github copilot", "which laptop is good for coding", "is this enough for placements",
              "video was too fast", "audio is low in the middle", "can you upload notes", "kubernetes playlist please",
              "how to resolve merge conflict", "python course chahiye", "what is the difference between git and github",
              "linux ka video laiye", "how to install git on mac", "jenkins tutorial please", "how to use vim editor"]
    cats = (["Confusion/Problem"] * 3 + ["Question"] * 6
            + ["Request", "Request", "Question", "Question", "Request", "Question", "Question", "Confusion/Problem",
               "Confusion/Problem", "Request", "Request", "Question", "Request", "Question", "Request", "Question",
               "Request", "Question"])
    df = frame(texts + filler, cats)
    cl, summ, _ = topics.discover(df)
    t = dict(zip(cl["comment_id"], cl["topic_id"]))
    # "ls -a is not working" (problem side) and "use ls -Force" (fix side) end up in ONE topic
    assert len({t[k] for k in ("k0", "k1", "k3", "k4", "k5")}) == 1 and t["k0"] != topics.UNCLUSTERED_ID
    name = summ.loc[summ["topic_id"] == t["k0"], "topic_name"].iloc[0]
    assert name == "ls -a"


def test_names_keep_the_commenters_spelling():
    assert topics.phrase_name(["mam GitDemo folder kaise banaya", "gitdemo folder kaha hai",
                               "GitDemo folder nahi mila"], ["gitdemo", "folder"]) == "GitDemo Folder"
    assert topics.topic_name(["devops"], None, topics._case_forms(["DevOps course", "devops please",
                                                                     "DevOps roadmap"])) == "DevOps"
    assert topics.topic_name(["git"], None, topics._case_forms(["git stash?", "git bash"])) == "Git"


FILLER = ["please make a video on docker", "react course please", "how to deploy on aws?", "what is a pull request",
          "pls make video on github copilot", "which laptop is good for coding", "is this enough for placements",
          "video was too fast", "audio is low in the middle", "can you upload slides", "kubernetes playlist please",
          "how to resolve merge conflict", "python course chahiye", "what is the difference between git and github",
          "linux ka video laiye", "how to install git on mac", "jenkins tutorial please", "how to use vim editor"]
FILLER_CATS = ["Request", "Request", "Question", "Question", "Request", "Question", "Question", "Confusion/Problem",
               "Confusion/Problem", "Request", "Request", "Question", "Request", "Question", "Request", "Question",
               "Request", "Question"]


def test_maths_written_different_ways_is_one_topic():
    """Real Gate Smashers comments: k^2, k², k square, ksquare, 'k ki power 2', K2 and k+k are the same doubt."""
    texts = ["why its not 2K ?", "sir why not 2k why k^2?", "Sir k+k =2k ???", "k+k=2K OR k KI PAOWER 2 ???",
             "Why ksquare not 2k when k is I/o time", "Time 13:47: k square or 2k?", "13:56  why k square ??",
             "I don't understand K2 term ?", "How it is K² I didnt understand Someone explain please"]
    df = frame(texts + FILLER, ["Question"] * 7 + ["Confusion/Problem"] * 2 + FILLER_CATS)
    cl, summ, _ = topics.discover(df)
    t = dict(zip(cl["comment_id"], cl["topic_id"]))
    ids = {t[f"k{i}"] for i in range(len(texts))}
    assert len(ids) == 1 and topics.UNCLUSTERED_ID not in ids
    name = summ.loc[summ["topic_id"] == ids.pop(), "topic_name"].iloc[0]
    assert "k²" in name and "Understand" not in name and "Time" not in name


def test_groups_with_only_generic_words_are_not_topics():
    """'smj nhi aata' + 'nice explanation' share no subject: no 'Samajh' or 'Explanation' topic."""
    texts = ["1 - (.7)^2 = 76% kaise nhi smj aaya someone explain", "Paid courses m smj nhi aata yha better smj aata h",
             "Thanks for nice explanation, will it work for negative number?", "what an explanation"]
    df = frame(texts + FILLER, ["Confusion/Problem", "Confusion/Problem", "Question", "Question"] + FILLER_CATS)
    cl, summ, _ = topics.discover(df)
    names = set(summ["topic_name"])
    assert not names & {"Samajh", "Explanation", "Understand", "Aata"}
    for n in names:
        assert not topics.is_generic(n.lower())


def test_same_request_in_several_spellings_is_one_topic():
    texts = ["Sir!can you provide me notes of operating system?", "operating system ki class ke notes bna rkhe h kya?",
             "Operating system notes chahiye", "Notes", "Sir please notes do", "please also provide notes with lectures"]
    df = frame(texts + FILLER, ["Request", "Question", "Request", "Request", "Request", "Request"] + FILLER_CATS)
    cl, summ, _ = topics.discover(df)
    t = dict(zip(cl["comment_id"], cl["topic_id"]))
    ids = {t[f"k{i}"] for i in range(len(texts))}
    assert len(ids) == 1 and topics.UNCLUSTERED_ID not in ids
    assert summ.loc[summ["topic_id"] == ids.pop(), "topic_name"].iloc[0].startswith("Study materials")


# Real comments from the Stanford CS229 run.
STANFORD = ["Who is starting in 2026 ?", "Who is starting in 2026?", "Who is starting the course in 2026?",
            "Starting in  03-sep-2026", "Started 17/4/2026", "Who Started in July 2026?", "Starting in 2026 !!",
            "Is there any hard resources for this course?", "Is there a text book for this course?",
            "books to refer", "Hello From where  can i get the handouts ?",
            "Still relevant in 2025?", "How relevant is this course now? Is there an update?",
            "Is this course outdated?", "Is this course outdated in 2026?",
            "Is Dr Ng still teaching this at Stanford?", "teaching starts at 34:36"]


def _stanford():
    cats = ["Question"] * len(STANFORD)
    cl, summ, _ = topics.discover(frame(STANFORD + FILLER, cats + FILLER_CATS))
    return cl, summ, dict(zip(cl["comment_id"], cl["topic_name"]))


def test_check_ins_are_one_topic_and_never_recommended():
    assert topics.is_checkin("Who is starting in 2026?") and topics.is_checkin("Started 17/4/2026")
    assert not topics.is_checkin("Starting today I always start this course but never finish it, any tips?")
    assert not topics.is_checkin("Who are doing it in 2025 can you give me further guideline and resource")
    cl, summ, name = _stanford()
    checkins = {name[f"k{i}"] for i in range(7)}
    assert len(checkins) == 1 and "check-in" in checkins.pop().lower()      # "Starting" + "Started" = one topic
    assert set(summ.loc[summ["topic_name"].str.contains("check-in"), "topic_kind"]) == {"check-in"}
    cl2 = cl.assign(is_canonical=cl["is_canonical"].astype(bool))
    recs, rejected = recommend.recommend(summ, cl2, pd.DataFrame(), pd.Timestamp(_ts(0)))
    assert not any("check-in" in r["topic_name"].lower() for r in recs)
    assert any("check-ins" in x["reason"] for x in rejected)


def test_same_need_in_different_words_is_one_named_topic():
    cl, summ, name = _stanford()
    materials = {name[f"k{i}"] for i in range(7, 11)}          # resources, text book, books, handouts
    relevance = {name[f"k{i}"] for i in range(11, 15)}         # relevant, outdated, update
    assert materials == {"Study materials"}
    assert relevance == {"Is the course still relevant?"}
    # "teaching" alone does not make a topic (one asks if Dr Ng still teaches, one gives a timestamp)
    assert name["k15"] == name["k16"] == "Unclustered"
    assert not set(summ["topic_name"]) & {"Starting", "Started", "Resource", "Teaching", "Book"}


def test_recommendation_needs_comments_that_mention_the_topic():
    cl, summ, _ = _stanford()
    cl = cl.assign(is_canonical=cl["is_canonical"].astype(bool))
    recs, _ = recommend.recommend(summ, cl, pd.DataFrame(), pd.Timestamp(_ts(0)))
    titles = {r["topic_name"]: r for r in recs}
    assert set(titles) == {"Study materials", "Is the course still relevant?"}
    for r in recs:                                  # evidence = real comments from that topic
        ids = set(cl.loc[cl["topic_name"] == r["topic_name"], "comment_id"])
        assert {c["comment_id"] for c in r["representative_comments"]} <= ids
        assert {c["text"] for c in r["representative_comments"]} <= set(STANFORD)
    # A topic glued by an incidental word is rejected: make the key terms absent from most comments
    summ2 = summ.copy()
    summ2.loc[summ2["topic_name"] == "Study materials", "top_terms"] = "hard; refer"
    recs2, rej2 = recommend.recommend(summ2, cl, pd.DataFrame(), pd.Timestamp(_ts(0)))
    assert "Study materials" not in {r["topic_name"] for r in recs2}
    assert any("don't consistently mention" in x["reason"] for x in rej2)


def test_one_word_topics_are_named_from_a_real_comment():
    assert topics.clean_comment_title("DO I GET RECORDED LECTURES") == "Do I get recorded lectures"
    assert topics.clean_comment_title("Mam,  can i join it now???") == "Can I join it now?"
    assert topics.clean_comment_title("27:40 how can i get .git") == "How can I get .git"
    assert topics.name_from_comments("Join", ["Sry too late I want to join dis course...can i", "Can i join it now"],
                                     [0.5, 0.6]) == "Can I join it now"
    assert topics.name_from_comments("Git", ["27:40 how can i get .git"], [0.9]) == "Git"     # a subject word
    assert recommend.make_title("Is the course still relevant?", "concept explanation").startswith("Is This Course")
    assert recommend.make_title("Study materials: Operating System", "tutorial").startswith("Operating System Study")
