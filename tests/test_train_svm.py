"""Tests for src.train_svm (no saved model or data files needed)."""

import pandas as pd
import pytest

from src.train_svm import (CLASSES, attach_labels, audit_labels, build_pipeline, evaluate,
                           leakage_checks, reliability, wilson_interval)

CORPUS = [
    ("what is a tuple ?", "Question"), ("how does recursion work ?", "Question"),
    ("please make a video on django", "Request"), ("please upload notes sir", "Request"),
    ("samajh nahi aaya error aa raha", "Confusion/Problem"), ("code not working error", "Confusion/Problem"),
    ("best video thank you ❤", "Praise"), ("thank you bhaiya best ❤", "Praise"),
    ("dm me on telegram earn money", "Spam"), ("earn money telegram link", "Spam"),
    ("day 5 done", "Other"), ("who is watching in 2026", "Other"),
]


def fitted(min_df=1):
    texts, labels = zip(*CORPUS)
    return build_pipeline(min_df=min_df).fit(list(texts), list(labels))


def test_pipeline_keeps_negations_question_marks_and_emoji():
    vocab = fitted().named_steps["tfidf"].vocabulary_
    for tok in ("nahi", "not", "?", "❤", "not working"):
        assert tok in vocab, tok


def test_pipeline_normalises_like_text_norm():
    vocab = fitted().named_steps["tfidf"].vocabulary_
    model = fitted()
    # "nhi" -> "nahi" and "didn't" -> "did not" happen inside the pipeline
    x = model.named_steps["tfidf"].transform(["samajh nhi aaya"])
    assert x[0, vocab["nahi"]] > 0


def test_pipeline_config():
    p = build_pipeline()
    tf, svm = p.named_steps["tfidf"], p.named_steps["svm"]
    assert tf.ngram_range == (1, 2) and tf.min_df == 2 and tf.sublinear_tf
    assert tf.stop_words is None
    assert svm.class_weight == "balanced"


def test_pipeline_predicts_known_labels():
    model = fitted()
    assert set(model.classes_) == set(CLASSES)
    assert model.predict(["please make a video on react"])[0] in CLASSES


def test_wilson_interval():
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(0, 4)
    assert lo == 0.0 and 0.4 < hi < 0.6
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi


def test_reliability_thresholds():
    assert reliability(2) == "insufficient"
    assert reliability(22) == "low"
    assert reliability(90) == "adequate"


def test_evaluate_reports_all_classes_and_flags_small_ones():
    y_true = ["Other"] * 20 + ["Praise"] * 12 + ["Spam"] * 2
    y_pred = ["Other"] * 20 + ["Praise"] * 10 + ["Other"] * 2 + ["Other"] * 2
    ev = evaluate(y_true, y_pred, with_ci=False)
    assert set(ev["per_class"]) == set(CLASSES)
    assert ev["per_class"]["Spam"]["reliability"] == "insufficient"
    assert ev["per_class"]["Question"]["support"] == 0
    assert ev["macro_f1_classes_with_support_ge_10"]["classes"] == ["Praise", "Other"]
    assert len(ev["confusion_matrix"]["rows_true_cols_pred"]) == len(CLASSES)


def test_audit_labels_detects_problems():
    df = pd.DataFrame({"comment_id": ["a", "a", "b"], "text": ["x", "y", "z"],
                       "label": ["Other"] * 3, "label_6class": ["Other", "", "Weird"]})
    audit = audit_labels(df)
    assert audit["duplicate_comment_ids"] == 1
    assert audit["missing_label_6class"] == 1
    assert audit["unknown_labels"] == ["Weird"]


def test_attach_labels_rejects_text_mismatch():
    labels = pd.DataFrame({"comment_id": ["a"], "text": ["hello"], "label": ["Other"],
                           "label_6class": ["Other"]})
    split = pd.DataFrame({"comment_id": ["a"], "text": ["changed"], "label": ["Other"]})
    with pytest.raises(SystemExit):
        attach_labels(split, labels)
    ok = attach_labels(split.assign(text="hello"), labels)
    assert ok["label_6class"].tolist() == ["Other"]


def test_leakage_checks_fail_on_shared_group():
    tr = pd.DataFrame({"comment_id": ["a"], "leakage_group": ["1"], "text": ["x"], "video_id": ["v"]})
    te = pd.DataFrame({"comment_id": ["b"], "leakage_group": ["1"], "text": ["y"], "video_id": ["w"]})
    with pytest.raises(SystemExit):
        leakage_checks(tr, te)
    te["leakage_group"] = "2"
    assert leakage_checks(tr, te)["shared_leakage_groups"] == 0
