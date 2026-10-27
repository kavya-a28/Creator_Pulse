"""Tests for src.generate_synthetic and the --synthetic training path."""

import pandas as pd
import pytest

from src.generate_synthetic import CLASSES, TEMPLATES, generate, near_duplicate_mask
from src.preprocess import normalize
from src.train_svm import LABEL_COLUMN, load_synthetic, with_extra


@pytest.fixture(scope="module")
def small():
    data, stats = generate(per_class=60, seed=7)
    return data, stats


def test_every_class_has_templates():
    assert set(TEMPLATES) == set(CLASSES)
    assert all(len(TEMPLATES[c]) >= 5 for c in CLASSES)


def test_balanced_and_well_formed(small):
    data, stats = small
    assert data[LABEL_COLUMN].value_counts().to_dict() == {c: 60 for c in CLASSES}
    assert data["comment_id"].is_unique and data["comment_id"].str.startswith("syn_").all()
    assert (data["video_id"] == "synthetic").all()
    assert data["text"].str.strip().str.len().gt(0).all()
    # 5-class label mirrors the labelling-guide merge.
    merged = data[LABEL_COLUMN].replace({"Confusion/Problem": "Question"})
    assert (data["label"] == merged).all()


def test_no_duplicate_normalised_text(small):
    data, _ = small
    assert data["text"].map(normalize).is_unique


def test_deterministic():
    a, _ = generate(per_class=30, seed=3)
    b, _ = generate(per_class=30, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_leakage_filter_drops_near_copies_of_real_comments():
    real = ["please make a video on django rest framework sir"]
    data, stats = generate(per_class=40, seed=1, real_texts=real)
    near, _ = near_duplicate_mask(data["text"].tolist(), real)
    assert not near.any()
    assert not data["text"].map(normalize).isin({normalize(real[0])}).any()


def test_mixed_language(small):
    data, _ = small
    assert {"en", "hinglish", "hi"} <= set(data["language"])


def test_synthetic_only_extends_training(tmp_path, small):
    data, _ = small
    path = tmp_path / "syn.csv"
    data.to_csv(path, index=False, encoding="utf-8-sig")
    syn = load_synthetic(path)
    train = pd.DataFrame({"text": ["a", "b"], LABEL_COLUMN: ["Other", "Praise"]})
    x, y = with_extra(train, syn)
    assert len(x) == len(y) == 2 + len(syn)
    x0, y0 = with_extra(train, None)
    assert list(x0) == ["a", "b"]


def test_load_synthetic_rejects_non_synthetic_ids(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"comment_id": ["Ugx123"], "text": ["hi"], LABEL_COLUMN: ["Other"]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        load_synthetic(path)
