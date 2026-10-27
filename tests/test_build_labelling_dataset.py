"""Tests for src.build_labelling_dataset (no API key or network needed)."""

import pandas as pd

from src.build_labelling_dataset import (
    LABEL_COLUMNS,
    LABELS,
    deduplicate,
    make_label_frame,
)


def _raw():
    return pd.DataFrame({
        "comment_id": ["c1", "c2", "c1", "c3"],
        "video_id": ["v1", "v1", "v2", "v2"],
        "text": ["Sir ye samajh nahi aaya 🤔", "बहुत बढ़िया ❤️", "dup", "What is 2:15 about?"],
        "published_at": ["2026-01-01T00:00:00Z"] * 4,
    })


def test_deduplicate_by_comment_id_keeps_first():
    out = deduplicate(_raw())
    assert list(out["comment_id"]) == ["c1", "c2", "c3"]
    assert out.loc[0, "text"] == "Sir ye samajh nahi aaya 🤔"


def test_label_frame_columns_and_empty_label():
    out = make_label_frame(deduplicate(_raw()))
    assert list(out.columns) == LABEL_COLUMNS
    assert (out["label"] == "").all()


def test_text_preserved_through_csv_roundtrip(tmp_path):
    path = tmp_path / "x.csv"
    make_label_frame(_raw()).to_csv(path, index=False, encoding="utf-8-sig")
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(back["text"]) == list(_raw()["text"])


def test_five_labels():
    assert LABELS == ["Question", "Request", "Praise", "Spam", "Other"]
