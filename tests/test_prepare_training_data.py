"""Tests for src.prepare_training_data (no API key or network needed)."""

import pandas as pd

from src.prepare_training_data import build_training_set, dedup_key


def _frames(rows):
    """rows: (comment_id, text, label, author_channel_id)"""
    labelled = pd.DataFrame(
        [(c, "v1", t, lab) for c, t, lab, _ in rows],
        columns=["comment_id", "video_id", "text", "label"],
    )
    raw = pd.DataFrame(
        [(c, a, "UCcreator") for c, _, _, a in rows],
        columns=["comment_id", "author_channel_id", "channel_id"],
    )
    return labelled, raw


def test_dedup_key_keeps_devanagari_and_drops_emoji_and_timestamps():
    assert dedup_key("बहुत बढ़िया ❤️") == "बहुत बढ़िया"
    assert dedup_key("12:03 Congratsss!!!") == "congratss"
    assert dedup_key("❤️❤") == dedup_key("❤")


def test_creator_comments_removed():
    lab, raw = _frames([
        ("c1", "Join my course: https://x.y", "Spam", "UCcreator"),
        ("c2", "How do I install pip?", "Question", "UCviewer"),
    ])
    train, removed = build_training_set(lab, raw)
    assert list(train.comment_id) == ["c2"]
    assert removed.loc[0, "reason"] == "creator-authored"


def test_near_duplicates_collapse_and_count():
    lab, raw = _frames([
        ("c1", "Congratulations Harry bhai 🎉🎉", "Praise", "UCa"),
        ("c2", "congratulations harry bhai", "Praise", "UCb"),
        ("c3", "Congratulation Harry Bhai!!!", "Praise", "UCc"),
        ("c4", "Please make a video on Django", "Request", "UCd"),
    ])
    train, removed = build_training_set(lab, raw)
    assert list(train.comment_id) == ["c1", "c4"]
    assert list(train.duplicate_count) == [3, 1]
    assert set(removed.kept_comment_id) == {"c1"}


def test_different_emoji_only_comments_are_not_merged():
    lab, raw = _frames([
        ("c1", "♥️🥰👍", "Praise", "UCa"),
        ("c2", "💟♾️🇦🇶🏔️👌🌍", "Other", "UCb"),
    ])
    train, _ = build_training_set(lab, raw)
    assert len(train) == 2


def test_mixed_label_group_keeps_one_per_label():
    lab, raw = _frames([
        ("c1", "Python course please", "Request", "UCa"),
        ("c2", "python course please!!", "Other", "UCb"),
    ])
    train, _ = build_training_set(lab, raw)
    assert sorted(train.label) == ["Other", "Request"]


def test_dedup_key_does_not_squeeze_digits():
    assert dedup_key("1000 subscribers") != dedup_key("100 subscribers")
