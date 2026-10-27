"""Tests for src.make_splits (no API key or network needed)."""

import numpy as np
import pandas as pd

from src.make_splits import choose_test_videos, leakage_groups, random_group_split


def _words(seed):
    """A short pseudo-random 'sentence' that shares little with any other."""
    rng = np.random.default_rng(seed)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    return " ".join("".join(rng.choice(letters, rng.integers(4, 9))) for _ in range(6))


def _df():
    rows = []
    labels = ["Question", "Request", "Praise", "Spam", "Other"]
    for v in range(10):
        for i in range(12):
            rows.append({
                "comment_id": f"c{v}_{i}",
                "video_id": f"video{v:05d}x",
                "channel_id": f"ch{v % 5}",
                "author_channel_id": f"a{v}_{i}",
                "label": labels[i % 5],
                "text": _words(v * 12 + i),
            })
    df = pd.DataFrame(rows)
    # near-duplicates and a repeat author that must stay together
    df.loc[0, "text"] = "Congratulations Harry bhai 🎉🎉"
    df.loc[60, "text"] = "congratulations harry bhai!!!"
    df.loc[5, "author_channel_id"] = df.loc[100, "author_channel_id"] = "same_author"
    return df


def test_leakage_groups_join_near_duplicates_and_authors():
    df = _df()
    g = leakage_groups(df)
    assert g[0] == g[60]
    assert g[5] == g[100]
    assert len(set(g)) == len(df) - 2


def test_random_split_keeps_groups_together_and_is_reproducible():
    df = _df()
    g = leakage_groups(df)
    f1 = random_group_split(df, g, seed=42)
    f2 = random_group_split(df, g, seed=42)
    assert (f1 == f2).all()
    for grp in set(g):
        assert len(set(f1[g == grp])) == 1
    assert set(f1) == {0, 1, 2, 3, 4}


def test_video_holdout_one_per_channel_all_classes():
    df = _df()
    vids = choose_test_videos(df, seed=42, n_draws=200)
    assert len(vids) == df.channel_id.nunique()
    test = df.video_id.isin(vids)
    assert set(df[test].label) == set(df.label) == set(df[~test].label)
    assert choose_test_videos(df, seed=42, n_draws=200) == vids
