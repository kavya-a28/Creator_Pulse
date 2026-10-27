"""
CreatorPulse — Build the training set from the labelled comments.

Input:
    data/labelled/comments_labelled.csv   (comment_id, video_id, text, label, ...)
    data/raw/educational_comments.csv     (for author_channel_id / channel_id)

Output:
    data/processed/training_comments.csv  one row per kept comment
    data/processed/removed_comments.csv   every dropped row, with the reason

Rules, applied in this order:

1. Creator-authored comments are excluded. A comment is creator-authored when
   its author_channel_id equals the channel_id of the channel that uploaded the
   video (pinned promos, announcements, the creator's own notes). These are
   not audience intent, so they never enter training, whatever their label.

2. Near-duplicates are collapsed to one row. Two comments are near-duplicates
   when their normalised text is identical, or when the character n-gram
   TF-IDF cosine similarity of the normalised texts is >= THRESHOLD. Groups
   are connected components of that similarity graph. Normalisation lowercases,
   applies NFKC, removes URLs, video timestamps (12:03), punctuation and
   emoji, and squeezes repeated characters ("congratsss" -> "congratss";
   digits are never squeezed).
   Hindi/Devanagari is kept intact. Emoji-only comments are compared by the
   set of emoji they use.

   Within a group, the first comment (file order) of each label is kept, so a
   group whose members were labelled differently keeps one example per label
   and is flagged in the log rather than silently losing a label. Each kept
   row gets duplicate_count = number of comments it stands for.

The original raw and labelled files are never modified.

Usage:
    python -m src.prepare_training_data
    python -m src.prepare_training_data --threshold 0.85
"""

import argparse
import re
import unicodedata

import pandas as pd

from src.config import LABELLED_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

THRESHOLD = 0.80

LABELLED_INPUT = LABELLED_DATA_DIR / "comments_labelled.csv"
RAW_INPUT = RAW_DATA_DIR / "educational_comments.csv"
TRAIN_OUTPUT = PROCESSED_DATA_DIR / "training_comments.csv"
REMOVED_OUTPUT = PROCESSED_DATA_DIR / "removed_comments.csv"
CSV_ENCODING = "utf-8-sig"

_URL = re.compile(r"https?://\S+")
_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_REPEAT = re.compile(r"([^\d\s])\1{2,}")  # never squeeze digits
_SPACE = re.compile(r"\s+")


_VARIATION_SELECTORS = {"︎", "️"}


def _is_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    # Variation selectors are category Mn, like Devanagari vowel signs, so
    # they are matched explicitly instead of stripping all Mn characters.
    return cat.startswith("S") or cat == "Cf" or ch in _VARIATION_SELECTORS


def dedup_key(text: str) -> str:
    """Normalised form of a comment used only for near-duplicate detection."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _URL.sub(" ", t)
    t = _TIMESTAMP.sub(" ", t)
    emoji = "".join(sorted({
        ch for ch in t
        if _is_symbol(ch) and ord(ch) > 0x2000 and ch not in _VARIATION_SELECTORS
    }))
    t = "".join(
        " " if _is_symbol(ch) or unicodedata.category(ch).startswith("P") else ch
        for ch in t
    )
    t = _SPACE.sub(" ", _REPEAT.sub(r"\1\1", t)).strip()
    if t:
        return t
    return f"emoji:{emoji}" if emoji else ""


def creator_mask(labelled: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    """True for comments written by the channel that uploaded the video."""
    meta = raw.set_index("comment_id")[["author_channel_id", "channel_id"]]
    m = labelled["comment_id"].map(
        lambda cid: cid in meta.index
        and meta.at[cid, "author_channel_id"] != ""
        and meta.at[cid, "author_channel_id"] == meta.at[cid, "channel_id"]
    )
    return m.astype(bool)


def near_duplicate_groups(keys: pd.Series, threshold: float = THRESHOLD) -> pd.Series:
    """Assign a group id to each row; rows in one group are near-duplicates."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    unique = pd.Series(keys.unique())
    # Text keys are compared fuzzily. Emoji-only and empty keys only match
    # exactly: character n-grams say little about which emoji were used.
    is_text = (unique != "") & ~unique.str.startswith("emoji:")
    text_keys = unique[is_text].reset_index(drop=True)
    exact_keys = unique[~is_text].reset_index(drop=True)

    key_to_group = {}
    if len(text_keys):
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(text_keys)
        adj = cosine_similarity(X, dense_output=False) >= threshold
        n, comp = connected_components(adj, directed=False)
        key_to_group.update(zip(text_keys, comp))
    else:
        n = 0
    key_to_group.update({k: n + i for i, k in enumerate(exact_keys)})
    return keys.map(key_to_group)


def build_training_set(labelled: pd.DataFrame, raw: pd.DataFrame, threshold: float = THRESHOLD):
    """Return (training_df, removed_df)."""
    df = labelled.reset_index(drop=True).copy()
    removed = []

    is_creator = creator_mask(df, raw)
    for _, r in df[is_creator].iterrows():
        removed.append({**r[["comment_id", "video_id", "text", "label"]].to_dict(),
                        "reason": "creator-authored", "kept_comment_id": "",
                        "group_has_mixed_labels": ""})
    df = df[~is_creator].reset_index(drop=True)

    df["_group"] = near_duplicate_groups(df["text"].map(dedup_key), threshold)
    mixed = df.groupby("_group")["label"].nunique() > 1
    df["_mixed"] = df["_group"].map(mixed)

    keep_idx = df.groupby(["_group", "label"], sort=False).head(1).index
    kept_for = df.loc[keep_idx].set_index(["_group", "label"])["comment_id"]
    counts = df.groupby(["_group", "label"]).size()

    for i, r in df.drop(index=keep_idx).iterrows():
        removed.append({**r[["comment_id", "video_id", "text", "label"]].to_dict(),
                        "reason": "near-duplicate",
                        "kept_comment_id": kept_for[(r["_group"], r["label"])],
                        "group_has_mixed_labels": bool(r["_mixed"])})

    train = df.loc[keep_idx].sort_index().copy()
    train["duplicate_count"] = [counts[(g, lab)] for g, lab in zip(train["_group"], train["label"])]
    train = train[["comment_id", "video_id", "text", "label", "duplicate_count"]].reset_index(drop=True)

    removed_df = pd.DataFrame(removed, columns=[
        "comment_id", "video_id", "text", "label", "reason",
        "kept_comment_id", "group_has_mixed_labels",
    ])
    return train, removed_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the training set from labelled comments.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Cosine similarity for near-duplicates (default {THRESHOLD}).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    existing = [p for p in (TRAIN_OUTPUT, REMOVED_OUTPUT) if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"[ERROR] Output already exists: {', '.join(map(str, existing))}. Use --force to overwrite.")

    labelled = pd.read_csv(LABELLED_INPUT, dtype=str, keep_default_na=False)
    raw = pd.read_csv(RAW_INPUT, dtype=str, keep_default_na=False)
    train, removed = build_training_set(labelled, raw, args.threshold)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUTPUT, index=False, encoding=CSV_ENCODING)
    removed.to_csv(REMOVED_OUTPUT, index=False, encoding=CSV_ENCODING)

    print(f"Labelled comments      : {len(labelled)}")
    print(f"Creator-authored removed: {(removed.reason == 'creator-authored').sum()}")
    print(f"Near-duplicates removed : {(removed.reason == 'near-duplicate').sum()}")
    print(f"Training comments      : {len(train)} -> {TRAIN_OUTPUT}")
    print(f"Removal log            : {REMOVED_OUTPUT}")
    print("\nLabel counts (before -> after):")
    before = labelled["label"].value_counts()
    after = train["label"].value_counts()
    for lab in before.index:
        print(f"  {lab:<10} {before[lab]:>4} -> {after.get(lab, 0):>4}")


if __name__ == "__main__":
    main()
