"""
CreatorPulse — Build the educational-comments dataset for NLP labelling.

Collects ~1,000 top-level comments from a handful of educational YouTube
channels by reusing the existing collectors:

    src.fetch_channel_videos.resolve_channel / fetch_channel_videos
    src.fetch_comments.fetch_comment_threads / threads_to_dataframe

Outputs:
    data/raw/educational_comments.csv        original, unmodified comments
    data/labelled/comments_to_label.csv      comment_id, video_id, text, empty label

Comment text is never cleaned or altered: emojis, Hindi (Devanagari) and
Hinglish are kept exactly as returned by the API. Duplicates are removed on
comment_id only. Both CSVs are written as UTF-8 with BOM so that Excel on
Windows shows Hindi and emojis correctly; pandas reads them normally.

Usage:
    python -m src.build_labelling_dataset
    python -m src.build_labelling_dataset --target 1000 --videos-per-channel 12
    python -m src.build_labelling_dataset --channels @3blue1brown @CodeWithHarry
"""

import argparse
import logging
import math
from typing import List, Optional

import pandas as pd
from googleapiclient.errors import HttpError

from src.config import (
    LABELLED_DATA_DIR,
    RAW_DATA_DIR,
    build_youtube_client,
    setup_logging,
)
from src.fetch_channel_videos import fetch_channel_videos, resolve_channel
from src.fetch_comments import fetch_comment_threads, threads_to_dataframe

logger = logging.getLogger(__name__)

# Educational channels: programming, technology, mathematics, exam preparation.
# A mix of English and Hindi/Hinglish audiences.
DEFAULT_CHANNELS = [
    "@CodeWithHarry",        # programming (Hindi/Hinglish)
    "@ApnaCollegeOfficial",  # programming / placements (Hindi/Hinglish)
    "@freecodecamp",         # programming / technology (English)
    "@3blue1brown",          # mathematics (English)
    "@PhysicsWallah",        # exam preparation — JEE/NEET (Hindi/Hinglish)
]

# Confusion/Problem was merged into Question (too few examples: 19 of 1,000).
LABELS = ["Question", "Request", "Praise", "Spam", "Other"]

RAW_OUTPUT = RAW_DATA_DIR / "educational_comments.csv"
LABEL_OUTPUT = LABELLED_DATA_DIR / "comments_to_label.csv"

RAW_COLUMNS = [
    "comment_id", "video_id", "text", "author_channel_id",
    "published_at", "like_count", "reply_count", "creator_replied",
    "channel_id", "channel_title", "video_title", "video_published_at",
]
LABEL_COLUMNS = ["comment_id", "video_id", "text", "label"]

CSV_ENCODING = "utf-8-sig"


def collect_channel_comments(
    youtube,
    channel_input: str,
    per_channel_target: int,
    videos_per_channel: int,
    per_video_cap: int,
) -> pd.DataFrame:
    """Collect up to `per_channel_target` comments from one channel.

    Comments are spread across several recent videos (at most
    `per_video_cap` per video) so one viral video does not dominate.
    """
    channel_id, _, channel_title = resolve_channel(youtube, channel_input)
    print(f"\n== {channel_title} ({channel_id})")

    videos = fetch_channel_videos(youtube, channel_id, limit=videos_per_channel)
    frames = []
    collected = 0

    for video in videos:
        if collected >= per_channel_target:
            break
        video_id = video["video_id"]
        try:
            # One API page = up to 100 threads = 1 quota unit.
            threads = fetch_comment_threads(youtube, video_id, max_pages=1)
        except HttpError as e:
            print(f"  - {video_id}: skipped (HTTP {e.resp.status})")
            continue

        if not threads:
            print(f"  - {video_id}: no comments")
            continue

        df = threads_to_dataframe(threads, video_id, channel_id)
        take = min(per_video_cap, per_channel_target - collected, len(df))
        df = df.head(take).copy()
        df["channel_id"] = channel_id
        df["channel_title"] = channel_title
        df["video_title"] = video.get("title", "")
        df["video_published_at"] = video.get("published_at", "")
        frames.append(df)
        collected += len(df)
        print(f"  - {video_id}: +{len(df)} (channel total {collected}) — {video.get('title', '')[:60]}")

    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.concat(frames, ignore_index=True)[RAW_COLUMNS]


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate comments by comment_id, keeping the first occurrence."""
    return df.drop_duplicates(subset="comment_id", keep="first").reset_index(drop=True)


def make_label_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the labelling sheet: id columns, original text, empty label."""
    out = raw[["comment_id", "video_id", "text"]].copy()
    out["label"] = ""
    return out[LABEL_COLUMNS]


def build_dataset(
    channels: List[str],
    target: int,
    videos_per_channel: int,
    per_video_cap: Optional[int],
) -> pd.DataFrame:
    youtube = build_youtube_client()
    per_channel = math.ceil(target / len(channels))
    cap = per_video_cap or max(20, math.ceil(per_channel / 4))

    frames = []
    for ch in channels:
        try:
            frames.append(
                collect_channel_comments(youtube, ch, per_channel, videos_per_channel, cap)
            )
        except (ValueError, HttpError) as e:
            print(f"\n[WARN] Skipping channel {ch}: {e}")

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("[ERROR] No comments collected from any channel.")

    raw = pd.concat(frames, ignore_index=True)
    before = len(raw)
    raw = deduplicate(raw)
    print(f"\nCollected {before} comments, {before - len(raw)} duplicate comment_id(s) removed.")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect educational YouTube comments and create a labelling sheet.",
    )
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS,
                        help="Channel handles, URLs or IDs (default: 5 educational channels).")
    parser.add_argument("--target", type=int, default=1000,
                        help="Approximate total number of comments (default: 1000).")
    parser.add_argument("--videos-per-channel", type=int, default=15,
                        help="Recent videos to consider per channel (default: 15).")
    parser.add_argument("--per-video-cap", type=int, default=None,
                        help="Max comments per video (default: ~1/4 of the per-channel share).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite comments_to_label.csv even if it already has labels.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # Never silently wipe labelling work.
    if LABEL_OUTPUT.exists() and not args.force:
        existing = pd.read_csv(LABEL_OUTPUT, dtype=str, keep_default_na=False)
        if "label" in existing and (existing["label"].str.strip() != "").any():
            raise SystemExit(
                f"[ERROR] {LABEL_OUTPUT} already contains labels. "
                "Re-run with --force to overwrite it."
            )

    raw = build_dataset(args.channels, args.target, args.videos_per_channel, args.per_video_cap)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LABELLED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_OUTPUT, index=False, encoding=CSV_ENCODING)
    make_label_frame(raw).to_csv(LABEL_OUTPUT, index=False, encoding=CSV_ENCODING)

    print("\n" + "=" * 60)
    print("  Educational comments dataset")
    print("=" * 60)
    print(f"  Raw comments   : {len(raw)} -> {RAW_OUTPUT}")
    print(f"  Labelling file : {LABEL_OUTPUT}")
    print(f"  Labels         : {', '.join(LABELS)}")
    print("\n  Per channel:")
    for title, n in raw["channel_title"].value_counts().items():
        print(f"    {title:<30} {n}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
