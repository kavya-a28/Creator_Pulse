"""
CreatorPulse — Channel video collection module.

Fetches recent videos from a YouTube channel and optionally collects
comments for each video.

Usage:
    python -m src.fetch_channel_videos --channel CHANNEL_ID --limit 20
    python -m src.fetch_channel_videos --channel CHANNEL_ID --limit 20 --fetch-comments
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from googleapiclient.errors import HttpError

from src.config import (
    RAW_DATA_DIR,
    build_youtube_client,
    setup_logging,
)
from src.fetch_comments import fetch_comment_threads, save_comments_csv, threads_to_dataframe
from src.url_utils import parse_channel_input

logger = logging.getLogger(__name__)


def resolve_channel(youtube, channel_input: str) -> tuple[str, str, str]:
    """Resolve a channel identifier (ID, handle @name, or URL) to (channel_id, uploads_playlist_id, channel_title).

    Args:
        youtube: YouTube API client.
        channel_input: Channel ID (e.g. UC...), handle (e.g. @Creator), or channel URL.

    Returns:
        tuple of (channel_id, uploads_playlist_id, channel_title).

    Raises:
        ValueError: If the channel is not found.
    """
    id_type, id_value = parse_channel_input(channel_input)

    items = []
    if id_type == "id":
        response = youtube.channels().list(
            part="contentDetails,snippet",
            id=id_value,
        ).execute()
        items = response.get("items", [])
    elif id_type == "handle":
        # 1. Try with forHandle
        response = youtube.channels().list(
            part="contentDetails,snippet",
            forHandle=id_value,
        ).execute()
        items = response.get("items", [])

        # 2. If not found and handle didn't have '@', try with '@'
        if not items and not id_value.startswith("@"):
            response = youtube.channels().list(
                part="contentDetails,snippet",
                forHandle=f"@{id_value}",
            ).execute()
            items = response.get("items", [])

        # 3. Fallback: try forUsername
        if not items:
            clean_name = id_value.lstrip("@")
            try:
                response = youtube.channels().list(
                    part="contentDetails,snippet",
                    forUsername=clean_name,
                ).execute()
                items = response.get("items", [])
            except Exception:
                pass
    elif id_type == "username":
        response = youtube.channels().list(
            part="contentDetails,snippet",
            forUsername=id_value,
        ).execute()
        items = response.get("items", [])
        if not items:
            response = youtube.channels().list(
                part="contentDetails,snippet",
                forHandle=id_value,
            ).execute()
            items = response.get("items", [])

    if not items:
        raise ValueError(
            f"Channel not found: '{channel_input}'. "
            "Check that the channel handle, URL, or ID is correct and public."
        )

    item = items[0]
    channel_id = item.get("id", id_value)
    channel_title = item.get("snippet", {}).get("title", channel_id)
    content_details = item.get("contentDetails", {})
    uploads_id = content_details.get("relatedPlaylists", {}).get("uploads", "")

    if not uploads_id and channel_id.startswith("UC"):
        uploads_id = "UU" + channel_id[2:]

    logger.info("Resolved channel '%s' -> ID: %s, Uploads: %s", channel_title, channel_id, uploads_id)
    return channel_id, uploads_id, channel_title


def get_uploads_playlist_id(youtube, channel_id: str) -> str:
    """Get the uploads playlist ID for a YouTube channel ID or handle/URL.

    Args:
        youtube: YouTube API client.
        channel_id: The YouTube channel ID, handle, or URL.

    Returns:
        The uploads playlist ID.

    Raises:
        ValueError: If the channel is not found.
    """
    _, uploads_id, _ = resolve_channel(youtube, channel_id)
    return uploads_id


def fetch_channel_videos(
    youtube,
    channel_id: str,
    limit: int = 20,
) -> list:
    """Fetch recent videos from a channel's uploads playlist.

    Args:
        youtube: YouTube API client.
        channel_id: The YouTube channel ID.
        limit: Maximum number of videos to fetch (default: 20).

    Returns:
        List of dicts with keys: video_id, title, published_at.

    Raises:
        ValueError: If the channel is not found.
        HttpError: For API errors.
    """
    uploads_id = get_uploads_playlist_id(youtube, channel_id)

    videos = []
    page_token = None

    while len(videos) < limit:
        # Fetch up to 50 per page (API maximum for playlistItems)
        per_page = min(50, limit - len(videos))

        request_kwargs = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_id,
            "maxResults": per_page,
        }
        if page_token:
            request_kwargs["pageToken"] = page_token

        try:
            response = youtube.playlistItems().list(**request_kwargs).execute()
        except HttpError as e:
            logger.error("API error fetching playlist items: %s", e)
            raise

        items = response.get("items", [])
        if not items:
            break

        for item in items:
            if len(videos) >= limit:
                break

            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})

            video_id = content_details.get("videoId", "")
            if not video_id:
                # Fallback to snippet.resourceId
                resource_id = snippet.get("resourceId", {})
                video_id = resource_id.get("videoId", "")

            if not video_id:
                logger.warning("Skipping playlist item with no video ID.")
                continue

            videos.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "published_at": content_details.get(
                    "videoPublishedAt",
                    snippet.get("publishedAt", ""),
                ),
            })

        page_count = len(videos)
        logger.info("Fetched %d/%d videos so far.", page_count, limit)

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    logger.info("Total videos collected: %d", len(videos))
    return videos


def save_channel_videos_csv(
    videos: list,
    channel_id: str,
) -> Path:
    """Save channel video metadata to a CSV file.

    Args:
        videos: List of video metadata dicts.
        channel_id: The channel ID (used in filename).

    Returns:
        Path to the saved CSV file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(videos, columns=["video_id", "title", "published_at"])
    output_path = RAW_DATA_DIR / f"channel_videos_{channel_id}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("Saved %d video records to %s", len(df), output_path)
    return output_path


def fetch_comments_for_videos(
    youtube,
    videos: list,
    channel_id: str,
    max_pages_per_video: Optional[int] = None,
) -> None:
    """Fetch comments for each video in the list.

    Args:
        youtube: YouTube API client.
        videos: List of video metadata dicts (must have 'video_id' key).
        channel_id: The channel ID of the video creator.
        max_pages_per_video: Max comment pages per video (None for all).
    """
    total = len(videos)
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, video in enumerate(videos, 1):
        video_id = video["video_id"]
        title = video.get("title", "Unknown")
        print(f"\n[{i}/{total}] Fetching comments for: {title} ({video_id})")

        try:
            threads = fetch_comment_threads(
                youtube, video_id, max_pages=max_pages_per_video,
            )

            if not threads:
                print(f"  → No comments found.")
                skip_count += 1
                continue

            df = threads_to_dataframe(threads, video_id, channel_id)
            csv_path = save_comments_csv(df, video_id)
            print(f"  → Saved {len(df)} comments to {csv_path}")
            success_count += 1

        except HttpError as e:
            status = e.resp.status
            if status == 403:
                print(f"  → Skipped (comments disabled or access denied).")
                skip_count += 1
            elif status == 404:
                print(f"  → Skipped (video not found or private).")
                skip_count += 1
            else:
                print(f"  → Error (HTTP {status}): {e}")
                error_count += 1

        except Exception as e:
            print(f"  → Unexpected error: {e}")
            error_count += 1

    print("\n" + "=" * 60)
    print("  Batch Comment Collection Summary")
    print("=" * 60)
    print(f"  Videos processed : {total}")
    print(f"  Successful       : {success_count}")
    print(f"  Skipped          : {skip_count}")
    print(f"  Errors           : {error_count}")
    print("=" * 60 + "\n")


def main() -> None:
    """Main entry point for the channel video collection CLI."""
    parser = argparse.ArgumentParser(
        description="Fetch recent videos from a YouTube channel.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        help="YouTube channel ID (UC...), handle (@Creator), or channel URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of videos to fetch (default: 20).",
    )
    parser.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Also fetch comments for each video.",
    )
    parser.add_argument(
        "--max-comment-pages",
        type=int,
        default=None,
        help="Max comment pages per video (default: all).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    print(f"\nFetching up to {args.limit} videos for channel: {args.channel}")

    # Build API client
    youtube = build_youtube_client()

    # Resolve channel identifier (URL, @handle, or ID)
    try:
        channel_id, uploads_id, channel_title = resolve_channel(youtube, args.channel)
        print(f"Channel resolved: {channel_title} (ID: {channel_id})")
    except ValueError as e:
        print(f"\n[ERROR] {e}\n")
        return
    except HttpError as e:
        print(f"\n[ERROR] API error: {e}\n")
        return

    # Fetch video list
    try:
        videos = fetch_channel_videos(youtube, channel_id, limit=args.limit)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n")
        return
    except HttpError as e:
        print(f"\n[ERROR] API error: {e}\n")
        return

    if not videos:
        print(f"\nNo videos found for channel '{channel_title}'.\n")
        return

    # Save video metadata using the resolved channel_id
    csv_path = save_channel_videos_csv(videos, channel_id)

    print(f"\nSaved {len(videos)} video(s) to {csv_path}")
    for v in videos[:5]:
        print(f"  - {v['title']} ({v['video_id']})")
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more")

    # Optionally fetch comments for each video
    if args.fetch_comments:
        print("\nStarting comment collection for all videos...")
        fetch_comments_for_videos(
            youtube, videos, channel_id,
            max_pages_per_video=args.max_comment_pages,
        )


if __name__ == "__main__":
    main()
