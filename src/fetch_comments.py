"""
CreatorPulse — YouTube comment collection module.

Fetches all public comments for a video using the YouTube Data API v3,
detects creator replies, and saves results to CSV.

Usage:
    python -m src.fetch_comments --video VIDEO_ID
    python -m src.fetch_comments --url "https://www.youtube.com/watch?v=VIDEO_ID"
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from googleapiclient.errors import HttpError

from src.config import (
    RAW_DATA_DIR,
    build_youtube_client,
    get_video_channel_id,
    setup_logging,
)
from src.url_utils import extract_video_id

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds

# HTTP status codes
RETRYABLE_STATUS_CODES = {429, 500, 503}


def fetch_comment_threads(
    youtube,
    video_id: str,
    max_pages: Optional[int] = None,
    order: Optional[str] = None,
) -> list:
    """Fetch all comment threads for a video using pagination.

    Args:
        youtube: YouTube API client.
        video_id: The YouTube video ID.
        max_pages: Maximum number of pages to fetch (None for all).
        order: Optional API sort order ("time" = newest first, "relevance").
            None keeps the API default (previous behaviour).

    Returns:
        List of comment thread items from the API.

    Raises:
        HttpError: For non-retryable API errors.
    """
    all_threads = []
    seen_ids = set()
    page_token = None
    page_count = 0

    while True:
        # Check page limit
        if max_pages is not None and page_count >= max_pages:
            logger.info("Reached max page limit (%d pages).", max_pages)
            break

        # Fetch one page with retry logic
        response = _fetch_page_with_retry(youtube, video_id, page_token, order=order)

        items = response.get("items", [])
        for item in items:
            comment_id = item["id"]
            if comment_id not in seen_ids:
                seen_ids.add(comment_id)
                all_threads.append(item)

        page_count += 1
        fetched_total = len(all_threads)
        logger.info(
            "Page %d: fetched %d threads (total unique: %d)",
            page_count, len(items), fetched_total,
        )

        # Check for next page
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return all_threads


def _fetch_page_with_retry(
    youtube,
    video_id: str,
    page_token: Optional[str],
    order: Optional[str] = None,
) -> dict:
    """Fetch a single page of comment threads with exponential backoff.

    Args:
        youtube: YouTube API client.
        video_id: The YouTube video ID.
        page_token: Pagination token (None for first page).

    Returns:
        The API response dictionary.

    Raises:
        HttpError: For permanent API errors after exhausting retries.
    """
    request_kwargs = {
        "part": "snippet,replies",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
    }
    if order:
        request_kwargs["order"] = order
    if page_token:
        request_kwargs["pageToken"] = page_token

    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = youtube.commentThreads().list(**request_kwargs).execute()
            return response

        except HttpError as e:
            status_code = e.resp.status

            # Handle specific permanent errors
            if status_code == 403:
                error_reason = _get_error_reason(e)

                if error_reason == "commentsDisabled":
                    print(
                        f"\n[ERROR] Comments are disabled for video '{video_id}'.\n"
                        "This video does not allow public comments.\n"
                    )
                    raise

                if error_reason in ("quotaExceeded", "dailyLimitExceeded"):
                    print(
                        "\n[ERROR] YouTube API quota exceeded.\n"
                        "Your daily quota (10,000 units) has been used up.\n"
                        "Check usage at: https://console.cloud.google.com/apis/dashboard\n"
                        "Quota resets at midnight Pacific Time.\n"
                    )
                    raise

                if error_reason in ("forbidden", "accessNotConfigured"):
                    print(
                        "\n[ERROR] API access denied.\n"
                        "Make sure YouTube Data API v3 is enabled in your GCP project:\n"
                        "  https://console.cloud.google.com/apis/library/youtube.googleapis.com\n"
                    )
                    raise

            if status_code == 404:
                print(
                    f"\n[ERROR] Video not found: '{video_id}'.\n"
                    "Check that the video ID is correct and the video is public.\n"
                )
                raise

            if status_code == 400:
                print(
                    f"\n[ERROR] Bad request (status 400).\n"
                    f"Details: {e}\n"
                    "Check that your API key is valid.\n"
                )
                raise

            # Retry on transient errors
            if status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                logger.warning(
                    "Transient error (HTTP %d) on attempt %d/%d. "
                    "Retrying in %.1f seconds...",
                    status_code, attempt, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue

            # Exhausted retries or unexpected error
            logger.error("API error (HTTP %d) after %d attempts.", status_code, attempt)
            raise

        except Exception as e:
            # Network errors, timeouts, etc.
            if attempt < MAX_RETRIES:
                logger.warning(
                    "Network error on attempt %d/%d: %s. Retrying in %.1f seconds...",
                    attempt, MAX_RETRIES, e, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            logger.error("Network error after %d attempts: %s", attempt, e)
            raise

    # Should not reach here, but just in case
    raise RuntimeError("Unexpected exit from retry loop")


def _get_error_reason(http_error: HttpError) -> str:
    """Extract the error reason string from an HttpError.

    Args:
        http_error: The HttpError from the API.

    Returns:
        The error reason string, or "unknown" if not found.
    """
    try:
        import json
        error_body = json.loads(http_error.content.decode("utf-8"))
        errors = error_body.get("error", {}).get("errors", [])
        if errors:
            return errors[0].get("reason", "unknown")
    except Exception:
        pass
    return "unknown"


def detect_creator_reply(
    thread_item: dict,
    creator_channel_id: str,
) -> Optional[bool]:
    """Check whether the video creator has replied to a comment thread.

    Args:
        thread_item: A single comment thread item from the API.
        creator_channel_id: The channel ID of the video's creator.

    Returns:
        True if a creator reply was found in the available replies.
        False if replies were checked and no creator reply was found,
              AND all replies were available (totalReplyCount matches).
        None if reply data is incomplete (API returns a subset), so
             we cannot be certain whether the creator replied.
    """
    snippet = thread_item.get("snippet", {})
    total_reply_count = snippet.get("totalReplyCount", 0)

    # No replies at all
    if total_reply_count == 0:
        return False

    # Check available replies
    replies_data = thread_item.get("replies", {})
    reply_comments = replies_data.get("comments", [])

    # No reply data available even though replies exist
    if not reply_comments:
        return None  # Uncertain — replies exist but weren't returned

    # Check each reply for creator authorship
    for reply in reply_comments:
        reply_snippet = reply.get("snippet", {})
        author_channel = reply_snippet.get("authorChannelId", {})
        author_id = author_channel.get("value", "")

        if author_id == creator_channel_id:
            return True

    # Did we see all replies?
    if len(reply_comments) >= total_reply_count:
        return False  # We checked all replies, creator didn't reply
    else:
        return None  # API only returned a subset (max 5), can't be sure


def threads_to_dataframe(
    threads: list,
    video_id: str,
    creator_channel_id: str,
) -> pd.DataFrame:
    """Convert API comment thread items into a pandas DataFrame.

    Columns (in order):
        comment_id, video_id, text, author_channel_id,
        published_at, like_count, reply_count, creator_replied

    Args:
        threads: List of comment thread items from the API.
        video_id: The YouTube video ID.
        creator_channel_id: The channel ID of the video's creator.

    Returns:
        A DataFrame with one row per top-level comment.
    """
    rows = []

    for thread in threads:
        snippet = thread.get("snippet", {})
        top_comment = snippet.get("topLevelComment", {})
        comment_snippet = top_comment.get("snippet", {})

        # Extract author channel ID safely
        author_channel = comment_snippet.get("authorChannelId", {})
        author_id = author_channel.get("value", "") if isinstance(author_channel, dict) else ""

        creator_replied = detect_creator_reply(thread, creator_channel_id)

        rows.append({
            "comment_id": top_comment.get("id", ""),
            "video_id": video_id,
            "text": comment_snippet.get("textDisplay", ""),
            "author_channel_id": author_id,
            "published_at": comment_snippet.get("publishedAt", ""),
            "like_count": comment_snippet.get("likeCount", 0),
            "reply_count": snippet.get("totalReplyCount", 0),
            "creator_replied": creator_replied,
        })

    df = pd.DataFrame(rows, columns=[
        "comment_id", "video_id", "text", "author_channel_id",
        "published_at", "like_count", "reply_count", "creator_replied",
    ])

    return df


def save_comments_csv(df: pd.DataFrame, video_id: str) -> Path:
    """Save the comments DataFrame to a CSV file.

    Args:
        df: The comments DataFrame.
        video_id: The YouTube video ID (used in filename).

    Returns:
        The path to the saved CSV file.
    """
    # Ensure output directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DATA_DIR / f"comments_{video_id}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("Saved %d comments to %s", len(df), output_path)
    return output_path


def print_summary(
    df: pd.DataFrame,
    video_id: str,
    csv_path: Path,
    warnings: list,
) -> None:
    """Print a summary of the collection results.

    Args:
        df: The comments DataFrame.
        video_id: The video ID.
        csv_path: Path to the saved CSV.
        warnings: List of warning messages encountered.
    """
    print("\n" + "=" * 60)
    print("  CreatorPulse — Comment Collection Summary")
    print("=" * 60)
    print(f"  Video ID       : {video_id}")
    print(f"  Comments saved : {len(df)}")
    print(f"  CSV file       : {csv_path}")

    if len(df) > 0:
        creator_true = (df["creator_replied"] == True).sum()  # noqa: E712
        creator_false = (df["creator_replied"] == False).sum()  # noqa: E712
        creator_unknown = df["creator_replied"].isna().sum()

        print(f"  Creator replied : {creator_true} confirmed")
        print(f"  No creator reply: {creator_false} confirmed")
        print(f"  Reply uncertain : {creator_unknown} (incomplete data)")

    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    print("=" * 60 + "\n")


def main() -> None:
    """Main entry point for the comment collection CLI."""
    parser = argparse.ArgumentParser(
        description="Fetch YouTube comments for a video and save to CSV.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=str,
        help="YouTube video ID (11 characters).",
    )
    group.add_argument(
        "--url",
        type=str,
        help="YouTube video URL.",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of API pages to fetch (default: all).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    warnings_list = []

    # Extract video ID
    try:
        if args.url:
            video_id = extract_video_id(args.url)
        else:
            video_id = extract_video_id(args.video)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n")
        return

    print(f"\nFetching comments for video: {video_id}")

    # Build API client
    youtube = build_youtube_client()

    # Get video creator's channel ID
    try:
        creator_channel_id = get_video_channel_id(youtube, video_id)
        logger.info("Video creator channel ID: %s", creator_channel_id)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n")
        return
    except HttpError as e:
        print(f"\n[ERROR] Failed to fetch video info: {e}\n")
        return

    # Fetch comments
    try:
        threads = fetch_comment_threads(
            youtube, video_id, max_pages=args.max_pages,
        )
    except HttpError:
        # Specific error messages already printed in _fetch_page_with_retry
        return

    if not threads:
        print(f"\nNo comments found for video '{video_id}'.")
        print("The video may have no comments or comments may be restricted.\n")
        return

    # API limitation warning
    incomplete_reply_count = 0
    for thread in threads:
        snippet = thread.get("snippet", {})
        total_replies = snippet.get("totalReplyCount", 0)
        available_replies = len(
            thread.get("replies", {}).get("comments", [])
        )
        if total_replies > available_replies:
            incomplete_reply_count += 1

    if incomplete_reply_count > 0:
        warnings_list.append(
            f"{incomplete_reply_count} comment(s) have more replies than the API returned. "
            "The 'creator_replied' field may be uncertain for these."
        )

    # Convert to DataFrame and save
    df = threads_to_dataframe(threads, video_id, creator_channel_id)
    csv_path = save_comments_csv(df, video_id)

    # Print summary
    print_summary(df, video_id, csv_path, warnings_list)


if __name__ == "__main__":
    main()
