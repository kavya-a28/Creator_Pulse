"""
CreatorPulse configuration module.

Handles API key loading, YouTube client creation, and shared settings.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LABELLED_DATA_DIR = DATA_DIR / "labelled"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def load_api_key() -> str:
    """Load the YouTube API key from the .env file.

    Returns:
        The API key string.

    Raises:
        SystemExit: If the API key is missing or empty.
    """
    # Load .env from project root
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key or api_key.strip() == "" or api_key.strip() == "your_api_key_here":
        print(
            "\n[ERROR] YouTube API key not configured.\n"
            "\n"
            "To fix this:\n"
            "  1. Go to https://console.cloud.google.com/apis/credentials\n"
            "  2. Create a project (or select an existing one)\n"
            "  3. Enable 'YouTube Data API v3'\n"
            "  4. Create an API key\n"
            "  5. Copy .env.example to .env:\n"
            "       copy .env.example .env\n"
            "  6. Paste your API key in .env:\n"
            "       YOUTUBE_API_KEY=AIza...\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return api_key.strip()


def build_youtube_client():
    """Build and return an authenticated YouTube Data API v3 client.

    Returns:
        A googleapiclient.discovery.Resource for the YouTube API.
    """
    from googleapiclient.discovery import build

    api_key = load_api_key()
    logger.info("Building YouTube API client...")

    # Use the classic API address if this computer can't resolve youtube.googleapis.com,
    # and the system HTTPS proxy if one is configured (see src.collect).
    from src.collect import _http_with_proxy, choose_endpoint, system_proxy
    kwargs = {"developerKey": api_key, "cache_discovery": False}
    ep = choose_endpoint()
    if ep["endpoint"]:
        logger.info("Using %s (youtube.googleapis.com could not be resolved).", ep["host"])
        kwargs["client_options"] = {"api_endpoint": ep["endpoint"]}
    proxy = system_proxy()
    if proxy:
        kwargs["http"] = _http_with_proxy(proxy)
    return build("youtube", "v3", **kwargs)


def get_video_channel_id(youtube, video_id: str) -> str:
    """Fetch the channel ID of the video's creator.

    Args:
        youtube: YouTube API client.
        video_id: The YouTube video ID.

    Returns:
        The channel ID of the video's uploader.

    Raises:
        ValueError: If the video is not found.
    """
    response = youtube.videos().list(
        part="snippet",
        id=video_id,
        fields="items/snippet/channelId",
    ).execute()

    items = response.get("items", [])
    if not items:
        raise ValueError(
            f"Video not found: '{video_id}'. "
            "Check that the video ID is correct and the video is public."
        )

    return items[0]["snippet"]["channelId"]


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application and configure stdout encoding.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
    """
    # Fix Windows terminal UTF-8 encoding for emojis and non-ASCII text
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
