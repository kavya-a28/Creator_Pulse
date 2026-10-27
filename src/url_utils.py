"""
YouTube URL parsing and video ID extraction utilities.

Supports standard watch URLs, shortened youtu.be links, Shorts, and mobile URLs.
"""

import re
from urllib.parse import parse_qs, urlparse

# YouTube video IDs are exactly 11 characters: alphanumeric, hyphens, underscores
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Extract and validate a YouTube video ID from a URL or raw ID.

    Supported formats:
        - Raw 11-character video ID
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtube.com/watch?v=VIDEO_ID
        - https://m.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - URLs with extra query parameters (&t=, &list=, etc.)

    Args:
        url_or_id: A YouTube URL or video ID string.

    Returns:
        The 11-character video ID.

    Raises:
        ValueError: If the input is not a valid YouTube URL or video ID.
    """
    if not url_or_id or not url_or_id.strip():
        raise ValueError("Empty input. Provide a YouTube video URL or 11-character video ID.")

    text = url_or_id.strip()

    # Check if it's already a bare video ID
    if VIDEO_ID_PATTERN.match(text):
        return text

    # Try to parse as URL
    parsed = urlparse(text)

    # Ensure it's actually a URL (has a scheme or starts with www/youtu)
    if not parsed.scheme and not text.startswith(("www.", "youtu")):
        raise ValueError(
            f"Invalid input: '{text}'. "
            "Provide a YouTube URL (e.g. https://www.youtube.com/watch?v=VIDEO_ID) "
            "or an 11-character video ID."
        )

    # Add scheme if missing so urlparse works correctly
    if not parsed.scheme:
        text = "https://" + text
        parsed = urlparse(text)

    hostname = (parsed.hostname or "").lower()

    # Validate hostname
    valid_hosts = {
        "www.youtube.com", "youtube.com", "m.youtube.com",
        "youtu.be", "www.youtu.be",
    }

    if hostname not in valid_hosts:
        raise ValueError(
            f"Not a YouTube URL: '{url_or_id}'. "
            "Supported domains: youtube.com, youtu.be"
        )

    video_id = None

    # youtu.be/VIDEO_ID
    if hostname in ("youtu.be", "www.youtu.be"):
        path = parsed.path.lstrip("/")
        if path:
            video_id = path.split("/")[0]

    # youtube.com/watch?v=VIDEO_ID
    elif parsed.path == "/watch":
        query_params = parse_qs(parsed.query)
        v_param = query_params.get("v")
        if v_param:
            video_id = v_param[0]

    # youtube.com/shorts/VIDEO_ID
    elif parsed.path.startswith("/shorts/"):
        parts = parsed.path.split("/")
        if len(parts) >= 3 and parts[2]:
            video_id = parts[2]

    # youtube.com/embed/VIDEO_ID
    elif parsed.path.startswith("/embed/"):
        parts = parsed.path.split("/")
        if len(parts) >= 3 and parts[2]:
            video_id = parts[2]

    if not video_id:
        raise ValueError(
            f"Could not extract video ID from: '{url_or_id}'. "
            "Supported formats: youtube.com/watch?v=ID, youtu.be/ID, youtube.com/shorts/ID"
        )

    # Validate extracted ID
    if not VIDEO_ID_PATTERN.match(video_id):
        raise ValueError(
            f"Extracted video ID '{video_id}' is invalid. "
            "YouTube video IDs are exactly 11 characters (letters, digits, hyphens, underscores)."
        )

    return video_id


def parse_channel_input(channel_input: str) -> tuple[str, str]:
    """Extract and categorize a YouTube channel identifier from a URL, handle, or ID.

    Supported formats:
        - Full URL with handle: https://www.youtube.com/@ChannelHandle
        - Full URL with channel ID: https://www.youtube.com/channel/UC...
        - Full URL with custom name: https://www.youtube.com/c/CustomName
        - Full URL with username: https://www.youtube.com/user/UserName
        - Handle starting with @: @ChannelHandle
        - Plain handle: ChannelHandle
        - Raw channel ID: UC... (24 characters)

    Args:
        channel_input: A channel URL, handle, or ID.

    Returns:
        tuple of (identifier_type, identifier_value) where identifier_type
        is 'id', 'handle', or 'username'.

    Raises:
        ValueError: If input is empty.
    """
    if not channel_input or not channel_input.strip():
        raise ValueError("Empty input. Provide a channel URL, handle (e.g. @Creator), or channel ID (UC...).")

    text = channel_input.strip()

    # Check for URL format
    if text.startswith(("http://", "https://", "www.", "youtube.com", "m.youtube.com")):
        if not text.startswith(("http://", "https://")):
            text = "https://" + text
        parsed = urlparse(text)
        path = parsed.path.strip("/")
        parts = path.split("/")

        if parts and parts[0]:
            # Handles like /@ChannelHandle or /@ChannelHandle/videos
            if parts[0].startswith("@"):
                return "handle", parts[0]
            # Standard channel ID like /channel/UC...
            if parts[0] == "channel" and len(parts) > 1:
                return "id", parts[1]
            # Custom channel URL like /c/CustomName
            if parts[0] == "c" and len(parts) > 1:
                return "handle", parts[1]
            # Legacy username URL like /user/Username
            if parts[0] == "user" and len(parts) > 1:
                return "username", parts[1]
            # Fallback for paths like /ChannelHandle
            return "handle", parts[0]

    # Handle prefixed with @
    if text.startswith("@"):
        return "handle", text

    # Standard YouTube channel IDs start with 'UC'
    if text.startswith("UC"):
        return "id", text

    # Otherwise treat as handle/username
    return "handle", text
