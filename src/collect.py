"""
CreatorPulse — comment collection service for the app.

Wraps the existing collectors (src.fetch_comments, src.fetch_channel_videos,
src.url_utils, src.config) with:

* URL validation that tells video URLs from channel URLs and explains
  unsupported inputs (playlists, search pages, non-YouTube links);
* a per-video comment cap and newest-first ordering;
* per-video capture notes (complete, newest N, or unknown order) used by the
  trend rules in src.insights;
* API errors turned into CollectionError with a plain-language message and a
  fix (missing key, invalid key, API not enabled, quota, comments disabled,
  video/channel not found, network).

Quota cost (YouTube Data API v3, 10,000 units/day by default):
    video URL    1 (video details) + 1 per 100 comments
    channel URL  1-3 (resolve) + 1 per 50 videos (list) + 1 per 50 videos
                 (details) + 1 per 100 comments per video
"""

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.config import PROJECT_ROOT
from src.url_utils import VIDEO_ID_PATTERN, extract_video_id, parse_channel_input

COMMENTS_PER_PAGE = 100
MAX_VIDEOS_LIMIT = 50
MAX_COMMENTS_LIMIT = 2000
RAW_COLUMNS = ["comment_id", "video_id", "text", "author_channel_id", "published_at", "like_count",
               "reply_count", "creator_replied", "channel_id", "channel_title", "video_title",
               "video_published_at"]
CHANNEL_ID_PATTERN = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
YOUTUBE_HOSTS = {"www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be",
                 "music.youtube.com"}


class CollectionError(Exception):
    """A collection problem with a user-facing message and a suggested fix."""

    def __init__(self, code: str, message: str, fix: str = "", detail: str = ""):
        super().__init__(message)
        self.code, self.message, self.fix, self.detail = code, message, fix, detail

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "fix": self.fix, "detail": self.detail}


@dataclass
class Target:
    kind: str                 # "video" or "channel"
    value: str                # video id, or channel input understood by resolve_channel
    url: str
    note: str = ""


@dataclass
class CollectionResult:
    comments: pd.DataFrame
    target: Target
    channel_id: str = ""
    channel_title: str = ""
    channel_thumbnail: str = ""                           # profile picture URL ("" if unavailable)
    videos: List[dict] = field(default_factory=list)      # per-video capture notes
    skipped_videos: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    api_units_used: int = 0
    collected_at: str = ""


# --------------------------------------------------------------------------
# URL validation
# --------------------------------------------------------------------------

def parse_target(text: str) -> Target:
    """Decide whether the input is a video or a channel. Raises CollectionError."""
    raw = (text or "").strip()
    if not raw:
        raise CollectionError("empty_url", "Paste a YouTube video or channel URL to start.")
    if VIDEO_ID_PATTERN.match(raw) and not raw.startswith("UC"):
        return Target("video", raw, raw)
    if CHANNEL_ID_PATTERN.match(raw):
        return Target("channel", raw, raw)
    if raw.startswith("@") and " " not in raw:
        return Target("channel", raw, raw)

    url = raw if re.match(r"^[a-z]+://", raw, re.I) else "https://" + raw
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise CollectionError(
            "not_youtube", "That doesn't look like a YouTube link.",
            "Use a link from youtube.com or youtu.be, for example https://www.youtube.com/watch?v=VIDEO_ID "
            "or https://www.youtube.com/@ChannelName.")
    path = parsed.path.rstrip("/")
    query = parse_qs(parsed.query)

    if path in ("/playlist",) or (path == "" and "list" in query):
        raise CollectionError(
            "unsupported_playlist", "Playlist links aren't supported yet.",
            "Open one video from the playlist and paste its link, or paste the channel link instead.")
    if path in ("/results", "/feed/trending", "/feed/subscriptions") or path.startswith("/hashtag"):
        raise CollectionError(
            "unsupported_page", "Search, feed and hashtag pages can't be analyzed.",
            "Paste a link to a single video or to a channel.")
    if path == "/live" or path.startswith("/live/"):
        vid = path.split("/")[2] if path.count("/") >= 2 else ""
        if VIDEO_ID_PATTERN.match(vid):
            return Target("video", vid, raw)
    if host.endswith("youtu.be") or path in ("/watch",) or path.startswith(("/shorts/", "/embed/")):
        try:
            vid = extract_video_id(url)
        except ValueError as exc:
            raise CollectionError("invalid_video_url", "This video link is incomplete or malformed.",
                                  "Copy the link again from the video's Share button.", str(exc)) from exc
        note = "Playlist part of the link was ignored; only this video is analyzed." if "list" in query else ""
        return Target("video", vid, raw, note)
    first = path.strip("/").split("/")[0] if path.strip("/") else ""
    if first.startswith("@") or first in ("channel", "c", "user"):
        kind, value = parse_channel_input(url)
        if kind == "id" and not CHANNEL_ID_PATTERN.match(value):
            raise CollectionError("invalid_channel_url", "This channel ID doesn't look right.",
                                  "Channel IDs start with UC and are 24 characters long.")
        return Target("channel", url, raw)
    if not first:
        raise CollectionError("unsupported_page", "That's the YouTube home page, not a video or channel.",
                              "Paste a link to a single video or to a channel.")
    raise CollectionError(
        "unsupported_page", f"YouTube pages like '/{first}' can't be analyzed.",
        "Paste a video link (youtube.com/watch?v=..., youtu.be/..., /shorts/...) or a channel link "
        "(youtube.com/@name, /channel/UC...).")


# --------------------------------------------------------------------------
# Client and errors
# --------------------------------------------------------------------------

def api_key_configured() -> bool:
    """True if a YouTube API key is set (in the environment or .env). Never returns the key."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass
    key = (os.getenv("YOUTUBE_API_KEY") or "").strip()
    return bool(key) and key not in ("your_api_key_here", "YOUR_API_KEY_HERE")


API_HOST = "youtube.googleapis.com"          # what google-api-python-client 2.x uses by default
CLASSIC_HOST = "www.googleapis.com"          # the classic address; serves the same YouTube Data API v3


def host_resolves(host: str) -> bool:
    import socket
    try:
        socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return True
    except OSError:
        return False


def system_proxy() -> str:
    """HTTPS proxy from environment variables or Windows Internet settings ('' if none)."""
    import urllib.request
    try:
        return urllib.request.getproxies().get("https", "") or ""
    except Exception:
        return ""


def choose_endpoint() -> dict:
    """Which API address to use: the default one, or the classic one when the default can't be looked up.

    Some DNS servers (routers, ISPs, campus networks) fail to resolve
    youtube.googleapis.com while the browser still works (Chrome can use its own
    secure DNS). www.googleapis.com serves the same API.
    """
    if system_proxy():      # the proxy looks names up itself; local DNS doesn't matter
        return {"host": API_HOST, "endpoint": None, "fallback": False, "via_proxy": True}
    if host_resolves(API_HOST):
        return {"host": API_HOST, "endpoint": None, "fallback": False}
    if host_resolves(CLASSIC_HOST):
        return {"host": CLASSIC_HOST, "endpoint": f"https://{CLASSIC_HOST}/", "fallback": True}
    return {"host": API_HOST, "endpoint": None, "fallback": False, "unresolved": True}


def _http_with_proxy(proxy: str):
    import httplib2
    from urllib.parse import urlparse
    u = urlparse(proxy if "://" in proxy else f"http://{proxy}")
    info = httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, u.hostname, u.port or 8080,
                              proxy_user=u.username, proxy_pass=u.password)
    return httplib2.Http(timeout=60, proxy_info=info)


def build_client():
    """Build the YouTube client, or raise CollectionError.

    Uses the API key from .env (src.config.load_api_key), the classic API address
    when the default one can't be resolved, and the system HTTPS proxy if set.
    """
    if not api_key_configured():
        raise CollectionError(
            "missing_api_key", "No YouTube API key is configured.",
            "Copy .env.example to .env in the project folder, set YOUTUBE_API_KEY=<your key>, "
            "then restart the app. See the README for how to create a key.")
    from src.config import load_api_key
    try:
        key = load_api_key()
    except SystemExit as exc:   # load_api_key exits when the key is missing
        raise CollectionError("missing_api_key", "No YouTube API key is configured.",
                              "Set YOUTUBE_API_KEY in .env and restart the app.") from exc
    ep = choose_endpoint()
    if ep.get("unresolved"):
        raise CollectionError(
            "dns_failed", "Your computer couldn't look up Google's API address (youtube.googleapis.com).",
            "Your browser may work because it uses its own DNS. For details run "
            "`python -m src.check_connection` in the project folder. Usually changing the computer's DNS to 8.8.8.8 / 1.1.1.1, turning off a VPN or ad "
            "blocker, or trying another network (e.g. a phone hotspot) solves it.")
    try:
        from googleapiclient.discovery import build
        kwargs = {"developerKey": key, "cache_discovery": False}
        if ep["endpoint"]:
            kwargs["client_options"] = {"api_endpoint": ep["endpoint"]}
        proxy = system_proxy()
        if proxy:
            kwargs["http"] = _http_with_proxy(proxy)
        return build("youtube", "v3", **kwargs)
    except Exception as exc:
        raise CollectionError("client_error", "Could not create the YouTube API client.",
                              "Check your internet connection and try again.", type(exc).__name__) from exc


def _reason(exc) -> str:
    try:
        body = json.loads(exc.content.decode("utf-8"))
        errs = body.get("error", {}).get("errors", [])
        if errs:
            return errs[0].get("reason", "") or ""
        return body.get("error", {}).get("status", "") or ""
    except Exception:
        return ""


def translate_http_error(exc, context: str = "") -> CollectionError:
    """Map a googleapiclient HttpError to a CollectionError."""
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None
    reason = _reason(exc)
    detail = f"HTTP {status} {reason}".strip()
    if reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded"):
        return CollectionError("quota_exceeded", "The YouTube API daily quota has been used up.",
                               "Quota resets at midnight Pacific Time. Try again later, or analyze fewer "
                               "videos / fewer comments per video.", detail)
    if reason == "commentsDisabled":
        return CollectionError("comments_disabled", f"Comments are turned off for {context or 'this video'}.",
                               "Try a different video.", detail)
    content = ""
    try:
        content = exc.content.decode("utf-8", "replace")
    except Exception:
        pass
    if reason in ("keyInvalid", "API_KEY_INVALID") or (status == 400 and "API key" in content):
        return CollectionError("api_key_invalid", "YouTube rejected the API key.",
                               "Check YOUTUBE_API_KEY in .env (no quotes or spaces) and that the key is "
                               "not restricted to other APIs.", detail)
    if reason in ("accessNotConfigured", "forbidden", "ipRefererBlocked", "keyExpired") or status == 403:
        return CollectionError("api_forbidden", "The YouTube API refused the request.",
                               "Make sure 'YouTube Data API v3' is enabled for your Google Cloud project and "
                               "the key's restrictions allow it.", detail)
    if status == 404 or reason in ("videoNotFound", "channelNotFound", "playlistNotFound"):
        return CollectionError("not_found", f"{(context or 'The video').capitalize()} was not found.",
                               "It may be private, deleted or age-restricted. Check the link.", detail)
    if status in (500, 503):
        return CollectionError("api_unavailable", "YouTube's API is having problems right now.",
                               "Wait a minute and try again.", detail)
    return CollectionError("api_error", "The YouTube API returned an error.",
                           "Try again; if it keeps happening, check the details below.", detail)


def _call(fn, context: str = ""):
    from googleapiclient.errors import HttpError
    try:
        return fn()
    except HttpError as exc:
        raise translate_http_error(exc, context) from exc
    except CollectionError:
        raise
    except ValueError as exc:      # resolve_channel raises ValueError when not found
        raise CollectionError("not_found", str(exc), "Check the link and that it is public.") from exc
    except Exception as exc:       # transport errors after the collector's retries
        detail = f"{type(exc).__name__}: {str(exc)[:200]}"
        if type(exc).__name__ == "ServerNotFoundError" or "Unable to find the server" in str(exc):
            raise CollectionError(
                "dns_failed", "Your computer couldn't look up Google's API address.",
                "The browser may still work because it uses its own DNS. Run `python -m src.check_connection` "
                "for details; usually setting DNS to 8.8.8.8 / 1.1.1.1, turning off a VPN or ad blocker, or "
                "another network fixes it.", detail) from exc
        raise CollectionError("network", "Couldn't reach YouTube.",
                              "Check your internet connection (and any VPN or proxy) and try again.", detail) from exc


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def fetch_video_details(youtube, video_ids: List[str]) -> dict:
    """video_id -> snippet/statistics for up to 50 ids per call (1 unit each)."""
    out = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        resp = _call(lambda: youtube.videos().list(part="snippet,statistics", id=",".join(chunk),
                                                   maxResults=50).execute(), "the video")
        for item in resp.get("items", []):
            sn, st = item.get("snippet", {}), item.get("statistics", {})
            out[item["id"]] = {
                "video_id": item["id"], "title": sn.get("title", ""), "channel_id": sn.get("channelId", ""),
                "channel_title": sn.get("channelTitle", ""), "published_at": sn.get("publishedAt", ""),
                "comment_count": int(st["commentCount"]) if "commentCount" in st else None,
                "live": sn.get("liveBroadcastContent", "none"),
            }
    return out


def fetch_channel_thumbnail(youtube, channel_id: str) -> str:
    """URL of the channel's profile picture (1 API unit). Returns "" on any problem: the picture is
    cosmetic, so it never stops an analysis."""
    if not channel_id:
        return ""
    try:
        resp = youtube.channels().list(part="snippet", id=channel_id).execute()
        thumbs = ((resp.get("items") or [{}])[0].get("snippet") or {}).get("thumbnails") or {}
        for size in ("medium", "default", "high"):
            url = (thumbs.get(size) or {}).get("url", "")
            if url.startswith("https://"):
                return url
    except Exception:  # noqa: BLE001
        pass
    return ""


def plan_quota(kind: str, max_videos: int, max_comments: int) -> int:
    """Upper-bound estimate of API units for one analysis."""
    pages = math.ceil(max_comments / COMMENTS_PER_PAGE)
    if kind == "video":
        return 2 + pages                     # video details + channel picture + comment pages
    return 4 + 2 * math.ceil(max_videos / 50) + max_videos * pages


def _capture_note(df: pd.DataFrame, n_fetched: int, max_comments: int, available: Optional[int]) -> dict:
    """Was every comment captured, or only the newest N (verified), or an unknown subset?"""
    ts = pd.to_datetime(df["published_at"], utc=True, errors="coerce").reset_index(drop=True)
    # YouTube returns a pinned comment first, whatever its date; judge the order without it.
    pinned = bool(len(ts) > 1 and pd.notna(ts.iloc[0]) and pd.notna(ts.iloc[1]) and ts.iloc[0] < ts.iloc[1])
    if pinned:
        ts = ts.iloc[1:]
    newest_first = bool(ts.notna().all() and (ts.diff().dropna() <= pd.Timedelta(0)).all())
    truncated = n_fetched >= max_comments
    if not truncated:
        capture = "complete"
    elif newest_first:
        capture = "newest_n"
    else:
        capture = "sampled"
    return {"n_comments": int(len(df)), "available_comment_count": available, "truncated": bool(truncated),
            "newest_first_verified": newest_first, "capture": capture, "pinned_first": pinned,
            "complete_since": (ts.min().isoformat() if capture == "newest_n" and ts.notna().any() else None)}


def collect(target: Target, max_videos: int = 5, max_comments: int = 300, youtube=None,
            on_progress: Optional[Callable[[str], None]] = None) -> CollectionResult:
    """Collect top-level comments for a video or the latest videos of a channel."""
    from src.fetch_channel_videos import fetch_channel_videos, resolve_channel
    from src.fetch_comments import fetch_comment_threads, threads_to_dataframe

    max_videos = max(1, min(int(max_videos), MAX_VIDEOS_LIMIT))
    max_comments = max(1, min(int(max_comments), MAX_COMMENTS_LIMIT))
    youtube = youtube or build_client()
    say = on_progress or (lambda msg: None)
    units = 0
    res = CollectionResult(pd.DataFrame(columns=RAW_COLUMNS), target)

    if target.kind == "video":
        say("Checking the video")
        details = fetch_video_details(youtube, [target.value])
        units += 1
        if target.value not in details:
            raise CollectionError("not_found", "That video was not found.",
                                  "It may be private, deleted or age-restricted. Check the link.")
        videos = [details[target.value]]
        res.channel_id, res.channel_title = videos[0]["channel_id"], videos[0]["channel_title"]
    else:
        say("Finding the channel")
        ch_id, _, ch_title = _call(lambda: resolve_channel(youtube, target.value), "the channel")
        units += 2
        res.channel_id, res.channel_title = ch_id, ch_title
        say(f"Listing the latest {max_videos} videos of {ch_title}")
        listed = _call(lambda: fetch_channel_videos(youtube, ch_id, limit=max_videos), "the channel")
        units += 1 + math.ceil(max_videos / 50)
        if not listed:
            raise CollectionError("no_videos", f"{ch_title} has no public videos to analyze.",
                                  "Try another channel.")
        details = fetch_video_details(youtube, [v["video_id"] for v in listed])
        units += math.ceil(len(listed) / 50)
        videos = [details.get(v["video_id"], {"video_id": v["video_id"], "title": v.get("title", ""),
                                              "channel_id": ch_id, "channel_title": ch_title,
                                              "published_at": v.get("published_at", ""),
                                              "comment_count": None, "live": "none"}) for v in listed]

    frames = []
    for i, v in enumerate(videos, 1):
        vid, title = v["video_id"], v["title"] or v["video_id"]
        if v.get("comment_count") == 0:
            res.skipped_videos.append({"video_id": vid, "title": title, "reason": "no comments yet"})
            continue
        say(f"Collecting comments from video {i} of {len(videos)}: {title[:60]}")
        pages = math.ceil(max_comments / COMMENTS_PER_PAGE)
        try:
            threads = _call(lambda: fetch_comment_threads(youtube, vid, max_pages=pages, order="time"),
                            f"'{title[:60]}'")
        except CollectionError as err:
            if target.kind == "video" or err.code not in ("comments_disabled", "not_found"):
                if err.code == "quota_exceeded" and frames:
                    res.warnings.append("The API quota ran out part-way; results cover the videos collected "
                                        "before that.")
                    res.skipped_videos.append({"video_id": vid, "title": title, "reason": "quota exceeded"})
                    break
                raise
            res.skipped_videos.append({"video_id": vid, "title": title, "reason": err.message})
            continue
        units += max(1, math.ceil(len(threads) / COMMENTS_PER_PAGE))
        df = threads_to_dataframe(threads, vid, v.get("channel_id") or res.channel_id).head(max_comments)
        df["channel_id"] = v.get("channel_id") or res.channel_id
        df["channel_title"] = v.get("channel_title") or res.channel_title
        df["video_title"] = v["title"]
        df["video_published_at"] = v.get("published_at", "")
        note = _capture_note(df, len(threads), max_comments, v.get("comment_count"))
        res.videos.append({"video_id": vid, "title": v["title"], "published_at": v.get("published_at", ""),
                           **note})
        if df.empty:
            res.skipped_videos.append({"video_id": vid, "title": title, "reason": "no public comments returned"})
            continue
        frames.append(df)

    res.channel_thumbnail = fetch_channel_thumbnail(youtube, res.channel_id)
    units += 1
    if frames:
        res.comments = pd.concat(frames, ignore_index=True)[RAW_COLUMNS]
    res.api_units_used = units
    res.collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if res.comments.empty:
        reasons = "; ".join(f"{s['title'][:40]}: {s['reason']}" for s in res.skipped_videos[:3])
        raise CollectionError("no_comments", "No public comments could be collected.",
                              "Try a video with comments enabled, or a channel with more recent videos.",
                              reasons)
    return res
