"""Tests for src.fetch_channel_videos — Channel video collection."""

from unittest.mock import MagicMock

import pytest

from src.fetch_channel_videos import (
    fetch_channel_videos,
    get_uploads_playlist_id,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def make_channels_response(uploads_playlist_id="UU_creator_456"):
    """Build a channels.list API response."""
    return {
        "items": [
            {
                "contentDetails": {
                    "relatedPlaylists": {
                        "uploads": uploads_playlist_id,
                    }
                }
            }
        ]
    }


def make_playlist_items_response(videos, next_page_token=None):
    """Build a playlistItems.list API response."""
    items = []
    for v in videos:
        items.append({
            "snippet": {
                "title": v.get("title", "Test Video"),
                "publishedAt": v.get("published_at", "2026-01-01T00:00:00Z"),
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": v["video_id"],
                },
            },
            "contentDetails": {
                "videoId": v["video_id"],
                "videoPublishedAt": v.get("published_at", "2026-01-01T00:00:00Z"),
            },
        })

    response = {"items": items}
    if next_page_token:
        response["nextPageToken"] = next_page_token
    return response


# ── Uploads Playlist ID ──────────────────────────────────────────────


class TestUploadsPlaylistId:
    """Test uploads playlist ID extraction."""

    def test_valid_channel(self):
        """Should return the uploads playlist ID."""
        mock_youtube = MagicMock()
        mock_youtube.channels.return_value.list.return_value.execute.return_value = (
            make_channels_response("UU_creator_456")
        )

        result = get_uploads_playlist_id(mock_youtube, "UC_creator_456")
        assert result == "UU_creator_456"

    def test_channel_not_found(self):
        """Should raise ValueError for missing channel."""
        mock_youtube = MagicMock()
        mock_youtube.channels.return_value.list.return_value.execute.return_value = {
            "items": []
        }

        with pytest.raises(ValueError, match="Channel not found"):
            get_uploads_playlist_id(mock_youtube, "UC_nonexistent")

    def test_resolve_channel_by_handle_url(self):
        """Should resolve channel from URL with handle."""
        from src.fetch_channel_videos import resolve_channel

        mock_youtube = MagicMock()
        mock_youtube.channels.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "UCiKNX0TIMyKgGqQznhR1Xig",
                    "snippet": {"title": "LOL (Life of Limbachiyaas)"},
                    "contentDetails": {
                        "relatedPlaylists": {"uploads": "UUiKNX0TIMyKgGqQznhR1Xig"}
                    },
                }
            ]
        }

        channel_id, uploads_id, title = resolve_channel(
            mock_youtube, "https://www.youtube.com/@LifeOfLimbachiyaas"
        )
        assert channel_id == "UCiKNX0TIMyKgGqQznhR1Xig"
        assert uploads_id == "UUiKNX0TIMyKgGqQznhR1Xig"
        assert title == "LOL (Life of Limbachiyaas)"


# ── Video Limit Enforcement ──────────────────────────────────────────


class TestVideoLimitEnforcement:
    """Test that the limit parameter is respected."""

    def test_limit_respected(self):
        """Should not return more videos than requested."""
        mock_youtube = MagicMock()

        # Mock channels.list
        mock_youtube.channels.return_value.list.return_value.execute.return_value = (
            make_channels_response()
        )

        # Mock playlistItems.list with 10 videos
        videos = [{"video_id": f"vid_{i:03d}", "title": f"Video {i}"} for i in range(10)]
        mock_youtube.playlistItems.return_value.list.return_value.execute.return_value = (
            make_playlist_items_response(videos)
        )

        result = fetch_channel_videos(mock_youtube, "UC_creator_456", limit=5)
        assert len(result) == 5

    def test_fewer_videos_than_limit(self):
        """When channel has fewer videos than limit, return what's available."""
        mock_youtube = MagicMock()

        mock_youtube.channels.return_value.list.return_value.execute.return_value = (
            make_channels_response()
        )

        videos = [{"video_id": f"vid_{i:03d}", "title": f"Video {i}"} for i in range(3)]
        mock_youtube.playlistItems.return_value.list.return_value.execute.return_value = (
            make_playlist_items_response(videos)
        )

        result = fetch_channel_videos(mock_youtube, "UC_creator_456", limit=20)
        assert len(result) == 3

    def test_pagination_stops_at_limit(self):
        """Should stop paginating once limit is reached."""
        mock_youtube = MagicMock()

        mock_youtube.channels.return_value.list.return_value.execute.return_value = (
            make_channels_response()
        )

        # Page 1: 5 videos with next page
        page1_videos = [{"video_id": f"vid_p1_{i}", "title": f"P1 Video {i}"} for i in range(5)]
        # Page 2: 5 more videos
        page2_videos = [{"video_id": f"vid_p2_{i}", "title": f"P2 Video {i}"} for i in range(5)]

        mock_youtube.playlistItems.return_value.list.return_value.execute.side_effect = [
            make_playlist_items_response(page1_videos, next_page_token="page2"),
            make_playlist_items_response(page2_videos),
        ]

        # Request only 7 across both pages
        result = fetch_channel_videos(mock_youtube, "UC_creator_456", limit=7)
        assert len(result) == 7


# ── Video Data Extraction ───────────────────────────────────────────


class TestVideoDataExtraction:
    """Test that video metadata is correctly extracted."""

    def test_video_fields_extracted(self):
        """Should extract video_id, title, and published_at."""
        mock_youtube = MagicMock()
        mock_youtube.channels.return_value.list.return_value.execute.return_value = (
            make_channels_response()
        )

        videos = [{
            "video_id": "dQw4w9WgXcQ",
            "title": "Test Title",
            "published_at": "2026-03-15T10:00:00Z",
        }]
        mock_youtube.playlistItems.return_value.list.return_value.execute.return_value = (
            make_playlist_items_response(videos)
        )

        result = fetch_channel_videos(mock_youtube, "UC_creator_456", limit=5)
        assert len(result) == 1
        assert result[0]["video_id"] == "dQw4w9WgXcQ"
        assert result[0]["title"] == "Test Title"
        assert result[0]["published_at"] == "2026-03-15T10:00:00Z"
