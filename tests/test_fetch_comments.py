"""Tests for src.fetch_comments — Comment fetching, creator reply detection, CSV output."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.fetch_comments import (
    detect_creator_reply,
    fetch_comment_threads,
    threads_to_dataframe,
)


# ── Fixtures: realistic API response builders ────────────────────────


def make_thread(
    comment_id="Ugz_test_001",
    text="Great video!",
    author_channel_id="UC_viewer_123",
    video_id="dQw4w9WgXcQ",
    like_count=5,
    total_reply_count=0,
    replies=None,
    published_at="2026-01-15T10:00:00Z",
):
    """Build a realistic commentThread API item."""
    thread = {
        "id": comment_id,
        "snippet": {
            "channelId": "UC_creator_456",
            "videoId": video_id,
            "topLevelComment": {
                "id": comment_id,
                "snippet": {
                    "textDisplay": text,
                    "authorChannelId": {"value": author_channel_id},
                    "publishedAt": published_at,
                    "likeCount": like_count,
                    "videoId": video_id,
                    "channelId": "UC_creator_456",
                },
            },
            "totalReplyCount": total_reply_count,
            "isPublic": True,
        },
    }
    if replies is not None:
        thread["replies"] = {"comments": replies}
    return thread


def make_reply(author_channel_id="UC_viewer_789", text="Thanks!"):
    """Build a reply comment item."""
    return {
        "id": "Ugz_reply_001",
        "snippet": {
            "textDisplay": text,
            "authorChannelId": {"value": author_channel_id},
            "publishedAt": "2026-01-15T12:00:00Z",
            "likeCount": 1,
            "parentId": "Ugz_test_001",
        },
    }


# ── Creator Reply Detection ─────────────────────────────────────────


CREATOR_ID = "UC_creator_456"


class TestCreatorReplyDetection:
    """Test the three-state creator reply detection."""

    def test_no_replies_returns_false(self):
        """Comments with zero replies → False."""
        thread = make_thread(total_reply_count=0)
        assert detect_creator_reply(thread, CREATOR_ID) is False

    def test_creator_replied_returns_true(self):
        """Creator reply is present → True."""
        creator_reply = make_reply(author_channel_id=CREATOR_ID, text="Thanks!")
        thread = make_thread(
            total_reply_count=1,
            replies=[creator_reply],
        )
        assert detect_creator_reply(thread, CREATOR_ID) is True

    def test_other_viewer_replied_all_visible_returns_false(self):
        """Only non-creator replies, all replies visible → False."""
        viewer_reply = make_reply(author_channel_id="UC_other_viewer")
        thread = make_thread(
            total_reply_count=1,
            replies=[viewer_reply],
        )
        assert detect_creator_reply(thread, CREATOR_ID) is False

    def test_incomplete_replies_returns_none(self):
        """Replies exist but API returned subset → None (uncertain)."""
        viewer_reply = make_reply(author_channel_id="UC_other_viewer")
        thread = make_thread(
            total_reply_count=10,  # 10 total, but only 1 returned
            replies=[viewer_reply],
        )
        assert detect_creator_reply(thread, CREATOR_ID) is None

    def test_replies_claimed_but_empty_returns_none(self):
        """totalReplyCount > 0 but no replies in response → None."""
        thread = make_thread(total_reply_count=3)
        # No "replies" key at all
        assert detect_creator_reply(thread, CREATOR_ID) is None

    def test_creator_reply_among_multiple(self):
        """Creator reply mixed with viewer replies → True."""
        replies = [
            make_reply(author_channel_id="UC_viewer_A"),
            make_reply(author_channel_id=CREATOR_ID),
            make_reply(author_channel_id="UC_viewer_B"),
        ]
        thread = make_thread(total_reply_count=3, replies=replies)
        assert detect_creator_reply(thread, CREATOR_ID) is True


# ── DataFrame Conversion ────────────────────────────────────────────


class TestDataFrameConversion:
    """Test threads_to_dataframe output format and content."""

    def test_column_order(self):
        """CSV columns must be in exact specified order."""
        threads = [make_thread()]
        df = threads_to_dataframe(threads, "dQw4w9WgXcQ", CREATOR_ID)

        expected_columns = [
            "comment_id", "video_id", "text", "author_channel_id",
            "published_at", "like_count", "reply_count", "creator_replied",
        ]
        assert list(df.columns) == expected_columns

    def test_unicode_emoji_preservation(self):
        """Unicode, emojis, and Hinglish text must be preserved."""
        threads = [
            make_thread(
                comment_id="Ugz_unicode",
                text="यह बहुत अच्छा है! 🔥🎉 bahut mast video 💯",
            ),
        ]
        df = threads_to_dataframe(threads, "dQw4w9WgXcQ", CREATOR_ID)

        assert "🔥" in df.iloc[0]["text"]
        assert "🎉" in df.iloc[0]["text"]
        assert "यह बहुत अच्छा है" in df.iloc[0]["text"]
        assert "bahut mast video" in df.iloc[0]["text"]

    def test_missing_author_channel_id(self):
        """Handle comments with missing authorChannelId without crashing."""
        thread = make_thread()
        # Remove authorChannelId
        thread["snippet"]["topLevelComment"]["snippet"]["authorChannelId"] = {}
        df = threads_to_dataframe([thread], "dQw4w9WgXcQ", CREATOR_ID)

        assert df.iloc[0]["author_channel_id"] == ""

    def test_empty_threads_list(self):
        """Empty thread list produces empty DataFrame with correct columns."""
        df = threads_to_dataframe([], "dQw4w9WgXcQ", CREATOR_ID)

        assert len(df) == 0
        assert "comment_id" in df.columns
        assert "creator_replied" in df.columns


# ── Duplicate Prevention ─────────────────────────────────────────────


class TestDuplicatePrevention:
    """Test that duplicate comments are filtered during pagination."""

    def test_duplicate_threads_filtered(self):
        """Same comment ID across pages should only appear once."""
        # Simulate two pages returning overlapping comment IDs
        page1_response = {
            "items": [
                make_thread(comment_id="Ugz_001", text="First"),
                make_thread(comment_id="Ugz_002", text="Second"),
            ],
            "nextPageToken": "page2_token",
        }
        page2_response = {
            "items": [
                make_thread(comment_id="Ugz_002", text="Second (dup)"),
                make_thread(comment_id="Ugz_003", text="Third"),
            ],
        }

        mock_youtube = MagicMock()
        mock_list = MagicMock()
        mock_youtube.commentThreads.return_value.list.return_value = mock_list
        mock_list.execute.side_effect = [page1_response, page2_response]

        # Patch to use our mock
        with patch.object(
            mock_youtube.commentThreads(), "list", return_value=mock_list
        ):
            threads = fetch_comment_threads(mock_youtube, "dQw4w9WgXcQ")

        assert len(threads) == 3
        ids = [t["id"] for t in threads]
        assert ids == ["Ugz_001", "Ugz_002", "Ugz_003"]


# ── API Error Handling (Mocked) ──────────────────────────────────────


class TestAPIErrorHandling:
    """Test error handling with mocked API responses."""

    def _make_http_error(self, status_code, reason="error"):
        """Create a mock HttpError."""
        from googleapiclient.errors import HttpError

        resp = MagicMock()
        resp.status = status_code

        error_body = json.dumps({
            "error": {
                "errors": [{"reason": reason, "message": "Test error"}],
                "code": status_code,
                "message": "Test error",
            }
        }).encode("utf-8")

        return HttpError(resp, error_body)

    def test_comments_disabled_raises(self):
        """403 with commentsDisabled reason should raise."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads.return_value.list.return_value.execute.side_effect = (
            self._make_http_error(403, "commentsDisabled")
        )

        with pytest.raises(Exception):
            fetch_comment_threads(mock_youtube, "dQw4w9WgXcQ")

    def test_video_not_found_raises(self):
        """404 error should raise."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads.return_value.list.return_value.execute.side_effect = (
            self._make_http_error(404, "videoNotFound")
        )

        with pytest.raises(Exception):
            fetch_comment_threads(mock_youtube, "dQw4w9WgXcQ")

    def test_quota_exceeded_raises(self):
        """403 with quotaExceeded reason should raise."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads.return_value.list.return_value.execute.side_effect = (
            self._make_http_error(403, "quotaExceeded")
        )

        with pytest.raises(Exception):
            fetch_comment_threads(mock_youtube, "dQw4w9WgXcQ")
