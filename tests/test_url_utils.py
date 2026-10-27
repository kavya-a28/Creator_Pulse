"""Tests for src.url_utils — YouTube URL parsing and video ID extraction."""

import pytest

from src.url_utils import extract_video_id


class TestValidURLs:
    """Test extraction from valid YouTube URL formats."""

    def test_bare_video_id(self):
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_video_id_with_hyphens_underscores(self):
        assert extract_video_id("a1B-c2D_e3F") == "a1B-c2D_e3F"

    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_watch_url_without_www(self):
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_watch_url_http(self):
        url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_mobile_url(self):
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120&list=PLtest"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url_with_params(self):
        url = "https://youtu.be/dQw4w9WgXcQ?t=30"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_whitespace(self):
        url = "  https://www.youtube.com/watch?v=dQw4w9WgXcQ  "
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestInvalidInputs:
    """Test that invalid inputs raise ValueError."""

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Empty input"):
            extract_video_id("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError, match="Empty input"):
            extract_video_id("   ")

    def test_non_youtube_url(self):
        with pytest.raises(ValueError, match="Not a YouTube URL"):
            extract_video_id("https://www.google.com/search?q=test")

    def test_random_text(self):
        with pytest.raises(ValueError, match="Invalid input"):
            extract_video_id("not a url or id")

    def test_too_short_id(self):
        with pytest.raises(ValueError, match="Invalid input"):
            extract_video_id("abc123")

    def test_too_long_id(self):
        with pytest.raises(ValueError, match="Invalid input"):
            extract_video_id("dQw4w9WgXcQ_extra")

    def test_watch_url_without_v_param(self):
        with pytest.raises(ValueError, match="Could not extract"):
            extract_video_id("https://www.youtube.com/watch?list=PLtest")

    def test_youtube_homepage(self):
        with pytest.raises(ValueError, match="Could not extract"):
            extract_video_id("https://www.youtube.com/")

    def test_youtube_channel_url(self):
        with pytest.raises(ValueError, match="Could not extract"):
            extract_video_id("https://www.youtube.com/channel/UCtest12345678")

    def test_none_input(self):
        with pytest.raises((ValueError, TypeError)):
            extract_video_id(None)


from src.url_utils import parse_channel_input


class TestChannelInputParsing:
    """Test channel input categorization and extraction."""

    def test_url_with_handle(self):
        id_type, val = parse_channel_input("https://www.youtube.com/@LifeOfLimbachiyaas")
        assert id_type == "handle"
        assert val == "@LifeOfLimbachiyaas"

    def test_url_with_handle_and_subpath(self):
        id_type, val = parse_channel_input("https://www.youtube.com/@LifeOfLimbachiyaas/videos")
        assert id_type == "handle"
        assert val == "@LifeOfLimbachiyaas"

    def test_url_with_channel_id(self):
        id_type, val = parse_channel_input("https://www.youtube.com/channel/UCiKNX0TIMyKgGqQznhR1Xig")
        assert id_type == "id"
        assert val == "UCiKNX0TIMyKgGqQznhR1Xig"

    def test_raw_handle_with_at(self):
        id_type, val = parse_channel_input("@LifeOfLimbachiyaas")
        assert id_type == "handle"
        assert val == "@LifeOfLimbachiyaas"

    def test_raw_channel_id(self):
        id_type, val = parse_channel_input("UCiKNX0TIMyKgGqQznhR1Xig")
        assert id_type == "id"
        assert val == "UCiKNX0TIMyKgGqQznhR1Xig"

    def test_empty_channel_input(self):
        with pytest.raises(ValueError, match="Empty input"):
            parse_channel_input("")
