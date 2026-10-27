"""Tests for src.collect: URL validation, collection with a fake API client, and error mapping."""

import pandas as pd
import pytest

from src import collect
from src.collect import CollectionError, collect as run_collect, parse_target, translate_http_error
from tests.fake_youtube import FakeYouTube, http_error

CH = "UC" + "a" * 22


def comments(n_per_video=5, videos=("vid00000001", "vid00000002"), channel=CH):
    rows = []
    for v_i, v in enumerate(videos):
        for i in range(n_per_video):
            rows.append({"comment_id": f"{v}-c{i}", "video_id": v, "text": f"comment {i} on {v} please make python",
                         "author_channel_id": f"a{i}", "published_at": f"2026-10-{10 + i:02d}T10:00:00Z",
                         "like_count": "0", "reply_count": "0", "channel_id": channel, "channel_title": "Chan",
                         "video_title": f"Video {v_i}", "video_published_at": f"2026-10-0{v_i + 1}T00:00:00Z"})
    return pd.DataFrame(rows)


# ---------------- URL validation ----------------

@pytest.mark.parametrize("url,kind,value", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "video", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ?t=30", "video", "dQw4w9WgXcQ"),
    ("youtube.com/shorts/dQw4w9WgXcQ", "video", "dQw4w9WgXcQ"),
    ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=PLx", "video", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/live/dQw4w9WgXcQ", "video", "dQw4w9WgXcQ"),
    ("dQw4w9WgXcQ", "video", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/@CodeWithHarry", "channel", None),
    ("https://www.youtube.com/@CodeWithHarry/videos", "channel", None),
    (f"https://www.youtube.com/channel/{CH}", "channel", None),
    ("@ApnaCollege", "channel", "@ApnaCollege"),
    (CH, "channel", CH),
])
def test_parse_target_valid(url, kind, value):
    t = parse_target(url)
    assert t.kind == kind
    if value:
        assert t.value == value


@pytest.mark.parametrize("url,code", [
    ("", "empty_url"), ("   ", "empty_url"),
    ("https://example.com/watch?v=dQw4w9WgXcQ", "not_youtube"),
    ("not a url at all", "not_youtube"),
    ("https://www.youtube.com/playlist?list=PL123", "unsupported_playlist"),
    ("https://www.youtube.com/results?search_query=python", "unsupported_page"),
    ("https://www.youtube.com/", "unsupported_page"),
    ("https://www.youtube.com/watch?v=short", "invalid_video_url"),
    ("https://www.youtube.com/channel/UCbad", "invalid_channel_url"),
])
def test_parse_target_rejects_with_reason(url, code):
    with pytest.raises(CollectionError) as e:
        parse_target(url)
    assert e.value.code == code and e.value.message


def test_playlist_part_of_video_link_is_noted():
    assert "Playlist" in parse_target("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1").note


# ---------------- API key ----------------

def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(collect, "api_key_configured", lambda: False)
    with pytest.raises(CollectionError) as e:
        collect.build_client()
    assert e.value.code == "missing_api_key" and ".env" in e.value.fix


def test_api_key_placeholder_counts_as_missing(monkeypatch):
    monkeypatch.setattr(collect, "PROJECT_ROOT", collect.PROJECT_ROOT / "does-not-exist")
    monkeypatch.setenv("YOUTUBE_API_KEY", "your_api_key_here")
    assert collect.api_key_configured() is False
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    assert collect.api_key_configured() is False


# ---------------- Collection ----------------

def test_collect_video_with_cap_and_capture_note():
    fy = FakeYouTube(comments(n_per_video=8))
    res = run_collect(parse_target("vid00000001"), max_comments=5, youtube=fy)
    assert len(res.comments) == 5
    assert list(res.comments.columns) == collect.RAW_COLUMNS
    assert set(res.comments["video_title"]) == {"Video 0"} and res.channel_title == "Chan"
    note = res.videos[0]
    assert note["truncated"] and note["newest_first_verified"] and note["capture"] == "newest_n"
    assert note["complete_since"].startswith("2026-10-13")      # oldest of the newest 5
    # commentThreads was called with order=time (newest first)
    assert ("commentThreads", "vid00000001") in fy.calls


def test_collect_complete_capture_when_under_cap():
    res = run_collect(parse_target("vid00000001"), max_comments=50, youtube=FakeYouTube(comments(4)))
    assert res.videos[0]["capture"] == "complete" and len(res.comments) == 4


def test_collect_channel_skips_disabled_video():
    fy = FakeYouTube(comments(3), handles={"@chan": CH}, disabled={"vid00000002"})
    res = run_collect(parse_target("https://www.youtube.com/@chan"), max_videos=5, max_comments=10, youtube=fy)
    assert set(res.comments["video_id"]) == {"vid00000001"}
    assert res.skipped_videos and "turned off" in res.skipped_videos[0]["reason"]


def test_collect_single_video_comments_disabled_is_an_error():
    fy = FakeYouTube(comments(3), disabled={"vid00000001"})
    with pytest.raises(CollectionError) as e:
        run_collect(parse_target("vid00000001"), youtube=fy)
    assert e.value.code == "comments_disabled"


def test_collect_unknown_video_and_channel():
    fy = FakeYouTube(comments(3))
    with pytest.raises(CollectionError) as e:
        run_collect(parse_target("zzzzzzzzzzz"), youtube=fy)
    assert e.value.code == "not_found"
    with pytest.raises(CollectionError) as e:
        run_collect(parse_target("https://www.youtube.com/@nobody"), youtube=fy)
    assert e.value.code == "not_found"


def test_collect_no_comments():
    # The video exists (and reports comments) but the API returns no comment threads.
    fy_empty = FakeYouTube(comments(1))
    fy_empty.commentThreads = FakeYouTube(comments(1).iloc[0:0]).commentThreads
    with pytest.raises(CollectionError) as e:
        run_collect(parse_target("vid00000001"), youtube=fy_empty)
    assert e.value.code == "no_comments"


def test_quota_mid_channel_keeps_collected_videos():
    fy = FakeYouTube(comments(3), handles={"@chan": CH},
                     errors={"video:vid00000001": http_error(403, "quotaExceeded")})
    # videos are listed newest first: vid00000002 (published 10-02) is collected before vid00000001
    res = run_collect(parse_target("@chan"), max_videos=5, max_comments=10, youtube=fy)
    assert set(res.comments["video_id"]) == {"vid00000002"}
    assert any("quota" in w.lower() for w in res.warnings)


def test_quota_on_first_call_is_an_error():
    fy = FakeYouTube(comments(3), errors={"all": http_error(403, "quotaExceeded")})
    with pytest.raises(CollectionError) as e:
        run_collect(parse_target("vid00000001"), youtube=fy)
    assert e.value.code == "quota_exceeded" and "midnight Pacific" in e.value.fix


@pytest.mark.parametrize("status,reason,code", [
    (400, "keyInvalid", "api_key_invalid"), (403, "accessNotConfigured", "api_forbidden"),
    (403, "quotaExceeded", "quota_exceeded"), (403, "commentsDisabled", "comments_disabled"),
    (404, "videoNotFound", "not_found"), (503, "backendError", "api_unavailable"),
])
def test_http_errors_are_translated(status, reason, code):
    err = translate_http_error(http_error(status, reason))
    assert err.code == code and err.message and err.fix


def test_api_key_never_appears_in_errors(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "AIzaSECRET123")
    err = translate_http_error(http_error(400, "keyInvalid", "API key not valid"))
    assert "AIzaSECRET123" not in " ".join(err.to_dict().values())


def test_quota_plan():
    assert collect.plan_quota("video", 5, 300) == 5          # +1: the channel's profile picture
    assert collect.plan_quota("channel", 5, 100) == 4 + 2 + 5


def test_channel_picture_is_collected_and_never_breaks_collection():
    import pandas as pd

    df = pd.DataFrame([{"comment_id": "c1", "video_id": "v1", "text": "hi", "author_channel_id": "a",
                        "published_at": "2026-01-01T00:00:00Z", "like_count": 0, "reply_count": 0,
                        "creator_replied": False, "channel_id": "UCx", "channel_title": "Chan",
                        "video_title": "T", "video_published_at": "2026-01-01T00:00:00Z"}])
    fake = FakeYouTube(df)
    assert collect.fetch_channel_thumbnail(fake, "UCx") == "https://yt3.example.com/UCx=s240"
    assert collect.fetch_channel_thumbnail(fake, "") == ""

    class Broken:
        def channels(self):
            raise RuntimeError("network down")
    assert collect.fetch_channel_thumbnail(Broken(), "UCx") == ""


# ---------------- API address / DNS fallback ----------------

def _fake_key(monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(collect, "api_key_configured", lambda: True)
    monkeypatch.setattr(cfg, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(collect, "system_proxy", lambda: "")


def test_client_uses_default_address_when_it_resolves(monkeypatch):
    _fake_key(monkeypatch)
    monkeypatch.setattr(collect, "host_resolves", lambda h: True)
    yt = collect.build_client()
    assert yt._baseUrl.startswith("https://youtube.googleapis.com/")


def test_client_falls_back_to_classic_address(monkeypatch):
    _fake_key(monkeypatch)
    monkeypatch.setattr(collect, "host_resolves", lambda h: h == "www.googleapis.com")
    yt = collect.build_client()
    assert yt._baseUrl.startswith("https://www.googleapis.com/")
    req = yt.videos().list(part="id", id="dQw4w9WgXcQ")
    assert req.uri.startswith("https://www.googleapis.com/youtube/v3/videos")


def test_no_dns_at_all_gives_actionable_error(monkeypatch):
    _fake_key(monkeypatch)
    monkeypatch.setattr(collect, "host_resolves", lambda h: False)
    with pytest.raises(CollectionError) as e:
        collect.build_client()
    assert e.value.code == "dns_failed" and "8.8.8.8" in e.value.fix


def test_server_not_found_is_reported_as_dns_problem():
    import httplib2

    def boom():
        raise httplib2.ServerNotFoundError("Unable to find the server at youtube.googleapis.com")
    with pytest.raises(CollectionError) as e:
        collect._call(boom)
    assert e.value.code == "dns_failed" and "check_connection" in e.value.fix


def test_proxy_is_used_when_configured(monkeypatch):
    _fake_key(monkeypatch)
    monkeypatch.setattr(collect, "host_resolves", lambda h: True)
    monkeypatch.setattr(collect, "system_proxy", lambda: "http://proxy.local:3128")
    yt = collect.build_client()
    assert yt._http.proxy_info.proxy_host == "proxy.local" and yt._http.proxy_info.proxy_port == 3128


def test_check_connection_reports_each_step(monkeypatch):
    from src import check_connection as cc
    monkeypatch.setattr(collect, "api_key_configured", lambda: True)
    monkeypatch.setattr(collect, "system_proxy", lambda: "")
    monkeypatch.setattr(cc, "_dns", lambda h: (h == "www.googleapis.com", "x"))
    monkeypatch.setattr(collect, "host_resolves", lambda h: h == "www.googleapis.com")
    monkeypatch.setattr(cc, "_https", lambda h: (True, "reachable"))
    monkeypatch.setattr(collect, "build_client", lambda: FakeYouTube(comments(1, videos=("dQw4w9WgXcQ",))))
    res = {r["check"]: r for r in cc.run_checks()}
    assert not res["Look up youtube.googleapis.com"]["ok"]
    assert res["Look up www.googleapis.com"]["ok"]
    assert res["HTTPS connection to www.googleapis.com"]["ok"]
    assert res["YouTube API call with your key"]["ok"] and "classic" in res["YouTube API call with your key"]["detail"]


def test_proxy_skips_local_dns(monkeypatch):
    monkeypatch.setattr(collect, "system_proxy", lambda: "http://proxy.local:3128")
    monkeypatch.setattr(collect, "host_resolves", lambda h: False)
    ep = collect.choose_endpoint()
    assert ep["host"] == "youtube.googleapis.com" and not ep.get("unresolved")


def test_pinned_comment_does_not_break_newest_first_check():
    df = comments(n_per_video=8)
    df = df[df.video_id == "vid00000001"].copy()
    old = df.iloc[0].copy()
    old["comment_id"], old["published_at"] = "pinned", "2025-01-01T00:00:00Z"
    df = pd.concat([df, old.to_frame().T], ignore_index=True)
    fy = FakeYouTube(df, pinned={"pinned"})
    res = run_collect(parse_target("vid00000001"), max_comments=5, youtube=fy)
    note = res.videos[0]
    assert res.comments["comment_id"].iloc[0] == "pinned"
    assert note["pinned_first"] and note["newest_first_verified"] and note["capture"] == "newest_n"
    assert not note["complete_since"].startswith("2025-01-01")
