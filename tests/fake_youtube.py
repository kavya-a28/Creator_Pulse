"""A stand-in for googleapiclient's YouTube client, for tests only.

Implements the call chain the collectors use (channels / playlistItems /
videos / commentThreads ... .list(**kw).execute()) over an in-memory table of
comments, and raises real googleapiclient HttpError objects for configured
failures. It never touches the network.
"""

import json
import time

import httplib2
import pandas as pd
from googleapiclient.errors import HttpError


def http_error(status: int, reason: str, message: str = "") -> HttpError:
    body = {"error": {"code": status, "message": message or reason,
                      "errors": [{"reason": reason, "message": message or reason}]}}
    return HttpError(httplib2.Response({"status": status}), json.dumps(body).encode("utf-8"))


class _Req:
    def __init__(self, fn, delay=0.0):
        self._fn, self._delay = fn, delay

    def execute(self):
        if self._delay:
            time.sleep(self._delay)
        return self._fn()


class _Resource:
    delay = 0.0

    def __init__(self, fn):
        self._fn = fn

    def list(self, **kw):
        return _Req(lambda: self._fn(**kw), _Resource.delay)


class FakeYouTube:
    """comments: DataFrame with comment_id, video_id, text, author_channel_id,
    published_at, like_count, reply_count, channel_id, channel_title,
    video_title, video_published_at.

    errors: {"video:<id>": HttpError, "channel": HttpError, "all": HttpError}
    handles: {"@name": channel_id}
    """

    def __init__(self, comments: pd.DataFrame, errors=None, handles=None, disabled=(), delay: float = 0.0,
                 pinned=()):
        self.pinned = set(pinned)     # comment ids YouTube shows first (pinned), whatever their date
        _Resource.delay = delay       # simulated network latency per API call
        self.c = comments.copy()
        self.errors = errors or {}
        self.handles = handles or {}
        self.disabled = set(disabled)
        self.calls = []

    # ------------------------------------------------------------------
    def _check(self, key):
        if "all" in self.errors:
            raise self.errors["all"]
        if key in self.errors:
            raise self.errors[key]

    def _channels(self):
        return (self.c.groupby("channel_id").agg(title=("channel_title", "first")).reset_index())

    def channels(self):
        def fn(part, id=None, forHandle=None, forUsername=None, **_):
            self.calls.append(("channels", id or forHandle or forUsername))
            self._check("channel")
            ch = self._channels()
            if forHandle:
                cid = self.handles.get(forHandle) or self.handles.get("@" + forHandle.lstrip("@"))
                ch = ch[ch["channel_id"] == cid]
            elif forUsername:
                ch = ch.iloc[0:0]
            else:
                ch = ch[ch["channel_id"] == id]
            return {"items": [{"id": r.channel_id, "snippet": {"title": r.title, "thumbnails": {
                                   "default": {"url": f"https://yt3.example.com/{r.channel_id}=s88"},
                                   "medium": {"url": f"https://yt3.example.com/{r.channel_id}=s240"}}},
                               "contentDetails": {"relatedPlaylists": {"uploads": "UU" + r.channel_id[2:]}}}
                              for r in ch.itertuples()]}
        return _Resource(fn)

    def _videos(self):
        v = (self.c.groupby("video_id").agg(title=("video_title", "first"), channel_id=("channel_id", "first"),
                                            channel_title=("channel_title", "first"),
                                            published=("video_published_at", "first"), n=("comment_id", "size"))
             .reset_index())
        return v.sort_values("published", ascending=False)

    def playlistItems(self):
        def fn(part, playlistId, maxResults=50, pageToken=None, **_):
            self.calls.append(("playlistItems", playlistId))
            cid = "UC" + playlistId[2:]
            v = self._videos()
            v = v[v["channel_id"] == cid]
            start = int(pageToken or 0)
            page = v.iloc[start:start + maxResults]
            out = {"items": [{"snippet": {"title": r.title, "resourceId": {"videoId": r.video_id}},
                              "contentDetails": {"videoId": r.video_id, "videoPublishedAt": r.published}}
                             for r in page.itertuples()]}
            if start + maxResults < len(v):
                out["nextPageToken"] = str(start + maxResults)
            return out
        return _Resource(fn)

    def videos(self):
        def fn(part, id, maxResults=50, **_):
            self.calls.append(("videos", id))
            self._check("videos")
            ids = id.split(",")
            v = self._videos()
            v = v[v["video_id"].isin(ids)]
            items = []
            for r in v.itertuples():
                stats = {} if r.video_id in self.disabled else {"commentCount": str(r.n)}
                items.append({"id": r.video_id,
                              "snippet": {"title": r.title, "channelId": r.channel_id,
                                          "channelTitle": r.channel_title, "publishedAt": r.published,
                                          "liveBroadcastContent": "none"},
                              "statistics": stats})
            return {"items": items}
        return _Resource(fn)

    def commentThreads(self):
        def fn(part, videoId, maxResults=100, pageToken=None, order=None, **_):
            self.calls.append(("commentThreads", videoId))
            self._check(f"video:{videoId}")
            if videoId in self.disabled:
                raise http_error(403, "commentsDisabled")
            rows = self.c[self.c["video_id"] == videoId]
            if order == "time":
                rows = rows.sort_values("published_at", ascending=False)
            if self.pinned:
                rows = pd.concat([rows[rows["comment_id"].isin(self.pinned)], rows[~rows["comment_id"].isin(self.pinned)]])
            start = int(pageToken or 0)
            page = rows.iloc[start:start + maxResults]
            items = [{"id": r.comment_id,
                      "snippet": {"totalReplyCount": int(r.reply_count or 0),
                                  "topLevelComment": {"id": r.comment_id, "snippet": {
                                      "textDisplay": r.text, "authorChannelId": {"value": r.author_channel_id},
                                      "publishedAt": r.published_at, "likeCount": int(r.like_count or 0)}}}}
                     for r in page.itertuples()]
            out = {"items": items}
            if start + maxResults < len(rows):
                out["nextPageToken"] = str(start + maxResults)
            return out
        return _Resource(fn)
