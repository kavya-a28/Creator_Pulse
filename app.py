"""
CreatorPulse — Streamlit app: paste a YouTube video or channel URL, get audience insights.

    streamlit run app.py

The page only handles presentation. The work is done by:
    src.collect          YouTube collection (existing collectors + validation)
    src.analysis         the pipeline (preprocess, classify, topics, insights, recommend)
    src.job_runner       runs an analysis in a background thread so page reruns never repeat it
    src.runs             per-run folders: data/runs/<run_id>/ (+ raw CSV in data/raw/app_runs/)
    src.dashboard_data   loads a run's outputs for display
"""

import json
import re

import pandas as pd
import streamlit as st

from src import ui_components as ui
from src.analysis import AnalysisSettings, target_key
from src.collect import CollectionError, api_key_configured, parse_target, plan_quota
from src.config import DATA_DIR
from src.dashboard_data import (ACTIONABLE, CLASSES, category_counts, export_comments, filter_comments,
                                load_run_data, topic_comments)
from src.job_runner import BusyError, JobRunner, effective_status
from src.runs import STAGES, Run, find_recent, list_runs

st.set_page_config(page_title="CreatorPulse", page_icon="▶️", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.CSS + ui.DASH_CSS + ui.MIX_CSS + ui.OVERVIEW_CSS + ui.MIXROW_CSS + ui.GRID_CSS + ui.RECS_CSS + ui.MOTION_CSS, unsafe_allow_html=True)

COMMENT_CHOICES = [50, 100, 200, 300, 500, 1000, 2000]
PREFS_PATH = DATA_DIR / "app_settings.json"
DEFAULT_PREFS = {"max_comments": 300, "exclude_spam": True}
VIDEOS_PER_RUN = 1          # one video per analysis (a channel link uses its latest upload)

PAGES = [  # key, label, icon
    ("new", "New Analysis", ":material/edit_square:"),
    ("overview", "Overview", ":material/space_dashboard:"),
    ("comments", "Comments", ":material/chat_bubble:"),
    ("insights", "Insights", ":material/bar_chart:"),
    ("recs", "Recommendations", ":material/lightbulb:"),
    ("runs", "Past Runs", ":material/history:"),
]


# --------------------------------------------------------------------------
# Shared resources, caches, state
# --------------------------------------------------------------------------

@st.cache_resource
def get_runner() -> JobRunner:
    return JobRunner()


@st.cache_data(show_spinner=False)
def cached_run_data(run_id: str, updated_at: str):
    """Keyed on updated_at, so a run's cache entry is replaced whenever its metadata changes."""
    return load_run_data(run_id)


@st.cache_data(ttl=3600, show_spinner=False)
def channel_picture(channel_id: str) -> str:
    """Profile picture for analyses made before pictures were collected (1 API unit, once per channel)."""
    try:
        from src.collect import build_client, fetch_channel_thumbnail
        return fetch_channel_thumbnail(build_client(), channel_id)
    except Exception:  # noqa: BLE001
        return ""


def run_avatar(run, status: str) -> str:
    coll = run.meta.get("collection") or {}
    pic = coll.get("channel_thumbnail")
    if pic is None and coll.get("channel_id") and status != "running" and api_key_configured():
        pic = channel_picture(coll["channel_id"])
        if pic:                                    # remember it, so it is fetched only once
            try:
                run.meta["collection"]["channel_thumbnail"] = pic
                run.save()
            except Exception:  # noqa: BLE001
                pass
    return pic or ""


@st.cache_data(ttl=5, show_spinner=False)
def cached_runs():
    return list_runs()


def load_prefs() -> dict:
    try:
        return {**DEFAULT_PREFS, **json.loads(PREFS_PATH.read_text(encoding="utf-8"))}
    except Exception:
        return dict(DEFAULT_PREFS)


runner = get_runner()
ss = st.session_state
ss.setdefault("page", "new")
ss.setdefault("run_id", None)          # run shown on the dashboard pages
ss.setdefault("active_run", None)      # run started from New Analysis (progress shown there)
ss.setdefault("form_error", None)
ss.setdefault("duplicate_of", None)
ss.setdefault("focus_rec", None)
# Comments Explorer filters. Kept here (not only in the widgets) so they survive
# leaving the page: Streamlit forgets a widget's value once it isn't drawn.
NO_FILTERS = {"cats": [], "topic": None, "q": "", "dates": None}
FILTER_WIDGETS = ("w_cats", "w_topic", "w_q", "w_dates")
ss.setdefault("cx", dict(NO_FILTERS))
ss.setdefault("cx_run", None)
if "max_comments" not in ss:
    _p = load_prefs()
    ss.max_comments, ss.exclude_spam = _p["max_comments"], _p["exclude_spam"]


def go(page: str):
    ss.page, ss.focus_rec = page, None


def set_filters(**f):
    """Replace the Comments Explorer filters (the widgets re-read them on their next draw)."""
    ss.cx = {**NO_FILTERS, **f}
    for k in FILTER_WIDGETS:
        ss.pop(k, None)


def open_comments(cats=None, topic=None):
    """Navigate to the Comments Explorer showing only these comments."""
    set_filters(cats=list(cats or []), topic=topic)
    ss.cx_run = ss.run_id
    ss.page, ss.focus_rec = "comments", None


def open_run(run_id: str, page: str = "overview"):
    ss.run_id, ss.page = run_id, page


def run_title(m: dict) -> str:
    return (m.get("collection") or {}).get("channel_title") or m.get("source_url", "Analysis")


def default_run_id():
    """Most recent run that produced comments to show."""
    for m in cached_runs():
        if (m.get("totals") or {}).get("comments_analyzed") or "comments_predicted" in (m.get("outputs") or {}):
            return m["run_id"]
    return None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def sidebar():
    with st.sidebar:
        st.markdown(ui.brand_html(), unsafe_allow_html=True)
        for key, label, icon in PAGES:
            st.button(label, key=f"nav_{key}", icon=icon, on_click=go, args=(key,), use_container_width=True,
                      type="primary" if ss.page == key else "secondary")
        st.markdown(ui.SIDE_NOTE, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# New Analysis
# --------------------------------------------------------------------------

def current_settings() -> AnalysisSettings:
    return AnalysisSettings(max_videos=VIDEOS_PER_RUN, max_comments=int(ss.max_comments),
                            exclude_spam=bool(ss.exclude_spam))


def submit():
    """Analyze button: validate quickly, then start the background analysis exactly once."""
    ss.form_error, ss.duplicate_of = None, None
    url = (ss.get("url") or "").strip()
    settings = current_settings()
    try:
        target = parse_target(url)
    except CollectionError as err:
        ss.form_error = err.to_dict()
        return
    if not api_key_configured():
        ss.form_error = {"message": "No YouTube API key is configured.",
                         "fix": "Put YOUTUBE_API_KEY=<your key> in the .env "
                                "file in the project folder, then restart the app."}
        return
    if not ss.pop("force_new", False):
        recent = find_recent(target_key(url), settings.as_dict())
        if recent:
            ss.duplicate_of = recent["run_id"]
            return
    try:
        run_id = runner.start(url, settings)
    except BusyError as exc:
        ss.form_error = {"message": str(exc), "fix": ""}
        return
    cached_runs.clear()
    ss.active_run = run_id


def clear_messages():
    ss.form_error, ss.duplicate_of = None, None


def force_submit():
    ss.force_new = True
    submit()


def new_analysis_page():
    st.markdown(ui.header_html("Analyze a YouTube Video or Channel",
                               "Paste a YouTube video URL or channel URL to collect comments and get "
                               "AI-powered insights."), unsafe_allow_html=True)
    busy = bool(runner.active())
    with st.container(key="url_card"):
        c0, c1, c2 = st.columns([0.075, 0.705, 0.22], vertical_alignment="center", gap="small")
        c0.markdown(f'<div class="cp-vtile">{ui.VIDEO_TILE}</div>', unsafe_allow_html=True)
        c1.text_input("YouTube video or channel URL", key="url", label_visibility="collapsed",
                      placeholder="https://www.youtube.com/watch?v=example", on_change=clear_messages)
        c2.button("Analyze", key="analyze", icon=":material/auto_awesome:", type="primary",
                  use_container_width=True, on_click=submit, disabled=busy)
        d1, d2 = st.columns([0.74, 0.26], vertical_alignment="center")
        d1.markdown(ui.support_html(), unsafe_allow_html=True)
        with d2:
            with st.popover("Advanced Options", icon=":material/settings:", use_container_width=True):
                advanced_options()
        if ss.form_error:
            st.error(f"**{ss.form_error['message']}**  \n{ss.form_error.get('fix', '')}", icon=":material/error:")
        if ss.duplicate_of:
            duplicate_notice()

    rid = ss.active_run
    if rid is None and busy:
        rid = next(iter(runner.active()))
    if rid:
        try:
            m = Run.load(rid).meta
        except Exception:
            ss.active_run, m = None, None
        if m is not None:
            status = effective_status(m, runner.active())
            if status == "running":
                live_progress(rid)
                return
            if status.startswith("completed") and ss.active_run == rid:
                ss.active_run, ss.celebrate = None, rid   # finished while the page wasn't polling: show results
                open_run(rid)
                st.rerun()
            finished_progress(m)
            return
    st.markdown(ui.stepper_html(), unsafe_allow_html=True)
    st.markdown(ui.empty_state_html(), unsafe_allow_html=True)


def advanced_options():
    st.select_slider("Most recent comments to collect", options=COMMENT_CHOICES, key="max_comments",
                     help="Comments are taken newest first, up to this number. Replies to comments are not "
                          "collected. If the video has fewer comments, all of them are used.")
    st.toggle("Exclude spam from topic discovery", key="exclude_spam",
              help="Spam is still classified and counted; it just isn't grouped into topics or recommended.")
    kind = "video"
    try:
        kind = parse_target(ss.get("url", "")).kind if ss.get("url") else "video"
    except CollectionError:
        pass
    units = plan_quota(kind, VIDEOS_PER_RUN, int(ss.max_comments))
    st.caption(f"One video per analysis; a channel link analyzes the channel's latest upload. Uses up to about "
               f"{units} of your 10,000 daily YouTube API units.")


def duplicate_notice():
    try:
        prev = Run.load(ss.duplicate_of).meta
    except Exception:
        ss.duplicate_of = None
        return
    st.warning(f"This link was analyzed with the same settings at {ui.fmt_time(prev['created_at'])}. "
               "Open that result, or collect the comments again.", icon=":material/history:")
    a, b, _ = st.columns([1.5, 1.3, 2])
    a.button("Open previous result", key="open_dup", on_click=open_run, args=(ss.duplicate_of,))
    b.button("Analyze again", key="force_dup", on_click=force_submit)


def _progress_body(m: dict):
    """Live progress read from the run's own stage records (updated by the pipeline as it works)."""
    st.markdown(ui.stepper_html(ui.step_states(m["stages"])), unsafe_allow_html=True)
    st.markdown(ui.progress_card_html(m), unsafe_allow_html=True)


@st.fragment(run_every=1.0)
def live_progress(run_id: str):
    m = Run.load(run_id).meta
    _progress_body(m)
    status = effective_status(m, runner.active())
    if status != "running":
        cached_runs.clear()
        if status.startswith("completed"):
            ss.active_run, ss.celebrate = None, run_id
            open_run(run_id)
        st.rerun()


def finished_progress(m: dict):
    status = effective_status(m, runner.active())
    st.markdown(ui.stepper_html(ui.step_states(m["stages"])), unsafe_allow_html=True)
    if status == "interrupted":
        st.error("This analysis was interrupted (the app was closed or restarted while it ran). Click Analyze "
                 "to start it again.", icon=":material/error:")
    elif status == "failed":
        show_error(m)
    else:
        st.success("Analysis finished.", icon=":material/check_circle:")
    a, b, _ = st.columns([1.3, 1.1, 3])
    if "comments_predicted" in (m.get("outputs") or {}):
        a.button("Open results", key="open_finished", icon=":material/space_dashboard:", on_click=open_run,
                 args=(m["run_id"],))
    b.button("Clear", key="clear_finished", on_click=lambda: ss.update(active_run=None, form_error=None))


def show_error(m: dict):
    err = m.get("error") or {}
    label = dict(STAGES).get(m.get("failed_stage"), "The analysis")
    st.error(f"**{label} failed. {err.get('message', '')}**  \n{err.get('fix', '')}", icon=":material/error:")
    if err.get("detail"):
        with st.expander("Technical details"):
            st.code(err["detail"], language=None)


# --------------------------------------------------------------------------
# Past Runs
# --------------------------------------------------------------------------

def runs_table(runs, prefix):
    active = runner.active()
    for m in runs:
        with st.container(border=True):
            a, b, c = st.columns([3.2, 1.6, 1], vertical_alignment="center")
            t = m.get("totals") or {}
            status = effective_status(m, active)
            a.markdown(f"**{ui.esc(run_title(m))}**  \n<span class='cp-muted'>{ui.esc(m.get('source_url', ''))}</span>",
                       unsafe_allow_html=True)
            facts = ", ".join(x for x in [
                f"{t['comments_analyzed']} comments" if t.get("comments_analyzed") else "",
                f"{t['topics']} topics" if t.get("topics") is not None else "",
                f"{t['recommendations']} ideas" if t.get("recommendations") is not None else ""] if x)
            b.markdown(f"{ui.status_html(status)}<br><span class='cp-muted'>{ui.esc(ui.fmt_time(m.get('created_at')))}"
                       f"</span><br><span class='cp-muted'>{ui.esc(facts)}</span>", unsafe_allow_html=True)
            has_data = "comments_predicted" in (m.get("outputs") or {})
            if status == "running":
                c.button("Progress", key=f"{prefix}_prog_{m['run_id']}", on_click=lambda r=m["run_id"]:
                         ss.update(active_run=r, page="new"))
            else:
                c.button("Open", key=f"{prefix}_open_{m['run_id']}", on_click=open_run, args=(m["run_id"],),
                         disabled=not has_data and status != "failed")


def runs_page():
    st.markdown(ui.header_html("Past Runs", "Every analysis is saved, so you can reopen it at any time."),
                unsafe_allow_html=True)
    st.write("")
    runs = cached_runs()
    if not runs:
        st.info("No analyses yet. Paste a link on New Analysis to start.", icon=":material/info:")
        return
    runs_table(runs, "runs")


# --------------------------------------------------------------------------
# Dashboard pages
# --------------------------------------------------------------------------

DASH_SUBS = {"overview": "Key insights from your YouTube audience",
             "comments": "Every analyzed comment, with its predicted category and topic",
             "topics": "What your viewers keep asking about", "insights": "Questions, requests, problems and trends",
             "recs": "Video ideas backed by your viewers' comments"}
DASH_TITLES = {"overview": "Overview", "comments": "Comments Explorer", "topics": "Topics", "insights": "Insights",
               "recs": "Recommendations"}


def dashboard_page(page: str):
    rid = ss.run_id or default_run_id()
    if not rid:
        st.markdown(ui.header_html(DASH_TITLES[page]), unsafe_allow_html=True)
        st.info("Nothing to show yet. Analyze a video or channel first.", icon=":material/info:")
        st.button("New analysis", key="dash_new", icon=":material/auto_awesome:", type="primary", on_click=go,
                  args=("new",))
        return
    ss.run_id = rid
    try:
        run = Run.load(rid)
    except Exception:
        st.markdown(ui.header_html(DASH_TITLES[page]), unsafe_allow_html=True)
        st.error("This analysis could not be opened. Its folder may have been moved or deleted.")
        ss.run_id = None
        return
    m = run.meta
    status = effective_status(m, runner.active())
    st.markdown(ui.hero_html(DASH_TITLES[page], DASH_SUBS[page]), unsafe_allow_html=True)
    st.markdown(ui.metabar_html(status, run_title(m), m["source_url"], ui.fmt_time(m.get("created_at")),
                                run_avatar(run, status)), unsafe_allow_html=True)
    if status == "failed":
        show_error(m)
    if page == "overview":
        for w in m.get("warnings", []):
            st.warning(w, icon=":material/warning:")
    d = cached_run_data(rid, m.get("updated_at", ""))
    if d.comments is None:
        st.markdown(ui.stages_html(m["stages"]), unsafe_allow_html=True)
        return
    if page == "overview":
        if ss.get("celebrate") == rid:                   # first view after the analysis finished
            ss.celebrate = None
            n_top = int((d.topics["topic_id"] != -1).sum()) if d.topics is not None else 0
            n_rec = len((d.recs or {}).get("recommendations", []))
            st.markdown(ui.success_html(len(d.comments), n_top, n_rec), unsafe_allow_html=True)
            st.toast(f"Analysis complete: {len(d.comments)} comments analyzed", icon=":material/check_circle:")
        overview(d, run)
    else:
        st.write("")
        {"comments": lambda: comments_tab(d, rid), "topics": lambda: topics_tab(d),
         "insights": lambda: insights_tab(d), "recs": lambda: recs_tab(d, rid)}[page]()


def open_rec(rank: int):
    ss.focus_rec, ss.open_rec, ss.open_rec_run, ss.page = rank, rank, ss.run_id, "recs"


def toggle_rec(rank: int):
    ss.open_rec = None if ss.get("open_rec") == rank else rank


def click_row(row_id: str, html_row: str, label: str, cb, args=()):
    """A whole row that is one button (see ui_components: row_* / go_*)."""
    with st.container(key=f"row_{row_id}"):
        st.markdown(html_row, unsafe_allow_html=True)
        st.button(label, key=f"go_{row_id}", on_click=cb, args=args)


def card_head(ic, title, sub, link=None):
    """Card heading in the reference style; `link` = (label, key, callback, args) adds a "View all" link."""
    if not link:
        st.markdown(ui.head3_html(ic, title, sub), unsafe_allow_html=True)
        return
    st.markdown(ui.head3_html(ic, title, sub), unsafe_allow_html=True)
    label, key, cb, args = link
    st.button(label, key=f"hlink_{key}", icon=":material/chevron_right:", icon_position="right", on_click=cb,
             args=args)


def overview(d, run):
    real = (d.topics[d.topics["topic_id"] != -1].sort_values(["n_unique_comments", "topic_id"], ascending=[False, True])
            if d.topics is not None else None)
    recs = (d.recs or {}).get("recommendations", [])
    st.markdown(ui.kpi_tiles_html([
        ("chat", len(d.comments), "Comments analyzed"),
        ("play", d.comments["video_id"].nunique(), "Videos analyzed"),
        ("layers", len(real) if real is not None else "–", "Topics discovered"),
        ("bulb", len(recs) if d.recs is not None else "–", "Content recommendations"),
    ]), unsafe_allow_html=True)

    with st.container(key="grid1"):
        left, right = st.columns(2, gap="medium")
        with left, st.container(key="card_cats"):
            card_head("chat", "Comment categories", "Click a category to view matching comments →")
            cc = category_counts(d.comments)
            top = float(cc["comments"].max()) if len(cc) else 0
            for r in cc.itertuples():
                click_row(f"cat_{ui.slug(r.category)}", ui.cat_row3_html(r.category, r.comments, r.percent, top),
                          f"Show {r.category} comments", open_comments, ([r.category],))
        with right, st.container(key="card_topics"):
            card_head("layers", "Top audience topics", "Topics with the highest number of unique comments",
                      ("View all", "topics", go, ("topics",)) if real is not None and not real.empty else None)
            if real is None:
                st.caption("Topic discovery did not run.")
            elif real.empty:
                st.caption("No group of similar comments was found yet.")
            else:
                mx = int(real["n_unique_comments"].max())
                for i, r in enumerate(real.head(6).itertuples()):
                    click_row(f"topic_{i}", ui.topic_row3_html(i, r.topic_name, r.n_unique_comments, mx),
                              f"Show comments about {r.topic_name}", open_comments, (None, r.topic_name))

    with st.container(key="grid2"):
        left, right = st.columns(2, gap="medium")
        with left, st.container(key="card_recent"):
            card_head("chat", "Recent comments", "Latest comments from this analysis",
                      ("View all comments", "comments", open_comments, ()))
            recent = d.comments.sort_values("published_dt", ascending=False, na_position="last").head(6)
            st.markdown(ui.recent_table_html(recent), unsafe_allow_html=True)
        with right, st.container(key="card_recs"):
            card_head("bulb", "Top content opportunities", "Video ideas based on audience questions and problems",
                      ("View all", "recs", go, ("recs",)) if recs else None)
            if not recs:
                st.caption(no_recs_reason(d))
            for i, r in enumerate(recs[:5]):
                click_row(f"rec_{i}", ui.opp_row3_html(i, r), f"Open recommendation {r['rank']}", open_rec,
                          (r["rank"],))


def no_recs_reason(d) -> str:
    if d.recs is None:
        return "Recommendations were not generated in this run."
    rej = d.recs.get("topics_not_recommended", [])
    if not rej:
        return ("No recurring topic was found, so there is nothing to recommend yet. More comments "
                "(or more videos from the channel) give the model more to work with.")
    return (f"None of the {len(rej)} topic(s) were raised by enough different people yet (at least 3 are needed). "
            "Try more comments per video or more videos.")


NOT_GROUPED = "Not grouped into a topic"


def _sync_filter(name, widget_key):
    ss.cx = {**ss.cx, name: ss.get(widget_key)}


def remove_filter(name, value=None):
    f = dict(ss.cx)
    if name == "cats":
        f["cats"] = [c for c in f["cats"] if c != value]
    else:
        f[name] = NO_FILTERS[name]
    set_filters(**f)


def comments_tab(d, run_id):
    df = d.comments
    if ss.cx_run != run_id:                          # another analysis: start unfiltered
        set_filters()
        ss.cx_run = run_id
    f = ss.cx
    topics = sorted(t for t in df["topic_name"].unique() if t and t != "Unclustered")
    topic_opts = ["All topics"] + topics + [NOT_GROUPED]
    days = df["published_dt"].dropna().dt.date
    dmin, dmax = (days.min(), days.max()) if len(days) else (None, None)
    # Seed the widgets from the saved filters (after navigation they start empty).
    if f["topic"] not in topic_opts:
        f["topic"] = None
    ss.setdefault("w_cats", [c for c in f["cats"] if c in CLASSES])
    ss.setdefault("w_topic", f["topic"] or "All topics")
    ss.setdefault("w_q", f["q"])
    if dmin is not None:
        ss.setdefault("w_dates", tuple(f["dates"]) if f["dates"] else (dmin, dmax))

    with st.container(key="card_comments"):
        c1, c2, c3, c4 = st.columns([1.3, 1.25, 1.45, 1.2], vertical_alignment="bottom")
        c1.multiselect("Category", CLASSES, key="w_cats", placeholder="All categories",
                       on_change=_sync_filter, args=("cats", "w_cats"))
        c2.selectbox("Topic", topic_opts, key="w_topic",
                     on_change=lambda: ss.update(cx={**ss.cx, "topic": None if ss.w_topic == "All topics"
                                                     else ss.w_topic}))
        if dmin is not None:
            c3.date_input("Posted between (UTC)", key="w_dates", min_value=dmin, max_value=dmax, format="DD/MM/YYYY",
                          on_change=lambda: ss.update(cx={**ss.cx, "dates": _date_range(ss.w_dates, dmin, dmax)}))
        c4.text_input("Search", key="w_q", placeholder="Search comment text", on_change=_sync_filter,
                      args=("q", "w_q"))
        f = ss.cx

        out = filter_comments(df, f["cats"], None if f["topic"] in (None, NOT_GROUPED) else f["topic"], None, f["q"])
        if f["topic"] == NOT_GROUPED:
            out = out[out["topic_name"].isin(["", "Unclustered"])]
        if f["dates"]:
            day = out["published_dt"].dt.date
            out = out[(day >= f["dates"][0]) & (day <= f["dates"][1])]

        chips = ([(f"Category: {c}", "cats", c) for c in f["cats"]]
                 + ([(f"Topic: {f['topic']}", "topic", None)] if f["topic"] else [])
                 + ([(f"Posted {f['dates'][0]:%d %b} – {f['dates'][1]:%d %b %Y}", "dates", None)] if f["dates"] else [])
                 + ([(f'Search: "{f["q"]}"', "q", None)] if f["q"] else []))
        a, b = st.columns([5, 1.1], vertical_alignment="center")
        a.markdown(f'<div class="cp-count"><b>{len(out)}</b> matching comment{"s" if len(out) != 1 else ""} '
                   f'<span class="cp-muted">of {len(df)}</span></div>', unsafe_allow_html=True)
        b.button("Clear filters", key="clear_filters", icon=":material/filter_alt_off:", on_click=set_filters,
                 disabled=not chips, use_container_width=True)
        if chips:
            with st.container(key="chips"):
                for i, (label, name, value) in enumerate(chips):
                    st.button(label, key=f"chip_{i}", icon=":material/close:", icon_position="right",
                              on_click=remove_filter, args=(name, value), help="Remove this filter")
        view = out.sort_values("published_dt", ascending=False, na_position="last").assign(
            when=out["published_dt"].dt.tz_convert(None),
            topic=out["topic_name"].replace({"Unclustered": ""}))
        st.dataframe(view[["text", "predicted_category", "topic", "video_title", "when"]], use_container_width=True,
                     hide_index=True, height=min(420, 38 + 35 * max(len(view), 1)), key="comments_table", column_config={
                         "text": st.column_config.TextColumn("Comment", width="large"),
                         "predicted_category": st.column_config.TextColumn("Category", width=125),
                         "topic": st.column_config.TextColumn(
                             "Topic", width=140, help="Blank: not grouped (only questions, requests and "
                                                           "problems are grouped into topics)"),
                         "video_title": st.column_config.TextColumn("Video", width=170),
                         "when": st.column_config.DatetimeColumn("Posted (UTC)", format="D MMM YYYY, HH:mm"),
                     })
        a, b, _ = st.columns([1.5, 1.4, 1.3])
        a.download_button("Download these comments (CSV)", export_comments(out), file_name=f"comments_{run_id}.csv",
                          mime="text/csv", icon=":material/download:", key=f"dl_f_{run_id}")
        b.download_button("Download all comments (CSV)", export_comments(df), file_name=f"comments_all_{run_id}.csv",
                          mime="text/csv", icon=":material/download:", key=f"dl_a_{run_id}")


def _date_range(v, dmin, dmax):
    """date_input value -> (start, end), or None when it is the whole period or only half chosen."""
    if not isinstance(v, (tuple, list)) or len(v) != 2:
        return ss.cx.get("dates")
    if tuple(v) == (dmin, dmax):
        return None
    return tuple(v)


def topics_tab(d):
    if d.topics is None:
        st.caption("Topic discovery did not run for this analysis.")
        return
    tp = d.topics
    real = tp[tp["topic_id"] != -1].sort_values(["n_unique_comments", "topic_id"], ascending=[False, True])
    unc = tp[tp["topic_id"] == -1]
    n_unc = int(unc["n_comments_raw"].iloc[0]) if len(unc) else 0
    with st.container(key="card_topics_mix"):
        st.markdown(ui.card_head_html("layers", f"{len(real)} topic(s) found",
                                      f"From the questions, requests and problems; click a topic to see its comments. "
                                      f"{n_unc} comment(s) didn't match others closely enough and are not grouped."),
                    unsafe_allow_html=True)
        if real.empty:
            st.caption("No group of at least two similar comments yet.")
            return
        st.markdown(ui.mix_legend_html(), unsafe_allow_html=True)
        mx = int((real["n_question"] + real["n_request"] + real["n_confusion_problem"]).max())
        for i, r in enumerate(real.head(12).itertuples()):
            click_row(f"tmix_{i}", ui.mix_row_html(r, mx), f"Show comments about {r.topic_name}", open_comments,
                      (None, r.topic_name))
    with st.container(key="card_topics_table"):
        table = real[["topic_name", "n_unique_comments", "n_comments_raw", "n_unique_commenters", "n_videos",
                      "n_question", "n_request", "n_confusion_problem", "representative_comment"]]
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "topic_name": "Topic", "n_unique_comments": "Unique comments", "n_comments_raw": "All comments",
            "n_unique_commenters": "People", "n_videos": "Videos", "n_question": "Questions",
            "n_request": "Requests", "n_confusion_problem": "Problems",
            "representative_comment": st.column_config.TextColumn("Representative comment", width="large")})
    with st.container(key="card_topics_inspect"):
        st.markdown(ui.card_head_html("help", "Inspect a topic", "Every comment in the topic, most typical first"),
                    unsafe_allow_html=True)
        names = dict(zip(real["topic_name"], real["topic_id"]))
        choice = st.selectbox("Topic", list(names) + (["Unclustered"] if n_unc else []), key="topic_pick")
        tid = names.get(choice, -1)
        row = tp[tp["topic_id"] == tid].iloc[0]
        if tid != -1:
            st.markdown(ui.quote_html(row["representative_comment"], "Representative comment"), unsafe_allow_html=True)
            st.caption(f"Appears in {row['n_videos']} video(s): {row['video_titles']}")
            if row.get("note"):
                st.caption(row["note"])
        if tid != -1:
            st.button(f"Open these comments in Comments Explorer", key="topic_to_comments",
                      icon=":material/chat_bubble:", on_click=open_comments, args=(None, choice))
        g = topic_comments(d, tid)
        st.dataframe(g[["text", "predicted_category", "video_title", "near_dup_group_size", "similarity_to_topic"]],
                     hide_index=True, use_container_width=True, column_config={
                         "text": st.column_config.TextColumn("Comment", width="large"),
                         "predicted_category": "Category", "video_title": "Video",
                         "near_dup_group_size": st.column_config.NumberColumn(
                             "Copies", help="How many near-identical copies of this comment exist"),
                         "similarity_to_topic": st.column_config.NumberColumn("Fit", format="%.2f",
                                                                              help="Similarity to the topic")})


def _topic_list(items, empty):
    if not items:
        st.caption(empty)
    for t in items:
        st.markdown(f"<b>{ui.fmt_name(t['topic_name'])}</b>: {t.get('n_unique_with_this_intent', t['n_unique_comments'])} "
                    f"unique comment(s), {t['n_videos']} video(s)", unsafe_allow_html=True)
        st.markdown(ui.quote_html(t.get("example_comment") or t["representative_comment"]), unsafe_allow_html=True)


def insights_tab(d):
    ins = d.insights
    if ins is None:
        st.caption("Insights were not calculated for this analysis.")
        return
    mc = ins["most_common"]
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, key, ic, title, empty in [(c1, "Question", "help", "Most common questions", "No recurring questions yet."),
                                       (c2, "Request", "request", "Most frequent requests", "No recurring requests yet."),
                                       (c3, "Confusion/Problem", "alert", "Recurring problems",
                                        "No recurring problems yet.")]:
        with col, st.container(key=f"card_ins_{ic}"):
            st.markdown(ui.card_head_html(ic, title, "Grouped by topic, unique comments"), unsafe_allow_html=True)
            _topic_list(mc[key]["top_topics"][:4], empty)

    a, b = st.columns(2, gap="medium")
    with a, st.container(key="card_ins_tutorial"):
        st.markdown(ui.card_head_html("star", "May deserve a dedicated tutorial",
                                      "Questions or problems raised by 3+ different people"), unsafe_allow_html=True)
        ded = ins["repeated_questions_needing_a_dedicated_video"]
        if ded:
            for t in ded:
                st.markdown(f"<b>{ui.fmt_name(t['topic_name'])}</b>: {ui.esc(t['reason'])}", unsafe_allow_html=True)
        else:
            st.caption("No question or problem was raised by 3 or more different people yet.")
    with b, st.container(key="card_ins_repeat"):
        st.markdown(ui.card_head_html("repeat", "Posted more than once", "The same comment, copied or repeated"),
                    unsafe_allow_html=True)
        reps = [(k, r) for k in ACTIONABLE for r in mc[k]["most_repeated_comments"]]
        if not reps:
            st.caption("No question, request or problem was posted more than once.")
        for k, r in reps[:5]:
            st.markdown(ui.quote_html(r["text"], f"{k}, {r['copies']} copies from {r['distinct_commenters']} "
                                                 "commenter(s)"), unsafe_allow_html=True)

    with st.container(key="card_ins_trend"):
        st.markdown(ui.card_head_html("trend", "Topic growth", "Observed change over time, not overall popularity"),
                    unsafe_allow_html=True)
        tr = ins["trends"]
        if tr.get("status") == "ok":
            st.markdown(f"Compared the last {tr['window_days']} days with the {tr['window_days']} days before, on "
                        f"{len(tr['eligible_videos'])} video(s) whose comments were fully captured for both periods. "
                        "Growth means a larger share of comments, not just many comments.")
            moving = list(tr.get("topics", []))
            if moving:
                st.dataframe(pd.DataFrame(moving), hide_index=True, use_container_width=True)
            else:
                st.caption("No topic changed enough to call it growing or declining.")
        else:
            st.info("**Growth can't be measured yet.** " + tr.get("reason", ""), icon=":material/info:")
            st.caption("Topic counts on this dashboard describe this snapshot of comments, not a trend.")

    cross = ins["topics_across_multiple_videos"]
    if cross:
        with st.container(key="card_ins_cross"):
            st.markdown(ui.card_head_html("layers", "Topics across several videos", "Recurring beyond one video"),
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(cross)[["topic_name", "n_videos", "n_unique_comments", "dominant_intent"]],
                         hide_index=True, use_container_width=True,
                         column_config={"topic_name": "Topic", "n_videos": "Videos",
                                        "n_unique_comments": "Unique comments", "dominant_intent": "Main intent"})

    with st.container(key="card_ins_reply"):
        st.markdown(ui.card_head_html("reply", "Creator replies", "Based on the reply data YouTube returned"),
                    unsafe_allow_html=True)
        rep = pd.DataFrame(ins["creator_replies"]["by_category"])
        if not rep.empty:
            st.dataframe(rep, hide_index=True, use_container_width=True, column_config={
                "predicted_category": "Category", "comments": "Comments", "creator_reply_found": "Creator replied",
                "no_creator_reply_at_collection": "No creator reply (all replies seen)",
                "reply_data_incomplete": "Unknown (replies not all returned)", "threads_with_any_reply": "Any reply"})
            st.caption(ins["creator_replies"]["meaning"])


FORMAT_LABEL = {"debugging walkthrough": "Debugging guide", "comparison": "Comparison",
                "tutorial series": "Tutorial series", "practice session": "Practice session",
                "roadmap / career Q&A": "Roadmap / career Q&A", "concept explanation": "Concept explanation",
                "tutorial": "Tutorial"}
FACTOR_HELP = {"frequency": ("Frequency", 40, "how many different people raised it"),
               "recurrence": ("Recurrence", 25, "how many videos it appears in"),
               "recency": ("Recency", 15, "how recent the latest comment is"),
               "intent": ("Intent", 20, "problems weigh most, then requests, then questions")}


def why_text(r: dict) -> str:
    """The data-based part of the recommendation's reason (who asked, how much, how recent)."""
    keep = [x for x in re.split(r"(?<=\.)\s+", r["reason"]) if not x.startswith(("Trend:", "Score driven"))]
    return " ".join(keep)


def takeaway(r: dict) -> str:
    """One-line summary built only from this recommendation's own numbers."""
    people = r["n_unique_commenters"] if r.get("n_unique_commenters") is not None else r["n_unique_comments"]
    topic = r["topic_name"]
    about = "this" if topic.endswith("?") or len(topic.split()) > 5 else topic
    fmt = FORMAT_LABEL.get(r["suggested_format"], r["suggested_format"]).lower()
    if r["main_intent"] == "Confusion/Problem":
        return f"{people} viewers are stuck on {about}. A step-by-step {fmt} that walks through the fix can address it."
    if r["main_intent"] == "Request":
        return f"{people} viewers asked for content on {about}. A dedicated {fmt} video can meet this demand."
    return (f"{people} viewers have questions about {about}. A clear, beginner-friendly {fmt} can answer them "
            "in one place.")


def recs_tab(d, run_id):
    if d.recs is None:
        st.caption("Recommendations were not generated for this analysis.")
        return
    recs = d.recs["recommendations"]
    if not recs:
        st.info(no_recs_reason(d), icon=":material/info:")
        return
    focus = ss.get("focus_rec")
    if focus:
        recs = sorted(recs, key=lambda r: r["rank"] != focus)      # the one opened from Overview first
    if ss.get("open_rec_run") != run_id:              # another analysis: open its first idea
        ss.open_rec, ss.open_rec_run = recs[0]["rank"], run_id
    when = dict(zip(d.comments["comment_id"], d.comments["published_dt"]))
    for r in recs:
        is_open = ss.open_rec == r["rank"]
        with st.container(key="reccard_open" if is_open else f"reccard_{r['rank']}"):
            fmt = FORMAT_LABEL.get(r["suggested_format"], r["suggested_format"])
            meta = (f"Topic: {r['topic_name']}  ·  {fmt}  ·  Mostly {r['main_intent'].lower()}s  ·  "
                    f"{r['n_videos']} video{'s' if r['n_videos'] != 1 else ''}")
            click_row(f"rechead_{r['rank']}", ui.rec_card_head_html(r, meta, is_open),
                      ("Collapse" if is_open else "Expand") + f" recommendation {r['rank']}", toggle_rec, (r["rank"],))
            if not is_open:
                continue
            st.markdown(ui.rec_why_html(why_text(r), takeaway(r)), unsafe_allow_html=True)
            st.markdown(ui.rec_scores_html(r), unsafe_allow_html=True)
            samples = r["representative_comments"][:4]
            with st.container(key=f"samplehead_{r['rank']}"):
                st.markdown(f'<div class="cp-rc-sec">{ui.icon("msg", 20, ui.INDIGO)}<b>Sample comments '
                            f'({len(samples)})</b></div>', unsafe_allow_html=True)
                st.button("View all comments", key=f"recview_{r['rank']}", icon=":material/arrow_forward:",
                          icon_position="right", on_click=open_comments, args=(None, r["topic_name"]))
            st.markdown(ui.rec_samples_html([(c["text"], c["predicted_category"], ui.rel_time(when.get(c["comment_id"])))
                                             for c in samples]), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

def guarded(fn, *args):
    """Show a plain-language message instead of a Python traceback."""
    try:
        fn(*args)
    except st.errors.StreamlitAPIException:
        raise
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in ("RerunException", "StopException"):
            raise
        st.error("**Something went wrong while showing this page.** The analysis files are unchanged; "
                 "try reopening the analysis from the sidebar.", icon=":material/error:")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}", language=None)



# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

sidebar()
PAGE_FN = {"new": new_analysis_page, "runs": runs_page}
if ss.page not in PAGE_FN and ss.page not in DASH_TITLES:
    ss.page = "new"
if ss.page in PAGE_FN:
    guarded(PAGE_FN[ss.page])
else:
    guarded(dashboard_page, ss.page)
