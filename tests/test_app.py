"""End-to-end tests for the URL-to-insights app.

The YouTube API is replaced by tests/fake_youtube.FakeYouTube (same call chain,
real HttpError objects), run folders go to a temporary directory, and the real
saved model is used for classification.
"""

import json
import time
from pathlib import Path

import pandas as pd
import pytest

import src.collect as collect_mod
import src.runs as runs_mod
from src.analysis import AnalysisFailed, AnalysisSettings, run_analysis
from src.dashboard_data import category_counts, export_comments, filter_comments, load_run_data
from src.job_runner import BusyError, JobRunner, effective_status
from src.runs import Run, find_recent, list_runs
from tests.fake_youtube import FakeYouTube, http_error

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "raw" / "educational_comments.csv"
MODEL_OK = (ROOT / "models" / "creatorpulse_svm.joblib").exists()
pytestmark = pytest.mark.skipif(not (REAL.exists() and MODEL_OK), reason="needs the collected data and saved model")

VIDEO = "GeVCpohRU8Y"       # "What is MCP?" (50 real comments in the collected data)


@pytest.fixture()
def real_comments():
    return pd.read_csv(REAL, dtype=str, keep_default_na=False, encoding="utf-8-sig")


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(runs_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runs_mod, "APP_RAW_DIR", tmp_path / "raw")
    return tmp_path


def apna(real):
    return real[real["channel_title"] == "Apna College"]["channel_id"].iloc[0]


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def test_video_analysis_end_to_end(dirs, real_comments):
    fy = FakeYouTube(real_comments)
    seen = []
    run = run_analysis(f"https://www.youtube.com/watch?v={VIDEO}", AnalysisSettings(max_comments=300),
                       progress=lambda r: seen.append([s["status"] for s in r.meta["stages"]]), youtube=fy)
    m = run.meta
    assert m["status"] == "completed"
    assert [s["status"] for s in m["stages"]] == ["done"] * 8
    # progress was reported stage by stage and never showed a later stage done before an earlier one
    for snap in seen:
        done = [st == "done" for st in snap]
        assert done == sorted(done, reverse=True)
    assert m["totals"]["comments_collected"] == 50 and m["totals"]["videos_analyzed"] == 1
    assert all(v is True for k, v in m["consistency_checks"].items() if k != "recommendations_have_3_to_5_examples")
    # outputs exist inside the run folder; raw CSV in the app raw folder, unique per run
    for name, path in m["outputs"].items():
        assert Path(path).exists(), name
    assert run.raw_path.parent == dirs / "raw" and run.run_id in run.raw_path.name
    pred = pd.read_csv(run.path("comments_predicted.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert set(pred["predicted_category"]) <= {"Question", "Request", "Confusion/Problem", "Praise", "Spam", "Other"}
    # The trend rules received the collector's capture notes and collection time
    ins = json.loads(run.path("audience_insights.json").read_text(encoding="utf-8"))
    assert ins["trends"]["data_end_source"] == "collection time"


def test_predictions_match_the_saved_model_directly(dirs, real_comments):
    from src.batch_predict import default_model_path
    from src.predict import load_model
    run = run_analysis(VIDEO, youtube=FakeYouTube(real_comments))
    pred = pd.read_csv(run.path("comments_predicted.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    direct = load_model(default_model_path()).predict(pred["text"].tolist())
    assert list(pred["predicted_category"]) == list(direct)


def test_two_runs_never_overwrite_each_other(dirs, real_comments):
    fy = FakeYouTube(real_comments)
    a = run_analysis(VIDEO, youtube=fy)
    b = run_analysis(VIDEO, youtube=fy)
    assert a.run_id != b.run_id and a.raw_path != b.raw_path
    assert a.raw_path.exists() and b.raw_path.exists()
    assert {m["run_id"] for m in list_runs()} == {a.run_id, b.run_id}


def test_channel_analysis_and_recommendations_cite_real_comments(dirs, real_comments):
    fy = FakeYouTube(real_comments, handles={"@ApnaCollege": apna(real_comments)})
    run = run_analysis("https://www.youtube.com/@ApnaCollege", AnalysisSettings(max_videos=4, max_comments=50),
                       youtube=fy)
    assert run.meta["totals"]["videos_analyzed"] == 4
    recs = json.loads(run.path("content_recommendations.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(run.raw_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    texts = dict(zip(raw["comment_id"], raw["text"]))
    for r in recs["recommendations"]:
        for c in r["representative_comments"]:
            assert texts[c["comment_id"]] == c["text"]            # quoted verbatim from collected data
        assert set(r["evidence_comment_ids"]) <= set(texts)


def test_failure_keeps_earlier_outputs_and_names_the_stage(dirs, real_comments, monkeypatch):
    import src.topics as topics_mod

    def boom(*a, **k):
        raise RuntimeError("simulated clustering failure")
    monkeypatch.setattr(topics_mod, "run", boom)
    with pytest.raises(AnalysisFailed) as e:
        run_analysis(VIDEO, youtube=FakeYouTube(real_comments))
    run = e.value.run
    m = run.meta
    assert m["status"] == "failed" and m["failed_stage"] == "topics"
    assert [s["status"] for s in m["stages"]][:5] == ["done", "done", "done", "done", "failed"]
    assert all(s["status"] == "skipped" for s in m["stages"][5:])
    assert run.path("comments_predicted.csv").exists() and run.raw_path.exists()
    assert "Discovering audience topics failed" in m["error"]["message"]
    assert "Traceback" not in m["error"]["message"] + m["error"]["fix"]
    assert (run.dir / "error.log").exists()


def test_api_errors_fail_the_collect_stage_with_a_fix(dirs, real_comments):
    for err, code in [(http_error(403, "quotaExceeded"), "quota_exceeded"),
                      (http_error(400, "keyInvalid", "API key not valid"), "api_key_invalid"),
                      (http_error(403, "accessNotConfigured"), "api_forbidden")]:
        with pytest.raises(AnalysisFailed) as e:
            run_analysis(VIDEO, youtube=FakeYouTube(real_comments, errors={"all": err}))
        m = e.value.run.meta
        assert m["failed_stage"] == "collect" and m["error"]["code"] == code and m["error"]["fix"]


def test_comments_disabled_and_missing_video(dirs, real_comments):
    with pytest.raises(AnalysisFailed) as e:
        run_analysis(VIDEO, youtube=FakeYouTube(real_comments, disabled={VIDEO}))
    assert e.value.run.meta["error"]["code"] == "comments_disabled"
    with pytest.raises(AnalysisFailed) as e:
        run_analysis("https://youtu.be/AAAAAAAAAAA", youtube=FakeYouTube(real_comments))
    assert e.value.run.meta["error"]["code"] == "not_found"


def test_invalid_url_and_missing_key_fail_validation(dirs, real_comments, monkeypatch):
    with pytest.raises(AnalysisFailed) as e:
        run_analysis("https://example.com/x", youtube=FakeYouTube(real_comments))
    assert e.value.run.meta["failed_stage"] == "validate"
    monkeypatch.setattr(collect_mod, "api_key_configured", lambda: False)
    with pytest.raises(AnalysisFailed) as e:
        run_analysis(VIDEO)
    assert e.value.run.meta["error"]["code"] == "missing_api_key"


def test_missing_model_is_reported_before_using_quota(dirs, real_comments, tmp_path):
    fy = FakeYouTube(real_comments)
    with pytest.raises(AnalysisFailed) as e:
        run_analysis(VIDEO, youtube=fy, model_path=tmp_path / "nope.joblib")
    assert e.value.run.meta["error"]["code"] == "model_missing"
    assert fy.calls == []                       # no API call was made


def test_spam_toggle(dirs, real_comments):
    run = run_analysis(VIDEO, AnalysisSettings(exclude_spam=False), youtube=FakeYouTube(real_comments))
    assert run.meta["status"].startswith("completed")
    assert run.meta["settings"]["exclude_spam"] is False


# --------------------------------------------------------------------------
# Background runner, duplicates, run listing, dashboard data
# --------------------------------------------------------------------------

def test_job_runner_blocks_duplicate_and_concurrent_runs(dirs, real_comments):
    jr = JobRunner(youtube_factory=lambda: FakeYouTube(real_comments, delay=0.2))
    rid = jr.start(VIDEO, AnalysisSettings())
    with pytest.raises(BusyError):
        jr.start(f"https://youtu.be/{VIDEO}", AnalysisSettings())      # same video, other URL form
    with pytest.raises(BusyError):
        jr.start("https://www.youtube.com/watch?v=ausLKMojXaY", AnalysisSettings())
    jr.wait(rid, timeout=120)
    assert Run.load(rid).meta["status"] == "completed"
    assert not jr.active()
    assert len(list_runs()) == 1


def test_find_recent_and_interrupted_status(dirs, real_comments):
    from src.analysis import target_key
    run_analysis(VIDEO, AnalysisSettings(), youtube=FakeYouTube(real_comments))
    assert find_recent(target_key(f"https://youtu.be/{VIDEO}"), AnalysisSettings().as_dict())
    assert find_recent(target_key(VIDEO), AnalysisSettings(max_comments=50).as_dict()) is None
    stale = Run.create("https://youtu.be/x", {})
    assert effective_status(stale.meta, running_ids={}) == "interrupted"
    assert effective_status(stale.meta, running_ids={stale.run_id: "k"}) == "running"


def test_dashboard_data_filters_and_exports(dirs, real_comments):
    run = run_analysis(VIDEO, youtube=FakeYouTube(real_comments))
    d = load_run_data(run.run_id)
    assert d.available["comments"] and d.available["recs"]
    cc = category_counts(d.comments)
    assert cc["comments"].sum() == len(d.comments)
    q = filter_comments(d.comments, ["Request"])
    assert len(q) and set(q["predicted_category"]) == {"Request"}
    s = filter_comments(d.comments, query="MCP")
    assert all("mcp" in t.lower() for t in s["text"])
    csv = export_comments(d.comments).decode("utf-8-sig")
    assert csv.splitlines()[0].startswith("comment_id,text,predicted_category,topic_name")
    assert len(pd.read_csv(pd.io.common.StringIO(csv))) == len(d.comments)


def test_dashboard_data_for_partial_run(dirs, real_comments, monkeypatch):
    import src.insights as ins_mod
    monkeypatch.setattr(ins_mod, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(AnalysisFailed) as e:
        run_analysis(VIDEO, youtube=FakeYouTube(real_comments))
    d = load_run_data(e.value.run.run_id)
    assert d.available["comments"] and d.available["topics"]
    assert not d.available["insights"] and not d.available["recs"]


# --------------------------------------------------------------------------
# Streamlit UI (in-process AppTest)
# --------------------------------------------------------------------------

def _wait_all_done(runs_dir: Path, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        metas = [json.loads(p.read_text(encoding="utf-8")) for p in runs_dir.glob("*/metadata.json")]
        if metas and all(m["status"] != "running" for m in metas):
            return metas
        time.sleep(0.2)
    raise TimeoutError


@pytest.fixture()
def app(dirs, real_comments, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()
    st.cache_data.clear()
    fake = FakeYouTube(real_comments, handles={"@ApnaCollege": apna(real_comments)})
    monkeypatch.setattr(collect_mod, "build_client", lambda: fake)
    monkeypatch.setattr(collect_mod, "api_key_configured", lambda: True)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    yield at, fake
    at.get_runner = None


def test_ui_landing_renders(app):
    at, _ = app
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Analyze a YouTube Video or Channel" in md and "Paste a YouTube URL to get started" in md
    assert at.session_state["page"] == "new"                 # opens straight on the paste-URL page
    assert at.text_input(key="url").placeholder == "https://www.youtube.com/watch?v=example"
    assert at.button(key="analyze").label == "Analyze"
    for key in ("new", "overview", "comments", "insights", "recs", "runs"):
        assert at.button(key=f"nav_{key}")
    assert not [b for b in at.button if b.key == "nav_settings"]      # no Settings page
    assert not [b for b in at.button if b.key == "nav_topics"]        # Topics is not in the sidebar
    assert not [b for b in at.button if b.key == "nav_home"]          # no Home page


def test_ui_every_page_renders_without_a_run(app):
    at, _ = app
    for key in ("overview", "comments", "insights", "recs", "runs", "new"):
        at.button(key=f"nav_{key}").click().run()
        assert not at.exception, key
        assert not at.error, key


def test_ui_invalid_url_shows_error_and_creates_no_run(app, dirs):
    at, fake = app
    at.text_input(key="url").input("https://www.youtube.com/playlist?list=PL1").run()
    at.button(key="analyze").click().run()
    assert any("Playlist links aren't supported" in e.value for e in at.error)
    assert not list((dirs / "runs").glob("*")) if (dirs / "runs").exists() else True
    assert fake.calls == []


def test_ui_missing_api_key(app, monkeypatch):
    at, fake = app
    monkeypatch.setattr(collect_mod, "api_key_configured", lambda: False)
    at.text_input(key="url").input(f"https://www.youtube.com/watch?v={VIDEO}").run()
    at.button(key="analyze").click().run()
    assert any("No YouTube API key" in e.value for e in at.error)
    assert fake.calls == []


def test_ui_full_flow_collects_once_and_shows_dashboard(app, dirs):
    at, fake = app
    at.text_input(key="url").input(f"https://www.youtube.com/watch?v={VIDEO}").run()
    at.button(key="analyze").click().run()
    assert at.session_state["active_run"]
    metas = _wait_all_done(dirs / "runs")
    assert len(metas) == 1 and metas[0]["status"] == "completed"
    at.run()
    # Success feedback is shown once, when the finished analysis first opens
    assert any("Analysis complete" in m.value for m in at.markdown) and at.session_state["celebrate"] is None
    at.run()
    assert not any("Analysis complete" in m.value for m in at.markdown)
    calls_after_run = list(fake.calls)
    assert sum(c[0] == "commentThreads" for c in calls_after_run) == 1
    # Re-render several times (as Streamlit does on every interaction): no new collection.
    for _ in range(3):
        at.run()
    assert fake.calls == calls_after_run
    assert not at.exception
    # Finished -> the app opens the Overview of this run
    assert at.session_state["page"] == "overview" and at.session_state["run_id"] == metas[0]["run_id"]
    md = " ".join(m.value for m in at.markdown)
    from src.dashboard_data import load_run_data
    n = len(load_run_data(metas[0]["run_id"], dirs / "runs").comments)
    assert "Comments analyzed" in md and f">{n}<" in md       # real number from this run
    for key in ("comments", "insights", "recs", "runs", "overview"):
        at.button(key=f"nav_{key}").click().run()
        assert not at.exception, key
    at.button(key="nav_comments").click().run()
    # Filtering the comment table works
    at.multiselect(key="w_cats").set_value(["Request"]).run()
    assert not at.exception
    # Same URL again from the landing page -> offered the previous result, nothing collected
    at.button(key="nav_new").click().run()
    at.text_input(key="url").input(f"https://www.youtube.com/watch?v={VIDEO}").run()
    at.button(key="analyze").click().run()
    assert at.session_state["duplicate_of"] == metas[0]["run_id"]
    assert fake.calls == calls_after_run


def test_ui_failed_run_shows_stage_and_fix(app, dirs, monkeypatch):
    at, fake = app
    fake.disabled.add(VIDEO)
    at.text_input(key="url").input(f"https://youtu.be/{VIDEO}").run()
    at.button(key="analyze").click().run()
    _wait_all_done(dirs / "runs")
    at.run()
    assert any("Collecting comments failed" in e.value and "turned off" in e.value for e in at.error)
    assert not at.exception


def test_metadata_save_survives_windows_file_locks(dirs, monkeypatch):
    """Windows: os.replace fails with 'Access is denied' while another reader has the file open."""
    import os
    real = os.replace
    fails = {"n": 0}

    def flaky(src, dst):
        if fails["n"] < 5:
            fails["n"] += 1
            raise PermissionError(13, "Access is denied")
        return real(src, dst)
    monkeypatch.setattr(runs_mod.os, "replace", flaky)
    run = Run.create("https://youtu.be/x", {})
    run.start_stage("validate")
    assert fails["n"] == 5 and Run.load(run.run_id).meta["stages"][0]["status"] == "running"
    assert not list(run.dir.glob("*.tmp"))

    def always(src, dst):
        raise PermissionError(13, "Access is denied")
    monkeypatch.setattr(runs_mod.os, "replace", always)
    monkeypatch.setattr(runs_mod.time, "sleep", lambda s: None)
    run.finish_stage("validate")                     # falls back to writing in place
    assert Run.load(run.run_id).meta["stages"][0]["status"] == "done"
    assert not list(run.dir.glob("*.tmp"))


def test_ui_analyzes_exactly_one_video_even_for_a_channel_link(app, dirs):
    at, fake = app
    assert not at.number_input                       # no "videos to analyze" setting
    at.text_input(key="url").input("https://www.youtube.com/@ApnaCollege").run()
    at.button(key="analyze").click().run()
    metas = _wait_all_done(dirs / "runs")
    assert metas[0]["settings"]["max_videos"] == 1
    assert len(metas[0]["collection"]["videos"]) == 1
    assert sum(c[0] == "commentThreads" for c in fake.calls) == 1


# --------------------------------------------------------------------------
# Overview -> Comments Explorer interactions
# --------------------------------------------------------------------------

def _analyzed(app, dirs):
    at, fake = app
    at.text_input(key="url").input(f"https://www.youtube.com/watch?v={VIDEO}").run()
    at.button(key="analyze").click().run()
    metas = _wait_all_done(dirs / "runs")
    at.run()
    assert at.session_state["page"] == "overview"
    from src.dashboard_data import load_run_data
    return at, load_run_data(metas[0]["run_id"], dirs / "runs")


def _table(at):
    return at.dataframe[0].value


def _count_text(at):
    return next(m.value for m in at.markdown if "matching comment" in m.value)


def test_ui_overview_is_compact(app, dirs):
    at, d = _analyzed(app, dirs)
    md = " ".join(m.value for m in at.markdown)
    for label in ("Comments analyzed", "Videos analyzed", "Topics discovered", "Content recommendations"):
        assert label in md
    assert "Comments collected" not in md                      # four KPI cards only
    assert not [b for b in at.button if b.key == "change_run"]   # no Change analysis button
    assert 'url(&quot;https://yt3.example.com/' in md and "=s240" in md      # channel profile picture
    assert not [e for e in at.expander if "Run details" in e.label]
    md = " ".join(m.value for m in at.markdown)
    assert "Recent comments" in md and "Comment categories" in md and "Top content opportunities" in md


def test_ui_every_category_row_opens_comments_filtered_to_it(app, dirs):
    at, d = _analyzed(app, dirs)
    from src.dashboard_data import CLASSES
    from src import ui_components as ui
    for cat in CLASSES:
        at.button(key="nav_overview").click().run()
        at.button(key=f"go_cat_{ui.slug(cat)}").click().run()
        assert not at.exception, cat
        assert at.session_state["page"] == "comments", cat
        assert at.multiselect(key="w_cats").value == [cat], cat
        expected = int((d.comments["predicted_category"] == cat).sum())
        assert f"<b>{expected}</b> matching" in _count_text(at), cat
        tbl = _table(at)
        assert len(tbl) == expected and set(tbl["predicted_category"]) <= {cat}, cat
        assert at.button(key="chip_0").label == f"Category: {cat}"


def test_ui_filter_survives_navigation_and_can_be_cleared(app, dirs):
    at, d = _analyzed(app, dirs)
    at.button(key="go_cat_Question").click().run()
    n_q = int((d.comments["predicted_category"] == "Question").sum())
    # Leave the page and come back: still filtered, without choosing the filter again
    for page in ("insights", "recs"):
        at.button(key=f"nav_{page}").click().run()
        at.button(key="nav_comments").click().run()
        assert at.multiselect(key="w_cats").value == ["Question"], page
        assert len(_table(at)) == n_q, page
    # Add a search on top, then remove the category chip only
    at.text_input(key="w_q").input("sir").run()
    assert len(_table(at)) <= n_q
    at.button(key="chip_0").click().run()          # Category: Question
    assert at.multiselect(key="w_cats").value == []
    assert at.text_input(key="w_q").value == "sir"
    # Clear filters removes everything
    at.button(key="clear_filters").click().run()
    assert at.multiselect(key="w_cats").value == [] and at.text_input(key="w_q").value == ""
    assert len(_table(at)) == len(d.comments)
    assert at.button(key="clear_filters").disabled
    # Back to Overview
    at.button(key="nav_overview").click().run()
    assert at.session_state["page"] == "overview" and not at.exception
    assert any("Comment categories" in m.value for m in at.markdown)


def test_ui_multiple_categories_and_view_all(app, dirs):
    at, d = _analyzed(app, dirs)
    at.button(key="go_cat_Request").click().run()
    at.multiselect(key="w_cats").set_value(["Request", "Praise"]).run()
    want = int(d.comments["predicted_category"].isin(["Request", "Praise"]).sum())
    assert len(_table(at)) == want
    at.button(key="nav_overview").click().run()
    at.button(key="hlink_comments").click().run()      # "View all comments" opens unfiltered
    assert at.session_state["page"] == "comments" and len(_table(at)) == len(d.comments)
    assert at.multiselect(key="w_cats").value == []


def test_ui_topic_and_recommendation_rows_open_their_comments(app, dirs):
    at, d = _analyzed(app, dirs)
    real = d.topics[d.topics["topic_id"] != -1].sort_values(["n_unique_comments", "topic_id"], ascending=[False, True])
    if real.empty:
        pytest.skip("no topics in this sample")
    name = real.iloc[0]["topic_name"]
    at.button(key="go_topic_0").click().run()
    assert at.session_state["page"] == "comments" and at.selectbox(key="w_topic").value == name
    tbl = _table(at)
    assert len(tbl) == int(real.iloc[0]["n_comments_raw"]) and set(tbl["topic"]) == {name}
    # Topics page: rows are clickable too
    at.button(key="nav_overview").click().run()
    at.button(key="hlink_topics").click().run()          # Topics page via "View all" on the Overview
    at.button(key="go_tmix_0").click().run()
    assert at.session_state["page"] == "comments" and at.selectbox(key="w_topic").value == name
    recs = (d.recs or {}).get("recommendations", [])
    if recs:
        at.button(key="nav_overview").click().run()
        at.button(key="go_rec_0").click().run()
        assert at.session_state["page"] == "recs" and at.session_state["focus_rec"] == recs[0]["rank"]
        at.button(key=f"recview_{recs[0]['rank']}").click().run()
        assert at.session_state["page"] == "comments"
        assert at.selectbox(key="w_topic").value == recs[0]["topic_name"]
        assert len(_table(at)) == recs[0]["n_comments_raw"]
