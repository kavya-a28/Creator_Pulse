"""
CreatorPulse — presentation helpers for the Streamlit app (light theme):
CSS, inline icons, the progress stepper, charts and small HTML fragments.
No data processing happens here.
"""

import html
import re
from datetime import datetime

import pandas as pd

# Design tokens (light, indigo accent). Chart series colors were checked with the
# dataviz palette validator on a white surface (CVD, normal-vision and 3:1 contrast pass).
BG, SIDEBAR, CARD, LINE = "#F7F8FC", "#F3F4FB", "#FFFFFF", "#E6E8F2"
INK, MUTED, FAINT = "#12163A", "#5D6487", "#9AA0BC"
INDIGO, INDIGO_SOFT, AMBER = "#4F46E5", "#EEF0FF", "#D97706"
BAR = INDIGO
INTENT_COLORS = {"Question": "#2a78d6", "Request": "#eb6834", "Confusion/Problem": "#199e70"}
CHART_FONT = "'Plus Jakarta Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"

CATEGORY_HELP = {
    "Question": "Asks something answerable, or for advice",
    "Request": "Asks you to make or share something",
    "Confusion/Problem": "Stuck, an error, or didn't understand",
    "Praise": "Thanks and appreciation",
    "Spam": "Promotion, scams, link-dropping",
    "Other": "Jokes, attendance, general remarks",
}

# --------------------------------------------------------------------------
# Icons (simple line icons drawn for this app)
# --------------------------------------------------------------------------

def _svg(body: str, size: int = 24, stroke: str = "currentColor", fill: str = "none", vb: str = "0 0 24 24") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="{vb}" fill="{fill}" stroke="{stroke}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>')


LOGO = ('<svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true"><rect width="34" height="34" rx="9" '
        f'fill="{INDIGO}"/><path d="M13 10.5v13l10.5-6.5z" fill="#fff"/></svg>')
ICON_CHAT = _svg('<path d="M21 11.5a8.4 8.4 0 0 1-12.2 7.5L3 21l2-5.3A8.5 8.5 0 1 1 21 11.5z"/>', 24, "#2a78d6")
ICON_BARS = _svg('<path d="M5 20v-6M10 20V9M15 20v-9M20 20V4"/>', 24, "#7C3AED")
ICON_BULB = _svg('<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3z"/>',
                 24, AMBER)
ICON_LINK = _svg('<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
                 20, MUTED)
ICON_HELP = _svg('<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.3 2.4c-.5.2-.8.7-.8 1.2v.4M12 17h.01"/>',
                 18, MUTED)
VIDEO_TILE = ('<svg width="56" height="44" viewBox="0 0 56 44" aria-hidden="true"><rect x="1" y="1" width="54" height="42" '
              f'rx="11" fill="{INDIGO_SOFT}" stroke="#D9DCFB"/><path d="M23 14v16l13-8z" fill="{INDIGO}"/></svg>')
EMPTY_ART = f"""<svg class="cp-art" width="200" height="110" viewBox="0 0 290 160" aria-hidden="true">
<ellipse cx="150" cy="86" rx="125" ry="70" fill="#EEF0FB"/>
<rect x="72" y="26" width="168" height="104" rx="12" fill="#fff" stroke="{LINE}" stroke-width="1.5"/>
<path d="M72 38a12 12 0 0 1 12-12h144a12 12 0 0 1 12 12v8H72z" fill="#E4E7FB"/>
<circle cx="86" cy="36" r="2.6" fill="#B8BEF5"/><circle cx="95" cy="36" r="2.6" fill="#B8BEF5"/><circle cx="104" cy="36" r="2.6" fill="#B8BEF5"/>
<rect x="88" y="62" width="48" height="34" rx="8" fill="{INDIGO_SOFT}"/><path d="M106 70v18l14-9z" fill="{INDIGO}"/>
<rect x="148" y="66" width="76" height="9" rx="4.5" fill="#E4E7F2"/><rect x="148" y="83" width="50" height="9" rx="4.5" fill="#EEF0F6"/>
<path d="M40 52l10 9M33 72h16" stroke="{INDIGO}" stroke-width="4" stroke-linecap="round"/>
</svg>"""


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {{ --bg:{BG}; --side:{SIDEBAR}; --card:{CARD}; --line:{LINE}; --ink:{INK}; --muted:{MUTED}; --faint:{FAINT};
  --indigo:{INDIGO}; --indigo-soft:{INDIGO_SOFT}; --font:{CHART_FONT}; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg); }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"] {{ font-family: var(--font); color: var(--ink); }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: var(--font); color: var(--ink); letter-spacing: -0.02em; }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1220px; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: var(--side); border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] {{ padding-top: .2rem; padding-bottom: .5rem; }}
[data-testid="stSidebarHeader"] {{ height: 2.2rem; min-height: 2.2rem; padding-top: .4rem; padding-bottom: 0; }}
/* The sidebar fits the window: no scrollbar */
[data-testid="stSidebarContent"] {{ overflow: hidden; }}
.cp-brand {{ display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.45rem; letter-spacing:-.02em; }}
.cp-brand .a {{ color: var(--ink); }} .cp-brand .b {{ color: var(--indigo); }}
.cp-tag {{ color: var(--muted); font-size: .85rem; line-height: 1.35; margin: .3rem 0 .7rem; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button {{ justify-content:flex-start; border:none; background:transparent;
  color: var(--ink); font-weight: 500; padding: .55rem .8rem; border-radius: 10px; box-shadow:none; min-height: 2.35rem; padding-top: .35rem; padding-bottom: .35rem; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover {{ background: #E9EBF8; color: var(--ink); }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"],
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] {{
  background: #E8E9FD; color: var(--indigo); font-weight: 600; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button p {{ font-size: .98rem; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button > div {{ justify-content: flex-start; width: 100%; gap: .35rem; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] {{ margin-bottom: -.75rem; }}
.cp-side-rule {{ border-top: 1px solid var(--line); margin: .9rem .2rem .5rem; }}
.cp-side-card {{ background: #ECEEFA; border-radius: 14px; padding: 1.1rem 1.1rem 1.2rem; margin-top: 1.6rem; }}
.cp-side-card b {{ display:block; font-weight:800; font-size: 1.02rem; line-height:1.3; margin: .5rem 0 .55rem; }}
.cp-side-card span {{ color: var(--muted); font-size: .86rem; line-height: 1.5; }}

/* Page header */
.cp-head h1 {{ font-size: clamp(1.6rem, 2.6vw, 2.15rem); font-weight: 800; margin: 0 0 .25rem; line-height: 1.15; padding: 0; }}
.cp-head p {{ color: var(--muted); font-size: 1rem; margin: 0; max-width: 80ch; }}

/* The URL card */
.st-key-url_card {{ background: var(--card); border: 1px solid var(--line) !important; border-radius: 16px;
  box-shadow: 0 6px 24px rgba(18,22,58,.06); padding: 1.1rem 1.3rem .7rem; margin-top: 1rem; }}
.st-key-url [data-testid="stTextInputRootElement"] {{ height: 54px; border-radius: 12px; border: 1px solid #D9DCEA;
  background: #fff; }}
.st-key-url [data-testid="stTextInputRootElement"]:focus-within {{ border-color: var(--indigo);
  box-shadow: 0 0 0 4px rgba(79,70,229,.14); }}
.st-key-url input {{ height: 52px; font-size: 1.08rem; padding-left: 3rem; color: var(--ink); background: transparent;
  background-image: url("data:image/svg+xml;utf8,{ICON_LINK.replace('"', "'").replace('#', '%23').replace('<', '%3C').replace('>', '%3E')}");
  background-repeat: no-repeat; background-position: 1rem center; }}
.st-key-analyze button {{ height: 54px; border-radius: 12px; font-weight: 700; font-size: 1.12rem;
  background: var(--indigo); color: #fff; border: none; box-shadow: 0 6px 16px rgba(79,70,229,.28); }}
.st-key-analyze button:hover {{ background: #4338CA; color: #fff; }}
.st-key-analyze button:focus-visible {{ outline: 3px solid #A5B4FC; outline-offset: 2px; }}
.st-key-analyze button p {{ font-size: 1.12rem; font-weight: 700; color: #fff; }}
.st-key-analyze button:disabled, .st-key-analyze button[disabled] {{ background: #A5A9E8 !important; color: #fff !important;
  box-shadow: none; opacity: 1; }}
.st-key-analyze button:disabled p, .st-key-analyze button:disabled span {{ color: #fff !important; }}
.cp-vtile {{ display:flex; align-items:center; height: 54px; }}
.cp-support {{ display:flex; align-items:center; gap:.5rem; color: var(--muted); font-size: .93rem; padding-top: .45rem; }}
[data-testid="stPopoverButton"] {{ background: var(--indigo-soft); color: var(--indigo); border: 1px solid #DDE0FB;
  border-radius: 10px; }}
[data-testid="stPopoverButton"] p, [data-testid="stPopoverButton"] span {{ color: var(--indigo); font-weight: 600; }}
[data-testid="stPopoverButton"]:hover {{ background: #E3E6FF; border-color: #C9CDF8; }}

/* Stepper */
.cp-stepper {{ display:flex; align-items:flex-start; margin: 1.3rem 0 .2rem; }}
.cp-step {{ flex: 1; display:flex; flex-direction:column; align-items:center; position: relative; text-align:center; }}
.cp-step:not(:last-child)::after {{ content:""; position:absolute; top: 21px; left: calc(50% + 28px); right: calc(-50% + 28px);
  height: 2px; background: #DADDEB; }}
.cp-step.done:not(:last-child)::after {{ background: var(--indigo); }}
.cp-dot {{ width: 42px; height: 42px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
  font-weight: 600; background: #EEF0F7; color: var(--muted); border: 1px solid #E1E4F0; }}
.cp-step.active .cp-dot {{ background: var(--indigo); color: #fff; border-color: var(--indigo);
  box-shadow: 0 0 0 6px rgba(79,70,229,.12); }}
.cp-step.done .cp-dot {{ background: var(--indigo); color: #fff; border-color: var(--indigo); }}
.cp-step.failed .cp-dot {{ background: #FEE2E2; color: #B91C1C; border-color: #FCA5A5; }}
.cp-step .lbl {{ margin-top: .55rem; font-size: .92rem; color: var(--muted); max-width: 15ch; line-height: 1.3;
  overflow-wrap: normal; word-break: keep-all; hyphens: none; }}
.cp-step.active .lbl {{ color: var(--indigo); font-weight: 700; }}
.cp-step.done .lbl {{ color: var(--ink); }}
.cp-step.failed .lbl {{ color: #B91C1C; font-weight: 600; }}
@media (prefers-reduced-motion: no-preference) {{
  .cp-step.active.running .cp-dot {{ animation: cp-pulse 1.4s ease-in-out infinite; }}
  @keyframes cp-pulse {{ 0%,100% {{ box-shadow: 0 0 0 6px rgba(79,70,229,.12); }} 50% {{ box-shadow: 0 0 0 10px rgba(79,70,229,.06); }} }}
}}
@media (max-width: 800px) {{ .cp-step .lbl {{ font-size: .75rem; }} .cp-dot {{ width: 34px; height: 34px; }}
  .cp-step:not(:last-child)::after {{ top: 17px; }} }}

/* Empty state + features */
.cp-empty {{ text-align:center; margin: .6rem auto 0; max-width: 760px; }}
@media (max-height: 820px) {{ .cp-art {{ display: none; }} }}
.cp-empty h2 {{ font-size: 1.4rem; font-weight: 800; margin: .3rem 0 .3rem; padding: 0; }}
.cp-empty p {{ color: var(--muted); font-size: 1rem; line-height: 1.5; margin: 0 auto; max-width: 60ch; }}
.cp-features {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); background: var(--card);
  border: 1px solid var(--line); border-radius: 16px; margin: 1rem auto 0; max-width: 880px; }}
.cp-feat {{ display:flex; align-items:center; gap: .9rem; padding: .9rem 1.2rem; }}
.cp-feat + .cp-feat {{ border-left: 1px solid var(--line); }}
.cp-feat .ic {{ width: 48px; height: 48px; border-radius: 50%; display:flex; align-items:center; justify-content:center; flex:none; }}
.cp-feat b {{ display:block; font-weight: 600; }} .cp-feat span {{ color: var(--muted); }}
@media (max-width: 800px) {{ .cp-features {{ grid-template-columns: 1fr; }} .cp-feat + .cp-feat {{ border-left:none; border-top: 1px solid var(--line); }} }}

/* Progress card */
.cp-now {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.2rem; margin-top: 1rem; }}
.cp-now b {{ font-weight: 700; }} .cp-now span {{ color: var(--muted); }}

/* Dashboard */
.cp-runbar {{ display:flex; flex-wrap:wrap; align-items:center; gap:.5rem 1.2rem; color: var(--muted); font-size:.95rem; margin-top:.4rem; }}
.cp-runbar a {{ color: var(--indigo); word-break: break-all; }}
.cp-status {{ display:inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .82rem; font-weight: 700; }}
.cp-status.ok {{ background: #DCFCE7; color: #166534; }} .cp-status.warn {{ background: #FEF3C7; color: #92400E; }}
.cp-status.bad {{ background: #FEE2E2; color: #991B1B; }} .cp-status.run {{ background: var(--indigo-soft); color: var(--indigo); }}
.cp-kpis {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: .9rem; margin: 1.3rem 0 1.1rem; }}
.cp-kpi {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; }}
.cp-kpi b {{ display:block; font-weight: 800; font-size: 1.9rem; line-height: 1.15; }}
.cp-kpi span {{ color: var(--muted); font-size: .9rem; }}
@media (max-width: 900px) {{ .cp-kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
.cp-quote {{ border-left: 3px solid #D5D8F6; padding: .3rem 0 .3rem .9rem; margin: .45rem 0; }}
.cp-quote small {{ display:block; color: var(--faint); font-size: .8rem; margin-top: .15rem; }}
.cp-rec-title {{ font-weight: 800; font-size: 1.2rem; margin: 0; }}
.cp-score {{ font-weight: 800; font-size: 2rem; color: var(--indigo); line-height: 1; text-align: right; }}
.cp-score small {{ display:block; font-size: .8rem; color: var(--muted); font-weight: 500; }}
.cp-muted, .cp-note {{ color: var(--muted); }}
.cp-stages {{ list-style:none; padding:0; margin:.4rem 0 0; }}
.cp-stages li {{ display:flex; gap:.8rem; padding:.4rem 0; color: var(--faint); }}
.cp-stages li .ic {{ width:1.3rem; text-align:center; font-weight:700; }}
.cp-stages li.done {{ color: var(--ink); }} .cp-stages li.done .ic {{ color: #16A34A; }}
.cp-stages li.running {{ color: var(--indigo); }} .cp-stages li.failed {{ color: #B91C1C; }}
.cp-stages li small {{ display:block; color: var(--muted); font-size:.86rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--card); border-radius: 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.2rem; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


# --------------------------------------------------------------------------
# HTML fragments
# --------------------------------------------------------------------------

def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def brand_html() -> str:
    return (f'<div class="cp-brand">{LOGO}<span><span class="a">Creator</span><span class="b">Pulse</span></span></div>'
            '<div class="cp-tag">Turn YouTube comments into your next great video idea.</div>')


ICON_BULB_PATH = ('<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 '
                  '12 3z"/>')


def header_html(title: str, sub: str = "") -> str:
    p = f"<p>{esc(sub)}</p>" if sub else ""
    return f'<div class="cp-head"><h1>{esc(title)}</h1>{p}</div>'


def support_html() -> str:
    return f'<div class="cp-support">{ICON_HELP}<span>Supported: YouTube video URL or channel URL/ID</span></div>'


def empty_state_html() -> str:
    return (f'<div class="cp-empty">{EMPTY_ART}<h2>Paste a YouTube URL to get started</h2>'
            '<p>We will collect comments, analyze them using NLP, discover key topics, and suggest content ideas '
            'for you.</p></div>' + features_html())


def features_html() -> str:
    items = [(ICON_CHAT, "#E7F0FC", "Analyze", "audience comments"),
             (ICON_BARS, "#F1EAFE", "Discover", "recurring topics"),
             (ICON_BULB, "#FEF3DC", "Get content", "recommendations")]
    cells = "".join(f'<div class="cp-feat"><div class="ic" style="background:{bg}">{ic}</div>'
                    f'<div><b>{esc(a)}</b><span>{esc(b)}</span></div></div>' for ic, bg, a, b in items)
    return f'<div class="cp-features">{cells}</div>'


# The 7 user-facing steps and the pipeline stages behind each one.
STEPS = [("Validate URL", ["validate"]), ("Collect comments", ["collect"]), ("Preprocess text", ["preprocess"]),
         ("Classify comments", ["classify"]), ("Find topics", ["topics", "insights"]),
         ("Generate recommendations", ["recommend"]), ("Complete", ["dashboard"])]


def step_states(stages) -> list:
    """Status per step from the run's stage list: done / running / failed / pending / skipped."""
    by = {s["key"]: s["status"] for s in stages}
    out = []
    for _, keys in STEPS:
        st = [by.get(k, "pending") for k in keys]
        if "failed" in st:
            out.append("failed")
        elif all(x == "done" for x in st):
            out.append("done")
        elif "running" in st or ("done" in st and "pending" in st):
            out.append("running")
        elif all(x == "skipped" for x in st):
            out.append("skipped")
        else:
            out.append("pending")
    return out


def stepper_html(states=None) -> str:
    """states=None: idle page (step 1 highlighted as the next thing to do)."""
    idle = states is None
    cells = []
    for i, (label, _) in enumerate(STEPS):
        stt = "pending" if idle else states[i]
        cls = {"done": "done", "running": "active running", "failed": "failed"}.get(stt, "")
        if idle and i == 0:
            cls = "active"
        mark = "✓" if stt == "done" else ("!" if stt == "failed" else str(i + 1))
        cells.append(f'<div class="cp-step {cls}"><div class="cp-dot">{mark}</div><div class="lbl">{esc(label)}</div></div>')
    return f'<div class="cp-stepper" role="list">{"".join(cells)}</div>'


def fmt_time(iso: str) -> str:
    """ISO timestamp -> local time of this computer, e.g. '26 Oct 2026, 10:55 (UTC+05:30)'."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone()
        off = dt.utcoffset()
        mins = int(off.total_seconds() // 60) if off is not None else 0
        sign = "+" if mins >= 0 else "-"
        return dt.strftime("%d %b %Y, %H:%M") + f" (UTC{sign}{abs(mins) // 60:02d}:{abs(mins) % 60:02d})"
    except Exception:
        return str(iso)


STATUS_STYLE = {"completed": ("ok", "Completed"), "completed_with_warnings": ("warn", "Completed with warnings"),
                "failed": ("bad", "Failed"), "running": ("run", "Running"), "interrupted": ("bad", "Interrupted")}


def status_html(status: str) -> str:
    cls, label = STATUS_STYLE.get(status, ("run", status))
    return f'<span class="cp-status {cls}">{label}</span>'


ICONS = {"pending": "○", "running": "◐", "done": "✓", "failed": "✕", "skipped": "–"}


def stages_html(stages) -> str:
    rows = []
    for s in stages:
        detail = f"<small>{esc(s.get('detail'))}</small>" if s.get("detail") and s["status"] != "pending" else ""
        secs = f" ({s['seconds']:.1f}s)" if s.get("seconds") is not None and s["status"] == "done" else ""
        rows.append(f'<li class="{s["status"]}"><span class="ic">{ICONS.get(s["status"], "○")}</span>'
                    f'<div>{esc(s["label"])}{secs}{detail}</div></li>')
    return f'<ul class="cp-stages">{"".join(rows)}</ul>'


def quote_html(text: str, meta: str = "") -> str:
    m = f"<small>{esc(meta)}</small>" if meta else ""
    return f'<div class="cp-quote">{esc(text)}{m}</div>'


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Overview (dashboard) components
# --------------------------------------------------------------------------

# One color per category, used everywhere. Validated on white (adjacent pairs
# pass the normal-vision floor; every bar also carries a text label).
CATEGORY_COLORS = {"Question": "#2a78d6", "Request": "#eb6834", "Confusion/Problem": "#199e70",
                   "Praise": "#e87ba4", "Spam": "#eda100", "Other": "#A3ABC2"}

_I = {
    "chat": '<path d="M21 11.5a8.4 8.4 0 0 1-12.2 7.5L3 21l2-5.3A8.5 8.5 0 1 1 21 11.5z"/>',
    "doc": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
    "play": '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M10 9v6l5-3z"/>',
    "layers": '<path d="M12 3 2 8l10 5 10-5z"/><path d="m2 13 10 5 10-5"/>',
    "bulb": ICON_BULB_PATH,
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/>',
    "external": '<path d="M14 4h6v6M20 4l-9 9"/><path d="M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.3 2.4c-.5.2-.8.7-.8 1.2v.4M12 17h.01"/>',
    "request": '<path d="M12 5v14M5 12h14"/>',
    "alert": '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17h.01"/>',
    "repeat": '<path d="M17 2l4 4-4 4"/><path d="M3 11V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4"/><path d="M21 13v2a3 3 0 0 1-3 3H3"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    "reply": '<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 5 5v6"/>',
    "star": '<path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1 6.2L12 17.3 6.5 20.2l1-6.2L3 9.6l6.2-.9z"/>',
}
KPI_STYLE = {"chat": ("#E7F0FC", "#2a78d6"), "doc": ("#F1EAFE", "#7C3AED"), "play": ("#E7F0FC", "#2a78d6"),
             "layers": ("#F1EAFE", "#7C3AED"), "bulb": ("#FEF3DC", "#D97706"),
             "help": ("#E7F0FC", "#2a78d6"), "request": ("#FDECE4", "#C2410C"), "alert": ("#E3F5EE", "#137a57"),
             "repeat": ("#F1EAFE", "#7C3AED"), "trend": ("#E7F0FC", "#2a78d6"), "reply": ("#EEF0FF", "#4F46E5"),
             "star": ("#FEF3DC", "#D97706")}
RANK_STYLE = [("#FCE7F3", "#BE185D"), ("#E0E7FF", "#4338CA"), ("#FEF3C7", "#B45309"), ("#E5E7EB", "#374151")]


def icon(name: str, size: int = 22, color: str = "currentColor") -> str:
    return _svg(_I[name], size, color)


def run_meta_html(status: str, channel: str, url: str, analyzed: str) -> str:
    cls, label = STATUS_STYLE.get(status, ("run", status))
    chk = icon("check", 15) if cls == "ok" else ""
    return (f'<div class="cp-meta"><span class="cp-status {cls}">{chk}{label}</span>'
            f'<b>{esc(channel)}</b>'
            f'<a href="{esc(url)}" target="_blank" rel="noopener">{icon("play", 17, MUTED)}<span>{esc(url)}</span>'
            f'{icon("external", 15)}</a><span class="sep"></span>'
            f'<span>{icon("calendar", 16, MUTED)} Analyzed {esc(analyzed)}</span></div>')


def kpi_cards_html(items) -> str:
    """items: (icon_name, value, label)."""
    cells = []
    for ic, value, label in items:
        bg, fg = KPI_STYLE[ic]
        cells.append(f'<div class="cp-kpi2"><div class="ic" style="background:{bg}">{icon(ic, 24, fg)}</div>'
                     f'<div><b>{esc(value)}</b><span>{esc(label)}</span></div></div>')
    return f'<div class="cp-kpis2">{"".join(cells)}</div>'


def card_head_html(ic: str, title: str, sub: str) -> str:
    bg, fg = KPI_STYLE.get(ic, ("#EEF0FF", INDIGO))
    return (f'<div class="cp-cardhead"><div class="ic" style="background:{bg}">{icon(ic, 20, fg)}</div>'
            f'<div><h3>{esc(title)}</h3><p>{esc(sub)}</p></div></div>')


def _nice_max(v: float) -> tuple:
    """Axis maximum and tick step giving 4-8 round ticks."""
    import math
    if v <= 0:
        return 1, 1
    for step in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]:
        top = math.ceil(v * 1.05 / step) * step
        if top / step <= 6:
            return top, step
    return v, v


_CMD = re.compile(r"(?<![\w-])([a-z][\w.]*(?:\s+-{1,2}[\w-]+)+)")


def fmt_name(name: str) -> str:
    """Escape a topic name / title and show command-like parts (ls -a, git push -u) in a code font,
    so "ls" can't be misread as "Is"."""
    return _CMD.sub(lambda m: f'<code class="cp-cmd">{m.group(1)}</code>', esc(name))


DASH_CSS = f"""
<style>
.cp-meta {{ display:flex; flex-wrap:wrap; align-items:center; gap:.5rem 1.1rem; color: var(--muted); font-size:.95rem; }}
.cp-meta .cp-status {{ display:inline-flex; align-items:center; gap:.35rem; padding:.25rem .7rem; }}
.cp-meta .cp-status svg {{ color:#16A34A; }}
.cp-meta b {{ color: var(--ink); font-weight:700; }}
.cp-meta a {{ display:inline-flex; align-items:center; gap:.35rem; color: var(--indigo); text-decoration: underline;
  text-underline-offset: 2px; word-break: break-all; }}
.cp-meta span {{ display:inline-flex; align-items:center; gap:.35rem; }}
.cp-meta .sep {{ width:1px; height:1.2rem; background: var(--line); }}
.cp-kpis2 {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: .9rem; margin: 1.1rem 0 1rem; }}
.cp-kpi2 {{ display:flex; align-items:center; gap:.7rem; background:#fff; border:1px solid var(--line); border-radius:16px;
  padding: .95rem .9rem; box-shadow: 0 2px 10px rgba(18,22,58,.03); }}
.cp-kpi2 .ic {{ width:42px; height:42px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }}
.cp-kpi2 > div:last-child {{ min-width: 0; }}
.cp-kpi2 b {{ display:block; font-weight:800; font-size:1.65rem; line-height:1.1; }}
[data-testid="stButtonGroup"] button p {{ font-size: .85rem; white-space: nowrap; }}
.cp-kpi2 span {{ display:block; color: var(--muted); font-size:.8rem; line-height:1.25; word-break: keep-all;
  overflow-wrap: normal; hyphens: none; }}
@media (max-width: 1100px) {{ .cp-kpis2 {{ grid-template-columns: repeat(3, minmax(0,1fr)); }} }}
@media (max-width: 700px) {{ .cp-kpis2 {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
[class*="st-key-card_"] {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding: 1.1rem 1.2rem 1rem;
  box-shadow: 0 2px 10px rgba(18,22,58,.03); }}
.cp-cardhead {{ display:flex; gap:.8rem; align-items:flex-start; }}
.cp-cardhead .ic {{ width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }}
.cp-cardhead h3 {{ font-size:1.18rem !important; font-weight:800; margin:0 !important; padding:0 !important; }}
.cp-cardhead p {{ color: var(--muted); font-size:.9rem; margin:.1rem 0 0; }}
.cp-bars {{ display:grid; grid-template-columns: max-content 1fr 6.2rem; column-gap: .9rem; row-gap: .95rem;
  align-items:center; margin-top: .9rem; }}
.cp-bars .lbl {{ text-align:right; color: var(--muted); font-size:.9rem; }}
.cp-bars .trk {{ height: 26px; background:#F1F2F8; border-radius: 6px; position:relative; overflow:hidden; }}
.cp-bars .trk i {{ position:absolute; left:0; top:0; bottom:0; border-radius:6px; }}
.cp-bars .val b {{ font-weight:700; }} .cp-bars .val span {{ color: var(--muted); font-size:.88rem; }}
.cp-bars .axis {{ position:relative; height:1.2rem; color: var(--muted); font-size:.8rem; margin-top:-.3rem; }}
.cp-bars .axis span {{ position:absolute; transform: translateX(-50%); }}
.cp-bars .cap {{ text-align:center; color: var(--muted); font-size:.85rem; margin-top:-.6rem; }}
.cp-ranks {{ display:grid; grid-template-columns: 26px max-content 1fr max-content; gap: .85rem .8rem; align-items:center;
  margin: .9rem 0 .5rem; }}
.cp-ranks .rk, .cp-recrow .rk {{ width:26px; height:26px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:.85rem; font-weight:700; }}
.cp-ranks .nm {{ font-weight:600; font-size:.95rem; max-width: 16rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.cp-ranks .trk {{ height:10px; background:#EEEFF6; border-radius:99px; overflow:hidden; min-width: 4rem; }}
.cp-ranks .trk i {{ display:block; height:100%; border-radius:99px; background: linear-gradient(90deg,#8B8CF6,#6D5DF2); }}
.cp-ranks .ct {{ color: var(--muted); font-size:.85rem; }}
.cp-recrow {{ display:flex; align-items:center; gap:.9rem; }}
.cp-recrow .rk {{ width:36px; height:36px; font-size:1rem; flex:none; }}
.cp-recrow b {{ display:block; font-weight:700; font-size:.97rem; }}
.cp-recrow span {{ color: var(--muted); font-size:.88rem; }}
[class*="st-key-recrow_"] {{ border:1px solid var(--line); border-radius:12px; padding:.6rem .8rem; background:#FCFCFE; }}
[class*="st-key-recopen_"] button {{ border:none; background:transparent; color: var(--muted); }}
[class*="st-key-recopen_"] button:hover {{ color: var(--indigo); background: var(--indigo-soft); }}
.st-key-card_recfocus {{ border: 2px solid var(--indigo) !important; box-shadow: 0 0 0 4px rgba(79,70,229,.10) !important; }}
[class*="st-key-card_rec"] {{ margin-bottom: .6rem; }}
code.cp-cmd {{ font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: .92em; background: #EEF0FA;
  color: var(--ink); padding: .05em .35em; border-radius: 5px; }}
[data-testid="stExpander"] details {{ background:#fff; border-radius:16px; border-color: var(--line); }}
</style>
"""


MIX_CSS = """
<style>
.cp-legend { display:flex; flex-wrap:wrap; gap: .4rem 1.2rem; color: var(--muted); font-size:.88rem; margin:.8rem 0 .4rem; }
.cp-legend span { display:inline-flex; align-items:center; gap:.4rem; }
.cp-legend i { width:12px; height:12px; border-radius:3px; display:inline-block; }
.cp-mix { display:grid; grid-template-columns: max-content 1fr max-content; gap: .7rem .9rem; align-items:center; }
.cp-mix .lbl { font-weight:600; font-size:.93rem; max-width: 16rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.cp-mix .stk { display:flex; gap:2px; height:18px; background:#F1F2F8; border-radius:5px; overflow:hidden; }
.cp-mix .stk i { display:block; height:100%; }
.cp-mix .val { color: var(--muted); font-size:.85rem; }
</style>
"""


# --------------------------------------------------------------------------
# Clickable rows (Overview, Topics) and the Comments Explorer
# --------------------------------------------------------------------------
# A clickable row is a keyed container "row_<id>" holding the row's HTML and an
# invisible full-size button "go_<id>" laid over it, so the whole row is one
# real (keyboard-focusable) button that runs a Streamlit callback.

PILL = {"Question": ("#E3EEFB", "#1F5FAF"), "Request": ("#FDEBE2", "#A9481B"),
        "Confusion/Problem": ("#E0F3EB", "#136B4C"), "Praise": ("#FCE9F1", "#A23D6C"),
        "Spam": ("#FDF1D6", "#7A5400"), "Other": ("#EEF0F4", "#4B5469")}
CHEVRON = _svg('<path d="m9 6 6 6-6 6"/>', 18, "currentColor")


def slug(s: str) -> str:
    return re.sub(r"\W+", "_", str(s)).strip("_")


def pill_html(cat: str) -> str:
    bg, fg = PILL.get(cat, ("#EEF0F4", "#4B5469"))
    return f'<span class="cp-pill" style="background:{bg};color:{fg}">{esc(cat)}</span>'


def cat_row_html(cat: str, n: int, pct: float, top: float) -> str:
    w = 0 if not top else 100 * n / top
    return (f'<div class="cp-catrow"><div class="lbl">{esc(cat)}</div>'
            f'<div class="trk"><i style="width:{w:.2f}%;background:{CATEGORY_COLORS.get(cat, INDIGO)}"></i></div>'
            f'<div class="val"><b>{n}</b> <span>({pct:.0f}%)</span></div><div class="chev">{CHEVRON}</div></div>')


def cat_axis_html(top: float, step: float) -> str:
    ticks = "".join(f'<span style="left:{100 * t / top:.2f}%">{t}</span>' for t in range(0, int(top) + 1, int(step)))
    return (f'<div class="cp-cataxis"><div></div><div class="axis">{ticks}</div><div></div><div></div></div>'
            '<div class="cp-catcap">Comments</div>')


def topic_row_html(i: int, name: str, n: int, mx: int) -> str:
    bg, fg = RANK_STYLE[min(i, 3)]
    return (f'<div class="cp-toprow"><div class="rk" style="background:{bg};color:{fg}">{i + 1}</div>'
            f'<div class="nm" title="{esc(name)}">{fmt_name(name)}</div>'
            f'<div class="trk"><i style="width:{100 * n / max(mx, 1):.1f}%"></i></div>'
            f'<div class="ct">{n} unique comment{"s" if n != 1 else ""}</div></div>')


def opp_row_html(i: int, r: dict) -> str:
    bg, fg = RANK_STYLE[min(i, 3)]
    people = r["n_unique_commenters"] if r.get("n_unique_commenters") is not None else r["n_unique_comments"]
    return (f'<div class="cp-opprow"><div class="rk" style="background:{bg};color:{fg}">{r["rank"]}</div>'
            f'<div class="tx"><b title="{esc(r["suggested_title"])}">{fmt_name(r["suggested_title"])}</b>'
            f'<span>Priority {r["priority_score"]:.1f} · {people} people · {r["n_unique_comments"]} comments</span></div>'
            f'<div class="chev">{CHEVRON}</div></div>')


def rel_time(ts, now=None) -> str:
    """'2 days ago' style age of a timestamp (pandas Timestamp, UTC)."""
    if ts is None or pd.isna(ts):
        return ""
    now = now or pd.Timestamp.now(tz="UTC")
    s = max(0, int((now - ts).total_seconds()))
    for unit, secs in (("year", 31536000), ("month", 2592000), ("week", 604800), ("day", 86400), ("hour", 3600),
                       ("minute", 60)):
        if s >= secs:
            k = s // secs
            return f"{k} {unit}{'s' if k != 1 else ''} ago"
    return "just now"


def recent_table_html(df: pd.DataFrame, now=None) -> str:
    rows = []
    for r in df.itertuples():
        topic = r.topic_name if r.topic_name and r.topic_name != "Unclustered" else ""
        rows.append(f'<tr><td class="tx" title="{esc(r.text)}">{esc(r.text)}</td><td>{pill_html(r.predicted_category)}</td>'
                    f'<td class="tp">{fmt_name(topic) if topic else "<span class=cp-muted>—</span>"}</td>'
                    f'<td class="dt">{esc(rel_time(r.published_dt, now))}</td></tr>')
    return ('<table class="cp-recent"><thead><tr><th>Comment</th><th>Category</th><th>Topic</th><th>Posted</th></tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table>')


def rec_head_html(r: dict) -> str:
    """Compact recommendation row: rank, title, topic/format, priority, evidence."""
    bg, fg = RANK_STYLE[min(r["rank"] - 1, 3)]
    people = r["n_unique_commenters"] if r.get("n_unique_commenters") is not None else r["n_unique_comments"]
    return (f'<div class="cp-rechead"><div class="rk" style="background:{bg};color:{fg}">{r["rank"]}</div>'
            f'<div class="tx"><b>{fmt_name(r["suggested_title"])}</b><span>{esc(r["_meta"])}</span></div>'
            f'<div class="pri"><b>{r["priority_score"]:.0f}</b><span>priority / 100</span></div>'
            f'<div class="ev"><b>{people}</b><span>people</span></div>'
            f'<div class="ev"><b>{r["n_unique_comments"]}</b><span>comments</span></div></div>')


OVERVIEW_CSS = """
<style>
.cp-head h1 { font-size: clamp(1.5rem, 2.2vw, 1.9rem) !important; margin-bottom: .35rem; }
.cp-meta { font-size: .88rem; gap: .35rem .9rem; }
.cp-meta a span { max-width: 26rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; }
.cp-kpis2 { grid-template-columns: repeat(4, minmax(0,1fr)); gap: .8rem; margin: .9rem 0 .8rem; }
.cp-kpi2 { padding: .8rem .9rem; }
.cp-kpi2 b { font-size: 1.5rem; }
@media (max-width: 1100px) { .cp-kpis2 { grid-template-columns: repeat(2, minmax(0,1fr)); } }
[class*="st-key-card_"] { padding: .95rem 1.05rem .9rem; }
.cp-cardhead h3 { font-size: 1.05rem !important; }
.cp-cardhead p { font-size: .84rem; }
.cp-cardhead .ic { width: 36px; height: 36px; }

/* clickable rows: container row_* + invisible full-size button go_* */
[class*="st-key-row_"] { position: relative; gap: 0 !important; border-radius: 10px; transition: background .12s; }
[class*="st-key-row_"]:hover { background: #F3F3FE; }
[class*="st-key-row_"] [data-testid="stMarkdownContainer"], [class*="st-key-row_"] .stMarkdown,
[class*="st-key-row_"] [data-testid="stElementContainer"] { margin: 0 !important; }
[data-testid="stElementContainer"]:has(> .stMarkdown style) { display: none; }   /* CSS-only blocks take no space */
.st-key-card_cats, .st-key-card_topics, .st-key-card_recs, .st-key-card_topics_mix { gap: .3rem !important; }
.st-key-card_cats > div:nth-child(1), .st-key-card_topics > div:nth-child(1), .st-key-card_recs > div:nth-child(1),
.st-key-card_topics_mix > div:nth-child(1) { margin-bottom: .45rem; }
[class*="st-key-row_"]:hover .chev { color: var(--indigo); }
[class*="st-key-go_"] { position: absolute !important; inset: 0; z-index: 2; margin: 0 !important; width: 100% !important; }
[class*="st-key-go_"] .stButton, [class*="st-key-go_"] .stButton > div:first-child,
[class*="st-key-go_"] .stButton > div:first-child > *, [class*="st-key-go_"] button {
  display: block; width: 100% !important; height: 100% !important; min-height: 0 !important; }
[class*="st-key-go_"] button { opacity: 0; cursor: pointer; padding: 0; border: none; }
[class*="st-key-row_"]:has(button:focus-visible) { outline: 2px solid var(--indigo); outline-offset: 1px; }
.cp-catrow { display: grid; grid-template-columns: 8.6rem 1fr 5.4rem 1.2rem; gap: .8rem; align-items: center;
  padding: .5rem .55rem; }
.cp-catrow .lbl { color: var(--ink); font-size: .9rem; white-space: nowrap; }
.cp-catrow .trk { height: 22px; background: #F1F2F8; border-radius: 6px; position: relative; overflow: hidden; }
.cp-catrow .trk i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; }
.cp-catrow .val { font-size: .9rem; white-space: nowrap; } .cp-catrow .val b { font-weight: 700; }
.cp-catrow .val span { color: var(--muted); }
.chev { color: #B4B8CC; display: flex; align-items: center; }
.cp-cataxis { display: grid; grid-template-columns: 8.6rem 1fr 5.4rem 1.2rem; gap: .8rem; padding: .2rem .55rem 0; }
.cp-cataxis .axis { position: relative; height: 1.1rem; color: var(--muted); font-size: .76rem;
  border-top: 1px solid #E6E8F0; padding-top: .15rem; }
.cp-cataxis .axis span { position: absolute; transform: translateX(-50%); }
.cp-catcap { text-align: center; color: var(--muted); font-size: .78rem; margin: .1rem 0 .1rem; }
.cp-toprow { display: grid; grid-template-columns: 24px minmax(6rem, max-content) minmax(2.5rem, 1fr) max-content; gap: .7rem; align-items: center;
  padding: .32rem .45rem; }
.cp-toprow .rk, .cp-opprow .rk, .cp-rechead .rk { width: 24px; height: 24px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-size: .8rem; font-weight: 700; flex: none; }
.cp-toprow .nm { font-weight: 600; font-size: .9rem; max-width: 14rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cp-toprow .trk { height: 9px; background: #EEEFF6; border-radius: 99px; overflow: hidden; }
.cp-toprow .trk i { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg,#9A9BF7,#6D5DF2); }
.cp-toprow .ct { color: var(--muted); font-size: .8rem; text-align: right; white-space: nowrap; }
.cp-opprow { display: grid; grid-template-columns: 30px minmax(0,1fr) 1.2rem; gap: .75rem; align-items: center;
  padding: .45rem .45rem; border-bottom: 1px solid #EEF0F5; }
.cp-opprow .rk { width: 30px; height: 30px; font-size: .9rem; }
.cp-opprow .tx b { display: block; font-weight: 600; font-size: .9rem; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
.cp-opprow .tx span { color: var(--muted); font-size: .8rem; }
[class*="st-key-hlink_"] button { border: none; background: transparent; color: var(--indigo); padding: .1rem .3rem;
  min-height: 0; justify-content: flex-end; }
[class*="st-key-hlink_"] button p { color: var(--indigo); font-weight: 600; font-size: .85rem; }
[class*="st-key-hlink_"] button:hover { background: var(--indigo-soft); }
[class*="st-key-hlink_"] { display: flex; justify-content: flex-end; }
.cp-recent { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: .87rem; margin-top: .5rem; }
.cp-recent th { text-align: left; font-weight: 600; color: var(--muted); font-size: .8rem; background: #F7F8FC;
  padding: .45rem .6rem; border-bottom: 1px solid var(--line); }
.cp-recent th:nth-child(1) { width: 52%; } .cp-recent th:nth-child(2) { width: 16%; }
.cp-recent th:nth-child(3) { width: 18%; } .cp-recent th:nth-child(4) { width: 14%; }
.cp-recent td { padding: .45rem .6rem; border-bottom: 1px solid #EEF0F5; vertical-align: middle; }
.cp-recent td.tx, .cp-recent td.tp { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cp-recent td.dt { color: var(--muted); white-space: nowrap; }
.cp-pill { display: inline-block; padding: .12rem .55rem; border-radius: 6px; font-size: .78rem; font-weight: 600;
  white-space: nowrap; }
@media (max-width: 700px) { .cp-recent th:nth-child(3), .cp-recent td:nth-child(3) { display: none; }
  .cp-catrow, .cp-cataxis { grid-template-columns: 6.8rem 1fr 4.6rem 1rem; } }

/* Comments Explorer */
[class*="st-key-chip_"] button { border-radius: 999px; border: 1px solid #C9CDF8; background: var(--indigo-soft);
  padding: .1rem .75rem; min-height: 0; }
[class*="st-key-chip_"] button p { color: var(--indigo); font-weight: 600; font-size: .84rem; }
[class*="st-key-chip_"] button:hover { background: #DFE2FF; border-color: var(--indigo); }
.st-key-chips { flex-direction: row !important; flex-wrap: wrap; gap: .45rem !important; align-items: center; }
.st-key-chips > div { width: auto !important; flex: none !important; }
.cp-count { font-size: .92rem; } .cp-count b { font-weight: 700; }
.st-key-clear_filters { margin-bottom: .7rem; }
/* Narrow windows: the Overview's two columns stack instead of squeezing */
@media (max-width: 1150px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_cats) { flex-direction: column; }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_cats) > [data-testid="stColumn"] {
    width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
}

/* Recommendations */
.cp-rechead { display: grid; grid-template-columns: 30px minmax(0,1fr) 6.2rem 4.4rem 5rem; gap: .9rem; align-items: center; }
.cp-rechead .rk { width: 30px; height: 30px; font-size: .9rem; }
.cp-rechead .tx b { display: block; font-weight: 700; font-size: .98rem; }
.cp-rechead .tx span { color: var(--muted); font-size: .82rem; }
.cp-rechead .pri, .cp-rechead .ev { text-align: center; }
.cp-rechead .pri b { display: block; color: var(--indigo); font-weight: 800; font-size: 1.25rem; line-height: 1.1; }
.cp-rechead .ev b { display: block; font-weight: 700; font-size: 1.1rem; line-height: 1.1; }
.cp-rechead span { color: var(--muted); font-size: .74rem; }
@media (max-width: 700px) { .cp-rechead { grid-template-columns: 30px minmax(0,1fr) 4.4rem; }
  .cp-rechead .ev { display: none; } }
[class*="st-key-card_rec"] [data-testid="stExpander"] details { border-radius: 10px; }
</style>
"""


def mix_legend_html() -> str:
    legend = "".join(f'<span><i style="background:{c}"></i>{esc(k)}</span>' for k, c in INTENT_COLORS.items())
    return f'<div class="cp-legend">{legend}</div>'


def mix_row_html(r, mx: int) -> str:
    """One topic: stacked Question / Request / Problem bar + counts (a clickable row on the Topics page)."""
    parts = [("Question", r.n_question), ("Request", r.n_request), ("Confusion/Problem", r.n_confusion_problem)]
    segs = "".join(f'<i style="width:{100 * v / max(mx, 1):.2f}%;background:{INTENT_COLORS[k]}" title="{esc(k)}: {v}"></i>'
                   for k, v in parts if v)
    word = {"Question": "question", "Request": "request", "Confusion/Problem": "problem"}
    desc = ", ".join(f"{v} {word[k]}{'s' if v != 1 else ''}" for k, v in parts if v)
    return (f'<div class="cp-mixrow"><div class="lbl" title="{esc(r.topic_name)}">{fmt_name(r.topic_name)}</div>'
            f'<div class="stk">{segs}</div><div class="val">{esc(desc)}</div><div class="chev">{CHEVRON}</div></div>')


MIXROW_CSS = """
<style>
.cp-mixrow { display:grid; grid-template-columns: minmax(0, 13rem) 1fr 11rem 1.2rem; gap: .9rem; align-items:center;
  padding: .42rem .55rem; }
.cp-mixrow .lbl { font-weight:600; font-size:.92rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.cp-mixrow .stk { display:flex; gap:2px; height:16px; background:#F1F2F8; border-radius:5px; overflow:hidden; }
.cp-mixrow .stk i { display:block; height:100%; }
.cp-mixrow .val { color: var(--muted); font-size:.84rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
@media (max-width: 700px) { .cp-mixrow { grid-template-columns: minmax(0, 8rem) 1fr 1rem; } .cp-mixrow .val { display:none; } }
</style>
"""


# --------------------------------------------------------------------------
# Overview v3 (grid of equal-height cards, as in the reference design)
# --------------------------------------------------------------------------

# Category colors as in the reference: problems red, praise green.
CATEGORY_COLORS.update({"Confusion/Problem": "#e5484d", "Praise": "#22a06b"})
INTENT_COLORS["Confusion/Problem"] = "#e5484d"
PILL.update({"Confusion/Problem": ("#FDE8E8", "#B42318"), "Praise": ("#E3F5EC", "#137A4B")})

KPI_TINT = {"chat": ("linear-gradient(135deg,#F3EFFF,#FBF9FF)", "#E2DAFB", "#EAE3FF", "#6D4AE6"),
            "play": ("linear-gradient(135deg,#EAF2FF,#F7FAFF)", "#D6E4FB", "#DDE9FD", "#2563EB"),
            "layers": ("linear-gradient(135deg,#E6F7EF,#F6FCF9)", "#CDEBDC", "#D6F2E4", "#16A34A"),
            "bulb": ("linear-gradient(135deg,#FFF5E1,#FFFBF2)", "#F5E3BC", "#FDEDC8", "#D97706")}

HERO_ART = """<svg class="cp-hero-art" width="230" height="96" viewBox="0 0 230 96" aria-hidden="true">
<defs><linearGradient id="hg1" x1="0" x2="1"><stop offset="0" stop-color="#8B7CF6"/><stop offset="1" stop-color="#5B4AE8"/></linearGradient>
<linearGradient id="hg2" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#A5B4FC"/><stop offset="1" stop-color="#E0E7FF"/></linearGradient></defs>
<circle cx="150" cy="10" r="70" fill="#EEF0FF" opacity=".7"/>
<rect x="40" y="14" width="88" height="70" rx="14" fill="#fff" stroke="#E6E8F5"/>
<rect x="58" y="30" width="52" height="38" rx="10" fill="url(#hg1)"/><path d="M78 40v18l15-9z" fill="#fff"/>
<rect x="150" y="56" width="14" height="28" rx="3" fill="url(#hg2)"/><rect x="170" y="40" width="14" height="44" rx="3" fill="url(#hg2)"/>
<rect x="190" y="22" width="14" height="62" rx="3" fill="url(#hg2)"/></svg>"""


def hero_html(title: str, sub: str) -> str:
    return f'<div class="cp-hero"><div><h1>{esc(title)}</h1><p>{esc(sub)}</p></div>{HERO_ART}</div>'


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", str(name or "").strip()) if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def metabar_html(status: str, channel: str, url: str, analyzed: str, avatar: str = "") -> str:
    short = re.sub(r"^https?://(www[.])?", "", str(url))
    cls, label = STATUS_STYLE.get(status, ("run", status))
    chk = icon("check", 15) if cls == "ok" else ""
    return (f'<div class="cp-metabar"><span class="cp-status {cls}">{chk}{label}</span>'
            + (f'<span class="ch"><i class="av pic" style="--av:url(&quot;{esc(avatar)}&quot;)" '
               f'title="{esc(channel)}">{esc(_initials(channel))}</i>{esc(channel)}</span>' if avatar.startswith("https://")
               else f'<span class="ch"><i class="av">{esc(_initials(channel))}</i>{esc(channel)}</span>')
            + f'<a href="{esc(url)}" target="_blank" rel="noopener" title="Open on YouTube">{yt_icon(22)}'
            f'<span>{esc(short)}</span>'
            f'{icon("external", 14)}</a>'
            f'<span class="dt">{icon("calendar", 16, MUTED)}Analyzed {esc(analyzed)}</span></div>')


def kpi_tiles_html(items) -> str:
    """items: (icon_name, value, label) — tinted cards as in the reference."""
    cells = []
    for ic, value, label in items:
        bg, border, circle, fg = KPI_TINT[ic]
        cells.append(f'<div class="cp-tile" style="background:{bg};border-color:{border}">'
                     f'<div class="ic" style="background:{circle}">{icon(ic, 30, fg)}</div>'
                     f'<div><b>{esc(value)}</b><span>{esc(label)}</span></div></div>')
    return f'<div class="cp-tiles">{"".join(cells)}</div>'


def head3_html(ic: str, title: str, sub: str) -> str:
    bg, fg = {"chat": ("#EEF0FF", "#4F46E5"), "layers": ("#F1EAFE", "#6D28D9"), "bulb": ("#FFF4DB", "#D97706")}.get(
        ic, ("#EEF0FF", INDIGO))
    return (f'<div class="cp-h3"><div class="ic" style="background:{bg}">{icon(ic, 22, fg)}</div>'
            f'<div><h3>{esc(title)}</h3><p>{esc(sub)}</p></div></div>')


def cat_row3_html(cat: str, n: int, pct: float, top: float) -> str:
    color = CATEGORY_COLORS.get(cat, INDIGO)
    w = 0 if not top else 100 * n / top
    return (f'<div class="cp-r3 cat"><i class="dot" style="background:{color}"></i><div class="lbl">{esc(cat)}</div>'
            f'<div class="trk" data-tip="{n} comments · {pct:.1f}% of analyzed"><i style="width:{w:.2f}%;background:{color}"></i></div>'
            f'<div class="val"><b>{n}</b> ({pct:.0f}%)</div><div class="chev">{CHEVRON}</div></div>')


def topic_row3_html(i: int, name: str, n: int, mx: int) -> str:
    bg, fg = RANK_STYLE[min(i, 3)]
    return (f'<div class="cp-r3 top"><div class="rk" style="background:{bg};color:{fg}">{i + 1}</div>'
            f'<div class="nm" title="{esc(name)}">{fmt_name(name)}</div>'
            f'<div class="ct">{n} comment{"s" if n != 1 else ""}</div>'
            f'<div class="trk" data-tip="{n} unique comment{"s" if n != 1 else ""} · click to read them">'
            f'<i style="width:{100 * n / max(mx, 1):.1f}%"></i></div><div class="chev">{CHEVRON}</div></div>')


def opp_row3_html(i: int, r: dict) -> str:
    bg, fg = RANK_STYLE[min(i, 3)]
    return (f'<div class="cp-r3 opp"><div class="rk" style="background:{bg};color:{fg}">{r["rank"]}</div>'
            f'<div class="tx"><b title="{esc(r["suggested_title"])}">{fmt_name(r["suggested_title"])}</b>'
            f'<span>{r["n_unique_comments"]} comments &nbsp;•&nbsp; Priority {r["priority_score"]:.1f}</span></div>'
            f'<div class="chev">{CHEVRON}</div></div>')


SIDE_NOTE = (f'<div class="cp-sidenote"><div class="ic">{_svg(ICON_BULB_PATH, 20, "#D97706")}</div>'
             '<b>Better content starts with your audience.</b>'
             '<span>Analyze comments, discover what they want, and create videos that matter.</span></div>')


GRID_CSS = """
<style>
/* Wider page, less empty space */
[data-testid="stMainBlockContainer"].block-container, .block-container { max-width: 1480px !important;
  padding: 1rem 1.6rem 1.2rem !important; }
[data-testid="stAppViewContainer"] { background: radial-gradient(1200px 380px at 85% -60px, #ECEBFF 0%, rgba(236,235,255,0) 70%), var(--bg); }

/* Hero */
.cp-hero { display:flex; justify-content:space-between; align-items:flex-start; position:relative; min-height: 70px; }
.cp-hero h1 { font-size: 2rem !important; font-weight: 800; margin: 0 !important; padding: 0 !important; line-height: 1.15; }
.cp-hero p { color: var(--muted); margin: .2rem 0 0; font-size: .98rem; }
.cp-hero-art { margin-top: -14px; margin-right: 6px; flex: none; }
@media (max-width: 900px) { .cp-hero-art { display:none; } }

/* Meta strip */
.cp-metabar { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem 1.4rem; font-size:.88rem; color: var(--muted);
  background: rgba(255,255,255,.85); border: 1px solid var(--line); border-radius: 14px; padding: .55rem .9rem;
  margin: .2rem 0 .1rem; min-width: 0; }
.cp-metabar .cp-status { display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .75rem; }
.cp-metabar .cp-status svg { color:#16A34A; }
.cp-metabar .ch { display:inline-flex; align-items:center; gap:.5rem; color: var(--ink); font-weight:600; }
.cp-metabar .ch i { font-style:normal; width:28px; height:28px; border-radius:50%; background:#E8E9FD; color: var(--indigo);
  display:inline-flex; align-items:center; justify-content:center; font-size:.72rem; font-weight:800; }
.cp-metabar a { display:inline-flex; align-items:center; gap:.4rem; color: var(--indigo); min-width:0; }
.cp-metabar a span { text-decoration: underline; text-underline-offset: 2px; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; max-width: 21rem; }
.cp-metabar .dt { display:inline-flex; align-items:center; gap:.4rem; }
.cp-metabar .ch i.av { position: relative; overflow: hidden; width: 32px; height: 32px; box-shadow: 0 0 0 2px #fff, 0 0 0 3px #E0E3FA; }
.cp-metabar .ch i.av.pic::after { content: ""; position: absolute; inset: 0; border-radius: 50%;
  background: var(--av) center / cover no-repeat; animation: cpFade .4s both; }

/* KPI tiles */
.cp-tiles { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1rem; margin: .3rem 0 .2rem; }
.cp-tile { display:flex; align-items:center; gap: .85rem; border:1px solid; border-radius: 16px; padding: 1rem 1rem;
  box-shadow: 0 2px 10px rgba(18,22,58,.03); }
.cp-tile .ic { width:54px; height:54px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.cp-tile b { display:block; font-size: 1.85rem; font-weight: 800; line-height: 1.1; }
.cp-tile > div:last-child { min-width: 0; }
.cp-tile span { display:block; color: #3F4560; font-size: .9rem; margin-top: .15rem; line-height: 1.25; }
@media (max-width: 1100px) { .cp-tiles { grid-template-columns: repeat(2, minmax(0,1fr)); } }

/* Equal-height cards in each row */
.st-key-grid1, .st-key-grid2 { margin-top: .1rem; }
.st-key-grid1 [data-testid="stHorizontalBlock"], .st-key-grid2 [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-grid1 [data-testid="stColumn"] > div, .st-key-grid2 [data-testid="stColumn"] > div { height: 100%; }
.st-key-grid1 [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
.st-key-grid2 [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] { flex: 1 1 auto; }
.st-key-grid1 [class*="st-key-card_"], .st-key-grid2 [class*="st-key-card_"] { height: 100%; justify-content: flex-start; }
.st-key-card_cats, .st-key-card_topics, .st-key-card_recs, .st-key-card_recent { gap: 0 !important;
  padding: 1rem 1.1rem .7rem !important; }
.st-key-card_cats > div:nth-child(1), .st-key-card_topics > div:nth-child(1), .st-key-card_recs > div:nth-child(1),
.st-key-card_recent > div:nth-child(1) { margin-bottom: .7rem; }
.st-key-grid2 { margin-top: -.2rem; }

/* Card heading */
.cp-h3 { display:flex; gap:.8rem; align-items:center; padding-right: 6rem; margin-bottom: .5rem; }
[class*="st-key-card_"] { position: relative; }
[class*="st-key-card_"] [class*="st-key-hlink_"] { position: absolute !important; top: 1.35rem; right: 1rem; z-index: 3;
  width: auto !important; }
.cp-h3 .ic { width:44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.cp-h3 h3 { font-size: 1.12rem !important; font-weight: 800; margin: 0 !important; padding: 0 !important; }
.cp-h3 p { color: var(--muted); font-size: .84rem; margin: .1rem 0 0; }

/* Rows */
.cp-r3 { display:grid; align-items:center; gap: .8rem; padding: .5rem .6rem; min-height: 46px; border-bottom: 1px solid #EEF0F5; }
[class*="st-key-row_"]:last-of-type .cp-r3 { border-bottom: none; }
.cp-r3 .chev { color:#A8ADC2; display:flex; justify-content:flex-end; }
.cp-r3.cat { grid-template-columns: 16px 9.5rem minmax(3rem,1fr) 5.6rem 1.1rem; }
.cp-r3.cat .dot { width:16px; height:16px; border-radius:50%; display:block; }
.cp-r3.cat .lbl { font-size:.92rem; white-space:nowrap; }
.cp-r3 .trk { height: 12px; background:#EEF0F6; border-radius:99px; overflow:hidden; position:relative; }
.cp-r3 .trk i { display:block; height:100%; border-radius:99px; }
.cp-r3.cat .val { font-size:.9rem; color: var(--muted); white-space:nowrap; } .cp-r3.cat .val b { color: var(--ink); font-weight:700; }
.cp-r3 .rk { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:.85rem; font-weight:700; }
.cp-r3.top { grid-template-columns: 28px minmax(5rem, 11rem) 6.5rem minmax(3rem,1fr) 1.1rem; }
.cp-r3.top .nm { font-weight:600; font-size:.92rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.cp-r3.top .ct { color: var(--muted); font-size:.86rem; white-space:nowrap; }
.cp-r3.top .trk i { background: linear-gradient(90deg,#7C83F6,#5B4AE8); }
.cp-r3.opp { grid-template-columns: 36px minmax(0,1fr) 1.1rem; padding: .42rem .6rem; }
.cp-r3.opp .rk { width:36px; height:36px; font-size:.95rem; }
.cp-r3.opp .tx { min-width: 0; }
.cp-r3.opp .tx b { display:-webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow:hidden;
  font-weight:600; font-size:.92rem; line-height: 1.35; overflow-wrap: anywhere; }
.cp-r3.opp .tx span { color: var(--muted); font-size:.8rem; }
[class*="st-key-row_"] { border-radius: 10px; }
[class*="st-key-row_"]:hover { background: #F4F4FE; }
[class*="st-key-row_"]:hover .cp-r3 { border-bottom-color: transparent; }

/* Recent comments table */
.st-key-card_recent .cp-recent { margin-top: 0; font-size: .86rem; }
.st-key-card_recent .cp-recent th { background:#F5F6FA; border-bottom:none; }
.st-key-card_recent .cp-recent th:first-child { border-radius: 8px 0 0 8px; } .st-key-card_recent .cp-recent th:last-child { border-radius: 0 8px 8px 0; }
.st-key-card_recent .cp-recent td { padding: .7rem .5rem; overflow: hidden; }
.st-key-card_recent .cp-recent th { padding: .45rem .5rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.st-key-card_recent .cp-recent th:nth-child(1) { width: 47%; } .st-key-card_recent .cp-recent th:nth-child(2) { width: 22%; }
.st-key-card_recent .cp-recent th:nth-child(3) { width: 13%; } .st-key-card_recent .cp-recent th:nth-child(4) { width: 18%; }
.st-key-card_recent .cp-recent td.dt { white-space: normal; line-height: 1.25; font-size: .82rem; }
.st-key-card_recent .cp-recent .cp-pill { max-width: 100%; overflow: hidden; text-overflow: ellipsis; vertical-align: middle; }
.st-key-card_recent [data-testid="stMarkdownContainer"] { overflow: hidden; }
@media (max-width: 1400px) {
  .st-key-card_recent .cp-recent th:nth-child(3), .st-key-card_recent .cp-recent td:nth-child(3) { display: none; }
  .st-key-card_recent .cp-recent th:nth-child(1) { width: 52%; } .st-key-card_recent .cp-recent th:nth-child(2) { width: 26%; }
  .st-key-card_recent .cp-recent th:nth-child(4) { width: 22%; } }

/* Sidebar note card, shown only when the window is tall enough (the sidebar never scrolls) */
[data-testid="stSidebarContent"] { position: relative; }
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.cp-sidenote) { position: static; }
.cp-sidenote { position: absolute; bottom: 18px; left: 16px; right: 16px; background: linear-gradient(160deg,#fff,#F3F1FF);
  border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1rem 1.1rem; }
.cp-sidenote .ic { width:36px; height:36px; border-radius:50%; background:#FFF4DB; display:flex; align-items:center;
  justify-content:center; margin-bottom:.6rem; }
.cp-sidenote b { display:block; font-weight:800; font-size:1.02rem; line-height:1.3; margin-bottom:.4rem; }
.cp-sidenote span { color: var(--muted); font-size:.84rem; line-height:1.5; }
@media (max-height: 760px) { .cp-sidenote { display:none; } }

@media (max-width: 1150px) {
  .st-key-grid1 [data-testid="stHorizontalBlock"], .st-key-grid2 [data-testid="stHorizontalBlock"] { flex-direction: column; }
  .st-key-grid1 [data-testid="stColumn"], .st-key-grid2 [data-testid="stColumn"] { width:100% !important; flex: 1 1 100% !important;
    min-width: 100% !important; }
}
</style>
"""


# --------------------------------------------------------------------------
# Recommendations page (expandable cards)
# --------------------------------------------------------------------------

_I.update({
    "users": '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M18 14a6 6 0 0 1 3.5 6"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "msg": '<path d="M4 5h16v11H9l-5 4z"/><path d="M8 9h8M8 12h5"/>',
    "chart": '<path d="M4 20V4M4 20h16"/><path d="M8 16v-4M12 16V8M16 16v-6"/>',
    "up": '<path d="m6 15 6-6 6 6"/>', "down": '<path d="m6 9 6 6 6-6"/>',
})
SCORE_TILES = [("frequency", "users", "Frequency", 40, "How many different people raised it", "#EEEBFF", "#6D4AE6", "#7C6BF2"),
               ("recurrence", "play", "Recurrence", 25, "How many videos it appears in", "#E3F5EC", "#16A34A", "#22A06B"),
               ("recency", "clock", "Recency", 15, "How recent the latest comment is", "#FFF1E3", "#D97706", "#F59E0B"),
               ("intent", "msg", "Intent", 20, "Problems weigh most, then requests, then questions", "#F3E8FF",
                "#9333EA", "#4F46E5")]


def rec_card_head_html(r: dict, meta: str, is_open: bool) -> str:
    bg, fg = RANK_STYLE[min(r["rank"] - 1, 3)]
    people = r["n_unique_commenters"] if r.get("n_unique_commenters") is not None else r["n_unique_comments"]
    return (f'<div class="cp-rc-head"><div class="rk" style="background:{bg};color:{fg}">{r["rank"]}</div>'
            f'<div class="tx"><b>{fmt_name(r["suggested_title"])}</b><span>{esc(meta)}</span></div>'
            f'<div class="pri"><b>{r["priority_score"]:.0f}<small> / 100</small></b><span>Priority score</span></div>'
            f'<div class="num"><b>{people}</b><span>people</span></div>'
            f'<div class="num"><b>{r["n_unique_comments"]}</b><span>comments</span></div>'
            f'<div class="tog">{icon("up" if is_open else "down", 20)}</div></div>')


def rec_why_html(text: str, takeaway: str) -> str:
    return (f'<div class="cp-rc-why"><div class="why"><div class="h">{icon("chart", 22, "#6D4AE6")}<b>Why this idea?</b></div>'
            f'<p>{esc(text)}</p></div>'
            f'<div class="take">{icon("bulb", 22, "#6D4AE6")}<div><b>Key takeaway</b><p>{esc(takeaway)}</p></div></div></div>')


def rec_scores_html(r: dict) -> str:
    cells = []
    for key, ic, label, mx, why, bg, fg, bar in SCORE_TILES:
        pts = float(r["contributions"].get(key, 0))
        cells.append(f'<div class="t"><div class="ic" style="background:{bg}">{icon(ic, 22, fg)}</div><div class="bd">'
                     f'<b>{pts:.1f}<small> / {mx}</small></b><span class="l">{label}</span>'
                     f'<div class="bar"><i style="width:{min(100, 100 * pts / mx):.1f}%;background:{bar}"></i></div>'
                     f'<span class="w">{esc(why)}</span></div></div>')
    return (f'<div class="cp-rc-sec">{icon("chart", 20, INDIGO)}<b>Score breakdown</b></div>'
            f'<div class="cp-rc-scores">{"".join(cells)}</div>')


def rec_samples_html(samples) -> str:
    """samples: (text, category, age) of real comments."""
    cards = "".join(f'<div class="q"><p><i>&ldquo;</i>{esc(t)}</p><div class="ft">{pill_html(c)}<span>{esc(age)}</span></div></div>'
                    for t, c, age in samples)
    return f'<div class="cp-rc-samples">{cards}</div>'


RECS_CSS = """
<style>
[class*="st-key-reccard_"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:.55rem 1.1rem .55rem;
  box-shadow: 0 2px 10px rgba(18,22,58,.03); gap:.4rem !important; margin-bottom:.35rem; position:relative; }
.cp-rc-head { display:grid; grid-template-columns: 46px minmax(0,1fr) 7.6rem 5rem 5.6rem 1.6rem; gap:1rem; align-items:center;
  padding:.55rem .2rem; }
.cp-rc-head .rk { width:46px; height:46px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:1.15rem; }
.cp-rc-head .tx b { display:block; font-weight:800; font-size:1.08rem; line-height:1.3; }
.cp-rc-head .tx span { color: var(--muted); font-size:.86rem; }
.cp-rc-head .pri { background:#EEF0FF; border-radius:12px; padding:.45rem .6rem; text-align:center; }
.cp-rc-head .pri b { display:block; color: var(--indigo); font-weight:800; font-size:1.45rem; line-height:1.1; }
.cp-rc-head .pri b small { color:#3F4560; font-weight:500; font-size:.85rem; }
.cp-rc-head .pri span, .cp-rc-head .num span { color: var(--muted); font-size:.76rem; }
.cp-rc-head .num { text-align:center; } .cp-rc-head .num b { display:block; font-weight:800; font-size:1.25rem; }
.cp-rc-head .tog { color: var(--indigo); display:flex; justify-content:center; }
[class*="st-key-row_rechead_"]:hover { background: transparent; }
[class*="st-key-row_rechead_"]:hover .tx b { color: var(--indigo); }
.st-key-reccard_open .cp-rc-head { border-bottom: 1px solid #EEF0F5; padding-bottom: .9rem; }
.cp-rc-why { display:grid; grid-template-columns: minmax(0,1.9fr) minmax(0,1fr); gap:1.2rem; align-items:start; margin-top:.5rem; }
.cp-rc-why .h { display:flex; align-items:center; gap:.6rem; } .cp-rc-why .h b { font-size:1.08rem; font-weight:800; }
.cp-rc-why .why p { margin:.4rem 0 0 2.15rem; color:#2B3150; font-size:.93rem; line-height:1.6; }
.cp-rc-why .take { display:flex; gap:.8rem; background:#F4F2FF; border:1px solid #E6E1FB; border-radius:14px; padding:1rem 1.1rem; }
.cp-rc-why .take b { font-weight:700; } .cp-rc-why .take p { margin:.35rem 0 0; color: var(--muted); font-size:.88rem; line-height:1.5; }
.cp-rc-sec { display:flex; align-items:center; gap:.55rem; margin:.9rem 0 .6rem; } .cp-rc-sec b { font-weight:800; }
.cp-rc-scores { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.8rem; }
.cp-rc-scores .t { display:flex; gap:.8rem; border:1px solid var(--line); border-radius:14px; padding:.9rem .9rem; }
.cp-rc-scores .ic { width:48px; height:48px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.cp-rc-scores .bd { flex:1; min-width:0; }
.cp-rc-scores b { display:block; font-weight:800; font-size:1.2rem; } .cp-rc-scores b small { font-weight:500; font-size:.85rem; color:#3F4560; }
.cp-rc-scores .l { display:block; color: var(--muted); font-size:.85rem; }
.cp-rc-scores .bar { height:8px; border-radius:99px; background:#EEF0F6; overflow:hidden; margin:.55rem 0 .5rem; }
.cp-rc-scores .bar i { display:block; height:100%; border-radius:99px; }
.cp-rc-scores .w { display:block; color: var(--muted); font-size:.78rem; line-height:1.4; }
.cp-rc-samples { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.8rem; margin-bottom:.4rem; }
.cp-rc-samples .q { border:1px solid var(--line); border-radius:14px; padding:.9rem 1rem .8rem; display:flex; flex-direction:column;
  justify-content:space-between; min-width:0; }
.cp-rc-samples .q p { margin:0 0 .8rem; font-size:.9rem; line-height:1.5; overflow-wrap:anywhere; display:-webkit-box;
  -webkit-line-clamp:5; -webkit-box-orient:vertical; overflow:hidden; }
.cp-rc-samples .q p i { font-style:normal; color:#A5A9D6; font-size:1.5rem; line-height:0; margin-right:.35rem; vertical-align:-.45rem; }
.cp-rc-samples .ft { display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
.cp-rc-samples .ft span:last-child { color: var(--muted); font-size:.78rem; white-space:nowrap; }
[class*="st-key-samplehead_"] { position: relative; margin-top: .5rem; margin-bottom: .3rem; }
[class*="st-key-samplehead_"] [class*="st-key-recview_"] { position: absolute !important; right: 0; top: .95rem;
  width: auto !important; z-index: 2; }
[class*="st-key-samplehead_"] .cp-rc-sec { margin: 1.1rem 0 .2rem; }
[class*="st-key-reccard_"] { padding-bottom: .9rem !important; }
.st-key-reccard_open { padding-bottom: 1.1rem !important; }
.cp-rc-why .take > svg, .cp-rc-why .h > svg, .cp-rc-sec > svg { flex: none; width: 24px; height: 24px; }
[class*="st-key-recview_"] button { border:none; background:transparent; color: var(--indigo); padding:.1rem .3rem; min-height:0; }
[class*="st-key-recview_"] button p { color: var(--indigo); font-weight:600; font-size:.9rem; }
[class*="st-key-recview_"] button:hover { background: var(--indigo-soft); }
@media (max-width: 1150px) {
  .cp-rc-why { grid-template-columns: 1fr; }
  .cp-rc-scores, .cp-rc-samples { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .cp-rc-head { grid-template-columns: 40px minmax(0,1fr) 6.4rem 1.4rem; }
  .cp-rc-head .num { display:none; } }
</style>
"""


# --------------------------------------------------------------------------
# Motion layer, YouTube icons, live pipeline progress, success feedback
# --------------------------------------------------------------------------

def yt_icon(size: int = 20) -> str:
    """YouTube play-button mark, used next to YouTube links and inputs."""
    w, h = size, round(size * 0.72)
    return (f'<svg class="cp-yt" width="{w}" height="{h}" viewBox="0 0 28 20" aria-label="YouTube" role="img">'
            '<rect width="28" height="20" rx="5.5" fill="#FF0033"/><path d="M11.2 5.6v8.8l7.6-4.4z" fill="#fff"/></svg>')


VIDEO_TILE = (f'<div class="cp-yt-tile">{yt_icon(30)}</div>')

HERO_ART = """<svg class="cp-hero-art" width="230" height="96" viewBox="0 0 230 96" aria-hidden="true">
<defs><linearGradient id="hg2" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#A5B4FC"/><stop offset="1" stop-color="#E0E7FF"/></linearGradient>
<linearGradient id="hgr" x1="0" x2="1"><stop offset="0" stop-color="#FF4D6D"/><stop offset="1" stop-color="#FF0033"/></linearGradient></defs>
<circle cx="150" cy="10" r="70" fill="#EEF0FF" opacity=".7"/>
<g class="cp-hero-card"><rect x="40" y="14" width="88" height="70" rx="14" fill="#fff" stroke="#E6E8F5"/>
<rect x="58" y="30" width="52" height="38" rx="11" fill="url(#hgr)"/><path d="M78 40v18l15-9z" fill="#fff"/></g>
<rect class="cp-hb b1" x="150" y="56" width="14" height="28" rx="3" fill="url(#hg2)"/>
<rect class="cp-hb b2" x="170" y="40" width="14" height="44" rx="3" fill="url(#hg2)"/>
<rect class="cp-hb b3" x="190" y="22" width="14" height="62" rx="3" fill="url(#hg2)"/></svg>"""

_I.update({"cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>'})

# Pipeline phases as the creator thinks of them, mapped onto the real run stages.
PHASES = [("YouTube collection", ["validate", "collect"], "yt"),
          ("ML classification", ["preprocess", "classify"], "cpu"),
          ("Topic discovery", ["topics", "insights"], "layers"),
          ("Recommendations", ["recommend", "dashboard"], "bulb")]


def pipeline_progress(stages) -> float:
    """Share of the run's stages finished (a running stage counts half). Real progress, not a timer."""
    if not stages:
        return 0.0
    pts = sum(1.0 if s["status"] in ("done", "skipped") else 0.5 if s["status"] == "running" else 0.0 for s in stages)
    return pts / len(stages)


def progress_card_html(meta: dict) -> str:
    stages = meta["stages"]
    by = {s["key"]: s for s in stages}
    pct = round(100 * pipeline_progress(stages))
    done_n = sum(s["status"] in ("done", "skipped") for s in stages)
    current = next((s for s in stages if s["status"] == "running"), None)
    failed = any(s["status"] == "failed" for s in stages)
    cells = []
    for name, keys, ic in PHASES:
        ss_ = [by[k] for k in keys if k in by]
        stat = [s["status"] for s in ss_]
        if "failed" in stat:
            cls, chip = "failed", "Failed"
        elif stat and all(x in ("done", "skipped") for x in stat):
            cls, chip = "done", "Done"
        elif "running" in stat or ("done" in stat and "pending" in stat):
            cls, chip = "running", "Running"
        else:
            cls, chip = "pending", "Waiting"
        detail = next((s.get("detail") for s in reversed(ss_) if s.get("detail")), "") or (
            "Queued" if cls == "pending" else "")
        secs = sum(float(s.get("seconds") or 0) for s in ss_)
        tail = f'<em>{secs:.1f}s</em>' if cls == "done" and secs else ""
        mark = (icon("check", 15) if cls == "done" else '<i class="spin"></i>' if cls == "running" else
                icon("alert", 14) if cls == "failed" else "")
        art = yt_icon(26) if ic == "yt" else icon(ic, 22, INDIGO)
        cells.append(f'<div class="ph {cls}"><div class="top"><div class="pi">{art}</div>'
                     f'<span class="chip">{mark}{chip}</span></div><b>{esc(name)}</b>'
                     f'<span class="d" title="{esc(detail)}">{esc(detail)}</span>{tail}</div>')
    now = (f'<div class="now"><i class="pulse"></i><b>{esc(current["label"])}</b>'
           f'<span>{esc(current.get("detail") or "Working")}</span></div>') if current else ""
    url = re.sub(r"^https?://(www[.])?", "", str(meta.get("source_url", "")))
    return (f'<div class="cp-prog{" failed" if failed else ""}"><div class="hd"><div class="l">{yt_icon(30)}<div>'
            f'<b>Analyzing your video</b><span>{esc(url)}</span></div></div>'
            f'<div class="pc"><b>{pct}%</b><span>{done_n} of {len(stages)} stages</span></div></div>'
            f'<div class="bar" role="progressbar" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100">'
            f'<i style="width:{pct}%"></i></div>{now}<div class="phases">{"".join(cells)}</div></div>')


def success_html(n_comments: int, n_topics, n_recs) -> str:
    return (f'<div class="cp-success"><svg viewBox="0 0 52 52" width="34" height="34" aria-hidden="true">'
            f'<circle cx="26" cy="26" r="24" fill="#E7F8EF"/><path class="ck" d="M15 27l7 7 15-16" fill="none" '
            f'stroke="#16A34A" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            f'<div><b>Analysis complete</b><span>{n_comments} comments analyzed · {n_topics} topics · '
            f'{n_recs} video ideas</span></div></div>')


MOTION_CSS = """
<style>
:root { --grad: linear-gradient(135deg, #6366F1 0%, #7C3AED 100%); --grad-hover: linear-gradient(135deg, #5458E8 0%, #6D28D9 100%);
  --ease: cubic-bezier(.2,.7,.2,1); --shadow-sm: 0 2px 10px rgba(18,22,58,.04); --shadow-md: 0 10px 28px rgba(79,70,229,.12); }

/* ---------- Page entry: fade/slide in, staggered ---------- */
@keyframes cpFadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes cpFade { from { opacity: 0; } to { opacity: 1; } }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > * { animation: cpFadeUp .45s var(--ease) both; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(2) { animation-delay: .03s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(3) { animation-delay: .06s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(4) { animation-delay: .09s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(5) { animation-delay: .12s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(6) { animation-delay: .15s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(7) { animation-delay: .18s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(8) { animation-delay: .21s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(9) { animation-delay: .24s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(n+10) { animation-delay: .27s; }
.cp-tiles .cp-tile { animation: cpFadeUp .5s var(--ease) both; }
.cp-tiles .cp-tile:nth-child(2) { animation-delay: .06s; } .cp-tiles .cp-tile:nth-child(3) { animation-delay: .12s; }
.cp-tiles .cp-tile:nth-child(4) { animation-delay: .18s; }

/* ---------- Buttons ---------- */
.stApp button { transition: transform .18s var(--ease), box-shadow .2s var(--ease), background .2s, border-color .2s, color .2s; }
.stApp button[data-testid="stBaseButton-primary"], .st-key-analyze button {
  background: var(--grad) !important; border: none !important; color: #fff !important;
  box-shadow: 0 6px 18px rgba(99,102,241,.32); }
.stApp button[data-testid="stBaseButton-primary"] p, .stApp button[data-testid="stBaseButton-primary"] span { color: #fff !important; }
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover, .st-key-analyze button:hover {
  background: var(--grad-hover) !important; transform: translateY(-1px); box-shadow: 0 10px 24px rgba(99,102,241,.38); }
.stApp [data-testid="stMain"] button:active { transform: translateY(0) scale(.98); }
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover { border-color: #A5B4FC; color: var(--indigo);
  transform: translateY(-1px); box-shadow: 0 6px 16px rgba(79,70,229,.10); }
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover p { color: var(--indigo); }
.stApp button:disabled, .stApp button[disabled] { background: #E9EAF3 !important; color: #A2A7BD !important; box-shadow: none !important;
  transform: none !important; cursor: not-allowed; }
.stApp button:disabled p, .stApp button:disabled span { color: #A2A7BD !important; }
.st-key-analyze button:disabled, .st-key-analyze button[disabled] { background: #C9CBEF !important; box-shadow: none !important; }
.st-key-analyze button:disabled p, .st-key-analyze button:disabled span { color: #fff !important; }
.st-key-analyze button:disabled::after { display: none; }
.stApp button:focus-visible { outline: 3px solid #C7D2FE !important; outline-offset: 2px; }
.stApp button [data-testid="stIconMaterial"] { transition: transform .2s var(--ease); }
.stApp [data-testid="stMain"] button:hover [data-testid="stIconMaterial"] { transform: translateX(2px); }
.st-key-analyze button { position: relative; overflow: hidden; }
.st-key-analyze button::after { content: ""; position: absolute; inset: 0; background: linear-gradient(110deg, transparent 30%,
  rgba(255,255,255,.28) 50%, transparent 70%); transform: translateX(-120%); }
.st-key-analyze button:hover::after { transform: translateX(120%); transition: transform .8s var(--ease); }
/* Overlay row buttons stay invisible */
[class*="st-key-go_"] button, [class*="st-key-go_"] button:hover { transform: none !important; box-shadow: none !important;
  background: transparent !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] [class*="st-key-nav_"] button { position: relative; overflow: hidden;
  transition: background .2s var(--ease), color .2s, padding-left .2s var(--ease); }
[data-testid="stSidebar"] [class*="st-key-nav_"] button::before { content: ""; position: absolute; left: 0; top: 22%; bottom: 22%;
  width: 3px; border-radius: 0 3px 3px 0; background: var(--grad); transform: scaleY(0); transition: transform .25s var(--ease); }
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, #E8E9FD, #F1EEFF) !important; box-shadow: none !important; color: var(--indigo) !important; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] p { color: var(--indigo) !important; font-weight: 600; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"]::before { transform: scaleY(1); }
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover { padding-left: 1.05rem; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button [data-testid="stIconMaterial"] { color: #6B7194; transition: color .2s, transform .2s var(--ease); }
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover [data-testid="stIconMaterial"],
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] {
  color: var(--indigo); transform: scale(1.08); }
.cp-brand svg { transition: transform .35s var(--ease); } .cp-brand:hover svg { transform: rotate(-8deg) scale(1.06); }
.cp-sidenote { animation: cpFadeUp .6s .2s var(--ease) both; }

/* ---------- Cards ---------- */
[class*="st-key-card_"], [class*="st-key-reccard_"], .cp-tile, .cp-metabar, .st-key-url_card, .cp-rc-scores .t, .cp-rc-samples .q {
  transition: box-shadow .25s var(--ease), border-color .25s var(--ease), transform .25s var(--ease); }
[class*="st-key-card_"]:hover, [class*="st-key-reccard_"]:hover, .st-key-url_card:hover { box-shadow: var(--shadow-md) !important;
  border-color: #D9D6FB !important; }
.cp-tile:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
.cp-tile .ic svg { transition: transform .3s var(--ease); } .cp-tile:hover .ic svg { transform: scale(1.12) rotate(-4deg); }
.cp-rc-scores .t:hover, .cp-rc-samples .q:hover { transform: translateY(-2px); border-color: #D9D6FB; box-shadow: var(--shadow-md); }
.cp-h3 .ic svg { transition: transform .3s var(--ease); }
[class*="st-key-card_"]:hover .cp-h3 .ic svg { transform: scale(1.1); }
.cp-hero-art .cp-hb { transform-box: fill-box; transform-origin: bottom; animation: cpGrow .8s var(--ease) both; }
.cp-hero-art .b2 { animation-delay: .1s; } .cp-hero-art .b3 { animation-delay: .2s; }
@keyframes cpGrow { from { transform: scaleY(.15); } to { transform: scaleY(1); } }
.cp-hero-art .cp-hero-card { animation: cpFloat 5s ease-in-out infinite; }
@keyframes cpFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-3px); } }

/* ---------- Rows, bars, tooltips ---------- */
.cp-r3, .cp-rc-head { transition: transform .2s var(--ease); }
[class*="st-key-row_"]:hover .cp-r3 { transform: translateX(3px); }
[class*="st-key-row_"] .chev, .cp-rc-head .tog { transition: transform .2s var(--ease), color .2s; }
[class*="st-key-row_"]:hover .chev { transform: translateX(3px); color: var(--indigo); }
.cp-r3 .trk i, .cp-catrow .trk i, .cp-mixrow .stk i, .cp-rc-scores .bar i { animation: cpBar .9s var(--ease) both;
  transform-origin: left; transition: filter .2s; }
@keyframes cpBar { from { transform: scaleX(0); } to { transform: scaleX(1); } }
[class*="st-key-row_"]:hover .trk i { filter: brightness(1.08) saturate(1.15); }
.cp-r3 .trk[data-tip] { overflow: visible; }
.cp-r3 .trk[data-tip]::after { content: attr(data-tip); position: absolute; left: 50%; bottom: calc(100% + 8px);
  transform: translate(-50%, 4px); background: #1E1B4B; color: #fff; font-size: .75rem; padding: .3rem .55rem; border-radius: 6px;
  white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity .15s, transform .15s; z-index: 5; }
[class*="st-key-row_"]:hover .cp-r3 .trk[data-tip]::after { opacity: 1; transform: translate(-50%, 0); }
.cp-r3 .trk[data-tip] i { border-radius: 99px; }

/* ---------- Recommendation cards ---------- */
[class*="st-key-reccard_"] { animation: cpFadeUp .45s var(--ease) both; }
.st-key-reccard_open { border-color: #CFCBFA !important; box-shadow: 0 12px 32px rgba(79,70,229,.10) !important; }
.st-key-reccard_open .cp-rc-why, .st-key-reccard_open .cp-rc-sec, .st-key-reccard_open .cp-rc-scores,
.st-key-reccard_open .cp-rc-samples { animation: cpFadeUp .35s var(--ease) both; }
.st-key-reccard_open .cp-rc-scores { animation-delay: .05s; } .st-key-reccard_open .cp-rc-samples { animation-delay: .1s; }
.cp-rc-samples .q:nth-child(2) { animation-delay: .13s; } .cp-rc-samples .q:nth-child(3) { animation-delay: .16s; }
[class*="st-key-row_rechead_"]:hover .cp-rc-head .tog { transform: translateY(1px) scale(1.15); }
.cp-rc-head .pri { transition: transform .2s var(--ease); } [class*="st-key-row_rechead_"]:hover .cp-rc-head .pri { transform: scale(1.04); }

/* ---------- Filters ---------- */
.stApp [data-baseweb="tag"] { background: var(--grad) !important; border-radius: 8px !important; animation: cpFade .25s both; }
.stApp [data-baseweb="tag"] span { color: #fff !important; }
.stApp [data-baseweb="select"] > div, .stApp [data-testid="stTextInputRootElement"], .stApp [data-testid="stDateInputField"] {
  transition: border-color .2s, box-shadow .2s; }
.stApp [data-baseweb="select"] > div:hover, .stApp [data-testid="stTextInputRootElement"]:hover { border-color: #A5B4FC; }
.stApp [data-baseweb="select"] > div:focus-within, .stApp [data-testid="stTextInputRootElement"]:focus-within {
  border-color: var(--indigo) !important; box-shadow: 0 0 0 3px rgba(99,102,241,.16); }
[class*="st-key-chip_"] button { animation: cpFade .25s both; }
[class*="st-key-chip_"] button:hover { transform: translateY(-1px); }
.cp-pill { transition: transform .15s; } tr:hover .cp-pill { transform: scale(1.05); }
.cp-recent tbody tr { transition: background .15s; } .cp-recent tbody tr:hover { background: #F7F6FF; }

/* ---------- YouTube marks ---------- */
.cp-yt { flex: none; display: inline-block; vertical-align: middle; filter: drop-shadow(0 2px 4px rgba(255,0,51,.18)); }
.cp-yt-tile { width: 56px; height: 46px; border-radius: 12px; background: #FFF1F3; border: 1px solid #FFD5DC; display: flex;
  align-items: center; justify-content: center; transition: transform .25s var(--ease); }
.st-key-url_card:hover .cp-yt-tile { transform: scale(1.06) rotate(-3deg); }
.cp-metabar a .cp-yt { margin-right: .1rem; }

/* ---------- Live pipeline progress ---------- */
.cp-prog { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 1.2rem 1.3rem 1.1rem; margin-top: 1rem;
  box-shadow: 0 10px 30px rgba(79,70,229,.08); }
.cp-prog .hd { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.cp-prog .hd .l { display: flex; align-items: center; gap: .8rem; min-width: 0; }
.cp-prog .hd .l b { display: block; font-weight: 800; font-size: 1.1rem; }
.cp-prog .hd .l span { color: var(--muted); font-size: .85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block; }
.cp-prog .pc { text-align: right; } .cp-prog .pc b { display: block; font-size: 1.9rem; font-weight: 800; line-height: 1;
  background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.cp-prog .pc span { color: var(--muted); font-size: .8rem; }
.cp-prog .bar { height: 10px; border-radius: 99px; background: #EEF0F8; overflow: hidden; margin: .9rem 0 .8rem; }
.cp-prog .bar i { display: block; height: 100%; border-radius: 99px; background: var(--grad); position: relative; overflow: hidden;
  transition: width .6s var(--ease); }
.cp-prog .bar i::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, transparent, rgba(255,255,255,.45),
  transparent); animation: cpSheen 1.6s linear infinite; }
@keyframes cpSheen { from { transform: translateX(-100%); } to { transform: translateX(100%); } }
.cp-prog .now { display: flex; align-items: center; gap: .55rem; font-size: .9rem; margin-bottom: .9rem; min-width: 0; }
.cp-prog .now b { font-weight: 700; white-space: nowrap; } .cp-prog .now span { color: var(--muted); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.cp-prog .pulse { width: 9px; height: 9px; border-radius: 50%; background: #6366F1; flex: none; box-shadow: 0 0 0 0 rgba(99,102,241,.5);
  animation: cpPulse 1.4s ease-out infinite; }
@keyframes cpPulse { 0% { box-shadow: 0 0 0 0 rgba(99,102,241,.45); } 100% { box-shadow: 0 0 0 10px rgba(99,102,241,0); } }
.cp-prog .phases { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: .7rem; }
.cp-prog .ph { border: 1px solid var(--line); border-radius: 14px; padding: .8rem .85rem; background: #FBFBFE; min-width: 0;
  transition: border-color .3s, background .3s, box-shadow .3s; }
.cp-prog .ph .top { display: flex; justify-content: space-between; align-items: center; margin-bottom: .55rem; }
.cp-prog .ph .pi { width: 40px; height: 40px; border-radius: 12px; background: #EEF0FF; display: flex; align-items: center; justify-content: center; }
.cp-prog .ph b { display: block; font-weight: 700; font-size: .92rem; }
.cp-prog .ph .d { display: block; color: var(--muted); font-size: .78rem; margin-top: .15rem; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
.cp-prog .ph em { font-style: normal; color: #16A34A; font-size: .75rem; font-weight: 600; }
.cp-prog .chip { display: inline-flex; align-items: center; gap: .3rem; font-size: .72rem; font-weight: 700; padding: .18rem .55rem;
  border-radius: 99px; background: #EEF0F6; color: #6B7194; }
.cp-prog .ph.running { border-color: #C7C3FA; background: #F7F6FF; box-shadow: 0 6px 18px rgba(99,102,241,.12); }
.cp-prog .ph.running .chip { background: #E8E9FD; color: var(--indigo); }
.cp-prog .ph.done .chip { background: #DCFCE7; color: #166534; } .cp-prog .ph.done .pi { background: #E7F8EF; }
.cp-prog .ph.failed { border-color: #FCA5A5; background: #FFF5F5; } .cp-prog .ph.failed .chip { background: #FEE2E2; color: #991B1B; }
.cp-prog .spin { width: 11px; height: 11px; border-radius: 50%; border: 2px solid #C7D2FE; border-top-color: var(--indigo);
  animation: cpSpin .8s linear infinite; display: inline-block; }
@keyframes cpSpin { to { transform: rotate(360deg); } }
.cp-prog.failed .bar i { background: #F87171; } .cp-prog.failed .bar i::after { display: none; }
@media (max-width: 900px) { .cp-prog .phases { grid-template-columns: repeat(2, minmax(0,1fr)); } }

/* ---------- Success feedback ---------- */
.cp-success { display: flex; align-items: center; gap: .8rem; background: linear-gradient(90deg, #ECFDF3, #F5FFF9);
  border: 1px solid #BBF7D0; border-radius: 14px; padding: .6rem .9rem; margin: .5rem 0 .1rem;
  animation: cpPop .5s var(--ease) both, cpOut .6s 6s var(--ease) forwards; overflow: hidden; }
.cp-success b { display: block; font-weight: 700; color: #166534; } .cp-success span { color: #3F6B52; font-size: .85rem; }
.cp-success .ck { stroke-dasharray: 40; stroke-dashoffset: 40; animation: cpDraw .6s .25s var(--ease) forwards; }
@keyframes cpDraw { to { stroke-dashoffset: 0; } }
@keyframes cpPop { from { opacity: 0; transform: translateY(-6px) scale(.98); } to { opacity: 1; transform: none; } }
@keyframes cpOut { to { opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0; margin: 0; border-width: 0; } }

/* ---------- Respect reduced motion ---------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .cp-success { display: none; }
}
</style>
"""
