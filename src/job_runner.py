"""
CreatorPulse — background analysis jobs for the Streamlit app.

Streamlit re-runs the page script on every interaction and stops a running
script when the user clicks something. Running the analysis inside the page
script would therefore kill it half-way. Instead the app starts the analysis
in a worker thread; the thread only writes files (metadata.json and stage
outputs), and the page polls metadata.json to show live progress.

One process-wide JobRunner (created once via st.cache_resource) guarantees:
  * at most MAX_CONCURRENT analyses run at a time;
  * the same target cannot be analyzed twice concurrently;
  * re-runs of the page never start collection again.
"""

import threading
from pathlib import Path
from typing import Dict, Optional

from src.analysis import AnalysisFailed, AnalysisSettings, run_analysis, target_key
from src.runs import Run, resolve_dirs

MAX_CONCURRENT = 1


class BusyError(Exception):
    pass


class JobRunner:
    def __init__(self, runs_dir: Optional[Path] = None, raw_dir: Optional[Path] = None, youtube_factory=None):
        self._runs_dir, self._raw_dir = runs_dir, raw_dir     # None: resolved when a run starts
        self.youtube_factory = youtube_factory      # tests inject a fake client here
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}
        self._keys: Dict[str, str] = {}

    def active(self) -> Dict[str, str]:
        """run_id -> target key of analyses still running."""
        with self._lock:
            for rid in [r for r, t in self._threads.items() if not t.is_alive()]:
                self._threads.pop(rid, None)
                self._keys.pop(rid, None)
            return dict(self._keys)

    def is_running(self, run_id: str) -> bool:
        return run_id in self.active()

    def start(self, url: str, settings: AnalysisSettings, model_path=None) -> str:
        key = target_key(url)
        with self._lock:
            alive = {r: k for r, k in self._keys.items() if self._threads[r].is_alive()}
            if key in alive.values():
                raise BusyError("This link is already being analyzed. Its progress is shown below.")
            if len(alive) >= MAX_CONCURRENT:
                raise BusyError("Another analysis is still running. Wait for it to finish first.")
            runs_dir, raw_dir = resolve_dirs(self._runs_dir, self._raw_dir)
            run = Run.create(url.strip(), settings.as_dict(), runs_dir, raw_dir)
            yt = self.youtube_factory() if self.youtube_factory else None
            t = threading.Thread(target=self._work, args=(url, settings, run, yt, model_path, runs_dir, raw_dir),
                                 name=f"analysis-{run.run_id}", daemon=True)
            self._threads[run.run_id] = t
            self._keys[run.run_id] = key
            t.start()
        return run.run_id

    def _work(self, url, settings, run, yt, model_path, runs_dir, raw_dir):
        try:
            run_analysis(url, settings, youtube=yt, runs_dir=runs_dir, raw_dir=raw_dir,
                         model_path=model_path, run=run)
        except AnalysisFailed:
            pass            # status and error are already in metadata.json
        except Exception as exc:   # pragma: no cover - last resort
            try:
                run.fail("validate", {"code": "internal", "message": "The analysis stopped unexpectedly.",
                                      "fix": "Try again.", "detail": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def wait(self, run_id: str, timeout: Optional[float] = None) -> None:
        t = self._threads.get(run_id)
        if t:
            t.join(timeout)


def effective_status(meta: dict, running_ids) -> str:
    """'interrupted' for runs left 'running' by a stopped app; otherwise the stored status."""
    # A run is only executed by a worker thread of this app process, so a run
    # still marked "running" without a live worker was cut off (app stopped).
    status = meta.get("status", "unknown")
    if status == "running" and meta.get("run_id") not in running_ids:
        return "interrupted"
    return status
