"""
CreatorPulse — analysis run storage.

Every analysis started from the app gets its own run:

    data/raw/app_runs/comments_<run_id>.csv    raw comments as collected (never overwritten)
    data/runs/<run_id>/metadata.json           status, timestamps, source URL, settings,
                                               per-stage progress, errors, output paths
    data/runs/<run_id>/processed/...           every stage output for this run

metadata.json is rewritten atomically after each stage, so a run that fails
part-way keeps the outputs and status of the stages that finished.
"""

import json
import os
import time
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config import DATA_DIR, RAW_DATA_DIR

# Overridable so a separate data folder can be used (e.g. for a demo or tests).
RUNS_DIR = Path(os.environ.get("CREATORPULSE_RUNS_DIR", DATA_DIR / "runs"))
APP_RAW_DIR = Path(os.environ.get("CREATORPULSE_APP_RAW_DIR", RAW_DATA_DIR / "app_runs"))

STAGES = [
    ("validate", "Validating YouTube URL"),
    ("collect", "Collecting comments"),
    ("preprocess", "Preprocessing text"),
    ("classify", "Classifying comments"),
    ("topics", "Discovering audience topics"),
    ("insights", "Calculating insights"),
    ("recommend", "Generating video recommendations"),
    ("dashboard", "Preparing dashboard"),
]
STAGE_LABEL = dict(STAGES)


def resolve_dirs(runs_dir=None, raw_dir=None):
    """Defaults are read at call time, so RUNS_DIR / APP_RAW_DIR can be changed (tests, demos)."""
    import src.runs as mod
    return Path(runs_dir or mod.RUNS_DIR), Path(raw_dir or mod.APP_RAW_DIR)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:24]
    return s or "run"


def new_run_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{_slug(label)}_{secrets.token_hex(3)}"


def _write_json(path: Path, data: dict, attempts: int = 40) -> None:
    """Write JSON atomically (temp file + rename).

    On Windows the rename fails with "Access is denied" while another thread or
    program (the page polling progress, OneDrive, antivirus) has the file open
    for a moment, so it is retried for ~4 seconds before giving up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    delay = 0.02
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 1.5, 0.25)
    # Last resort: write in place (readers retry on a partial file).
    try:
        path.write_text(text, encoding="utf-8")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def read_json(path: Path, attempts: int = 20) -> dict:
    """Read JSON, retrying briefly if the file is locked or caught mid-write."""
    delay = 0.02
    for i in range(attempts):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 0.25)


class Run:
    """A run directory plus its metadata. Not thread-safe; one writer per run."""

    def __init__(self, run_id: str, runs_dir: Optional[Path] = None, raw_dir: Optional[Path] = None):
        runs_dir, raw_dir = resolve_dirs(runs_dir, raw_dir)
        self.run_id = run_id
        self.dir = Path(runs_dir) / run_id
        self.processed = self.dir / "processed"
        self.raw_dir = Path(raw_dir)
        self.meta_path = self.dir / "metadata.json"
        self.meta: dict = {}
        self._t0: dict = {}

    # ---------------- creation / loading ----------------
    @classmethod
    def create(cls, source_url: str, settings: dict, runs_dir: Optional[Path] = None,
               raw_dir: Optional[Path] = None, label: str = "") -> "Run":
        run = cls(new_run_id(label or source_url), runs_dir, raw_dir)
        run.processed.mkdir(parents=True, exist_ok=False)
        run.meta = {
            "run_id": run.run_id, "source_url": source_url, "settings": settings,
            "status": "running", "created_at": now_iso(), "updated_at": now_iso(), "finished_at": None,
            "failed_stage": None, "error": None, "warnings": [],
            "stages": [{"key": k, "label": lbl, "status": "pending", "started_at": None,
                        "finished_at": None, "seconds": None, "detail": ""} for k, lbl in STAGES],
            "target": None, "collection": None, "totals": {}, "outputs": {},
        }
        run.save()
        return run

    @classmethod
    def load(cls, run_id: str, runs_dir: Optional[Path] = None, raw_dir: Optional[Path] = None) -> "Run":
        run = cls(run_id, runs_dir, raw_dir)
        run.meta = read_json(run.meta_path)
        return run

    def save(self) -> None:
        self.meta["updated_at"] = now_iso()
        _write_json(self.meta_path, self.meta)

    # ---------------- stage bookkeeping ----------------
    def stage(self, key: str) -> dict:
        return next(s for s in self.meta["stages"] if s["key"] == key)

    def start_stage(self, key: str, detail: str = "") -> None:
        s = self.stage(key)
        s.update(status="running", started_at=now_iso(), detail=detail)
        self._t0[key] = time.perf_counter()
        self.save()

    def update_stage(self, key: str, detail: str) -> None:
        self.stage(key)["detail"] = detail
        self.save()

    def finish_stage(self, key: str, detail: str = "") -> None:
        s = self.stage(key)
        s.update(status="done", finished_at=now_iso(), detail=detail or s.get("detail", ""))
        t0 = self._t0.get(key)
        s["seconds"] = round(time.perf_counter() - t0, 2) if t0 is not None else None
        self.save()

    def fail(self, key: str, error: dict) -> None:
        s = self.stage(key)
        s.update(status="failed", finished_at=now_iso(), detail=error.get("message", ""))
        for later in self.meta["stages"]:
            if later["status"] == "pending":
                later["status"] = "skipped"
        self.meta.update(status="failed", failed_stage=key, error=error, finished_at=now_iso())
        self.save()

    def complete(self) -> None:
        self.meta.update(status="completed_with_warnings" if self.meta["warnings"] else "completed",
                         finished_at=now_iso())
        self.save()

    def add_output(self, name: str, path: Path) -> None:
        self.meta["outputs"][name] = str(Path(path).resolve())
        self.save()

    @property
    def raw_path(self) -> Path:
        return self.raw_dir / f"comments_{self.run_id}.csv"

    def path(self, name: str) -> Path:
        return self.processed / name


def list_runs(runs_dir: Optional[Path] = None) -> List[dict]:
    """Metadata of every run, newest first. Unreadable runs are skipped."""
    runs_dir, _ = resolve_dirs(runs_dir, None)
    runs = []
    if not Path(runs_dir).exists():
        return runs
    for meta in Path(runs_dir).glob("*/metadata.json"):
        try:
            runs.append(read_json(meta, attempts=5))
        except Exception:
            continue
    return sorted(runs, key=lambda m: m.get("created_at", ""), reverse=True)


def find_recent(source_key: str, settings: dict, within_minutes: int = 30,
                runs_dir: Optional[Path] = None) -> Optional[dict]:
    """A finished run of the same target and settings in the last `within_minutes`, if any."""
    now = datetime.now(timezone.utc)
    for m in list_runs(runs_dir):
        if m.get("status", "").startswith("completed") and (m.get("target") or {}).get("key") == source_key \
                and m.get("settings") == settings:
            try:
                age = (now - datetime.fromisoformat(m["created_at"])).total_seconds() / 60
            except Exception:
                continue
            if age <= within_minutes:
                return m
    return None
