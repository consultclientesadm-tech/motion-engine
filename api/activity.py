"""Marca a última vez que alguém usou a API (para o desligamento automático do pod)."""
from __future__ import annotations

import os
import time
from pathlib import Path

from settings import get_settings


def stamp_path() -> Path:
    return Path(get_settings().work_dir) / "last_activity"


def touch() -> None:
    p = stamp_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def idle_seconds(now: float | None = None) -> float:
    p = stamp_path()
    now = time.time() if now is None else now
    try:
        return now - os.path.getmtime(p)
    except OSError:
        return float("inf")
