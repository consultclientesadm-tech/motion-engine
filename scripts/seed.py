#!/usr/bin/env python
"""Prepara um ambiente de dev: cria tabelas, bucket + lifecycle e (opcional) um job de exemplo.

  python -m scripts.seed                     # só infraestrutura
  python -m scripts.seed --demo URL_VIDEO    # cria um job de exemplo e enfileira
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.s3 import ensure_bucket_and_lifecycle  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", metavar="VIDEO_URL")
    p.add_argument("--sheet", default="s3://motion-engine/uploads/demo_sheet.png")
    a = p.parse_args()
    from alembic import command
    from alembic.config import Config
    command.upgrade(Config(str(Path(__file__).resolve().parents[1] / "alembic.ini")), "head")
    ensure_bucket_and_lifecycle()
    print("tabelas e bucket ok")
    if a.demo:
        from fastapi.testclient import TestClient

        from api.main import app
        with TestClient(app) as c:
            r = c.post("/jobs", json={"video_url": a.demo, "budget_usd": 10,
                                      "characters": [{"name": "A", "sheet_url": a.sheet, "voice_mode": "generate"}]})
            print(r.status_code, r.json())


if __name__ == "__main__":
    main()
