"""Desliga o pod do RunPod quando ninguém usa.

Roda dentro do pod (o start do pod sobe isto em segundo plano). Desliga quando
NÃO há job ativo E a API está sem uso há IDLE_MINUTES (padrão 15).

    python -m scripts.idle_shutdown            # laço
    python -m scripts.idle_shutdown --once     # só verifica (exit 0 = desligaria)

Para desligar usa `runpodctl stop pod $RUNPOD_POD_ID` (vem nas imagens do
RunPod) ou, sem ele, a API GraphQL com RUNPOD_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request

from api import activity
from db.models import ACTIVE_STATES, Job
from db.session import session_scope

log = logging.getLogger("idle_shutdown")


def active_jobs() -> int:
    with session_scope() as s:
        return s.query(Job).filter(Job.state.in_(list(ACTIVE_STATES))).count()


def should_stop(idle_min: float) -> tuple[bool, str]:
    n = active_jobs()
    if n:
        return False, f"{n} job(s) ativo(s)"
    idle = activity.idle_seconds()
    if idle < idle_min * 60:
        return False, f"usado há {idle / 60:.1f} min"
    return True, f"sem jobs e sem uso há {min(idle, 1e9) / 60:.0f} min"


def stop_pod() -> None:
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        raise RuntimeError("RUNPOD_POD_ID não definido (não estou num pod do RunPod?)")
    if shutil.which("runpodctl"):
        subprocess.run(["runpodctl", "stop", "pod", pod], check=True)
        return
    key = os.environ["RUNPOD_API_KEY"]
    body = json.dumps({"query": f'mutation {{ podStop(input: {{podId: "{pod}"}}) {{ id desiredStatus }} }}'})
    req = urllib.request.Request("https://api.runpod.io/graphql", data=body.encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    urllib.request.urlopen(req, timeout=30).read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-minutes", type=float, default=float(os.environ.get("IDLE_MINUTES", 15)))
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    activity.touch()  # o pod acabou de ligar: conta como uso
    if a.once:
        stop, why = should_stop(a.idle_minutes)
        print(why)
        return 0 if stop else 1
    while True:
        try:
            stop, why = should_stop(a.idle_minutes)
            if stop:
                log.info("desligando o pod: %s", why)
                stop_pod()
                return 0
        except Exception as e:  # noqa: BLE001 - nunca derruba o laço por um erro passageiro
            log.warning("verificação falhou: %s", e)
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
