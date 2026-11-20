#!/usr/bin/env python
"""Estima o custo de um job ANTES de gastar.

Exemplos:
  python -m scripts.cost_estimate --minutes 30 --people 3
  python -m scripts.cost_estimate --minutes 40 --people 3 --budget 30 --json
  python -m scripts.cost_estimate --video clip.mp4 --people 3   # mede a duração com ffprobe

Sai com código 2 se a estimativa não couber em --budget (útil em script/CI).
Os tempos por frame vêm de api/costs.py e podem ser ajustados por variável
de ambiente COST_* depois de calibrar no seu hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.costs import estimate  # noqa: E402
from settings import get_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--minutes", type=float)
    g.add_argument("--seconds", type=float)
    g.add_argument("--video", help="arquivo/URL para medir a duração com ffprobe")
    p.add_argument("--people", type=int, default=3)
    p.add_argument("--resolution", choices=["720p", "1080p"], default="1080p")
    p.add_argument("--all-frames", action="store_true", help="processa também trechos sem pessoa")
    p.add_argument("--people-ratio", type=float, default=0.6, help="fração do vídeo com pessoa (padrão 0.6)")
    p.add_argument("--speech-seconds", type=float, default=0.0, help="segundos de fala gerada (ativa lipsync)")
    p.add_argument("--gpu-usd-hour", type=float, help="sobrescreve GPU_USD_PER_HOUR")
    p.add_argument("--budget", type=float, help="teto do job; sai com 2 se não couber")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    if a.video:
        from workers.media import probe
        duration = probe(a.video).duration_s
    else:
        duration = a.seconds if a.seconds is not None else a.minutes * 60

    s = get_settings()
    if a.gpu_usd_hour:
        s = s.model_copy(update={"gpu_usd_per_hour": a.gpu_usd_hour})
    est = estimate(duration, people=a.people, target_resolution=a.resolution,
                   only_people=not a.all_frames, people_ratio=a.people_ratio,
                   speech_s=a.speech_seconds, lipsync=a.speech_seconds > 0, s=s)

    fits = a.budget is None or est.total_usd <= a.budget
    if a.json:
        print(json.dumps({**est.to_dict(), "budget_usd": a.budget, "fits": fits}, indent=2))
    else:
        print(f"Vídeo: {duration / 60:.1f} min | {a.people} pessoas | {a.resolution} | "
              f"{est.blocks} blocos de {s.block_seconds:.0f}s ({est.blocks_with_people} com pessoa)")
        print(f"GPU: US$ {s.gpu_usd_per_hour:.2f}/h")
        for k, v in est.usd.items():
            secs = est.gpu_seconds.get(k)
            extra = f"  ({secs / 3600:.2f} h GPU)" if secs else ""
            print(f"  {k:<8} US$ {v:8.2f}{extra}")
        print(f"  {'TOTAL':<8} US$ {est.total_usd:8.2f}  ({est.gpu_hours:.2f} h GPU)")
        if a.budget is not None:
            print(("✅ cabe" if fits else "❌ não cabe") + f" no teto de US$ {a.budget:.2f}")
    return 0 if fits else 2


if __name__ == "__main__":
    sys.exit(main())
