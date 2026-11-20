#!/usr/bin/env python
"""Liga o Motion Engine no RunPod e devolve a URL da /ui (mesma regra do deploy/launcher).

1. Já há um pod POD_NAME ligado? usa ele.
2. Senão tenta religar cada pod POD_NAME parado.
3. Se a GPU dele foi alugada por outro, apaga SÓ esse pod (o Network Volume fica)
   e cria outro no mesmo volume com a primeira GPU livre da lista.

    RUNPOD_API_KEY=... python -m scripts.runpod_launch            # liga e espera a /ui
    python -m scripts.runpod_launch --stop                         # para os pods ligados
    python -m scripts.runpod_launch --status

A chave vem de RUNPOD_API_KEY ou de ~/.runpod/config.toml (nunca é impressa).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

GQL = "https://api.runpod.io/graphql"
POD_NAME = "motion-engine"
NETWORK_VOLUME_ID = "kl3al7r6s6"
DATA_CENTER = "CA-MTL-1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
START_CMD = ("bash -c 'mkdir -p /workspace/motion/logs; (bash /workspace/motion/start.sh "
             "> /workspace/motion/logs/boot.log 2>&1 &); exec /start.sh'")
GPU_TYPES = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 4090", "NVIDIA L40S",
             "NVIDIA RTX A5000"]


def ui_url(pod_id: str) -> str:
    return f"https://{pod_id}-8000.proxy.runpod.net/ui"


def api_key() -> str:
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    txt = open(os.path.expanduser("~/.runpod/config.toml")).read()
    m = re.search(r'apiKey\s*=\s*"?([^"\n]+)"?', txt)
    if not m:
        raise SystemExit("sem RUNPOD_API_KEY nem ~/.runpod/config.toml")
    return m.group(1).strip()


def http_gql(query: str, variables: dict | None = None) -> dict:
    req = urllib.request.Request(
        GQL, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}",
                 "User-Agent": "motion-engine-launch"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    except urllib.error.HTTPError as e:
        return {"errors": [{"message": f"HTTP {e.code}: {e.read().decode()[:300]}"}]}


def http_ready(pod_id: str) -> bool:
    try:
        with urllib.request.urlopen(f"https://{pod_id}-8000.proxy.runpod.net/health", timeout=8) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


@dataclass
class Launcher:
    gql: callable = http_gql
    ready: callable = http_ready
    sleep: callable = time.sleep
    gpus: list[str] = field(default_factory=lambda: list(GPU_TYPES))
    log: callable = print

    def pods(self) -> list[dict]:
        r = self.gql("{ myself { pods { id name desiredStatus costPerHr } } }")
        return [p for p in r["data"]["myself"]["pods"] if p["name"] == POD_NAME]

    def _ok(self, r: dict, key: str) -> dict | None:
        if r.get("errors"):
            self.log("  RunPod: " + r["errors"][0].get("message", "erro"))
            return None
        return (r.get("data") or {}).get(key)

    def resume(self, pod_id: str) -> bool:
        r = self.gql('mutation($id: String!) { podResume(input:{podId:$id, gpuCount:1}) '
                     '{ id desiredStatus } }', {"id": pod_id})
        return bool(self._ok(r, "podResume"))

    def terminate(self, pod_id: str) -> None:
        # apaga só o pod; o Network Volume (todo o motor) continua
        self.gql('mutation($id: String!) { podTerminate(input:{podId:$id}) }', {"id": pod_id})

    def create(self) -> str | None:
        q = ('mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input:$i) '
             '{ id costPerHr } }')
        for gpu in self.gpus:
            i = {"cloudType": "SECURE", "gpuCount": 1, "gpuTypeId": gpu, "name": POD_NAME,
                 "imageName": IMAGE, "dockerArgs": START_CMD, "ports": "8000/http,22/tcp",
                 "volumeMountPath": "/workspace", "containerDiskInGb": 20,
                 "networkVolumeId": NETWORK_VOLUME_ID, "dataCenterId": DATA_CENTER}
            pod = self._ok(self.gql(q, {"i": i}), "podFindAndDeployOnDemand")
            if pod:
                self.log(f"  pod novo {pod['id']} com {gpu} (US$ {pod.get('costPerHr')}/h)")
                return pod["id"]
            self.log(f"  sem {gpu} livre")
        return None

    def start(self) -> str | None:
        pods = self.pods()
        running = [p for p in pods if p["desiredStatus"] == "RUNNING"]
        if running:
            return running[0]["id"]
        for p in pods:
            if self.resume(p["id"]):
                self.log(f"  religado {p['id']}")
                return p["id"]
            self.log(f"  {p['id']} sem GPU na máquina dele: apagando só o pod")
            self.terminate(p["id"])
        return self.create()

    def wait_ready(self, pod_id: str, timeout_s: float = 900, every: float = 10) -> bool:
        t = 0.0
        while t < timeout_s:
            if self.ready(pod_id):
                return True
            self.sleep(every)
            t += every
        return False

    def stop(self) -> list[str]:
        ids = [p["id"] for p in self.pods() if p["desiredStatus"] == "RUNNING"]
        for i in ids:
            self.gql('mutation($id: String!) { podStop(input:{podId:$id}) { id } }', {"id": i})
        return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    L = Launcher()
    if a.stop:
        print("parados:", L.stop() or "nenhum ligado")
        return 0
    if a.status:
        for p in L.pods():
            print(p["id"], p["desiredStatus"], ui_url(p["id"]))
        return 0
    pod = L.start()
    if not pod:
        print("nenhuma GPU livre da lista agora; tente de novo em alguns minutos")
        return 2
    if not a.no_wait:
        print("esperando a /ui subir…")
        if not L.wait_ready(pod):
            print("o pod ligou mas a /ui não respondeu em 15 min; veja /workspace/motion/logs")
            return 3
    print(ui_url(pod))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
