from __future__ import annotations

from scripts.runpod_launch import Launcher, ui_url


class FakeRP:
    """Simula a API GraphQL do RunPod."""

    def __init__(self, pods, resumable=(), free_gpus=()):
        self.pods = pods
        self.resumable, self.free = set(resumable), list(free_gpus)
        self.calls = []

    def __call__(self, q, v=None):
        v = v or {}
        self.calls.append(q.split("(")[0].split("{")[-1].strip() or q[:20])
        if "myself" in q:
            return {"data": {"myself": {"pods": self.pods}}}
        if "podResume" in q:
            if v["id"] in self.resumable:
                return {"data": {"podResume": {"id": v["id"], "desiredStatus": "RUNNING"}}}
            return {"errors": [{"message": "There are not enough free GPUs on the host machine"}],
                    "data": {"podResume": None}}
        if "podTerminate" in q:
            self.pods = [p for p in self.pods if p["id"] != v["id"]]
            self.terminated = getattr(self, "terminated", []) + [v["id"]]
            return {"data": {"podTerminate": None}}
        if "podFindAndDeployOnDemand" in q:
            gpu = v["i"]["gpuTypeId"]
            assert v["i"]["networkVolumeId"] and v["i"]["name"] == "motion-engine"
            if gpu in self.free:
                return {"data": {"podFindAndDeployOnDemand": {"id": "novo-" + gpu[-4:], "costPerHr": 0.5}}}
            return {"errors": [{"message": "no instances"}], "data": {"podFindAndDeployOnDemand": None}}
        if "podStop" in q:
            return {"data": {"podStop": {"id": v["id"]}}}
        raise AssertionError(q)


def L(rp):
    return Launcher(gql=rp, ready=lambda i: True, sleep=lambda s: None, log=lambda *a: None)


def pod(i, st="EXITED", name="motion-engine"):
    return {"id": i, "name": name, "desiredStatus": st, "costPerHr": 0.59}


def test_uses_running_pod():
    rp = FakeRP([pod("a", "RUNNING")])
    assert L(rp).start() == "a"


def test_resumes_stopped_pod():
    rp = FakeRP([pod("a")], resumable={"a"})
    assert L(rp).start() == "a"
    assert not getattr(rp, "terminated", [])


def test_no_gpu_terminates_only_our_pod_and_creates_in_preference_order():
    rp = FakeRP([pod("a"), pod("x", name="traduface")], free_gpus={"NVIDIA L40S", "NVIDIA RTX A5000"})
    new = L(rp).start()
    assert new == "novo-L40S"                        # primeira livre da lista
    assert rp.terminated == ["a"]                    # o pod de outro nome nunca é tocado


def test_nothing_free_returns_none():
    rp = FakeRP([], free_gpus=())
    assert L(rp).start() is None


def test_stop_only_running():
    rp = FakeRP([pod("a", "RUNNING"), pod("b")])
    assert L(rp).stop() == ["a"]


def test_ui_url():
    assert ui_url("abc") == "https://abc-8000.proxy.runpod.net/ui"
