from __future__ import annotations

import io
import shutil
import subprocess

import pytest


class FakeBody:
    def __init__(self, data: bytes):
        self.data = data

    def iter_chunks(self, size):
        for i in range(0, len(self.data), size):
            yield self.data[i:i + size]


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def get_object(self, Bucket, Key, Range=None):
        data = self.objects[Key]
        if Range:
            a, _, b = Range[6:].partition("-")
            a, b = int(a), int(b) if b else len(data) - 1
            part = data[a:b + 1]
            return {"Body": FakeBody(part), "ContentLength": len(part),
                    "ContentRange": f"bytes {a}-{a + len(part) - 1}/{len(data)}"}
        return {"Body": FakeBody(data), "ContentLength": len(data)}


@pytest.fixture()
def fake_s3(monkeypatch):
    from storage import s3
    fake = FakeS3()

    def upload(path, key, content_type=None):
        with open(path, "rb") as f:
            fake.objects[key] = f.read()
        return key

    monkeypatch.setattr(s3, "client", lambda: fake)
    monkeypatch.setattr(s3, "bucket", lambda: "b")
    monkeypatch.setattr(s3, "upload", upload)
    return fake


def test_page_and_root_redirect(client):
    r = client.get("/ui")
    assert r.status_code == 200 and "Motion Engine" in r.text
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui"


def test_upload_image_and_reject_unknown(client, fake_s3):
    r = client.post("/ui/upload", files={"file": ("ana.png", b"\x89PNG fake", "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "image" and body["url"].startswith("s3://b/uploads/")
    assert len(fake_s3.objects) == 1
    r = client.post("/ui/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 422


def test_upload_rejects_broken_video(client, fake_s3):
    r = client.post("/ui/upload", files={"file": ("clip.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 422
    assert fake_s3.objects == {}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="sem ffmpeg")
def test_upload_video_measures_duration(client, fake_s3, tmp_path):
    out = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10",
                    "-t", "2", "-pix_fmt", "yuv420p", str(out)], check=True)
    r = client.post("/ui/upload", files={"file": ("c.mp4", io.BytesIO(out.read_bytes()), "video/mp4")})
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "video" and abs(r.json()["duration_s"] - 2) < 0.2


def test_video_stream_only_when_done_and_supports_range(client, fake_s3, db):
    from db.models import Job, JobState
    job = client.post("/jobs", json={
        "video_url": "s3://b/clip.mp4", "duration_s": 60,
        "characters": [{"name": "Ana", "sheet_url": "s3://b/ana.png"}]}).json()
    assert client.get(f"/ui/jobs/{job['id']}/video").status_code == 409

    with db.session_scope() as s:
        j = s.get(Job, job["id"])
        j.state, j.output_key = JobState.done, "jobs/x/output.mp4"
    fake_s3.objects["jobs/x/output.mp4"] = bytes(range(256)) * 4

    r = client.get(f"/ui/jobs/{job['id']}/video")
    assert r.status_code == 200 and len(r.content) == 1024
    r = client.get(f"/ui/jobs/{job['id']}/video", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))
    assert r.headers["content-range"] == "bytes 10-19/1024"
    r = client.get(f"/ui/jobs/{job['id']}/video?download=1")
    assert "attachment" in r.headers["content-disposition"]
    assert client.get("/ui/jobs/nao-existe/video").status_code == 404


def test_shutdown_button(client, monkeypatch):
    import threading

    from scripts import idle_shutdown
    calls = []
    monkeypatch.setattr(idle_shutdown, "stop_pod", lambda: calls.append(1))
    monkeypatch.setattr(threading, "Timer", lambda t, f: type("T", (), {"start": lambda self: f()})())
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert client.post("/ui/shutdown").status_code == 400          # fora do RunPod
    monkeypatch.setenv("RUNPOD_POD_ID", "abc")
    client.post("/jobs", json={"video_url": "s3://b/clip.mp4", "duration_s": 30,
                               "characters": [{"name": "Ana", "sheet_url": "s3://b/ana.png"}]})
    assert client.post("/ui/shutdown").status_code == 409          # job em produção
    assert calls == []
    r = client.post("/ui/shutdown?force=true")
    assert r.status_code == 200 and calls == [1]
