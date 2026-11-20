def payload(**kw):
    base = {"video_url": "s3://b/clip.mp4", "duration_s": 1800, "budget_usd": 30,
            "characters": [{"name": n, "sheet_url": f"s3://b/{n}.png"} for n in ("Ana", "Bia", "Caio")]}
    base.update(kw)
    return base


def test_create_get_cancel_resume(client):
    r = client.post("/jobs", json=payload())
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["state"] == "queued" and job["estimated_usd"] > 0
    assert client.queued == [job["id"]]

    assert client.get(f"/jobs/{job['id']}").json()["id"] == job["id"]
    assert client.post(f"/jobs/{job['id']}/resume").status_code == 409  # ainda ativo
    assert client.post(f"/jobs/{job['id']}/cancel").json()["state"] == "cancelled"
    r = client.post(f"/jobs/{job['id']}/resume", json={"budget_usd": 40})
    assert r.status_code == 200 and r.json()["budget_usd"] == 40
    assert client.queued == [job["id"], job["id"]]


def test_resume_after_worker_died(client, monkeypatch):
    """Worker morreu no meio do render: estado ativo, mas sem trava no Redis."""
    from api.routes import jobs
    from db.models import Job, JobState
    from db.session import session_scope
    job = client.post("/jobs", json=payload()).json()
    with session_scope() as s:
        s.get(Job, job["id"]).state = JobState.rendering
    assert client.post(f"/jobs/{job['id']}/resume").status_code == 409  # trava viva
    monkeypatch.setattr(jobs, "job_is_locked", lambda job_id: False)
    r = client.post(f"/jobs/{job['id']}/resume")
    assert r.status_code == 200 and r.json()["state"] == "queued"


def test_reject_when_estimate_exceeds_job_budget(client):
    r = client.post("/jobs", json=payload(duration_s=3 * 3600))
    assert r.status_code == 402
    assert r.json()["detail"]["scope"] == "do job"
    assert client.queued == []


def test_reject_when_daily_budget_exhausted(client):
    r = client.post("/jobs", json=payload(daily_budget_usd=1))
    assert r.status_code == 402
    assert r.json()["detail"]["scope"] == "diário"


def test_script_must_reference_characters(client):
    r = client.post("/jobs", json=payload(script=[{"character": "Zé", "start_s": 1, "text": "oi"}]))
    assert r.status_code == 422


def test_map_tracks_moves_ids_between_characters(client):
    job = client.post("/jobs", json=payload()).json()
    chars = client.get("/characters", params={"job_id": job["id"]}).json()
    a, b = chars[0]["id"], chars[1]["id"]
    assert client.put(f"/characters/{a}/tracks", json={"track_ids": [1, 2]}).json()["track_ids"] == [1, 2]
    client.put(f"/characters/{b}/tracks", json={"track_ids": [2, 5]})
    assert client.get(f"/characters/{a}").json()["track_ids"] == [1]


def test_estimate_endpoint(client):
    r = client.post("/jobs/estimate", json={"duration_s": 1800, "people": 3})
    assert r.status_code == 200 and r.json()["total_usd"] < 30


def test_health(client):
    assert client.get("/health").json()["ok"] is True
