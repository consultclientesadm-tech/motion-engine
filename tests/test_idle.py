import os
import time

from api import activity
from db.models import Job, JobState
from scripts import idle_shutdown


def test_idle_rules(db, tmp_path, monkeypatch):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    from settings import get_settings
    get_settings.cache_clear()
    try:
        assert activity.idle_seconds() == float("inf")
        activity.touch()
        assert idle_shutdown.should_stop(15)[0] is False          # acabou de usar
        old = time.time() - 20 * 60
        os.utime(activity.stamp_path(), (old, old))
        assert idle_shutdown.should_stop(15)[0] is True           # 20 min parado
        with db.session_scope() as s:
            s.add(Job(video_url="x", budget_usd=1, daily_budget_usd=1, state=JobState.rendering))
        assert idle_shutdown.should_stop(15) == (False, "1 job(s) ativo(s)")
    finally:
        get_settings.cache_clear()
