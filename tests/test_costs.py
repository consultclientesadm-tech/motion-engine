from api import costs
from scripts.cost_estimate import main


def test_estimate_30min_3people_fits_30usd():
    est = costs.estimate(30 * 60, people=3)
    assert est.blocks == 60
    assert est.blocks_with_people == 36
    assert 0 < est.total_usd < 30


def test_3h_does_not_fit_30usd():
    assert costs.estimate(3 * 3600, people=3).total_usd > 30


def test_720p_skips_upscale_and_only_people_is_cheaper():
    full = costs.estimate(600, people=2, only_people=False)
    some = costs.estimate(600, people=2, only_people=True)
    assert some.total_usd < full.total_usd
    assert costs.estimate(600, people=2, target_resolution="720p").usd["upscale"] == 0


def test_lipsync_only_when_speech():
    no = costs.estimate(600, people=2, lipsync=False)
    yes = costs.estimate(600, people=2, lipsync=True, speech_s=60)
    assert yes.total_usd > no.total_usd


def test_cli_exit_codes(capsys):
    assert main(["--minutes", "30", "--people", "3", "--budget", "30"]) == 0
    assert "cabe" in capsys.readouterr().out
    assert main(["--minutes", "180", "--people", "3", "--budget", "30", "--json"]) == 2
