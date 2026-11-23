"""Claude chooses the video model from model_guide: the guide's facts and suggestion, and the candidate order when a
model is preferred (goes first, approved order follows) or insisted on (the only candidate)."""

import pytest

from vidimg import cli
from vidimg import pipeline_video as pv
from vidimg.probe.video import VideoProfile, _flags


def _profile(**kw) -> VideoProfile:
    base = dict(path="x", width=720, height=576, fps=25.0, frames=750, duration=30.0, codec="h264",
                bitrate_kbps=None, bits_per_pixel=None, interlaced_ratio=0.0, field_order="progressive",
                blockiness=None, blur=None)
    base.update(kw)
    p = VideoProfile(**base)
    p.flags = _flags(p)
    return p


def _labels(cands):
    return [c[0].split(" ", 1)[1] for c in cands]


def test_preferred_model_goes_first_then_the_approved_order():
    p = _profile()
    _, cands = pv.plan_candidates(p, 2, "anime", "cugan", strict=False)
    assert _labels(cands) == ["cugan", "esrgan-v3", "general-v3"]
    _, cands = pv.plan_candidates(p, 2, "anime", "cugan-pro", strict=False)
    assert _labels(cands)[0] == "cugan-pro" and "esrgan-v3" in _labels(cands)
    _, cands = pv.plan_candidates(p, None, "real", "bsrnet", strict=False)
    assert _labels(cands) == ["bsrnet", "realesrnet", "general-v3"]


def test_insisted_model_is_the_only_candidate():
    _, cands = pv.plan_candidates(_profile(), 2, "anime", "cugan", strict=True)
    assert _labels(cands) == ["cugan"]
    why, cands = pv.plan_candidates(_profile(), 2, "anime", "realesrnet", strict=True)
    assert cands == [] and "does not apply" in why


def test_a_preferred_model_that_does_not_fit_falls_back_to_the_approved_order():
    why, cands = pv.plan_candidates(_profile(), 2, "anime", "realesrnet", strict=False)
    assert _labels(cands) == ["esrgan-v3", "cugan", "general-v3"] and "approved order" in why


def test_guide_suggestions_follow_the_measurements():
    g = pv.model_guide(_profile(), 2, "anime")
    assert g["suggested"] == "esrgan-v3"
    assert {m["model"] for m in g["models"] if m["applies"]} == {"esrgan-v3", "cugan", "cugan-pro", "janai-sd", "janai-hd",
                                                                 "general-v3"}
    g = pv.model_guide(_profile(faces_max=2, face_min_height=60), 2, "anime")
    assert any("cugan-pro" in r for r in g["why"])                         # realistic drawn faces: check the sheet
    g = pv.model_guide(_profile(faces_max=1, face_min_height=80), None, "real")
    assert g["suggested"] == "realesrnet"
    long = _profile(frames=25 * 3600 * 3, duration=3 * 3600.0, faces_max=1, face_min_height=80)
    g = pv.model_guide(long, None, "real")
    assert g["suggested"] == "general-v3"
    est = {m["model"]: m.get("est_hours") for m in g["models"]}
    assert est["realesrnet"] > pv.FREE_NONGAN_MAX_HOURS and est["realesrnet"] > 10 * est["general-v3"]
    assert "model_strict" in g["how"]


def test_cli_model_and_prefer_exclude_each_other():
    with pytest.raises(SystemExit):
        cli.main(["restore-video", "x.mp4", "--model", "cugan", "--prefer", "cugan"])
