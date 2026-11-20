"""Free engine (the only engine): planners, dispatch, licences, the no-paid-software guarantee. No models run."""

import pytest

from vidimg import config
from vidimg import pipeline as pi
from vidimg import pipeline_video as pv
from vidimg.adapters import free_sr
from vidimg.probe.video import VideoProfile

class _Faces:
    count, min_height, small = 3, 60, True


class _Img:
    width, height, flags, jpeg_quality, faces = 640, 480, ["low_resolution"], 80, _Faces()


def test_free_photo_plan_is_non_gan_bsrnet_first():
    why, cands = pi.plan_candidates(_Img())
    assert [c[1]["model"] for c in cands] == ["bsrnet_x4", "realesrnet_x4", "swinir_real_x4"]
    assert all(c[1]["engine"] == "free" and c[1]["scale"] == 4 for c in cands)
    assert "non-GAN" in why
    gan = {k for k, m in free_sr.manifest().items() if m["training"].startswith("gan")}
    assert not gan & {c[1]["model"] for c in cands}            # never a GAN model for photos


def _vp(dur, flags=()):
    p = VideoProfile(path="v.mp4", width=1280, height=720, fps=30.0, frames=int(dur * 30), duration=dur,
                     codec="h264", bitrate_kbps=900, bits_per_pixel=0.1, interlaced_ratio=0.0, field_order=None,
                     blockiness=1.0, blur=5.0)
    p.flags = list(flags)
    return p


def test_free_video_plan_and_speed_rule():
    why, cands = pv.plan_candidates(_vp(60))              # 1 min of 720p: ~1.1 h estimated
    assert [c[0] for c in cands] == ["free realesrnet", "free bsrnet", "free general-v3"]
    st = cands[0][1]()[0]
    assert isinstance(st, free_sr.FreeSR) and st.height == 1080 and st.out_size(1280, 720) == (1920, 1080)
    long_why, long_cands = pv.plan_candidates(_vp(3600))  # 1 h of 720p: ~64 h estimated
    assert [c[0] for c in long_cands] == ["free general-v3"] and "would need" in long_why
    # interlaced: double rate after deinterlacing counts twice (60 s -> ~2.1 h; 100 s -> ~3.5 h)
    assert pv.plan_candidates(_vp(60, flags=("interlaced",)))[1][0][0] == "free realesrnet"
    assert [c[0] for c in pv.plan_candidates(_vp(100, flags=("interlaced",)))[1]] == ["free general-v3"]
    assert [c[0] for c in pv.plan_candidates(_vp(100))[1]][0] == "free realesrnet"


def test_dispatch_free_models_or_video2x(monkeypatch, tmp_path):
    from vidimg.adapters import video2x
    seen = {}
    monkeypatch.setattr(free_sr, "run", lambda *a, **k: seen.setdefault("free", True))
    monkeypatch.setattr(video2x, "run_stage", lambda *a, **k: seen.setdefault("video2x", True))
    pv.run_engine(tmp_path / "a.mp4", tmp_path / "o" / "b.mp4", [free_sr.FreeSR()])
    pv.run_engine(tmp_path / "a.mp4", tmp_path / "o" / "b.mp4", [video2x.AnimeSR()])
    assert seen == {"free": True, "video2x": True}


def test_free_models_are_free_licences():
    for key, m in free_sr.manifest().items():
        assert m["license"] in ("BSD-3-Clause", "Apache-2.0", "MIT"), key
        # official GitHub releases; NAFNet's authors publish their weights only on Google Drive (README link):
        # allowed for an optional model with a fixed file name (SHA-256 pinned in lock.json on first download)
        assert m["url"].startswith("https://github.com/") or (
            m["url"].startswith("https://drive.usercontent.google.com/") and m.get("optional") and m.get("file")), key


def test_paid_software_is_never_started():
    from vidimg import proc
    with pytest.raises(proc.PaidSoftwareRefused):
        proc.run("x", [r"C:\Program Files\Topaz Labs LLC\Topaz Gigapixel AI\gigapixel.exe", "--help"])


def test_no_paid_engine_left_in_the_tool():
    import importlib.util
    from vidimg import media
    for mod in ("vidimg.adapters.gigapixel", "vidimg.adapters.photo_ai", "vidimg.adapters.video_ai",
                "vidimg.adapters.still_via_tvai"):
        assert importlib.util.find_spec(mod) is None, mod
    assert not hasattr(config, "media_engine")
    import shutil
    if shutil.which("ffprobe"):                             # absent on the CI runner: nothing to check there
        assert "topaz" not in media.ffprobe().lower()
