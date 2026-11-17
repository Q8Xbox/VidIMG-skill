"""Video pipeline planning and cut detection (no engines run)."""

import numpy as np

from vidimg import pipeline_video as pv
from vidimg.adapters import avisynth as avs
from vidimg.guard.video import _is_cut
from vidimg.probe.video import VideoProfile


def _vp(w, h, flags=(), duration=60.0, cuts=()):
    p = VideoProfile(path="v.mp4", width=w, height=h, fps=30.0, frames=int(duration * 30), duration=duration,
                     codec="h264", bitrate_kbps=900, bits_per_pixel=0.1, interlaced_ratio=0.0, field_order=None,
                     blockiness=1.0, blur=5.0, scene_cuts=list(cuts))
    p.flags = list(flags)
    return p


def test_target_size():
    assert pv.target_size(_vp(640, 352))[0] == 704
    assert pv.target_size(_vp(1280, 720))[0] == 1080
    assert pv.target_size(_vp(1920, 1080))[0] is None


def test_field_handling_is_classical_not_dione():
    why, c = pv.plan_candidates(_vp(720, 480, flags=("interlaced",)))
    assert [label for label, _ in c][0] == "free realesrnet"     # models only ever see progressive frames
    assert "deinterlace first" in why
    pre, pwhy = pv.plan_prepass(_vp(720, 480, flags=("interlaced",)))
    assert isinstance(pre[0], avs.Deinterlace) and pre[0].method == "qtgmc" and pre[0].tff
    assert pre[0].source_match == 3 and pre[0].lossless == 2 and "QTGMC" in pwhy
    bff = _vp(720, 480, flags=("interlaced", "noisy"))
    bff.field_order = "bff"
    pre = pv.plan_prepass(bff)[0][0]
    assert not pre.tff and pre.tr2 == 2
    tc = pv.plan_prepass(_vp(720, 480, flags=("telecine",)))[0]
    assert isinstance(tc[0], avs.IVTC)
    assert pv.plan_prepass(_vp(1280, 720))[0] == []


def test_fallback_is_the_ffmpeg_field_chain():
    vf, rate = pv._fallback_vf(_vp(720, 480, flags=("interlaced",)))
    assert vf.startswith("bwdif=mode=send_field") and rate == 2.0
    vf, rate = pv._fallback_vf(_vp(720, 480, flags=("telecine",)))
    assert vf.startswith("fieldmatch") and vf.endswith("decimate") and rate == 0.8


def test_previews_avoid_scene_cuts():
    starts = pv._preview_starts(_vp(640, 360, duration=100.0, cuts=(21.0,)))
    assert len(starts) == 3 and all(not (s <= 21.0 <= s + pv.PREVIEW_SECONDS) for s in starts)
    assert pv._preview_starts(_vp(640, 360, duration=4.0)) == [0.0]


def test_cut_detector():
    rng = np.random.default_rng(0)
    a = np.clip(rng.normal(60, 10, (120, 160, 3)), 0, 255).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-3, 4, a.shape), 0, 255).astype(np.uint8)   # same shot
    c = np.clip(rng.normal(190, 15, (120, 160, 3)), 0, 255).astype(np.uint8)              # new shot
    assert not _is_cut(a, b) and _is_cut(a, c)
