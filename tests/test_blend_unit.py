"""Blended-field detection and the sRestore step: the per-frame test, the decision, the rate choice, the
worker's frame selection and the planned stages. Synthetic images only (no video files, no VapourSynth)."""

from fractions import Fraction

import cv2
import numpy as np
import pytest

from vidimg import pipeline_video as pv
from vidimg import vs_runner
from vidimg.adapters import avisynth as avs
from vidimg.adapters import vapoursynth as vsb
from vidimg.probe import blend
from vidimg.probe.video import VideoProfile, _flags


def _texture(h=240, w=360, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.0)
    img = (img - img.min()) / (img.max() - img.min())
    return img * 200 + 25


def _shift(img: np.ndarray, dx: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def test_a_mix_of_the_neighbours_is_a_blend_and_real_motion_is_not():
    t = _texture()
    a, b = _u8(_shift(t, -6)), _u8(_shift(t, 6))
    moving = blend.judge(a, _u8(t), b)                         # the picture half way: real motion
    mixed = blend.judge(a, _u8(0.4 * _shift(t, -6) + 0.6 * _shift(t, 6)), b)
    assert moving is not None and not moving["blend"]
    assert mixed is not None and mixed["blend"] and abs(mixed["alpha"] - 0.4) < 0.05


def test_sub_pixel_motion_and_still_frames_are_not_judged():
    t = _texture()
    # a slow pan is genuinely indistinguishable from a mix (100 % false blends before the flow rule)
    assert blend.judge(_u8(_shift(t, -0.3)), _u8(t), _u8(_shift(t, 0.3))) is None
    assert blend.judge(_u8(t), _u8(t), _u8(t)) is None


def test_decision_needs_enough_frames_and_more_than_one_window():
    assert blend.BlendStats(judged=300, blended=60, share=0.2, windows=[0.2, 0.3, 0.1]).is_blended
    assert not blend.BlendStats(judged=300, blended=15, share=0.05, windows=[0.05, 0.05]).is_blended
    assert not blend.BlendStats(judged=20, blended=10, share=0.5, windows=[0.5]).is_blended
    # one dissolve in one window cannot flag a whole video
    assert not blend.BlendStats(judged=300, blended=36, share=0.12, windows=[0.6, 0.0, 0.0, 0.0, 0.0]).is_blended


def test_held_pictures_in_motion_are_counted():
    t = _texture()
    pics = [_u8(_shift(t, 4 * k)) for k in range(12)]
    assert blend.held_share(pics) == 0.0
    held = [pics[k // 2] for k in range(24)]                    # every picture shown twice (on twos)
    assert 0.4 <= blend.held_share(held) <= 0.6


def test_candidate_rates():
    f = Fraction
    assert pv.deblend_rates(f(50)) == [f(24000, 1001), f(25), f(30000, 1001)]
    assert pv.deblend_rates(f(60000, 1001)) == [f(24000, 1001), f(25), f(30000, 1001)]
    assert pv.deblend_rates(f(30000, 1001)) == [f(24000, 1001), f(25)]
    assert pv.deblend_rates(f(25)) == [f(24000, 1001)]
    assert pv.deblend_rates(f(24000, 1001)) == []


def test_rate_choice_on_the_measured_trials():
    f24, f25, f30 = Fraction(24000, 1001), Fraction(25), Fraction(30000, 1001)
    # PAL -> NTSC field blend of a 25 fps original: 29.97 leaves as few blends but holds pictures
    assert pv.pick_rate({f24: 0.219, f25: 0.229, f30: 0.202}, {f24: 0.263, f25: 0.250, f30: 0.351}) == f25
    # NTSC film -> PAL field blend (23.976 original): no blends at any rate; 25 holds 3.0 points more
    assert pv.pick_rate({f24: 0.0, f25: 0.0, f30: 0.003}, {f24: 0.456, f25: 0.486, f30: 0.594}) == f24
    # progressive 29.97 copy of a 25 fps original: the blend share decides
    assert pv.pick_rate({f24: 0.198, f25: 0.095}, {f24: 0.166, f25: 0.194}) == f25
    assert pv.pick_rate({}) is None


def test_worker_selection_matches_changefps():
    sel = vs_runner.selection(50, Fraction(50), Fraction(25))
    assert sorted(sel) == list(range(0, 50, 2)) and set(sel.values()) == {1}
    sel = vs_runner.selection(1000, Fraction(25), Fraction(24000, 1001))
    assert sum(sel.values()) == int(1000 * Fraction(24000, 1001) / 25)
    assert all(v == 1 for v in sel.values())                   # going down never repeats a frame


def test_srestore_stage_and_order():
    st = avs.SRestore(frate="25", in_fps="60000/1001")
    assert st.rate() == Fraction(25) / Fraction(60000, 1001)
    with pytest.raises(NotImplementedError):
        st.lines()                                             # AviSynth+ cannot run it (GRunT missing)
    order = avs.order([st, avs.Deinterlace()])
    assert isinstance(order[0], avs.Deinterlace) and order[1] is st
    sp = vsb.spec("in.nut", [st], "c.lwi", {}, Fraction(60000, 1001), trim=(10, 20))
    assert sp["stages"] == [{"kind": "srestore", "frate": "25", "in_fps": "60000/1001"}]
    assert sp["trim"] == [10, 20]
    assert vsb.supports([avs.Deinterlace(), st])


def _profile(**kw) -> VideoProfile:
    base = dict(path="x", width=720, height=576, fps=25.0, frames=1000, duration=40.0, codec="h264",
                bitrate_kbps=None, bits_per_pixel=None, interlaced_ratio=0.0, field_order="progressive",
                blockiness=None, blur=None)
    base.update(kw)
    p = VideoProfile(**base)
    p.flags = _flags(p)
    return p


def test_blended_interlaced_is_never_telecine():
    p = _profile(interlaced_ratio=0.91, field_order="tff", repeated_field_ratio=0.33, blend_share=0.24,
                 blend={"is_blended": True})
    assert "blended" in p.flags and "interlaced" in p.flags and "telecine" not in p.flags
    stages, why = pv.plan_prepass(p)
    assert [type(s) for s in stages] == [avs.Deinterlace, avs.SRestore] and "sRestore" in why
    # the same fields without blends stay telecine
    assert "telecine" in _profile(interlaced_ratio=0.91, field_order="tff", repeated_field_ratio=0.33).flags


def test_blended_progressive_plans_srestore_only():
    p = _profile(blend_share=0.2, blend={"is_blended": True})
    stages, _ = pv.plan_prepass(p)
    assert [type(s) for s in stages] == [avs.SRestore]
    assert pv.plan_prepass(_profile())[0] == []
