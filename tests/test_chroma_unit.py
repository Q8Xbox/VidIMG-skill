"""Chroma shift: the measurement on synthetic 4:2:0 frames with known shifts, the decision rules, the stage
in both engines and its place in the plan. No video files, no VapourSynth."""

from fractions import Fraction

import numpy as np

from vidimg import pipeline_video as pv
from vidimg.adapters import avisynth as avs
from vidimg.adapters import vapoursynth as vsb
from vidimg.probe import chroma
from vidimg.probe.video import VideoProfile, _flags


def _shapes(h=288, w=384, seed=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat coloured rectangles (cartoon-like): full-resolution Y, U, V with edges in the same places."""
    rng = np.random.default_rng(seed)
    y = np.full((h, w), 120.0)
    u = np.full((h, w), 128.0)
    v = np.full((h, w), 128.0)
    for _ in range(40):
        y0, x0 = rng.integers(0, h - 40), rng.integers(0, w - 40)
        hh, ww = rng.integers(20, 90), rng.integers(20, 90)
        y[y0:y0 + hh, x0:x0 + ww] = rng.uniform(40, 220)
        u[y0:y0 + hh, x0:x0 + ww] = rng.uniform(70, 190)
        v[y0:y0 + hh, x0:x0 + ww] = rng.uniform(70, 190)
    return y, u, v


def _subsample(p: np.ndarray) -> np.ndarray:
    """4:2:0, left siting: the same rule the measurement assumes for the luma."""
    return chroma.luma_on_chroma_grid(p)


def _frame(dx=0, dy=0, seed=1):
    y, u, v = _shapes(seed=seed)
    u, v = (np.roll(np.roll(p, dx, axis=1), dy, axis=0) for p in (u, v))
    return y.astype(np.uint8), _subsample(u).astype(np.uint8), _subsample(v).astype(np.uint8)


def test_aligned_planes_measure_zero_and_shifts_are_recovered():
    r0 = chroma.measure_frame(*_frame())
    assert max(abs(c) for c in (*r0["u"], *r0["v"])) < 0.3
    r = chroma.measure_frame(*_frame(dx=4, dy=-2))
    assert abs(r["u"][0] - 4) < 0.4 and abs(r["u"][1] + 2) < 0.4
    assert abs(r["v"][0] - 4) < 0.4 and abs(r["v"][1] + 2) < 0.4


def test_colourless_frames_are_not_measured():
    y, _, _ = _frame()
    grey = np.full((y.shape[0] // 2, y.shape[1] // 2), 128, np.uint8)
    assert chroma.measure_frame(y, grey, grey) is None


def _stats(u, v, spreads=None, frames=12) -> chroma.ChromaShift:
    # u, v given as excess over the baseline; stored as measured values
    return chroma.ChromaShift(u=(u[0] + chroma.BASE_X, u[1] + chroma.BASE_Y),
                              v=(v[0] + chroma.BASE_X, v[1] + chroma.BASE_Y),
                              spreads=spreads or {"u": [0.2, 0.2], "v": [0.2, 0.2]}, frames=frames)


def test_decision_and_correction():
    st = _stats((1.7, -0.3), (1.6, -0.3))                       # Sample01-like
    assert st.is_shifted and st.correction() == {"u": [-1.75, 0.0], "v": [-1.5, 0.0]}
    assert not _stats((1.0, 0.5), (0.2, 1.0)).is_shifted        # Sample06-like: below 1.25 px
    assert not _stats((2.0, 0.0), (2.0, 0.0), frames=2).is_shifted
    # Sample04-like: vertical steady, horizontal scattered -> vertical only
    st = _stats((-1.6, 4.1), (1.7, 4.3), {"u": [0.53, 0.57], "v": [1.79, 0.18]})
    assert st.is_shifted and st.correction() == {"u": [0.0, -4.0], "v": [0.0, -4.25]}
    # one plane scattered on an axis: neither plane is moved on it (no new fringes)
    st = _stats((1.7, 0.8), (1.9, 1.2), {"u": [0.36, 0.36], "v": [0.84, 0.25]})
    assert not st.is_shifted
    # each plane on its own: both 4 px low, Cr alone 2.2 px to the right, Cb about in place (an owner cartoon)
    st = _stats((0.59, 3.97), (2.21, 4.09), {"u": [0.59, 0.44], "v": [0.48, 0.49]})
    assert st.is_shifted and st.correction() == {"u": [0.0, -4.0], "v": [-2.25, -4.0]}
    # ... but not when the other plane cannot be placed at all on that axis
    assert _stats((0.2, 0.0), (2.2, 0.0), {"u": [1.4, 0.2], "v": [0.3, 0.2]}).correction() == {"u": [0.0, 0.0], "v": [0.0, 0.0]}
    # a component far off whose frames nearly agree is worth more frames; a settled one is not
    assert _stats((0.59, 3.97), (2.21, 4.09), {"u": [0.59, 0.44], "v": [0.57, 0.49]}).nearly()
    assert not _stats((1.7, -0.3), (1.6, -0.3)).nearly() and not _stats((0.2, 0.0), (2.2, 0.0), {"u": [0.2, 0.2], "v": [1.6, 0.2]}).nearly()
    # interlaced: horizontal part only
    assert _stats((2.0, 2.0), (2.0, 2.0)).correction(horizontal_only=True) == {"u": [-2.0, 0.0], "v": [-2.0, 0.0]}


def test_stage_in_both_engines():
    cs = avs.ChromaShift(u_dx=-1.75, u_dy=0.0, v_dx=-1.5, v_dy=-4.0)
    text = "\n".join(cs.lines())
    assert "ExtractU().Spline36Resize(__y.Width() / 2, __y.Height() / 2, src_left=0.875, src_top=0)" in text
    assert "ExtractV().Spline36Resize(__y.Width() / 2, __y.Height() / 2, src_left=0.75, src_top=2)" in text
    assert text.splitlines()[-1].startswith("CombinePlanes(__y, __u, __v")
    script = avs.Script("in.mp4", [avs.Deinterlace(), cs], "c.lwi", Fraction(25)).text()
    assert not avs.lint(script)
    assert script.index("QTGMC(") < script.index("ExtractU(")             # after field handling
    sp = vsb.spec("in.mp4", [cs], "c.lwi", {}, Fraction(25))
    assert sp["stages"] == [{"kind": "chroma", "u_dx": -1.75, "u_dy": 0.0, "v_dx": -1.5, "v_dy": -4.0}]
    assert cs.rate() == 1 and vsb.supports([cs])


def _profile(**kw) -> VideoProfile:
    base = dict(path="x", width=720, height=576, fps=25.0, frames=1000, duration=40.0, codec="h264",
                bitrate_kbps=None, bits_per_pixel=None, interlaced_ratio=0.0, field_order="progressive",
                blockiness=None, blur=None)
    base.update(kw)
    p = VideoProfile(**base)
    p.flags = _flags(p)
    return p


def _chroma_dict(u, v):
    return _stats(u, v).to_dict()


def test_plan_places_the_correction():
    p = _profile(chroma=_chroma_dict((0.0, 4.1), (0.0, 4.3)))
    stages, why = pv.plan_prepass(p)
    assert [type(s) for s in stages] == [avs.ChromaShift] and "Cb 4 px up" in why
    p = _profile(interlaced_ratio=0.9, field_order="tff", chroma=_chroma_dict((2.0, 3.0), (2.0, 3.0)))
    stages, _ = pv.plan_prepass(p)
    assert [type(s) for s in stages] == [avs.Deinterlace, avs.ChromaShift]
    assert stages[1].u_dy == 0.0 and stages[1].u_dx == -2.0          # interlaced: horizontal only
    p = _profile(blend_share=0.2, blend={"is_blended": True}, chroma=_chroma_dict((2.0, 0.0), (2.0, 0.0)))
    assert [type(s) for s in pv.plan_prepass(p)[0]] == [avs.ChromaShift, avs.SRestore]
    assert pv.plan_prepass(_profile(chroma=_chroma_dict((0.3, 0.2), (0.1, 0.4))))[0] == []


def test_ffmpeg_fallback_uses_whole_chroma_pixels():
    assert pv._fallback_chroma_vf([avs.ChromaShift(u_dx=-4, v_dx=-3, v_dy=2)]) == "chromashift=cbh=-2:cbv=0:crh=-2:crv=1"
    assert pv._fallback_chroma_vf([avs.ChromaShift(u_dx=-0.75)]) is None
    assert pv._fallback_chroma_vf([avs.Deinterlace()]) is None


def test_low_frame_rate_clips_are_not_flagged_blended():
    assert "blended" not in _profile(fps=9.99, blend={"is_blended": True}).flags
