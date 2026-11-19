"""Tape defects: head-switching band, flagging, clipped highlights, the opt-in bottom mask and the notes.
Synthetic frames only (no video files, no AviSynth/VapourSynth)."""

from fractions import Fraction

import cv2
import numpy as np

from vidimg import pipeline_video as pv
from vidimg.adapters import avisynth as avs
from vidimg.adapters import vapoursynth as vsb
from vidimg.probe import tape
from vidimg.probe.video import VideoProfile, _flags


def _picture(h=480, w=720, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3.0)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 180 + 40).astype(np.uint8)


def _head_switching(y: np.ndarray, rows=8, shift=25) -> np.ndarray:
    out = y.copy()
    out[-rows:] = np.roll(out[-rows:], shift, axis=1)       # the picture continues, displaced sideways
    return out


def test_row_shift_recovers_a_sideways_displacement():
    y = _picture()
    s, p = tape.row_shift(y[100], np.roll(y[100], 7))
    assert abs(s - 7) < 0.5 and p > 0.9


def test_head_switching_band_is_found_and_content_edges_are_not():
    y = _picture()
    assert tape.bottom_break(_head_switching(y)) == (8, 0)
    assert tape.bottom_break(y) == (0, 0)
    overlay = y.copy()                                          # a caption/ticker: unrelated texture, no shift
    overlay[-10:] = _picture(seed=7)[-10:]
    assert tape.bottom_break(overlay)[0] == 0
    bordered = _head_switching(y)                               # a capture's own black border below the band
    bordered[-6:] = 16
    rows, border = tape.bottom_break(bordered)
    assert border == 6 and rows >= 2


def test_flagging_needs_a_bend_that_grows_towards_the_top():
    y = _picture()
    bent = y.copy()
    for r in range(24):
        bent[r] = np.roll(bent[r], int(round(10 * ((24 - r) / 24) ** 2)))
    assert tape.top_bend(bent) >= tape.FLAG_PX
    assert tape.top_bend(y) < tape.FLAG_PX
    sheared = np.stack([np.roll(row, r // 6) for r, row in enumerate(y)])   # a diagonal everywhere, not a bend
    assert tape.top_bend(sheared) < tape.FLAG_PX
    for step in (1, 5):          # a diagonal only near the top (an owner cartoon's curtain / staircase): not a bend
        local = y.copy()
        for r in range(60):
            local[r] = np.roll(y[r], (60 - r) * step // 2)
        assert tape.top_bend(local) < tape.FLAG_PX


def test_only_solid_superwhite_patches_count_as_clipped():
    frames = []
    for k in range(10):
        f = _picture(seed=k)
        f[100:180, 200:320] = 254                                # a blown-out window
        frames.append(f)
    d = tape.summarize(frames)
    assert d.clipped and d.clipped_pixels > 0.02
    text = []
    for k in range(10):
        f = _picture(seed=k)
        f[300:340:4, 100:600] = 254                              # thin superwhite lettering strokes
        text.append(f)
    assert not tape.summarize(text).clipped
    assert not tape.summarize(frames, limited_range=False).clipped   # full range: 255 is just white


def test_summary_decides_on_a_consistent_band():
    y = _picture()
    frames = [_head_switching(_picture(seed=k), shift=10 + 3 * k) for k in range(8)]
    d = tape.summarize(frames)
    assert d.head_switching and d.head_switching_rows == 8 and d.head_switching_share == 1.0
    assert not tape.summarize([y] * 8).head_switching


def _profile(**kw) -> VideoProfile:
    base = dict(path="x", width=720, height=480, fps=29.97, frames=1000, duration=40.0, codec="h264",
                bitrate_kbps=None, bits_per_pixel=None, interlaced_ratio=0.0, field_order="progressive",
                blockiness=None, blur=None)
    base.update(kw)
    p = VideoProfile(**base)
    p.flags = _flags(p)
    return p


HS = {"head_switching": True, "head_switching_rows": 8, "head_switching_share": 0.9, "bottom_border_rows": 0}


def test_flags_notes_and_the_opt_in_mask():
    p = _profile(tape=HS | {"flagging": True, "flagging_px": 7.7, "flagging_share": 0.44, "clipped": True,
                            "clipped_share": 0.3})
    assert {"head_switching", "top_flagging", "clipped_highlights"} <= set(p.flags)
    notes = pv.tape_notes(p)
    assert "left as it is" in notes[0] and "time-base corrector" in notes[1] and "cannot be restored" in notes[2]
    assert pv.plan_prepass(p)[0] == []                          # reported only, unless asked
    stages, why = pv.plan_prepass(p, mask_tape_noise=True)
    assert [type(s) for s in stages] == [avs.MaskBottom] and stages[0].rows == 10 and "masked" in why
    assert "masked black" in pv.tape_notes(p, masked=True)[0]


def test_mask_rows_round_up_and_follow_field_handling():
    p = _profile(interlaced_ratio=0.9, field_order="tff", tape=HS | {"head_switching_rows": 7})
    stages, _ = pv.plan_prepass(p, mask_tape_noise=True)
    assert [type(s) for s in stages] == [avs.Deinterlace, avs.MaskBottom]
    assert stages[1].rows == 12                                 # 7 + 2 margin -> multiple of 4 (interlaced)
    assert pv.mask_stage(_profile(tape=HS | {"head_switching_rows": 5, "bottom_border_rows": 4})).rows == 12
    assert pv.mask_stage(_profile()) is None


def test_mask_stage_in_both_engines():
    m = avs.MaskBottom(rows=10)
    assert m.lines() == ["Crop(0, 0, 0, -10)", "AddBorders(0, 0, 0, 10)"] and m.rate() == 1
    script = avs.Script("in.mp4", [avs.Deinterlace(), m], "c.lwi", Fraction(30000, 1001)).text()
    assert not avs.lint(script) and script.index("QTGMC(") < script.index("Crop(")
    sp = vsb.spec("in.mp4", [m], "c.lwi", {}, Fraction(25))
    assert sp["stages"] == [{"kind": "mask", "rows": 10}] and vsb.supports([m])
