"""Black borders and padded repeat frames: measuring them, the two corrections' own checks, and Claude's part in
the automatic flow (findings to judge on close-ups, recipes compared by eye, a review of the result)."""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vidimg import auto, jobs
from vidimg.adapters import avisynth as avs
from vidimg.probe import framing

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="ffmpeg is not installed")


# ----------------------------------------------------------------------------- measuring
def _steps(n: int, copy_every: int | None = None, twos: bool = False) -> np.ndarray:
    s = np.full(n, 2.0)
    if copy_every:
        s[copy_every - 1::copy_every] = 0.05
    if twos:
        s[1::2] = 0.02
    return s


def test_a_copy_every_sixth_frame_is_a_25_fps_original():
    st = framing.summarize([_steps(240, 6)] * 3, 29.97, [0.0, 100.0, 200.0])
    assert st.is_padded and st.cycle == 6 and st.original_fps == "25" and 0.15 < st.share < 0.18
    assert st.example_s is not None
    five = framing.summarize([_steps(240, 5)] * 3, 29.97)
    assert five.cycle == 5 and Fraction(five.original_fps) == Fraction(24000, 1001)


def test_held_drawings_and_other_rates_are_not_padding():
    assert not framing.summarize([_steps(240, twos=True)] * 3, 29.97).is_padded      # animation on twos
    assert not framing.summarize([_steps(240, 6)] * 3, 25.0).is_padded               # not a 29.97 file
    assert not framing.summarize([_steps(240)] * 3, 29.97).is_padded                 # no copies at all
    assert not framing.summarize([_steps(40, 6)], 29.97).is_padded                   # too few to call a cycle
    still = np.full(240, 0.01)                                                       # a still scene: nothing moves
    assert framing.summarize([still], 29.97).judged == 0


def test_crop_keeps_even_edges_and_refuses_what_is_not_a_bar():
    assert framing.even_crop(640, 480, {"left": 8, "right": 22, "top": 0, "bottom": 4}) == (610, 476, 8, 0)
    assert framing.even_crop(640, 480, {"left": 7, "right": 3, "top": 5, "bottom": 0}) == (632, 476, 6, 4)
    assert framing.even_crop(640, 480, {"left": 1, "right": 1, "top": 0, "bottom": 0}) is None
    assert framing.even_crop(640, 480, {"left": 200, "right": 200, "top": 0, "bottom": 0}) is None
    assert framing.Borders(left=2, top=2).any is False and framing.Borders(right=4).any is True


def test_repeat_removal_only_chooses_frames():
    st = avs.DropRepeats(frate="25/1", in_fps="30000/1001")
    assert st.rate() == Fraction(1001, 1200) and st.lines() == ["TDecimate(mode=2, rate=25.000000)"]
    text = avs.Script(Path("C:/in/a.mp4"), [st], Path("C:/work/a.lwi"), Fraction(30000, 1001)).text()
    assert avs.lint(text) == [] and "AssumeFPS(25, 1)" in text


# ----------------------------------------------------------------------------- the checks, on real files
def _make(dst: Path, vf: str, rate: str = "25", seconds: int = 4) -> Path:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size=320x240:rate={rate}:duration={seconds}", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-vf", vf, "-c:v", "ffv1", "-pix_fmt", "yuv420p",
                    "-c:a", "flac", "-shortest", str(dst)], check=True, stdin=subprocess.DEVNULL, capture_output=True)
    return dst


@needs_ffmpeg
def test_borders_are_measured_and_the_crop_check_tells_bars_from_picture(tmp_path):
    from vidimg.guard.video import verify_crop

    barred = _make(tmp_path / "barred.mkv", "crop=290:236:0:0,pad=320:240:8:0:black")
    b = framing.borders(barred, frames=20)
    assert (b.left, b.right, b.top, b.bottom) == (8, 22, 0, 4) and b.any
    crop = framing.even_crop(320, 240, b.to_dict())
    assert crop == (290, 236, 8, 0)
    good = tmp_path / "good.mkv"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(barred), "-vf",
                    "crop=290:236:8:0", "-c:v", "ffv1", "-c:a", "copy", str(good)], check=True, stdin=subprocess.DEVNULL, capture_output=True)
    rep = verify_crop(barred, good, crop=crop)
    assert rep.passed, rep.failed()
    # cutting into the picture: what was removed is not black
    deep = tmp_path / "deep.mkv"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(barred), "-vf",
                    "crop=200:236:60:0", "-c:v", "ffv1", "-c:a", "copy", str(deep)], check=True, stdin=subprocess.DEVNULL, capture_output=True)
    assert "removed" in verify_crop(barred, deep, crop=(200, 236, 60, 0)).failed()
    # a "crop" that also changed the picture it kept
    soft = tmp_path / "soft.mkv"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(barred), "-vf",
                    "crop=290:236:8:0,gblur=sigma=1", "-c:v", "ffv1", "-c:a", "copy", str(soft)], check=True, stdin=subprocess.DEVNULL, capture_output=True)
    assert "picture" in verify_crop(barred, soft, crop=crop).failed()
    assert not framing.borders(_make(tmp_path / "full.mkv", "null"), frames=20).any


@needs_ffmpeg
def test_the_repeat_check_passes_copies_removed_and_fails_pictures_dropped(tmp_path):
    from vidimg.guard.video import verify_derepeat

    original = _make(tmp_path / "original.mkv", "null", seconds=8)
    padded = _make(tmp_path / "padded.mkv", "fps=30000/1001", seconds=8)            # copies fill the higher rate
    found = framing.repeats(padded, count=1, seconds=8.0)
    assert found.is_padded and found.cycle == 6 and found.original_fps == "25"
    assert not framing.repeats(original, count=1, seconds=8.0).is_padded
    rep = verify_derepeat(padded, original, in_fps="30000/1001", out_fps="25", windows=2, seconds=3.0)
    assert rep.passed, [(g.name, g.detail) for g in rep.gates if g.status == "fail"]
    # a lower rate than the original drops real pictures, not only copies
    thinned = _make(tmp_path / "thinned.mkv", "fps=15", seconds=8)
    bad = verify_derepeat(padded, thinned, in_fps="30000/1001", out_fps="15", windows=2, seconds=3.0)
    assert "pictures" in bad.failed()


# ----------------------------------------------------------------------------- Claude's part in the flow
def _plan(out: Path) -> dict:
    video = {"content": "anime", "model": "esrgan-v3", "model_strict": False, "scale": 2, "texture": None,
             "mask_tape_noise": False, "preview_only": False, "skip": False, "remove_repeats": False,
             "crop_borders": False}
    plan = {"version": 1, "created": "now", "out": str(out), "inputs": [], "ignored_files": 0, "contact_sheets": [],
            "items": [
                {"id": "v1", "type": "video", "source": "C:/in/clip.mp4", "guide": {}, "decision": dict(video),
                 "decided_by": "analysis", "reasons": [], "state": "pending", "closeups": "C:/out/survey/v1_closeups.jpg",
                 "findings": [{"finding": "black_borders", "what": "black bars", "decision": "crop_borders"}]},
                {"id": "s1", "type": "series", "source": "C:/in/Show", "folder": "C:/in/Show", "episodes": [1, 2, 3],
                 "analysed_episode": "C:/in/Show/Show - 02.mkv", "guide": {},
                 "decision": {k: v for k, v in video.items() if k not in ("preview_only", "remove_repeats", "crop_borders")},
                 "decided_by": "analysis", "reasons": [], "state": "pending"}]}
    out.mkdir(parents=True, exist_ok=True)
    auto.save(plan)
    return plan


def test_findings_name_the_decision_and_what_to_look_at():
    prof = SimpleNamespace(flags=["black_borders", "padded_repeats"], fps=29.97,
                           borders={"left": 8, "right": 22, "top": 0, "bottom": 4},
                           repeats={"cycle": 6, "share": 0.083, "original_fps": "25"})
    got = auto._findings(prof)
    assert [f["decision"] for f in got] == ["crop_borders", "remove_repeats"] and all(f["look"] for f in got)
    assert "every 6th frame" in got[1]["what"] and "22" in got[0]["what"]
    assert auto._findings(SimpleNamespace(flags=["sd"], fps=25.0, borders={}, repeats={})) == []


def test_framing_is_decided_per_video_and_the_summary_points_at_the_closeups(tmp_path):
    _plan(tmp_path)
    plan = auto.decide(tmp_path, {"v1": {"crop_borders": "true", "remove_repeats": True, "reason": "bars are black"}})
    v1 = plan["items"][0]
    assert v1["decision"]["crop_borders"] is True and v1["decision"]["remove_repeats"] is True
    with pytest.raises(ValueError):
        auto.decide(tmp_path, {"s1": {"crop_borders": True}})          # a series is decided as one: not offered
    with pytest.raises(ValueError):
        auto.decide(tmp_path, {"v1": {"crop_borders": "maybe"}})
    s = auto.summary(auto.load(tmp_path))
    assert s["items"][0]["closeups"] and s["items"][0]["findings"] and "closeups" in s["next"] and "try_recipes" in s["next"]


def test_a_review_is_asked_for_recorded_and_can_send_an_item_back(tmp_path):
    plan = _plan(tmp_path)
    plan["items"][0].update(state="delivered", delivered="x_restored.mp4", result={"chosen": "anime esrgan-v3"})
    plan["items"][1].update(state="skipped")
    plan["report"] = str(auto.write_report(plan))
    auto.save(plan)
    assert "review_results" in auto.summary(auto.load(tmp_path))["next"]
    for bad in ({"v1": {"verdict": "fine", "notes": "x"}}, {"v1": {"verdict": "good"}}, {"s1": {"verdict": "good", "notes": "x"}},
                {"zz": {"verdict": "good", "notes": "x"}}):
        with pytest.raises(ValueError):
            auto.review(tmp_path, bad)
    plan = auto.review(tmp_path, {"v1": {"verdict": "acceptable", "notes": "lines clean; the sky is a little flat"}})
    assert plan["items"][0]["review"]["verdict"] == "acceptable" and auto.summary(plan)["next"] is None
    assert "the sky is a little flat" in Path(plan["report"]).read_text(encoding="utf-8")
    plan = auto.review(tmp_path, {"v1": {"verdict": "redo", "notes": "backgrounds turned to blobs"}})
    assert plan["items"][0]["state"] == "pending"
    plan = auto.decide(tmp_path, {"v1": {"texture": 1, "model": "cugan"}})       # a new recipe: the old review goes
    assert "review" not in plan["items"][0] and plan["items"][0]["state"] == "pending"


def test_recipes_are_validated_and_their_outcome_is_kept_for_the_decision(tmp_path, monkeypatch):
    from vidimg import pipeline_video
    from vidimg import texture as texture_pass
    from vidimg.report import closeups

    _plan(tmp_path)
    for bad in ([{"model": "cugan"}], [{"model": "cugan"}] * 5, [{"model": "gfpgan"}, {"model": "cugan"}],
                [{"skip": True}, {"model": "cugan"}], [{"texture": 3}, {"texture": 1}]):
        with pytest.raises(ValueError):
            auto.trial(tmp_path, "v1", bad, log=lambda m: None)
    with pytest.raises(ValueError):
        auto.trial(tmp_path, "s1", [{"crop_borders": True}, {"model": "cugan"}], log=lambda m: None)
    with pytest.raises(ValueError):
        auto.trial(tmp_path, "nope", [{}, {}], log=lambda m: None)
    seen = []

    def fake(src, out, **kw):
        seen.append(kw)
        rejected = kw["model"] == "cugan"
        att = SimpleNamespace(label=f"anime {kw['model']}", previews=3, passed_previews=0 if rejected else 3,
                              failed=[["flicker"], [], ["flicker"]] if rejected else [[], [], []], vmaf=[97.0] * 3)
        return SimpleNamespace(attempts=[att], notes=["preview only: full run not started"], chosen=None)

    monkeypatch.setattr(pipeline_video, "restore_video", fake)
    monkeypatch.setattr(closeups, "frames_at", lambda *a, **k: [])
    monkeypatch.setattr(texture_pass, "apply", lambda *a, **k: SimpleNamespace(ok=False))
    plan = auto.trial(tmp_path, "v1", [{"name": "anime-v3", "model": "esrgan-v3", "crop_borders": True},
                                       {"name": "cugan", "model": "cugan", "texture": 0.5}], log=lambda m: None)
    assert [k["model"] for k in seen] == ["esrgan-v3", "cugan"] and all(k["preview_only"] and k["model_strict"] for k in seen)
    assert seen[0]["crop_borders"] is True and seen[1]["crop_borders"] is False and seen[0]["scale"] == 2
    got = plan["items"][0]["trials"]
    assert [(r["name"], r["guard_passed"]) for r in got["recipes"]] == [("anime-v3", True), ("cugan", False)]
    assert got["recipes"][1]["failed_gates"] == ["flicker"] and "guard passed" in got["next"]
    assert plan["items"][0]["decision"]["model"] == "esrgan-v3"                 # a trial decides nothing
    assert not (tmp_path / "trials" / "v1" / "run").exists()
    assert auto.summary(plan)["items"][0]["trials"]["recipes"][0]["model"] == "anime esrgan-v3"
    page = auto.write_report(plan).read_text(encoding="utf-8")
    assert "Recipes compared" in page and "rejected by the guard: flicker" in page and "black bars" in page


def test_the_recipe_trial_is_a_job_kind():
    assert jobs.KINDS["try-recipes"] == "try-recipes"


# ----------------------------------------------------------------------------- close-ups
def test_closeups_show_line_art_not_the_channel_logo(tmp_path):
    from vidimg.report import closeups

    rng = np.random.default_rng(1)
    frames = []
    for k in range(3):
        f = np.full((240, 320, 3), 60 + 60 * k, np.uint8)
        f[20:50, 250:300] = rng.integers(0, 255, (30, 50, 3), dtype=np.uint8) if k == 0 else frames[0][20:50, 250:300]
        y = 120 + 20 * k                                         # line art that differs from frame to frame
        f[y:y + 30:14, 40:140] = 0
        frames.append(f)
    assert closeups.busiest(frames[1], (80, 60))[0] > 150        # alone, the logo is the busiest spot
    mask = closeups.overlay_mask(frames)
    assert mask[30, 270] and not mask[150, 80]
    x, y = closeups.busiest(frames[1], (80, 60), ignore=mask)
    assert x < 150 and 80 <= y <= 150                            # with the mask, the lines
    assert closeups.overlay_mask(frames[:1]) is None
    sheet = closeups.recipe_sheet([("source", [frames[0], None]), ("1 anime-v3", [frames[1], frames[2]])],
                                  tmp_path / "recipes.jpg", cell=200)
    assert sheet and sheet.exists()
    assert closeups.recipe_sheet([("source", [])], tmp_path / "none.jpg") is None


@needs_ffmpeg
def test_closeup_frames_follow_the_tools_colour_rule(tmp_path):
    from vidimg.report import closeups

    def read(path, size, matrix):
        raw = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", "0.2", "-i", str(path),
                              "-frames:v", "1", "-vf", f"scale=in_color_matrix={matrix},format=bgr24", "-f", "rawvideo",
                              "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
        return np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3).astype(int)

    for size, rule, other in (((1280, 720), "bt709", "bt601"), ((640, 480), "bt601", "bt709")):   # untagged files
        clip = tmp_path / f"{size[1]}.mkv"
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=size={size[0]}x{size[1]}:rate=25:duration=1", "-pix_fmt", "yuv420p", "-c:v", "ffv1",
                        str(clip)], check=True, stdin=subprocess.DEVNULL, capture_output=True)
        got = closeups.frames_at(clip, 0.2)[0].astype(int)
        assert float(np.abs(got - read(clip, size, rule)).mean()) < 0.01       # HD as BT.709, SD as BT.601 ...
        assert float(np.abs(got - read(clip, size, other)).mean()) > 1.0       # ... and the two do differ
    assert len(closeups.frames_at(clip, 0.0, 3)) == 3


@needs_ffmpeg
def test_a_frame_can_be_asked_for_by_its_number(tmp_path):
    from vidimg.report import closeups

    clip = tmp_path / "count.mkv"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=10:duration=2", "-pix_fmt", "yuv420p", "-c:v", "ffv1", str(clip)],
                   check=True, stdin=subprocess.DEVNULL, capture_output=True)
    run = closeups.frames_at(clip, 0.0, 8)
    for n in (0, 5, 7):
        assert np.array_equal(closeups.frames_at(clip, 99.0, index=n)[0], run[n])     # the time is ignored
    assert not np.array_equal(run[5], run[6])
