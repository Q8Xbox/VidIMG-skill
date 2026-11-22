"""Damaged frames (probe/damage.py): candidates, the survey's sheet, and making the chosen ones again."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vidimg import auto, recipe
from vidimg import pipeline_video as pv
from vidimg.guard import GuardReport
from vidimg.guard import video as gv
from vidimg.probe import damage, scenes
from vidimg.report import closeups

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="ffmpeg is not installed")


def _picture(seed: int, light: float = 128.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.resize((rng.random((54, 72)) * 255).astype(np.float32), (360, 270), interpolation=cv2.INTER_CUBIC)
    return np.clip(cv2.GaussianBlur(base, (0, 0), 0.8) - base.mean() + light, 0, 255).astype(np.uint8)


def _torn(frame: np.ndarray, seed: int = 3) -> np.ndarray:
    """Rows of the frame pushed sideways and a band of noise, as a tape dropout leaves them."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for y in range(90, 200, 4):
        out[y:y + 4] = np.roll(out[y:y + 4], int(rng.integers(20, 60)), axis=1)
    out[140:170] = rng.integers(0, 255, (30, 360), dtype=np.uint8)
    return out


def _clip() -> list[np.ndarray]:
    a = _picture(1)
    seq = [a.copy() for _ in range(40)]
    seq[10] = _torn(a)                                   # a lone torn frame
    seq[20] = np.full_like(a, 235)                       # a white flash between identical frames
    seq[30], seq[31] = _torn(a, 5), _torn(a, 6)          # two damaged frames in a row: not found (see the module)
    return seq


def test_a_frame_that_differs_from_two_agreeing_neighbours_is_a_candidate():
    got = damage.find(iter(_clip()))
    by = {d["frame"]: d for d in got.found}
    assert got.frames == 40 and sorted(by) == [10, 20] and got.total == 2
    assert by[10]["lone"] and by[10]["looks"] == "other" and 0.2 < by[10]["share"] < 0.7
    assert by[20]["lone"] and by[20]["looks"] == "flash" and by[20]["share"] == 1.0
    assert got.to_dict()["lone"] == [10, 20] and got.to_dict()["count"] == 2
    # a strobe: candidates side by side are a run, and none of them is offered
    a, lit = _picture(1), np.clip(_picture(1).astype(np.int16) + 60, 0, 255).astype(np.uint8)
    strobe = damage.find(iter([a, a, a, lit, a, lit, a, lit, a, a, a]))
    assert [d["frame"] for d in strobe.found] == [3, 4, 5, 6, 7] and not any(d["lone"] for d in strobe.found)


def test_cuts_movement_and_ghosts_are_not_candidates():
    a, b = _picture(1), _picture(2, 90.0)
    assert damage.find(iter([a] * 10 + [b] * 10)).found == []                           # a cut
    assert damage.find(iter([np.roll(a, 5 * k, axis=1) for k in range(20)])).found == []   # steady movement
    mix = ((a.astype(np.float32) + b) / 2).astype(np.uint8)
    assert damage.find(iter([a] * 8 + [mix] + [b] * 8)).found == []                     # a mix of two scenes
    small = a.copy()
    small[100:112, 100:118] = 0                                                         # a blink: too little of the frame
    assert damage.find(iter([a] * 5 + [small] + [a] * 5)).found == []
    assert damage.find(iter([])).frames == 0


def test_the_measure_of_a_remade_frame_falls():
    a = _picture(1)
    assert damage.stand_out(a, _torn(a), a) > 5 * damage.stand_out(a, a, a) + 1.0
    assert damage.stand_out(a, a, a) == 0.0


def test_the_same_pass_serves_the_scene_map_and_the_candidates(monkeypatch):
    finder = damage.Finder()
    got = scenes.find(iter(_clip()), fps=25.0, watch=finder.add)
    assert got.frames == 40 and finder.result().frames == 40 and len(finder.result().found) == 2
    monkeypatch.setattr(damage.media, "probe", lambda p, exact=False: {"frames": 41})
    monkeypatch.setattr(damage.ghosts, "read_gray_exact", lambda p: iter(_clip()))
    assert "40 frames of 41" in damage.scan(Path("x.mkv")).error


def test_the_survey_lists_the_lone_candidates_largest_first_and_keeps_flashes_apart(monkeypatch, tmp_path):
    got = damage.find(iter(_clip()))
    seen = {}

    def sheet(src, dst, found, title=""):
        seen.update(found=found, title=title)
        return Path(dst)

    monkeypatch.setattr(closeups, "damage_sheet", sheet)
    said = []
    out = auto._damaged("v1", Path("C:/in/clip.mkv"), tmp_path, got, said.append)
    assert out["count"] == 1 and out["frames"] == [10] and out["flashes"] == 1 and out["in_runs"] == 0
    assert out["sheet"].endswith("v1_damaged.jpg") and "1 candidate damaged frames" in seen["title"]
    assert json.loads(Path(out["list"]).read_text(encoding="utf-8"))["lone"] == [10, 20]
    assert "1 candidate damaged frames, 1 one-tone flashes, 0 in runs" in said[-1]
    assert auto._damaged("v2", Path("C:/in/clip.mkv"), tmp_path, damage.Damage(frames=9), said.append) is None


def test_the_sheet_shows_each_candidate_between_its_neighbours(monkeypatch, tmp_path):
    asked = {}

    def numbered(src, numbers):
        asked["n"] = sorted(set(numbers))
        return {n: cv2.cvtColor(_picture(n % 7), cv2.COLOR_GRAY2BGR) for n in numbers}

    monkeypatch.setattr(closeups, "frames_numbered", numbered)
    found = damage.find(iter(_clip())).found
    out = closeups.damage_sheet(Path("x.mkv"), tmp_path / "d.jpg", found, title="v1 x.mkv")
    assert out and asked["n"] == [9, 10, 11, 19, 20, 21]
    img = cv2.imread(str(out))
    assert img.shape[1] == 1440 and img.shape[0] > 2 * 300
    monkeypatch.setattr(closeups, "frames_numbered", lambda src, numbers: {})
    assert closeups.damage_sheet(Path("x.mkv"), tmp_path / "none.jpg", found) is None


# ----------------------------------------------------------------------------- the recipe and the second file
def test_frames_to_make_again_are_named_in_the_recipe():
    got, advice = recipe.validate({"faces": "drawn", "rebuild_damaged": [16060, 1184, 1184]})
    assert got["rebuild_damaged"] == [1184, 16060]
    assert any("2 frame(s) are made again" in a and "created" in a and "faithful file keeps them" in a for a in advice)
    assert "2 damaged frame(s) made again in a second file" in recipe.describe(got)
    assert recipe.validate({"faces": "drawn"})[0]["rebuild_damaged"] == [] and recipe.validate(got)[0] == got
    assert recipe.from_decision({"content": "anime"}, [])["rebuild_damaged"] == []
    for bad in (True, "all", [1.5], [-1], [True], list(range(recipe.REBUILD_MAX + 1))):
        with pytest.raises(ValueError, match="rebuild_damaged is a list"):
            recipe.validate({"faces": "drawn", "rebuild_damaged": bad})


def test_named_frames_are_made_again_in_a_second_file_and_the_faithful_file_keeps_them(monkeypatch, tmp_path):
    from vidimg import rebuild as rebuild_mod

    faithful = tmp_path / "clip_restored.mp4"
    faithful.write_bytes(b"faithful")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    frames = {"clip.mp4": 120}
    monkeypatch.setattr(pv, "lossless_copy", lambda a, b: Path(b).write_bytes(b"feed") and None)
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"fps": 25.0, "frames": frames.get(Path(p).name, 120)})
    for name in ("fix_av_offset", "fix_aspect"):
        monkeypatch.setattr(pv.media, name, lambda a, b: None)
    monkeypatch.setattr(pv.media, "encoder_args", lambda name: ["-c:v", "copy"])
    monkeypatch.setattr(pv, "carry_side_streams", lambda a, b: None)
    calls = []

    monkeypatch.setattr(pv, "_ffmpeg", lambda: "ffmpeg")

    def again(feed, dst, found, work, drawn=True, log=print, what=""):
        calls.append(list(found))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"remade")
        return {"ok": True, "frames": 120, "replaced": len(found), "error": None}

    monkeypatch.setattr(rebuild_mod, "rebuild", again)
    monkeypatch.setattr(gv, "verify_remade", lambda a, b, **k: GuardReport(True, []))

    class Done:
        returncode, stderr = 0, ""

    def encode(args, **kw):
        Path(args[-1]).write_bytes(Path(args[args.index("-i") + 1]).read_bytes())
        return Done()

    monkeypatch.setattr(pv.subprocess, "run", encode)

    def record():
        return pv.VideoRecord(str(source), {}, "", delivered=str(faithful), passed=True)

    rcp = {"interpolate": None, "rebuild_ghosts": False, "rebuild_damaged": [0, 10, 40, 41, 77, 119]}
    rec = pv.add_interpolation(record(), rcp, tmp_path / "work", "hevc10", True, log=lambda m: None)
    copy = tmp_path / "clip_restored_frames_rebuilt.mp4"
    # the first and the last frame, and two frames side by side, have no sound neighbour: left as they are
    assert calls == [[10, 77]] and rec.interpolated == str(copy) and copy.read_bytes() == b"remade"
    assert faithful.read_bytes() == b"faithful" and rec.delivered == str(faithful)
    assert any("4 of the 6 damaged frames named were left as they are" in n for n in rec.notes)
    assert "2 damaged frames were made again from their neighbours" in rec.notes[-1]
    assert "keeps the damaged frames as they are" in rec.notes[-1] and "not original picture" in rec.notes[-1]
    assert rec.prepass["rebuild_damaged"]["guard_passed"] is True and rec.prepass["rebuild_damaged"]["by_time"] is False
    # after a step that removed a fifth of the frames, the named frames are found by their time
    frames["clip.mp4"] = 150
    calls.clear()
    rec = pv.add_interpolation(record(), {**rcp, "rebuild_damaged": [50, 100]}, tmp_path / "w2", "hevc10", True, log=lambda m: None)
    assert calls == [[40, 80]] and rec.prepass["rebuild_damaged"]["by_time"] is True
    frames["clip.mp4"] = 120
    # the check fails: no second file at all
    monkeypatch.setattr(gv, "verify_remade", lambda a, b, **k: GuardReport(False, [gv.Gate("remade", "fail", 0.2, 0.75, "")]))
    rec = pv.add_interpolation(record(), {**rcp, "rebuild_damaged": [10]}, tmp_path / "w3", "hevc10", True, log=lambda m: None)
    assert rec.interpolated is None and "damaged frames not made again: its check failed: remade" in rec.notes[-1]
    # nothing named: nothing happens
    rec = pv.add_interpolation(record(), {"interpolate": None, "rebuild_ghosts": False, "rebuild_damaged": []},
                               tmp_path / "w4", "hevc10", True, log=lambda m: None)
    assert rec.interpolated is None and rec.notes == []


@needs_ffmpeg
def test_a_remade_copy_must_leave_the_other_frames_alone_and_calm_the_named_ones(tmp_path):
    a = _picture(1)
    seq = [np.roll(a, k, axis=1) for k in range(30)]
    broken = list(seq)
    broken[12] = _torn(seq[12])
    fixed = list(seq)

    def write(path, frames):
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                        "-s", "360x270", "-r", "25", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p10le", str(path)],
                       input=b"".join(f.tobytes() for f in frames), check=True, capture_output=True)
        return path

    src = write(tmp_path / "broken.mkv", broken)
    good = gv.verify_remade(src, write(tmp_path / "fixed.mkv", fixed), rebuilt=[12])
    assert good.passed, good.failed()
    same = gv.verify_remade(src, write(tmp_path / "same.mkv", broken), rebuilt=[12])
    assert same.failed() == ["remade"]                                   # the frame stands out as much as before
    spoiled = list(fixed)
    spoiled[13] = seq[20]                                                # the frame beside the remade one changed
    assert "untouched" in gv.verify_remade(src, write(tmp_path / "spoiled.mkv", spoiled), rebuilt=[12]).failed()
