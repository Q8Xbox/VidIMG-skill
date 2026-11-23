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
    seq[30], seq[31] = _torn(a, 5), _torn(a, 6)          # two damaged frames in a row: found as a run
    return seq


def test_a_frame_that_differs_from_two_agreeing_neighbours_is_a_candidate():
    got = damage.find(iter(_clip()))
    by = {d["frame"]: d for d in got.found}
    assert got.frames == 40 and sorted(by) == [10, 20] and got.total == 2
    assert by[10]["lone"] and by[10]["looks"] == "other" and 0.2 < by[10]["share"] < 0.7
    assert by[20]["lone"] and by[20]["looks"] == "flash" and by[20]["share"] == 1.0
    assert got.to_dict()["lone"] == [10, 20] and got.to_dict()["count"] == 2
    assert [(r["first"], r["last"]) for r in got.runs] == [(30, 31)] and got.runs[0]["share"] > 0.2
    # a strobe: candidates side by side are a run, and none of them is offered
    a, lit = _picture(1), np.clip(_picture(1).astype(np.int16) + 60, 0, 255).astype(np.uint8)
    strobe = damage.find(iter([a, a, a, lit, a, lit, a, lit, a, a, a]))
    assert [d["frame"] for d in strobe.found] == [3, 4, 5, 6, 7] and not any(d["lone"] for d in strobe.found)
    assert strobe.runs == []                                            # a sound frame inside it: not a run of damage
    three = [a] * 6 + [_torn(a, 5), _torn(a, 6), _torn(a, 7)] + [a] * 6
    assert [(r["first"], r["last"]) for r in damage.find(iter(three)).runs] == [(6, 8)]
    four = [a] * 6 + [_torn(a, k) for k in (5, 6, 7, 8)] + [a] * 6     # the scene holds still: a longer run is found
    assert [(r["first"], r["last"]) for r in damage.find(iter(four)).runs] == [(6, 9)]
    seven = [a] * 6 + [_torn(a, k) for k in range(5, 12)] + [a] * 6    # longer than is looked for anywhere
    assert damage.find(iter(seven)).runs == []
    drift = [np.roll(a, 2 * k, axis=1) for k in range(16)]              # a slow pan: after four frames it has moved on
    for k in (6, 7, 8, 9):
        drift[k] = _torn(drift[k], k)
    assert damage.find(iter(drift)).runs == []
    moving = [np.roll(a, 6 * k, axis=1) for k in range(20)]              # the frames either side do not agree
    moving[9], moving[10] = _torn(moving[9]), _torn(moving[10])
    assert damage.find(iter(moving)).runs == []


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

    def sheet(src, dst, found, title="", runs=None):
        seen.update(found=found, title=title, runs=runs)
        return Path(dst)

    monkeypatch.setattr(closeups, "damage_sheet", sheet)
    said = []
    out = auto._damaged("v1", Path("C:/in/clip.mkv"), tmp_path, got, said.append)
    assert out["count"] == 1 and out["frames"] == [10] and out["flashes"] == 1 and out["in_runs"] == 0
    assert out["runs"] == [[30, 31]] and seen["runs"] == got.runs
    assert out["sheet"].endswith("v1_damaged.jpg") and "1 candidate damaged frames" in seen["title"]
    assert json.loads(Path(out["list"]).read_text(encoding="utf-8"))["lone"] == [10, 20]
    assert "1 candidate damaged frames (0 read like tape damage), 1 runs of two to six (0), 1 one-tone flashes, 0 in a flicker" in said[-1]
    only_runs = damage.Damage(frames=40, runs=got.runs)
    assert auto._damaged("v3", Path("C:/in/clip.mkv"), tmp_path, only_runs, said.append)["runs"] == [[30, 31]]
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
    tall = closeups.damage_sheet(Path("x.mkv"), tmp_path / "r.jpg", found, runs=[{"first": 30, "last": 31, "share": 0.4}])
    assert asked["n"] == [9, 10, 11, 19, 20, 21, 29, 30, 31, 32]         # a run with the sound frame either side
    assert cv2.imread(str(tall)).shape[0] > img.shape[0]
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
    calls, asked_regions = [], []

    monkeypatch.setattr(pv, "_ffmpeg", lambda: "ffmpeg")

    def again(feed, dst, found, work, drawn=True, log=print, what="", regions=False):
        calls.append(list(found))
        asked_regions.append(regions)
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
    # the first and the last frame have no sound frame on one side: left. Two frames side by side are a run
    assert calls == [[10, 40, 41, 77]] and asked_regions == [True]
    assert rec.interpolated == str(copy) and copy.read_bytes() == b"remade"
    assert rec.prepass["rebuild_damaged"]["in_runs"] == 2
    assert rec.prepass["rebuild_damaged"]["asked"] == 6 and rec.prepass["rebuild_damaged"]["made_from_neighbours"] == 4
    assert faithful.read_bytes() == b"faithful" and rec.delivered == str(faithful)
    assert any("2 of the 6 damaged frames named were left as they are" in n for n in rec.notes)
    assert "4 damaged frames were made again from their neighbours" in rec.notes[-1]
    assert "keeps the damaged frames as they are" in rec.notes[-1] and "not original picture" in rec.notes[-1]
    assert rec.prepass["rebuild_damaged"]["guard_passed"] is True and rec.prepass["rebuild_damaged"]["by_time"] is False
    # after a step that removed a fifth of the frames, the named frames are found by their time
    frames["clip.mp4"] = 150
    calls.clear()
    rec = pv.add_interpolation(record(), {**rcp, "rebuild_damaged": [50, 100]}, tmp_path / "w2", "hevc10", True, log=lambda m: None)
    assert calls == [[40, 80]] and rec.prepass["rebuild_damaged"]["by_time"] is True
    frames["clip.mp4"] = 120
    # four named frames in a row: made where the scene holds still, left where it moves
    four = {**rcp, "rebuild_damaged": [20, 21, 22, 23, 60]}
    calls.clear()
    monkeypatch.setattr(pv, "_still_runs", lambda feed, runs: (runs, []))
    rec = pv.add_interpolation(record(), four, tmp_path / "w5", "hevc10", True, log=lambda m: None)
    assert calls == [[20, 21, 22, 23, 60]] and rec.prepass["rebuild_damaged"]["left"] == 0
    calls.clear()
    monkeypatch.setattr(pv, "_still_runs", lambda feed, runs: ([r for r in runs if r[1] - r[0] < 3],
                                                               [[a, z, 0.8] for a, z in runs if z - a >= 3]))
    rec = pv.add_interpolation(record(), four, tmp_path / "w6", "hevc10", True, log=lambda m: None)
    assert calls == [[60]] and any("4 of the 5 damaged frames named were left" in n and "a scene that moves" in n for n in rec.notes)
    assert rec.prepass["rebuild_damaged"]["runs_in_movement"] == [[20, 23, 0.8]]
    # seven in a row are left whatever the scene does
    calls.clear()
    rec = pv.add_interpolation(record(), {**rcp, "rebuild_damaged": [20, 21, 22, 23, 24, 25, 26, 60]}, tmp_path / "w7",
                               "hevc10", True, log=lambda m: None)
    assert calls == [[60]]
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


# ----------------------------------------------------------------------------- runs, and only the damaged part
def test_named_frames_side_by_side_are_runs():
    from vidimg import rebuild

    assert rebuild.runs_of([41, 10, 40, 77, 10]) == [[10, 10], [40, 41], [77, 77]]
    assert rebuild.runs_of([]) == []


def _yuv(y: np.ndarray, tone: int = 512) -> bytes:
    h, w = y.shape
    c = np.full((h // 2, w // 2), tone, "<u2").tobytes()
    return (y.astype("<u2") * 4).tobytes() + c + c


def test_only_the_part_that_differs_is_taken_from_the_made_frame():
    from vidimg import rebuild

    w, h = 360, 270
    good = _picture(1)
    torn = good.copy()
    torn[120:150] = 255 - torn[120:150]                                  # one band of the frame is damaged
    made = np.clip(good.astype(np.int16) + 2, 0, 255).astype(np.uint8)   # the made frame: the picture, a touch off
    out, share = rebuild.blend_region(_yuv(torn), _yuv(made, 600), w, h)
    y, u, _ = rebuild._planes(out, w, h)
    y8 = (y // 4).astype(np.int16)
    assert 0.05 < share < 0.5
    assert np.array_equal(y8[:60], torn[:60]) and np.array_equal(y8[220:], torn[220:])   # the sound part: the stored frame
    assert np.abs(y8[125:145] - made[125:145]).max() <= 1                # the damaged band: the made frame
    assert u[0, 0] == 512 and u[67, 90] == 600                           # colour follows the same region
    whole, share = rebuild.blend_region(_yuv(255 - good), _yuv(made), w, h)
    assert share == 1.0 and whole == _yuv(made)                          # nothing sound left to keep
    same, share = rebuild.blend_region(_yuv(good), _yuv(made), w, h)
    assert share == 0.0 and same == _yuv(good)                           # the stored frame agrees: nothing is created


@needs_ffmpeg
def test_a_run_is_made_from_the_sound_frames_either_side_and_the_rest_passes_through(tmp_path, monkeypatch):
    from vidimg import rebuild

    a = _picture(1)
    clean = [np.roll(a, k, axis=0) for k in range(24)]                   # slow, steady movement
    broken = list(clean)
    broken[5] = _torn(clean[5])
    for i in (12, 13):
        broken[i] = _torn(clean[i], i)

    def write(path, frames, rate="25"):
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                        "-s", "360x270", "-r", rate, "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p10le", str(path)],
                       input=b"".join(f.tobytes() for f in frames), check=True, capture_output=True)
        return path

    def read(path):
        raw = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path), "-pix_fmt", "gray",
                              "-f", "rawvideo", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
        return [np.frombuffer(raw[k * 97200:(k + 1) * 97200], np.uint8).reshape(270, 360) for k in range(len(raw) // 97200)]

    asked = []

    def rife(src, dst, processor, fps_multiplier=2, **kw):               # stands in for RIFE: straight lines between frames
        asked.append(fps_multiplier)
        fr = [f.astype(np.float32) for f in read(src)]
        out = []
        for x, y in zip(fr, fr[1:]):
            out += [((x * (fps_multiplier - j) + y * j) / fps_multiplier).astype(np.uint8) for j in range(fps_multiplier)]
        write(dst, out + [fr[-1].astype(np.uint8)])
        return SimpleNamespace(ok=True, error=None)

    monkeypatch.setattr(rebuild.video2x, "run", rife)
    feed = write(tmp_path / "feed.mkv", broken)
    out = tmp_path / "made" / "feed.mkv"
    info = rebuild.rebuild(feed, out, [13, 5, 12], tmp_path / "work", log=lambda m: None, what="damaged frames", regions=True)
    assert info["ok"] and info["replaced"] == 3 and info["runs"] == 1 and sorted(asked) == [2, 3]
    assert 0.3 < info["picture_created"] < 3.0                           # less than three whole frames were created
    got, stored = read(out), read(feed)
    assert len(got) == 24
    for i in range(24):
        if i not in (5, 12, 13):
            assert np.array_equal(got[i], stored[i]), i                  # every other frame as stored
    # the run: each frame close to the clean one (a third and two thirds of the way between frames 11 and 14)
    for i in (5, 12, 13):
        assert float(np.abs(got[i].astype(int) - clean[i]).mean()) < 6.0 < float(np.abs(broken[i].astype(int) - clean[i]).mean()), i
    assert np.array_equal(got[12][:40], stored[12][:40])                 # well above the tear: the stored frame itself
    rep = gv.verify_remade(feed, out, rebuilt=[5, 12, 13])
    assert rep.passed, rep.failed()
    # more than six in a row: refused, the caller leaves such runs out
    assert "more than 6" in rebuild.rebuild(feed, out, list(range(3, 10)), tmp_path / "w2", log=lambda m: None)["error"]
    # a long run is made only between two frames that show nearly the same picture
    assert pv._still_runs(feed, [[5, 5], [12, 13]]) == ([[5, 5], [12, 13]], [])          # nothing long: nothing read
    calm = write(tmp_path / "calm.mkv", [a] * 24)
    kept, moving = pv._still_runs(calm, [[5, 5], [8, 11]])
    assert kept == [[5, 5], [8, 11]] and moving == []                                    # a held picture: still enough
    far = write(tmp_path / "far.mkv", [np.roll(a, 9 * k, axis=0) for k in range(24)])    # nine pixels a frame
    kept, moving = pv._still_runs(far, [[5, 5], [8, 11]])
    assert kept == [[5, 5]] and moving[0][:2] == [8, 11] and moving[0][2] > 0.3


def test_a_long_run_needs_two_ends_that_show_the_same_picture():
    from vidimg import rebuild

    a = _picture(1)
    assert rebuild.still_enough(a, a) == (True, 0.0)
    talking = a.copy()
    talking[100:150, 150:230] = 255 - talking[100:150, 150:230]         # a mouth moves: a small part of the frame
    ok, share = rebuild.still_enough(a, talking)
    assert ok and 0 < share < 0.2
    ok, share = rebuild.still_enough(a, _picture(2))
    assert not ok and share > 0.6


# ----------------------------------------------------------------------------- the hint: reads like tape damage
def _readings(n: int = 200, level: float = 0.0004) -> np.ndarray:
    return np.full(n, level, np.float64)


def test_a_frame_reads_like_tape_damage_when_it_is_high_and_stands_out_from_the_frames_around_it():
    t = _readings()
    t[50] = 0.0030                                                       # high, and alone
    assert damage.tape_like(t, 50) and not damage.tape_like(t, 51)
    t[100] = 0.0009                                                      # stands out, but under the level
    assert not damage.tape_like(t, 100)
    busy = _readings(level=0.0040)                                       # a busy scene: every frame reads high
    busy[60] = 0.0050
    assert not damage.tape_like(busy, 60)                                # high, but like the frames around it
    assert not damage.tape_like(t, 999) and not damage.tape_like(None, 5)
    # frames of a run do not count as "the frames around it"
    run = _readings()
    run[80:84] = 0.0030
    assert all(damage.tape_like(run, i, (80, 83)) for i in range(80, 84))
    edge = _readings(3)
    edge[1] = 0.01
    assert not damage.tape_like(edge, 1)                                 # nothing far enough away to compare with


def test_candidates_and_runs_are_marked_only_with_readings_for_exactly_this_video():
    got = damage.find(iter(_clip()))
    t = _readings(40)
    t[10] = 0.0030                                                       # the torn frame
    t[30:32] = 0.0030                                                    # the torn run
    damage.mark_tape(got, t)
    by = {d["frame"]: d for d in got.found}
    assert by[10]["tape"] is True and by[20]["tape"] is False            # the flash is not tape damage
    assert got.runs[0]["tape"] is True
    fresh = damage.find(iter(_clip()))
    assert "tape" not in damage.mark_tape(fresh, _readings(39)).found[0]  # a reading short: nothing is marked
    assert "tape" not in damage.mark_tape(fresh, None).found[0]


def test_the_readings_come_from_one_ffmpeg_pass(monkeypatch):
    seen = {}

    def run(args, **kw):
        seen["args"] = args
        return SimpleNamespace(stdout="frame:0 pts:0\nlavfi.signalstats.TOUT=0.000412\nframe:1 pts:1\n"
                                      "lavfi.signalstats.TOUT=0.003\nlavfi.signalstats.TOUT=1e-05\n", returncode=0)

    monkeypatch.setattr(damage.media, "ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("subprocess.run", run)
    got = damage.outliers(Path("x.mkv"))
    assert list(got) == [0.000412, 0.003, 1e-05]
    graph = seen["args"][seen["args"].index("-vf") + 1]
    assert "signalstats=stat=tout" in graph and "file=-" in graph and "passthrough" in seen["args"]
    monkeypatch.setattr("subprocess.run", lambda args, **kw: SimpleNamespace(stdout="", returncode=1))
    assert damage.outliers(Path("x.mkv")) is None

    def broken(args, **kw):
        raise OSError("no ffmpeg")

    monkeypatch.setattr("subprocess.run", broken)
    assert damage.outliers(Path("x.mkv")) is None


def test_the_survey_lists_and_shows_tape_like_candidates_first(monkeypatch, tmp_path):
    a = _picture(1)
    seq = [a.copy() for _ in range(60)]
    seq[10] = _torn(a)
    small = a.copy()
    small[60:230, 40:320] = 255 - small[60:230, 40:320]                  # a larger change, drawn on purpose
    seq[40] = small
    got = damage.find(iter(seq))
    assert [d["frame"] for d in sorted(got.found, key=lambda d: -d["share"])] == [40, 10]
    t = _readings(60)
    t[10] = 0.0030
    monkeypatch.setattr(damage, "outliers", lambda src: t)
    seen = {}
    monkeypatch.setattr(closeups, "damage_sheet", lambda src, dst, found, title="", runs=None: seen.update(found=found) or Path(dst))
    said = []
    out = auto._damaged("v1", Path("C:/in/clip.mkv"), tmp_path, got, said.append)
    assert out["frames"] == [10, 40] and out["tape_like"] == [10] and out["tape_like_runs"] == []
    assert "2 candidate damaged frames (1 read like tape damage)" in said[-1]
    assert {d["frame"]: d["tape"] for d in seen["found"]} == {10: True, 40: False}
    # without readings the order is by size, as before
    monkeypatch.setattr(damage, "outliers", lambda src: None)
    plain = auto._damaged("v2", Path("C:/in/clip.mkv"), tmp_path, damage.find(iter(seq)), said.append)
    assert plain["frames"] == [40, 10] and plain["tape_like"] == []


def test_the_sheet_puts_tape_like_candidates_first_and_says_so(monkeypatch, tmp_path):
    asked = {}

    def numbered(src, numbers):
        asked["n"] = list(numbers)
        return {n: cv2.cvtColor(_picture(n % 7), cv2.COLOR_GRAY2BGR) for n in numbers}

    monkeypatch.setattr(closeups, "frames_numbered", numbered)
    monkeypatch.setattr(closeups, "DAMAGE_SHOWN", 1)                     # room for one: it must be the tape-like one
    found = [{"frame": 40, "share": 0.9, "looks": "other", "lone": True, "tape": False},
             {"frame": 10, "share": 0.2, "looks": "other", "lone": True, "tape": True}]
    assert closeups.damage_sheet(Path("x.mkv"), tmp_path / "d.jpg", found) and asked["n"] == [9, 10, 11]
