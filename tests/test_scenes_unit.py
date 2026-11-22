"""The scene map (probe/scenes.py): cuts, kinds of scene, and what the survey shows of them."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vidimg import auto
from vidimg.probe import scenes
from vidimg.report import closeups


def _picture(seed: int, light: float = 128.0, contrast: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.resize((rng.random((54, 72)) * 255).astype(np.float32), (360, 270), interpolation=cv2.INTER_CUBIC)
    base = cv2.GaussianBlur(base, (0, 0), 0.8)
    return np.clip((base - base.mean()) * contrast + light, 0, 255).astype(np.uint8)


def _night(seed: int) -> np.ndarray:                 # dark and nearly flat
    return _picture(seed, light=30.0, contrast=0.04)


def _moving(base: np.ndarray, k: int) -> np.ndarray:
    """The picture with a small figure walking across it: most of the frame holds still."""
    out = base.copy()
    out[100:160, 20 + 4 * k:80 + 4 * k] = _picture(7, float(base.mean()))[:60, :60]
    return out


def _clip() -> list[np.ndarray]:
    day, other = _picture(1, 170.0), _picture(2, 110.0)
    seq = [_night(5)] * 30                                               # a held night scene
    seq += [_moving(day, k) for k in range(40)]                         # a bright scene with movement in it
    seq += [_picture(9, 240.0, 0.02)] * 2                               # a two-frame flash
    seq += [_moving(other, k) for k in range(40)]                       # another, less bright
    seq += [_night(6)] * 30                                              # night again
    return seq


def test_a_video_is_split_at_its_cuts_and_short_stretches_join_the_scene_before():
    got = scenes.find(iter(_clip()))
    assert got.error is None and got.frames == 142
    assert [s[:2] for s in got.scenes] == [[0, 29], [30, 71], [72, 111], [112, 141]]      # the flash joined the day
    assert got.scenes[0][0] == 0 and all(a[1] + 1 == b[0] for a, b in zip(got.scenes, got.scenes[1:]))
    assert got.to_dict()["count"] == 4 and "scenes" not in got.brief()
    assert scenes.find(iter([])).error


def test_scenes_are_sorted_into_kinds_by_light_motion_and_detail():
    got = scenes.find(iter(_clip()))
    kinds = {k["id"]: k for k in got.kinds}
    night, first_day, second_day = got.scenes[0][2], got.scenes[1][2], got.scenes[2][2]
    assert got.scenes[3][2] == night and night != first_day
    assert kinds[night]["name"] == "dark, still, flat" and kinds[night]["scenes"] == 2 and kinds[night]["frames"] == 60
    assert kinds[first_day]["name"] == "bright, moving, detailed"
    assert kinds[second_day]["name"] == "mid, moving, detailed"
    assert abs(sum(k["share"] for k in got.kinds) - 1) < 0.01 and got.kinds[0]["frames"] == max(k["frames"] for k in got.kinds)
    for k in got.kinds:                                                  # every kind names frames to look at, inside it
        assert k["example_frames"] and all(any(a <= n <= z and kid == k["id"] for a, z, kid in got.scenes)
                                           for n in k["example_frames"])
    assert scenes.spans_of(got.scenes, night) == [[0, 29], [112, 141]]
    assert scenes.spans_of([[0, 9, "k1"], [10, 19, "k1"], [20, 29, "k2"]], "k1") == [[0, 19]]


def test_a_fast_pan_is_one_scene():
    wide = _picture(3)
    pan = [np.roll(wide, 40 * k, axis=1) for k in range(30)]             # a quarter of the frame every three frames
    got = scenes.find(iter(pan))
    assert len(got.scenes) == 1 and got.kinds[0]["name"].split(", ")[1] == "fast"


def test_a_cut_blended_over_one_frame_is_found_unless_the_frame_rate_is_low():
    a, b = _picture(1, 170.0), _picture(2, 90.0)
    mix = ((a.astype(np.float32) + b) / 2).astype(np.uint8)
    seq = [_moving(a, k) for k in range(20)] + [mix] + [_moving(b, k) for k in range(20)]
    got = scenes.find(iter(seq), fps=25.0)
    assert len(got.scenes) == 2 and got.scenes[0][1] in (19, 20)
    half = ((a.astype(np.float32) * 0.75 + b.astype(np.float32) * 0.25)).astype(np.uint8)      # two soft steps
    more = ((a.astype(np.float32) * 0.25 + b.astype(np.float32) * 0.75)).astype(np.uint8)
    soft = [a] * 12 + [half, more] + [b] * 12
    # the frames either side of a blended frame differ where consecutive frames do not
    near = [i for i, c in enumerate(scenes.measure(iter(soft), two_apart=True)["cut"]) if c]
    plain = [i for i, c in enumerate(scenes.measure(iter(soft), two_apart=False)["cut"]) if c]
    assert set(plain) <= set(near) and len(near) > len(plain)
    assert len(scenes.find(iter(soft), fps=25.0).scenes) == 2
    # at 10 fps frames two apart are a fifth of a second apart: only consecutive frames are compared
    assert [s[:2] for s in scenes.find(iter(soft), fps=10.0).scenes] == [s for s in scenes.split(scenes.measure(iter(soft), two_apart=False)["cut"])]


def test_small_kinds_join_the_nearest_larger_one_and_the_list_stays_short():
    def stat(first, length, key):
        return {"first": first, "last": first + length - 1, "light": 100.0, "motion": 1.0, "detail": 80.0, "dips": 0, "key": key}

    stats, at = [], 0
    for key, length in [((1, 1, 1), 600), ((0, 0, 0), 300), ((1, 2, 1), 10), ((0, 0, 1), 5)]:
        stats.append(stat(at, length, key))
        at += length
    kinds, ids = scenes._kinds(stats, at)
    assert [k["name"] for k in kinds] == ["mid, moving, detailed", "dark, still, flat"]
    assert ids[(1, 2, 1)] == "k1" and ids[(0, 0, 1)] == "k2"            # each went to the kind one level away
    assert kinds[0]["scenes"] == 2 and kinds[0]["frames"] == 610 and sum(k["frames"] for k in kinds) == at
    many = [stat(100 * i, 100, (i % 3, (i // 3) % 3, i // 9)) for i in range(18)]      # eighteen kinds of equal size
    kinds, ids = scenes._kinds(many, 1800)
    assert len(kinds) == scenes.MAX_KINDS and set(ids.values()) == {k["id"] for k in kinds}
    assert sum(k["frames"] for k in kinds) == 1800
    one, _ = scenes._kinds([stat(0, 50, (1, 1, 1))], 50)                # a single scene is still a kind
    assert len(one) == 1 and one[0]["share"] == 1.0


def test_a_reader_that_loses_frames_gives_no_map(monkeypatch):
    monkeypatch.setattr(scenes.media, "probe", lambda p, exact=False: {"frames": 150})
    monkeypatch.setattr(scenes.ghosts, "read_gray_exact", lambda p: iter(_clip()))
    got = scenes.scan(Path("x.mkv"))
    assert got.scenes == [] and "142 frames of 150" in got.error
    monkeypatch.setattr(scenes.media, "probe", lambda p, exact=False: {"frames": 142})
    assert len(scenes.scan(Path("x.mkv")).scenes) == 4


def test_the_sheet_shows_every_kind_with_a_frame_and_its_lines(monkeypatch, tmp_path):
    seen = []

    def frame(src, t, count=1, index=None):
        seen.append(round(t * 25))
        return [cv2.cvtColor(_picture(int(t * 25)), cv2.COLOR_GRAY2BGR)]

    monkeypatch.setattr(closeups.media, "probe", lambda p: {"fps": 25.0})
    monkeypatch.setattr(closeups, "frames_at", frame)
    kinds = [{"id": "k1", "name": "mid, moving, detailed", "scenes": 12, "share": 0.7, "light": 120.0, "motion": 2.5,
              "detail": 90.0, "dips": 0.02, "example_frames": [100, 900]},
             {"id": "k2", "name": "dark, still, flat", "scenes": 3, "share": 0.3, "light": 30.0, "motion": 0.1,
              "detail": 5.0, "dips": 0.0, "example_frames": [400]}]
    out = closeups.kinds_sheet(Path("x.mkv"), tmp_path / "kinds.jpg", kinds, title="v1 x.mkv")
    assert out and out.exists() and seen == [100, 900, 400]
    img = cv2.imread(str(out))
    assert img.shape[1] == 1440 and img.shape[0] > 2 * 270
    monkeypatch.setattr(closeups, "frames_at", lambda *a, **k: [])
    assert closeups.kinds_sheet(Path("x.mkv"), tmp_path / "none.jpg", kinds) is None


def test_the_survey_keeps_the_map_in_its_own_file_and_goes_on_without_one(monkeypatch, tmp_path):
    found = scenes.find(iter(_clip()))
    monkeypatch.setattr(scenes, "scan", lambda src, watch=None: found)
    monkeypatch.setattr(closeups, "kinds_sheet", lambda src, dst, kinds, title="": Path(dst))
    said = []
    got, damaged = auto._scene_map("v1", Path("C:/in/clip.mkv"), tmp_path, said.append)
    assert damaged is None and got["count"] == 4 and got["kinds"] == found.kinds and got["sheet"].endswith("v1_scene_kinds.jpg")
    stored = json.loads(Path(got["map"]).read_text(encoding="utf-8"))
    assert stored["scenes"] == found.scenes and stored["frames"] == 142
    assert "4 scenes of" in said[-1]
    monkeypatch.setattr(scenes, "scan", lambda src, watch=None: scenes.Scenes(frames=3, error="read 3 frames of 9"))
    assert auto._scene_map("v2", Path("C:/in/clip.mkv"), tmp_path, said.append) == (None, None) and "no scene map" in said[-1]

    def broken(src, watch=None):
        raise OSError("gone")

    monkeypatch.setattr(scenes, "scan", broken)
    assert auto._scene_map("v3", Path("C:/in/clip.mkv"), tmp_path, said.append) == (None, None) and "failed" in said[-1]
