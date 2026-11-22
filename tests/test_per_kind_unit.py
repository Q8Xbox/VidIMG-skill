"""Settings by kind of scene: the recipe's `per_kind` and a step's `kinds`, and how the pipeline runs them."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vidimg import auto, library, recipe
from vidimg import pipeline_video as pv
from vidimg import texture
from vidimg.adapters import avisynth as avs
from vidimg.guard import GuardReport
from vidimg.guard import video as gv
from vidimg.probe import scenes

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="ffmpeg is not installed")

BASE = {"faces": "drawn", "steps": [{"block": "deblock", "preset": "light"}, {"block": "dehalo", "kinds": ["k3", "k1"]}],
        "models": ["esrgan-v3"], "scale": 2, "texture": 1.0,
        "per_kind": {"k2": {"texture": 0.4, "steps": {"deblock": "strong"}},
                     "k4": {"texture": 0, "steps": {"deblock": False}}}}
MAP = scenes.Scenes(frames=100, scenes=[[0, 29, "k1"], [30, 49, "k2"], [50, 59, "k4"], [60, 79, "k2"], [80, 99, "k3"]],
                    kinds=[{"id": f"k{i}", "name": "mid, moving, flat", "example_frames": [f]}
                           for i, f in ((1, 15), (2, 40), (3, 90), (4, 55))])


# ----------------------------------------------------------------------------- the recipe
def test_a_recipe_says_what_a_kind_of_scene_gets_differently():
    got, advice = recipe.validate(BASE)
    assert got["steps"][1]["kinds"] == ["k1", "k3"] and "kinds" not in got["steps"][0]
    assert got["per_kind"]["k2"] == {"texture": 0.4, "steps": {"deblock": {"preset": "strong", "settings": {"quant1": 30, "quant2": 34}}}}
    assert got["per_kind"]["k4"] == {"texture": 0.0, "steps": {"deblock": False}}
    assert any("k1, k2, k3, k4" in a and "this video" in a for a in advice)
    assert recipe.validate(got)[0] == got                               # the stored form reads back as itself
    assert recipe.uses_kinds(got) and recipe.kinds_named(got) == ["k1", "k2", "k3", "k4"]
    plain = recipe.validate({"faces": "drawn", "steps": ["deblock"]})[0]
    assert plain["per_kind"] == {} and not recipe.uses_kinds(plain)
    assert recipe.from_decision({"content": "anime"}, [])["per_kind"] == {}
    said = recipe.describe(got)
    assert "dehalo (medium)" not in said and "dehalo [only k1, k3]" in said
    assert "in k2: texture 0.4, deblock strong" in said and "in k4: texture 0, deblock off" in said


def test_each_kind_gets_the_base_step_its_own_settings_or_nothing():
    got = recipe.validate(BASE)[0]
    deblock, dehalo = got["steps"]
    assert recipe.step_settings(got, deblock, "k1") == {"quant1": 18, "quant2": 20}
    assert recipe.step_settings(got, deblock, "k2") == {"quant1": 30, "quant2": 34}
    assert recipe.step_settings(got, deblock, "k4") is None
    assert recipe.step_settings(got, deblock, "k6") == {"quant1": 18, "quant2": 20}     # a kind the recipe does not name
    assert recipe.step_settings(got, dehalo, "k3") is not None and recipe.step_settings(got, dehalo, "k2") is None
    assert [recipe.texture_for(got, k, 1.0) for k in ("k1", "k2", "k4")] == [1.0, 0.4, 0.0]
    own = recipe.validate({**BASE, "per_kind": {"k2": {"steps": {"deblock": {"settings": {"quant1": 12}}}}}})[0]
    assert own["per_kind"]["k2"]["steps"]["deblock"]["settings"] == {"quant1": 12, "quant2": 26}


@pytest.mark.parametrize("change, said", [
    ({"per_kind": {"night": {"texture": 0.5}}}, "not a kind id"),
    ({"per_kind": {"k2": {"model": "cugan"}}}, "texture and/or steps"),
    ({"per_kind": {"k2": {"texture": 1.5}}}, "between 0 and 1"),
    ({"per_kind": {"k2": {"steps": {"degrain": "light"}}}}, "not one of the recipe's steps"),
    ({"per_kind": {"k2": {"steps": {"dehalo": "strong"}}}}, "limited to k1, k3"),
    ({"per_kind": {"k2": {"steps": {"deblock": "extreme"}}}}, "no preset"),
    ({"per_kind": {"k2": {"steps": {"deblock": True}}}}, "preset name"),
    ({"per_kind": ["k2"]}, "per_kind is an object"),
    ({"steps": [{"block": "deblock", "kinds": "k2"}]}, "list of kind ids"),
    ({"steps": [{"block": "deblock", "kinds": []}]}, "list of kind ids"),
    ({"steps": [{"block": "deflicker", "kinds": ["k2"]}], "per_kind": {}}, "cannot be limited to kinds"),
    ({"steps": [{"block": "crop_borders", "kinds": ["k2"]}], "per_kind": {}}, "cannot be limited to kinds"),
    ({"steps": ["stabilise"], "per_kind": {"k2": {"steps": {"stabilise": False}}}}, "cannot differ by kind"),
])
def test_what_cannot_differ_by_kind_is_refused(change, said):
    with pytest.raises(ValueError, match=said):
        recipe.validate({**BASE, **change})


def test_a_step_that_changes_the_frame_count_first_is_said():
    got, advice = recipe.validate({"faces": "drawn", "steps": ["remove_repeats", {"block": "deblock", "kinds": ["k2"]}]})
    assert any("remove_repeats changes the frame count before deblock" in a and "one frame off" in a for a in advice)
    assert not any("one frame off" in a for a in recipe.validate(BASE)[1])


# ----------------------------------------------------------------------------- the map on other frame counts
def test_the_map_follows_a_changed_frame_count_and_joins_equal_neighbours():
    assert scenes.at_count(MAP.scenes, 100, 100) == MAP.scenes and scenes.at_count(MAP.scenes, 100, 100) is not MAP.scenes
    fewer = scenes.at_count(MAP.scenes, 100, 80)                        # every fifth frame was a copy
    assert fewer == [[0, 23, "k1"], [24, 39, "k2"], [40, 47, "k4"], [48, 63, "k2"], [64, 79, "k3"]]
    assert all(a[1] + 1 == b[0] for a, b in zip(fewer, fewer[1:]))
    tiny = scenes.at_count([[0, 0, "k1"], [1, 99, "k2"]], 100, 10)      # a scene too short to keep a frame
    assert tiny[0][0] == 0 and tiny[-1][1] == 9
    assert scenes.ranges_by(MAP.scenes, {"k1": 0, "k2": 1, "k4": 1, "k3": 0}) == [[0, 29, 0], [30, 79, 1], [80, 99, 0]]
    assert scenes.ranges_by(MAP.scenes, {"k2": 0.4}) == [[0, 29, None], [30, 49, 0.4], [50, 59, None], [60, 79, 0.4], [80, 99, None]]
    assert scenes.kind_at(MAP.scenes, 55) == "k4" and scenes.kind_at(MAP.scenes, 400) is None


def test_the_script_computes_every_variant_on_the_whole_clip_and_takes_each_stretch_from_its_own():
    lines = avs.ByRanges([["Deblock_QED(quant1=18, quant2=20)"], []], [[0, 29, 0], [30, 59, 1], [60, 99, 0]]).lines()
    assert lines == ["__in = last", "last = __in", "Deblock_QED(quant1=18, quant2=20)", "__v0 = last", "__v1 = __in",
                     "__out = __v0.Trim(0, end=29)", "__out = __out ++ __v1.Trim(30, end=59)",
                     "__out = __out ++ __v0.Trim(60, end=99)", "__out"]
    assert avs.ByRanges().rate() == 1 and avs.ByRanges in avs._ORDER


# ----------------------------------------------------------------------------- the pipeline
def _ctx(tmp_path, rcp, found=MAP):
    rec = SimpleNamespace(notes=[], prepass={})
    return {"base": tmp_path / "base.nut", "work": tmp_path, "rec": rec, "log": lambda m: None, "drawn": True,
            "src": tmp_path / "in.mkv", "recipe": rcp, **({"scene_map": found} if found is not False else {})}


def test_kinds_with_the_same_settings_share_a_variant(monkeypatch, tmp_path):
    blocks = library.load()[0]
    got = recipe.validate(BASE)[0]
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"frames": 100})
    plan = pv.by_kind_plan(_ctx(tmp_path, got), got["steps"], blocks)
    assert [[b.name for b, _ in var] for var in plan["variants"]] == [["deblock", "dehalo"], ["deblock"], []]
    assert plan["of_kind"] == {"k1": 0, "k2": 1, "k3": 0, "k4": 2}
    assert plan["variants"][1][0][1] == {"quant1": 30, "quant2": 34}
    assert plan["ranges"] == [[0, 29, 0], [30, 49, 1], [50, 59, 2], [60, 79, 1], [80, 99, 0]]
    assert plan["also"] == [14, 39, 54] and plan["untouched"] == [54]
    assert "k1, k3: deblock (quant1 18, quant2 20), dehalo" in plan["said"] and "k4: left as it is" in plan["said"]
    # after a step that dropped a fifth of the frames the stretches are found by their share of the time
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"frames": 80})
    assert pv.by_kind_plan(_ctx(tmp_path, got), got["steps"], blocks)["ranges"][-1] == [64, 79, 0]
    plain = recipe.validate({"faces": "drawn", "steps": ["deblock"]})[0]
    assert pv.by_kind_plan(_ctx(tmp_path, plain), plain["steps"], blocks) is None
    # only the first step of the recipe is in this group, and no kind changes it
    assert pv.by_kind_plan(_ctx(tmp_path, {**got, "per_kind": {}}), got["steps"][:1], blocks) is None
    assert pv.by_kind_plan(_ctx(tmp_path, got, found=None), got["steps"], blocks) is False


def test_the_scene_map_is_made_once_and_its_absence_is_said(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(scenes, "scan", lambda src: calls.append(src) or MAP)
    c = _ctx(tmp_path, recipe.validate({**BASE, "per_kind": {"k9": {"texture": 0.2}}})[0], found=False)
    assert pv.scene_map_for(c) is MAP and pv.scene_map_for(c) is MAP and len(calls) == 1
    assert "names k9, which this video does not have" in c["rec"].notes[-1]
    monkeypatch.setattr(scenes, "scan", lambda src: scenes.Scenes(error="read 3 frames of 9"))
    c = _ctx(tmp_path, recipe.validate(BASE)[0], found=False)
    assert pv.scene_map_for(c) is None and "no scene map (read 3 frames of 9)" in c["rec"].notes[-1]
    assert pv.scene_map_for(c) is None and len(c["rec"].notes) == 1


def test_a_pass_by_kind_renders_the_variants_and_is_checked_on_every_kind(monkeypatch, tmp_path):
    blocks = library.load()[0]
    got = recipe.validate(BASE)[0]
    base = tmp_path / "base.nut"
    base.write_bytes(b"x" * 10)
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"frames": 100})
    seen = {}

    def render(src, dst, stages, work=None):
        seen["stages"] = stages
        Path(dst).write_bytes(b"out")
        return SimpleNamespace(ok=True, error=None, details={"script": "s", "frames": 100})

    def check(src, dst, **kw):
        seen["check"] = kw
        return GuardReport(True, [])

    monkeypatch.setattr(pv.avs, "available", lambda: (True, ""))
    monkeypatch.setattr(pv.avs, "render", render)
    monkeypatch.setattr(pv.avs, "_rate", lambda p: (1, 1, {}))
    monkeypatch.setattr(pv, "verify_cleanup", check)
    for name in ("colour_hint", "write_colour_sidecar"):
        monkeypatch.setattr(pv.media, name, lambda *a: {})
    for name in ("fix_av_offset", "fix_aspect"):
        monkeypatch.setattr(pv.media, name, lambda a, b: None)
    c = _ctx(tmp_path, got)
    assert pv.run_steps(c, got["steps"], blocks)
    stage = seen["stages"][0]
    assert isinstance(stage, avs.ByRanges) and len(seen["stages"]) == 1 and len(stage.variants) == 3
    assert stage.variants[1] == ["Deblock_QED(quant1=30, quant2=34)"] and stage.variants[2] == []
    assert stage.ranges[2] == [50, 59, 2]
    # the block level is not demanded: one kind gets no deblocking; a frame of each variant is judged, and the
    # kind left out must be the input's
    assert seen["check"]["deblocked"] is False and seen["check"]["also"] == [14, 39, 54] and seen["check"]["untouched"] == [54]
    assert c["rec"].notes[-1].startswith("clean-up before the model: by kind of scene: k1, k3: deblock")
    assert c["base"].name == "pass.nut"
    # every kind this video has gets the same: an ordinary pass
    same = recipe.validate({"faces": "drawn", "steps": ["deblock"], "per_kind": {"k7": {"steps": {"deblock": "strong"}}}})[0]
    base.write_bytes(b"y" * 12)
    c = _ctx(tmp_path, same)
    assert pv.run_steps(c, same["steps"], blocks)
    assert isinstance(seen["stages"][0], avs.Lines) and seen["check"]["deblocked"] is True and seen["check"]["also"] is None
    # a step asked for only in kinds the video does not have is not run at all
    none = recipe.validate({"faces": "drawn", "steps": [{"block": "dehalo", "kinds": ["k8"]}]})[0]
    seen.clear()
    c = _ctx(tmp_path, none)
    assert pv.run_steps(c, none["steps"], blocks) and not seen and "does not have" in c["rec"].notes[-1]
    # without a scene map nothing is set by kind, and the step is not spread over the whole video either
    c = _ctx(tmp_path, got, found=None)
    assert pv.run_steps(c, got["steps"], blocks) and not seen and "no scene map" in c["rec"].notes[-1]


def test_the_pass_check_takes_the_extra_frames_and_holds_left_out_stretches_to_the_input(monkeypatch):
    seen = {}

    def whole(src, dst, **kw):
        seen.update(kw)
        return GuardReport(True, [gv.Gate("frames", "pass", 1, 1, "")])

    monkeypatch.setattr(gv, "verify_video", whole)
    monkeypatch.setattr(gv, "untouched_gate", lambda a, b, frames: gv.Gate("untouched", "fail", 31.0, 50.0, str(frames)))
    rep = gv.verify_cleanup(Path("a"), Path("b"), also=[5, 9], untouched=[7])
    assert seen["also"] == [5, 9] and not rep.passed and rep.failed() == ["untouched"]
    assert any("leaves out were changed" in s for s in rep.suggestions)
    assert gv.verify_cleanup(Path("a"), Path("b")).passed


@needs_ffmpeg
def test_texture_strength_follows_the_stretches(tmp_path):
    rng = np.random.default_rng(4)
    w, h, n = 64, 48, 6
    flat = np.full((h, w), 120, np.uint8)
    grainy = np.clip(120 + rng.normal(0, 6, (n, h, w)), 0, 255).astype(np.uint8)

    def write(path, frames):
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                        "-s", f"{w}x{h}", "-r", "10", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p10le", str(path)],
                       input=b"".join(f.tobytes() for f in frames), check=True, capture_output=True)
        return path

    def read(path):
        raw = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path), "-pix_fmt", "gray",
                              "-f", "rawvideo", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
        return [np.frombuffer(raw[k * w * h:(k + 1) * w * h], np.uint8).reshape(h, w).astype(np.float32) for k in range(n)]

    base, model = write(tmp_path / "base.mkv", list(grainy)), write(tmp_path / "model.mkv", [flat] * n)
    r = texture.apply(base, model, tmp_path / "out.mkv", 1.0, "ffv1", strengths=[[0, 1, 0.0], [4, 5, 0.5]])
    assert r.ok and r.details["stretches"] == 2
    given = [float(np.abs(o - m).mean()) for o, m in zip(read(tmp_path / "out.mkv"), read(model))]
    assert given[0] < 0.1 and given[1] < 0.1                            # no texture back: the model's own frame
    assert given[2] > 0.8 and given[3] > 0.8                            # full strength
    assert 0.3 * given[2] < given[4] < 0.8 * given[2]                   # half strength


# ----------------------------------------------------------------------------- deciding
def test_a_decision_may_only_name_kinds_the_video_has():
    got = recipe.validate(BASE)[0]
    item = {"type": "video", "scenes": {"kinds": [{"id": "k1"}, {"id": "k2"}, {"id": "k3"}, {"id": "k4"}]}}
    auto._check_kinds("v1", item, got)
    auto._check_kinds("v1", {"type": "video"}, recipe.validate({"faces": "drawn"})[0])
    with pytest.raises(ValueError, match="no kind k4 in this video .its kinds: k1, k2, k3"):
        auto._check_kinds("v1", {"type": "video", "scenes": {"kinds": [{"id": "k1"}, {"id": "k2"}, {"id": "k3"}]}}, got)
    with pytest.raises(ValueError, match="no scene map"):
        auto._check_kinds("v1", {"type": "video", "scenes": None}, got)
    with pytest.raises(ValueError, match="each episode has its own kinds"):
        auto._check_kinds("s1", {**item, "type": "series"}, got)


def test_a_trial_knows_which_kind_each_preview_shows(monkeypatch, tmp_path):
    from vidimg import statefile

    where = tmp_path / "v1_scenes.json"
    statefile.write_json(where, MAP.to_dict())
    monkeypatch.setattr(pv, "_preview_starts", lambda prof: [0.2, 3.4, 8.0])
    monkeypatch.setattr("vidimg.probe.video.VideoProfile", lambda **kw: SimpleNamespace(fps=kw["fps"]))
    rec = SimpleNamespace(profile={"fps": 10.0})
    assert auto._preview_kinds({"scenes": {"map": str(where)}}, rec) == ["k1", "k4", "k3"]
    assert auto._preview_kinds({"scenes": None}, rec) == []
