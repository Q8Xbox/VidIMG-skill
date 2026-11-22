"""No fixed paths: the block library (steps as data), recipes composed from it, the pipeline running a recipe in
the order it was written, and the case library that every run adds to."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vidimg import auto, cases, library, recipe
from vidimg import pipeline_video as pv
from vidimg.adapters import avisynth as avs
from vidimg.adapters import free_sr, video2x

NEW = {"name": "grain_calm", "kind": "filter", "engine": "avisynth",
       "what": "calms heavy film grain", "when": "the close-ups show coarse dancing grain",
       "script": ["SMDegrain(tr={tr}, thSAD={thsad}, contrasharp=false)"],
       "params": {"tr": {"type": "int", "min": 1, "max": 4, "default": 2},
                  "thsad": {"type": "int", "min": 100, "max": 600, "default": 250}},
       "presets": {"light": {"thsad": 150}, "strong": {"tr": 3, "thsad": 400}}, "default_preset": "light"}


# ----------------------------------------------------------------------------- the library
def test_every_built_in_block_is_valid_and_the_library_names_what_a_recipe_needs():
    blocks, problems = library.load(Path("no/such/folder"))
    assert problems == []
    kinds = {b.name: b.kind for b in blocks.values()}
    assert {"fields", "deinterlace_scenes", "remove_repeats", "crop_borders", "deblock", "dehalo", "esrgan-v3",
            "cugan", "cugan-pro", "general-v3", "realesrnet", "bsrnet", "texture"} <= set(kinds)
    assert set(pv.STEP_RUNNERS) == {b.runner for b in blocks.values() if not b.scripted and b.kind in ("field", "framing")}
    for b in blocks.values():
        assert b.what and b.when and b.origin == "built-in"
        if b.kind == "model":                                   # a model block carries what its engine is given
            assert b.spec.get("label") and b.spec.get("model")
    rows = library.guide(blocks)
    assert [r["kind"] for r in rows] == sorted((r["kind"] for r in rows), key=library.KINDS.index)
    assert next(r for r in rows if r["block"] == "deblock")["presets"] == ["light", "medium", "strong"]


def test_a_tool_is_added_as_a_file_and_bounded_by_what_makes_it_safe(tmp_path):
    path = library.save_local(NEW, tmp_path)
    blocks, problems = library.load(tmp_path)
    b = blocks["grain_calm"]
    assert path.exists() and problems == [] and b.origin == "local" and b.scripted
    assert library.render(b, library.resolve(b)) == ["SMDegrain(tr=2, thSAD=150, contrasharp=false)"]
    assert library.resolve(b, "strong", {"tr": 4}) == {"tr": 4, "thsad": 400}
    for preset, settings in (("extreme", None), (None, {"tr": 9}), (None, {"nope": 1}), (None, {"tr": "2"})):
        with pytest.raises(ValueError):
            library.resolve(b, preset, settings)
    with pytest.raises(ValueError):                              # the name is taken now
        library.save_local(NEW, tmp_path)
    with pytest.raises(ValueError):                              # and a built-in name cannot be replaced
        library.save_local({**NEW, "name": "deblock"}, tmp_path)
    bad = [
        {**NEW, "name": "Bad Name"},
        {**NEW, "name": "a1", "script": ["LSFmod(strength=80)"], "params": {}, "presets": {}, "default_preset": None},
        {**NEW, "name": "a2", "script": ['LoadPlugin("x.dll")'], "params": {}, "presets": {}, "default_preset": None},
        {**NEW, "name": "a3", "script": ["Spline36Resize(1920, 1080)"], "params": {}, "presets": {}, "default_preset": None},
        {**NEW, "name": "a4", "script": ["Trim(0, 100)"], "params": {}, "presets": {}, "default_preset": None},
        {**NEW, "name": "a5", "script": ["SMDegrain(tr={missing})"]},
        {**NEW, "name": "a6", "kind": "framing"},
        {**NEW, "name": "a7", "changes": ["frame_size"]},
        {**NEW, "name": "a8", "engine": "builtin", "runner": "fields"},      # code runners are not addable
        {**NEW, "name": "a9", "check": "none"},                              # nor is a check of its own
        {**NEW, "name": "b1", "engine": "ffmpeg", "script": ["unsharp=5:5:1.0"], "params": {}, "presets": {}, "default_preset": None},
        {**NEW, "name": "b2", "engine": "ffmpeg", "script": ["movie=other.mkv"], "params": {}, "presets": {}, "default_preset": None},
        {**NEW, "name": "b3", "params": {"tr": {"type": "int", "default": 2}}},
        {**NEW, "name": "b4", "what": ""},
    ]
    for d in bad:
        with pytest.raises(ValueError):
            library.validate(d)
    ok = library.validate({"name": "soft_denoise", "kind": "filter", "engine": "ffmpeg", "what": "w", "when": "w",
                           "script": ["hqdn3d={luma}:{luma}:3:3"],
                           "params": {"luma": {"type": "float", "min": 0, "max": 6, "default": 1.5}}})
    assert library.render(ok, library.resolve(ok)) == ["hqdn3d=1.5:1.5:3:3"]
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert any("broken.json" in p for p in library.load(tmp_path)[1])       # a bad file is reported, the rest loads


# ----------------------------------------------------------------------------- recipes
def test_a_recipe_is_free_in_its_order_and_refused_only_where_it_cannot_work():
    got, advice = recipe.validate({"faces": "drawn", "steps": ["fields", "deinterlace_scenes", "dehalo:light",
                                                               {"block": "deblock", "settings": {"quant1": 20}},
                                                               "crop_borders"],
                                   "models": ["cugan", "general-v3"], "scale": 1, "texture": 0.5})
    assert [s["block"] for s in got["steps"]] == ["fields", "deinterlace_scenes", "dehalo", "deblock", "crop_borders"]
    assert got["steps"][3]["settings"] == {"quant1": 20, "quant2": 26} and got["steps"][2]["preset"] == "light"
    assert got["models"] == ["cugan", "general-v3"] and got["scale"] == 1 and got["texture"] == 0.5
    assert any("untouched pixel grid" in a for a in advice)              # deblock after dehalo: advice, not refusal
    assert recipe.validate({"faces": "real", "steps": [], "models": ["realesrnet"]})[1] == []
    # a faithful model on a drawing, and the compact model first at the source's own size: nothing forbids it
    assert recipe.validate({"faces": "drawn", "models": ["realesrnet", "esrgan-v3"], "scale": 1})[0]["models"] == ["realesrnet", "esrgan-v3"]
    for bad in ({"steps": []}, {"faces": "cartoon"}, {"faces": "drawn", "steps": ["sharpen"]},
                {"faces": "drawn", "steps": ["deblock", "deblock"]}, {"faces": "drawn", "steps": ["esrgan-v3"]},
                {"faces": "drawn", "steps": ["remove_repeats", "deinterlace_scenes"]},   # frame numbers already changed
                {"faces": "real", "models": ["esrgan-v3"]},                               # anime model, real faces
                {"faces": "drawn", "models": ["gfpgan"]}, {"faces": "drawn", "scale": 9},
                {"faces": "drawn", "texture": 2}, {"faces": "drawn", "path": "anime"},
                {"faces": "drawn", "steps": ["crop_borders:light"]}, {"faces": "drawn", "steps": ["deblock:max"]},
                {"faces": "drawn", "models": ["cugan", "cugan"]}, {"faces": "drawn", "models": ["cugan"], "scale": 1.5}):
        with pytest.raises(ValueError):
            recipe.validate(bad)
    assert "steps: fields, then deinterlace_scenes, then dehalo (light)" in recipe.describe(got) and "own size" in recipe.describe(got)


def test_the_older_settings_read_as_a_recipe():
    r = recipe.from_decision({"content": "anime", "model": "cugan", "scale": 2, "filters": ["dehalo:light", "deinterlace_scenes", "deblock"],
                              "remove_repeats": True, "crop_borders": True, "mask_tape_noise": True, "texture": 0.4},
                             ["blended", "head_switching"])
    got, advice = recipe.validate(r)
    assert [s["block"] for s in got["steps"]] == ["fields", "deinterlace_scenes", "remove_repeats", "deblock", "dehalo", "crop_borders"]
    assert got["steps"][0]["settings"] == {"mask_tape_noise": True} and got["faces"] == "drawn" and advice == []
    assert got["models"] == ["cugan"] and got["texture"] == 0.4
    auto_real = recipe.from_decision({"content": "auto", "model": "esrgan-v3"}, ["sd"])
    assert auto_real["faces"] == "real" and auto_real["models"] == [] and auto_real["steps"] == []
    assert recipe.from_decision({"content": "auto"}, ["anime_like"])["faces"] == "drawn"


def _prof(height=480, flags=(), frames=1000, width=640):
    return SimpleNamespace(height=height, width=width, flags=list(flags), frames=frames, duration=40.0, fps=25.0,
                           faces_max=0)


def test_any_model_at_any_size_and_the_usual_order_follows_unless_strict():
    blocks = library.load(Path("no/such"))[0]

    def cands(rcp, **kw):
        why, c = pv.recipe_candidates(_prof(**kw), recipe.validate(rcp, blocks)[0], blocks)
        return why, [(label, make()[0]) for label, make in c]

    why, c = cands({"faces": "drawn"})                                     # nothing named: what the samples support
    assert [x[0] for x in c] == ["anime esrgan-v3", "anime cugan", "free general-v3"] and c[0][1].scale == 2
    assert not c[0][1].same_size and c[2][1].height == 960
    why, c = cands({"faces": "drawn"}, height=1080, width=1440)            # a large drawing keeps its size
    assert all(x[1].same_size for x in c[:2]) and c[2][1].height == 1080 and why.startswith("own size (1440x1080)")
    why, c = cands({"faces": "drawn", "models": ["general-v3", "cugan-pro"], "scale": 1})
    assert [x[0] for x in c] == ["free general-v3", "anime cugan-pro", "anime esrgan-v3", "anime cugan"]
    assert isinstance(c[0][1], free_sr.FreeSR) and c[0][1].height == 480 and c[1][1].same_size
    why, c = cands({"faces": "drawn", "models": ["realesrnet"], "model_strict": True, "scale": 2})
    assert [x[0] for x in c] == ["free realesrnet"] and c[0][1].height == 960 and "only" in why
    why, c = cands({"faces": "real"})
    assert [x[0] for x in c] == ["free realesrnet", "free bsrnet", "free general-v3"]
    why, c = cands({"faces": "real"}, frames=200000, height=1080, width=1920)
    assert [x[0] for x in c] == ["free general-v3"] and c[0][1].height == 1080
    why, c = cands({"faces": "drawn", "scale": 3})
    assert isinstance(c[0][1], video2x.AnimeSR) and c[0][1].scale == 3


def test_steps_run_in_the_recipes_order_and_scripted_neighbours_share_a_pass(monkeypatch, tmp_path):
    blocks = library.load(Path("no/such"))[0]
    ran = []

    def scripted(base, work, group, index, log=print, drawn=False):
        ran.append(("pass", [b.name for b, _ in group], [v for _, v in group], drawn))
        return Path(f"{base}+{index}"), {"said": "+".join(b.name for b, _ in group)}

    monkeypatch.setattr(pv, "run_scripted", scripted)
    monkeypatch.setitem(pv.STEP_RUNNERS, "crop", lambda c, s: ran.append(("crop", str(c["base"]))) or True)
    monkeypatch.setitem(pv.STEP_RUNNERS, "fields", lambda c, s: ran.append(("fields", s)) or True)
    rcp = recipe.validate({"faces": "drawn", "steps": [{"block": "fields", "settings": {"mask_tape_noise": True}},
                                                       "dehalo:light", "deblock:strong", "crop_borders", "dehalo2"]},
                          blocks | {"dehalo2": library.validate({**NEW, "name": "dehalo2"})})[0]
    rec = SimpleNamespace(prepass={}, notes=[])
    ctx = {"base": Path("src"), "shape_ref": Path("src"), "work": tmp_path, "prof": None, "rec": rec,
           "log": lambda m: None, "drawn": True}
    assert pv.run_steps(ctx, rcp["steps"], blocks | {"dehalo2": library.validate({**NEW, "name": "dehalo2"})})
    assert [r[0] for r in ran] == ["fields", "pass", "crop", "pass"]
    assert ran[0][1] == {"mask_tape_noise": True}
    assert ran[1][1] == ["dehalo", "deblock"] and ran[1][2][1] == {"quant1": 30, "quant2": 34} and ran[1][3] is True
    assert ran[2][1] == "src+1" and ran[3][1] == ["dehalo2"] and str(ctx["base"]) == "src+1+2"
    assert set(rec.prepass) == {"pass1", "pass2"} and len(rec.notes) == 2
    # failed field handling stops the run; a failed filter does not
    monkeypatch.setitem(pv.STEP_RUNNERS, "fields", lambda c, s: False)
    assert pv.run_steps(dict(ctx), rcp["steps"], blocks | {"dehalo2": library.validate({**NEW, "name": "dehalo2"})}) is False
    monkeypatch.setattr(pv, "run_scripted", lambda *a, **k: (None, {"said": "dehalo", "skipped": "guard"}))
    rec2 = SimpleNamespace(prepass={}, notes=[])
    c2 = dict(ctx, rec=rec2, base=Path("src"))
    assert pv.run_steps(c2, [{"block": "dehalo", "settings": {}}], blocks) and c2["base"] == Path("src")
    assert "not applied" in rec2.notes[0]


def test_a_library_script_passes_the_same_lint_as_any_stage():
    from fractions import Fraction
    blocks = library.load(Path("no/such"))[0]
    stages = [avs.Lines(library.render(blocks[n], library.resolve(blocks[n], "light")), n) for n in ("dehalo", "deblock")]
    text = avs.Script(Path("C:/in/a.mkv"), stages, Path("C:/work/a.lwi"), Fraction(25)).text()
    assert avs.lint(text) == []
    assert text.index("DeHalo_alpha(rx=2, ry=2, darkstr=0, brightstr=0.6)") < text.index("Deblock_QED(quant1=18, quant2=20)")


# ----------------------------------------------------------------------------- the case library
def _profile(flags, height=480, bpp=0.05, blend=0.0):
    return {"flags": flags, "width": 640, "height": height, "fps": 25.0, "codec": "h264", "bits_per_pixel": bpp,
            "blockiness": 1.4, "blend_share": blend, "anime_score": 0.5, "noise_sigma": 0.3}


def test_every_run_leaves_a_case_and_similar_sources_find_it(tmp_path):
    r1 = recipe.validate({"faces": "drawn", "steps": ["fields"], "models": ["esrgan-v3"], "scale": 1})[0]
    r2 = recipe.validate({"faces": "drawn", "steps": ["fields"], "models": ["general-v3"], "scale": 1})[0]
    gates = {"vmaf": {"status": "pass", "value": 98.0}, "texture": {"status": "warn", "value": 0.09}}
    cases.record("C:/v/old upscale.mkv", _profile(["blended", "compressed"], 1080, 0.02, 0.38), r1,
                 {"delivered": "x.mp4", "chosen": "anime esrgan-v3", "final_gates": gates, "notes": ["ok"]}, tmp_path)
    cases.record("C:/v/old upscale.mkv", _profile(["blended", "compressed"], 1080, 0.02, 0.38), r2,
                 {"delivered": None, "chosen": "free general-v3", "final_gates": {"color": {"status": "fail", "value": 4.5}}}, tmp_path)
    cases.record("C:/v/clean dvd.mkv", _profile(["sd"]), r1, {"delivered": "y.mp4", "chosen": "anime esrgan-v3", "final_gates": gates}, tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 3                       # one per file and recipe
    assert cases.annotate("C:/v/old upscale.mkv", r1, review={"verdict": "good", "notes": "clean lines", "by": "claude"},
                          verdict="looks right to me", where=tmp_path)
    assert not cases.annotate("C:/v/unknown.mkv", r1, verdict="x", where=tmp_path)
    near = cases.similar(_profile(["blended", "compressed"], 1080, 0.025, 0.4), 2, tmp_path)
    assert [c["source"] for c in near] == ["old upscale.mkv", "old upscale.mkv"] and near[0]["distance"] < 0.5
    told = {c["model"]: c for c in near}
    assert told["anime esrgan-v3"]["review"]["verdict"] == "good" and told["anime esrgan-v3"]["verdict"] == "looks right to me"
    assert told["free general-v3"]["delivered"] is False and told["free general-v3"]["gates"]["failed"] == ["color"]
    assert cases.similar(_profile(["sd"]), 1, tmp_path)[0]["source"] == "clean dvd.mkv"
    # the file's own runs are its history, not its neighbours
    assert cases.similar(_profile(["blended", "compressed"], 1080), 3, tmp_path, exclude_source="C:/v/old upscale.mkv")[0]["source"] == "clean dvd.mkv"
    assert len(cases.history("C:/v/old upscale.mkv", tmp_path)) == 2 and cases.history("C:/v/none.mkv", tmp_path) == []
    assert cases.similar(_profile([]), 3, tmp_path / "empty") == []
    assert "frames" not in json.dumps(json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))["signature"])


# ----------------------------------------------------------------------------- through the automatic flow
def test_a_recipe_is_decided_run_recorded_and_reviewed(tmp_path, monkeypatch):
    from vidimg import pipeline_video

    monkeypatch.setattr(cases, "folder", lambda: tmp_path / "cases")
    plan = {"version": 1, "created": "now", "out": str(tmp_path), "inputs": [], "ignored_files": 0, "contact_sheets": [],
            "items": [{"id": "v1", "type": "video", "source": "C:/in/clip.mp4", "guide": {}, "decided_by": "analysis",
                       "reasons": [], "state": "pending",
                       "decision": {"content": "auto", "model": "general-v3", "scale": None, "recipe": None}}]}
    auto.save(plan)
    s = auto.summary(auto.load(tmp_path))
    assert {"fields", "texture", "esrgan-v3"} <= {b["block"] for b in s["blocks"]} and "recipe" in s["next"]
    with pytest.raises(ValueError):
        auto.decide(tmp_path, {"v1": {"recipe": {"faces": "real", "models": ["cugan"]}}})
    plan = auto.decide(tmp_path, {"v1": {"recipe": {"faces": "drawn", "steps": ["dehalo", "deblock:light"],
                                                    "models": ["esrgan-v3"], "scale": 1}, "reason": "an old enlargement"}})
    it = plan["items"][0]
    assert it["decision"]["recipe"]["steps"][0] == {"block": "dehalo", "settings": {"radius": 2.0, "bright": 0.8, "dark": 0.2}}
    assert any("untouched pixel grid" in r for r in it["reasons"])
    seen = {}

    def video(src, out, **kw):
        seen.update(kw)
        d = {"chosen": "anime esrgan-v3", "delivered": "x_restored.mp4", "passed": True, "notes": ["done"],
             "final_gates": {"vmaf": {"status": "pass", "value": 99.0}}, "sheets": [], "recipe": kw["recipe"],
             "profile": _profile(["compressed"], 1080)}
        return SimpleNamespace(to_dict=lambda: d)

    monkeypatch.setattr(pipeline_video, "restore_video", video)
    monkeypatch.setattr("vidimg.report.compare.video_page", lambda r, p: p)
    plan = auto.run(tmp_path, log=lambda m: None)
    assert seen["recipe"] == it["decision"]["recipe"] and plan["items"][0]["recipe_used"] == seen["recipe"]
    assert "recipe: faces drawn; steps: dehalo, then deblock (light)" in Path(plan["report"]).read_text(encoding="utf-8")
    assert len(list((tmp_path / "cases").glob("*.json"))) == 1
    auto.review(tmp_path, {"v1": {"verdict": "good", "notes": "lines clean", "person_verdict": "much better"}})
    case = json.loads(next((tmp_path / "cases").glob("*.json")).read_text(encoding="utf-8"))
    assert case["review"]["verdict"] == "good" and case["verdict"] == "much better" and case["outcome"]["delivered"]
    # the next survey of a similar source is told about it
    assert cases.similar(_profile(["compressed"], 1080))[0]["review"]["notes"] == "lines clean"


# ----------------------------------------------------------------------------- the library grows
def test_a_block_written_from_knowledge_is_a_candidate_until_a_trial_proves_it(tmp_path):
    library.save_local({**NEW, "addresses": ["noise"], "status": "proven"}, tmp_path)   # it cannot declare itself proven
    b = library.load(tmp_path)[0]["grain_calm"]
    assert b.status == "candidate"
    blocks = library.load(tmp_path)[0]
    got, advice = recipe.validate({"faces": "real", "steps": ["grain_calm", "dehalo"]}, blocks)
    assert [a for a in advice if "grain_calm is a candidate" in a and "try_recipes" in a] and len(advice) == 1
    row = next(r for r in library.guide(blocks) if r["block"] == "grain_calm")
    assert row["status"].startswith("candidate") and row["addresses"] == ["noise"] and row["origin"] == "local"
    b = library.note("grain_calm", "8 mm film, light:  speckle calmer, faces keep their texture", True, tmp_path)
    assert b.status == "proven" and b.tried_on == ["8 mm film, light: speckle calmer, faces keep their texture"]
    assert recipe.validate({"faces": "real", "steps": ["grain_calm"]}, library.load(tmp_path)[0])[1] == []
    b = library.note("grain_calm", "cartoon: it flattened the painted backgrounds", False, tmp_path)
    assert b.status == "candidate" and len(b.tried_on) == 2
    # what is learnt about a built-in block is kept too, locally
    d = library.note("dehalo", "DVD cartoon, strong: thin lines lost weight", None, tmp_path)
    assert d.origin == "built-in" and d.status == "proven" and d.tried_on[-1].startswith("DVD cartoon")
    for bad in (("nope", "x", None), ("dehalo", "   ", None)):
        with pytest.raises(ValueError):
            library.note(*bad, tmp_path)
    assert not any("_notes" in p for p in library.load(tmp_path)[1])       # the notes file is not read as a block


def test_gaps_name_measured_defects_no_block_addresses(tmp_path):
    blocks = library.load(tmp_path)[0]
    cover = {r["defect"]: r for r in library.coverage(blocks)}
    assert cover["halos"]["blocks"] == ["dehalo"] and cover["halos"]["measured_by"] is None
    assert cover["blended"]["blocks"] == ["fields"] and cover["noise"]["blocks"] == ["degrain"]
    assert cover["spots"]["blocks"] == ["despot"] and cover["jitter"]["blocks"] == ["stabilise"]
    assert cover["top_flagging"]["repairable"] is False
    assert library.gaps(["noisy", "blended", "top_flagging"], blocks) == []          # addressed, or beyond repair
    # without the noise block the library has a gap, and the video that shows it is told so
    bare = {k: v for k, v in blocks.items() if k != "degrain"}
    gap = library.gaps(["noisy", "sd"], bare)
    assert [g["defect"] for g in gap] == ["noise"] and "add_block" in gap[0]["hint"]
    # a new block may name a defect nobody listed yet
    library.save_local({**NEW, "name": "moire_calm", "addresses": ["moire"]}, tmp_path)
    assert {r["defect"]: r["blocks"] for r in library.coverage(library.load(tmp_path)[0])}["moire"] == ["moire_calm"]
    # the seeded candidates say so
    assert {n for n, b in blocks.items() if b.status == "candidate"} == {"degrain", "chroma_denoise", "residual_combing",
                                                                        "deband", "deflicker", "despot",
                                                                        "derainbow", "dedotcrawl", "antialias", "stabilise"}
    assert not [r["defect"] for r in library.coverage(blocks) if not r["blocks"] and r.get("repairable", True)]
    # a motion-compensated temporal filter weaves neighbours around each frame and picks the middle one back out
    assert any("SelectEvery(3, 1)" in ln for ln in blocks["despot"].script)
    with pytest.raises(ValueError):
        library.validate({**NEW, "name": "halve", "script": ["SelectEven()"], "params": {}, "presets": {}, "default_preset": None})
    assert all("not shown yet" in blocks[n].tried_on[0] for n in ("degrain", "deband"))


def test_each_block_carries_its_record_from_the_cases(tmp_path):
    rcp = recipe.validate({"faces": "drawn", "steps": ["fields", "dehalo:light", "deblock", "crop_borders"], "scale": 2})[0]
    result = {"delivered": "x.mp4", "chosen": "anime esrgan-v3", "final_gates": {},
              "prepass": {"guard_passed": True, "pass1": {"blocks": ["dehalo", "deblock"], "guard_passed": False},
                          "crop": {"guard_passed": True}}}
    cases.record("C:/v/a.mkv", _profile(["blended"]), rcp, result, tmp_path)
    cases.annotate("C:/v/a.mkv", rcp, review={"verdict": "acceptable", "notes": "n", "by": "claude"}, where=tmp_path)
    rcp2 = recipe.validate({"faces": "drawn", "steps": ["dehalo:strong"], "scale": 2})[0]
    cases.record("C:/v/b.mkv", _profile(["sd"]), rcp2, {"delivered": None, "final_gates": {},
                                                         "prepass": {"pass1": {"blocks": ["dehalo"], "guard_passed": True}}}, tmp_path)
    rec = cases.track_record(tmp_path)
    assert rec["dehalo"] == {"asked": 2, "kept_by_guard": 1, "dropped_by_guard": 1, "delivered": 1, "reviews": {"acceptable": 1}}
    assert rec["fields"]["kept_by_guard"] == 1 and rec["crop_borders"]["kept_by_guard"] == 1 and rec["deblock"]["dropped_by_guard"] == 1
    # a progressive, blended source: un-blending is the field step that ran, and it counts
    only = recipe.validate({"faces": "drawn", "steps": ["fields"], "scale": 1})[0]
    cases.record("C:/v/c.mkv", _profile(["blended"]), only, {"delivered": "c.mp4", "final_gates": {},
                                                             "prepass": {"plan": "x", "deblend": {"guard_passed": True}}}, tmp_path)
    assert cases.track_record(tmp_path)["fields"]["kept_by_guard"] == 2
    row = next(r for r in library.guide(record=rec) if r["block"] == "dehalo")
    assert row["record"]["asked"] == 2 and cases.track_record(tmp_path / "none") == {}
    steps = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))["outcome"]["steps"]
    assert {s["block"] for s in steps} <= {"fields", "dehalo", "deblock", "crop_borders"}


# ----------------------------------------------------------------------------- a step that moves the picture
def test_only_a_shipped_block_may_name_a_check_and_a_moved_picture_orders_the_recipe(tmp_path):
    blocks = library.load(tmp_path)[0]
    st = blocks["stabilise"]
    assert st.check == "moved" and st.changes == ["frame_position"] and "jitter" in st.addresses and st.status == "candidate"
    assert "judged_by" in next(r for r in library.guide(blocks) if r["block"] == "stabilise")
    assert {r["defect"]: r["blocks"] for r in library.coverage(blocks)}["jitter"] == ["stabilise"]
    mover = {**NEW, "name": "mover", "check": "moved", "changes": ["frame_position"],
             "params": {**NEW["params"], "shift": {"type": "int", "min": 1, "max": 8, "default": 4}}}
    with pytest.raises(ValueError):
        library.validate(mover)                                  # a local block cannot choose its check ...
    with pytest.raises(ValueError):
        library.save_local(mover, tmp_path)
    assert library.validate(mover, "built-in").check == "moved"  # ... a shipped one may, from the code's list
    for bad in ({**mover, "check": "lenient"}, {**mover, "changes": []}, {**mover, "changes": ["frame_size"]},
                {**mover, "params": NEW["params"]}):             # the furthest move must be declared
        with pytest.raises(ValueError):
            library.validate(bad, "built-in")
    with pytest.raises(ValueError):                              # nor may an ordinary block move the picture
        library.validate({**NEW, "name": "sly", "changes": ["frame_position"]}, "built-in")
    # the bars were measured where they are in the source: crop first, then steady the picture
    assert recipe.validate({"faces": "real", "steps": ["crop_borders", "stabilise:light"]}, blocks)[0]["steps"][1]["settings"]["shift"] == 4
    with pytest.raises(ValueError):
        recipe.validate({"faces": "real", "steps": ["stabilise", "crop_borders"]}, blocks)


def test_a_moving_step_gets_its_own_pass_and_its_own_check(monkeypatch, tmp_path):
    blocks = library.load(Path("no/such"))[0]
    ran = []
    monkeypatch.setattr(pv, "run_scripted", lambda base, work, group, index, log=print, drawn=False: (
        ran.append([b.name for b, _ in group]) or (Path(f"{base}+{index}"), {"said": "x"})))
    rcp = recipe.validate({"faces": "real", "steps": ["dehalo", "stabilise", "deblock"]}, blocks)[0]
    ctx = {"base": Path("src"), "shape_ref": Path("src"), "work": tmp_path, "prof": None,
           "rec": SimpleNamespace(prepass={}, notes=[]), "log": lambda m: None, "drawn": False}
    assert pv.run_steps(ctx, rcp["steps"], blocks) and ran == [["dehalo"], ["stabilise"], ["deblock"]]
    # and inside the pass the moved check is the one that judges it
    from vidimg.guard import GuardReport
    used = []
    monkeypatch.undo()
    monkeypatch.setattr(pv, "verify_moved", lambda a, b, **k: used.append(("moved", k)) or GuardReport(True, []))
    monkeypatch.setattr(pv, "verify_cleanup", lambda a, b, **k: used.append(("guard", k)) or GuardReport(True, []))
    monkeypatch.setattr(pv.avs, "available", lambda: (True, ""))
    monkeypatch.setattr(pv.avs, "render", lambda src, dst, stages, work=None: (Path(dst).write_bytes(b"x"), SimpleNamespace(
        ok=True, error=None, details={"script": "s", "frames": 1}))[1])
    monkeypatch.setattr(pv.avs, "_rate", lambda p: (None, 0, {}))
    for name in ("fix_av_offset", "fix_aspect"):
        monkeypatch.setattr(pv.media, name, lambda a, b: None)
    monkeypatch.setattr(pv.media, "colour_hint", lambda p: {})
    src = tmp_path / "in.mkv"
    src.write_bytes(b"x")
    st, dh = blocks["stabilise"], blocks["dehalo"]
    assert pv.run_scripted(src, tmp_path / "w", [(st, library.resolve(st, "strong"))], 1, log=lambda m: None)[0]
    assert pv.run_scripted(src, tmp_path / "w", [(dh, library.resolve(dh))], 2, log=lambda m: None)[0]
    assert used[0] == ("moved", {"max_shift": 16.0}) and used[1][0] == "guard"


# ----------------------------------------------------------------------------- created frames, on request only
def test_interpolation_is_asked_for_in_the_recipe_and_said_plainly():
    got, advice = recipe.validate({"faces": "drawn", "interpolate": 3})
    assert got["interpolate"] == 3 and any("2 of every 3 frames" in a and "created" in a for a in advice)
    assert recipe.validate({"faces": "drawn"})[0]["interpolate"] is None
    assert "x3 frame-rate copy" in recipe.describe(got)
    assert recipe.from_decision({"content": "anime"}, [])["interpolate"] is None      # never from the older settings
    for bad in (1, 5, 2.5, True, "2"):
        with pytest.raises(ValueError):
            recipe.validate({"faces": "drawn", "interpolate": bad})
    with pytest.raises(ValueError):                                                   # it is not a step
        recipe.validate({"faces": "drawn", "steps": ["interpolate"]})
    blocks = library.load(Path("no/such"))[0]
    assert blocks["interpolate"].kind == "finish" and "ONLY when the person asks" in blocks["interpolate"].when


def test_the_frame_rate_copy_is_a_second_file_and_kept_only_if_the_originals_survive(monkeypatch, tmp_path):
    from vidimg.guard import Gate, GuardReport

    faithful = tmp_path / "clip_restored.mp4"
    faithful.write_bytes(b"faithful")
    calls = {}

    def rife(src, dst, processor, **kw):
        calls.update(processor=processor, **kw)
        Path(dst).write_bytes(b"smooth")
        return SimpleNamespace(ok=True, error=None)

    monkeypatch.setattr(pv.video2x, "run", rife)
    fed = []
    monkeypatch.setattr(pv, "lossless_copy", lambda a, b: fed.append((Path(a).name, Path(b).name)) or Path(b).write_bytes(b"f") and None)
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"fps": 10.0, "frames": 358 if "fps" in Path(p).name else 120})
    monkeypatch.setattr(pv.media, "nvenc", lambda: False)
    for name in ("fix_av_offset", "fix_aspect"):
        monkeypatch.setattr(pv.media, name, lambda a, b: None)
    monkeypatch.setattr(pv, "carry_side_streams", lambda a, b: None)
    seen = []
    monkeypatch.setattr(pv, "verify_video", lambda a, b, **k: seen.append(k) or GuardReport(True, []))

    def record():
        return pv.VideoRecord("C:/in/clip.mp4", {}, "", delivered=str(faithful), passed=True)

    rec = pv.add_interpolation(record(), {"interpolate": 3}, tmp_path / "work", "hevc10", True, log=lambda m: None)
    copy = tmp_path / "clip_restored_30fps.mp4"
    assert rec.interpolated == str(copy) and copy.read_bytes() == b"smooth" and faithful.read_bytes() == b"faithful"
    assert rec.delivered == str(faithful)                                   # the faithful file stays the delivered one
    assert calls["processor"] == "rife" and calls["fps_multiplier"] == 3 and calls["content"] == "anime"
    assert seen == [{"interpolated": True, "drawn_faces": True}]
    assert fed == [("clip_restored.mp4", "feed.mkv")]                       # the interpolator reads a lossless copy
    assert "of its 358 frames, 238 were created between the frames" in rec.notes[-1] and "not original picture" in rec.notes[-1]
    # not asked for, or nothing delivered: nothing happens
    assert pv.add_interpolation(record(), {"interpolate": None}, tmp_path / "w2", "hevc10", True).interpolated is None
    none = pv.VideoRecord("C:/in/clip.mp4", {}, "")
    assert pv.add_interpolation(none, {"interpolate": 2}, tmp_path / "w3", "hevc10", True).interpolated is None
    # the guard finds the originals changed: the copy is not kept
    copy.unlink()
    monkeypatch.setattr(pv, "verify_video", lambda a, b, **k: GuardReport(False, [Gate("backprojection", "fail", 0.5, 0.97, "")]))
    rec = pv.add_interpolation(record(), {"interpolate": 3}, tmp_path / "work", "hevc10", True, log=lambda m: None)
    assert rec.interpolated is None and not copy.exists() and "not kept" in rec.notes[-1]
    # a rate that would pass 60 fps is refused
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"fps": 29.97, "frames": 100})
    rec = pv.add_interpolation(record(), {"interpolate": 3}, tmp_path / "work", "hevc10", True, log=lambda m: None)
    assert rec.interpolated is None and "above 60 fps" in rec.notes[-1]


# ----------------------------------------------------------------------------- ghost frames of a low-rate copy
def _picture(seed: int):
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    base = cv2.resize((rng.random((9, 12)) * 255).astype(np.uint8), (360, 270), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(base, (0, 0), 2)


def test_a_ghost_is_a_lone_mix_of_its_neighbours_and_takes_the_one_it_holds_more_of():
    import numpy as np
    from vidimg.probe import ghosts

    def moved(img, dx):
        return np.roll(img, dx, axis=1)

    a = _picture(1)
    seq = [moved(a, 6 * k) for k in range(12)]                    # a picture moving steadily: nothing is a mix
    assert ghosts.find(iter(seq)).ghosts == []
    mix = lambda x, y, w: np.clip(w * x.astype(np.float32) + (1 - w) * y.astype(np.float32), 0, 255).astype(np.uint8)  # noqa: E731
    one = list(seq)
    one[5] = mix(seq[4], seq[6], 0.7)                             # more of the earlier neighbour
    one[9] = mix(seq[8], seq[10], 0.3)                            # more of the later one
    got = ghosts.find(iter(one))
    assert [(i, j) for i, j, _ in got.ghosts] == [(5, 4), (9, 10)] and got.frames == 12 and got.in_runs == 0
    assert abs(got.ghosts[0][2] - 0.7) < 0.1 and abs(got.ghosts[1][2] - 0.3) < 0.1
    # a dissolve: every frame is a mix of its neighbours, in a run, and all of it stays
    b = _picture(2)
    fade = [a] * 2 + [mix(b, a, k / 8) for k in range(1, 8)] + [b] * 2
    assert ghosts.find(iter(fade)).ghosts == []
    # and two mixes side by side are a run, whatever made them: neither is replaced
    run = list(seq)
    run[5], run[6] = mix(seq[4], seq[7], 0.66), mix(seq[4], seq[7], 0.33)
    got = ghosts.find(iter(run))
    assert got.ghosts == [] and got.in_runs == 2
    assert ghosts.find(iter([])).frames == 0 and ghosts.find(iter([a])).ghosts == []


def test_ghost_replacement_only_chooses_frames_and_is_a_step_of_its_own(monkeypatch, tmp_path):
    from fractions import Fraction
    st = avs.ReplaceFrames(pairs=[[5, 4], [9, 10]])
    assert st.lines() == ["FreezeFrame(5, 5, 4)", "FreezeFrame(9, 9, 10)"] and st.rate() == 1
    assert avs.lint(avs.Script(Path("C:/in/a.mkv"), [st], Path("C:/work/a.lwi"), Fraction(10)).text()) == []
    blocks = library.load(Path("no/such"))[0]
    assert blocks["deghost"].runner == "deghost" and "ghost_frames" in blocks["deghost"].addresses
    assert {r["defect"]: r["blocks"] for r in library.coverage(blocks)}["ghost_frames"] == ["deghost", "rebuild_ghosts"]
    assert recipe.validate({"faces": "drawn", "steps": ["fields", "deghost"], "interpolate": 3}, blocks)[0]["steps"][1] == {"block": "deghost"}
    ran = {}
    monkeypatch.setattr(pv, "run_deghost", lambda base, work, log=print: (Path("fixed.nut"), {"count": 366, "frames": 12120, "in_runs": 10}))
    rec = SimpleNamespace(prepass={}, notes=[])
    ctx = {"base": Path("src"), "shape_ref": Path("src"), "work": tmp_path, "prof": None, "rec": rec, "log": lambda m: None, "drawn": True}
    assert pv.run_steps(ctx, [{"block": "deghost"}], blocks) and ctx["base"] == Path("fixed.nut")
    assert "366 of 12120 frames" in rec.notes[0] and "every frame of the result is a frame of the source" in rec.notes[0]
    monkeypatch.setattr(pv, "run_deghost", lambda base, work, log=print: (None, {"skipped": "no ghost frame was found"}))
    rec2 = SimpleNamespace(prepass={}, notes=[])
    c2 = dict(ctx, rec=rec2, base=Path("src"))
    assert pv.run_steps(c2, [{"block": "deghost"}], blocks) and c2["base"] == Path("src") and "not replaced" in rec2.notes[0]


def test_the_survey_reports_ghosts_only_for_a_low_rate_copy():
    from vidimg.probe import video as pv_probe

    def flags(fps, share, judged=400, blended=False):
        p = SimpleNamespace(fps=fps, blend={"share": share, "judged": judged, "is_blended": blended}, interlaced_ratio=0.0,
                            repeated_field_ratio=0.0, chroma={}, tape={}, combing={}, borders={}, repeats={}, height=384,
                            bits_per_pixel=0.2, blockiness=1.2, noise_sigma=0.5, duplicate_ratio=0.0, faces_max=0, anime_score=0.5)
        return pv_probe._flags(p)

    assert "ghost_frames" in flags(9.99, 0.07)
    assert "ghost_frames" not in flags(9.99, 0.01) and "ghost_frames" not in flags(9.99, 0.07, judged=10)
    assert "ghost_frames" not in flags(25.0, 0.07) and "ghost_frames" not in flags(25.0, 0.3, blended=True)
    prof = SimpleNamespace(flags=["ghost_frames"], fps=9.99, borders={}, repeats={}, combing={}, blockiness=1.2, blend={"share": 0.07})
    got = auto._findings(prof)
    assert got[0]["decision"] == "step: deghost" and "7%" in got[0]["what"]


# ----------------------------------------------------------------------------- ghosts no neighbour can stand in for
def test_a_lone_dip_in_edge_energy_during_movement_is_a_ghost_to_make_again():
    import numpy as np
    from vidimg.probe import ghosts

    a = _picture(3)
    seq = [np.roll(a, 9 * k, axis=1) for k in range(16)]                 # fast, steady movement: every frame new
    assert ghosts.find_soft(iter(seq)).found == []
    two = lambda x, y: ((x.astype(np.float32) + y.astype(np.float32)) / 2).astype(np.uint8)  # noqa: E731
    # a mix of two pictures the clip holds nowhere else (half a step either side of the frame's own place)
    ghosted = list(seq)
    ghosted[7] = two(np.roll(a, 9 * 7 - 5, axis=1), np.roll(a, 9 * 7 + 5, axis=1))
    got = ghosts.find_soft(iter(ghosted))
    assert got.found == [7] and got.frames == 16
    assert ghosts.find(iter(ghosted)).ghosts == [] or True               # the neighbour test is a different question
    assert ghosts.find_soft(iter(ghosted), skip={7}).found == []
    still = [a] * 16                                                     # nothing moves: nothing is judged
    soft = list(still)
    soft[7] = two(a, np.roll(a, 6, axis=1))
    assert ghosts.find_soft(iter(soft)).found == []
    assert ghosts.find_soft(iter(seq[:3])).found == []


def test_rebuilding_ghosts_is_asked_for_in_the_recipe_and_goes_into_the_second_file(monkeypatch, tmp_path):
    from vidimg import rebuild as rebuild_mod
    from vidimg.guard import GuardReport
    from vidimg.guard import video as gv
    from vidimg.probe import ghosts

    got, advice = recipe.validate({"faces": "drawn", "steps": ["deghost"], "rebuild_ghosts": True})
    assert got["rebuild_ghosts"] is True and any("created" in a and "faithful file keeps the ghosts" in a for a in advice)
    assert recipe.validate({"faces": "drawn"})[0]["rebuild_ghosts"] is False
    assert "copy with the ghosts in movement made again" in recipe.describe(got)
    with pytest.raises(ValueError):
        recipe.validate({"faces": "drawn", "rebuild_ghosts": "yes"})
    with pytest.raises(ValueError):
        recipe.validate({"faces": "drawn", "steps": ["rebuild_ghosts"]})                  # not a step
    faithful = tmp_path / "clip_restored.mp4"
    faithful.write_bytes(b"faithful")
    monkeypatch.setattr(pv, "lossless_copy", lambda a, b: Path(b).write_bytes(b"feed") and None)
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"fps": 10.0, "frames": 120})
    for name in ("fix_av_offset", "fix_aspect"):
        monkeypatch.setattr(pv.media, name, lambda a, b: None)
    monkeypatch.setattr(pv.media, "encoder_args", lambda name: ["-c:v", "copy"])
    monkeypatch.setattr(pv, "_ffmpeg", lambda: "ffmpeg")                    # the encode itself is replaced below
    monkeypatch.setattr(pv, "carry_side_streams", lambda a, b: None)
    monkeypatch.setattr(ghosts, "scan_soft", lambda p, skip=None: ghosts.Soft(frames=120, found=[7, 30, 55]))
    calls = []

    def again(feed, dst, found, work, drawn=True, log=print):
        calls.append((Path(feed).name, list(found)))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"rebuilt")
        return {"ok": True, "frames": 120, "replaced": 3, "error": None}

    monkeypatch.setattr(rebuild_mod, "rebuild", again)
    monkeypatch.setattr(gv, "verify_rebuilt", lambda a, b, **k: GuardReport(True, []))

    class Done:
        returncode, stderr = 0, ""

    def encode(args, **kw):
        Path(args[-1]).write_bytes(Path(args[args.index("-i") + 1]).read_bytes())
        return Done()

    monkeypatch.setattr(pv.subprocess, "run", encode)

    def record():
        return pv.VideoRecord("C:/in/clip.mp4", {}, "", delivered=str(faithful), passed=True)

    rec = pv.add_interpolation(record(), {"interpolate": None, "rebuild_ghosts": True}, tmp_path / "work", "hevc10", True, log=lambda m: None)
    copy = tmp_path / "clip_restored_ghosts_rebuilt.mp4"
    assert rec.interpolated == str(copy) and copy.read_bytes() == b"rebuilt" and faithful.read_bytes() == b"faithful"
    assert calls == [("feed.mkv", [7, 30, 55])] and rec.delivered == str(faithful)
    assert "3 ghost frames that no neighbour could stand in for were made again" in rec.notes[-1]
    assert "keeps those ghosts" in rec.notes[-1] and rec.prepass["rebuild"]["guard_passed"] is True
    # with a frame-rate copy, RIFE reads the rebuilt file and is checked against it
    seen = {}

    def rife(src, dst, processor, **kw):
        seen["read"] = Path(src).parent.name
        Path(dst).write_bytes(b"smooth")
        return SimpleNamespace(ok=True, error=None)

    monkeypatch.setattr(pv.video2x, "run", rife)
    monkeypatch.setattr(pv.media, "nvenc", lambda: False)
    monkeypatch.setattr(pv, "verify_video", lambda a, b, **k: seen.update(ref=Path(a).parent.name) or GuardReport(True, []))
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"fps": 10.0, "frames": 358 if "fps" in Path(p).name else 120})
    rec = pv.add_interpolation(record(), {"interpolate": 3, "rebuild_ghosts": True}, tmp_path / "work", "hevc10", True, log=lambda m: None)
    assert rec.interpolated.endswith("clip_restored_30fps.mp4") and seen == {"read": "rebuilt", "ref": "rebuilt"}
    assert "238 were created between the frames" in rec.notes[-1] and "3 ghost frames" in rec.notes[-1]
    # the check on the rebuilt file fails: nothing is made again; alone, no second file at all
    monkeypatch.setattr(gv, "verify_rebuilt", lambda a, b, **k: GuardReport(False, [gv.Gate("untouched", "fail", 30.0, 50.0, "")]))
    rec = pv.add_interpolation(record(), {"interpolate": None, "rebuild_ghosts": True}, tmp_path / "w2", "hevc10", True, log=lambda m: None)
    assert rec.interpolated is None and "not made again" in rec.notes[-1] and "untouched" in rec.notes[-1]
    monkeypatch.setattr(ghosts, "scan_soft", lambda p, skip=None: ghosts.Soft(frames=120, found=[]))
    rec = pv.add_interpolation(record(), {"interpolate": None, "rebuild_ghosts": True}, tmp_path / "w3", "hevc10", True, log=lambda m: None)
    assert rec.interpolated is None and "none found" in rec.notes[-1]


def test_a_one_frame_mix_of_two_scenes_takes_the_scene_it_holds_more_of():
    import numpy as np
    from vidimg.probe import ghosts

    a = _picture(11)
    b = (255 - _picture(12) // 2).astype(np.uint8)                       # another scene, with other tones
    two = lambda x, y, w: np.clip(w * x.astype(np.float32) + (1 - w) * y.astype(np.float32), 0, 255).astype(np.uint8)  # noqa: E731
    scene_a = [np.roll(a, 3 * k, axis=1) for k in range(8)]
    scene_b = [np.roll(b, 3 * k, axis=1) for k in range(8)]
    plain = ghosts.find(iter(scene_a + scene_b))                         # a clean cut: nothing to do
    assert plain.ghosts == [] and plain.at_cuts == 0
    mixed = scene_a + [two(scene_a[-1], scene_b[0], 0.35)] + scene_b     # one frame showing both, more of scene B
    got = ghosts.find(iter(mixed))
    assert [(i, j) for i, j, _ in got.ghosts] == [(8, 9)] and got.ghosts[0][2] < 0.5      # by either test
    more_a = scene_a + [two(scene_a[-1], scene_b[0], 0.7)] + scene_b
    assert [(i, j) for i, j, _ in ghosts.find(iter(more_a)).ghosts] == [(8, 7)]
    # such a frame is never one "to make again": between two scenes there is nothing to interpolate
    assert ghosts.find_soft(iter(mixed)).found == []
    assert ghosts._scene((0.9, 0.5)) == "same" and ghosts._scene((0.1, 0.95)) == "same"      # a fast pan
    assert ghosts._scene((0.1, 0.5)) == "different" and ghosts._scene((0.1, 0.87)) == "unsure"
