"""How long a recipe will take: stage rates learnt from the runs timed on this machine, built-in rates before that."""

import json
from pathlib import Path

from vidimg import auto, cases, estimate, recipe

SD = {"width": 640, "height": 480, "frames": 30000}
HD = {"width": 1440, "height": 1080, "frames": 30000}


def _recipe(**kw):
    return recipe.validate({"faces": "drawn", **kw})[0]


def test_the_size_a_recipe_writes_and_the_model_it_starts_with():
    assert estimate.out_size(640, 480, _recipe()) == (1280, 960, 2.0)            # a small drawing: x2
    assert estimate.out_size(1440, 1080, _recipe()) == (1440, 1080, 1.0)         # a large one keeps its size
    assert estimate.out_size(640, 480, _recipe(scale=1)) == (640, 480, 1.0)
    assert estimate.out_size(1280, 720, {"faces": "real", "scale": None}) == (1920, 1080, 1.5)
    assert estimate.first_model(_recipe(), 1000, 640, 480) == "anime esrgan-v3"
    assert estimate.first_model(_recipe(models=["cugan"]), 1000, 640, 480) == "anime cugan"
    assert estimate.first_model({"faces": "real", "models": []}, 2000, 640, 480) == "free realesrnet"
    assert estimate.first_model({"faces": "real", "models": []}, 200000, 1920, 1080) == "free general-v3"


def test_before_any_run_is_timed_the_built_in_rates_are_used_and_said_so(tmp_path):
    e = estimate.estimate(SD, _recipe(steps=["fields"], models=["esrgan-v3"], scale=2), tmp_path)
    assert e["model"] == "anime esrgan-v3" and e["output"] == "1280x960" and "built-in rates" in e["basis"]
    assert set(e["stages_min"]) == {"steps", "previews", "model", "finish", "check"}
    assert e["range"][0] < e["minutes"] < e["range"][1]
    # the stages that used to be left out are a large part of the whole
    assert e["stages_min"]["finish"] > 0.5 * e["stages_min"]["model"]
    # a drawing kept at its own size: the anime model works at x2 inside, four times the pixels of the result
    own = estimate.estimate(HD, _recipe(models=["esrgan-v3"], scale=1), tmp_path)
    twice = estimate.estimate(HD, _recipe(models=["general-v3"], scale=1), tmp_path)
    assert own["stages_min"]["model"] > 100 and own["output"] == "1440x1080" and twice["model"] == "free general-v3"
    # asked-for extras are counted, nothing else is
    more = estimate.estimate(SD, _recipe(models=["esrgan-v3"], scale=2, interpolate=3, rebuild_ghosts=True), tmp_path)
    assert more["stages_min"]["second_file"] > 8 and "steps" not in more["stages_min"]
    assert estimate.estimate({"width": 640}, _recipe(), tmp_path) is None and estimate.estimate({}, _recipe(), tmp_path) is None


def test_timed_runs_on_this_machine_replace_the_built_in_rates(tmp_path):
    rcp = _recipe(steps=["fields"], models=["esrgan-v3"], scale=2, interpolate=3)
    record = {"delivered": "x.mp4", "chosen": "anime esrgan-v3", "final_gates": {},
              "timings": {"steps": 100.0, "previews": 60.0, "model": 1000.0, "finish": 500.0, "check": 100.0, "second_file": 900.0},
              "work_done": {"frames": 10000, "in_megapixels": 0.3, "out_megapixels": 1.2, "same_size_anime": False}}
    cases.record("C:/v/a.mkv", {"flags": []}, rcp, record, tmp_path)
    got = estimate.learned(tmp_path)
    assert got["model:anime esrgan-v3"] == [12.0] and got["finish"] == [30.0] and got["steps"] == [30.0]
    assert got["second_file"] == [40.0]
    # the same video again: the estimate is what the run took
    e = estimate.estimate({"width": 632, "height": 474, "frames": 10000}, rcp, tmp_path)
    assert abs(e["stages_min"]["model"] - 1000 / 60) < 1 and abs(e["stages_min"]["finish"] - 500 / 60) < 1
    assert "timed runs on this machine" in e["basis"] and "anime esrgan-v3" in e["basis"]
    # another model has not been timed here: its own built-in rate, and the basis says the rest is built in
    other = estimate.estimate(SD, _recipe(models=["cugan"], scale=2), tmp_path)
    assert other["model"] == "anime cugan" and "built-in" in other["basis"]
    # a drawing kept at its own size is rated by what the model writes inside (x2)
    own = dict(record, work_done={"frames": 10000, "in_megapixels": 1.5, "out_megapixels": 1.5, "same_size_anime": True},
               timings={"model": 6000.0, "finish": 1000.0})
    cases.record("C:/v/b.mkv", {"flags": []}, _recipe(models=["esrgan-v3"], scale=1), own, tmp_path)
    assert sorted(estimate.learned(tmp_path)["model:anime esrgan-v3"]) == [10.0, 12.0]
    # a case without timings (an older one) teaches nothing and breaks nothing
    (tmp_path / "old.json").write_text(json.dumps({"source": "o.mkv", "recipe": rcp, "outcome": {"model": "anime esrgan-v3"}}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert len(estimate.learned(tmp_path)["finish"]) == 2


def test_a_run_writes_its_timings_into_its_case_and_a_later_note_keeps_them(tmp_path):
    rcp = _recipe(models=["esrgan-v3"], scale=2)
    rec = {"delivered": "x.mp4", "chosen": "anime esrgan-v3", "final_gates": {}, "timings": {"model": 50.0, "finish": 40.0},
           "work_done": {"frames": 100, "in_megapixels": 0.3, "out_megapixels": 1.2}}
    cases.record("C:/v/a.mkv", {"flags": []}, rcp, rec, tmp_path)
    cases.record("C:/v/a.mkv", {"flags": []}, rcp, {"delivered": "x.mp4", "final_gates": {}}, tmp_path)   # recorded again, untimed
    cases.annotate("C:/v/a.mkv", rcp, verdict="fine", where=tmp_path)
    c = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert c["timings"] == {"model": 50.0, "finish": 40.0} and c["work_done"]["frames"] == 100 and c["verdict"] == "fine"


def test_the_survey_gives_an_estimate_for_the_recipe_as_it_stands(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, "folder", lambda: tmp_path / "cases")
    plan = {"version": 1, "created": "now", "out": str(tmp_path), "inputs": [], "ignored_files": 0, "contact_sheets": [],
            "items": [{"id": "v1", "type": "video", "source": "C:/in/clip.mp4", "guide": {}, "decided_by": "analysis",
                       "reasons": [], "state": "pending", "work": dict(HD),
                       "proposed_recipe": _recipe(models=["general-v3"], scale=1),
                       "decision": {"content": "auto", "recipe": None}},
                      {"id": "v2", "type": "video", "source": "C:/in/old.mp4", "guide": {}, "decided_by": "analysis",
                       "reasons": [], "state": "pending", "decision": {"content": "auto", "recipe": None}}]}
    auto.save(plan)
    s = auto.summary(auto.load(tmp_path))
    first = s["items"][0]["estimate"]
    assert first["model"] == "free general-v3" and first["minutes"] > 60 and "estimate" in s["next"]
    assert "estimate" not in s["items"][1]                                  # surveyed before sizes were kept
    plan = auto.decide(tmp_path, {"v1": {"recipe": {"faces": "drawn", "models": ["esrgan-v3"], "scale": 1}}})
    again = auto.summary(plan)["items"][0]["estimate"]
    assert again["model"] == "anime esrgan-v3" and again["minutes"] != first["minutes"]
