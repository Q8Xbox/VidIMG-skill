"""How long a recipe will take on this machine, from the runs that were timed on it.

The first estimates were one number per model, taken from a benchmark, and they were about half the truth on whole
videos: 25 minutes said, 56 taken; 2.5 hours said, 4.5 taken (owner samples, 2026-11-20). They counted the model
and nothing else. A run is five stages, and on a large frame the final encode alone can take as long as the model:

    steps       every step before the model is a lossless pass over the whole video, plus its check
    previews    three short clips per model tried, each checked
    model       by far the largest for the slow models; an anime model at the source's own size works at 2x inside
    finish      texture return and the final encode
    check       the final guard
    second file a frame-rate copy or rebuilt ghosts, when asked for

Each stage has a rate: megapixel-frames per second. Every run records its timings and what it worked on in its
case (cases.py), and the estimate uses the median rate of the earlier runs for each stage (for the model: of the
runs with that model). Where no run has been timed yet, the built-in rates below are used: they come from the
owner's runs on one machine (RTX 5060 Ti) and are a starting point, not a promise.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from . import cases

# megapixel-frames per second, per stage (see the module text). For the model: megapixels the model WRITES.
BUILT_IN = {
    "model:anime esrgan-v3": 26.0,      # 24-31 measured on four whole videos (SD x2, and 1440x1080 at its own size)
    "model:anime cugan": 20.0,          # not timed on a whole video yet; a little slower than anime-v3 on previews
    "model:anime cugan-pro": 20.0,
    "model:free general-v3": 8.0,       # one whole 1080p video at its own size
    "model:free realesrnet": 0.45,      # megapixels READ per second (0.47 fps at 1280x720)
    "model:free bsrnet": 0.45,
    "finish": 27.0,                     # megapixels read AND written: the texture pass reads the model's input
                                        # beside its output, so a video kept at its own size (a large input) costs
                                        # far more per output pixel than an enlarged SD one. 19-42 measured on
                                        # six whole videos this way; per output megapixel alone it was 10-35
    "steps": 45.0,                      # megapixels read, per step, render and check together
    "second_file": 25.0,                # megapixel-frames of the copy
}
FIXED = {"previews": 90.0, "check": 150.0, "rebuild_ghosts": 480.0}      # seconds, largely independent of size
SLOW_INPUT_MODELS = ("free realesrnet", "free bsrnet")                  # rated by what they read, not what they write
LABELS = {"esrgan-v3": "anime esrgan-v3", "cugan": "anime cugan", "cugan-pro": "anime cugan-pro",
          "general-v3": "free general-v3", "realesrnet": "free realesrnet", "bsrnet": "free bsrnet"}
SPREAD = 0.3                            # the range given around the estimate


def out_size(width: int, height: int, recipe: dict) -> tuple[int, int, float]:
    """(width, height, scale) the recipe will write: its scale, or the tool's default for the source's size."""
    scale = recipe.get("scale")
    if scale is None:
        if recipe.get("faces") == "drawn":
            scale = 2 if height <= 720 else 1
        else:
            scale = 2 if height <= 540 else (1080 / height if height <= 800 else 1)
    return round(width * scale / 2) * 2, round(height * scale / 2) * 2, float(scale)


def first_model(recipe: dict, frames: int, width: int, height: int) -> str:
    if recipe.get("models"):
        return LABELS.get(recipe["models"][0], recipe["models"][0])
    if recipe.get("faces") == "drawn":
        return "anime esrgan-v3"
    hours = frames * width * height / 1e6 / BUILT_IN["model:free realesrnet"] / 3600
    return "free realesrnet" if hours <= 3.0 else "free general-v3"


def learned(where: Path | None = None) -> dict:
    """Rates measured on this machine: {stage: [megapixel-frames per second, one per timed run]}."""
    where = where if where is not None else cases.folder()
    out: dict = {}
    for f in where.glob("*.json") if where.is_dir() else []:
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        t, w = c.get("timings") or {}, c.get("work_done") or {}
        frames, mp_in, mp_out = w.get("frames"), w.get("in_megapixels"), w.get("out_megapixels")
        if not (frames and mp_in and mp_out):
            continue
        label = (c.get("outcome") or {}).get("model")
        if label and t.get("model", 0) > 5:
            if label in SLOW_INPUT_MODELS:
                work = frames * mp_in
            else:
                work = frames * mp_out * (4 if w.get("same_size_anime") else 1)     # x2 inside, scaled back
            out.setdefault(f"model:{label}", []).append(work / t["model"])
        if t.get("finish", 0) > 5:
            out.setdefault("finish", []).append(frames * (mp_out + mp_in) / t["finish"])
        n_steps = len((c.get("recipe") or {}).get("steps") or [])
        if n_steps and t.get("steps", 0) > 5:
            out.setdefault("steps", []).append(frames * mp_in * n_steps / t["steps"])
        mult = (c.get("recipe") or {}).get("interpolate")
        if mult and t.get("second_file", 0) > 5:
            out.setdefault("second_file", []).append(frames * mult * mp_out / t["second_file"])
    return out


def estimate(work: dict, recipe: dict, where: Path | None = None) -> dict | None:
    """Minutes for this recipe on this video. work: {width, height, frames} of the source.
    Returns the total, a range, the stages, and what it rests on. None when the size is not known."""
    width, height, frames = (work or {}).get("width"), (work or {}).get("height"), (work or {}).get("frames")
    if not (width and height and frames):
        return None
    seen = learned(where)
    used: list[str] = []

    def rate(key: str) -> float:
        if seen.get(key):
            used.append(key)
            return float(median(seen[key]))
        return BUILT_IN.get(key, BUILT_IN["model:free general-v3"])

    ow, oh, scale = out_size(width, height, recipe)
    mp_in, mp_out = width * height / 1e6, ow * oh / 1e6
    label = first_model(recipe, frames, width, height)
    if label in SLOW_INPUT_MODELS:
        model_work = frames * mp_in
    else:
        model_work = frames * mp_out * (4 if label.startswith("anime") and scale == 1 else 1)
    steps = len(recipe.get("steps") or [])
    stages = {"steps": steps * frames * mp_in / rate("steps") if steps else 0.0,
              "previews": FIXED["previews"] * max(1, len(recipe.get("models") or []) or 1),
              "model": model_work / rate(f"model:{label}"),
              "finish": frames * (mp_out + mp_in) / rate("finish"),
              "check": FIXED["check"]}
    extra = 0.0
    if recipe.get("interpolate"):
        extra += frames * recipe["interpolate"] * mp_out / rate("second_file")
    if recipe.get("rebuild_ghosts"):
        extra += FIXED["rebuild_ghosts"]
    if extra:
        stages["second_file"] = extra
    total = sum(stages.values())
    timed = sorted(set(used))
    return {"minutes": round(total / 60), "range": [round(total * (1 - SPREAD) / 60), round(total * (1 + SPREAD) / 60)],
            "stages_min": {k: round(v / 60, 1) for k, v in stages.items() if v},
            "model": label, "output": f"{ow}x{oh}",
            "basis": (f"timed runs on this machine for: {', '.join(k.split(':')[-1] for k in timed)}" if timed else
                      "built-in rates (no run has been timed on this machine yet)")
                     + ("" if len(timed) >= 3 else "; the rest from built-in rates measured on another setup")}
