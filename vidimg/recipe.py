"""A recipe: what one video gets, composed from the block library (library.py) by whoever supervises the
restoration. There are no paths in the pipeline; it runs the recipe.

    {"faces": "drawn" | "real",              how the guard judges faces: advisory on drawings, strict on people
     "steps": ["fields", {"block": "deblock", "preset": "light"}, "crop_borders"],
                                              the steps before the model, in the order they run
     "models": ["esrgan-v3", "cugan"],        tried in this order on the previews; the first the guard passes is used
     "model_strict": false,                   true: no fallback beyond the listed models
     "scale": 1 | 2 | ... | null,             1 = keep the size; null = the tool's default for the source's size
     "texture": 0..1 | null,                  source texture put back after the model (null = default)
     "interpolate": 2 | 3 | 4 | null}         ONLY when the person asks: also write a copy at that many times the frame
                                              rate, with frames created between the originals (RIFE). The faithful
                                              file is still delivered; the copy says what part of it is invented
     "rebuild_ghosts": true | false           ONLY when the person asks: in that second file (or in one at the original
                                              rate when interpolate is null) the ghost frames no neighbour can stand
                                              in for are made again from their neighbours (created frames too)
     "rebuild_damaged": [1184, 16060]         ONLY when the person asks: these frames of the SOURCE (numbered from 0),
                                              which whoever supervises judged to be damage (a torn or dropped-out
                                              frame, not a flash drawn on purpose), are made again from their two
                                              neighbours in that second file (created frames too). Frames named
                                              side by side (up to 3; up to 6 where the scene holds still) are made
                                              as a run from the frames around it
     "per_kind": {"k3": {"texture": 0.5, "steps": {"deblock": "strong", "dehalo": false}}}
                                              what a kind of scene gets differently (the kinds are the ones the survey
                                              found in THIS video, probe/scenes.py): its own texture strength, another
                                              preset or settings for a step of the recipe, or that step left out
                                              (false). A step may also say {"block": ..., "kinds": ["k2", "k5"]}: it
                                              then runs in those kinds only.

One model serves the whole video: a change of model at every cut would be a change of look at every cut. What may
differ by kind is how strongly a picture filter works and how much texture comes back. Only script filters on
AviSynth judged by the video guard can differ by kind (they are computed on the whole clip and chosen per stretch,
so a filter that looks at neighbouring frames sees the same frames as ever); steps that change the frame count,
the size or the position are for the whole video.

`faces` is a statement about the content, separate from the choice of model: routing by "anime or real" used to set
both at once, and a cartoon kept at its own size was judged as if its faces were people's (owner sample,
2026-11-21). Two things are refused because they cannot work or are never right: a step that relies on the
source's frame numbers after a step that changes them, and an anime model on faces declared real. Everything
else about order is advice, returned as warnings.
"""

from __future__ import annotations

import re

from . import config, library

FACES = ("drawn", "real")
FIELD_FLAGS = {"interlaced", "telecine", "blended", "chroma_shift"}
KEYS = {"faces", "steps", "models", "model_strict", "scale", "texture", "interpolate", "rebuild_ghosts", "per_kind",
        "rebuild_damaged"}
REBUILD_MAX = 500               # frames one recipe may name: more than that is not "a few damaged frames"
KIND = re.compile(r"^k[1-9]\d?$")
INTERPOLATE = (2, 3, 4)
ANIME_MODELS = {"esrgan-v3", "cugan", "cugan-pro"}


def _step(s) -> dict:
    if isinstance(s, str):
        name, _, preset = s.strip().partition(":")
        return {"block": name.strip(), **({"preset": preset.strip()} if preset.strip() else {})}
    if isinstance(s, dict) and ("block" in s or "name" in s):
        d = dict(s)
        d["block"] = str(d.pop("block", None) or d.pop("name")).strip()
        if "strength" in d:                                  # the word the clean-up filters were first given
            d["preset"] = d.pop("strength")
        extra = set(d) - {"block", "preset", "settings", "kinds"}
        if extra:
            raise ValueError(f"step {d['block']}: unknown key(s) {sorted(extra)}; a step has block, preset, settings, kinds")
        return d
    raise ValueError(f"a step is a block name or {{block, preset, settings}}, got {s!r}")


def by_kind_ok(b) -> bool:
    """May this block's settings differ from one kind of scene to another?"""
    return bool(b.scripted and b.engine == "avisynth" and b.check == "video_guard")


def _kind_ids(value, where: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(k, str) and KIND.match(k) for k in value):
        raise ValueError(f"{where}: kinds is a list of kind ids as the survey names them (k1, k2, ...)")
    return sorted(set(value), key=lambda k: int(k[1:]))


def _per_kind(raw, steps: list[dict], blocks: dict) -> dict:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("per_kind is an object: {kind id: {texture, steps}}")
    by_name = {s["block"]: s for s in steps}
    out: dict = {}
    for kind in sorted(raw, key=str):
        o = raw[kind]
        if not isinstance(kind, str) or not KIND.match(kind):
            raise ValueError(f"per_kind: {kind!r} is not a kind id (the survey names them k1, k2, ...)")
        if not isinstance(o, dict) or set(o) - {"texture", "steps"}:
            raise ValueError(f"per_kind {kind}: an object with texture and/or steps")
        entry: dict = {}
        t = o.get("texture")
        if t is not None:
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 1:
                raise ValueError(f"per_kind {kind}: texture must be between 0 and 1")
            entry["texture"] = float(t)
        changed: dict = {}
        if o.get("steps") is not None and not isinstance(o["steps"], dict):
            raise ValueError(f"per_kind {kind}: steps is an object {{block: preset, settings, or false}}")
        for name, how in (o.get("steps") or {}).items():
            base = by_name.get(name)
            if base is None:
                raise ValueError(f"per_kind {kind}: {name} is not one of the recipe's steps; list it in `steps` "
                                 "(with `kinds` when only some kinds get it): the order of the steps is set there")
            b = blocks[name]
            if not by_kind_ok(b):
                raise ValueError(f"per_kind {kind}: {name} cannot differ by kind (only script filters on AviSynth "
                                 "judged by the video guard can; this step is for the whole video)")
            if how is False:
                changed[name] = False
                continue
            if base.get("kinds") and kind not in base["kinds"]:
                raise ValueError(f"per_kind {kind}: {name} is limited to {', '.join(base['kinds'])} in `steps`")
            if isinstance(how, str):
                how = {"preset": how}
            if not isinstance(how, dict) or set(how) - {"preset", "settings"} or not b.params:
                raise ValueError(f"per_kind {kind}: {name} takes a preset name, {{preset, settings}}, or false"
                                 + ("" if b.params else " (it has no settings: only false)"))
            changed[name] = {**({"preset": how["preset"]} if how.get("preset") else {}),
                             "settings": library.resolve(b, how.get("preset"), how.get("settings"))}
        if changed:
            entry["steps"] = changed
        if entry:
            out[kind] = entry
    return out


def uses_kinds(recipe: dict) -> bool:
    return bool(recipe.get("per_kind")) or any(s.get("kinds") for s in recipe.get("steps") or [])


def kinds_named(recipe: dict) -> list[str]:
    named = set(recipe.get("per_kind") or {})
    for s in recipe.get("steps") or []:
        named |= set(s.get("kinds") or [])
    return sorted(named, key=lambda k: int(k[1:]))


def step_settings(recipe: dict, step: dict, kind: str | None) -> dict | None:
    """The settings one step runs with in a kind of scene; None when it is left out there."""
    how = (((recipe.get("per_kind") or {}).get(kind) or {}).get("steps") or {}).get(step["block"]) if kind else None
    if how is False or (step.get("kinds") and kind not in step["kinds"]):
        return None
    return dict((how or step).get("settings") or {})


def texture_for(recipe: dict, kind: str | None, default: float) -> float:
    t = ((recipe.get("per_kind") or {}).get(kind) or {}).get("texture") if kind else None
    return float(default if t is None else t)


def validate(recipe: dict, blocks: dict | None = None) -> tuple[dict, list[str]]:
    """The recipe in one shape, and advice about it. ValueError for what cannot run."""
    blocks = blocks if blocks is not None else library.load()[0]
    if not isinstance(recipe, dict):
        raise ValueError("a recipe is an object with faces, steps, models, scale, texture")
    extra = set(recipe) - KEYS
    if extra:
        raise ValueError(f"unknown recipe key(s) {sorted(extra)}; a recipe has: {', '.join(sorted(KEYS))}")
    faces = recipe.get("faces")
    if faces not in FACES:
        raise ValueError(f"faces must be one of {FACES}: say whether the faces in this video are drawn or real")
    warnings: list[str] = []
    steps, seen, changed_count, changed_size, filters, moved = [], set(), None, None, [], None
    for raw in recipe.get("steps") or []:
        s = _step(raw)
        b = blocks.get(s["block"])
        if b is None:
            raise ValueError(f"unknown block {s['block']!r}; the library has: {', '.join(sorted(blocks))}")
        if b.kind in ("model", "finish"):
            raise ValueError(f"{b.name} is a {b.kind} block: models go in `models`, texture in `texture`")
        if b.name in seen:
            raise ValueError(f"{b.name} is listed twice")
        seen.add(b.name)
        values = library.resolve(b, s.get("preset"), s.get("settings")) if b.params else {}
        if not b.params and (s.get("preset") or s.get("settings")):
            raise ValueError(f"{b.name} has no presets or settings")
        if "source_frame_numbers" in b.needs and changed_count:
            raise ValueError(f"{b.name} works on the source's own frame numbers and must come before {changed_count}, "
                             "which changes them")
        if "source_positions" in b.needs and moved:
            raise ValueError(f"{b.name} was measured where things are in the source and must come before {moved}, "
                             "which moves the picture")
        if "frame_position" in b.changes:
            moved = moved or b.name
        if "untouched_grid" in b.needs and (changed_size or filters):
            warnings.append(f"{b.name} works best on the untouched pixel grid: put it before "
                            f"{changed_size or filters[0]}")
        if b.kind == "field" and b.name == "fields" and steps:
            warnings.append("fields (interlace, telecine, blends) normally comes first: every later step expects "
                            "clean progressive frames")
        if b.status == "candidate":
            warnings.append(f"{b.name} is a candidate block (not yet shown to help): compare the recipe with and "
                            "without it (try_recipes) before a full run, then record what you saw (note_block)")
        if "frame_count" in b.changes:
            changed_count = changed_count or b.name
        if "frame_size" in b.changes:
            changed_size = changed_size or b.name
        if b.kind == "filter":
            filters.append(b.name)
        only = None
        if s.get("kinds") is not None:
            only = _kind_ids(s["kinds"], f"step {b.name}")
            if not by_kind_ok(b):
                raise ValueError(f"{b.name} cannot be limited to kinds of scene (only script filters on AviSynth "
                                 "judged by the video guard can; this step is for the whole video)")
        steps.append({"block": b.name, **({"preset": s["preset"]} if s.get("preset") else {}),
                      **({"settings": values} if values else {}), **({"kinds": only} if only else {})})
    scale = recipe.get("scale")
    if scale is not None:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 1 <= scale <= 4:
            raise ValueError("scale must be a number from 1 (keep the size) to 4, or null for the tool's default")
        scale = int(scale) if float(scale).is_integer() else float(scale)
    models = []
    for m in recipe.get("models") or []:
        b = blocks.get(str(m))
        if b is None or b.kind != "model":
            raise ValueError(f"unknown model {m!r}; the library has: "
                             f"{', '.join(sorted(n for n, x in blocks.items() if x.kind == 'model'))}")
        if "drawn_faces" in b.needs and faces != "drawn":
            raise ValueError(f"{b.name} is an anime model: it reshapes real faces and is only used when the faces "
                             "are declared drawn")
        ok = b.spec.get("scales")
        if ok and scale is not None and scale not in ok:
            raise ValueError(f"{b.name} works at x{', x'.join(map(str, ok))} (1 = enlarged then scaled back), not x{scale:g}")
        if b.name in models:
            raise ValueError(f"model {b.name} is listed twice")
        models.append(b.name)
    texture = recipe.get("texture")
    if texture is not None and (isinstance(texture, bool) or not isinstance(texture, (int, float)) or not 0 <= texture <= 1):
        raise ValueError("texture must be between 0 and 1, or null for the default")
    more = recipe.get("interpolate")
    if more is not None:
        if isinstance(more, bool) or more not in INTERPOLATE:
            raise ValueError(f"interpolate must be one of {INTERPOLATE} (how many times the frame rate), or null")
        warnings.append(f"interpolate x{more}: {more - 1} of every {more} frames of that file will be created, not "
                        "original picture; it is written beside the faithful file, only because the person asked")
    again = recipe.get("rebuild_ghosts")
    if again not in (None, True, False):
        raise ValueError("rebuild_ghosts must be true or false")
    if again:
        warnings.append("rebuild_ghosts: ghost frames that no neighbour can stand in for (fast movement, cuts) are made "
                        "again from their neighbours in a second file; those frames are created, not original "
                        "picture, and the faithful file keeps the ghosts")
    per = _per_kind(recipe.get("per_kind"), steps, blocks)
    remake = recipe.get("rebuild_damaged") or []
    if (not isinstance(remake, (list, tuple)) or len(remake) > REBUILD_MAX
            or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in remake)):
        raise ValueError(f"rebuild_damaged is a list of at most {REBUILD_MAX} frame numbers of the source (from 0)")
    remake = sorted(set(remake))
    if remake:
        warnings.append(f"rebuild_damaged: {len(remake)} frame(s) are made again from their neighbours in a second "
                        "file; those frames are created, not original picture, and the faithful file keeps them "
                        "as they are. Frames named side by side are made as a run from the frames either side "
                        "of it (up to 3 in a row, up to 6 where the scene holds still; other runs are left). Only "
                        "the part of a frame that differs "
                        "from the made one is replaced")
    out = {"faces": faces, "steps": steps, "models": models, "model_strict": bool(recipe.get("model_strict")),
           "scale": scale, "texture": None if texture is None else float(texture),
           "interpolate": None if more is None else int(more), "rebuild_ghosts": bool(again), "per_kind": per,
           "rebuild_damaged": remake}
    if uses_kinds(out):
        warnings.append(f"settings by kind of scene ({', '.join(kinds_named(out))}): the kinds are the ones the survey "
                        "found in this video (scenes.kinds); in another video the same ids are other scenes")
        first = next((s["block"] for s in steps if s.get("kinds") or any(
            s["block"] in (o.get("steps") or {}) for o in per.values())), None)
        if changed_count and first and [s["block"] for s in steps].index(changed_count) < [s["block"] for s in steps].index(first):
            warnings.append(f"{changed_count} changes the frame count before {first}: the scenes are then found by "
                            "their time, and a stretch may begin or end one frame off its cut")
    return out, warnings


def from_decision(d: dict, flags: list[str] | set[str]) -> dict:
    """The recipe the older decision keys describe (content, model, scale, filters, crop_borders, ...), in the
    order the tool used to fix: field work, interlaced stretches, repeats, picture filters, crop."""
    from . import cleanup

    flags = set(flags or [])
    content = d.get("content") or "auto"
    faces = "drawn" if content == "anime" or (content == "auto" and "anime_like" in flags) else "real"
    wanted = cleanup.parse(d.get("filters"))
    steps: list = []
    if flags & FIELD_FLAGS or (d.get("mask_tape_noise") and "head_switching" in flags):
        steps.append({"block": "fields", "settings": {"mask_tape_noise": True}} if d.get("mask_tape_noise") else "fields")
    steps += [{"block": f["name"]} for f in wanted if f["name"] == "deinterlace_scenes"]
    if d.get("remove_repeats"):
        steps.append("remove_repeats")
    steps += [{"block": f["name"], **({"preset": f["strength"]} if f.get("strength") else {})}
              for f in wanted if f["name"] != "deinterlace_scenes"]
    if d.get("crop_borders"):
        steps.append("crop_borders")
    model = d.get("model")
    if faces == "real" and model in ANIME_MODELS:
        model = None                                   # an anime model picked before the content was called real
    return {"faces": faces, "steps": steps, "models": [model] if model in config.VIDEO_MODEL_CHOICES else [],
            "model_strict": bool(d.get("model_strict")), "scale": d.get("scale"), "texture": d.get("texture"),
            "interpolate": None, "rebuild_ghosts": False, "per_kind": {}, "rebuild_damaged": []}


def describe(recipe: dict) -> str:
    def one(s: dict) -> str:
        return (s["block"] + (f" ({s['preset']})" if s.get("preset") else "")
                + (f" [only {', '.join(s['kinds'])}]" if s.get("kinds") else ""))
    parts = [f"faces {recipe['faces']}"]
    if recipe.get("steps"):
        parts.append("steps: " + ", then ".join(one(s) for s in recipe["steps"]))
    if recipe.get("models"):
        parts.append("models: " + ", ".join(recipe["models"]) + (" only" if recipe.get("model_strict") else ", then the usual order"))
    parts.append("own size" if recipe.get("scale") == 1 else (f"x{recipe['scale']:g}" if recipe.get("scale") else "default size"))
    if recipe.get("texture") is not None:
        parts.append(f"texture {recipe['texture']:g}")
    if recipe.get("interpolate"):
        parts.append(f"plus a x{recipe['interpolate']} frame-rate copy (created frames)")
    if recipe.get("rebuild_ghosts"):
        parts.append("ghosts in movement made again in the copy" if recipe.get("interpolate")
                     else "plus a copy with the ghosts in movement made again (created frames)")
    if recipe.get("rebuild_damaged"):
        parts.append(f"{len(recipe['rebuild_damaged'])} damaged frame(s) made again in a second file (created frames)")
    for kind, o in (recipe.get("per_kind") or {}).items():
        bits = [f"texture {o['texture']:g}"] if o.get("texture") is not None else []
        bits += [f"{name} {'off' if how is False else how.get('preset') or 'own settings'}"
                 for name, how in (o.get("steps") or {}).items()]
        parts.append(f"in {kind}: " + ", ".join(bits))
    return "; ".join(parts)
