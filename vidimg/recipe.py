"""A recipe: what one video gets, composed from the block library (library.py) by whoever supervises the
restoration. There are no paths in the pipeline; it runs the recipe.

    {"faces": "drawn" | "real",              how the guard judges faces: advisory on drawings, strict on people
     "steps": ["fields", {"block": "deblock", "preset": "light"}, "crop_borders"],
                                              the steps before the model, in the order they run
     "models": ["esrgan-v3", "cugan"],        tried in this order on the previews; the first the guard passes is used
     "model_strict": false,                   true: no fallback beyond the listed models
     "scale": 1 | 2 | ... | null,             1 = keep the size; null = the tool's default for the source's size
     "texture": 0..1 | null}                  source texture put back after the model (null = default)

`faces` is a statement about the content, separate from the choice of model: routing by "anime or real" used to set
both at once, and a cartoon kept at its own size was judged as if its faces were people's (owner sample,
2026-11-21). Two things are refused because they cannot work or are never right: a step that relies on the
source's frame numbers after a step that changes them, and an anime model on faces declared real. Everything
else about order is advice, returned as warnings.
"""

from __future__ import annotations

from . import config, library

FACES = ("drawn", "real")
FIELD_FLAGS = {"interlaced", "telecine", "blended", "chroma_shift"}
KEYS = {"faces", "steps", "models", "model_strict", "scale", "texture"}
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
        extra = set(d) - {"block", "preset", "settings"}
        if extra:
            raise ValueError(f"step {d['block']}: unknown key(s) {sorted(extra)}; a step has block, preset, settings")
        return d
    raise ValueError(f"a step is a block name or {{block, preset, settings}}, got {s!r}")


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
        steps.append({"block": b.name, **({"preset": s["preset"]} if s.get("preset") else {}),
                      **({"settings": values} if values else {})})
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
    out = {"faces": faces, "steps": steps, "models": models, "model_strict": bool(recipe.get("model_strict")),
           "scale": scale, "texture": None if texture is None else float(texture)}
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
            "model_strict": bool(d.get("model_strict")), "scale": d.get("scale"), "texture": d.get("texture")}


def describe(recipe: dict) -> str:
    def one(s: dict) -> str:
        return s["block"] + (f" ({s['preset']})" if s.get("preset") else "")
    parts = [f"faces {recipe['faces']}"]
    if recipe.get("steps"):
        parts.append("steps: " + ", then ".join(one(s) for s in recipe["steps"]))
    if recipe.get("models"):
        parts.append("models: " + ", ".join(recipe["models"]) + (" only" if recipe.get("model_strict") else ", then the usual order"))
    parts.append("own size" if recipe.get("scale") == 1 else (f"x{recipe['scale']:g}" if recipe.get("scale") else "default size"))
    if recipe.get("texture") is not None:
        parts.append(f"texture {recipe['texture']:g}")
    return "; ".join(parts)
