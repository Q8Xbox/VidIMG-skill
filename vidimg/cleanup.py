"""The clean-up catalogue: classical filters that whoever supervises the restoration (Claude, or a person) may
compose for one video, before the model sees it.

A restorer writes a script for each source; a fixed chain fits none of them well. The freedom here is bounded on
purpose: only the filters below, only the listed strengths, in an order the tool fixes (field work first, blocks on
the untouched 8x8 grid, halos last), every script linted (no sharpening before the model, private plugin set only)
and every pass judged by its own guard before anything is built on it. A filter is chosen because the close-ups
show its defect, never because a number alone says so.

    deinterlace_scenes   a progressive video with interlaced stretches: only the measured spans are deinterlaced
    deblock              8x8 block edges of a starved MPEG encode, which a model would sharpen into a grid
    dehalo               light halos beside dark lines (edge enhancement baked into an old master)

Recipes arrive as a list: ["deinterlace_scenes", {"name": "deblock", "strength": "light"}, "dehalo:strong"].
"""

from __future__ import annotations

from .adapters import avisynth as avs

STRENGTHS = ("light", "medium", "strong")

CATALOGUE: dict[str, dict] = {
    "deinterlace_scenes": {
        "what": "deinterlaces only the measured interlaced stretches of an otherwise progressive video, at the "
                "same rate; every other frame is passed through untouched",
        "when": "the analysis found interlaced spans AND the comb close-up shows horizontal comb teeth on moving "
                "edges (not thin drawn stripes)",
        "strengths": None,
    },
    "deblock": {
        "what": "smooths the 8x8 block edges of a starved MPEG encode (Deblock_QED), strongest on the block "
                "borders and gentle inside the blocks",
        "when": "the line close-ups show a grid of square blocks in flat colours or around lines; light for faint "
                "blocks, strong only for heavy ones (it softens fine texture too)",
        "strengths": {"light": {"quant1": 18, "quant2": 20}, "medium": {"quant1": 24, "quant2": 26},
                      "strong": {"quant1": 30, "quant2": 34}},
    },
    "dehalo": {
        "what": "pulls back the light halo that old edge enhancement left beside dark lines (DeHalo_alpha); the "
                "lines keep their weight",
        "when": "the line close-ups show a bright rim running along dark lines on flat colour (not a drawn "
                "highlight, not white beside black by design)",
        "strengths": {"light": {"bright": 0.6, "dark": 0.0}, "medium": {"bright": 0.8, "dark": 0.2},
                      "strong": {"bright": 1.0, "dark": 0.4}},
    },
}
ORDER = ("deinterlace_scenes", "deblock", "dehalo")


def parse(filters) -> list[dict]:
    """A recipe's filter list in one shape: [{"name": ..., "strength": ...}], in the tool's order, each filter at
    most once. Raises ValueError on anything outside the catalogue."""
    if filters in (None, "", [], ()):
        return []
    if isinstance(filters, str):
        filters = [x for x in filters.replace(";", ",").split(",") if x.strip()]
    if not isinstance(filters, (list, tuple)):
        raise ValueError("filters must be a list")
    out: dict[str, dict] = {}
    for f in filters:
        if isinstance(f, str):
            name, _, strength = f.strip().partition(":")
            f = {"name": name.strip(), **({"strength": strength.strip()} if strength.strip() else {})}
        if not isinstance(f, dict) or "name" not in f:
            raise ValueError(f"a filter is a name or {{name, strength}}, got {f!r}")
        name = str(f["name"]).strip().lower()
        if name not in CATALOGUE:
            raise ValueError(f"unknown filter {name!r}; the catalogue has: {', '.join(ORDER)}")
        extra = set(f) - {"name", "strength"}
        if extra:
            raise ValueError(f"{name}: unknown setting(s) {sorted(extra)}; only strength can be set")
        if name in out:
            raise ValueError(f"{name} is listed twice")
        levels = CATALOGUE[name]["strengths"]
        strength = f.get("strength")
        if levels is None:
            if strength not in (None, ""):
                raise ValueError(f"{name} has no strength")
            out[name] = {"name": name}
            continue
        strength = str(strength or "medium").strip().lower()
        if strength not in levels:
            raise ValueError(f"{name}: strength must be one of {', '.join(STRENGTHS)}")
        out[name] = {"name": name, "strength": strength}
    return [out[n] for n in ORDER if n in out]


def describe(filters: list[dict]) -> str:
    return ", ".join(f["name"] + (f" ({f['strength']})" if f.get("strength") else "") for f in filters)


def picture_stages(filters: list[dict]) -> list:
    """The stages that work on the picture of every frame (one pass, in the tool's order)."""
    stages = []
    for f in filters:
        if f["name"] == "deblock":
            stages.append(avs.Deblock(**CATALOGUE["deblock"]["strengths"][f["strength"]]))
        elif f["name"] == "dehalo":
            stages.append(avs.Dehalo(**CATALOGUE["dehalo"]["strengths"][f["strength"]]))
    return stages


def wants_spans(filters: list[dict]) -> bool:
    return any(f["name"] == "deinterlace_scenes" for f in filters)


def guide() -> list[dict]:
    """The catalogue for whoever decides: name, what it does, when to use it, the strengths."""
    return [{"name": n, "what": CATALOGUE[n]["what"], "when": CATALOGUE[n]["when"],
             "strengths": list(CATALOGUE[n]["strengths"]) if CATALOGUE[n]["strengths"] else None} for n in ORDER]
