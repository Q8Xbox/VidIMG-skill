"""The case library: what was done to each video and how it turned out, kept so the next decision does not
start from nothing.

After a video is restored its case is written to <output root>/library/cases/: the measurements that describe
the source, the recipe, what the guard said, the review of the result and the person's verdict when there is
one. At the next survey the most similar earlier cases are shown beside the new video ("a source like this got
recipe A and was judged good; recipe B was sent back because the backgrounds smeared"), so whoever composes the
recipe builds on experience instead of on the measurements alone.

A case is local and plain JSON. It holds no frames; the source is named by its file name only.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from . import config

FEATURES = ("height", "fps", "bits_per_pixel", "blockiness", "blend_share", "anime_score", "noise_sigma")


def folder() -> Path:
    return config.OUTPUT_ROOT / "library" / "cases"


def signature(profile: dict) -> dict:
    """What describes a source for comparison: its flags and a few measurements."""
    return {"flags": sorted(profile.get("flags") or []), "width": profile.get("width"), "codec": profile.get("codec"),
            **{k: profile.get(k) for k in FEATURES}}


def _key(source: str, recipe: dict) -> str:
    # one spelling per file: the same path arrives with either kind of slash
    src = hashlib.sha1(str(source).replace("\\", "/").lower().encode("utf-8")).hexdigest()[:10]
    rec = hashlib.sha1(json.dumps(recipe, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
    return f"{src}_{rec}"


def record(source: Path | str, profile: dict, recipe: dict, result: dict, where: Path | None = None) -> Path:
    """Write (or update) the case of one run. `result` is the video record: delivered, chosen, final_gates, notes."""
    where = where if where is not None else folder()
    where.mkdir(parents=True, exist_ok=True)
    path = where / f"{_key(str(source), recipe)}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    gates = result.get("final_gates") or {}
    pre = result.get("prepass") or {}
    kept = {b: bool(i.get("guard_passed")) for i in pre.values() if isinstance(i, dict) for b in i.get("blocks") or []}
    for block, key in (("deinterlace_scenes", "spans"), ("remove_repeats", "derepeat"), ("crop_borders", "crop")):
        if isinstance(pre.get(key), dict):
            kept[block] = bool(pre[key].get("guard_passed"))
    # field handling ran as a field pass, as un-blending (its own guarded pass, the only one for a progressive
    # source that is blended), or both: kept when either was
    ran = [x for x in (pre.get("guard_passed"), (pre.get("deblend") or {}).get("guard_passed")) if x is not None]
    if ran:
        kept["fields"] = any(ran)
    steps = [{"block": s["block"], **({"preset": s["preset"]} if s.get("preset") else {}),
              "kept": kept.get(s["block"])} for s in recipe.get("steps") or []]     # None: it found nothing to do
    case = {"source": Path(str(source)).name, "source_key": _key(str(source), {}).split("_")[0],
            "when": time.strftime("%Y-%m-%d %H:%M"), "signature": signature(profile), "recipe": recipe,
            "outcome": {"delivered": bool(result.get("delivered")), "model": result.get("chosen"), "steps": steps,
                        "failed": sorted(k for k, g in gates.items() if g.get("status") == "fail"),
                        "warned": sorted(k for k, g in gates.items() if g.get("status") == "warn"),
                        "vmaf": (gates.get("vmaf") or {}).get("value"),
                        "notes": [str(n)[:300] for n in (result.get("notes") or [])][:12]},
            **({"timings": result["timings"], "work_done": result.get("work_done") or {}} if result.get("timings") else
               {k: old[k] for k in ("timings", "work_done") if k in old}),
            **{k: old[k] for k in ("review", "verdict") if k in old}}
    path.write_text(json.dumps(case, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def annotate(source: Path | str, recipe: dict, *, review: dict | None = None, verdict: str | None = None,
             where: Path | None = None) -> bool:
    """Add the review of the result (verdict + notes) or the person's own words to a recorded case."""
    path = (where if where is not None else folder()) / f"{_key(str(source), recipe)}.json"
    if not path.exists():
        return False
    case = json.loads(path.read_text(encoding="utf-8"))
    if review is not None:
        case["review"] = {k: review.get(k) for k in ("verdict", "notes", "by")}
    if verdict is not None:
        case["verdict"] = str(verdict)[:500]
    path.write_text(json.dumps(case, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return True


def distance(a: dict, b: dict) -> float:
    """0 = the same kind of source. Flags count most (they name the defects), then size, then the measurements."""
    fa, fb = set(a.get("flags") or []), set(b.get("flags") or [])
    d = 2.0 * (1 - len(fa & fb) / len(fa | fb)) if fa | fb else 0.0
    for k, weight in (("height", 1.0), ("fps", 0.5), ("bits_per_pixel", 0.5), ("blockiness", 0.5)):
        x, y = a.get(k), b.get(k)
        if x and y and x > 0 and y > 0:
            d += weight * min(1.0, abs(math.log(x / y)))
        elif (x is None) != (y is None):
            d += weight * 0.5
    for k in ("blend_share", "anime_score"):
        x, y = a.get(k), b.get(k)
        if x is not None and y is not None:
            d += 0.5 * min(1.0, abs(x - y))
    return round(d, 3)


def similar(profile: dict, n: int = 3, where: Path | None = None, exclude_source: str | None = None) -> list[dict]:
    """The n recorded cases most like this source, nearest first, in the short form a decision needs."""
    where = where if where is not None else folder()
    if not where.is_dir():
        return []
    sig = signature(profile)
    skip = _key(exclude_source, {}).split("_")[0] if exclude_source else None
    found = []
    for f in where.glob("*.json"):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if skip and c.get("source_key") == skip:
            continue                                        # the same file's own earlier runs are listed apart
        found.append((distance(sig, c.get("signature") or {}), c))
    found.sort(key=lambda x: x[0])
    return [brief(c) | {"distance": d} for d, c in found[:n]]


def history(source: Path | str, where: Path | None = None) -> list[dict]:
    """Earlier runs of this very file (other recipes tried on it)."""
    where = where if where is not None else folder()
    key = _key(str(source), {}).split("_")[0]
    out = []
    for f in sorted(where.glob(f"{key}_*.json")) if where.is_dir() else []:
        try:
            out.append(brief(json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return out


def track_record(where: Path | None = None) -> dict:
    """Per block, over every recorded case: how often it was asked for, how often its pass was kept by the guard,
    how often the video was delivered, and how the results it was part of were reviewed."""
    where = where if where is not None else folder()
    out: dict = {}
    for f in where.glob("*.json") if where.is_dir() else []:
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        o = c.get("outcome") or {}
        verdict = (c.get("review") or {}).get("verdict")
        for s in o.get("steps") or []:
            r = out.setdefault(s["block"], {"asked": 0, "kept_by_guard": 0, "dropped_by_guard": 0, "delivered": 0, "reviews": {}})
            r["asked"] += 1
            r["kept_by_guard"] += s.get("kept") is True
            r["dropped_by_guard"] += s.get("kept") is False
            r["delivered"] += bool(o.get("delivered"))
            if verdict:
                r["reviews"][verdict] = r["reviews"].get(verdict, 0) + 1
    return out


def brief(c: dict) -> dict:
    s, o = c.get("signature") or {}, c.get("outcome") or {}
    row = {"source": c.get("source"), "when": c.get("when"),
           "was": f"{s.get('width')}x{s.get('height')} @ {s.get('fps')} fps; " + (", ".join(s.get("flags") or []) or "no flags"),
           "recipe": c.get("recipe"), "delivered": o.get("delivered"), "model": o.get("model")}
    if o.get("failed") or o.get("warned"):
        row["gates"] = {"failed": o.get("failed"), "warned": o.get("warned")}
    for k in ("review", "verdict"):
        if c.get(k):
            row[k] = c[k]
    return row
