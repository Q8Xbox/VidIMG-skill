"""The block library: every restoration step is an entry in a list, not a branch in the code.

A block is one JSON file that says what the step does, when to use it, how it runs and how it is judged. The
pipeline has no paths: whoever supervises a restoration (Claude, or a person) reads the library and composes a
recipe for one video (recipe.py). The library grows without code changes:

    vidimg/blocks/*.json                 built in, shipped with the tool
    <output root>/library/blocks/*.json  local: added on this machine while restoring (lab mode, or by hand);
                                         a block that proves itself can be promoted to the built-in set

Two kinds of entry:
  * built-in runners ("engine": "builtin"): steps that need code of their own (field handling, un-blending,
    the models). The entry describes them; "runner" names the function.
  * scripted blocks ("engine": "avisynth" or "ffmpeg"): a script template with settings. Adding one is adding a
    file. They are bounded by what makes a new step safe to try on someone's video:
      - picture filters only: same frames, same size, same rate;
      - the script passes the linter (private plugin set, nothing that sharpens before the model, no file or
        plugin loading), ffmpeg filters come from an allow-list;
      - the check is not theirs to choose: every scripted block is judged by the full video guard, and may add
        a measurement that must improve ("effect"). A block cannot define a weaker check for itself.

The library is meant to grow from two sides. From knowledge: whoever supervises knows what restorers do about
a defect, and writes the block when a video shows one that no block addresses (the survey lists such gaps, from
the defects in defects.json: what each looks like, what measures it if anything, which blocks address it). From
the videos: a block written that way starts as a "candidate"; it becomes "proven" when a side-by-side trial showed
that it helps, and every use is counted in the case library, so each block carries its own track record.

The fidelity guard is code and stays code. Nothing in this folder can change what counts as faithful.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

BUILTIN_DIR = Path(__file__).resolve().parent / "blocks"
KINDS = ("field", "framing", "filter", "model", "finish")
ENGINES = ("builtin", "avisynth", "ffmpeg")
SCRIPTED = ("avisynth", "ffmpeg")
CHANGES = ("frame_count", "frame_rate", "frame_size", "frame_position")
NEEDS = ("source_frame_numbers", "source_positions", "untouched_grid", "drawn_faces")
# How a scripted block's pass is judged. The checks are code (guard/video.py). A local block is always judged by
# the full video guard; only a block shipped with the tool may name another check from this list, and only one
# written for exactly what that block does.
CHECKS = ("video_guard", "moved")
EFFECTS = {"blockiness": "fall"}                # measurements a scripted block may promise to improve
STATUS = ("candidate", "proven")
DEFECTS_FILE = Path(__file__).resolve().parent / "defects.json"
NOTES_FILE = "_notes.json"                      # local: what was learnt about blocks (built-in ones included)
PARAM_TYPES = ("int", "float", "choice")

# what a scripted block may not contain, whatever the linter of the engine says
AVS_FORBIDDEN = re.compile(
    r"\b(LoadPlugin|LoadCPlugin|LoadVirtualDubPlugin|Import|Eval|ImageWriter|ImageSource|WriteFile\w*|"
    r"ConditionalReader|\w*Source|DirectShowSource|Prefetch|AssumeFPS|ChangeFPS|ConvertFPS|SelectOdd|SelectEven|"
    r"Trim|Crop\w*|AddBorders|\w*Resize\w*|LSFmod|LSFplus|CAS|Sharpen|UnsharpMask|aWarpSharp2?)\s*\(", re.I)
FFMPEG_ALLOWED = {
    # picture clean-up that keeps every frame, its size and its timing, and does not sharpen
    "atadenoise", "bilateral", "bm3d", "chromanr", "dctdnoiz", "deband", "deblock", "dedot", "deflicker",
    "fftdnoiz", "fspp", "gradfun", "hqdn3d", "median", "nlmeans", "owdenoise", "pp7", "removegrain", "spp",
    "uspp", "vaguedenoiser", "yaepblur",
}
# Interleave / SelectEvery are allowed: motion-compensated temporal filters (the main class of restoration tools:
# spot removal, temporal medians) weave the compensated neighbours around each frame and pick the middle one back
# out. A script that got the count wrong is caught twice: the render compares the frame count it produced with the
# one expected, and the video guard has its own frames gate.
NAME = re.compile(r"^[a-z][a-z0-9_-]{1,39}$")


def local_dir() -> Path:
    return config.OUTPUT_ROOT / "library" / "blocks"


@dataclass
class Block:
    name: str
    kind: str
    what: str
    when: str
    engine: str = "builtin"
    runner: str | None = None                   # built-in runners
    spec: dict = field(default_factory=dict)    # built-in runners: what the runner is given (a model's names)
    script: list[str] = field(default_factory=list)       # scripted blocks: template lines, {setting} placeholders
    params: dict = field(default_factory=dict)  # setting -> {type, min, max, choices, default}
    presets: dict = field(default_factory=dict)  # name -> {setting: value}
    default_preset: str | None = None
    changes: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    effect: dict | None = None                  # {"metric": "blockiness"}: must improve, or the pass is dropped
    softens: str | None = None                  # what the step costs (every filter costs something)
    origin: str = "built-in"
    tried_on: list[str] = field(default_factory=list)     # where it was tried and what was seen (short notes)
    addresses: list[str] = field(default_factory=list)    # the defects it is for (names from defects.json, or new)
    check: str = "video_guard"                  # which of CHECKS judges its pass (built-in blocks only may differ)
    status: str | None = None                   # candidate: written from knowledge, not yet shown to help; proven

    @property
    def scripted(self) -> bool:
        return self.engine in SCRIPTED

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], {})}


def _fail(name: str, why: str):
    raise ValueError(f"block {name!r}: {why}")


def validate(d: dict, origin: str = "local") -> Block:
    """A block definition as a Block, or ValueError saying what is wrong with it."""
    if not isinstance(d, dict):
        raise ValueError("a block is a JSON object")
    name = str(d.get("name", ""))
    if not NAME.match(name):
        raise ValueError(f"block name {name!r}: lower-case letters, digits, _ and - only, 2-40 characters")
    known = set(Block.__dataclass_fields__) - {"origin"}
    extra = set(d) - known - {"version", "origin"}
    if extra:
        _fail(name, f"unknown field(s) {sorted(extra)}")
    for key in ("kind", "what", "when"):
        if not str(d.get(key, "")).strip():
            _fail(name, f"{key} is missing")
    b = Block(**{k: d[k] for k in known if k in d}, origin=origin)
    if b.status is None:
        b.status = "proven" if origin == "built-in" else "candidate"
    if b.status not in STATUS:
        _fail(name, f"status must be one of {STATUS}")
    if not isinstance(b.addresses, list) or not all(isinstance(x, str) and NAME.match(x) for x in b.addresses):
        _fail(name, "addresses: a list of defect names (lower-case, digits, _ and -)")
    if b.kind not in KINDS:
        _fail(name, f"kind must be one of {KINDS}")
    if b.engine not in ENGINES:
        _fail(name, f"engine must be one of {ENGINES}")
    if set(b.changes) - set(CHANGES) or set(b.needs) - set(NEEDS):
        _fail(name, f"changes come from {CHANGES}, needs from {NEEDS}")
    for p, spec in b.params.items():
        if not isinstance(spec, dict) or spec.get("type") not in PARAM_TYPES or "default" not in spec:
            _fail(name, f"setting {p}: needs a type ({', '.join(PARAM_TYPES)}) and a default")
        if spec["type"] == "choice" and not spec.get("choices"):
            _fail(name, f"setting {p}: a choice needs its choices")
        if spec["type"] != "choice" and not ("min" in spec and "max" in spec):
            _fail(name, f"setting {p}: a number needs min and max")
    for pname, values in b.presets.items():
        if set(values) - set(b.params):
            _fail(name, f"preset {pname} sets unknown setting(s) {sorted(set(values) - set(b.params))}")
    if b.default_preset is not None and b.default_preset not in b.presets:
        _fail(name, f"default_preset {b.default_preset!r} is not one of its presets")
    if b.effect is not None and (b.effect.get("metric") not in EFFECTS):
        _fail(name, f"effect metric must be one of {sorted(EFFECTS)}")
    if not b.scripted:
        if origin != "built-in":
            _fail(name, "only the built-in set may hold code runners; a local block is a script (avisynth or ffmpeg)")
        if not b.runner:
            _fail(name, "a built-in runner needs its runner")
        return b
    if b.check not in CHECKS:
        _fail(name, f"check must be one of {CHECKS}")
    if b.check != "video_guard" and origin != "built-in":
        _fail(name, "a local block cannot choose its check: every added block is judged by the full video guard")
    # ---- scripted: the bounds that make a new step safe to try
    if b.kind != "filter":
        _fail(name, "a scripted block is a picture filter (kind: filter)")
    moved = b.check == "moved"
    if set(b.changes) - ({"frame_position"} if moved else set()):
        _fail(name, "a scripted block keeps every frame, its size and its rate (no 'changes')")
    if moved and (b.changes != ["frame_position"] or "shift" not in b.params):
        _fail(name, "a block judged by the moved check declares changes: [frame_position] and a `shift` setting "
                    "(the furthest it may move the picture, in pixels)")
    if not b.script or not all(isinstance(ln, str) and ln.strip() for ln in b.script):
        _fail(name, "script: one or more non-empty lines")
    used = set(re.findall(r"\{(\w+)\}", "\n".join(b.script)))
    if used - set(b.params):
        _fail(name, f"the script uses setting(s) it does not declare: {sorted(used - set(b.params))}")
    for values in [None, *b.presets.values()]:
        check_script(b, render(b, resolve(b, None, values)))
    return b


def resolve(b: Block, preset: str | None = None, settings: dict | None = None) -> dict:
    """The settings one use of the block runs with: defaults, then the preset, then explicit values, each checked
    against its range."""
    out = {p: spec["default"] for p, spec in b.params.items()}
    preset = preset or b.default_preset
    if preset is not None:
        if preset not in b.presets:
            _fail(b.name, f"no preset {preset!r} (it has: {', '.join(b.presets) or 'none'})")
        out.update(b.presets[preset])
    for p, v in (settings or {}).items():
        if p not in b.params:
            _fail(b.name, f"unknown setting {p!r} (it has: {', '.join(b.params) or 'none'})")
        out[p] = v
    for p, v in out.items():
        spec = b.params[p]
        if spec["type"] == "choice":
            if v not in spec["choices"]:
                _fail(b.name, f"{p} must be one of {spec['choices']}")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _fail(b.name, f"{p} must be a number")
        if not spec["min"] <= v <= spec["max"]:
            _fail(b.name, f"{p} must be between {spec['min']} and {spec['max']}")
        out[p] = int(v) if spec["type"] == "int" else float(v)
    return out


def render(b: Block, values: dict) -> list[str]:
    def one(v):
        return f"{v:g}" if isinstance(v, float) else str(v)
    return [re.sub(r"\{(\w+)\}", lambda m: one(values[m.group(1)]), ln) for ln in b.script]


def check_script(b: Block, lines: list[str]) -> None:
    """The bounds of a scripted block, on the lines it would actually run."""
    text = "\n".join(lines)
    if b.engine == "avisynth":
        m = AVS_FORBIDDEN.search(text)
        if m:
            _fail(b.name, f"{m.group(1)} is not allowed in a block (no loading, no file access, no change of frame "
                          "count, size or rate, no sharpening before the model)")
        if any(ln.strip().lower().startswith(("return", "global ")) for ln in lines):
            _fail(b.name, "no return / global in a block's script")
        return
    for part in text.replace("\n", ",").split(","):
        fname = part.strip().split("=", 1)[0].strip()
        if fname and fname not in FFMPEG_ALLOWED:
            _fail(b.name, f"ffmpeg filter {fname!r} is not on the allow-list ({', '.join(sorted(FFMPEG_ALLOWED))})")
    if re.search(r"[\[\];]|file\s*=|filename\s*=", text):
        _fail(b.name, "a plain filter chain only: no labels, no files")


def _read(folder: Path, origin: str, into: dict, problems: list[str]) -> None:
    if not folder.is_dir():
        return
    for f in sorted(folder.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            b = validate(json.loads(f.read_text(encoding="utf-8")), origin)
        except (ValueError, OSError, TypeError) as e:
            problems.append(f"{f.name}: {e}")
            continue
        if b.name in into:
            problems.append(f"{f.name}: a block named {b.name!r} already exists ({into[b.name].origin}); ignored")
            continue
        into[b.name] = b


def load(local: Path | None = None) -> tuple[dict[str, Block], list[str]]:
    """Every usable block (built-in first; a local file cannot replace a built-in one) and what was skipped."""
    blocks: dict[str, Block] = {}
    problems: list[str] = []
    folder = local if local is not None else local_dir()
    _read(BUILTIN_DIR, "built-in", blocks, problems)
    _read(folder, "local", blocks, problems)
    for name, n in _notes(folder).items():                   # what was learnt since the block was written
        b = blocks.get(name)
        if b is None:
            continue
        b.tried_on = list(b.tried_on) + [x for x in n.get("tried_on", []) if x not in b.tried_on]
        if n.get("status") in STATUS:
            b.status = n["status"]
    return blocks, problems


def _notes(folder: Path) -> dict:
    try:
        return json.loads((folder / NOTES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def note(name: str, text: str, proven: bool | None = None, local: Path | None = None) -> Block:
    """Record what a trial or a run showed about a block ("on a VHS cartoon, light: the rim is gone, lines keep
    their weight"), and whether it is now proven (it helped) or back to candidate (it did not). Kept locally, for
    built-in blocks too."""
    folder = local if local is not None else local_dir()
    blocks, _ = load(folder)
    if name not in blocks:
        raise ValueError(f"no block named {name!r}")
    text = " ".join(str(text).split())[:300]
    if not text:
        raise ValueError("say what was seen")
    notes = _notes(folder)
    entry = notes.setdefault(name, {"tried_on": []})
    if text not in entry["tried_on"]:
        entry["tried_on"].append(text)
    if proven is not None:
        entry["status"] = "proven" if proven else "candidate"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / NOTES_FILE).write_text(json.dumps(notes, indent=1, ensure_ascii=False), encoding="utf-8")
    return load(folder)[0][name]


def defects() -> dict:
    """The defects a restorer looks for: what each looks like and what measures it, if anything."""
    try:
        return json.loads(DEFECTS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def coverage(blocks: dict[str, Block] | None = None) -> list[dict]:
    """Every known defect with the blocks that address it. A defect with none is a gap in the library: the next
    video that shows it is the occasion to write the block."""
    blocks = blocks if blocks is not None else load()[0]
    known = dict(defects())
    for b in blocks.values():                                 # a new block may name a defect nobody listed yet
        for d in b.addresses:
            known.setdefault(d, {"looks_like": f"(named by the block {b.name})", "measured_by": None})
    rows = []
    for name, d in known.items():
        by = sorted(b.name for b in blocks.values() if name in b.addresses)
        rows.append({"defect": name, "looks_like": d.get("looks_like"), "measured_by": d.get("measured_by"),
                     "blocks": by, **({"repairable": False, "why": d["why"]} if d.get("repairable") is False else {}),
                     **({"note": d["note"]} if d.get("note") else {})})
    return rows


def gaps(flags: list[str] | set[str], blocks: dict[str, Block] | None = None) -> list[dict]:
    """Defects this video was MEASURED to have that no block addresses."""
    flags = set(flags or [])
    return [{"defect": r["defect"], "looks_like": r["looks_like"],
             "hint": "no block addresses this yet: if the close-ups confirm it, write one from what you know "
                     "(add_block), compare with and without it (try_recipes), and note what you saw (note_block)"}
            for r in coverage(blocks) if r["measured_by"] in flags and not r["blocks"] and r.get("repairable", True)]


def save_local(d: dict, local: Path | None = None) -> Path:
    """Add a scripted block to the local library (validated first). The name must be new."""
    b = validate(d, "local")
    folder = local if local is not None else local_dir()
    blocks, _ = load(folder)
    if b.name in blocks:
        raise ValueError(f"a block named {b.name!r} already exists ({blocks[b.name].origin})")
    b.status = "candidate"                                   # whatever the definition says: it has shown nothing yet
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{b.name}.json"
    path.write_text(json.dumps({**b.to_dict(), "version": 1}, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def guide(blocks: dict[str, Block] | None = None, record: dict | None = None) -> list[dict]:
    """The library for whoever composes a recipe: what each block does, when to use it, its presets and settings,
    whether it is proven, and (record: cases.track_record()) how it has done on earlier videos."""
    blocks = blocks if blocks is not None else load()[0]
    order = {k: i for i, k in enumerate(KINDS)}
    out = []
    for b in sorted(blocks.values(), key=lambda b: (order[b.kind], b.origin != "built-in", b.name)):
        row = {"block": b.name, "kind": b.kind, "what": b.what, "when": b.when}
        if b.presets:
            row["presets"] = list(b.presets)
        if b.params:
            row["settings"] = {p: ({"choices": s["choices"]} if s["type"] == "choice" else {"min": s["min"], "max": s["max"]})
                               for p, s in b.params.items()}
        for k in ("addresses", "changes", "needs", "softens", "tried_on"):
            if getattr(b, k):
                row[k] = getattr(b, k)
        if b.check != "video_guard":
            row["judged_by"] = {"moved": "a check of its own: each frame is lined up with its source first, then "
                                         "must match it; the movement must fall"}.get(b.check, b.check)
        if b.status == "candidate":
            row["status"] = "candidate: not yet shown to help; compare with and without it before a full run"
        if record and record.get(b.name):
            row["record"] = record[b.name]
        if b.origin != "built-in":
            row["origin"] = b.origin
        out.append(row)
    return out
