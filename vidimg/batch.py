"""Photo batches (phase 5): grouped by model, checkpointed per photo, resumable.

Same decisions as pipeline.restore_image (plan, candidate order, guard, clean pass first, otherwise the
first pass with warnings, otherwise the original stays), organised for throughput:

* rounds: round r runs every unfinished photo's r-th candidate. Photos whose candidate has identical
  settings go into ONE model call (the model is loaded once per call: about 10 s saved per photo);
* checkpoint: each photo's state (profile, attempts, outcome) is written to <out>/items/<key>.json after
  every step, so a stopped or crashed batch resumes where it was: finished photos are skipped and an
  unfinished one continues from its next candidate;
* outputs and the review page are the same as `python -m vidimg restore`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from . import config, guard, proc
from .adapters import free_sr, mark_output
from .pipeline import Attempt, RestoreRecord, carry_color_profile, choose_scale, deblur_step, plan_candidates
from .probe import image as probe

STATE_VERSION = 1


def _key(src: Path) -> str:
    """Stable per-photo id: stem + short hash of the full path (two folders may hold the same name)."""
    return f"{src.stem[:40]}-{hashlib.sha1(str(src.resolve()).lower().encode()).hexdigest()[:8]}"


class Item:
    def __init__(self, src: Path, out: Path):
        self.src, self.key = src, _key(src)
        self.path = out / "items" / f"{self.key}.json"
        self.work = out / "attempts" / self.key
        st = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.state = st if st.get("version") == STATE_VERSION and st.get("source") == str(src) else \
            {"version": STATE_VERSION, "source": str(src), "status": "new", "attempts": [], "notes": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def done(self) -> bool:
        return self.state["status"] in ("delivered", "kept_original", "skipped")

    def record(self) -> RestoreRecord:
        s = self.state
        rec = RestoreRecord(s["source"], s.get("profile", {}), s.get("plan", ""), s.get("delivered"),
                            s["status"] == "delivered")
        rec.attempts = ([Attempt(**s["deblur_attempt"])] if s.get("deblur_attempt") else []) +             [Attempt(**a) for a in s["attempts"]]
        rec.notes = list(s["notes"])
        return rec


def _prepare(it: Item, policy: config.Policy, log) -> None:
    """Profile, optional denoise, plan. Idempotent: a resumed item keeps what it already has."""
    s = it.state
    if s["status"] != "new":
        return
    it.work.mkdir(parents=True, exist_ok=True)
    prof = probe.analyze(it.src)
    s["profile"] = prof.to_dict()
    s["stage_in"] = str(it.src)
    if "noisy" in prof.flags:
        r = free_sr.denoise(it.src, it.work / "denoise")
        if r.ok:
            s["stage_in"] = str(r.output)
            s["notes"].append(f"SCUNet denoise first (noise sigma {prof.noise_sigma})")
        else:
            s["notes"].append(f"denoise skipped: {r.error}")
    # opt-in deblur; kept out of "attempts" (their count indexes the next upscale candidate)
    stage, datt, dnote = deblur_step(it.src, Path(s["stage_in"]), prof, policy, it.work, log)
    s["stage_in"], s["deblurred"] = str(stage), bool(datt and datt.passed)
    if datt:
        s["deblur_attempt"] = asdict(datt)
    if dnote:
        s["notes"].append(dnote)
    if choose_scale(prof) == 1:
        if s["deblurred"]:                              # large photo: the guarded deblur is the result
            s.update(plan="deblur only (already large)", engine="free", candidates=[], status="judged")
            s["notes"].append("already large: deblur only, no upscale")
            return
        s["status"] = "skipped"
        s["notes"].append("already large: no upscale planned")
        return
    s["plan"], cands = plan_candidates(prof, policy)
    s["engine"] = "free"
    s["candidates"] = [[label, params] for label, params in cands[:policy.max_refine_rounds + 2]]
    s["status"] = "planned"


def _finish(it: Item, out: Path, taken: set[str]) -> None:
    s = it.state
    atts = [Attempt(**a) for a in s["attempts"]]
    clean = next((a for a in atts if a.passed and not a.warnings), None)
    chosen = clean or next((a for a in atts if a.passed), None)
    if not chosen and s.get("deblurred") and s.get("deblur_attempt"):
        d = Attempt(**s["deblur_attempt"])
        if d.passed and not d.warnings:                 # no upscale passed: the guarded deblur alone
            chosen = d
            s["notes"].append("no upscale passed the guard: delivering the deblurred photo at its original size")
    if chosen:
        name = f"{it.src.stem}_restored.png"
        if name.lower() in taken:                       # same file name from two folders
            name = f"{it.key}_restored.png"
        taken.add(name.lower())
        final = out / name
        shutil.copyfile(chosen.output, final)
        s.update(status="delivered", delivered=str(final))
        note = carry_color_profile(it.src, final)
        if note:
            s["notes"].append(note)
        if chosen.warnings:
            s["notes"].append(f"delivered with warnings {chosen.warnings}: no candidate passed cleanly")
    else:
        s["status"] = "kept_original"
        s["notes"].append("no candidate passed the guard: the original stays the deliverable")


def _judge(it: Item, label: str, params: dict, output: Path | None, error: str | None, seconds: float | None, log) -> bool:
    """Guard one candidate output; returns True when the photo is finished (clean pass)."""
    s = it.state
    if output is None:
        s["attempts"].append(asdict(Attempt(label, params, None, False, [], {}, error)))
        log(f"   {it.src.name:<28} {label:<12} engine error: {error}")
        return False
    rep = guard.verify_image(it.src, output, deblur=bool(s.get("deblurred")))
    gates = {g.name: {"status": g.status, "value": g.value, "threshold": g.threshold} for g in rep.gates}
    att = Attempt(label, params, str(output), rep.passed, rep.failed(), gates, seconds=seconds, warnings=rep.warnings())
    s["attempts"].append(asdict(att))
    idv = gates.get("identity", {}).get("value")
    state = ("PASS" + (" (warn " + ",".join(att.warnings) + ")" if att.warnings else "")) if rep.passed \
        else "FAIL " + ",".join(rep.failed())
    log(f"   {it.src.name:<28} {label:<12} {state}  identity={idv}")
    return rep.passed and not att.warnings


def _run_group(items: list[Item], label: str, params: dict, rnd: int, out: Path, log) -> None:
    """One model call for every photo in the group; a photo missing from the output is retried alone."""
    folder = out / "rounds" / f"r{rnd}_{label.replace('+', '_')}_{hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]}"
    suffix = f"_{label.replace('+', '_')}"
    inputs = [Path(it.state["stage_in"]) for it in items]
    # one folder per model call: outputs are named by input stem, and photos with the same stem run in separate
    # calls of the same round; in a shared folder the later call overwrote the earlier photo's picture
    folder = folder / ("call-" + hashlib.sha1("|".join(sorted(str(i) for i in inputs)).encode()).hexdigest()[:8])
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "call.done"
    if marker.exists():                                      # resume: this exact call already finished
        per = json.loads(marker.read_text(encoding="utf-8")).get("seconds_per_image")
        log(f"   reusing finished model call ({label}, {len(items)} photos)")
    else:                                                    # one model load for the whole group
        import time as _t
        t0 = _t.monotonic()
        free_sr.upscale_images(inputs, folder, model=params["model"], scale=params["scale"], suffix=suffix)
        per = (_t.monotonic() - t0) / len(items)
        marker.write_text(json.dumps({"exit": 0, "seconds_per_image": per}), encoding="utf-8")
    for it, inp in zip(items, inputs):
        produced = folder / f"{inp.stem}{suffix}.png"
        if not (produced.exists() and produced.stat().st_size > 0):
            single = free_sr.upscale(inp, folder / "retry", model=params["model"], scale=params["scale"], suffix=suffix)
            produced = single.output if single.ok else None
            err = None if single.ok else (single.error or "no output from the grouped call or the single retry")
        else:
            err = None
        finished = _judge(it, label, params, produced, err, per, log)
        if finished or len(it.state["attempts"]) >= len(it.state["candidates"]):
            it.state["status"] = "judged"
        it.save()


def restore_images(files: list[Path], out: Path, *, policy: config.Policy = config.DEFAULT_POLICY, log=print) -> list[dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mark_output(out)
    items = [Item(Path(f), out) for f in files]
    resumed = sum(1 for it in items if it.state["status"] not in ("new", "failed"))
    log(f"{len(items)} photos" + (f" ({resumed} resumed from checkpoints)" if resumed else ""))
    for n, it in enumerate(items, 1):
        if it.state["status"] in ("new", "failed"):         # a photo that broke last time is tried again
            log(f"[{n}/{len(items)}] analyse {it.src.name}")
            it.state.update(status="new", attempts=[], notes=[])
            try:
                _prepare(it, policy, log)
            except Exception as e:  # noqa: BLE001  (one unreadable photo must not stop the others)
                it.state.update(status="failed", attempts=[])
                it.state["notes"] = [f"could not be prepared ({type(e).__name__}: {e}): the original stays the deliverable"]
                log(f"   {it.src.name}: skipped, {type(e).__name__}: {e}")
            it.save()
    rnd = 0
    while True:
        pending = [it for it in items if it.state["status"] == "planned"
                   and len(it.state["attempts"]) < len(it.state["candidates"])]
        if not pending:
            break
        groups: dict[str, list[Item]] = defaultdict(list)
        for it in pending:
            label, params = it.state["candidates"][len(it.state["attempts"])]
            groups[json.dumps([label, params], sort_keys=True)].append(it)
        log(f"round {rnd + 1}: {len(pending)} photos in {len(groups)} "
            "model call(s)")
        for key, members in groups.items():
            label, params = json.loads(key)
            # outputs are named by input stem: photos with the same stem go into separate calls
            by_stem: dict[str, list[Item]] = defaultdict(list)
            for it in members:
                by_stem[Path(it.state["stage_in"]).stem.lower()].append(it)
            while any(by_stem.values()):
                batch = [lst.pop(0) for lst in by_stem.values() if lst]
                _run_group(batch, label, params, rnd, out, log)
        rnd += 1
    taken: set[str] = set()
    for it in items:
        if it.state["status"] in ("planned", "judged") and not it.done:
            _finish(it, out, taken)
            it.save()
        elif it.state.get("delivered"):
            taken.add(Path(it.state["delivered"]).name.lower())
    return [it.record().to_dict() for it in items]
