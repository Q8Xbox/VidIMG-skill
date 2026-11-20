"""Image restoration pipeline (free tools only): analyse -> denoise if noisy -> plan -> restore -> guard.

Open non-GAN models (BSRNet, RealESRNet, SwinIR) on PyTorch CUDA; Video2X anime models for cartoons;
SCUNet for noisy photos. Candidates are tried from the planned choice
towards more conservative ones; the first output the guard accepts is delivered. If none passes,
nothing is delivered as "restored": the record says why and lists the attempts, and the original
stays the deliverable (the person can still look at the rejected attempts).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config, guard
from .adapters import free_sr, mark_output, video2x
from .probe import image as probe


@dataclass
class Attempt:
    label: str
    params: dict
    output: str | None
    passed: bool
    failed_gates: list[str]
    gates: dict
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    seconds: float | None = None


@dataclass
class RestoreRecord:
    source: str
    profile: dict
    plan: str
    delivered: str | None
    passed: bool
    attempts: list[Attempt] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def choose_scale(p: probe.ImageProfile) -> int:
    side = max(p.width, p.height)
    return 4 if side <= 900 else 2 if side <= 2000 else 1


FREE_PHOTO_MODELS = [("bsrnet", "bsrnet_x4"), ("realesrnet", "realesrnet_x4"), ("swinir", "swinir_real_x4")]


def plan_candidates(p: probe.ImageProfile, policy: config.Policy = config.DEFAULT_POLICY) -> tuple[str, list[tuple[str, dict]]]:
    """Ordered free candidates, most faithful first."""
    scale = choose_scale(p)
    if "anime_like" in p.flags:
        # anime / cartoon (no real face detected): Video2X's anime models
        cands = [("anime cugan", {"engine": "video2x", "processor": "realcugan", "model": "models-se", "scale": scale}),
                 ("anime esrgan", {"engine": "video2x", "processor": "realesrgan", "model": "realesr-animevideov3",
                                   "scale": scale})]
        return f"x{scale} ({p.width}x{p.height}); anime/cartoon: Video2X Real-CUGAN first", cands
    # ground truth on the person's photos (vidimg/bench_free.py): BSRNet true identity 0.811 and +1.58 dB over
    # bicubic, RealESRNet 0.798, SwinIR-real 0.782. Non-GAN (PSNR-trained) models only: GAN models invent
    # texture and scored lower. Accepted by the person on 19 photos (phase 6, 2026-11-16).
    cands = [(label, {"engine": "free", "model": m, "scale": scale}) for label, m in FREE_PHOTO_MODELS]
    why = (f"x{scale} ({p.width}x{p.height}); non-GAN models, BSRNet first (most faithful on your ground truth)"
           + (f"; {p.faces.count} faces, smallest {p.faces.min_height}px" if p.faces.count else ""))
    return why, cands


DEBLUR_MODEL = "nafnet_gopro"


def deblur_step(src: Path, stage_in: Path, prof: probe.ImageProfile, policy: config.Policy, work: Path,
                log=print) -> tuple[Path, Attempt | None, str | None]:
    """Opt-in deblur (policy.deblur) of a photo the probe flagged soft/blurred: NAFNet-GoPro at the
    original size, accepted only when the guard's deblur mode passes it (the result must re-blur into
    the input). Returns (next stage input, the attempt or None, a note or None). Measured on the owner's
    photos: motion blur +4.6 dB / true identity 0.52 -> 0.64; defocus +0.9 dB; sharp photos are harmed,
    which is why it never runs on photos the probe did not flag (PLAN.md 3.6)."""
    if not policy.deblur:
        return stage_in, None, None
    if "soft_or_blurred" not in prof.flags:
        return stage_in, None, f"deblur requested but the photo is not soft (sharpness {prof.sharpness}): skipped"
    if not free_sr.model_available(DEBLUR_MODEL):
        return stage_in, None, "deblur requested but the model is missing: run python -m vidimg.free_install"
    r = free_sr.upscale(stage_in, work / "deblur", model=DEBLUR_MODEL, scale=1.0, suffix="_deblurred")
    if not r.ok:
        return stage_in, None, f"deblur failed: {r.error}"
    rep = guard.verify_image(src, r.output, deblur=True)
    gates = {g.name: {"status": g.status, "value": g.value, "threshold": g.threshold} for g in rep.gates}
    att = Attempt("deblur", {"engine": "free", "model": DEBLUR_MODEL, "scale": 1}, str(r.output), rep.passed,
                  rep.failed(), gates, seconds=r.details.get("seconds") if r.details else None, warnings=rep.warnings())
    log(f"   {'deblur':<12} {'PASS' if rep.passed else 'FAIL ' + ','.join(rep.failed())}")
    if not rep.passed:
        return stage_in, att, "deblur rejected by the guard: continuing without it"
    return Path(r.output), att, f"NAFNet-GoPro deblur first (sharpness {prof.sharpness})"


def restore_image(src: Path, out_dir: Path, *, policy: config.Policy = config.DEFAULT_POLICY,
                  max_attempts: int | None = None, log=print) -> RestoreRecord:
    src = Path(src)
    out_dir = Path(out_dir)
    work = out_dir / "attempts" / src.stem
    work.mkdir(parents=True, exist_ok=True)
    mark_output(out_dir)
    prof = probe.analyze(src)
    rec = RestoreRecord(str(src), prof.to_dict(), "", None, False)
    stage_in = src
    if "noisy" in prof.flags:
        r = free_sr.denoise(src, work / "denoise")
        if r.ok:
            stage_in = r.output
            rec.notes.append(f"SCUNet denoise first (noise sigma {prof.noise_sigma})")
        else:
            rec.notes.append(f"denoise skipped: {r.error}")
    stage_in, datt, dnote = deblur_step(src, stage_in, prof, policy, work, log)
    deblurred = bool(datt and datt.passed)
    if datt:
        rec.attempts.append(datt)
    if dnote:
        rec.notes.append(dnote)
    rec.plan, cands = plan_candidates(prof, policy)
    if choose_scale(prof) == 1:
        if deblurred:                                   # large photo: the deblurred image is the result
            final = out_dir / f"{src.stem}_restored.png"
            shutil.copyfile(stage_in, final)
            rec.delivered, rec.passed = str(final), True
            rec.notes.append("already large: deblur only, no upscale")
        else:
            rec.notes.append("already large: no upscale planned; nothing to do in this slice")
        (work / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
        return rec
    limit = max_attempts or policy.max_refine_rounds + 2
    fallback: Attempt | None = None        # first output that passed but carried warnings
    chosen: Attempt | None = None
    for label, params in cands[:limit]:
        if params.get("engine") == "video2x":
            r = video2x.run(stage_in, work / f"{stage_in.stem}_{label.replace(' ', '_')}.png", params["processor"],
                            content="anime", scale=params["scale"], model=params["model"], codec="png", extra={})
        else:
            r = free_sr.upscale(stage_in, work, model=params["model"], scale=params["scale"],
                                suffix=f"_{label.replace('+', '_')}")
        if not r.ok:
            rec.attempts.append(Attempt(label, params, None, False, [], {}, r.error))
            log(f"   {label:<12} engine error: {r.error}")
            continue
        rep = guard.verify_image(src, r.output, deblur=deblurred)
        gates = {g.name: {"status": g.status, "value": g.value, "threshold": g.threshold} for g in rep.gates}
        att = Attempt(label, params, str(r.output), rep.passed, rep.failed(), gates,
                      seconds=r.run.seconds if r.run else None, warnings=rep.warnings())
        rec.attempts.append(att)
        idv = gates.get("identity", {}).get("value")
        state = ("PASS" + (" (warn " + ",".join(att.warnings) + ")" if att.warnings else "")) if rep.passed \
            else "FAIL " + ",".join(rep.failed())
        log(f"   {label:<12} {state}  identity={idv}")
        if rep.passed and not att.warnings:
            chosen = att
            break
        if rep.passed and fallback is None:
            fallback = att
    chosen = chosen or fallback
    if not chosen and deblurred and datt and not datt.warnings:
        chosen = datt                                   # no upscale passed: the guarded deblur alone
        rec.notes.append("no upscale passed the guard: delivering the deblurred photo at its original size")
    if chosen:
        final = out_dir / f"{src.stem}_restored.png"
        shutil.copyfile(chosen.output, final)
        rec.delivered, rec.passed = str(final), True
        if chosen.warnings:
            rec.notes.append(f"delivered with warnings {chosen.warnings}: no candidate passed cleanly")
    if not rec.passed:
        rec.notes.append("no candidate passed the guard: the original stays the deliverable")
    (work / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
    return rec
