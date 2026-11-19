"""One automatic entry point over everything VidIMG can do: survey -> decide -> run -> one report.

The photo pipeline, the video pipeline and the series runner each need someone to analyse a file, choose
the content type and the model, and start a job. This module does that for any mix of files and folders:

* survey(paths, out): finds photos, single videos and numbered series (three or more numbered episodes in
  one folder), analyses them (photos: every file; a series: one episode in the middle), and writes
  <out>/plan.json with one item per unit of work and a decision already filled in for each: content, model
  (from pipeline_video.model_guide), scale, texture, and the reasons. It also writes contact sheets: the
  frame sheets of all videos on a few labelled images, so Claude checks every "drawn or real?" call by
  looking at one picture instead of analysing file by file.
* decide(out, {item: {...}}): corrections after looking (content, model, scale, texture, deblur, skip, ...).
  Changing the content re-picks the model from the guide unless a model is given.
* run(out): executes the plan in order, resumable (finished items are skipped; photo batches, long videos and
  series keep their own checkpoints), and writes <out>/index.html: one page with every item, what was
  decided and why, what was delivered, and links to the detailed reports.

Nothing here weakens the rules of the pipelines underneath: every result still has to pass the fidelity
guard, originals are never written to, opt-in steps (deblur, tape mask) stay off unless a decision turns
them on, and a wrong automatic choice costs time, never fidelity (a chosen video model goes first, the
approved order follows).
"""

from __future__ import annotations

import html
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Callable

from . import config

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".mts", ".m2ts", ".3gp", ".webm",
             ".vob"}
SERIES_MIN = 3                 # numbered episodes in one folder that make it a series
SHEETS_PER_PAGE = 6            # video frame sheets per contact sheet
PLAN_VERSION = 1
VIDEO_KEYS = {"content", "model", "model_strict", "scale", "texture", "mask_tape_noise", "preview_only", "skip"}
PHOTO_KEYS = {"deblur", "skip"}
DONE = ("delivered", "kept_original", "partly_delivered", "skipped", "preview_done")


# ----------------------------------------------------------------------------- finding the work
def _is_output(p: Path) -> bool:
    """VidIMG's own results and work folders are never inputs."""
    parts = {x.lower() for x in p.parts}
    return (p.stem.endswith("_restored") or any(x.startswith("vidimg x") or x.startswith("vidimg restored")
                                                for x in parts)
            or bool(parts & {"work", "attempts", "rounds", "sheets", "comparison", "previews", "deblend_trials"}))


def scan(paths: list[Path]) -> dict:
    """{"photos": [...], "videos": [...], "series": [{"folder", "episodes": [[n, path], ...]}], "ignored": n}."""
    from . import series as series_mod

    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.is_file()]
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"not found: {p}")
    seen: set[str] = set()
    photos, videos, ignored = [], [], 0
    for f in files:
        key = str(f.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        ext = f.suffix.lower()
        if _is_output(f) or ext not in IMAGE_EXT | VIDEO_EXT:
            ignored += 1
        elif ext in IMAGE_EXT:
            photos.append(f)
        else:
            videos.append(f)
    by_folder: dict[Path, list[Path]] = {}
    for v in videos:
        by_folder.setdefault(v.parent, []).append(v)
    series, singles = [], []
    for folder, vids in by_folder.items():
        numbered: dict[int, Path] = {}
        for v in vids:
            n = series_mod.episode_number(v.name)
            if n is not None and n not in numbered:
                numbered[n] = v
        if len(numbered) >= SERIES_MIN:
            series.append({"folder": folder, "episodes": sorted(numbered.items())})
            singles += [v for v in vids if v not in numbered.values()]
        else:
            singles += vids
    return {"photos": photos, "videos": singles, "series": series, "ignored": ignored}


# ----------------------------------------------------------------------------- survey
def _video_item(item_id: str, src: Path, out: Path, log) -> dict:
    """Analyse one video and fill in its decisions from the model guide."""
    from . import pipeline_video
    from .probe import video as pv

    prof = pv.analyze(src)
    guide = pipeline_video.model_guide(prof)
    sheet = pipeline_video.frames_sheet(src, out / "survey" / f"{item_id}_frames.jpg")
    pre = pipeline_video.plan_prepass(prof)[1]
    anime = "anime_like" in prof.flags
    it = {"id": item_id, "source": str(src), "profile_flags": prof.flags,
          "size": guide["signals"]["size"], "faces": guide["signals"]["faces"],
          "prepass": pre or None, "tape_advice": pipeline_video.tape_notes(prof) or None,
          "frames_sheet": str(sheet) if sheet else None,
          "guide": {k: guide.get(k) for k in ("suggested", "why", "suggested_if_anime", "suggested_if_real")},
          "est_hours": {m["model"]: m["est_hours"] for m in guide["models"] if m.get("est_hours") is not None},
          # the probe's reading stands until someone looks at the sheet (it often reads old cartoons as real footage)
          "decision": {"content": "anime" if anime else "auto", "model": guide["suggested"], "model_strict": False,
                       "scale": None, "texture": None, "mask_tape_noise": False, "preview_only": False, "skip": False},
          "decided_by": "analysis", "reasons": list(guide["why"]),
          "needs_look": "confirm on the contact sheet: drawn (anime) or real footage", "state": "pending"}
    log(f"   {item_id}: {src.name}: {it['size']}; flags {', '.join(prof.flags) or 'none'}; "
        f"suggested {guide['suggested']}")
    return it


def _contact_sheets(items: list[dict], out: Path) -> list[str]:
    """The videos' frame sheets, labelled with their item ids, a few per image."""
    import cv2
    import numpy as np

    rows = []
    for it in items:
        if not it.get("frames_sheet"):
            continue
        img = cv2.imread(it["frames_sheet"])
        if img is None:
            continue
        img = cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])), interpolation=cv2.INTER_AREA)
        bar = np.full((34, 960, 3), 30, np.uint8)
        label = f"{it['id']}  {Path(it['source']).name}"[:90]
        cv2.putText(bar, label.encode("ascii", "replace").decode(), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        rows.append(np.vstack([bar, img]))
    pages = []
    for k in range(0, len(rows), SHEETS_PER_PAGE):
        p = out / "survey" / f"contact_{k // SHEETS_PER_PAGE + 1}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.vstack(rows[k:k + SHEETS_PER_PAGE]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        pages.append(str(p))
    return pages


def survey(paths: list[Path], out: Path, log: Callable[[str], None] = print) -> dict:
    """Find and analyse everything under `paths`; write and return the plan (see the module text)."""
    from .probe import image as pi

    out = Path(out)
    for p in map(Path, paths):
        folder = p if p.is_dir() else p.parent
        if out.resolve() == folder.resolve():
            raise ValueError(f"output folder {out} is the folder of the originals; choose another")
    found = scan(paths)
    (out / "survey").mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    log(f"found {len(found['photos'])} photos, {len(found['videos'])} videos, {len(found['series'])} series "
        f"({found['ignored']} other files ignored)")
    if found["photos"]:
        counts = {"noisy": 0, "soft_or_blurred": 0, "anime_like": 0, "faces": 0, "large": 0}
        for n, f in enumerate(found["photos"], 1):
            try:
                prof = pi.analyze(f)
            except Exception as e:  # noqa: BLE001  (an unreadable photo is reported by the run, not here)
                log(f"   photo {f.name}: analysis failed ({e})")
                continue
            for k in ("noisy", "soft_or_blurred", "anime_like", "faces"):
                counts[k] += k in prof.flags
            counts["large"] += max(prof.width, prof.height) > 2000
            if n % 25 == 0:
                log(f"   analysed {n}/{len(found['photos'])} photos")
        reasons = ["non-GAN models, most faithful first (BSRNet, RealESRNet, SwinIR); SCUNet first on noisy photos"]
        if counts["soft_or_blurred"]:
            reasons.append(f"{counts['soft_or_blurred']} photo(s) look soft or blurred: deblur stays off unless the "
                           "person asks for blur removal (decision deblur=true)")
        if counts["large"]:
            reasons.append(f"{counts['large']} photo(s) are already large (over 2000 px): no upscale for those")
        items.append({"id": "photos", "type": "photos", "sources": [str(f) for f in found["photos"]],
                      "count": len(found["photos"]), "counts": counts, "decision": {"deblur": False, "skip": False},
                      "decided_by": "analysis", "reasons": reasons, "state": "pending"})
    for k, v in enumerate(found["videos"], 1):
        try:
            items.append({"type": "video", **_video_item(f"v{k}", v, out, log)})
        except Exception as e:  # noqa: BLE001
            items.append({"id": f"v{k}", "type": "video", "source": str(v), "state": "error",
                          "error": f"analysis failed: {type(e).__name__}: {e}", "decision": {"skip": True}})
            log(f"   v{k}: {v.name}: analysis failed ({e})")
    for k, s in enumerate(found["series"], 1):
        eps = s["episodes"]
        sample = eps[len(eps) // 2][1]
        try:
            it = _video_item(f"s{k}", sample, out, log)
        except Exception as e:  # noqa: BLE001
            items.append({"id": f"s{k}", "type": "series", "folder": str(s["folder"]), "state": "error",
                          "error": f"analysis failed: {type(e).__name__}: {e}", "decision": {"skip": True}})
            continue
        it.update(type="series", folder=str(s["folder"]), analysed_episode=it.pop("source"), source=str(s["folder"]),
                  episodes=[n for n, _ in eps], episode_files=[str(f) for _, f in eps])
        it["decision"]["scale"] = 2
        it["decision"].pop("preview_only")
        it["reasons"].append(f"{len(eps)} numbered episodes: one decision for the whole series, analysed on "
                             f"episode {eps[len(eps) // 2][0]:02d}")
        items.append(it)
    plan = {"version": PLAN_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "out": str(out),
            "inputs": [str(p) for p in paths], "ignored_files": found["ignored"], "items": items,
            "contact_sheets": _contact_sheets([i for i in items if i["type"] != "photos"], out)}
    save(plan)
    return plan


# ----------------------------------------------------------------------------- plan file
def plan_path(out: Path) -> Path:
    out = Path(out)
    return out if out.suffix == ".json" else out / "plan.json"


def load(out: Path) -> dict:
    p = plan_path(out)
    if not p.exists():
        raise FileNotFoundError(f"no plan at {p}: run the survey first")
    return json.loads(p.read_text(encoding="utf-8"))


def save(plan: dict) -> None:
    p = Path(plan["out"]) / "plan.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)


def summary(plan: dict) -> dict:
    """What Claude and the person need at a glance (the full detail stays in plan.json)."""
    rows = []
    for it in plan["items"]:
        row = {"id": it["id"], "type": it["type"], "state": it.get("state"), "decision": it.get("decision"),
               "decided_by": it.get("decided_by"), "reasons": it.get("reasons")}
        if it["type"] == "photos":
            row.update(count=it["count"], counts=it["counts"])
        else:
            row.update(source=it.get("source"), size=it.get("size"), faces=it.get("faces"), flags=it.get("profile_flags"),
                       prepass=it.get("prepass"), tape_advice=it.get("tape_advice"), guide=it.get("guide"),
                       est_hours=it.get("est_hours"), needs_look=it.get("needs_look"))
            if it["type"] == "series":
                row.update(episodes=it.get("episodes"))
        for k in ("result", "error", "report", "delivered"):
            if it.get(k) is not None:
                row[k] = it[k]
        rows.append(row)
    states = [it.get("state") for it in plan["items"]]
    return {"plan": str(plan_path(Path(plan["out"]))), "out": plan["out"], "items": rows,
            "contact_sheets": plan.get("contact_sheets") or [],
            "report": plan.get("report"),
            "counts": {s: states.count(s) for s in sorted(set(map(str, states)))},
            "next": ("LOOK at each contact sheet, correct content/model with decide_plan for any item the analysis "
                     "got wrong (drawn vs real is the common one), then run_plan. Tell the person in a few lines "
                     "what you decided and why.") if any(s == "pending" for s in states) else None}


# ----------------------------------------------------------------------------- decide
def decide(out: Path, decisions: dict[str, dict], by: str = "claude") -> dict:
    """Apply corrections: {item_id: {content, model, model_strict, scale, texture, mask_tape_noise, preview_only,
    deblur, skip, reason}}. Unknown items or keys raise; a changed content re-picks the model from the guide."""
    plan = load(out)
    index = {it["id"]: it for it in plan["items"]}
    for item_id, d in decisions.items():
        if item_id not in index:
            raise ValueError(f"no item {item_id!r} in the plan (items: {', '.join(index)})")
        it = index[item_id]
        d = dict(d)
        reason = d.pop("reason", None)
        allowed = PHOTO_KEYS if it["type"] == "photos" else VIDEO_KEYS - ({"preview_only"} if it["type"] == "series" else set())
        bad = set(d) - allowed
        if bad:
            raise ValueError(f"{item_id}: unknown decision(s) {sorted(bad)}; allowed: {sorted(allowed)}")
        if "content" in d and d["content"] not in ("auto", "anime", "real"):
            raise ValueError(f"{item_id}: content must be auto, anime or real")
        if d.get("model") is not None and d["model"] not in config.VIDEO_MODEL_CHOICES:
            raise ValueError(f"{item_id}: model must be one of {config.VIDEO_MODEL_CHOICES}")
        if d.get("texture") is not None and not 0 <= float(d["texture"]) <= 1:
            raise ValueError(f"{item_id}: texture must be between 0 and 1")
        if d.get("scale") is not None and not 1 <= float(d["scale"]) <= 4:
            raise ValueError(f"{item_id}: scale must be between 1 and 4")
        if "content" in d and "model" not in d and it["type"] != "photos":
            pick = (it.get("guide") or {}).get(f"suggested_if_{d['content']}")
            if pick:
                d["model"] = pick
                it.setdefault("reasons", []).append(f"content set to {d['content']}: model {pick} from the guide")
        it["decision"].update(d)
        it["decided_by"] = by
        it.pop("needs_look", None)
        if reason:
            it.setdefault("reasons", []).append(str(reason))
        if it.get("state") in ("error", "kept_original", "skipped"):
            it["state"] = "pending"                     # a new decision is a new try
    save(plan)
    return plan


# ----------------------------------------------------------------------------- run
def _run_photos(it: dict, out: Path, log) -> None:
    from . import batch
    from .report import compare

    files = [Path(f) for f in it["sources"]]
    dst = out / "photos"
    dst.mkdir(parents=True, exist_ok=True)
    policy = replace(config.DEFAULT_POLICY, deblur=bool(it["decision"].get("deblur")))
    records = batch.restore_images(files, dst, policy=policy, log=log)
    sheets = {}
    for d in records:
        f = Path(d["source"])
        shown = d["delivered"] or next((x["output"] for x in d["attempts"] if x.get("output")), None)
        if shown:
            sheets[f.name] = compare.sheet(f, Path(shown), dst / "sheets" / f"{f.stem}.jpg")
    (dst / "records.json").write_text(json.dumps(records, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    page = compare.review_page(records, sheets, dst / "sheets" / "index.html")
    passed = sum(bool(r["passed"]) for r in records)
    skipped = sum(any("already large" in n for n in r.get("notes", [])) and not r["passed"] for r in records)
    it.update(result={"delivered": passed, "total": len(records), "already_large": skipped}, report=str(page),
              state="delivered" if passed == len(records) - skipped and passed else
              ("partly_delivered" if passed else "kept_original"))


def _run_video(it: dict, out: Path, log) -> None:
    from . import pipeline_video
    from .report import compare

    d = it["decision"]
    src = Path(it["source"])
    dst = out / "videos" / f"{it['id']}_{src.stem[:60]}"
    rec = pipeline_video.restore_video(src, dst, preview_only=bool(d.get("preview_only")), scale=d.get("scale"),
                                       content=d.get("content") or "auto", model=d.get("model"),
                                       model_strict=bool(d.get("model_strict")),
                                       mask_tape_noise=bool(d.get("mask_tape_noise")), log=log)
    r = rec.to_dict()
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "record.json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    page = compare.video_page(r, dst / "report.html")
    gates = {k: g.get("status") for k, g in (r.get("final_gates") or {}).items()}
    it.update(result={"chosen": r.get("chosen"), "passed": r.get("passed"), "notes": r.get("notes"),
                      "warnings": [k for k, s in gates.items() if s == "warn"]},
              delivered=r.get("delivered"), report=str(page),
              state="delivered" if r.get("delivered") else ("preview_done" if d.get("preview_only") and r.get("chosen")
                                                            else "kept_original"))


def _run_series(it: dict, out: Path, log) -> None:
    from . import series

    d = it["decision"]
    folder = Path(it["folder"])
    dst = out / "series" / f"{it['id']}_{folder.name[:60]}"
    status = series.restore_series(folder, dst, set(it["episodes"]), scale=d.get("scale") or 2,
                                   content=d.get("content") or "auto", model=d.get("model"),
                                   model_strict=bool(d.get("model_strict")),
                                   mask_tape_noise=bool(d.get("mask_tape_noise")), log=log)
    eps = {k: v for k, v in status.items() if not k.startswith("_")}
    done = sorted(k for k, v in eps.items() if v.get("delivered"))
    reports = sorted(dst.glob("Series Comparison Report*.html"))
    it.update(result={"delivered": len(done), "total": len(it["episodes"]),
                      "not_delivered": sorted(set(eps) - set(done))},
              report=str(reports[-1]) if reports else None,
              state="delivered" if len(done) == len(it["episodes"]) else ("partly_delivered" if done else "kept_original"))


def run(out: Path, log: Callable[[str], None] = print) -> dict:
    """Execute the plan at `out`; finished items are skipped, so a stopped run continues where it was."""
    plan = load(out)
    out = Path(plan["out"])
    runners = {"photos": _run_photos, "video": _run_video, "series": _run_series}
    todo = [it for it in plan["items"] if it.get("state") not in DONE]
    log(f"plan: {len(plan['items'])} items, {len(todo)} to do")
    for n, it in enumerate(plan["items"], 1):
        if it.get("state") in DONE:
            continue
        if it["decision"].get("skip"):
            it["state"] = "skipped"
            save(plan)
            continue
        name = it["id"] + " " + (Path(it["source"]).name if it.get("source") else f"({it.get('count')} photos)")
        log(f"[{n}/{len(plan['items'])}] {name}: {json.dumps(it['decision'], default=str)}")
        t0 = time.time()
        texture_default, it["state"] = config.TEXTURE_STRENGTH, "running"
        it.pop("error", None)
        save(plan)
        try:
            if it["decision"].get("texture") is not None:
                config.TEXTURE_STRENGTH = float(it["decision"]["texture"])
            runners[it["type"]](it, out, lambda m, i=it["id"]: log(f"{i}: {str(m).strip()}"))
        except Exception as e:  # noqa: BLE001  (one bad item must not stop the rest)
            it.update(state="error", error=f"{type(e).__name__}: {e}")
            log(f"{it['id']}: ERROR {e}\n{traceback.format_exc()}")
        finally:
            config.TEXTURE_STRENGTH = texture_default
        it["minutes"] = round((time.time() - t0) / 60, 1)
        save(plan)
        log(f"{it['id']}: {it['state']} in {it['minutes']} min")
    plan["report"] = str(write_report(plan))
    plan["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(plan)
    log(f"FINISHED: report {plan['report']}")
    return plan


# ----------------------------------------------------------------------------- one report
_STATE_TEXT = {"delivered": "Delivered", "partly_delivered": "Partly delivered", "kept_original": "Original kept",
               "skipped": "Skipped", "error": "Error", "pending": "Not run yet", "running": "Stopped mid-run",
               "preview_done": "Preview only"}


def write_report(plan: dict) -> Path:
    """<out>/index.html: every item, the decision and its reasons, the outcome, links to the detailed reports."""
    out = Path(plan["out"])
    e = html.escape

    def rel(p: str | None) -> str:
        if not p:
            return ""
        try:
            return Path(p).resolve().relative_to(out.resolve()).as_posix()
        except ValueError:
            return Path(p).as_uri()

    rows = []
    for it in plan["items"]:
        d = it.get("decision") or {}
        if it["type"] == "photos":
            what = f"{it['count']} photos"
            dec = "deblur on" if d.get("deblur") else "standard (deblur off)"
            r = it.get("result") or {}
            res = f"{r.get('delivered', 0)} of {r.get('total', it['count'])} delivered" + (
                f"; {r['already_large']} already large" if r.get("already_large") else "")
        else:
            what = Path(it.get("source") or "").name + (f" ({len(it.get('episodes', []))} episodes)"
                                                         if it["type"] == "series" else "")
            dec = ", ".join(x for x in (
                f"content {d.get('content')}", f"model {d.get('model')}" if d.get("model") else None,
                f"x{d['scale']:g}" if d.get("scale") else "automatic size",
                f"texture {d['texture']:g}" if d.get("texture") is not None else None,
                "tape noise masked" if d.get("mask_tape_noise") else None) if x)
            r = it.get("result") or {}
            if it["type"] == "series":
                res = f"{r.get('delivered', 0)} of {r.get('total', 0)} episodes delivered"
            else:
                res = (f"model used: {r.get('chosen')}" if r.get("chosen") else "") + (
                    f"; warnings: {', '.join(r['warnings'])}" if r.get("warnings") else "")
        state = it.get("state", "pending")
        why = "".join(f"<li>{e(str(x))}</li>" for x in (it.get("reasons") or []))
        notes = "".join(f"<li>{e(str(x))}</li>" for x in ((it.get("result") or {}).get("notes") or []))
        tape = "".join(f"<li>{e(str(x))}</li>" for x in (it.get("tape_advice") or []))
        link = f'<a href="{e(rel(it.get("report")))}">detailed report</a>' if it.get("report") else ""
        rows.append(f"""<section class="{e(state)}"><h2>{e(it['id'])} · {e(what)}</h2>
<p class="st">{e(_STATE_TEXT.get(state, state))}{' · ' + e(res) if res else ''}{' · ' + str(it['minutes']) + ' min' if it.get('minutes') else ''} {link}</p>
{f'<p class="err">{e(it["error"])}</p>' if it.get('error') else ''}
<p><b>Decision</b> ({e(str(it.get('decided_by', 'analysis')))}): {e(dec)}</p>
{f'<p>Classical pre-pass planned: {e(it["prepass"])}</p>' if it.get('prepass') else ''}
{f'<ul class="why">{why}</ul>' if why else ''}{f'<p><b>What happened</b></p><ul>{notes}</ul>' if notes else ''}
{f'<p><b>Tape defects (cannot be fixed in software)</b></p><ul>{tape}</ul>' if tape else ''}</section>""")
    states = [it.get("state", "pending") for it in plan["items"]]
    ok = sum(s in ("delivered", "partly_delivered") for s in states)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VidIMG restoration run</title><style>
:root{{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--fg:#1c1f24;--mut:#5c6470;--ok:#1a7f4b;--bad:#b3261e;--line:#dfe3e8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14161a;--card:#1d2026;--fg:#e8eaee;--mut:#9aa3af;--ok:#5fd39a;--bad:#ff8a80;--line:#2c313a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:24px 16px}} h1{{font-size:1.4rem;margin:0 0 4px}} h2{{font-size:1.02rem;margin:0 0 4px;word-break:break-word}}
section{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:14px 16px;margin:12px 0}}
section.delivered{{border-left-color:var(--ok)}} section.error,section.kept_original{{border-left-color:var(--bad)}}
.st{{font-weight:600;margin:0 0 6px}} .err{{color:var(--bad)}} .mut{{color:var(--mut)}} ul{{margin:4px 0 8px;padding-left:20px}} .why{{color:var(--mut)}}
a{{color:inherit}}</style></head><body><main>
<h1>VidIMG restoration run</h1>
<p class="mut">{e(plan.get('created', ''))} · {len(plan['items'])} item(s) · {ok} delivered or partly delivered ·
every delivered file passed the fidelity guard; anything else kept its original. Originals were not modified.</p>
{''.join(rows)}</main></body></html>"""
    p = out / "index.html"
    p.write_text(page, encoding="utf-8")
    return p
