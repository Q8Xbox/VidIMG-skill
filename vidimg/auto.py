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

from . import imgfile

import html
import json
import shutil
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Callable

from . import cases, cleanup, config, library, statefile
from . import recipe as recipe_mod
from .adapters import OUTPUT_MARKER, mark_output

from .media import IMAGE_EXT, VIDEO_EXT  # noqa: E402  (one list for every entry point)
SERIES_MIN = 3                 # numbered episodes in one folder that make it a series
SHEETS_PER_PAGE = 6            # video frame sheets per contact sheet
PLAN_VERSION = 1
VIDEO_KEYS = {"content", "model", "model_strict", "scale", "texture", "mask_tape_noise", "preview_only", "skip",
              "remove_repeats", "crop_borders", "filters", "recipe"}
FRAMING_KEYS = {"remove_repeats", "crop_borders"}      # single videos only (a series is decided as one)
TRIAL_KEYS = {"content", "model", "scale", "texture", "remove_repeats", "crop_borders", "filters", "recipe"}
BLOCKS_WORTH_A_LOOK = 1.8      # blockdetect mean from which block edges showed on the owner samples (1.9-3.0;
                               # the clean ones read 1.1-1.3)
TRIAL_RECIPES = (2, 4)         # recipes compared in one trial
REVIEW_VERDICTS = ("good", "acceptable", "redo")
PHOTO_KEYS = {"deblur", "skip"}
DONE = ("delivered", "kept_original", "partly_delivered", "skipped", "preview_done")


# ----------------------------------------------------------------------------- finding the work
OWN_FOLDERS = {"work", "attempts", "rounds", "sheets", "comparison", "previews", "deblend_trials", "survey", "items",
               "trials"}


def _results_folder(d: Path) -> bool:
    """`d` is a folder VidIMG wrote results into: it carries the marker, or (results from before the marker) a
    plan, photo checkpoints or a video's work checkpoints."""
    try:
        return ((d / OUTPUT_MARKER).exists() or (d / "plan.json").exists() or any(d.glob("items/*.json"))
                or any(d.glob("attempts/*/record.json")) or any(d.glob("work/*/profile.json")))
    except OSError:
        return False


def _is_output(p: Path) -> bool:
    """VidIMG's own results and work folders are never inputs. A folder named like one of the tool's work folders
    only counts inside a results folder: a person's own "Work" or "Previews" folder holds their media."""
    if p.stem.endswith("_restored") or any(x.lower().startswith(("vidimg x", "vidimg restored")) for x in p.parts):
        return True
    return any(d.name.lower() in OWN_FOLDERS and _results_folder(d.parent) for d in p.parents)


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
    findings = _findings(prof)
    try:
        from .report import closeups
        ghost = None
        if "ghost_frames" in prof.flags:
            from .probe import ghosts
            ghost = ghosts.example(src)
        close = closeups.sheet(src, out / "survey" / f"{item_id}_closeups.jpg", borders=prof.borders,
                               repeats=prof.repeats, combing=prof.combing, ghost=ghost,
                               title=f"{item_id}  {src.name}")
    except Exception as e:  # noqa: BLE001  (an aid for the decision; the survey goes on without it)
        close = None
        log(f"   {item_id}: close-ups failed ({e})")
    scene_map, damaged = _scene_map(item_id, src, out, log)
    if damaged and (damaged.get("frames") or damaged.get("runs")):
        findings.append({"finding": "damaged_frames",
                         "what": f"{damaged['count']} single frames show something neither neighbour shows "
                                 f"(candidates: {', '.join(map(str, damaged['frames'][:12]))}"
                                 + (", ..." if damaged["count"] > 12 else "") + ")"
                                 + (f", and {len(damaged['runs'])} runs of two to six frames stand out together"
                                    if damaged.get("runs") else "")
                                 + (f"; {len(damaged.get('tape_like') or [])} frames and "
                                    f"{len(damaged.get('tape_like_runs') or [])} runs read like tape damage and are "
                                    "listed and shown first" if damaged.get("tape_like") or damaged.get("tape_like_runs")
                                    else "") + "; a model sharpens real damage",
                         "decision": "recipe: rebuild_damaged: [frame numbers] (created frames, in a second file; only "
                                     "the frames you judged to be damage, and only when the person wants them made again)",
                         "look": "damaged.sheet: tape tearing, dropouts or broken blocks, or a frame drawn that way "
                                 "(a flash, lightning, an impact frame, a blink)?"})
    start = {"content": "anime" if anime else "auto", "model": guide["suggested"]}
    try:
        proposed = recipe_mod.validate(recipe_mod.from_decision(start, prof.flags))[0]
    except ValueError:
        proposed = None
    it = {"id": item_id, "source": str(src), "profile_flags": prof.flags,
          "findings": findings or None, "closeups": str(close) if close else None,
          "proposed_recipe": proposed, "scenes": scene_map, "damaged": damaged,
          "work": {"width": prof.width, "height": prof.height, "frames": prof.frames, "fps": prof.fps},
          "gaps": library.gaps(prof.flags) or None,
          "similar_cases": cases.similar(prof.to_dict(), 3, exclude_source=str(src)) or None,
          "history": cases.history(src) or None,
          "size": guide["signals"]["size"], "faces": guide["signals"]["faces"],
          "prepass": pre or None, "tape_advice": pipeline_video.tape_notes(prof) or None,
          "frames_sheet": str(sheet) if sheet else None,
          "guide": {k: guide.get(k) for k in ("suggested", "why", "suggested_if_anime", "suggested_if_real")},
          "est_hours": {m["model"]: m["est_hours"] for m in guide["models"] if m.get("est_hours") is not None},
          # the probe's reading stands until someone looks at the sheet (it often reads old cartoons as real footage)
          "decision": {"content": "anime" if anime else "auto", "model": guide["suggested"], "model_strict": False,
                       "scale": None, "texture": None, "mask_tape_noise": False, "preview_only": False, "skip": False,
                       "remove_repeats": False, "crop_borders": False, "filters": [], "recipe": None},
          "decided_by": "analysis", "reasons": list(guide["why"]),
          "needs_look": "confirm on the contact sheet: drawn (anime) or real footage"
                        + ("; judge each finding on the close-ups" if findings else "")
                        + ("; look at the kinds of scene on scenes.sheet" if (scene_map or {}).get("sheet") else ""),
          "state": "pending"}
    log(f"   {item_id}: {src.name}: {it['size']}; flags {', '.join(prof.flags) or 'none'}; "
        f"suggested {guide['suggested']}")
    return it


def _scene_map(item_id: str, src: Path, out: Path, log) -> tuple[dict | None, dict | None]:
    """Where the video cuts and what kinds of scene it holds (probe.scenes): the whole map in a file of its own,
    the kinds and a sheet showing each in the plan. The same pass over the frames looks for damaged ones
    (probe.damage). Aids for the decision: the survey goes on without them."""
    try:
        from .probe import damage, scenes
        from .report import closeups
        t0 = time.time()
        finder = damage.Finder()
        found = scenes.scan(src, watch=finder.add)
        if found.error:
            log(f"   {item_id}: no scene map ({found.error})")
            return None, None
        damaged = _damaged(item_id, src, out, finder.result(), log)
        where = out / "survey" / f"{item_id}_scenes.json"
        statefile.write_json(where, found.to_dict())
        sheet = closeups.kinds_sheet(src, out / "survey" / f"{item_id}_scene_kinds.jpg", found.kinds,
                                     title=f"{item_id}  {src.name}: {len(found.scenes)} scenes of {len(found.kinds)} kinds")
        log(f"   {item_id}: {len(found.scenes)} scenes of {len(found.kinds)} kinds ({time.time() - t0:.0f} s)")
        return ({"count": len(found.scenes), "kinds": found.kinds, "map": str(where), "sheet": str(sheet) if sheet else None},
                damaged)
    except Exception as e:  # noqa: BLE001
        log(f"   {item_id}: scene map failed ({e})")
        return None, None


DAMAGED_LISTED = 60             # candidate frame numbers carried in the plan (all of them are in the list file)


def _damaged(item_id: str, src: Path, out: Path, got, log) -> dict | None:
    """Candidate damaged frames for the plan: the lone ones that are not flashes, largest first, and a sheet."""
    try:
        from .probe import damage
        from .report import closeups
        lone = [d for d in got.found if d["lone"]]
        if not lone and not got.runs:
            return None
        damage.mark_tape(got, damage.outliers(src))             # which of them read like tape damage: looked at first
        flashes = [d for d in lone if d["looks"] == "flash"]
        cands = sorted((d for d in lone if d["looks"] != "flash"), key=lambda d: (not d.get("tape"), -d["share"]))
        runs = sorted(got.runs, key=lambda r: (not r.get("tape"), -r["share"]))
        tape = [d["frame"] for d in cands if d.get("tape")]
        tape_runs = [[r["first"], r["last"]] for r in runs if r.get("tape")]
        where = out / "survey" / f"{item_id}_damaged.json"
        statefile.write_json(where, got.to_dict())
        sheet = closeups.damage_sheet(src, out / "survey" / f"{item_id}_damaged.jpg", got.found, runs=got.runs,
                                      title=f"{item_id}  {src.name}: {len(cands)} candidate damaged frames, {len(runs)} runs")
        log(f"   {item_id}: {len(cands)} candidate damaged frames ({len(tape)} read like tape damage), {len(runs)} runs "
            f"of two to six ({len(tape_runs)}), {len(flashes)} one-tone flashes, {got.total - len(lone)} in a flicker")
        return {"count": len(cands), "frames": [d["frame"] for d in cands[:DAMAGED_LISTED]], "flashes": len(flashes),
                "tape_like": tape[:DAMAGED_LISTED], "tape_like_runs": tape_runs[:DAMAGED_LISTED],
                "runs": [[r["first"], r["last"]] for r in runs[:DAMAGED_LISTED]],
                "in_runs": got.total - len(lone), "list": str(where), "sheet": str(sheet) if sheet else None}
    except Exception as e:  # noqa: BLE001
        log(f"   {item_id}: damaged-frame candidates failed ({e})")
        return None


def _findings(prof) -> list[dict]:
    """Defects that are measured but corrected only on a decision, in plain words, with the decision that
    corrects each. Whoever decides looks at the close-ups first: the measurement cannot tell a bar from a dark
    edge, nor padding from a held drawing."""
    out = []
    b, r = prof.borders or {}, prof.repeats or {}
    c = getattr(prof, "combing", None) or {}
    if "partly_interlaced" in prof.flags:
        out.append({"finding": "partly_interlaced",
                    "what": f"{len(c.get('spans') or [])} interlaced stretches ({c.get('span_frames')} of "
                            f"{c.get('frames')} frames) in a progressive video; enlarging keeps the comb teeth",
                    "decision": "filters: deinterlace_scenes",
                    "look": "the comb close-ups: horizontal teeth on moving edges, or thin drawn stripes?"})
    if (getattr(prof, "blockiness", None) or 0) >= BLOCKS_WORTH_A_LOOK:
        out.append({"finding": "block_edges",
                    "what": f"compression block edges (level {prof.blockiness:.1f}; clean sources read about 1.2); "
                            "a model sharpens them into a grid",
                    "decision": "filters: deblock (light, medium or strong)",
                    "look": "the line close-ups: a grid of square blocks in flat colours or along lines?"})
    if "ghost_frames" in prof.flags:
        bl = getattr(prof, "blend", None) or {}
        out.append({"finding": "ghost_frames",
                    "what": f"about {bl.get('share', 0):.0%} of the moving frames are double images: mixes of the "
                            f"picture before and the picture after (a {prof.fps:g} fps copy made with blending); a model "
                            "redraws them with crisp double outlines",
                    "decision": "step: deghost",
                    "look": "the frame strip: does a frame show two drawings at once that its neighbours show singly?"})
    if "black_borders" in prof.flags:
        out.append({"finding": "black_borders",
                    "what": f"black bars: left {b.get('left')}, right {b.get('right')}, top {b.get('top')}, bottom "
                            f"{b.get('bottom')} px; enlarging keeps and sharpens them",
                    "decision": "crop_borders", "look": "the frame with the red outline: is everything outside it black?"})
    if "padded_repeats" in prof.flags:
        out.append({"finding": "padded_repeats",
                    "what": f"every {r.get('cycle')}th frame is a copy of the one before ({r.get('share', 0):.0%} of "
                            f"moving frames): a {r.get('original_fps')} fps original padded to {prof.fps:g} fps, "
                            "which judders",
                    "decision": "remove_repeats", "look": "the strip of consecutive frames: are the frames marked COPY "
                                                          "the same drawing mid-movement?"})
    return out


def _contact_sheets(items: list[dict], out: Path) -> list[str]:
    """The videos' frame sheets, labelled with their item ids, a few per image."""
    import cv2
    import numpy as np

    rows = []
    for it in items:
        if not it.get("frames_sheet"):
            continue
        img = imgfile.imread(it["frames_sheet"])
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
        imgfile.imwrite(str(p), np.vstack(rows[k:k + SHEETS_PER_PAGE]), [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    mark_output(out)
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
    statefile.write_json(Path(plan["out"]) / "plan.json", plan)


def _save_run(plan: dict, current: dict | None = None) -> None:
    """The runner's save. It holds the plan in memory for hours, so decisions made meanwhile (decide_plan) are on
    disk only: they are taken over here instead of being overwritten. The runner itself never changes a decision,
    so any difference comes from a person or Claude.
    - an item not started yet runs with the new decision;
    - the item running now finishes with the old one and is then set back to pending (`current`);
    - a finished item that decide() set back to pending stays pending for the next run."""
    path = Path(plan["out"]) / "plan.json"
    with statefile.locked(path):
        disk = {it.get("id"): it for it in statefile.read_json(path).get("items", [])}
        for it in plan["items"]:
            d = disk.get(it["id"])
            if not d or d.get("decision") == it.get("decision"):
                continue
            it["decision"], it["decided_by"] = d["decision"], d.get("decided_by", it.get("decided_by"))
            it["reasons"] = d.get("reasons", it.get("reasons"))
            if "needs_look" not in d:
                it.pop("needs_look", None)
            if it is current:
                it["redo"] = True
            elif d.get("state") == "pending" and it.get("state") in ("error", "kept_original", "skipped"):
                it["state"] = "pending"
        statefile.write_json(path, plan)


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
                       est_hours=it.get("est_hours"), needs_look=it.get("needs_look"),
                       findings=it.get("findings"), closeups=it.get("closeups"),
                       proposed_recipe=it.get("proposed_recipe"), similar_cases=it.get("similar_cases"),
                       history=it.get("history"), gaps=it.get("gaps"), scenes=it.get("scenes"),
                       damaged=it.get("damaged"))
            if it.get("state") == "pending" and it.get("work"):
                # for the recipe as it stands now (the decided one, else the proposal); for a series, per episode
                from . import estimate as estimate_mod
                now = (it.get("decision") or {}).get("recipe") or it.get("proposed_recipe")
                if now:
                    row["estimate"] = estimate_mod.estimate(it["work"], now)
            if it["type"] == "series":
                row.update(episodes=it.get("episodes"))
        for k in ("result", "error", "report", "delivered", "trials", "review"):
            if it.get(k) is not None:
                row[k] = it[k]
        rows.append(row)
    states = [it.get("state") for it in plan["items"]]
    return {"plan": str(plan_path(Path(plan["out"]))), "out": plan["out"], "items": rows,
            "contact_sheets": plan.get("contact_sheets") or [],
            "report": plan.get("report"),
            **({"blocks": library.guide(record=cases.track_record()),
                # every defect a restorer looks for and the blocks that address it; "blocks": [] is a gap
                "defects": library.coverage()} if any(s == "pending" for s in states) else {}),
            "counts": {s: states.count(s) for s in sorted(set(map(str, states)))},
            "next": _next_step(plan, states)}


def _next_step(plan: dict, states: list) -> str | None:
    if any(s == "pending" for s in states):
        return ("LOOK at each contact sheet AND each video's closeups image, read its similar_cases and history "
                "(what earlier videos like it, and this file itself, got and how that was judged), then compose "
                "its `recipe` with decide_plan from `blocks`: say whether the faces are drawn or real, list the "
                "steps the close-ups call for in the order you want them, name the models, the scale (1 keeps "
                "the size) and the texture. proposed_recipe is the analysis' starting point, not a path to "
                "follow. Each video's `scenes` says what kinds of scene it is made of (by light, "
                "motion and detail, with the share of the video each holds and how often double images occur in "
                "it) and scenes.sheet shows frames of each kind. Compose the recipe for the kinds "
                "that hold most of the video, then look at the others on the sheet: where a kind wants "
                "something else (a dark or flat kind shows smoothing first; a kind full of double images or "
                "blocks may want a stronger filter), say so in the recipe's `per_kind`: {kind id: {texture, "
                "steps: {block: preset | {preset, settings} | false}}}, or limit a step to some kinds with "
                "its `kinds`. One model serves the whole video; only script filters and the texture strength "
                "can differ by kind, and only for a single video (not a series). Leave per_kind out when the "
                "kinds look alike: every difference is something more to check. An item's `damaged` "
                "lists single frames that show something neither neighbour shows, and damaged.sheet shows the "
                "largest between their neighbours: tape tearing and dropouts look like that, and so do frames "
                "drawn on purpose (a flash, lightning, an impact frame, a blink). No model can restore a torn "
                "frame; it can only be made again from its neighbours, which is a created frame. Tell the "
                "person what you found, and ONLY if they want such frames made again name the ones you judged "
                "to be damage in the recipe's `rebuild_damaged`: [frame numbers]; they are then made again in "
                "a second file and the faithful file keeps them as they are. `damaged.tape_like` and tape_like_runs are the candidates that also read "
                "like analog tape damage (a hint that was right for 20 of 21 on a torn cartoon, and wrong for a "
                "few drawn frames on clean ones): look at those first, and still judge every frame you name. "
                "`damaged.runs` are two to six "
                "frames in a row that stand out together (more than three only where the scene holds still): to have such a run made again, name every frame "
                "of it. Check each neighbour on the sheet as well: a frame is made from the frames either "
                "side of it, so when a neighbour is damaged too, name it with the frame (a run), or leave "
                "both. GROW THE LIBRARY: `defects` lists what a restorer looks for and which blocks address "
                "each; an item's `gaps` are defects it was measured to have that no block addresses. When the "
                "close-ups show a defect with no block (measured or not), write the block from what you know "
                "about restoring it (add_block), compare the recipe with and without it (try_recipes), and "
                "record what you saw (note_block). Each item's `estimate` says how long its recipe will take "
                "(minutes, a range, the stages, and whether it rests on runs timed on this machine): tell the "
                "person before a long run. A candidate block is used in a full run only after such a "
                "comparison; a block's `record` is how it did on earlier videos. When "
                "two or more recipes are plausible (a step or not, which strength, which model) and you cannot "
                "tell which will look better, compare them first with try_recipes and look at its sheet. Then "
                "run_plan. Tell the person in a few lines what you decided and why.")
    unseen = [it["id"] for it in plan["items"] if it["type"] == "video" and it.get("delivered") and not it.get("review")]
    if unseen:
        return (f"LOOK at the before/after sheets of {', '.join(unseen)} (result.sheets) and record what you see with "
                "review_results: good, acceptable (say what is not ideal), or redo (then change the decision with "
                "decide_plan and run_plan again). The guard passed these; your review is about how they look. "
                "result.clips are short videos of the source beside the result (and of the places where frames were "
                "made again, slowed down): you cannot play them, so point the person to them; they show what "
                "stills cannot, such as flicker and single torn frames.")
    return None


# ----------------------------------------------------------------------------- decide
def decide(out: Path, decisions: dict[str, dict], by: str = "claude") -> dict:
    """Apply corrections: {item_id: {content, model, model_strict, scale, texture, mask_tape_noise, preview_only,
    deblur, skip, reason}}. Unknown items or keys raise; a changed content re-picks the model from the guide."""
    with statefile.locked(plan_path(out)):                  # against the runner's saves (see _save_run)
        return _decide(out, decisions, by)


def _decide(out: Path, decisions: dict[str, dict], by: str) -> dict:
    plan = load(out)
    index = {it["id"]: it for it in plan["items"]}
    for item_id, d in decisions.items():
        if item_id not in index:
            raise ValueError(f"no item {item_id!r} in the plan (items: {', '.join(index)})")
        it = index[item_id]
        d = dict(d)
        reason = d.pop("reason", None)
        allowed = PHOTO_KEYS if it["type"] == "photos" else VIDEO_KEYS - (
            {"preview_only"} | FRAMING_KEYS if it["type"] == "series" else set())
        bad = set(d) - allowed
        if bad:
            raise ValueError(f"{item_id}: unknown decision(s) {sorted(bad)}; allowed: {sorted(allowed)}")
        d = _typed(item_id, d)
        if "content" in d and d["content"] not in ("auto", "anime", "real"):
            raise ValueError(f"{item_id}: content must be auto, anime or real")
        if d.get("model") is not None and d["model"] not in config.VIDEO_MODEL_CHOICES:
            raise ValueError(f"{item_id}: model must be one of {config.VIDEO_MODEL_CHOICES}")
        if d.get("texture") is not None and not 0 <= float(d["texture"]) <= 1:
            raise ValueError(f"{item_id}: texture must be between 0 and 1")
        if d.get("scale") is not None and not 1 <= float(d["scale"]) <= 4:
            raise ValueError(f"{item_id}: scale must be between 1 and 4")
        if "filters" in d:
            try:
                d["filters"] = cleanup.parse(d["filters"])
            except ValueError as e:
                raise ValueError(f"{item_id}: {e}") from None
        if d.get("recipe") is not None:
            try:
                d["recipe"], advice = recipe_mod.validate(d["recipe"])
            except ValueError as e:
                raise ValueError(f"{item_id}: recipe: {e}") from None
            _check_kinds(item_id, it, d["recipe"])
            it.setdefault("reasons", []).extend(f"advice on the recipe: {a}" for a in advice)
        if "content" in d and "model" not in d and it["type"] != "photos":
            pick = (it.get("guide") or {}).get(f"suggested_if_{d['content']}")
            if pick:
                d["model"] = pick
                it.setdefault("reasons", []).append(f"content set to {d['content']}: model {pick} from the guide")
        it["decision"].update(d)
        if any(k in d for k in ("content", "model", "scale", "texture", "filters", "recipe") + tuple(FRAMING_KEYS)) and it.get("review"):
            it.pop("review")                            # a new recipe: the old look no longer stands
            if it.get("state") == "delivered":
                it["state"] = "pending"
        it["decided_by"] = by
        it.pop("needs_look", None)
        if reason:
            it.setdefault("reasons", []).append(str(reason))
        if it.get("state") in ("error", "kept_original", "skipped"):
            it["state"] = "pending"                     # a new decision is a new try
    save(plan)
    return plan


def _check_kinds(item_id: str, it: dict, recipe: dict) -> None:
    """A recipe may set things by kind of scene only for kinds this video has."""
    named = recipe_mod.kinds_named(recipe)
    if not named:
        return
    if it["type"] == "series":
        raise ValueError(f"{item_id}: recipe: a series is decided as one and each episode has its own kinds of "
                         "scene; settings by kind are for single videos")
    known = [k["id"] for k in (it.get("scenes") or {}).get("kinds") or []]
    if not known:
        raise ValueError(f"{item_id}: recipe: this video has no scene map, so nothing can be set by kind of scene")
    missing = [k for k in named if k not in known]
    if missing:
        raise ValueError(f"{item_id}: recipe: no kind {', '.join(missing)} in this video (its kinds: {', '.join(known)})")


def _preview_kinds(it: dict, rec) -> list:
    """The kind of scene each preview of a trial falls in (its middle frame), from the survey's map."""
    from . import pipeline_video
    from .probe import scenes
    from .probe import video as pv
    try:
        found = statefile.read_json(Path(it["scenes"]["map"]))
        prof = pv.VideoProfile(**rec.profile)
        half = pipeline_video.PREVIEW_SECONDS / 2
        return [scenes.kind_at(found["scenes"], min(found["frames"] - 1, round((s + half) * (prof.fps or 25.0))))
                for s in pipeline_video._preview_starts(prof)]
    except Exception:  # noqa: BLE001  (the trial then shows the recipe's own texture strength everywhere)
        return []


NUMBER_KEYS = {"scale", "texture"}
FLAG_KEYS = {"model_strict", "mask_tape_noise", "preview_only", "skip", "deblur", "remove_repeats", "crop_borders"}


def _typed(item_id: str, d: dict) -> dict:
    """Decisions in the types the pipelines and the report expect. A value that arrives as text ("2", "true")
    passed the range check and was stored as text; the run then worked and the report crashed at the very end."""
    out = dict(d)
    for k in NUMBER_KEYS & set(out):
        if out[k] is None:
            continue
        try:
            if isinstance(out[k], bool):
                raise ValueError
            n = float(out[k])
        except (TypeError, ValueError):
            raise ValueError(f"{item_id}: {k} must be a number, got {out[k]!r}") from None
        out[k] = int(n) if k == "scale" and n.is_integer() else n
    for k in FLAG_KEYS & set(out):
        v = out[k].strip().lower() if isinstance(out[k], str) else out[k]
        if v in (True, "true", "yes", 1):
            out[k] = True
        elif v in (False, "false", "no", 0):
            out[k] = False
        else:
            raise ValueError(f"{item_id}: {k} must be true or false, got {out[k]!r}")
    return out


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
            # keyed by the whole path, named by the photo's own id: two photos called IMG_1 shared one sheet
            sheets[str(f)] = compare.sheet(f, Path(shown), dst / "sheets" / f"{batch._key(f)}.jpg")
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
                                       mask_tape_noise=bool(d.get("mask_tape_noise")),
                                       remove_repeats=bool(d.get("remove_repeats")),
                                       crop_borders=bool(d.get("crop_borders")), filters=d.get("filters"),
                                       recipe=d.get("recipe"), log=log)
    r = rec.to_dict()
    dst.mkdir(parents=True, exist_ok=True)
    if r.get("recipe") and not d.get("preview_only"):
        try:                                                # the case library: this run, for the next decision
            cases.record(src, r.get("profile") or {}, r["recipe"], r)
            it["recipe_used"] = r["recipe"]
        except OSError as e:
            log(f"   case not recorded ({e})")
    (dst / "record.json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    page = compare.video_page(r, dst / "report.html")
    gates = {k: g.get("status") for k, g in (r.get("final_gates") or {}).items()}
    it.update(result={"chosen": r.get("chosen"), "passed": r.get("passed"), "notes": r.get("notes"),
                      "warnings": [k for k, s in gates.items() if s == "warn"], "sheets": r.get("sheets") or [],
                      "clips": r.get("clips") or [],
                      **({"interpolated": r["interpolated"]} if r.get("interpolated") else {})},
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
                                   mask_tape_noise=bool(d.get("mask_tape_noise")), filters=d.get("filters"),
                                   recipe=d.get("recipe"), log=log)
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
        _save_run(plan)                                     # take over decisions made since the last save
        if it["decision"].get("skip"):
            it["state"] = "skipped"
            _save_run(plan)
            continue
        name = it["id"] + " " + (Path(it["source"]).name if it.get("source") else f"({it.get('count')} photos)")
        log(f"[{n}/{len(plan['items'])}] {name}: {json.dumps(it['decision'], default=str)}")
        t0 = time.time()
        texture_default, it["state"] = config.TEXTURE_STRENGTH, "running"
        it.pop("error", None)
        _save_run(plan, it)
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
        _save_run(plan, it)
        if it.pop("redo", False):
            it["state"] = "pending"
            it.setdefault("reasons", []).append("the decision changed while this item was running: it ran with the "
                                                "old one and is pending again for the new one")
            _save_run(plan)
            log(f"{it['id']}: decision changed during the run: pending again (run the plan once more)")
        log(f"{it['id']}: {it['state']} in {it['minutes']} min")
    plan["report"] = str(write_report(plan))
    plan["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_run(plan)
    log(f"FINISHED: report {plan['report']}")
    return plan


# ----------------------------------------------------------------------------- recipe trials
def trial(out: Path, item_id: str, recipes: list[dict], log: Callable[[str], None] = print) -> dict:
    """Run the video's guarded previews once per recipe and lay the results side by side for a look.
    recipes: 2 to 4 dicts with any of content, model, scale, texture, remove_repeats, crop_borders (the rest comes
    from the item's current decision) and an optional name. A recipe's model is the only candidate of its column,
    so the sheet shows that model or says the guard rejected it. Nothing is delivered; the plan keeps the outcome
    under the item's `trials` for decide_plan to act on."""
    from . import pipeline_video
    from . import texture as texture_pass
    from .media import probe as media_probe
    from .report import closeups

    plan = load(out)
    out = Path(plan["out"])
    it = next((x for x in plan["items"] if x["id"] == item_id), None)
    if it is None or it["type"] not in ("video", "series"):
        raise ValueError(f"no video item {item_id!r} in the plan")
    if not TRIAL_RECIPES[0] <= len(recipes) <= TRIAL_RECIPES[1]:
        raise ValueError(f"give {TRIAL_RECIPES[0]} to {TRIAL_RECIPES[1]} recipes to compare")
    allowed = TRIAL_KEYS - (FRAMING_KEYS if it["type"] == "series" else set())
    typed = []
    for n, r in enumerate(recipes, 1):
        r = dict(r)
        name = str(r.pop("name", "") or f"recipe {n}")
        bad = set(r) - allowed
        if bad:
            raise ValueError(f"recipe {n}: unknown key(s) {sorted(bad)}; allowed: {sorted(allowed)} and name")
        r = _typed(f"recipe {n}", r)
        if r.get("model") is not None and r["model"] not in config.VIDEO_MODEL_CHOICES:
            raise ValueError(f"recipe {n}: model must be one of {config.VIDEO_MODEL_CHOICES}")
        if r.get("texture") is not None and not 0 <= float(r["texture"]) <= 1:
            raise ValueError(f"recipe {n}: texture must be between 0 and 1")
        if "filters" in r:
            try:
                r["filters"] = cleanup.parse(r["filters"])
            except ValueError as e:
                raise ValueError(f"recipe {n}: {e}") from None
        if r.get("recipe") is not None:
            try:
                r["recipe"] = recipe_mod.validate(r["recipe"])[0]
            except ValueError as e:
                raise ValueError(f"recipe {n}: {e}") from None
            _check_kinds(f"recipe {n}", it, r["recipe"])
        typed.append((name, r))
    src = Path(it.get("analysed_episode") or it["source"])
    tdir = out / "trials" / item_id
    run_dir = tdir / "run"
    columns: list[tuple[str, list]] = []
    results = []
    half = pipeline_video.PREVIEW_SECONDS / 2
    for n, (name, r) in enumerate(typed, 1):
        d = {**it["decision"], **r}
        log(f"{item_id} trial {n}/{len(typed)} {name}: {json.dumps(r, default=str)}")
        rec = pipeline_video.restore_video(src, run_dir, preview_only=True, scale=d.get("scale"),
                                           content=d.get("content") or "auto", model=d.get("model"),
                                           model_strict=bool(d.get("model")),
                                           mask_tape_noise=bool(d.get("mask_tape_noise")),
                                           remove_repeats=bool(d.get("remove_repeats")),
                                           crop_borders=bool(d.get("crop_borders")), filters=d.get("filters"),
                                           recipe=d.get("recipe"), log=log)
        att = rec.attempts[-1] if rec.attempts else None
        work = run_dir / "work" / pipeline_video.work_key(run_dir, src)
        asked = (d.get("recipe") or {}).get("texture") if d.get("recipe") else d.get("texture")
        everywhere = config.TEXTURE_STRENGTH if asked is None else float(asked)
        # each preview is shown with what its kind of scene gets (the steps already ran on the whole video)
        by_kind = recipe_mod.uses_kinds(d.get("recipe") or {})
        kinds_of = _preview_kinds(it, rec) if by_kind else []
        shots: list = []
        sources: list = []
        for i in range(att.previews if att else 0):
            clip = work / "previews" / f"src_{i}.mkv"
            made = work / "previews" / "out" / f"{att.label.replace(' ', '_')}_{i}.mkv"
            shown = made
            strength = recipe_mod.texture_for(d["recipe"], kinds_of[i], everywhere) if i < len(kinds_of) else everywhere
            if made.exists() and strength > 0:
                tex = tdir / f"r{n}_{i}.mkv"
                tdir.mkdir(parents=True, exist_ok=True)
                if texture_pass.apply(clip, made, tex, strength, "ffv1").ok:
                    shown = tex
            # the same frame NUMBER of the preview and of its source clip (they hold the same frames)
            middle = (media_probe(clip, exact=True)["frames"] // 2) if clip.exists() else 0
            fr = closeups.frames_at(shown, half, index=middle) if shown.exists() else []
            shots.append(fr[0] if fr else None)
            fr = closeups.frames_at(clip, half, index=middle) if clip.exists() else []
            sources.append(fr[0] if fr else None)
        for f in tdir.glob(f"r{n}_*.mkv"):
            f.unlink(missing_ok=True)
        if not columns:
            columns.append(("source", sources))
        passed = bool(att and att.passed_previews == att.previews and att.previews)
        failed = sorted({g for f in (att.failed if att else []) for g in f})
        columns.append((f"{n} {name}" + ("" if passed else " (GUARD REJECTED)"), shots))
        results.append({"n": n, "name": name, "recipe": r, "model": att.label if att else None,
                        "guard_passed": passed, "failed_gates": failed, "vmaf": att.vmaf if att else [],
                        **({"preview_kinds": kinds_of} if kinds_of else {}),
                        "notes": [x for x in rec.notes if "preview only" not in x]})
        log(f"{item_id} trial {n} {name}: {'passed the guard' if passed else 'REJECTED by the guard: ' + ', '.join(failed)}")
    sheet = closeups.recipe_sheet(columns, tdir / "recipes.jpg")
    shutil.rmtree(run_dir, ignore_errors=True)              # previews and passes of the trial: several GB
    outcome = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "sheet": str(sheet) if sheet else None, "recipes": results,
               "next": "LOOK at the sheet (first column is the source, then one column per recipe, one row per "
                       "preview). Choose only among recipes the guard passed, say why, and set it with decide_plan."}
    path = out / "plan.json"
    with statefile.locked(path):
        disk = statefile.read_json(path)
        for x in disk.get("items", []):
            if x.get("id") == item_id:
                x["trials"] = outcome
        statefile.write_json(path, disk)
    return disk


# ----------------------------------------------------------------------------- review
def review(out: Path, reviews: dict[str, dict], by: str = "claude") -> dict:
    """Record what the delivered results look like: {item_id: {verdict: good | acceptable | redo, notes: "..."}}.
    The guard decides whether a result is faithful; this is the look that follows, shown in the report. `redo`
    sets the item back to pending (change its decision with decide_plan, then run the plan again)."""
    path = plan_path(out)
    with statefile.locked(path):
        plan = load(out)
        index = {it["id"]: it for it in plan["items"]}
        for item_id, r in reviews.items():
            if item_id not in index:
                raise ValueError(f"no item {item_id!r} in the plan (items: {', '.join(index)})")
            verdict, notes = str(r.get("verdict", "")).strip().lower(), str(r.get("notes", "")).strip()
            if verdict not in REVIEW_VERDICTS:
                raise ValueError(f"{item_id}: verdict must be one of {REVIEW_VERDICTS}")
            if not notes:
                raise ValueError(f"{item_id}: say what you saw (notes)")
            it = index[item_id]
            if not it.get("delivered") and it.get("state") not in ("delivered", "partly_delivered"):
                raise ValueError(f"{item_id}: nothing was delivered to review")
            it["review"] = {"verdict": verdict, "notes": notes, "by": by, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            person = str(r.get("person_verdict", "")).strip()
            if person:
                it["person_verdict"] = person
            if it.get("recipe_used") and it.get("source"):
                try:                                        # the look joins the case, for the next decision
                    cases.annotate(it["source"], it["recipe_used"], review=it["review"], verdict=person or None)
                except OSError:
                    pass
            if verdict == "redo":
                it["state"] = "pending"
        if plan.get("report"):
            plan["report"] = str(write_report(plan))
        save(plan)
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
                "tape noise masked" if d.get("mask_tape_noise") else None,
                "copied frames removed" if d.get("remove_repeats") else None,
                "black borders cut" if d.get("crop_borders") else None,
                f"clean-up: {cleanup.describe(d['filters'])}" if d.get("filters") else None) if x)
            if d.get("recipe"):
                dec = "recipe: " + recipe_mod.describe(d["recipe"])
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
        found = "".join(f"<li>{e(str(x.get('what')))}</li>" for x in (it.get("findings") or []))
        tr = it.get("trials") or {}
        tried = "".join(
            f"<li>{e(str(x.get('name')))} ({e(str(x.get('model')))}): "
            f"{'passed the guard' if x.get('guard_passed') else 'rejected by the guard: ' + e(', '.join(x.get('failed_gates') or []))}</li>"
            for x in tr.get("recipes") or [])
        rv = it.get("review") or {}
        rows.append(f"""<section class="{e(state)}"><h2>{e(it['id'])} · {e(what)}</h2>
<p class="st">{e(_STATE_TEXT.get(state, state))}{' · ' + e(res) if res else ''}{' · ' + str(it['minutes']) + ' min' if it.get('minutes') else ''} {link}</p>
{f'<p class="err">{e(it["error"])}</p>' if it.get('error') else ''}
<p><b>Decision</b> ({e(str(it.get('decided_by', 'analysis')))}): {e(dec)}</p>
{f'<p>Classical pre-pass planned: {e(it["prepass"])}</p>' if it.get('prepass') else ''}
{f'<ul class="why">{why}</ul>' if why else ''}{f'<p><b>What happened</b></p><ul>{notes}</ul>' if notes else ''}
{f'<p><b>Found by the analysis (corrected only when decided)</b></p><ul>{found}</ul>' if found else ''}
{f'<p><b>Recipes compared</b> (<a href="{e(rel(tr.get("sheet")))}">side by side</a>)</p><ul>{tried}</ul>' if tried else ''}
{f'<p><b>Review of the result</b> ({e(str(rv.get("by")))}): {e(str(rv.get("verdict")))}. {e(str(rv.get("notes")))}</p>' if rv else ''}
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
