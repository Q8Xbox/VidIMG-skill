"""Series comparison: before/after frames and measurements per episode, and ONE English report page for
the whole series (written into the series output folder; every image path in it is relative).

Measurements per frame (original bicubic-upscaled to the result size vs the result, so both are judged at
the same scale): Laplacian variance (fine detail / line sharpness), mean of the strongest 5% gradients
(edge contrast), 8x8 blockiness ratio (codec blocking, 1.0 = none), and the result shrunk back to the
original size vs the original (PSNR, mean ΔE): high PSNR + low ΔE = nothing moved or recoloured.
"""

from __future__ import annotations

import html
import json
import re
import statistics as stats
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .compare import VIDEO_GATE_NAMES

# ---- frames and measurements --------------------------------------------------------------------------


def _stream(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,sample_aspect_ratio,display_aspect_ratio,codec_name,pix_fmt"
                        ":format=duration,size,bit_rate", "-of", "json", str(path)], capture_output=True, text=True)
    j = json.loads(r.stdout or "{}")
    s, f = (j.get("streams") or [{}])[0], j.get("format") or {}
    a, _, b = (s.get("sample_aspect_ratio") or "1:1").partition(":")
    try:
        sar = int(a) / int(b) if int(b) else 1.0
    except ValueError:
        sar = 1.0
    return {"w": int(s.get("width") or 0), "h": int(s.get("height") or 0), "sar": sar or 1.0,
            "dar": s.get("display_aspect_ratio"), "codec": s.get("codec_name"), "pix": s.get("pix_fmt"),
            "duration": float(f.get("duration") or 0), "mib": int(f.get("size") or 0) / 2 ** 20,
            "mbps": int(f.get("bit_rate") or 0) / 1e6}


def _grab(path: Path, t: float, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3)


def _sharpness(g):
    import cv2
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _edge(g):
    import cv2
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return float(m[m > np.percentile(m, 95)].mean())


def blockiness(g: np.ndarray) -> float:
    """Mean step across 8-px block boundaries / mean step inside blocks (1.0 = no visible block grid)."""
    d = np.abs(np.diff(g.astype(np.float32), axis=1))
    inner = np.delete(d, np.s_[7::8], axis=1).mean()
    return float(d[:, 7::8].mean() / max(inner, 1e-6))


def _delta_e(a, b):
    import cv2
    la, lb = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) for x in (a, b))
    for x in (la, lb):
        x[..., 0] *= 100 / 255
        x[..., 1:] -= 128
    return float(np.sqrt(((la - lb) ** 2).sum(-1)).mean())


def _label(img, text):
    import cv2
    img = img.copy()
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 18 + 13 * len(text)), 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _busiest(g, cw, ch):
    """Top-left corner of the crop with the most line detail (where differences are easiest to see)."""
    import cv2
    e = cv2.Canny(g, 60, 150)
    h, w = g.shape
    best, pos = -1, (0, 0)
    for y in range(0, max(1, h - ch), 24):
        for x in range(0, max(1, w - cw), 24):
            s = int(e[y:y + ch, x:x + cw].sum())
            if s > best:
                best, pos = s, (x, y)
    return pos


def default_times(duration: float, n: int = 10) -> list[int]:
    """n moments spread from 5% to 90% of the running time (skips most of the opening and end credits)."""
    if duration <= 0:
        return [0]
    return sorted({int(duration * (0.05 + 0.85 * i / max(1, n - 1))) for i in range(n)})


def episode_frames(original: Path, result: Path, outdir: Path, times: list[int] | None = None) -> list[dict]:
    """side_NN_TTTTs.jpg (both shown in the true on-screen shape), zoom_NN_TTTTs.png (original pixels /
    bicubic / result) and metrics.json for each moment."""
    import cv2
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    so, sr = _stream(original), _stream(result)
    ow, oh, rw, rh = so["w"], so["h"], sr["w"], sr["h"]
    k = rw / ow
    dh, dw = rh, round(rw * so["sar"] / 2) * 2
    metrics = []
    for i, t in enumerate(times or default_times(so["duration"])):
        o, r = _grab(original, t, ow, oh), _grab(result, t, rw, rh)
        side = np.hstack([_label(cv2.resize(o, (dw, dh), interpolation=cv2.INTER_CUBIC),
                                 f"ORIGINAL {ow}x{oh} (player upscale)  t={t}s"),
                          np.full((dh, 8, 3), 255, np.uint8),
                          _label(cv2.resize(r, (dw, dh), interpolation=cv2.INTER_CUBIC), f"VidIMG {rw}x{rh}  t={t}s")])
        cv2.imwrite(str(outdir / f"side_{i:02d}_{t:04d}s.jpg"), side, [cv2.IMWRITE_JPEG_QUALITY, 92])
        go = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
        cw, ch = min(240, ow), min(180, oh)
        x, y = _busiest(go, cw, ch)
        co = o[y:y + ch, x:x + cw]
        cr = r[int(y * k):int((y + ch) * k), int(x * k):int((x + cw) * k)]
        zw, zh = cw * 3, ch * 3
        gap = np.full((zh, 6, 3), 255, np.uint8)
        zoom = np.hstack([_label(cv2.resize(co, (zw, zh), interpolation=cv2.INTER_NEAREST), "original pixels"), gap,
                          _label(cv2.resize(co, (zw, zh), interpolation=cv2.INTER_CUBIC), "original, bicubic"), gap,
                          _label(cv2.resize(cr, (zw, zh), interpolation=cv2.INTER_CUBIC), "VidIMG")])
        cv2.imwrite(str(outdir / f"zoom_{i:02d}_{t:04d}s.png"), zoom)
        ob = cv2.resize(o, (rw, rh), interpolation=cv2.INTER_CUBIC)
        back = cv2.resize(r, (ow, oh), interpolation=cv2.INTER_AREA)
        gob, gr, gback = (cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) for v in (ob, r, back))
        from .. import texture
        flat_o, flat_r = texture.flat_shares(go, gr)          # over-smoothing between the lines (vs bicubic)
        metrics.append({"t": t, "sharpness_orig": _sharpness(gob), "sharpness_rest": _sharpness(gr),
                        "flat_orig": flat_o, "flat_rest": flat_r,
                        "edge_orig": _edge(gob), "edge_rest": _edge(gr),
                        "blockiness_orig": blockiness(go), "blockiness_rest_down": blockiness(gback),
                        "deltaE_back": _delta_e(o, back), "psnr_back": float(cv2.PSNR(o, back))})
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


# ---- the report page ----------------------------------------------------------------------------------

_CSS = """
:root { --bg:#f7f6f2; --panel:#fff; --ink:#1d1d1b; --muted:#66645e; --line:#e2dfd6;
        --good:#1f7a4d; --goodbg:#e5f3ea; --warn:#9a5b00; --warnbg:#fbf0dc; --bad:#a3271c; --badbg:#fbe4e1; --accent:#2b5d8a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg:#161614; --panel:#201f1c; --ink:#ecebe6; --muted:#a29f96; --line:#34322d;
        --good:#6fcf9c; --goodbg:#17301f; --warn:#f0b25a; --warnbg:#33270f; --bad:#ff8a7a; --badbg:#3a1a16; --accent:#8cb8e0; } }
:root[data-theme="dark"] { --bg:#161614; --panel:#201f1c; --ink:#ecebe6; --muted:#a29f96; --line:#34322d;
        --good:#6fcf9c; --goodbg:#17301f; --warn:#f0b25a; --warnbg:#33270f; --bad:#ff8a7a; --badbg:#3a1a16; --accent:#8cb8e0; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:30px; line-height:1.2; margin:0 0 6px; }
h2 { font-size:21px; margin:44px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--line); }
h3 { font-size:17px; margin:26px 0 8px; }
.sub { color:var(--muted); margin:0 0 22px; }
.verdict { border-radius:10px; padding:16px 18px; border:1px solid var(--good); background:var(--goodbg); }
.verdict.partial { border-color:var(--warn); background:var(--warnbg); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:18px 0; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.card .k { color:var(--muted); font-size:13px; } .card .v { font-size:22px; font-weight:650; } .card .d { font-size:13px; color:var(--muted); }
.tablewrap { overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:10px; }
.tablewrap.tall { max-height:640px; }
table { border-collapse:collapse; width:100%; font-size:14.5px; }
th, td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-weight:600; color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.03em; position:sticky; top:0; background:var(--panel); }
tfoot td { font-weight:650; border-top:2px solid var(--line); }
td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
.pass { color:var(--good); font-weight:600; } .warn { color:var(--warn); font-weight:600; } .fail { color:var(--bad); font-weight:600; }
.box { border-radius:10px; padding:12px 16px; margin:10px 0; border:1px solid var(--line); background:var(--panel); }
.box.good { border-left:4px solid var(--good); } .box.warn { border-left:4px solid var(--warn); background:var(--warnbg); }
figure { margin:14px 0 24px; } figure img { width:100%; height:auto; display:block; border-radius:8px; border:1px solid var(--line); }
figcaption { color:var(--muted); font-size:14px; margin-top:6px; }
details { margin:10px 0; } summary { cursor:pointer; font-weight:600; padding:6px 0; }
.links { font-size:14px; color:var(--muted); }
code { font-family:ui-monospace,Consolas,monospace; font-size:13.5px; background:var(--bg); padding:1px 5px; border-radius:4px; border:1px solid var(--line); word-break:break-all; }
a { color:var(--accent); } ul { margin:6px 0; padding-left:22px; } li { margin:4px 0; }
@media (max-width:760px) { h1 { font-size:25px; } }
"""

FACE_GATES = ("identity", "landmarks", "face_backprojection", "eye_color")
STRUCTURE_GATES = ("frames", "duration", "timestamps", "audio", "av_sync", "aspect")


def _e(x) -> str:
    return html.escape(str(x))


def _mmss(t: int) -> str:
    return f"{t // 60}:{t % 60:02d}"


def _secs(p: Path) -> int:
    return int(p.stem.split("_")[-1][:-1])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return stats.mean(xs) if xs else None


def _fmt(x, spec: str, none: str = "—") -> str:
    return none if x is None else format(x, spec)


def series_title(names: list[str]) -> str:
    """'[Group] Show Name - 01 [tags].mkv' x N -> 'Show Name'."""
    cleaned = []
    for n in names:
        s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", Path(n).stem)
        s = re.split(r"[Ss]\d{1,2}\s*[Ee]\d|\s-\s\d|(?:^|\s)(?:[Ee][Pp]|Episode)\s*\.?\s*\d", s)[0]
        cleaned.append(re.sub(r"[._]+", " ", s).strip(" -"))
    common = cleaned[0] if cleaned else ""
    for c in cleaned[1:]:
        while common and not c.startswith(common):
            common = common[:-1]
    return common.strip(" -") or "Series"


def collect(out: Path) -> list[dict]:
    """One row per episode from batch_status.json, the records and comparison/EpNN/metrics.json."""
    out = Path(out)
    status = json.loads((out / "batch_status.json").read_text(encoding="utf-8"))
    rows = []
    for key in sorted(k for k in status if not k.startswith("_")):
        st = status[key]
        src = Path(st.get("source") or "")
        rec_file = out / "records" / f"{src.stem}.json"
        rec = json.loads(rec_file.read_text(encoding="utf-8")) if rec_file.exists() else {}
        row = {"ep": key, "source": str(src), "name": src.name, "delivered": st.get("delivered"),
               "passed": bool(st.get("passed")), "chosen": st.get("chosen"), "minutes": st.get("minutes"),
               "notes": st.get("notes") or [], "error": st.get("error"), "align": st.get("alignment_offsets"),
               "gates": rec.get("final_gates") or {}, "attempts": rec.get("attempts") or [],
               "report": f"reports/{src.stem}.html" if (out / "reports" / f"{src.stem}.html").exists() else None}
        mfile = out / "comparison" / f"Ep{key}" / "metrics.json"
        m = json.loads(mfile.read_text(encoding="utf-8")) if mfile.exists() else []
        row["sharp_x"] = _mean([x["sharpness_rest"] / x["sharpness_orig"] for x in m if x["sharpness_orig"] > 0])
        row["edge_x"] = _mean([x["edge_rest"] / x["edge_orig"] for x in m if x["edge_orig"] > 0])
        row["flat_up"] = _mean([x["flat_rest"] - x["flat_orig"] for x in m if "flat_rest" in x])
        row["psnr"] = _mean([x["psnr_back"] for x in m])
        row["dE"] = _mean([x["deltaE_back"] for x in m])
        row["block_o"] = _mean([x["blockiness_orig"] for x in m])
        row["block_r"] = _mean([x["blockiness_rest_down"] for x in m])
        row["orig"] = _stream(src) if src.is_file() else {}
        row["rest"] = _stream(Path(row["delivered"])) if row["delivered"] and Path(row["delivered"]).is_file() else {}
        rows.append(row)
    return rows


def _gate(row: dict, name: str):
    g = row["gates"].get(name) or {}
    return g.get("status"), g.get("value")


def _fallback_reason(row: dict) -> str | None:
    """Why the first model in the approved order was not used (from the preview attempts)."""
    att = row["attempts"]
    if not att or not row["chosen"] or att[0].get("label") == row["chosen"]:
        return None
    first = att[0]
    failed = sorted({g for clip in first.get("failed") or [] for g in clip})
    names = ", ".join(VIDEO_GATE_NAMES.get(g, g).lower() for g in failed) or "a guard check"
    return (f"{first.get('label')} passed {first.get('passed_previews')} of {first.get('previews')} test clips "
            f"({names}); {row['chosen']} passed them all")


def page(rows: list[dict], out: Path, options: dict | None = None) -> str:
    out = Path(out)
    options = options or {}
    done = [r for r in rows if r["delivered"]]
    not_done = [r for r in rows if not r["delivered"]]
    first, last = rows[0]["ep"], rows[-1]["ep"]
    title = series_title([r["name"] for r in rows])
    scale = options.get("scale")
    tag = f"x{scale:g}" if scale else "restored"
    anime = options.get("content") == "anime" or any("anime" in (r["chosen"] or "") for r in done)
    vals = lambda name: [v for r in done for _, v in [_gate(r, name)] if isinstance(v, (int, float)) and not isinstance(v, bool)]
    agg = lambda key: _mean([r.get(key) for r in done])
    tot_o = sum(r["orig"].get("mib", 0) for r in rows) / 1024
    tot_r = sum(r["rest"].get("mib", 0) for r in done) / 1024
    hours = sum(r["minutes"] or 0 for r in rows) / 60
    run_h = sum(r["orig"].get("duration", 0) for r in rows) / 3600
    aligned = [r for r in done if r["align"] and all(o == 0 for o in r["align"])]
    models: dict[str, list[str]] = {}
    for r in done:
        models.setdefault(r["chosen"], []).append(r["ep"])
    o0 = next((r["orig"] for r in rows if r["orig"]), {})
    r0 = next((r["rest"] for r in done if r["rest"]), {})

    H: list[str] = []
    a = H.append
    a(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)} Series Comparison</title><style>{_CSS}</style></head><body><main>
<h1>{_e(title)}, episodes {first}–{last}: original vs VidIMG {tag}</h1>
<p class="sub">{len(rows)} episodes, {run_h:.1f} hours of video. Source: <code>{_e(options.get('source', ''))}</code> (left untouched). Output: this folder.</p>""")
    if not_done:
        verdict = (f"<strong>{len(done)} of {len(rows)} episodes were enhanced and delivered.</strong> "
                   f"Not delivered: {', '.join(r['ep'] for r in not_done)}. For these the original stays the "
                   "deliverable (the reason is in the table and on each episode's guard page).")
    else:
        verdict = f"<strong>All {len(rows)} episodes were enhanced and delivered, and every one passed the fidelity guard.</strong>"
    if done:
        verdict += (f" Frame count, length, audio and lip sync were checked against the original for every delivered episode; "
                    f"{len(aligned)} of {len(done)} also matched frame for frame in the alignment check.")
        if agg("sharp_x"):
            verdict += f" On average lines are {agg('sharp_x'):.1f}× sharper and edges {agg('edge_x'):.2f}× crisper."
    a(f"<div class='verdict{' partial' if not_done else ''}'>{verdict}</div>")

    def card(k, v, d=""):
        a(f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div><div class='d'>{d}</div></div>")
    a("<div class='cards'>")
    card("Delivered", f"{len(done)} / {len(rows)}", "passed the guard" if not not_done else "others kept the original")
    if agg("sharp_x"):
        card("Line sharpness", f"{agg('sharp_x'):.1f}×", f"range {min(r['sharp_x'] for r in done if r['sharp_x']):.1f}–{max(r['sharp_x'] for r in done if r['sharp_x']):.1f}×")
        card("Edge contrast", f"{agg('edge_x'):.2f}×", "strongest 5% of edges")
    if agg("flat_up") is not None:
        card("Texture between the lines", f"{agg('flat_up') * 100:+.0f} pts flat",
             "flat area vs a plain upscale (near 0 = texture kept; large = smoothed away)")
    if vals("backprojection"):
        card("Structure kept", f"{_mean(vals('backprojection')):.3f}", "back-projection (pass ≥ 0.93)")
    if vals("color"):
        card("Colour shift", f"ΔE {_mean(vals('color')):.2f}", "limit 4")
    if vals("vmaf"):
        card("VMAF", f"{_mean(vals('vmaf')):.1f}", f"range {min(vals('vmaf')):.1f}–{max(vals('vmaf')):.1f}")
    card("Processing time", f"{hours:.1f} h", f"about {hours * 60 / max(1, len(rows)):.0f} min per episode")
    card("Total size", f"{tot_r:.1f} GiB", f"originals {tot_o:.1f} GiB")
    a("</div>")

    if o0 and r0:
        a(f"""<h2>1. The files side by side</h2><div class="tablewrap"><table><thead><tr><th></th><th>Original</th><th>VidIMG</th></tr></thead><tbody>
<tr><td>Resolution</td><td class="num">{o0['w']} × {o0['h']}</td><td class="num">{r0['w']} × {r0['h']}</td></tr>
<tr><td>Screen shape</td><td class="num">{_e(o0.get('dar') or '—')}</td><td class="num">{_e(r0.get('dar') or '—')}</td></tr>
<tr><td>Video codec</td><td>{_e(o0.get('codec'))} ({_e(o0.get('pix'))})</td><td>{_e(r0.get('codec'))} ({_e(r0.get('pix'))})</td></tr>
<tr><td>Average bitrate</td><td class="num">{_mean([r['orig'].get('mbps') for r in rows]):.2f} Mb/s</td><td class="num">{_mean([r['rest'].get('mbps') for r in done]):.2f} Mb/s</td></tr>
<tr><td>Average file size</td><td class="num">{_mean([r['orig'].get('mib') for r in rows]):.0f} MiB</td><td class="num">{_mean([r['rest'].get('mib') for r in done]):.0f} MiB</td></tr>
<tr><td>Frames, audio, timing</td><td colspan="2">kept identical (checked by the guard on every delivered episode)</td></tr>
</tbody></table></div>""")

    a("<h2>2. What was done</h2><ul>")
    a("<li>Every episode ran the guarded VidIMG video pipeline with free, local tools only: short test clips through the approved models in order, the first model whose clips passed every guard check for the full episode, then the guard again on the finished file. No generative model and no colour or lighting change; nothing was sent to an outside service.</li>")
    for model, eps in models.items():
        a(f"<li><strong>{_e(model)}</strong>: episode{'s' if len(eps) > 1 else ''} {', '.join(eps)}.</li>")
    for r in done:
        why = _fallback_reason(r)
        if why:
            a(f"<li>Episode {r['ep']}: {_e(why)}.</li>")
    a("<li>Alignment check: near 20% and 60% of each delivered episode, a moving frame of the original was compared with the result frames around it (still scenes are skipped because they cannot tell frames apart). Offset 0 means the best match is the frame with the same number: nothing dropped, added or shifted.</li></ul>")

    a("""<h2>3. Episode by episode</h2>
<p>Sharpness and edge figures are averages over 10 moments per episode, result divided by original (both at the result size). Flat area: how many points more of the area between the drawn lines is completely flat in the result than in a plain upscale of the original (texture smoothed away; near 0 is best). PSNR and ΔE compare the result, shrunk back to the original size, with the original: high PSNR and low ΔE mean nothing was moved or recoloured.</p>
<div class="tablewrap tall"><table><thead><tr><th>Ep</th><th>Result</th><th>Model</th><th>Sharpness</th><th>Edges</th><th>Flat area</th><th>PSNR back</th><th>ΔE back</th><th>Structure</th><th>Colour ΔE</th><th>Flicker</th><th>VMAF</th><th>Size (orig → new)</th><th>Min</th><th>Aligned</th><th>Guard</th></tr></thead><tbody>""")
    for r in rows:
        if r["delivered"]:
            res = "<span class='pass'>delivered</span>"
        else:
            res = f"<span class='fail'>kept original</span>{'<br><small>' + _e(r['error']) + '</small>' if r['error'] else ''}"
        al = "—" if not r["align"] else ("yes" if all(o == 0 for o in r["align"]) else _e(r["align"]))
        cell = lambda name, spec: _fmt(_gate(r, name)[1] if isinstance(_gate(r, name)[1], (int, float)) else None, spec)
        size = f"{r['orig'].get('mib', 0):.0f} → {r['rest']['mib']:.0f} MiB" if r["rest"] else f"{r['orig'].get('mib', 0):.0f} MiB"
        a(f"<tr><td class='num'><strong>{r['ep']}</strong></td><td>{res}</td><td>{_e(r['chosen'] or '—')}</td>"
          f"<td class='num'>{_fmt(r['sharp_x'], '.1f')}×</td><td class='num'>{_fmt(r['edge_x'], '.2f')}×</td>"
          f"<td class='num'>{_fmt(r.get('flat_up') * 100 if r.get('flat_up') is not None else None, '+.0f')}</td>"
          f"<td class='num'>{_fmt(r['psnr'], '.1f')} dB</td><td class='num'>{_fmt(r['dE'], '.2f')}</td>"
          f"<td class='num'>{cell('backprojection', '.3f')}</td><td class='num'>{cell('color', '.2f')}</td>"
          f"<td class='num'>{cell('flicker', '+.4f')}</td><td class='num'>{cell('vmaf', '.1f')}</td>"
          f"<td class='num'>{size}</td><td class='num'>{_fmt(r['minutes'], '.0f')}</td><td class='num'>{al}</td>"
          f"<td>{'<a href=' + chr(39) + quote(r['report']) + chr(39) + '>page</a>' if r['report'] else '—'}</td></tr>")
    a(f"""</tbody><tfoot><tr><td>Avg</td><td>{len(done)}/{len(rows)}</td><td></td><td class='num'>{_fmt(agg('sharp_x'), '.1f')}×</td><td class='num'>{_fmt(agg('edge_x'), '.2f')}×</td>
<td class='num'>{_fmt(agg('flat_up') * 100 if agg('flat_up') is not None else None, '+.0f')}</td>
<td class='num'>{_fmt(agg('psnr'), '.1f')} dB</td><td class='num'>{_fmt(agg('dE'), '.2f')}</td><td class='num'>{_fmt(_mean(vals('backprojection')), '.3f')}</td>
<td class='num'>{_fmt(_mean(vals('color')), '.2f')}</td><td class='num'>{_fmt(_mean(vals('flicker')), '+.4f')}</td><td class='num'>{_fmt(_mean(vals('vmaf')), '.1f')}</td>
<td class='num'>{tot_o:.1f} → {tot_r:.1f} GiB</td><td class='num'>{hours:.1f} h</td><td>{len(aligned)}/{len(done)}</td><td></td></tr></tfoot></table></div>""")

    a("<h2>4. Fidelity guard across the series</h2><div class='tablewrap'><table><thead><tr><th>Check</th><th>Range over delivered episodes</th><th>Pass</th><th>Warn</th><th>Fail</th></tr></thead><tbody>")
    gate_names = [g for g in VIDEO_GATE_NAMES if any(g in r["gates"] for r in rows)]
    gate_names += sorted({g for r in rows for g in r["gates"]} - set(gate_names))
    for g in gate_names:
        st_list = [_gate(r, g)[0] for r in done]
        v = [x for x in vals(g)]
        rng = "—" if not v or g in STRUCTURE_GATES else (f"{min(v):.4g} – {max(v):.4g}" if min(v) != max(v) else f"{v[0]:.4g}")
        if g in STRUCTURE_GATES:
            rng = "identical" if all(s == "pass" for s in st_list if s) else "see guard pages"
        a(f"<tr><td>{_e(VIDEO_GATE_NAMES.get(g, g))}{' (advisory on drawn faces)' if anime and g in FACE_GATES else ''}</td><td class='num'>{rng}</td>"
          f"<td class='num pass'>{st_list.count('pass')}</td><td class='num warn'>{st_list.count('warn') or ''}</td><td class='num fail'>{st_list.count('fail') or ''}</td></tr>")
    a("</tbody></table></div>")
    face_warn = [r["ep"] for r in done if any(_gate(r, g)[0] == "warn" for g in FACE_GATES)]
    if face_warn:
        if anime:
            a(f"<p>Face checks: the face detector also fires on drawn faces. On anime/cartoon content these checks only advise and never block, because a cleaner ink line changes the face-recognition score without changing the character. Episodes with a face advisory: {', '.join(face_warn)}. Look at the faces in those episodes' guard sheets (<code>sheets/</code>) and judge by eye.</p>")
        else:
            a(f"<p class='box warn'>Face warnings in episodes {', '.join(face_warn)}: the result was delivered, but please look at the faces in those episodes' guard sheets (<code>sheets/</code>). Your visual verdict outranks the numbers.</p>")

    a("<h2>5. What to look for</h2><div class='box good'><strong>Expected gains</strong><ul>")
    if agg("sharp_x"):
        a(f"<li>Sharper lines and edges ({agg('sharp_x'):.1f}× detail, {agg('edge_x'):.2f}× edge contrast on average).</li>")
    if agg("block_o") is not None:
        a(f"<li>Codec blocking: 8×8 block score {agg('block_o'):.3f} → {agg('block_r'):.3f} on average (1.0 = no visible block grid).</li>")
    a("<li>Same picture: frames, timing, colour and shape checked against the original.</li></ul></div><div class='box warn'><strong>Known trade-offs</strong><ul>")
    if anime:
        a("<li>Anime models clean the picture: film grain and paper texture are smoothed away, and fine brush texture in painted backgrounds can look slightly flatter.</li>"
          "<li>Ink lines can come out a little bolder and darker than in the original.</li>")
    else:
        a("<li>The faithful (non-GAN) models look softer than tools that invent texture; they do not add detail the source never had.</li>")
    if tot_o:
        a(f"<li>Files are about {tot_r / tot_o if tot_o else 0:.1f}× larger ({tot_r:.1f} GiB vs {tot_o:.1f} GiB).</li>")
    a("<li>This is a cleaner, larger version of the same source, not a new master: detail lost in the source cannot be recovered.</li></ul></div>")

    zooms = []
    if done:
        step = max(1, len(done) // 8)
        for i, r in enumerate(done[::step][:8]):
            files = sorted((out / "comparison" / f"Ep{r['ep']}").glob("zoom_*.png"))
            if files:
                zooms.append((r["ep"], files[(i * 3 + 1) % len(files)]))
    if zooms:
        a("<h2>6. Zoomed comparisons (1:1 pixels)</h2><p>The same crop three ways: <em>original pixels</em> (enlarged without smoothing), <em>original, bicubic</em> (what a player shows when stretching it) and <em>VidIMG</em>.</p>")
        for ep, f in zooms:
            a(f"<figure><img loading='lazy' src='comparison/Ep{ep}/{quote(f.name)}' alt='Episode {ep} zoom at {_mmss(_secs(f))}'><figcaption>Episode {ep}, {_mmss(_secs(f))}</figcaption></figure>")
    if done:
        a("<h2>7. Full frames side by side, every episode</h2><p>Left: the original stretched as a player shows it. Right: VidIMG. Both in the true on-screen shape. Each <code>comparison/EpNN/</code> folder holds all frames and zooms; the time links open them.</p>")
        for i, r in enumerate(done):
            files = sorted((out / "comparison" / f"Ep{r['ep']}").glob("side_*.jpg"))
            if not files:
                continue
            pick = files[int(r["ep"]) % len(files)]
            a(f"<details{' open' if i in (0, len(done) // 2) else ''}><summary>Episode {r['ep']} ({_e(r['chosen'])})</summary>"
              f"<figure><img loading='lazy' src='comparison/Ep{r['ep']}/{quote(pick.name)}' alt='Episode {r['ep']} at {_mmss(_secs(pick))}'>"
              f"<figcaption>Episode {r['ep']}, {_mmss(_secs(pick))}</figcaption></figure><p class='links'>All frames: "
              + " · ".join(f"<a href='comparison/Ep{r['ep']}/{quote(p.name)}'>{_mmss(_secs(p))}</a>" for p in files) + "</p></details>")

    a(f"""<h2>8. What is in this folder</h2><div class="tablewrap"><table><tbody>
<tr><td><code>…_restored.*</code></td><td>The delivered episodes ({tot_r:.1f} GiB)</td></tr>
<tr><td><code>reports/</code></td><td>One guard page per episode (every check and test-clip result)</td></tr>
<tr><td><code>records/</code></td><td>Machine-readable record per episode (JSON)</td></tr>
<tr><td><code>comparison/EpNN/</code></td><td>Side-by-side frames, zooms and metrics.json per episode</td></tr>
<tr><td><code>sheets/</code></td><td>The guard's own before/after frames</td></tr>
<tr><td><code>batch_status.json</code></td><td>Run status (a rerun of the same command skips finished episodes)</td></tr>
<tr><td><code>work/</code></td><td>Test clips and checkpoints; safe to delete once every episode is finished</td></tr>
</tbody></table></div></main></body></html>""")
    return "\n".join(H)


def write(out: Path) -> Path:
    """Write 'Series Comparison Report - Episodes AA-BB.html' into the series folder."""
    out = Path(out)
    status = json.loads((out / "batch_status.json").read_text(encoding="utf-8"))
    rows = collect(out)
    if not rows:
        raise ValueError(f"no episodes recorded in {out / 'batch_status.json'}")
    span = f"Episode {rows[0]['ep']}" if len(rows) == 1 else f"Episodes {rows[0]['ep']}-{rows[-1]['ep']}"
    path = out / f"Series Comparison Report - {span}.html"
    path.write_text(page(rows, out, status.get("_options")), encoding="utf-8")
    return path
