"""Phase 6 acceptance report: one local HTML page (English) with every delivered photo and
video, its guard gates, the ground-truth evidence behind the engine choices, and a verdict form.

The page never leaves the machine (personal photos). The person marks each item "same person" /
"changed" / "unsure"; "copy verdict" puts a JSON block on the clipboard to paste back into the chat,
where it is saved as workspace/acceptance/human_verdict.json.

Usage: python -m vidimg.report.acceptance
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from .. import config
from .compare import GATE_NAMES, VIDEO_GATE_NAMES

ROOT = config.WORKSPACE / "acceptance"
PHOTO_RUN = ROOT / "photos"
VIDEO_RECORDS = {
    # free tools only (2026-11-16); earlier videos made with the removed paid engine are not shown
    "WhatsApp video (2 min, 352p → 704p, open model)": config.WORKSPACE / "acceptance" / "video_whatsapp_free3" / "record.json",
    "Cartoon test clip (Video2X for anime)": config.WORKSPACE / "bench_routing" / "out_video" / "record.json",
}
STATUS = {"pass": "✅", "warn": "⚠️", "fail": "⛔", "n/a": "—"}


def _asset(src: str | Path, name: str) -> str | None:
    p = Path(src)
    if not p.exists():
        return None
    dst = ROOT / "assets" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dst)
    from urllib.parse import quote
    return "assets/" + html.escape(quote(name, safe="/"))


def _gates_table(gates: dict, names: dict) -> str:
    rows = "".join(f"<tr><td>{STATUS.get(g.get('status'), '')}</td><td>{html.escape(names.get(k, k))}</td>"
                   f"<td>{g.get('value')}</td><td class='mut'>{g.get('threshold', '')}</td></tr>"
                   for k, g in gates.items())
    return f"<table class='gates'>{rows}</table>" if rows else ""


def _verdict(item_id: str) -> str:
    opts = (("same", "Same person and features"), ("changed", "Something in the face changed"), ("unsure", "Not sure"))
    radios = "".join(f"<label><input type='radio' name='{html.escape(item_id)}' value='{v}'> {t}</label>" for v, t in opts)
    return (f"<div class='verdict' data-id='{html.escape(item_id)}'>{radios}"
            f"<input type='text' class='note' placeholder='Note (optional)'></div>")


def _photo_cards() -> tuple[str, dict]:
    recs = json.loads((PHOTO_RUN / "records.json").read_text(encoding="utf-8")) if (PHOTO_RUN / "records.json").exists() else []
    stats = {"total": len(recs), "delivered": 0, "warned": 0, "kept": 0, "skipped": 0}
    cards = []
    for r in recs:
        name = Path(r["source"]).name
        if not r["passed"]:
            key = "skipped" if any("already large" in n for n in r["notes"]) else "kept"
            stats[key] += 1
            if key == "kept":
                cards.append(f"<section class='card kept'><h3>{html.escape(name)}</h3>"
                             f"<p>⛔ Not delivered: no candidate passed the guard, so the original stays as it is.</p></section>")
            continue
        stats["delivered"] += 1
        atts = [a for a in r["attempts"] if a["passed"]]
        chosen = next((a for a in atts if not a.get("warnings")), atts[0] if atts else None)
        if chosen and chosen.get("warnings"):
            stats["warned"] += 1
        sheet = PHOTO_RUN / "sheets" / f"{Path(name).stem}.jpg"
        img = _asset(sheet, f"photo_{Path(name).stem}.jpg")
        warn = (f"<p class='warn'>⚠️ Delivered with a warning: {html.escape(', '.join(GATE_NAMES.get(w, w) for w in chosen['warnings']))}</p>"
                if chosen and chosen.get("warnings") else "")
        cards.append(f"""<section class='card'><h3>{html.escape(name)}</h3>
<p class='mut'>Model: <b>{html.escape(chosen['label'] if chosen else '?')}</b> · plan: {html.escape(r['plan'])}</p>{warn}
{f"<img src='{img}' loading='lazy' alt='Before and after'>" if img else ''}
<details><summary>Guard checks</summary>{_gates_table(chosen['gates'] if chosen else {}, GATE_NAMES)}</details>
{_verdict('photo:' + name)}</section>""")
    return "".join(cards), stats


def _video_cards() -> tuple[str, int]:
    cards, n = [], 0
    for title, path in VIDEO_RECORDS.items():
        if not path.exists():
            alt = list(path.parent.glob("work/*/record.json"))
            if not alt:
                continue
            path = alt[0]
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not rec.get("passed"):
            continue
        n += 1
        imgs = "".join(f"<img src='{a}' loading='lazy' alt='Frame'>" for i, s in enumerate(rec.get("sheets") or [])
                       if (a := _asset(s, f"video_{path.parent.name}_{i}.jpg")))
        pre = rec.get("prepass") or {}
        pre_txt = f" · classical pre-pass: {html.escape(str(pre.get('plan')))}" if pre.get("plan") else ""
        cards.append(f"""<section class='card'><h3>{html.escape(title)}</h3>
<p class='mut'>Model: <b>{html.escape(str(rec.get('chosen')))}</b>{pre_txt}</p>
{imgs}<details open><summary>Final guard checks</summary>{_gates_table(rec.get('final_gates') or {}, VIDEO_GATE_NAMES)}</details>
{_verdict('video:' + title)}</section>""")
    return "".join(cards), n


def _evidence() -> str:
    rows = []
    sf = config.WORKSPACE / "calibration" / "20261114-235009" / "summary_free.json"
    if sf.exists():
        s = json.loads(sf.read_text(encoding="utf-8"))
        names = {"free_bsrnet_x4": "BSRNet (default)", "free_realesrnet_x4": "RealESRNet",
                 "free_swinir_real_x4": "SwinIR", "free_lanczos": "Classical upscale (no AI)"}
        rows = "".join(f"<tr><td>{n}</td><td>{s[k]['gt_identity']}</td><td>{s[k]['psnr_vs_bicubic']:+}</td></tr>"
                       for k, n in names.items() if k in s)
    vf = config.WORKSPACE / "bench_free" / "video" / "summary.json"
    vrows = ""
    if vf.exists():
        v = json.loads(vf.read_text(encoding="utf-8"))
        vn = {"free_realesrnet_x4": "RealESRNet",
              "free_bsrnet_x4": "BSRNet", "free_general_x4v3": "general-v3 (fast)"}
        vrows = "".join(f"<tr><td>{n}</td><td>{v[k]['identity']}</td><td>{v[k]['vmaf']}</td><td>{v[k]['psnr']}</td></tr>"
                        for k, n in vn.items() if k in v)
    return f"""<section class='card'><h2>Why these engines? (measured against the real originals of your photos and clips)</h2>
<h3>Photos (12 photos, controlled degradation)</h3><table class='ev'><tr><th>Engine</th><th>Identity match with the original</th><th>Gain over a plain upscale (dB)</th></tr>{rows}</table>
<h3>Video (12 clips)</h3><table class='ev'><tr><th>Engine</th><th>Identity match</th><th>VMAF</th><th>PSNR</th></tr>{vrows}</table>
<p class='mut'>Higher is better in every column. Details: PLAN.md and REQUIREMENTS.md.</p></section>"""


def build() -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    photos, st = _photo_cards()
    videos, nv = _video_cards()
    eligible = st["total"] - st["skipped"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Acceptance Review</title><style>
:root{{--bg:#f5f6f8;--fg:#1c2230;--card:#fff;--line:#dde2ea;--mut:#5b6474;--acc:#1f6f5c;--warn:#9a5b00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--fg:#e6e9ef;--card:#1b1f27;--line:#2b313c;--mut:#9aa3b2;--acc:#5fc3a6;--warn:#f0b35a}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,"Segoe UI",Tahoma,sans-serif;line-height:1.65;padding:16px}}
main{{max-width:1500px;margin:auto}} h1{{margin:.2em 0}} h2{{font-size:1.15rem;margin:.2em 0 .6em}} h3{{font-size:1rem;margin:.2em 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:14px 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}} .stat b{{font-size:1.6rem;display:block;color:var(--acc)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:14px 0}} .card img{{width:100%;height:auto;border-radius:8px;margin:8px 0}}
.kept{{opacity:.85}} .mut{{color:var(--mut)}} .warn{{color:var(--warn);font-weight:600}}
table{{border-collapse:collapse;margin:6px 0}} td,th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:start}}
.verdict{{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:10px;border-top:1px dashed var(--line);margin-top:8px}}
.verdict .note{{flex:1;min-width:200px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}}
.bar{{position:sticky;bottom:0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
button{{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:1rem;cursor:pointer}}
textarea{{width:100%;min-height:90px;direction:ltr;font-family:Consolas,monospace;font-size:.8rem;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;display:none}}
</style></head><body><main>
<h1>Acceptance review (phase 6)</h1>
<p>The standard: <b>no change in anyone's identity</b>. You know the people in these photos, so your judgement decides. Each card shows the original enlarged (left) and the result (right), then the faces zoomed in, before/after. Pick a verdict for each item, then press "Copy verdict" and paste it into the chat.</p>
<div class='cards'>
<div class='stat'><b>{st['delivered']} / {eligible}</b>photos delivered (of those that could be enlarged)</div>
<div class='stat'><b>{st['warned']}</b>of them with a guard warning</div>
<div class='stat'><b>{st['kept']}</b>kept as the original (rejected by the guard)</div>
<div class='stat'><b>{st['skipped']}</b>already large, not processed</div>
<div class='stat'><b>{nv}</b>videos delivered</div>
<div class='stat'><b>Free tools only</b>no paid software at all</div>
</div>
{_evidence()}
<h2>Videos</h2>{videos}
<h2>Photos</h2>{photos}
<div class='bar'><button id='copy'>Copy verdict</button><span id='count' class='mut'></span><textarea id='out' readonly></textarea></div>
</main><script>
const KEY='vidimg-acceptance-verdict';
function collect(){{const v={{}};document.querySelectorAll('.verdict').forEach(d=>{{const c=d.querySelector('input[type=radio]:checked');
 const n=d.querySelector('.note').value.trim(); if(c||n) v[d.dataset.id]={{verdict:c?c.value:null,note:n}};}});return v;}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(collect()));}}catch(e){{}} count();}}
function count(){{const all=document.querySelectorAll('.verdict').length, v=collect();
 const done=Object.values(v).filter(x=>x.verdict).length; document.getElementById('count').textContent=`${{done}} of ${{all}} judged`;}}
try{{const s=JSON.parse(localStorage.getItem(KEY)||'{{}}');document.querySelectorAll('.verdict').forEach(d=>{{const x=s[d.dataset.id];if(!x)return;
 if(x.verdict){{const r=d.querySelector(`input[value="${{x.verdict}}"]`);if(r)r.checked=true;}} d.querySelector('.note').value=x.note||'';}});}}catch(e){{}}
document.addEventListener('change',save);document.addEventListener('input',save);count();
document.getElementById('copy').onclick=async()=>{{const t=JSON.stringify({{acceptance_verdict:collect()}},null,1);
 const o=document.getElementById('out');o.value=t;o.style.display='block';
 try{{await navigator.clipboard.writeText(t);document.getElementById('copy').textContent='Copied ✓';}}catch(e){{o.select();}}}};
</script></body></html>"""
    out = ROOT / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
