"""Phase 6 acceptance report: one local HTML page (Arabic, right-to-left) with every delivered photo and
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
from .compare import GATE_AR, VIDEO_GATE_AR

ROOT = config.WORKSPACE / "acceptance"
PHOTO_RUN = ROOT / "photos"
VIDEO_RECORDS = {
    # free tools only (2026-11-16); earlier videos made with the removed paid engine are not shown
    "فيديو واتساب (دقيقتان، 352p ← 704p، نموذج مفتوح)": config.WORKSPACE / "acceptance" / "video_whatsapp_free3" / "record.json",
    "مقطع كرتوني للاختبار (Video2X للأنمي)": config.WORKSPACE / "bench_routing" / "out_video" / "record.json",
}
STATUS = {"pass": "✅", "warn": "⚠️", "fail": "⛔", "n/a": "—"}


def _asset(src: str | Path, name: str) -> str | None:
    p = Path(src)
    if not p.exists():
        return None
    dst = ROOT / "assets" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dst)
    return f"assets/{html.escape(name)}"


def _gates_table(gates: dict, names: dict) -> str:
    rows = "".join(f"<tr><td>{STATUS.get(g.get('status'), '')}</td><td>{html.escape(names.get(k, k))}</td>"
                   f"<td dir='ltr'>{g.get('value')}</td><td dir='ltr' class='mut'>{g.get('threshold', '')}</td></tr>"
                   for k, g in gates.items())
    return f"<table class='gates'>{rows}</table>" if rows else ""


def _verdict(item_id: str) -> str:
    opts = (("same", "نفس الشخص والملامح"), ("changed", "تغيّر شيء في الوجه"), ("unsure", "غير متأكد"))
    radios = "".join(f"<label><input type='radio' name='{item_id}' value='{v}'> {t}</label>" for v, t in opts)
    return (f"<div class='verdict' data-id='{html.escape(item_id)}'>{radios}"
            f"<input type='text' class='note' placeholder='ملاحظة (اختياري)'></div>")


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
                cards.append(f"<section class='card kept'><h3 dir='ltr'>{html.escape(name)}</h3>"
                             f"<p>⛔ لم يُسلَّم: لم يجتز أي مرشح الحارس، فبقي الأصل كما هو.</p></section>")
            continue
        stats["delivered"] += 1
        atts = [a for a in r["attempts"] if a["passed"]]
        chosen = next((a for a in atts if not a.get("warnings")), atts[0] if atts else None)
        if chosen and chosen.get("warnings"):
            stats["warned"] += 1
        sheet = PHOTO_RUN / "sheets" / f"{Path(name).stem}.jpg"
        img = _asset(sheet, f"photo_{Path(name).stem}.jpg")
        warn = (f"<p class='warn'>⚠️ سُلّم مع تنبيه: {html.escape('، '.join(GATE_AR.get(w, w) for w in chosen['warnings']))}</p>"
                if chosen and chosen.get("warnings") else "")
        cards.append(f"""<section class='card'><h3 dir='ltr'>{html.escape(name)}</h3>
<p class='mut'>النموذج: <b dir='ltr'>{html.escape(chosen['label'] if chosen else '?')}</b> · الخطة: <span dir='ltr'>{html.escape(r['plan'])}</span></p>{warn}
{f"<img src='{img}' loading='lazy' alt='قبل وبعد'>" if img else ''}
<details><summary>بوابات الحارس</summary>{_gates_table(chosen['gates'] if chosen else {}, GATE_AR)}</details>
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
        imgs = "".join(f"<img src='{a}' loading='lazy' alt='إطار'>" for i, s in enumerate(rec.get("sheets") or [])
                       if (a := _asset(s, f"video_{path.parent.name}_{i}.jpg")))
        pre = rec.get("prepass") or {}
        pre_txt = f" · المعالجة الأولى: <span dir='ltr'>{html.escape(str(pre.get('plan')))}</span>" if pre.get("plan") else ""
        cards.append(f"""<section class='card'><h3>{html.escape(title)}</h3>
<p class='mut'>النموذج: <b dir='ltr'>{html.escape(str(rec.get('chosen')))}</b>{pre_txt}</p>
{imgs}<details open><summary>بوابات الحارس النهائية</summary>{_gates_table(rec.get('final_gates') or {}, VIDEO_GATE_AR)}</details>
{_verdict('video:' + title)}</section>""")
    return "".join(cards), n


def _evidence() -> str:
    rows = []
    sf = config.WORKSPACE / "calibration" / "20261114-235009" / "summary_free.json"
    if sf.exists():
        s = json.loads(sf.read_text(encoding="utf-8"))
        names = {"free_bsrnet_x4": "مجاني BSRNet (الافتراضي)", "free_realesrnet_x4": "مجاني RealESRNet",
                 "free_swinir_real_x4": "مجاني SwinIR", "free_lanczos": "تكبير تقليدي بلا ذكاء"}
        rows = "".join(f"<tr><td>{n}</td><td dir='ltr'>{s[k]['gt_identity']}</td><td dir='ltr'>{s[k]['psnr_vs_bicubic']:+}</td></tr>"
                       for k, n in names.items() if k in s)
    vf = config.WORKSPACE / "bench_free" / "video" / "summary.json"
    vrows = ""
    if vf.exists():
        v = json.loads(vf.read_text(encoding="utf-8"))
        vn = {"free_realesrnet_x4": "مجاني RealESRNet",
              "free_bsrnet_x4": "مجاني BSRNet", "free_general_x4v3": "مجاني general-v3 (سريع)"}
        vrows = "".join(f"<tr><td>{n}</td><td dir='ltr'>{v[k]['identity']}</td><td dir='ltr'>{v[k]['vmaf']}</td><td dir='ltr'>{v[k]['psnr']}</td></tr>"
                        for k, n in vn.items() if k in v)
    return f"""<section class='card'><h2>لماذا هذه المحركات؟ (مقابل الأصل الحقيقي لصورك ومقاطعك)</h2>
<h3>الصور (12 صورة، تدهور مضبوط)</h3><table class='ev'><tr><th>المحرك</th><th>تطابق الهوية مع الأصل</th><th>الكسب عن التكبير البسيط (dB)</th></tr>{rows}</table>
<h3>الفيديو (12 مقطعاً)</h3><table class='ev'><tr><th>المحرك</th><th>تطابق الهوية</th><th>VMAF</th><th>PSNR</th></tr>{vrows}</table>
<p class='mut'>الأعلى أفضل في كل الأعمدة. التفاصيل: PLAN.md وREQUIREMENTS.md.</p></section>"""


def build() -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    photos, st = _photo_cards()
    videos, nv = _video_cards()
    eligible = st["total"] - st["skipped"]
    page = f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>مراجعة القبول</title><style>
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
<h1>مراجعة القبول (المرحلة 6)</h1>
<p>المعيار: <b>لا تغيّر في هوية أي شخص</b>. أنت تعرف الأشخاص في الصور، فحكمك هو الفيصل. في كل بطاقة: الأصل مكبَّراً (يسار) والناتج (يمين)، ثم الوجوه مكبَّرة قبل/بعد. اختر لكل عنصر حكماً، ثم اضغط «نسخ الحكم» والصقه في المحادثة.</p>
<div class='cards'>
<div class='stat'><b>{st['delivered']} / {eligible}</b>صور سُلّمت (من القابلة للتكبير)</div>
<div class='stat'><b>{st['warned']}</b>منها مع تنبيه من الحارس</div>
<div class='stat'><b>{st['kept']}</b>بقي أصلها (رفضها الحارس)</div>
<div class='stat'><b>{st['skipped']}</b>كبيرة أصلاً فلم تُعالَج</div>
<div class='stat'><b>{nv}</b>فيديوهات سُلّمت</div>
<div class='stat'><b>أدوات مجانية فقط</b>بلا أي برنامج مدفوع</div>
</div>
{_evidence()}
<h2>الفيديو</h2>{videos}
<h2>الصور</h2>{photos}
<div class='bar'><button id='copy'>نسخ الحكم</button><span id='count' class='mut'></span><textarea id='out' readonly></textarea></div>
</main><script>
const KEY='vidimg-acceptance-verdict';
function collect(){{const v={{}};document.querySelectorAll('.verdict').forEach(d=>{{const c=d.querySelector('input[type=radio]:checked');
 const n=d.querySelector('.note').value.trim(); if(c||n) v[d.dataset.id]={{verdict:c?c.value:null,note:n}};}});return v;}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(collect()));}}catch(e){{}} count();}}
function count(){{const all=document.querySelectorAll('.verdict').length, v=collect();
 const done=Object.values(v).filter(x=>x.verdict).length; document.getElementById('count').textContent=`حُكم على ${{done}} من ${{all}}`;}}
try{{const s=JSON.parse(localStorage.getItem(KEY)||'{{}}');document.querySelectorAll('.verdict').forEach(d=>{{const x=s[d.dataset.id];if(!x)return;
 if(x.verdict){{const r=d.querySelector(`input[value="${{x.verdict}}"]`);if(r)r.checked=true;}} d.querySelector('.note').value=x.note||'';}});}}catch(e){{}}
document.addEventListener('change',save);document.addEventListener('input',save);count();
document.getElementById('copy').onclick=async()=>{{const t=JSON.stringify({{acceptance_verdict:collect()}},null,1);
 const o=document.getElementById('out');o.value=t;o.style.display='block';
 try{{await navigator.clipboard.writeText(t);document.getElementById('copy').textContent='نُسخ ✓';}}catch(e){{o.select();}}}};
</script></body></html>"""
    out = ROOT / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
