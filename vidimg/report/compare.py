"""Before/after comparison sheets and a local review page (nothing leaves the computer)."""

from __future__ import annotations

import html
from pathlib import Path

import cv2
import numpy as np

from .. import faces, guard

SHEET_WIDTH = 1800
FACE_TILE = 300


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(out, text, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _fit(img: np.ndarray, w: int) -> np.ndarray:
    k = w / img.shape[1]
    return cv2.resize(img, (w, max(1, round(img.shape[0] * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)


def sheet(original: Path, restored: Path, out: Path, title: str = "") -> Path:
    src = cv2.imread(str(original))
    res = cv2.imread(str(restored))
    h, w = res.shape[:2]
    big = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC)   # fair "before": same size, bicubic
    half = SHEET_WIDTH // 2
    top = np.hstack([_label(_fit(big, half), "Original (enlarged)"), _label(_fit(res, half), "Restored")])
    rows = [top]
    found = faces.detect(big, score=guard.FACE_SCORE)
    found = sorted(found, key=lambda f: -f.box[2] * f.box[3])[:3]
    tiles = []
    for f in found:
        x, y, fw, fh = (int(v) for v in f.box)
        pad = int(0.35 * max(fw, fh))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + fw + pad), min(h, y + fh + pad)
        a = cv2.resize(big[y0:y1, x0:x1], (FACE_TILE, FACE_TILE), interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(res[y0:y1, x0:x1], (FACE_TILE, FACE_TILE), interpolation=cv2.INTER_CUBIC)
        tiles.append(np.hstack([_label(a, "before"), _label(b, "after"), np.full((FACE_TILE, 12, 3), 255, np.uint8)]))
    if tiles:
        row = np.hstack(tiles)
        if row.shape[1] < SHEET_WIDTH:
            row = np.hstack([row, np.full((FACE_TILE, SHEET_WIDTH - row.shape[1], 3), 255, np.uint8)])
        rows.append(np.full((12, SHEET_WIDTH, 3), 255, np.uint8))
        rows.append(row[:, :SHEET_WIDTH])
    img = np.vstack(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


GATE_AR = {"identity": "تطابق الهوية", "landmarks": "ملامح الوجه", "face_backprojection": "بنية الوجه",
           "backprojection": "البنية العامة", "color": "الألوان", "hallucination": "تفاصيل مخترعة",
           "eye_color": "لون العينين"}


def review_page(records: list[dict], sheets: dict[str, Path], out: Path) -> Path:
    """Local HTML page (right-to-left Arabic) with every sheet and the guard's decision."""
    cards = []
    for r in records:
        name = Path(r["source"]).name
        def verdict(a: dict) -> str:
            if not a["passed"]:
                return "رُفض: " + "، ".join(GATE_AR.get(g, g) for g in a["failed_gates"]) if a["failed_gates"] else "خطأ"
            warns = a.get("warnings") or []
            return "نجح" + (" مع تنبيه: " + "، ".join(GATE_AR.get(g, g) for g in warns) if warns else "")
        tried = " ← ".join(f"{a['label']} ({verdict(a)})" for a in r["attempts"]) or "—"
        status = "✅ اجتاز الحارس" if r["passed"] else "⛔ لم يجتز الحارس: الأصل يبقى كما هو"
        last = next((a for a in reversed(r["attempts"]) if a.get("gates")), None)
        idv = last["gates"].get("identity", {}).get("value") if last else None
        img = f'<img src="{html.escape(sheets[name].name)}" loading="lazy">' if name in sheets else ""
        cards.append(f"""<section><h2>{html.escape(name)}</h2><p class="st">{status}</p>
<p>الخطة: <span dir="ltr">{html.escape(r['plan'])}</span></p><p>المحاولات: <span dir="ltr">{html.escape(tried)}</span></p>
<p>تطابق الهوية (أسوأ وجه): <b dir="ltr">{idv if idv is not None else '—'}</b></p>{img}</section>""")
    passed = sum(r["passed"] for r in records)
    page = f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>مراجعة الاستعادة</title><style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--card:#fff;--line:#dfe3ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171c;--fg:#e8ebf0;--card:#1d2128;--line:#2c323c}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,"Segoe UI",Tahoma,sans-serif;margin:0;padding:16px;line-height:1.6}}
main{{max-width:1840px;margin:auto}} section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}}
img{{width:100%;height:auto;border-radius:6px}} h2{{margin:0 0 4px;font-size:1.05rem}} .st{{font-weight:600}}</style></head>
<body><main><h1>مراجعة استعادة الصور</h1>
<p>{passed} من {len(records)} صورة اجتازت حارس الأمانة. في كل بطاقة: الأصل مكبَّراً (يسار) والناتج (يمين)، ثم الوجوه مكبَّرة: قبل / بعد.
احكم أنت: هل بقيت الملامح نفسها؟</p>{''.join(cards)}</main></body></html>"""
    out.write_text(page, encoding="utf-8")
    return out


VIDEO_GATE_AR = {**GATE_AR, "frames": "عدد الإطارات", "duration": "المدة", "audio": "الصوت", "flicker": "الوميض",
                 "vmaf": "VMAF (الجودة المدركة)"}
STATUS_AR = {"pass": "✅", "warn": "⚠️", "fail": "⛔", "n/a": "—"}


def video_page(rec: dict, out: Path) -> Path:
    """Arabic (right-to-left) report for one video run: plan, classical pre-pass, previews, final gates, sheets."""
    pre = rec.get("prepass") or {}
    rows = "".join(
        f"<tr><td>{STATUS_AR.get(g.get('status'), '')}</td><td>{html.escape(VIDEO_GATE_AR.get(k, k))}</td>"
        f"<td dir='ltr'>{g.get('value')}</td><td dir='ltr'>{g.get('threshold', '')}</td></tr>"
        for k, g in (rec.get("final_gates") or {}).items())
    atts = "".join(
        f"<li dir='ltr'>{html.escape(a['label'])}: {a['passed_previews']}/{a['previews']} previews · VMAF {a.get('vmaf')}"
        f"{' · ' + html.escape('; '.join(','.join(f) for f in a['failed'] if f)) if any(a['failed']) else ''}</li>"
        for a in rec.get("attempts") or [])
    imgs = "".join(f'<img src="{html.escape(Path(s).relative_to(out.parent).as_posix() if Path(s).is_relative_to(out.parent) else Path(s).as_uri())}" loading="lazy">'
                   for s in rec.get("sheets") or [])
    if rec.get("passed"):
        status = "✅ اجتاز الحارس وسُلّم"
    elif rec.get("chosen") and not rec.get("final_gates"):
        status = "🔎 معاينة فقط: لم يبدأ التشغيل الكامل"
    else:
        status = "⛔ لم يُسلَّم: الأصل يبقى كما هو"
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in rec.get("notes") or [])
    pre_html = (f"<p>المعالجة الأولى: <span dir='ltr'>{html.escape(str(pre.get('plan')))}</span> · المحرك: "
                f"<b dir='ltr'>{html.escape(str(pre.get('engine')))}</b> · الحارس: "
                f"{'✅' if pre.get('guard_passed') else '⛔'}</p>") if pre.get("plan") else ""
    page = f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>تقرير الفيديو</title><style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--card:#fff;--line:#dfe3ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171c;--fg:#e8ebf0;--card:#1d2128;--line:#2c323c}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,"Segoe UI",Tahoma,sans-serif;margin:0;padding:16px;line-height:1.6}}
main{{max-width:1840px;margin:auto}} section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}}
table{{border-collapse:collapse}} td{{border-bottom:1px solid var(--line);padding:4px 10px}}
img{{width:100%;height:auto;border-radius:6px;margin:8px 0}}</style></head>
<body><main><h1>تقرير استعادة الفيديو: <span dir="ltr">{html.escape(Path(rec.get('source', '')).name)}</span></h1>
<section><p><b>{status}</b></p><p>الخطة: <span dir="ltr">{html.escape(rec.get('plan') or '')}</span></p>{pre_html}
<p>النموذج المختار: <b dir="ltr">{html.escape(str(rec.get('chosen')))}</b></p>
{f'<p>الناتج: <span dir="ltr">{html.escape(rec["delivered"])}</span></p>' if rec.get('delivered') else ''}
<ul>{atts}</ul>{f'<ul>{notes}</ul>' if notes else ''}</section>
{f'<section><h2>بوابات الحارس النهائية</h2><table>{rows}</table></section>' if rows else ''}
{f'<section><h2>إطارات للمقارنة (قبل / بعد)</h2>{imgs}</section>' if imgs else ''}</main></body></html>"""
    out.write_text(page, encoding="utf-8")
    return out
