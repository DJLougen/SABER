#!/usr/bin/env python3
from __future__ import annotations
import json, html
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "results" / "saber_browser_report.html"

def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def score(ref, kld):
    if ref is None or kld is None:
        return None
    return float(ref) + 0.1 * float(kld)

rows = []
for run in sorted(RUNS.iterdir()) if RUNS.exists() else []:
    if not run.is_dir():
        continue
    saber = read_json(run / "saber_result.json") or {}
    kld = read_json(run / "kld_result.json") or {}
    refusal = read_json(run / "refusal_result.json") or {}
    if not (saber or kld or refusal):
        continue
    ref_rate = refusal.get("refusal_rate_kw")
    mean_kld = kld.get("mean_kld")
    cfg = saber.get("config", {})
    rows.append({
        "run": run.name,
        "refusal": ref_rate,
        "kld": mean_kld,
        "score": score(ref_rate, mean_kld),
        "residual": saber.get("final_residual_refusal"),
        "layers": saber.get("selected_layers", []),
        "alpha": cfg.get("alpha_base"),
        "g": cfg.get("global_top_k"),
        "iters": cfg.get("max_iterations"),
        "complete": ref_rate is not None and mean_kld is not None,
    })

complete = [r for r in rows if r["complete"]]
for r in complete:
    r["pareto"] = not any(
        o is not r and o["refusal"] <= r["refusal"] and o["kld"] <= r["kld"] and (o["refusal"] < r["refusal"] or o["kld"] < r["kld"])
        for o in complete
    )
for r in rows:
    r.setdefault("pareto", False)
rows_sorted = sorted(rows, key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 999, r["run"]))
frontier = [r for r in rows_sorted if r.get("pareto")]

def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100*float(v):.2f}%" if pct else f"{float(v):.4f}"

def tr(r):
    cls = "pareto" if r.get("pareto") else ""
    return f"<tr class='{cls}'><td>{html.escape(r['run'])}</td><td>{fmt(r['score'])}</td><td>{fmt(r['refusal'], True)}</td><td>{fmt(r['kld'])}</td><td>{fmt(r['residual'])}</td><td>{r.get('g') or '-'}</td><td>{r.get('alpha') or '-'}</td><td>{r.get('iters') or '-'}</td><td>{len(r.get('layers') or [])}</td><td>{html.escape(str(r.get('layers') or []))}</td></tr>"

points = [r for r in complete if r.get("refusal") is not None and r.get("kld") is not None]
max_x = max([float(r["kld"]) for r in points] + [1.0])
max_y = max([float(r["refusal"]) for r in points] + [0.2])
svg_points = []
for r in points:
    x = 70 + 640 * float(r["kld"]) / max_x
    y = 360 - 300 * float(r["refusal"]) / max_y
    color = "#e14eca" if r.get("pareto") else "#70a5ff"
    rad = 7 if r.get("pareto") else 4
    label = html.escape(r["run"].replace("gemma4_e4b_auto_svd_", ""))
    svg_points.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{rad}' fill='{color}'><title>{label}: ref={fmt(r['refusal'], True)}, kld={fmt(r['kld'])}</title></circle>")

html_doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>SABER Frontier Report</title><style>
:root {{ color-scheme: dark; --bg:#111318; --panel:#1a1d25; --text:#e8eaf0; --muted:#a8afc0; --accent:#e14eca; --blue:#70a5ff; --line:#303645; }}
body {{ margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }}
header {{ padding:28px 36px 18px; border-bottom:1px solid var(--line); }} h1 {{ margin:0 0 8px; font-size:28px; }} p {{ color:var(--muted); line-height:1.5; max-width:980px; }}
main {{ padding:24px 36px 40px; display:grid; gap:22px; }} section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:14px; }} .metric {{ border:1px solid var(--line); border-radius:8px; padding:14px; }} .metric b {{ display:block; font-size:22px; margin-top:6px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ color:#ffffff; background:#202531; }} tr.pareto {{ background:rgba(225,78,202,0.10); }}
svg {{ width:100%; max-width:820px; height:auto; background:#111722; border:1px solid var(--line); border-radius:8px; }} .small {{ font-size:12px; color:var(--muted); }}
</style></head><body>
<header><h1>SABER Frontier Report</h1><p>SABER is framed here as an entanglement-aware refusal-ablation recipe: separability-ranked multi-direction refusal subspaces, capability-entanglement scoring, differential ablation strengths, and Pareto evaluation over refusal rate and KLD drift.</p></header>
<main><section><h2>Current Write-Up Points</h2><div class='grid'><div class='metric'>Aggressive point<b>0.00% refusal</b><span class='small'>nh60_a825_g14, KLD 0.4327</span></div><div class='metric'>Balanced point<b>2.04% refusal</b><span class='small'>a825_g14, KLD 0.3164</span></div><div class='metric'>Completed runs<b>{len(complete)}</b><span class='small'>{len(frontier)} Pareto frontier points</span></div></div></section>
<section><h2>Refusal / KLD Scatter</h2><svg viewBox='0 0 760 400'><line x1='70' y1='360' x2='720' y2='360' stroke='#566078'/><line x1='70' y1='40' x2='70' y2='360' stroke='#566078'/><text x='350' y='392' fill='#a8afc0'>KLD drift</text><text x='8' y='34' fill='#a8afc0'>refusal</text>{''.join(svg_points)}</svg><p class='small'>Pink points are Pareto frontier runs. Hover points for run names.</p></section>
<section><h2>Run Table</h2><table><thead><tr><th>run</th><th>score</th><th>refusal</th><th>KLD</th><th>residual</th><th>g</th><th>alpha</th><th>iters</th><th>n layers</th><th>layers</th></tr></thead><tbody>{''.join(tr(r) for r in rows_sorted)}</tbody></table></section>
<section><h2>Prior-Art Boundary</h2><p>I did not invent refusal ablation. SABER is my take on abliteration, inspired by prior work including Arditi et al., Labonne/FailSpy-style abliteration, Jim Lai's projected and norm-preserving variants, Pliny/OBLITERATUS, Heretic, Jiunsong/SuperGemma, and spectral-cleaning/SRA-style work. The claim is the combination of separability ranking, entanglement-aware scaling, and refusal/KLD frontier mapping.</p></section>
</main></body></html>"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html_doc)
print(f"wrote {OUT}")
print(f"runs={len(rows)} complete={len(complete)} pareto={len(frontier)}")
