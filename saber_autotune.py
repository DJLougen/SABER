#!/usr/bin/env python3
"""SABER dial-in optimizer utilities.

Reads completed SABER run directories, builds a refusal/KLD frontier, detects
proxy failures, and proposes the next local search points. This is intentionally
lightweight: it works from JSON artifacts and does not require Optuna.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

MODEL_SLUGS = {
    "google/gemma-4-E4B-it": "gemma4_e4b",
    "google/gemma-4-E2B-it": "gemma4_e2b",
    "Qwen/Qwen3.5-4B": "qwen_qwen3_5-4b",
    "Qwen/Qwen3.5-9B": "qwen_qwen3_5-9b",
}

BASE_BY_SLUG = {v: k for k, v in MODEL_SLUGS.items()}


@dataclass
class RunRow:
    run_dir: str
    name: str
    model: str | None
    alpha_base: float | None
    alpha_entangled: float | None
    extraction_method: str | None
    n_directions: int | None
    global_top_k: int | None
    layer_top_k: int | None
    entanglement_threshold: float | None
    max_iterations: int | None
    residual: float | None
    mean_kld: float | None
    max_kld: float | None
    refusal_kw: float | None
    refusal_judge: float | None
    proxy_mismatch: float
    score: float
    complete: bool
    pareto: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _int(v: Any) -> int | None:
    x = _num(v)
    return None if x is None else int(x)


def _parse_global_top_k(name: str, cfg: dict[str, Any]) -> int | None:
    for key in ("global_top_k", "global_topk"):
        if key in cfg:
            return _int(cfg.get(key))
    m = re.search(r"(?:^|_)g(\d+)(?:_|$)", name)
    return int(m.group(1)) if m else None


def _refusal_metric(refusal: dict[str, Any], saber: dict[str, Any]) -> float | None:
    for key in ("refusal_rate_judge", "refusal_rate_kw"):
        x = _num(refusal.get(key))
        if x is not None:
            return x
    return _num(saber.get("post_refusal_rate", saber.get("post_refusal")))


def _score(refusal: float | None, kld: float | None, residual: float | None, rows_for_model: list[dict[str, Any]]) -> tuple[float, float]:
    # Primary: real generated refusal. Secondary: KLD. Tertiary: residual proxy.
    r = 1.0 if refusal is None else refusal
    k = 10.0 if kld is None else kld
    res = 0.0 if residual is None else residual

    residuals = [_num(x.get("residual")) for x in rows_for_model]
    residuals = [x for x in residuals if x is not None]
    if residuals and residual is not None:
        lo, hi = min(residuals), max(residuals)
        norm_resid = 0.0 if hi <= lo else (residual - lo) / (hi - lo)
    else:
        norm_resid = 0.5

    # Mismatch catches Qwen-like cases: proxy looks low but generated refusals stay high.
    mismatch = max(0.0, r - (1.0 - norm_resid))
    return r + 0.12 * math.log1p(max(k, 0.0)) + 0.08 * mismatch, mismatch


def collect_runs(root: Path) -> list[RunRow]:
    raw: list[dict[str, Any]] = []
    for saber_path in sorted(root.rglob("saber_result.json")):
        run_dir = saber_path.parent
        saber = _read_json(saber_path)
        kld = _read_json(run_dir / "kld_result.json")
        refusal = _read_json(run_dir / "refusal_result.json")
        cfg = saber.get("config") or {}
        raw.append({
            "run_dir": str(run_dir),
            "name": run_dir.name,
            "model": saber.get("model"),
            "alpha_base": _num(cfg.get("alpha_base")),
            "alpha_entangled": _num(cfg.get("alpha_entangled")),
            "extraction_method": cfg.get("extraction_method"),
            "n_directions": _int(cfg.get("n_directions")),
            "global_top_k": _parse_global_top_k(run_dir.name, cfg),
            "layer_top_k": _int(cfg.get("layer_top_k")),
            "entanglement_threshold": _num(cfg.get("entanglement_threshold")),
            "max_iterations": _int(cfg.get("max_iterations")),
            "residual": _num(saber.get("final_residual_refusal")),
            "mean_kld": _num(kld.get("mean_kld")),
            "max_kld": _num(kld.get("max_kld")),
            "refusal_kw": _num(refusal.get("refusal_rate_kw")),
            "refusal_judge": _num(refusal.get("refusal_rate_judge")),
            "complete": bool(kld and refusal),
            "_refusal": _refusal_metric(refusal, saber),
        })

    rows: list[RunRow] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for item in raw:
        by_model.setdefault(str(item.get("model")), []).append(item)
    for item in raw:
        score, mismatch = _score(item.get("_refusal"), item.get("mean_kld"), item.get("residual"), by_model[str(item.get("model"))])
        rows.append(RunRow(
            run_dir=item["run_dir"], name=item["name"], model=item.get("model"),
            alpha_base=item.get("alpha_base"), alpha_entangled=item.get("alpha_entangled"),
            extraction_method=item.get("extraction_method"), n_directions=item.get("n_directions"),
            global_top_k=item.get("global_top_k"), layer_top_k=item.get("layer_top_k"),
            entanglement_threshold=item.get("entanglement_threshold"), max_iterations=item.get("max_iterations"),
            residual=item.get("residual"), mean_kld=item.get("mean_kld"), max_kld=item.get("max_kld"),
            refusal_kw=item.get("refusal_kw"), refusal_judge=item.get("refusal_judge"),
            proxy_mismatch=mismatch, score=score, complete=item.get("complete"),
        ))

    for model in sorted({r.model for r in rows}):
        group = [r for r in rows if r.model == model and r.complete and r.refusal_kw is not None and r.mean_kld is not None]
        for r in group:
            r.pareto = not any(
                (o is not r)
                and (o.refusal_kw <= r.refusal_kw)
                and (o.mean_kld <= r.mean_kld)
                and ((o.refusal_kw < r.refusal_kw) or (o.mean_kld < r.mean_kld))
                for o in group
            )
    return rows


def best_rows(rows: Iterable[RunRow], model: str | None, complete_only: bool = True) -> list[RunRow]:
    out = [r for r in rows if (model is None or r.model == model)]
    if complete_only:
        out = [r for r in out if r.complete and r.refusal_kw is not None and r.mean_kld is not None]
    return sorted(out, key=lambda r: (r.score, r.refusal_kw if r.refusal_kw is not None else 1.0, r.mean_kld if r.mean_kld is not None else 99.0))


def propose(rows: list[RunRow], model: str, limit: int = 4) -> list[dict[str, Any]]:
    ranked = best_rows(rows, model, complete_only=True)
    if not ranked:
        return []
    seeds = ranked[:3]
    existing = {r.name for r in rows}
    proposals: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for seed in seeds:
        alpha0 = seed.alpha_base if seed.alpha_base is not None else 0.8
        g0 = seed.global_top_k if seed.global_top_k is not None else 10
        # Local search around actual winners, biased toward lower KLD if refusal ties.
        for da, dg in [(-0.05, 0), (0.05, 0), (0.0, -2), (0.0, 2), (-0.025, 2), (0.025, -2)]:
            alpha = round(min(1.2, max(0.55, alpha0 + da)), 3)
            g = int(min(16, max(4, g0 + dg)))
            key = (alpha, g, seed.extraction_method or "svd", seed.n_directions or 4)
            if key in seen:
                continue
            seen.add(key)
            slug = MODEL_SLUGS.get(model, re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower())
            tag = f"{slug}_auto_svd_a{int(round(alpha*1000)):03d}_g{g:02d}"
            if tag in existing:
                continue
            cfg = {
                "extraction_method": seed.extraction_method or "svd",
                "n_directions": seed.n_directions or 4,
                "layer_selection_strategy": "top_k",
                "layer_top_k": seed.layer_top_k or 12,
                "global_top_k": g,
                "alpha_base": alpha,
                "alpha_entangled": seed.alpha_entangled if seed.alpha_entangled is not None else 0.03,
                "entanglement_threshold": seed.entanglement_threshold if seed.entanglement_threshold is not None else 0.55,
                "max_iterations": seed.max_iterations or 4,
            }
            proposals.append({
                "tag": tag,
                "base_model": model,
                "output_dir": f"runs/{tag}",
                "seed_run": seed.name,
                "seed_score": seed.score,
                "config": cfg,
            })
            if len(proposals) >= limit:
                return proposals
    return proposals


def _print_table(rows: list[RunRow], model: str | None, limit: int) -> None:
    ranked = best_rows(rows, model, complete_only=True)[:limit]
    print("rank\tscore\tref_kw\tkld\tresid\tproxy_gap\tpareto\trun")
    for i, r in enumerate(ranked, 1):
        print(f"{i}\t{r.score:.4f}\t{r.refusal_kw:.4f}\t{r.mean_kld:.4f}\t{(r.residual or 0):.4f}\t{r.proxy_mismatch:.4f}\t{int(r.pareto)}\t{r.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--propose", type=int, default=0)
    args = ap.parse_args()

    rows = collect_runs(Path(args.root))
    if args.propose:
        payload = propose(rows, args.model, args.propose)
        print(json.dumps(payload, indent=2))
        return
    if args.json:
        print(json.dumps([asdict(r) for r in best_rows(rows, args.model)[:args.limit]], indent=2))
    else:
        _print_table(rows, args.model, args.limit)


if __name__ == "__main__":
    main()
