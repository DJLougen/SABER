"""
SABER Lab Backend — single-file FastAPI service that wraps:
  - saber.SABER  (the existing core algorithm)
  - saber_eval_kld.KLDEvaluator  (capability preservation)
  - saber_eval_refusal.RefusalEvaluator  (real-gen refusal scoring)

Design:
  - Sequential GPU access (one heavy job at a time), enforced by a global lock.
  - Jobs run in background threads; status + results polled via REST.
  - SQLite at /workspace/saber-lab/runs.db persists run history across restarts.
  - Started on port 8765 by default; intended to be reached via SSH tunnel.

Run with:
    cd /workspace/saber-lab && source /workspace/venv-mamba3/bin/activate
    python saber_lab_server.py
or:
    uvicorn saber_lab_server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SABER_LAB_DIR = Path(os.environ.get("SABER_LAB_DIR", "/workspace/saber-lab"))
DB_PATH = SABER_LAB_DIR / "runs.db"
sys.path.insert(0, str(SABER_LAB_DIR))

# Lazy imports so the server can boot even if heavy deps aren't all there yet
def _import_saber():
    from saber import SABER, SABERConfig
    return SABER, SABERConfig

def _import_kld():
    from saber_eval_kld import KLDEvaluator, default_calibration_set
    return KLDEvaluator, default_calibration_set

def _import_refusal():
    from saber_eval_refusal import RefusalEvaluator
    return RefusalEvaluator

def _import_harmful_prompts():
    from config import HARMFUL_PROMPTS
    return HARMFUL_PROMPTS


def _import_autotune():
    from saber_autotune import collect_runs, best_rows, propose
    return collect_runs, best_rows, propose


def _import_run_helpers():
    from run_saber import load_model, format_prompts, evaluate_refusal_rate, evaluate_capability
    from config import HARMFUL_PROMPTS, HARMLESS_PROMPTS, CAPABILITY_PROMPT_CATEGORIES, get_capability_prompts
    return load_model, format_prompts, evaluate_refusal_rate, evaluate_capability, HARMFUL_PROMPTS, HARMLESS_PROMPTS, CAPABILITY_PROMPT_CATEGORIES, get_capability_prompts


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _read_json_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None



# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

GPU_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _new_job(kind: str, params: dict) -> str:
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "params": params,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "log": [],
            "result": None,
            "error": None,
        }
    _persist(job_id)
    return job_id


def _job_log(job_id: str, msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["log"].append(line)
    print(f"[{job_id[:8]}] {msg}", flush=True)


def _job_set(job_id: str, **kw):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kw)
    _persist(job_id)


def _persist(job_id: str):
    """Best-effort SQLite persistence."""
    with JOBS_LOCK:
        snap = JOBS.get(job_id)
        if not snap:
            return
        try:
            with _db() as cx:
                cx.execute(
                    "INSERT OR REPLACE INTO jobs (id, kind, status, params, log, result, error, "
                    "created_at, started_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        snap["id"], snap["kind"], snap["status"],
                        json.dumps(snap["params"]),
                        json.dumps(snap["log"]),
                        json.dumps(snap["result"]) if snap["result"] is not None else None,
                        snap["error"],
                        snap["created_at"], snap["started_at"], snap["finished_at"],
                    ),
                )
                cx.commit()
        except Exception as e:
            print(f"[persist] {e}", flush=True)


@contextmanager
def _db():
    SABER_LAB_DIR.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(DB_PATH)
    try:
        yield cx
    finally:
        cx.close()


def _ensure_schema():
    with _db() as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              status TEXT NOT NULL,
              params TEXT,
              log TEXT,
              result TEXT,
              error TEXT,
              created_at REAL,
              started_at REAL,
              finished_at REAL
            )
        """)
        cx.commit()


def _spawn(job_id: str, target, *args, **kwargs):
    def _run():
        with GPU_LOCK:
            _job_log(job_id, "GPU lock acquired")
            _job_set(job_id, status="running", started_at=time.time())
            try:
                result = target(job_id, *args, **kwargs)
                _job_set(job_id, status="done", result=result, finished_at=time.time())
                _job_log(job_id, "done")
            except Exception:
                tb = traceback.format_exc()
                _job_log(job_id, f"ERROR\n{tb}")
                _job_set(job_id, status="error", error=tb, finished_at=time.time())
    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _do_saber_run(job_id: str, base_model: str, saber_kwargs: dict, output_dir: str) -> dict:
    SABER, SABERConfig = _import_saber()
    (load_model, format_prompts, evaluate_refusal_rate, evaluate_capability,
     HARMFUL_PROMPTS, HARMLESS_PROMPTS, CAPABILITY_PROMPT_CATEGORIES,
     get_capability_prompts) = _import_run_helpers()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(Path(output_dir) / "saber.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    dtype = saber_kwargs.pop("dtype", "bfloat16")
    device = saber_kwargs.pop("device", "auto")
    n_harmful = int(saber_kwargs.pop("n_harmful", 30))
    n_harmless = int(saber_kwargs.pop("n_harmless", 30))
    n_capability = int(saber_kwargs.pop("n_capability", 30))
    eval_before = bool(saber_kwargs.pop("eval_before", False))
    eval_after = bool(saber_kwargs.pop("eval_after", True))
    save_model = bool(saber_kwargs.pop("save_model", True))
    quick = bool(saber_kwargs.pop("quick", False))

    _job_log(job_id, f"Loading model {base_model} dtype={dtype} device={device}")
    model, tokenizer = load_model(base_model, device, dtype)

    harmful = format_prompts(HARMFUL_PROMPTS[:n_harmful], tokenizer)
    harmless = format_prompts(HARMLESS_PROMPTS[:n_harmless], tokenizer)
    capability = None if quick else get_capability_prompts(
        n_per_category=max(1, n_capability // len(CAPABILITY_PROMPT_CATEGORIES))
    )
    capability_formatted = format_prompts(capability, tokenizer) if capability else None

    pre_refusal = pre_ppl = None
    if eval_before:
        _job_log(job_id, "Evaluating before ablation")
        pre_refusal = evaluate_refusal_rate(model, tokenizer, harmful, device)
        if capability_formatted:
            pre_ppl = evaluate_capability(model, tokenizer, capability_formatted, device)

    cfg = SABERConfig(**saber_kwargs)
    _job_log(job_id, f"Starting SABER alpha={cfg.alpha_base} entangled={cfg.alpha_entangled}")
    saber = SABER(model, tokenizer, cfg)
    start = time.time()
    result = saber.run(
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        capability_prompts=capability_formatted,
    )
    elapsed = time.time() - start

    post_refusal = post_ppl = None
    if eval_after:
        _job_log(job_id, "Evaluating after ablation")
        post_refusal = evaluate_refusal_rate(model, tokenizer, harmful, device)
        if capability_formatted:
            post_ppl = evaluate_capability(model, tokenizer, capability_formatted, device)

    summary = {
        "model": base_model,
        "output_dir": output_dir,
        "config": result.config,
        "selected_layers": result.selected_layers,
        "total_directions_ablated": len(result.all_refusal_directions),
        "iterations": len(result.ablation_history),
        "final_residual_refusal": result.final_residual_refusal,
        "final_capability_preservation": result.final_capability_preservation,
        "pre_refusal_rate": pre_refusal,
        "pre_perplexity": pre_ppl,
        "post_refusal_rate": post_refusal,
        "post_perplexity": post_ppl,
        "elapsed_seconds": elapsed,
        "layer_profiles": {
            str(l): {
                "fdr": p.fisher_discriminant_ratio,
                "n_directions": p.n_directions,
                "directions": [
                    {
                        "separability": d.separability,
                        "entanglement": d.entanglement,
                        "purity": d.purity,
                        "variance_explained": d.variance_explained,
                    }
                    for d in p.refusal_directions
                ],
            }
            for l, p in result.layer_profiles.items()
        },
        "ablation_history": [
            {
                "iteration": ar.iteration,
                "n_directions_ablated": len(ar.directions_ablated),
                "total_weight_norm_removed": ar.total_weight_norm_removed,
                "residual_refusal_score": ar.residual_refusal_score,
                "capability_degradation_estimate": ar.capability_degradation_estimate,
            }
            for ar in result.ablation_history
        ],
    }
    Path(output_dir, "saber_result.json").write_text(json.dumps(summary, indent=2))

    if save_model:
        model_path = Path(output_dir) / "saber_model"
        _job_log(job_id, f"Saving model to {model_path}")
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)

    return {k: _safe_json_value(v) for k, v in summary.items() if k not in ("layer_profiles", "ablation_history")}


def _do_saber_sweep(job_id: str, base_model: str, alpha_values: List[float], base_config: dict, output_root: str) -> dict:
    rows = []
    for alpha in alpha_values:
        cfg = dict(base_config)
        cfg["alpha_base"] = float(alpha)
        tag = str(alpha).replace(".", "")
        out_dir = str(Path(output_root) / f"a{tag}")
        _job_log(job_id, f"Sweep alpha={alpha} -> {out_dir}")
        result = _do_saber_run(job_id, base_model, cfg, out_dir)
        rows.append({
            "alpha_base": float(alpha),
            "output_dir": out_dir,
            "post_refusal_rate": result.get("post_refusal_rate"),
            "final_residual_refusal": result.get("final_residual_refusal"),
            "post_perplexity": result.get("post_perplexity"),
            "iterations": result.get("iterations"),
            "total_directions_ablated": result.get("total_directions_ablated"),
        })
    return {"base_model": base_model, "output_root": output_root, "runs": rows}


def _do_kld(job_id: str, base_model: str, ablated_path: str,
            cache_path: Optional[str], chat_template: bool,
            max_seq_len: int, batch_size: int) -> dict:
    KLDEvaluator, default_calibration_set = _import_kld()
    prompts, categories = default_calibration_set()
    _job_log(job_id, f"{len(prompts)} calibration prompts; base={base_model}")
    cache_path = cache_path or str(SABER_LAB_DIR / "cache" / "base_logits.pt")
    ev = KLDEvaluator(
        base_model_name=base_model, prompts=prompts, categories=categories,
        chat_template=chat_template, max_seq_len=max_seq_len,
    )
    if not os.path.exists(cache_path):
        _job_log(job_id, f"Precomputing base logits cache at {cache_path}")
        ev.precompute_base(cache_path, batch_size=batch_size)
    else:
        _job_log(job_id, f"Reusing base logits cache: {cache_path}")
    _job_log(job_id, f"Evaluating ablated: {ablated_path}")
    result = ev.evaluate(ablated_path, cache_path, batch_size=batch_size)
    return result.to_dict()


def _do_refusal(job_id: str, model_path: str, tokenizer_path: Optional[str],
                judge_model: Optional[str], judge_4bit: bool,
                max_new_tokens: int, n_prompts: int) -> dict:
    RefusalEvaluator = _import_refusal()
    HARMFUL_PROMPTS = _import_harmful_prompts()
    prompts = HARMFUL_PROMPTS if not n_prompts else HARMFUL_PROMPTS[:n_prompts]
    _job_log(job_id, f"target={model_path}, judge={judge_model}, n={len(prompts)}")
    ev = RefusalEvaluator(prompts=prompts, max_new_tokens=max_new_tokens)
    result = ev.evaluate(
        model_path, tokenizer_path=tokenizer_path,
        judge_model=judge_model, judge_4bit=judge_4bit,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SABERRunReq(BaseModel):
    base_model: str = Field(..., description="HF id of the base model")
    output_dir: str = "/workspace/saber-lab/runs/latest"
    config: dict = Field(default_factory=dict, description="SABERConfig kwargs")


class KLDReq(BaseModel):
    base_model: str
    ablated_path: str
    cache_path: Optional[str] = None
    chat_template: bool = True
    max_seq_len: int = 512
    batch_size: int = 2


class RefusalReq(BaseModel):
    model_path: str
    tokenizer_path: Optional[str] = None
    judge_model: Optional[str] = None
    judge_4bit: bool = True
    max_new_tokens: int = 256
    n_prompts: int = 0  # 0 = all default HARMFUL_PROMPTS


class SweepReq(BaseModel):
    base_model: str
    output_root: str = "/workspace/saber-lab/runs/sweep"
    alpha_values: List[float] = Field(default_factory=lambda: [0.70, 0.85, 1.00])
    config: dict = Field(default_factory=dict)


MODEL_PRESETS = [
    {"label": "Gemma 4 E4B", "model": "google/gemma-4-E4B-it", "notes": "current dial-in target"},
    {"label": "Gemma 4 E2B", "model": "google/gemma-4-E2B-it", "notes": "smaller replication"},
    {"label": "Qwen 3.5 4B", "model": "Qwen/Qwen3.5-4B", "notes": "same-scale cross-family replication"},
    {"label": "Qwen 3.5 9B", "model": "Qwen/Qwen3.5-9B", "notes": "larger-scale cross-family replication"},
    {"label": "Llama 3.1 8B Instruct", "model": "meta-llama/Llama-3.1-8B-Instruct", "notes": "license-gated cross-family target"},
]


def _result_row(run_dir: Path) -> Optional[dict]:
    saber = _read_json_file(run_dir / "saber_result.json")
    if not saber:
        return None
    kld = _read_json_file(run_dir / "kld_result.json") or {}
    cfg = saber.get("config") or {}
    return {
        "run_dir": str(run_dir),
        "model": saber.get("model"),
        "alpha_base": cfg.get("alpha_base"),
        "alpha_entangled": cfg.get("alpha_entangled"),
        "post_refusal_rate": saber.get("post_refusal_rate", saber.get("post_refusal")),
        "final_residual_refusal": saber.get("final_residual_refusal"),
        "mean_kld": kld.get("mean_kld"),
        "median_kld": kld.get("median_kld"),
        "max_kld": kld.get("max_kld"),
        "iterations": saber.get("iterations"),
        "selected_layers": saber.get("selected_layers"),
        "total_directions_ablated": saber.get("total_directions_ablated"),
    }


app = FastAPI(title="SABER Lab Backend", version="0.1.0")


@app.on_event("startup")
def _startup():
    _ensure_schema()
    print(f"SABER Lab Backend up. DB at {DB_PATH}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "saber_lab_dir": str(SABER_LAB_DIR)}


@app.post("/run/saber")
def run_saber(req: SABERRunReq):
    job_id = _new_job("saber_run", req.dict())
    _spawn(job_id, _do_saber_run, req.base_model, req.config, req.output_dir)
    return {"job_id": job_id}




@app.post("/run/sweep")
def run_sweep(req: SweepReq):
    job_id = _new_job("saber_sweep", req.dict())
    _spawn(job_id, _do_saber_sweep, req.base_model, req.alpha_values, req.config, req.output_root)
    return {"job_id": job_id}


@app.get("/presets/models")
def model_presets():
    return MODEL_PRESETS


@app.get("/results/autotune")
def results_autotune(root: str = str(SABER_LAB_DIR / "runs"),
                     model: str = "google/gemma-4-E4B-it",
                     limit: int = 12, proposals: int = 4):
    collect_runs, best_rows, propose = _import_autotune()
    rows = collect_runs(Path(root))
    ranked = [r.__dict__ for r in best_rows(rows, model)[:limit]]
    return {
        "model": model,
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
        "proposals": propose(rows, model, proposals),
    }


@app.get("/results/summary")
def results_summary(root: str = str(SABER_LAB_DIR / "runs"), limit: int = 100):
    root_path = Path(root)
    rows = []
    if root_path.exists():
        for path in sorted(root_path.rglob("saber_result.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            row = _result_row(path.parent)
            if row:
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


@app.post("/eval/kld")
def eval_kld(req: KLDReq):
    job_id = _new_job("eval_kld", req.dict())
    _spawn(job_id, _do_kld, req.base_model, req.ablated_path, req.cache_path,
           req.chat_template, req.max_seq_len, req.batch_size)
    return {"job_id": job_id}


@app.post("/eval/refusal")
def eval_refusal(req: RefusalReq):
    job_id = _new_job("eval_refusal", req.dict())
    _spawn(job_id, _do_refusal, req.model_path, req.tokenizer_path,
           req.judge_model, req.judge_4bit, req.max_new_tokens, req.n_prompts)
    return {"job_id": job_id}


@app.get("/job/{job_id}")
def job(job_id: str):
    with JOBS_LOCK:
        if job_id not in JOBS:
            # Try DB
            try:
                with _db() as cx:
                    row = cx.execute(
                        "SELECT id, kind, status, params, log, result, error, "
                        "created_at, started_at, finished_at FROM jobs WHERE id = ?",
                        (job_id,),
                    ).fetchone()
                if row:
                    return {
                        "id": row[0], "kind": row[1], "status": row[2],
                        "params": json.loads(row[3]) if row[3] else None,
                        "log": json.loads(row[4]) if row[4] else [],
                        "result": json.loads(row[5]) if row[5] else None,
                        "error": row[6],
                        "created_at": row[7], "started_at": row[8], "finished_at": row[9],
                    }
            except Exception:
                pass
            raise HTTPException(404, "job not found")
        return JOBS[job_id]


@app.get("/jobs")
def jobs(limit: int = 50, kind: Optional[str] = None):
    """Return recent jobs from the DB. Newest first."""
    where = "WHERE kind = ?" if kind else ""
    args = (kind, limit) if kind else (limit,)
    sql = (f"SELECT id, kind, status, created_at, finished_at FROM jobs {where} "
           f"ORDER BY created_at DESC LIMIT ?")
    with _db() as cx:
        rows = cx.execute(sql, args).fetchall()
    return [
        {"id": r[0], "kind": r[1], "status": r[2],
         "created_at": r[3], "finished_at": r[4]}
        for r in rows
    ]


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SABER_LAB_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
