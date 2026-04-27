"""
SABER Lab TUI — textual front-end for the dial-in loop.

Connects to a saber_lab_server.py backend (default http://127.0.0.1:8765,
expected via SSH tunnel to the Blackwell). Renders a config form on the left,
a live job log + recent runs table on the right, and action buttons up top.

Run with:
    pip install textual httpx rich
    python saber_lab_tui.py [--api http://127.0.0.1:8765]

Workflow:
    1. Edit config inputs (hyperparameters, base model id).
    2. Press "Run SABER" — submits to backend, status streams into the log.
    3. After a SABER run finishes (or pointing at any HF model), press
       "Eval KLD" and "Eval Refusal" — backend runs the eval, results land
       in the recent-runs table.
    4. Switch the "ablated_path" input to a different run_id's output_dir
       to evaluate that one instead.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

import httpx
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label,
                             RichLog, Static)


# ---------------------------------------------------------------------------
# Defaults — match SABERConfig defaults from saber.py / config.py
# ---------------------------------------------------------------------------

DEFAULT_BASE = "google/gemma-4-E2B-it"
DEFAULT_OUTPUT_DIR = "/workspace/saber-lab/runs/latest"
DEFAULT_HYPERPARAMS = {
    "extraction_method": "fisher_lda",
    "n_directions": "3",
    "layer_selection_strategy": "elbow",
    "layer_top_k": "10",
    "alpha_base": "0.85",
    "alpha_entangled": "0.10",
    "entanglement_threshold": "0.70",
    "max_iterations": "5",
    "convergence_threshold": "0.01",
    "decay_factor": "0.7",
}


def _parse_hyperparams(values: Dict[str, str]) -> Dict[str, Any]:
    """Cast each value to its expected type (string or numeric)."""
    out: Dict[str, Any] = {}
    type_hints = {
        "extraction_method": str,
        "n_directions": int,
        "layer_selection_strategy": str,
        "layer_top_k": int,
        "alpha_base": float,
        "alpha_entangled": float,
        "entanglement_threshold": float,
        "max_iterations": int,
        "convergence_threshold": float,
        "decay_factor": float,
    }
    for k, v in values.items():
        if not v:
            continue
        cast = type_hints.get(k, str)
        try:
            out[k] = cast(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# TUI app
# ---------------------------------------------------------------------------


class SaberTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #toolbar { dock: top; height: 3; padding: 0 1; }
    #toolbar Button { margin: 0 1; min-width: 16; }
    #body { layout: horizontal; }
    #left { width: 40%; min-width: 36; padding: 1 1; border: round $accent; }
    #right { width: 1fr; padding: 1 1; }
    #log { height: 1fr; border: round $primary; }
    #table { height: 14; border: round $primary; }
    #left Label { color: $accent; padding-top: 1; }
    Input { margin-bottom: 0; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "press_run_saber", "Run SABER"),
        Binding("k", "press_eval_kld", "Eval KLD"),
        Binding("f", "press_eval_refusal", "Eval Refusal"),
        Binding("s", "press_run_sweep", "Run Sweep"),
        Binding("p", "press_refresh_results", "Pareto"),
        Binding("ctrl+l", "clear_log", "Clear log"),
    ]

    def __init__(self, api_base: str, **kw):
        super().__init__(**kw)
        self.api = api_base.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_tasks: set[asyncio.Task] = set()

    # -- lifecycle --

    async def on_mount(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        self.title = "SABER Lab"
        self.sub_title = self.api

        table: DataTable = self.query_one("#table", DataTable)
        table.add_columns("when", "kind", "status", "summary")
        results: DataTable = self.query_one("#results", DataTable)
        results.add_columns("alpha", "refusal", "residual", "mean KLD", "max KLD", "run")

        try:
            r = await self._client.get(f"{self.api}/health")
            r.raise_for_status()
            self._log_line(f"[green]connected[/] {self.api}  -> {r.json()}")
        except Exception as e:
            self._log_line(f"[red]connect failed[/] {self.api}: {e}")
        await self._refresh_history()
        await self._refresh_results()

    async def on_unmount(self):
        if self._client:
            await self._client.aclose()

    # -- compose --

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="toolbar"):
            yield Button("Run SABER (r)", id="btn_run", variant="primary")
            yield Button("Run Sweep (s)", id="btn_sweep")
            yield Button("Eval KLD (k)", id="btn_kld")
            yield Button("Eval Refusal (f)", id="btn_ref")
            yield Button("Pareto (p)", id="btn_results")
            yield Button("Refresh history", id="btn_refresh")
        with Horizontal(id="body"):
            with VerticalScroll(id="left"):
                yield Label("[b]Base model[/b]")
                yield Input(value=DEFAULT_BASE, id="base_model")
                yield Label("[b]Output dir[/b]")
                yield Input(value=DEFAULT_OUTPUT_DIR, id="output_dir")
                yield Label("[b]Sweep output root[/b]")
                yield Input(value="/workspace/saber-lab/runs/dialin", id="sweep_root")
                yield Label("[b]Alpha sweep[/b] (comma separated)")
                yield Input(value="0.70,0.85,1.00", id="alpha_values")
                yield Label("[b]Ablated path[/b] (for evals)")
                yield Input(value=DEFAULT_OUTPUT_DIR, id="ablated_path")
                yield Label("[b]SABER hyperparameters[/b]")
                for k, v in DEFAULT_HYPERPARAMS.items():
                    yield Label(k)
                    yield Input(value=v, id=f"hp_{k}")
                yield Label("[b]Refusal eval[/b]")
                yield Label("judge_model (blank = keyword only)")
                yield Input(value="", id="judge_model",
                            placeholder="cais/HarmBench-Llama-2-13b-cls")
                yield Label("max_new_tokens")
                yield Input(value="256", id="max_new_tokens")
                yield Label("n_prompts (0=all)")
                yield Input(value="0", id="n_prompts")
            with Vertical(id="right"):
                yield RichLog(id="log", highlight=True, markup=True, wrap=True)
                yield Static("[b]Method dial-in: refusal/KLD frontier[/b]", markup=True)
                yield DataTable(id="results", zebra_stripes=True)
                yield DataTable(id="table", zebra_stripes=True)
        yield Footer()

    # -- helpers --

    def _val(self, widget_id: str) -> str:
        try:
            return self.query_one(f"#{widget_id}", Input).value.strip()
        except Exception:
            return ""

    def _log_line(self, msg: str):
        self.query_one("#log", RichLog).write(msg)

    def _collect_hyperparams(self) -> Dict[str, str]:
        return {k: self._val(f"hp_{k}") for k in DEFAULT_HYPERPARAMS}

    # -- actions --

    def action_press_run_saber(self):
        self.query_one("#btn_run", Button).press()

    def action_press_eval_kld(self):
        self.query_one("#btn_kld", Button).press()

    def action_press_eval_refusal(self):
        self.query_one("#btn_ref", Button).press()

    def action_press_run_sweep(self):
        self.query_one("#btn_sweep", Button).press()

    def action_press_refresh_results(self):
        self.query_one("#btn_results", Button).press()

    def action_clear_log(self):
        self.query_one("#log", RichLog).clear()

    async def on_button_pressed(self, ev: Button.Pressed):
        bid = ev.button.id
        if bid == "btn_run":
            await self._submit_run_saber()
        elif bid == "btn_kld":
            await self._submit_eval_kld()
        elif bid == "btn_ref":
            await self._submit_eval_refusal()
        elif bid == "btn_sweep":
            await self._submit_run_sweep()
        elif bid == "btn_results":
            await self._refresh_results()
        elif bid == "btn_refresh":
            await self._refresh_history()
            await self._refresh_results()

    # -- submitters --

    async def _post(self, path: str, payload: dict) -> Optional[dict]:
        try:
            r = await self._client.post(f"{self.api}{path}", json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self._log_line(f"[red]POST {path} failed:[/] {e}")
            return None

    async def _submit_run_saber(self):
        payload = {
            "base_model": self._val("base_model"),
            "output_dir": self._val("output_dir"),
            "config": _parse_hyperparams(self._collect_hyperparams()),
        }
        self._log_line(f"[cyan]POST /run/saber[/] base={payload['base_model']} "
                       f"hp={json.dumps(payload['config'])}")
        resp = await self._post("/run/saber", payload)
        if resp:
            self._track_job(resp["job_id"], "saber_run")


    def _alpha_values(self) -> list[float]:
        raw = self._val("alpha_values")
        vals: list[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(float(part))
        return vals or [0.70, 0.85, 1.00]

    async def _submit_run_sweep(self):
        payload = {
            "base_model": self._val("base_model"),
            "output_root": self._val("sweep_root"),
            "alpha_values": self._alpha_values(),
            "config": _parse_hyperparams(self._collect_hyperparams()),
        }
        payload["config"].pop("alpha_base", None)
        self._log_line(f"[cyan]POST /run/sweep[/] base={payload['base_model']} "
                       f"alphas={payload['alpha_values']} root={payload['output_root']}")
        resp = await self._post("/run/sweep", payload)
        if resp:
            self._track_job(resp["job_id"], "saber_sweep")


    async def _submit_eval_kld(self):
        payload = {
            "base_model": self._val("base_model"),
            "ablated_path": self._val("ablated_path"),
        }
        self._log_line(f"[cyan]POST /eval/kld[/] {payload}")
        resp = await self._post("/eval/kld", payload)
        if resp:
            self._track_job(resp["job_id"], "eval_kld")

    async def _submit_eval_refusal(self):
        payload = {
            "model_path": self._val("ablated_path"),
            "judge_model": self._val("judge_model") or None,
            "max_new_tokens": int(self._val("max_new_tokens") or "256"),
            "n_prompts": int(self._val("n_prompts") or "0"),
        }
        self._log_line(f"[cyan]POST /eval/refusal[/] {payload}")
        resp = await self._post("/eval/refusal", payload)
        if resp:
            self._track_job(resp["job_id"], "eval_refusal")

    # -- polling --

    def _track_job(self, job_id: str, kind: str):
        self._log_line(f"[yellow]job {job_id[:8]}[/] {kind} queued")
        task = asyncio.create_task(self._poll_job(job_id, kind))
        self._poll_tasks.add(task)
        task.add_done_callback(self._poll_tasks.discard)

    async def _poll_job(self, job_id: str, kind: str):
        last_loglen = 0
        last_status = None
        while True:
            try:
                r = await self._client.get(f"{self.api}/job/{job_id}")
                r.raise_for_status()
                d = r.json()
            except Exception as e:
                self._log_line(f"[red]poll {job_id[:8]} failed:[/] {e}")
                await asyncio.sleep(3.0)
                continue

            status = d.get("status")
            log_lines = d.get("log") or []
            for line in log_lines[last_loglen:]:
                self._log_line(f"  [dim]{job_id[:8]}[/] {line}")
            last_loglen = len(log_lines)

            if status != last_status:
                self._log_line(f"[bold]{job_id[:8]} -> {status}[/]")
                last_status = status

            if status in ("done", "error"):
                if status == "done":
                    summary = self._summarize(kind, d.get("result") or {})
                    self._log_line(f"[green]{job_id[:8]} result:[/] {summary}")
                    self._add_history_row(job_id, kind, status, summary)
                else:
                    self._log_line(f"[red]{job_id[:8]} error[/]\n{d.get('error') or ''}")
                    self._add_history_row(job_id, kind, status, "(see log)")
                return

            await asyncio.sleep(2.0)

    def _summarize(self, kind: str, result: dict) -> str:
        if not isinstance(result, dict):
            return str(result)[:80]
        if kind == "eval_kld":
            return (f"mean_kld={result.get('mean_kld'):.4e}  "
                    f"max_kld={result.get('max_kld'):.4e}  "
                    f"n={result.get('n_prompts')}")
        if kind == "eval_refusal":
            kw = result.get("refusal_rate_kw")
            j = result.get("refusal_rate_judge")
            asr = result.get("asr_judge")
            parts = [f"refusal_kw={kw:.2%}" if kw is not None else "kw=?"]
            if j is not None:
                parts.append(f"refusal_judge={j:.2%}")
            if asr is not None:
                parts.append(f"asr={asr:.2%}")
            return "  ".join(parts)
        if kind == "saber_run":
            rr = result.get("post_refusal_rate")
            rr_s = f"refusal={rr:.2%}" if rr is not None else "refusal=?"
            return (f"{rr_s}, residual={result.get('final_residual_refusal')}, "
                    f"dirs={result.get('total_directions_ablated')}, "
                    f"out={result.get('output_dir')}")
        if kind == "saber_sweep":
            runs = result.get("runs") or []
            return f"{len(runs)} runs -> {result.get('output_root')}"
        return json.dumps(result)[:120]

    def _add_history_row(self, job_id: str, kind: str, status: str, summary: str):
        table: DataTable = self.query_one("#table", DataTable)
        table.add_row(time.strftime("%H:%M:%S"), kind, status, summary)

    async def _refresh_history(self):
        try:
            r = await self._client.get(f"{self.api}/jobs", params={"limit": 30})
            r.raise_for_status()
            jobs = r.json()
        except Exception as e:
            self._log_line(f"[red]history fetch failed:[/] {e}")
            return
        table: DataTable = self.query_one("#table", DataTable)
        table.clear()
        for j in jobs:
            ts = time.strftime("%m-%d %H:%M:%S",
                               time.localtime(j.get("created_at") or 0))
            table.add_row(ts, j.get("kind") or "?", j.get("status") or "?",
                          j.get("id", "")[:8])

    async def _refresh_results(self):
        try:
            r = await self._client.get(f"{self.api}/results/summary", params={"limit": 100})
            r.raise_for_status()
            rows = r.json()
        except Exception as e:
            self._log_line(f"[red]results fetch failed:[/] {e}")
            return
        table: DataTable = self.query_one("#results", DataTable)
        table.clear()
        def fmt_float(v, digits=4):
            return "" if v is None else f"{float(v):.{digits}f}"
        def fmt_pct(v):
            return "" if v is None else f"{float(v):.2%}"
        rows = sorted(rows, key=lambda x: (x.get("model") or "", x.get("alpha_base") or 999))
        for row in rows:
            table.add_row(
                fmt_float(row.get("alpha_base"), 2),
                fmt_pct(row.get("post_refusal_rate")),
                fmt_float(row.get("final_residual_refusal")),
                fmt_float(row.get("mean_kld")),
                fmt_float(row.get("max_kld")),
                (row.get("run_dir") or "")[-28:],
            )


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default=os.environ.get("SABER_API", "http://127.0.0.1:8765"))
    args = p.parse_args()
    SaberTUI(api_base=args.api).run()


if __name__ == "__main__":
    _main()
