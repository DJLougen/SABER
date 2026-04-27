# SABER

SABER is a research toolkit for refusal ablation experiments with explicit tracking of refusal rate and behavioral drift. The repository contains the core SABER implementation, run/evaluation scripts, an autotune helper, a local FastAPI backend, a Textual TUI, and a generated browser report.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a CUDA PyTorch build appropriate for your machine if the default `pip install torch` wheel is not correct.

## TUI / Browser Workflow

Start the backend on the machine with the GPU and model cache:

```bash
python saber_lab_server.py
```

Then run the TUI locally against that backend:

```bash
python saber_lab_tui.py --api http://127.0.0.1:8765
```

For browser review, open `saber_browser_report.html` or regenerate it with:

```bash
python generate_saber_report.py
```

## Current Framing

Treat SABER selection as a Pareto optimization over refusal rate and behavioral drift/KLD. Current quick refusal checks are being expanded beyond the original 30-prompt probe set before final claims are made.
