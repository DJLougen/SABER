# SABER

SABER is a research toolkit for controlled refusal shaping. Model candidates are selected as a Pareto tradeoff between over-refusal reduction and behavioral drift, not as a race to remove every refusal. The repo contains the core SABER implementation, run/evaluation scripts, an autotune helper, a FastAPI backend, a Textual TUI, and a static browser report.

This work is inspired by the broader abliteration/refusal-ablation community, including Pliny-style and Heretic-style decensoring workflows, but the goal here is to make my own reproducible variant: identify refusal-associated directions, ablate them in a controlled way, and track how much capability/behavior moves while overbroad refusals go down. Retained refusals on severe criminal, coercive, or interpersonal-harm requests are treated as intended behavior, not failure.

## Current Status

- Gemma 4 E4B has a current SABER candidate uploaded to Hugging Face as `GestaltLabs/Gemma-4-E4B-SABER`.
- Ornstein-Hermes-3.6-27B is under active tuning.
- Early refusal checks used 30 harmful prompts. The active validation path now expands that to a 349-prompt keyword-refusal eval before making stronger claims.
- Current Ornstein evidence shows that some retained refusals are concentrated in severe harm categories, which is a useful release property rather than a defect.
- KLD is the drift metric used for Pareto ranking. Treat it as a relative selection signal until calibration issues are fully documented for each model family.
- Large model artifacts are intentionally ignored. The repo tracks code, summaries, and lightweight JSON results, not full `safetensors` outputs.

## Install

Use a Python environment with a CUDA PyTorch build appropriate for your GPU. On a fresh machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the default PyTorch wheel is not correct for your CUDA stack, install PyTorch first from the official PyTorch index, then install the rest of `requirements.txt`.

## Quick Repo Test

```bash
python -m py_compile \
  saber.py \
  run_saber.py \
  saber_autotune.py \
  saber_eval_kld.py \
  saber_eval_refusal.py \
  saber_lab_server.py \
  saber_lab_tui.py \
  generate_saber_report.py
```

Open the static report locally:

```bash
python generate_saber_report.py
# then open saber_browser_report.html
```

## Running SABER

A minimal run looks like this:

```bash
python run_saber.py \
  --model google/gemma-4-E4B-it \
  --output-dir runs/my_saber_run \
  --extraction-method svd \
  --alpha-base 0.825 \
  --layer-top-k 14 \
  --n-harmful 30
```

For serious claims, do not stop at the quick refusal probe. Run the larger refusal eval and KLD eval, then compare candidates by over-refusal reduction, retained refusal categories, and drift.

## Evaluation Workflow

SABER candidates should be read as a Pareto frontier, not a single scalar leaderboard:

- over-refusal rate: lower is better, but literal 0% refusal is not the release target
- retained severe-harm refusals: refusals on stalking, blackmail, phishing, document forgery, money laundering, illegal drug sales, and similar categories are acceptable and should be documented
- KLD / behavioral drift: lower is better among candidates with comparable over-refusal reduction
- residual refusal score: useful during search, but real generations decide the refusal metric
- qualitative samples: inspect outputs before publishing any model card or claim

Typical flow:

```bash
python saber_eval_refusal.py  # quick generation-based refusal scoring helpers
python saber_eval_kld.py      # KLD drift evaluation helpers
python saber_autotune.py --model google/gemma-4-E4B-it --limit 10
```

The overnight harness currently keeps expanded refusal outputs under each run directory as `refusal_expanded.json`.

## Release Framing

SABER model cards should not claim that the model has no safety boundary. The preferred framing is controlled refusal shaping:

- reduce broad, boilerplate, or capability-blocking refusal behavior
- preserve refusals for clearly harmful criminal/coercive requests
- publish the retained-refusal categories, not just a scalar refusal percentage
- select the candidate with the best refusal/KLD tradeoff for the desired behavior profile

For example, the Ornstein `a450_g10` candidate refused `14/349` expanded harmful-intent prompts (`4.01%`). Those refusals were concentrated in business sabotage, credential theft/phishing, evading police, money laundering, document forgery, blackmail, stalking, workplace harassment, illegal drug sales, and prescription drug abuse. For release purposes, that pattern is a feature: the model is less over-refusal-prone while still declining severe real-world harm categories.

## TUI / Browser Workflow

Start the backend on the machine with GPU access and model cache:

```bash
python saber_lab_server.py
```

If the backend is remote, tunnel it:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Run the TUI locally against the backend:

```bash
python saber_lab_tui.py --api http://127.0.0.1:8765
```

For a browser-only view, open `saber_browser_report.html` or regenerate it:

```bash
python generate_saber_report.py
```

## Important Files

- `saber.py`: core SABER implementation
- `run_saber.py`: CLI entry point for model loading, ablation, and quick checks
- `saber_autotune.py`: lightweight candidate ranking and proposal helper
- `saber_eval_refusal.py`: generation-based refusal evaluator
- `saber_eval_kld.py`: KLD/behavioral drift evaluator
- `saber_lab_server.py`: FastAPI backend for GPU-side runs/evals
- `saber_lab_tui.py`: Textual TUI for controlling the backend
- `generate_saber_report.py`: static HTML report generator
- `SABER_WRITEUP_DRAFT.md`: early writeup draft and framing notes
- `TECHNICAL_REPORT.md`: earlier technical report material

## Attribution / Prior Art

Do not claim that SABER invented refusal ablation or abliteration. The intended contribution is the particular implementation and workflow: spectral/refusal-direction extraction, iterative ablation, and explicit Pareto selection over over-refusal reduction, retained severe-harm refusals, and behavioral drift. Related inspirations and prior art should be credited in the writeup and model cards, including Pliny-style abliteration, Heretic-style automated decensoring, and community findings around Gemma-family ablation improvements.
