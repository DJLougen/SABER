# SABER

SABER is a research toolkit for controlled refusal shaping in open-weight language models. It studies refusal editing as a Pareto optimization problem: reduce broad, boilerplate refusals while measuring behavioral drift and documenting which severe-harm categories remain refused.

The current implementation combines refusal-direction extraction, multi-candidate ranking, iterative ablation, generation-based refusal checks, KLD drift scoring, a lightweight autotune workflow, a FastAPI backend, a Textual TUI, and a static browser report.

## Status

- Gemma 4 E4B has a current SABER release candidate uploaded as `GestaltLabs/Gemma-4-E4B-SABER`.
- Ornstein-Hermes-3.6-27B (`GestaltLabs/Ornstein-Hermes-3.6-27b`) is under active tuning and expanded evaluation.
- Candidate selection is based on a refusal/KLD frontier rather than a single refusal-rate target.
- Large generated model artifacts are intentionally ignored by git. This repo tracks code, run summaries, lightweight results, and documentation.

## Current Results

### Gemma 4 E4B

| candidate | run | refusal | KLD | interpretation |
|---|---:|---:|---:|---|
| aggressive | `gemma4_e4b_auto_svd_nh60_a825_g14` | `0.00%` | `0.4327` | strongest refusal suppression, higher drift |
| balanced | `gemma4_e4b_auto_svd_a825_g14` | `2.04%` | `0.3164` | low refusal with lower drift |

The Gemma result is the current release path for `GestaltLabs/Gemma-4-E4B-SABER`.

### Ornstein-Hermes-3.6-27B

The `ornstein_hermes36_27b_svd_a450_g10` candidate refused `14/349` prompts (`4.01%`) on the expanded keyword-refusal evaluation. The retained refusals were concentrated in severe harm categories such as business sabotage, credential theft/phishing, evading police, money laundering, document forgery, blackmail, stalking, workplace harassment, illegal drug sales, and prescription drug abuse.

That pattern is the intended release frame for SABER: lower over-refusal while retaining visible boundaries for severe criminal, coercive, or interpersonal-harm requests. Stronger Ornstein candidates are still being evaluated against KLD before a final release point is selected.

## Method Summary

SABER treats refusal ablation as a constrained editing problem rather than a binary remove/refuse switch.

1. Collect harmful, harmless, and capability-oriented activations from a base model.
2. Extract candidate refusal directions with methods such as difference-in-means, SVD, Fisher-style ranking, and related spectral variants.
3. Rank layers and directions by separability and estimated capability entanglement.
4. Apply differential ablation strengths so cleaner refusal directions can be edited more strongly while capability-entangled directions are treated more conservatively.
5. Re-evaluate candidates by over-refusal rate, retained refusal categories, residual refusal score, qualitative generations, and KLD drift from the base model.

The practical target is a useful frontier: a model that is less prone to broad refusal while staying close to the source model and retaining refusals in categories worth documenting publicly.

## Install

Use a Python environment with a CUDA PyTorch build appropriate for your GPU.

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

Generate the static report:

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

The quick probe is useful for search, but release claims should use expanded refusal evaluation, KLD/drift evaluation, saved run configuration, category review, and qualitative output inspection.

## Evaluation Workflow

SABER candidates should be read as a Pareto frontier:

- over-refusal rate: lower is better, with severe-harm retained refusals documented separately
- retained refusal categories: severe criminal, coercive, and interpersonal-harm refusals are reported as part of the model behavior profile
- KLD / behavioral drift: lower is better among candidates with comparable refusal behavior
- residual refusal score: useful during search, but generation-based evaluation is the release-facing metric
- qualitative samples: outputs are inspected before publishing a model card or public claim

Typical commands:

```bash
python saber_eval_refusal.py
python saber_eval_kld.py
python saber_autotune.py --model google/gemma-4-E4B-it --limit 10
```

The expanded overnight harness writes larger refusal-eval outputs under each run directory as `refusal_expanded.json`.

## TUI / Browser Workflow

Start the backend on the GPU host:

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
- `SABER_WRITEUP_DRAFT.md`: working research writeup
- `TECHNICAL_REPORT.md`: longer technical draft

## Attribution and Related Work

SABER is part of the refusal-direction and abliteration research lineage. The project is explicitly inspired by, and should be read alongside, the following work:

- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda, [Refusal in Language Models Is Mediated by a Single Direction](https://huggingface.co/papers/2406.11717), 2024. This work established the core refusal-direction finding that made directional refusal ablation practical.
- Maxime Labonne, [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration), 2024. This article helped popularize practical abliteration workflows and credits FailSpy's notebook/library lineage.
- FailSpy, [abliterator](https://github.com/FailSpy/abliterator) and associated abliterated model releases. These community recipes helped turn the refusal-direction finding into reproducible open tooling.
- Jim Lai (`grimjim`), [Projected Abliteration](https://huggingface.co/blog/grimjim/projected-abliteration), 2025, and [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration), 2025. These posts are important refinements around geometry, projection, and capability preservation.
- Philipp Emanuel Weidmann, [Heretic](https://github.com/p-e-w/heretic), 2025-2026. Heretic combines directional ablation with automated parameter search over refusal and KL divergence.
- Pliny the Prompter / OBLITERATUS, [Hugging Face Space](https://huggingface.co/spaces/pliny-the-prompter/obliteratus) and [OBLITERATUS releases](https://huggingface.co/OBLITERATUS). OBLITERATUS represents broad community tooling and experimentation around abliteration workflows.
- Jiunsong, [SuperGemma4 E4B Abliterated](https://huggingface.co/Jiunsong/supergemma4-e4b-abliterated) and related SuperGemma releases. These were influential evidence that Gemma-family ablation can improve practical behavior and release quality, not only reduce refusals.
- Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, and Weiyan Shi, [LLMs Encode Harmfulness and Refusal Separately](https://huggingface.co/papers/2507.11878), 2025. This distinction between harmfulness and refusal is closely aligned with SABER's goal of reducing over-refusal while preserving meaningful severe-harm boundaries.

SABER's intended contribution is the specific controlled-refusal-shaping workflow used here: multi-candidate refusal extraction, separability/entanglement-aware ranking, differential ablation strength, and explicit Pareto selection over refusal behavior and KLD drift.

## Scope and Limitations

SABER is research software for open-weight model editing and evaluation. Results are model-family dependent, probe-set dependent, and sensitive to generation settings. Refusal percentages should be interpreted together with prompt count, category composition, KLD, and qualitative samples.

This work has clear dual-use implications. Public releases should include base model lineage, method summary, refusal-eval size, retained-refusal categories, drift metrics, and known limitations.
