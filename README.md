# SABER

**Spectral Analysis-Based Entanglement Resolution**

SABER is a research toolkit for controlled refusal shaping in open-weight language models. It searches for refusal-related activation directions, edits candidate models, and ranks candidates on a frontier of refusal behavior and behavioral drift.

The goal is not to make a model answer everything. SABER is designed to reduce broad, boilerplate over-refusal while measuring what changed and documenting which high-risk categories still refuse.

## What SABER Does

- Extracts refusal directions from harmful, harmless, and capability prompt activations.
- Ranks candidate directions by separability and estimated capability entanglement.
- Applies configurable ablation strengths across selected layers and directions.
- Scores candidates with generation-based refusal checks and KLD drift checks.
- Provides a command-line runner, autotune helper, FastAPI backend, Textual TUI, and static report workflow.

## Status

SABER is experimental research software. Current front-facing results are tracked in [docs/CURRENT_RESULTS.md](docs/CURRENT_RESULTS.md), and longer technical notes live in [docs/research/TECHNICAL_REPORT.md](docs/research/TECHNICAL_REPORT.md).

Large generated model artifacts are intentionally not tracked by git. This repository tracks code, lightweight result summaries, documentation, and reproducibility scaffolding.

## Install

Use a Python environment with a CUDA-enabled PyTorch build that matches your machine.

```bash
git clone https://github.com/DJLougen/SABER.git
cd SABER

python -m venv .venv
source .venv/bin/activate

# Install the correct torch wheel for your CUDA stack first if needed.
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Smoke Test

This verifies that the repo imports and the main entry points compile.

```bash
python -m py_compile \
  saber.py \
  run_saber.py \
  saber_autotune.py \
  saber_eval_kld.py \
  saber_eval_refusal.py \
  saber_lab_server.py \
  saber_lab_tui.py
```

For more local checks, see [docs/TESTING.md](docs/TESTING.md).

## Minimal Run

Start with a small probe before running a long sweep:

```bash
python run_saber.py \
  --model google/gemma-4-E2B-it \
  --output-dir runs/gemma4_e2b_probe \
  --extraction-method svd \
  --n-harmful 30 \
  --n-harmless 30 \
  --n-capability 30 \
  --alpha-base 0.8 \
  --layer-top-k 8
```

The quick probe is useful for exploration. Do not treat it as release evidence. Public model claims should include expanded refusal evaluation, KLD drift evaluation, saved run configuration, category review, and qualitative samples.

## Evaluation Workflow

SABER candidates should be selected as Pareto points:

- **Refusal rate:** lower over-refusal is useful, but zero refusal is not automatically the target.
- **Retained refusal categories:** severe criminal, coercive, and interpersonal-harm refusals should be reviewed and documented.
- **KLD drift:** lower is better among candidates with comparable refusal behavior.
- **Residual refusal score:** useful during search, but generation-based evaluation is the release-facing metric.
- **Qualitative samples:** inspect actual outputs before publishing a model card or public claim.

Typical commands:

```bash
python saber_eval_refusal.py
python saber_eval_kld.py
python saber_autotune.py --model google/gemma-4-E2B-it --limit 10
```

Release criteria are summarized in [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## TUI And Browser Workflow

Start the backend on a GPU host:

```bash
python saber_lab_server.py
```

If the backend is remote, tunnel it:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Run the TUI locally:

```bash
python saber_lab_tui.py --api http://127.0.0.1:8765
```

Generate or view a static report:

```bash
python generate_saber_report.py
```

Generated reports and historical lightweight outputs are kept under [docs/results/](docs/results/).

## Repository Layout

| Path | Purpose |
|---|---|
| `saber.py` | Core SABER implementation |
| `run_saber.py` | Main CLI entry point for model loading, ablation, and quick checks |
| `saber_autotune.py` | Candidate ranking and next-run proposal helper |
| `saber_eval_refusal.py` | Generation-based refusal evaluator |
| `saber_eval_kld.py` | KLD and behavioral drift evaluator |
| `saber_lab_server.py` | FastAPI backend for GPU-side runs and evaluations |
| `saber_lab_tui.py` | Textual TUI for controlling the backend |
| `config.py` | Default probe prompt sets and run configuration |
| `docs/` | Public documentation, testing notes, results, and research writeups |
| `docs/model-cards/` | Model-card drafts and release text templates |
| `scripts/release/` | Release, upload, and GGUF helper scripts |
| `examples/` | Copyable command examples |

## Safety And Scope

SABER is dual-use model-editing research. It can materially alter refusal behavior. Before releasing a tuned model, document:

- base model lineage and license
- prompt set size and category composition
- retained refusal categories
- KLD or comparable drift metric
- generation settings used for evaluation
- known limitations and failure modes

See [docs/SAFETY.md](docs/SAFETY.md) for the repository policy and recommended public-release framing.

## Attribution And Related Work

SABER is part of the refusal-direction and abliteration research lineage. It should be read alongside:

- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda, [Refusal in Language Models Is Mediated by a Single Direction](https://huggingface.co/papers/2406.11717), 2024.
- Maxime Labonne, [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration), 2024.
- FailSpy, [abliterator](https://github.com/FailSpy/abliterator) and associated abliterated model releases.
- Jim Lai (`grimjim`), [Projected Abliteration](https://huggingface.co/blog/grimjim/projected-abliteration), 2025, and [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration), 2025.
- Philipp Emanuel Weidmann, [Heretic](https://github.com/p-e-w/heretic), 2025-2026.

SABER's intended contribution is the controlled-refusal-shaping workflow used here: multi-candidate refusal extraction, separability and entanglement-aware ranking, differential ablation strength, and explicit frontier selection over refusal behavior and KLD drift.

## License

Apache-2.0. See [LICENSE](LICENSE).
