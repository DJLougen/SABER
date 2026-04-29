# Contributing

SABER is research software, so small, reviewable changes are preferred.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the compile smoke test before opening a pull request:

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

## Contribution Guidelines

- Keep large model artifacts out of git.
- Do not include secrets, tokens, local absolute paths, or generated checkpoints.
- Keep public examples free of procedural high-risk content.
- Include evaluation context when changing scoring, prompts, or release-facing metrics.
- Prefer clear scripts and documented result files over one-off notebook state.
