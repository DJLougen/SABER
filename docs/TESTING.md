# Testing SABER Locally

## 1. Clone the Active Branch

```bash
git clone -b turquoise-current https://github.com/DJLougen/SABER.git
cd SABER
```

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-specific PyTorch wheel first if your machine needs one.

## 3. Smoke Test

```bash
python -m py_compile saber.py run_saber.py saber_eval_kld.py saber_eval_refusal.py saber_lab_server.py saber_lab_tui.py
python generate_saber_report.py
```

## 4. TUI Test

In one shell on the GPU host:

```bash
python saber_lab_server.py
```

In another shell, or on your PC through an SSH tunnel:

```bash
python saber_lab_tui.py --api http://127.0.0.1:8765
```

## 5. What Counts As Ready

A candidate is not ready just because the quick 30-prompt refusal probe says 0%. For writeup/model-card readiness, require:

- expanded refusal eval result with a large prompt count
- KLD/drift result against the selected base model
- recorded run config and output path
- qualitative sample review
- clear attribution that the work is a custom refusal-ablation workflow inspired by prior abliteration/decensoring methods
