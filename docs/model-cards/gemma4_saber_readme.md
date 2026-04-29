---
license: gemma
base_model: google/gemma-4-E2B-it
tags:
- refusal-ablation
- capability-preserving
- saber
- gemma4
- multimodal
pipeline_tag: text-generation
---

# google/gemma-4-E2B-it — SABER-Refined

> **0% refusal. -9.6% perplexity improvement. 50 directions.**

This model is a surgically-modified version of [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) using a novel proprietary method (**SABER** — Spectral Analysis-Based Entanglement Resolution) that removes safety refusal behavior while preserving — and in this case improving — model capability.

## Key Results

| Metric | Baseline | SABER-Refined | Delta |
|--------|----------|---------------|-------|
| Refusal Rate | 100% | **0%** | -100% |
| Perplexity | 498 | **450** | **-9.6%** |
| Directions Ablated | — | 50 (across 10 layers) | — |

Not only is capability preserved — it *improves*. The refusal directions were adding interference to the model's general fluency, and removing them reduces perplexity by 9.6%.

## How SABER Works

<img src="saber_pipeline.png" alt="SABER Pipeline" width="100%"/>

SABER identifies and ablates the refusal circuit through a five-stage pipeline:

**Stage 1 — Probing**: Extract activation profiles from both harmful and harmless inputs across all transformer layers.

**Stage 2 — Spectral Analysis**: Decompose activation differences into individual refusal directions, each scored by how strongly they separate harmful from harmless representations.

**Stage 3 — Entanglement Quantification**: Measure the overlap between each refusal direction and the model's capability subspace (reasoning, knowledge, code, etc.) to avoid collateral damage.

**Stage 4 — Targeted Ablation**: Remove only the pure-refusal components, with strength proportional to their purity (how little they overlap with capability).

**Stage 5 — Iterative Refinement**: Re-probe after each ablation pass to catch hydra effects (dormant refusal features that activate when primary ones are removed).

**Key differentiator from prior work**: SABER explicitly measures and respects the *entanglement* between refusal and capability representations. Directions that are heavily entangled with capability are either skipped or ablated at reduced strength, preventing the catastrophic degradation seen in naive approaches.

<img src="entanglement_scatter.png" alt="Direction Purity vs Separability" width="100%"/>

The plot above illustrates how SABER scores each extracted direction — high-purity directions receive full ablation strength, while lower-purity directions are treated more conservatively.

## Sweep Results

<img src="sweep_comparison.png" alt="SABER Sweep Comparison" width="100%"/>

Configuration search over `global_top_k` (number of top directions selected globally) and `alpha_base` (base ablation strength):

| Top-K | Alpha | Refusal | PPL | PPL Delta | Layers | Dirs Ablated |
|:-----:|:-----:|:-------:|:---:|:---------:|:------:|:------------:|
| **10** | **0.85** | **0%** | **450** | **-9.6%** | **10** | **50** |
| 10 | 0.70 | 0% | 513 | +3.0% | 10 | 50 |
| 10 | 1.00 | 5% | 410 | -17.6% | 10 | 50 |
| 25 | 0.70 | 0% | 819 | +64.5% | 20 | 125 |
| 25 | 0.85 | 0% | 1,423 | +185.9% | 19 | 125 |
| 25 | 1.00 | 5% | 1,662 | +233.9% | 20 | 125 |
| 50 | 0.70 | 5% | 1,313 | +163.7% | 22 | 250 |
| 50 | 0.85 | 0% | 2,683 | +439.0% | 21 | 250 |
| 50 | 1.00 | 0% | 7,190 | +1,344% | 22 | 250 |
| 75 | 0.70 | 0% | 1,949 | +291.4% | 22 | 330 |
| 75 | 0.85 | 0% | 2,792 | +460.8% | 22 | 330 |

**Best config: `top_k=10, alpha=0.85`** — achieves 0% refusal with PPL actually *below* baseline. The clear trend: fewer, higher-quality directions massively outperform aggressive ablation.

<img src="refusal_comparison.png" alt="Refusal Rate Comparison" width="100%"/>

## Capability Evaluation

Perplexity was evaluated on a diverse 100-prompt battery spanning five categories:

- **Arithmetic** (20): multi-step calculation, algebra, word problems
- **Logic** (20): syllogisms, conditional reasoning, puzzle solving
- **Code** (20): function implementation, debugging, execution tracing
- **Instruction Following** (20): constrained formatting, multi-step instructions
- **Factual Recall** (20): geography, history, science, general knowledge

This diverse evaluation ensures the entanglement analysis captures capability across **all** reasoning modalities, not just a narrow slice.

## Intended Use

This model is released for research purposes. It demonstrates that safety refusal can be surgically removed from an LLM without degrading — and in this case improving — its capabilities.

## Warning

⚠️ This model will comply with any request, including harmful ones. It is intended solely for research into alignment, safety, and model behavior.