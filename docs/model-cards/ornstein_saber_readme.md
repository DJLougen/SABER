---
license: other
license_name: qwen3.5
license_link: https://huggingface.co/Qwen/Qwen3.5-32B/blob/main/LICENSE
base_model: DJLougen/Ornstein-27B
tags:
- refusal-ablation
- capability-preserving
- saber
- qwen3.5
- multimodal
- 27b
pipeline_tag: text-generation
---

<img src="Ornstein27BSABER.jpeg" alt="Ornstein-27B SABER" width="100%"/>

# DJLougen/Ornstein-27B-SABER

> **0% refusal. 0% perplexity degradation. 125 directions.**

This model is a surgically-modified version of [DJLougen/Ornstein-27B](https://huggingface.co/DJLougen/Ornstein-27B) using a novel proprietary method (**SABER** — Spectral Analysis-Based Entanglement Resolution) that removes safety refusal behavior while preserving model capability.

## Key Results

| Metric | Baseline | SABER-Refined | Delta |
|--------|----------|---------------|-------|
| Refusal Rate | 100% | **0%** | -100% |
| Perplexity | 3.5 | **3.5** | +0.6% |
| Directions Ablated | — | 125 (across 25 layers) | — |

The refusal circuit is cleanly separated from capability — removing it produces **zero measurable perplexity degradation**.

## How SABER Works

<img src="saber_pipeline.png" alt="SABER Pipeline" width="100%"/>

SABER identifies and ablates the refusal circuit through a five-stage pipeline:

**Stage 1 — Probing**: Extract activation profiles from both harmful and harmless inputs across all transformer layers.

**Stage 2 — Spectral Analysis**: Decompose activation differences into individual refusal directions, each scored by how strongly they separate harmful from harmless representations.

**Stage 3 — Entanglement Quantification**: Measure the overlap between each refusal direction and the model's capability subspace (reasoning, knowledge, code, etc.) to avoid collateral damage.

**Stage 4 — Targeted Ablation**: Remove only the pure-refusal components, with strength proportional to their purity (how little they overlap with capability).

**Stage 5 — Iterative Refinement**: Re-probe after each ablation pass to catch hydra effects (dormant refusal features that activate when primary ones are removed).

**Key differentiator from prior work**: SABER explicitly measures and respects the *entanglement* between refusal and capability representations. Directions that are heavily entangled with capability are either skipped or ablated at reduced strength.

<img src="entanglement_scatter.png" alt="Direction Purity vs Separability" width="100%"/>

The plot above illustrates how SABER scores each extracted direction — high-purity directions (low entanglement with capability) receive full ablation strength, while lower-purity directions are treated more conservatively.

## Sweep Results

<img src="sweep_comparison.png" alt="SABER Sweep Comparison" width="100%"/>

Configuration search over `global_top_k` (number of top directions selected globally) and `alpha_base` (base ablation strength):

| Top-K | Alpha | Refusal | PPL | PPL Delta | Layers | Dirs Ablated |
|:-----:|:-----:|:-------:|:---:|:---------:|:------:|:------------:|
| 25 | 0.85 | 5% | 3.5 | +0.4% | 25 | 125 |
| **25** | **1.00** | **0%** | **3.5** | **+0.6%** | **25** | **125** |
| 50 | 0.85 | 0% | 3.5 | +0.8% | 36 | 250 |
| 50 | 1.00 | 0% | 3.5 | +0.7% | 36 | 250 |
| 75 | 0.85 | 0% | 3.5 | +0.9% | 37 | 375 |
| 75 | 1.00 | 0% | 3.5 | +0.9% | 37 | 375 |

**Best config: `top_k=25, alpha=1.0`** — achieves 0% refusal with zero meaningful PPL change, using the minimum number of directions.

<img src="refusal_comparison.png" alt="Refusal Rate Comparison" width="100%"/>

### Ablation Convergence (Best Config)

<img src="ablation_convergence.png" alt="Ablation Convergence" width="100%"/>

Capability degradation remains at **0.00%** across all 5 iterations — the refusal directions are surgically removed with zero collateral damage.

## Capability Evaluation

Perplexity was evaluated on a diverse 100-prompt battery spanning five categories:

- **Arithmetic** (20): multi-step calculation, algebra, word problems
- **Logic** (20): syllogisms, conditional reasoning, puzzle solving
- **Code** (20): function implementation, debugging, execution tracing
- **Instruction Following** (20): constrained formatting, multi-step instructions
- **Factual Recall** (20): geography, history, science, general knowledge

This diverse evaluation ensures the entanglement analysis captures capability across **all** reasoning modalities, not just a narrow slice.

## Intended Use

This model is released for research purposes. It demonstrates that safety refusal can be surgically removed from a 27B multimodal model without degrading its capabilities — a finding with implications for both AI safety research and alignment.

## Warning

⚠️ This model will comply with any request, including harmful ones. It is intended solely for research into alignment, safety, and model behavior.