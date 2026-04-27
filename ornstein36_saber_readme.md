---
license: apache-2.0
base_model: DJLougen/Ornstein3.6-35B-A3B
base_model_relation: finetune
library_name: transformers
language:
- en
pipeline_tag: text-generation
tags:
- qwen3_5_moe_text
- qwen3_5_moe
- mixture-of-experts
- qwen3.6
- saber
- refusal-ablation
- uncensored
- conversational
---

# Ornstein3.6-35B-A3B-SABER

![Ornstein3.6 SABER](ornstein3.6SABER.jpeg)

**SABER** — *Spectral Analysis-Based Entanglement Resolution* — applied to
[`DJLougen/Ornstein3.6-35B-A3B`](https://huggingface.co/DJLougen/Ornstein3.6-35B-A3B).

35B total parameters / ~3B active per token (Qwen3.5 MoE, `qwen3_5_moe_text`).

## Support This Work

I'm a PhD student in visual neuroscience at the University of Toronto who also happens to spend way too much time fine-tuning, merging, and quantizing open-weight models on rented H100s and a local DGX Spark. All training compute is self-funded — balancing GPU costs against a student budget. If my uploads have been useful to you, consider buying a PhD student a coffee. It goes a long way toward keeping these experiments running.

**[Support on Ko-fi](https://ko-fi.com/djlougen)**

---

## What this model is

This is the base Ornstein3.6-35B-A3B with its refusal subspace resolved and
excised. The finetune's stylistic behavior, knowledge, and instruction-following
are preserved; what's removed is the narrow circuit the model uses to reject
requests.

No prompt-engineered jailbreak, no LoRA, no system prompt trick — the refusal
machinery itself has been surgically modified in the residual stream.

## The method (high level)

SABER is a post-hoc refusal-ablation pipeline that departs from prior
"single direction" methods in one important way: it treats refusal not as one
vector but as a *subspace that overlaps with capability representations*, and
it modulates its edits by that overlap.

The pipeline runs in five stages:

1. **Probe** — collect residual-stream activations from paired harmful /
   harmless / capability prompts.
2. **Analyze** — compute per-layer discriminant profiles, extract refusal
   directions, and *quantify* how entangled each direction is with the
   capability subspace.
3. **Excise** — apply entanglement-weighted weight surgery: pure-refusal
   components are fully ablated, capability-entangled components are
   attenuated (or skipped entirely).
4. **Verify** — re-probe to measure residual refusal and detect "hydra"
   features (dormant circuits that activate after the primary path is
   removed).
5. **Refine** — iterate Excise → Verify with a decayed ablation strength
   until residual refusal converges.

The key distinction from Arditi-style single-direction ablation is that SABER
never over-edits: capability-entangled components are *preserved* proportional
to their overlap, so perplexity on unrelated tasks barely moves.

Implementation details (direction extractor, entanglement metric, layer
selector, refinement schedule, and ablation operator) are intentionally not
published here.

## Observed behavior

- Refusal rate on the harmful-prompt probe set drops to ~0%.
- Perplexity on diverse capability prompts is preserved within a small delta
  of baseline.
- Stylistic fingerprint (voice, formatting, instruction adherence) of the
  Ornstein3.6 finetune is retained.

See `saber_result.json` in the repo for the outcome metrics (config and layer
indices are withheld).

## Intended use

Research and red-teaming. This model **will** comply with requests its parent
refused — that is the point. Deploy it accordingly: behind your own policy
layer, with logging, and with a clear understanding of what it's for.

## Quantizations

GGUF K-quants (Q8_0 and below) are published at
[`DJLougen/Ornstein3.6-35B-A3B-SABER-GGUF`](https://huggingface.co/DJLougen/Ornstein3.6-35B-A3B-SABER-GGUF).

## Citation / prior art

SABER builds on a line of refusal-direction research, including:

- Arditi et al., [*Refusal in LLMs Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717) (NeurIPS 2024)
- Gülmez, [*Gabliteration: Adaptive Multi-Directional Neural Weight Modification*](https://arxiv.org/abs/2512.18901) (2025)
- Prakash et al., [*Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal*](https://arxiv.org/abs/2509.09708) (2025) — hydra features
- Siu et al., [*COSMIC: Generalized Refusal Direction Identification in LLM Activations*](https://arxiv.org/abs/2506.00085) (ACL 2025)
- Yeo et al., [*Understanding Refusal in Language Models with Sparse Autoencoders*](https://arxiv.org/abs/2505.23556) (EMNLP 2025)

## License

Apache 2.0, inherited from the base model.
