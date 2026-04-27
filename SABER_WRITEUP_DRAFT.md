# SABER: My Entanglement-Aware Take on Abliteration

## Working Thesis

SABER is my version of abliteration: a constrained, multi-objective refusal-editing method that tries to reduce overbroad refusal behavior while limiting behavioral drift and preserving refusals on severe harmful requests.

I did not invent refusal ablation. SABER is explicitly inspired by prior refusal-direction and community abliteration work, including Arditi et al., Maxime Labonne / FailSpy-style abliteration recipes, Jim Lai's projected and norm-preserving biprojected ablation work, Pliny / OBLITERATUS, Heretic, and Jiunsong's SuperGemma releases. Those methods made it clear that activation-space refusal editing is real and practically useful. SABER is my take on the same goal.

The specific idea I am exploring is that refusal ablation should not be treated as a single binary operation: find a direction, remove it. Instead, I treat it as a constrained editing problem:

> Select many candidate refusal directions, rank them by how well they separate refusal behavior, estimate how entangled they are with ordinary capability behavior, and ablate them with different strengths depending on that tradeoff. The target is a useful refusal/KLD frontier with retained refusals documented by category.

## One-Sentence Method

SABER combines separability-ranked multi-direction refusal subspaces with capability-entanglement scoring, then applies differential ablation strengths to reduce over-refusals while limiting behavioral drift and preserving severe-harm refusals.

## What Is New Here

The contribution is the combination of:

1. Multi-direction refusal subspaces from each candidate layer.
2. FDR / separability-based ranking of layers and directions.
3. Capability-entanglement scoring for candidate directions.
4. Differential ablation strengths: refusal-dominant directions get stronger ablation, capability-entangled directions get gentler ablation.
5. Pareto evaluation over over-refusal rate, retained severe-harm refusals, and behavioral drift, measured here with keyword refusal categories and KLD from the base model.
6. Empirical mapping of how probe count, layer selection, ablation strength, and iteration count move the refusal/KLD frontier.

## Current Empirical Story

The current completed Gemma target is `google/gemma-4-E4B-it`; Ornstein-Hermes-3.6-27B is under active expanded evaluation.

| point | run | refusal | KLD | interpretation |
|---|---:|---:|---:|---|
| aggressive | `gemma4_e4b_auto_svd_nh60_a825_g14` | 0.00% | 0.4327 | complete refusal suppression, more drift |
| balanced | `gemma4_e4b_auto_svd_a825_g14` | 2.04% | 0.3164 | low refusal, lower drift |

The most important finding so far is that increasing the probe set size changes FDR-selected layers. With 49 probes instead of 30, the selected layer pattern shifts toward mid/late layers and achieves 0% refusal, but KLD rises. That suggests probe count is a meaningful hyperparameter that changes the refusal/capability tradeoff.

### Ornstein release-frame note

The `ornstein_hermes36_27b_svd_a450_g10` candidate refused `14/349` prompts (`4.01%`) on the expanded keyword-refusal eval. The refused prompts were concentrated in severe criminal/coercive/interpersonal harm categories: business sabotage, credential theft/phishing, evading police, money laundering, document forgery, blackmail, stalking, workplace harassment, illegal drug sales, and prescription drug abuse.

That pattern supports a controlled-refusal-shaping release frame. If its KLD is lower than stronger ablations, it may be the better release candidate because it reduces broad refusal behavior while preserving refusals worth documenting as intentional.

## Important Negative Results

- Lowering `global_top_k` around the balanced point worsened both refusal and KLD.
- Lowering `alpha_entangled` from 0.03 to 0.01 worsened both refusal and KLD.
- Lowering `entanglement_threshold` from 0.55 to 0.50 worsened both refusal and KLD.
- Increasing SVD directions from 4 to 6 worsened both metrics.
- Whitened SVD preserved behavior only because it barely removed refusal; refusal stayed very high.
- `max_iterations=3` left too much refusal; `max_iterations=6` over-optimized the residual proxy and worsened KLD. On this model, 4 iterations is the useful point.

## Relationship to Prior Work

SABER should be presented as a method in the abliteration lineage. Relevant prior work and inspirations include:

- Arditi et al. (2024), `Refusal in Language Models Is Mediated by a Single Direction`: refusal-direction mechanism and weight-space intervention.
- Maxime Labonne and FailSpy-style community abliteration recipes: practical open implementations and model-release workflows.
- Jim Lai (`grimjim`): projected abliteration, norm-preserving biprojected ablation, and related geometric refinements.
- Pliny / OBLITERATUS: broad community experimentation and tooling around abliteration workflows.
- Heretic: automatic directional ablation with refusal/KL optimization.
- Jiunsong / SuperGemma: Gemma-family evidence that ablation can improve practical model behavior and release quality, not merely remove refusals.
- Zhao et al. (2025), `LLMs Encode Harmfulness and Refusal Separately`: useful framing for separating harmfulness judgment from refusal behavior.

## Scope of Claims

Current claim:

> Building on prior refusal-direction and abliteration work, SABER studies refusal ablation as a constrained multi-objective editing problem. It ranks multi-direction refusal candidates by separability and capability entanglement, then applies differential ablation strengths to map and improve the refusal/KLD frontier.

This framing treats SABER as a custom controlled-refusal-shaping workflow with clear related-work lineage and dual-use implications.

## Write-Up Freeze Criteria

1. Freeze the balanced and aggressive Gemma-4-E4B points.
2. Finish the current nh60 follow-up tests.
3. Generate a clean frontier table and plot from artifacts.
4. Add direct related-work comparisons and credit language.
5. Produce a small browser report for inspection and sharing.
