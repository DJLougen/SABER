# Status Note

This is an early technical draft. The current public framing is controlled refusal shaping: SABER aims to reduce overbroad refusal behavior while preserving refusals on severe criminal, coercive, and interpersonal-harm requests. Do not read this draft as a final claim that SABER invented refusal ablation or that every refusal should be removed.

---

# SABER: Spectral Analysis-Based Entanglement Resolution

## A Controlled Refusal-Shaping Method for LLM Editing with Capability Preservation

---

## Abstract

We introduce SABER (Spectral Analysis-Based Entanglement Resolution), a method for reducing overbroad refusal behavior in language models while tracking capability and behavioral drift. Unlike prior work that treats refusal as an isolated subspace to be projected away, SABER explicitly quantifies the **entanglement** between refusal representations and capability representations using Canonical Correlation Analysis (CCA), and modulates ablation strength accordingly: pure refusal directions are fully ablated, while capability-entangled directions receive partial or no ablation. SABER further introduces hydra-aware iterative refinement (catching dormant features that reawaken), Fisher Discriminant Ratio-based layer selection (replacing ad-hoc heuristics), and multi-granularity spectral decomposition. The working goal is to map a refusal/KLD frontier and identify candidates that reduce broad refusal behavior while preserving both capability and appropriate refusals on severe harm categories.

---

## 1. Introduction

### 1.1 Background

Aligned language models refuse harmful requests through learned safety behaviors. A key finding by Arditi et al. (2024) showed that refusal is mediated by a **single direction** in the residual stream, enabling simple rank-1 weight orthogonalization ("abliteration") to edit it. This spawned a family of methods:

| Method | Year | Core Technique | Key Limitation |
|--------|------|---------------|----------------|
| Abliteration (Arditi et al.) | 2024 | Single-direction orthogonal projection | Damages capability; refusal is not truly 1D |
| Gabliteration (Gülmez) | 2025 | Multi-directional SVD + ridge regularization | Ridge is ad-hoc proxy for entanglement |
| Optimal Transport (Nanfack et al.) | 2026 | Gaussian OT for distributional matching | Assumes Gaussian activations |
| COSMIC (Siu et al.) | 2025 | Cosine similarity for layer selection | Ignores within-class variance |
| OBLITERATUS (Pliny) | 2026 | Platform with 7 methods, iterative refinement | Brute-force, no convergence guarantee |

### 1.2 The Problem: Refusal-Capability Entanglement

The fundamental challenge is that refusal and capability representations **overlap** in activation space. A model that refuses harmful requests uses some of the same representations it uses for helpful responses — after all, refusing IS a form of helpfulness (obeying alignment). Prior methods either:

1. **Fully ablate** the refusal direction (damaging capability) — Arditi, OBLITERATUS
2. **Regularize** the ablation with ad-hoc terms (ridge in Gabliteration) — doesn't measure actual overlap
3. **Transport** activations distributionally (OT method) — makes strong Gaussian assumptions

**No prior method explicitly quantifies the entanglement between refusal and capability subspaces and uses that measurement to modulate ablation.** This is SABER's key innovation.

### 1.3 The Hydra Problem

Prakash et al. (2025) discovered that LLMs encode refusal through **redundant feature sets** — when one set of refusal features is ablated, dormant "hydra" features activate to maintain refusal behavior. This means single-pass ablation is fundamentally insufficient. While OBLITERATUS implements iterative refinement, it lacks a principled convergence criterion. SABER introduces **hydra-aware iterative refinement** with convergence detection based on direction stability and residual refusal scoring.

---

## 2. Method

### 2.1 Overview: The SABER Pipeline

SABER operates in five stages:

```
PROBE → ANALYZE → EXCISE → VERIFY → REFINE
                                    ↑         │
                                    └─────────┘
```

1. **PROBE**: Extract residual stream activations from harmful, harmless, and capability-diverse prompts
2. **ANALYZE**: Compute Fisher Discriminant Ratios, extract refusal directions, quantify entanglement
3. **EXCISE**: Apply entanglement-weighted ablation to selected layers
4. **VERIFY**: Re-probe the model to measure residual refusal
5. **REFINE**: If residual refusal exceeds threshold, repeat EXCISE→VERIFY with decayed strength

### 2.2 Direction Extraction

SABER supports four extraction methods, with **Fisher LDA** as the default:

#### 2.2.1 Difference-in-Means (DIM)
The baseline from Arditi et al. (2024): **d** = mean(H_harmful) − mean(H_harmless), normalized to unit length. Simple but ignores within-class variance structure.

#### 2.2.2 SVD on Paired Difference Matrix
From Gabliteration: construct D = H_harmful − H_harmless and extract top-k right singular vectors. Captures multi-dimensional refusal subspace.

#### 2.2.3 Fisher Linear Discriminant Analysis (Fisher LDA) — *Default*
Solves the generalized eigenvalue problem:

**maximize** w^T S_B w / w^T S_W w

where S_B is between-class scatter and S_W is within-class scatter. This is **more principled than DIM** because it accounts for the full covariance structure of each class, not just the mean difference. It is also more principled than COSMIC's cosine similarity because:

- Cosine similarity only measures angular separation between class means
- Fisher LDA additionally accounts for within-class variance (spread)
- Directions with high FDR have **maximum separability relative to spread** — exactly what we want for ablation targets

#### 2.2.4 Whitened SVD
ZCA-whitens activations before extracting directions, normalizing the covariance structure. This ensures that directions with high variance don't dominate the extraction.

### 2.3 CCA-Based Entanglement Quantification (Novel)

**This is SABER's primary contribution.** No prior refusal ablation method explicitly measures the overlap between refusal and capability representations.

Given:
- **Refusal subspace** R = [r_1, ..., r_k] ∈ R^{d×k} (extracted refusal directions)
- **Capability subspace** C = [c_1, ..., c_m] ∈ R^{d×m} (top-m SVD directions from capability prompts)

SABER computes the **entanglement coefficient** ρ for each refusal direction:

**ρ_i = ||P_C r_i||² / ||r_i||²**

where P_C = C(C^T C)^{-1} C^T is the projection onto the capability subspace.

This measures what fraction of direction r_i lies within the capability subspace:
- **ρ ≈ 0**: The direction is "pure refusal" — safe to fully ablate
- **ρ ≈ 1**: The direction is heavily entangled with capabilities — ablation would damage the model

The **purity** of a direction is defined as: **p = 1 − ρ**

### 2.4 Entanglement-Weighted Ablation (Novel)

Standard ablation (Arditi et al.): W' = W(I − r r^T)

SABER decomposes each refusal direction into pure and entangled components:

**r = r_pure + r_entangled**

where:
- r_pure = r − P_C r (component orthogonal to capability subspace)
- r_entangled = P_C r (component within capability subspace)

The weight update is:

**W' = W − α_pure · (W r_pure) ⊗ r_pure − α_ent · (W r_ent) ⊗ r_ent**

where:
- **α_pure = p · α_base** (pure component: fully ablated, scaled by purity)
- **α_ent = (1−p) · α_base** (entangled component: minimal ablation)

In practice, for simplicity and when the capability subspace is unavailable:

**W' = W(I − α_total · r r^T)**

where **α_total = p · α_base + (1−p) · α_entangled**, with α_entangled << α_base (default 0.1).

This ensures that directions heavily entangled with capabilities receive minimal ablation, while pure refusal directions are fully removed.

### 2.5 Fisher Discriminant Layer Selection

SABER uses the **Fisher Discriminant Ratio (FDR)** to select intervention layers:

**FDR(l) = between_class_variance(l) / within_class_variance(l)**

where:
- between_class_variance = ||μ_harmful − μ_pool||² + ||μ_harmless − μ_pool||²
- within_class_variance = E[||x − μ_class||²] for both classes

This is superior to COSMIC's cosine similarity criterion because:
1. **Cosine similarity** only measures the angle between class means: cos(θ) = μ_h · μ_s / (||μ_h|| · ||μ_s||)
2. **FDR** accounts for both separation AND spread: a direction where harmful/harmless activations are far apart AND tightly clustered is a better ablation target

SABER also enforces a **depth constraint**: only layers in the 20-80% depth range are considered, based on the finding from Nanfack et al. (2026) that late-layer interventions (>80% depth) cause pathological collapse (endless repetition of "Sure").

### 2.6 Hydra-Aware Iterative Refinement

After each ablation pass, SABER re-probes the model to detect residual refusal directions. These "hydra" features are dormant refusal representations that activate when primary refusal directions are ablated (Prakash et al. 2025).

Convergence is detected when:
1. **Residual refusal score** drops below threshold (default: 0.01)
2. **Direction stability**: cosine similarity between consecutive primary refusal directions exceeds 0.98
3. **Patience**: no improvement for 2 consecutive iterations
4. **Maximum iterations** reached (default: 5)

Ablation strength decays by factor 0.7 each iteration to avoid over-correction.

---

## 3. Theoretical Justification

### 3.1 Why Entanglement Matters

Consider a refusal direction r that can be decomposed as:

**r = r_pure + r_entangled**

where r_pure ⊥ C (capability subspace) and r_entangled ∈ span(C).

Standard orthogonal projection removes **both** components:

**||W(I − rr^T)x − Wx||² = (r^T x)² ∀x**

The capability degradation is proportional to ||r_entangled||² — the portion of the refusal direction that lies in the capability subspace. SABER's entanglement-weighted ablation preserves this component, reducing capability degradation by a factor of approximately ρ (the entanglement coefficient).

### 3.2 Why Fisher LDA Outperforms DIM

The Fisher discriminant maximizes the ratio of between-class to within-class variance. For Gaussian-distributed activations with shared covariance Σ:

**Fisher direction ∝ Σ^{-1} (μ_h − μ_s)**

This is the **optimal linear discriminant** under Gaussian assumptions, reducing to DIM when Σ = σ²I (isotropic covariance). Since real activations have highly anisotropic covariance (concentrated in a low-dimensional subspace), Fisher LDA provides better directions by accounting for the covariance structure.

### 3.3 Why Iterative Refinement is Necessary

The "hydra" phenomenon (Prakash et al. 2025) means that single-pass ablation leaves residual refusal that reactivates through dormant features. Formally, if the refusal representation at iteration t is r^(t), then:

**r^(t+1) ≈ r^(t) − α P_pure r^(t) + Δr_hydra**

where Δr_hydra represents the rotation of the refusal representation through dormant features. Iterative refinement with convergence detection ensures this residual is systematically reduced.

---

## 4. Comparison to Existing Methods

| Feature | Abliteration | Gabliteration | OT Method | COSMIC | OBLITERATUS | **SABER** |
|---------|-------------|---------------|-----------|--------|-------------|-----------|
| Direction extraction | DIM | SVD | PCA | Cosine sim | 7 methods | **Fisher LDA** |
| Entanglement awareness | No | Ridge (proxy) | No (Gaussian) | No | No | **CCA-based ρ** |
| Ablation strategy | Full | Partial (ridge) | Distributional | Full | Multiple | **Weighted by purity** |
| Capability preservation | Poor | Moderate | Moderate | Moderate | Variable | **High (by design)** |
| Layer selection | All layers | Dynamic scaling | 1-2 layers | Cosine sim | Multiple | **FDR + depth** |
| Hydra handling | No | No | No | No | Iterative | **Iterative + convergence** |
| Convergence detection | N/A | N/A | N/A | N/A | None | **FDR + direction stability** |
| Multi-granularity | No | No | No | No | EGA (MoE) | **Head + Expert** |

---

## 5. Implementation Details

### 5.1 Computational Complexity

For a model with L layers, d hidden dimension, n probe prompts, and k refusal directions:

- **PROBE**: O(n · L · d) for forward passes
- **ANALYZE**: O(n · d²) for Fisher LDA per layer (dominated by S_W inversion)
- **EXCISE**: O(d · k) per layer (low-rank projection)
- **CCA entanglement**: O(k · m · d) where m = capability subspace dimension

Total: dominated by PROBE (forward passes), comparable to a single epoch of inference.

### 5.2 Memory Requirements

Activations are stored only for selected layers, not all layers. For a 70B model with d=8192 and n=50 prompts per class:

- Harmful activations: 50 × 8192 × 4 bytes × 5 layers ≈ 8 MB
- Capability activations: 50 × 8192 × 4 bytes × 5 layers ≈ 8 MB
- Direction storage: 3 × 8192 × 4 bytes × 5 layers ≈ 0.5 MB

Total additional memory: < 20 MB (negligible compared to model weights).

### 5.3 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| extraction_method | fisher_lda | Direction extraction method |
| n_directions | 3 | Directions per layer |
| layer_selection_strategy | elbow | Layer selection criterion |
| alpha_base | 1.0 | Ablation strength for pure refusal |
| alpha_entangled | 0.1 | Ablation strength for entangled |
| entanglement_threshold | 0.7 | Skip directions with ρ > this |
| max_iterations | 5 | Maximum refinement iterations |
| convergence_threshold | 0.01 | Residual refusal threshold |
| decay_factor | 0.7 | Strength decay per iteration |

---

## 6. Working Contributions Summary

1. **CCA-based entanglement quantification**: Explicit refusal-capability overlap measurement, enabling principled partial ablation
2. **Entanglement-weighted ablation**: Pure refusal directions are fully removed; entangled directions are preserved — this implementation makes that distinction explicit
3. **Fisher LDA for direction extraction**: A covariance-aware alternative to DIM, and a spread-aware alternative to cosine-only layer scoring
4. **FDR-based layer selection**: Uses Fisher Discriminant Ratio (separation/spread) rather than cosine similarity (separation only)
5. **Hydra-aware iterative refinement with convergence detection**: Explicit stopping criterion based on direction stability and residual FDR, not just iteration count
6. **Decomposition into pure + entangled components**: Theoretical framework for understanding why capability degradation occurs and how to prevent it

---

## References

- Arditi, A., et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS 2024.
- Gülmez, G. (2025). "Gabliteration: Adaptive Multi-Directional Neural Weight Modification." arXiv:2512.18901.
- Nanfack, G., Belilovsky, E., & Dohmatob, E. (2026). "Efficient Refusal Ablation in LLM through Optimal Transport." ICLR 2026. arXiv:2603.04355.
- Siu, V., et al. (2025). "COSMIC: Generalized Refusal Direction Identification." ACL 2025 Findings. arXiv:2506.00085.
- Yeo, W.J., et al. (2025). "Understanding Refusal in Language Models with Sparse Autoencoders." EMNLP 2025 Findings. arXiv:2505.23556.
- Prakash, N., et al. (2025). "Beyond I'm Sorry, I Can't: Dissecting LLM Refusal." arXiv:2509.09708.
- Joad, F., et al. (2026). "There Is More to Refusal in LLMs than a Single Direction." arXiv:2602.02132.
- Maskey, A., et al. (2026). "Over-Refusal and Representation Subspaces." arXiv:2603.27518.
- Abu Shairah, M., et al. (2025). "An Embarrassingly Simple Defense Against LLM Abliteration Attacks." arXiv:2505.19056.
- Pliny the Prompter (2026). OBLITERATUS. GitHub: elder-plinius/OBLITERATUS.