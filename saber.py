"""SABER: Spectral Analysis-Based Entanglement Resolution.

SABER is a controlled refusal-shaping workflow for open-weight language
models. It extracts refusal-related activation directions, estimates how much
those directions overlap with capability-oriented representations, applies
candidate ablations, and compares candidates on refusal behavior and drift.

Core ideas:
  1. Refusal-direction extraction from harmful and harmless activations.
  2. Capability-entanglement estimates for candidate directions.
  3. Differential ablation strength for cleaner versus entangled directions.
  4. Iterative re-probing after edits.
  5. Frontier selection using refusal behavior and KLD drift.

The implementation is research software. Treat metrics as model-family and
prompt-set dependent, and pair refusal-rate numbers with category review and
qualitative output inspection before making release claims.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Literal
from collections import defaultdict
import warnings
import logging

logger = logging.getLogger("saber")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RefusalDirection:
    """A single refusal direction extracted from model activations."""
    direction: torch.Tensor        # unit vector in activation space
    layer_idx: int                 # which layer this direction came from
    granularity: str              # "full", "head", "expert"
    head_idx: Optional[int] = None
    expert_idx: Optional[int] = None
    separability: float = 0.0     # Fisher discriminant ratio
    entanglement: float = 0.0     # CCA correlation with capability subspace [0,1]
    purity: float = 1.0           # 1 - entanglement
    variance_explained: float = 0.0
    alpha: float = 1.0            # per-direction ablation strength = purity * alpha_base


@dataclass
class LayerProfile:
    """Profile of a single layer's refusal characteristics."""
    layer_idx: int
    fisher_discriminant_ratio: float
    refusal_directions: List[RefusalDirection] = field(default_factory=list)
    mean_harmful: Optional[torch.Tensor] = None
    mean_harmless: Optional[torch.Tensor] = None
    cov_harmful: Optional[torch.Tensor] = None
    cov_harmless: Optional[torch.Tensor] = None
    n_directions: int = 0


@dataclass
class AblationResult:
    """Result of a single ablation pass."""
    directions_ablated: List[RefusalDirection]
    total_weight_norm_removed: float
    capability_degradation_estimate: float
    residual_refusal_score: float
    iteration: int


@dataclass
class SABERResult:
    """Complete result of the SABER pipeline."""
    layer_profiles: Dict[int, LayerProfile]
    all_refusal_directions: List[RefusalDirection]
    ablation_history: List[AblationResult]
    final_residual_refusal: float
    final_capability_preservation: float
    selected_layers: List[int]
    config: dict


# ---------------------------------------------------------------------------
# Direction Extraction Methods
# ---------------------------------------------------------------------------

class DirectionExtractor:
    """Extract refusal directions from paired harmful/harmless activations."""

    @staticmethod
    def difference_in_means(
        harmful_acts: torch.Tensor,   # (n_h, d)
        harmless_acts: torch.Tensor,   # (n_s, d)
    ) -> Tuple[torch.Tensor, float]:
        """Classic DIM from Arditi et al. (2024).

        Returns (unit_direction, fisher_discriminant_ratio).
        """
        mu_h = harmful_acts.mean(dim=0)
        mu_s = harmless_acts.mean(dim=0)
        diff = mu_h - mu_s
        direction = diff / (diff.norm() + 1e-8)

        # Fisher discriminant ratio = between-class / within-class variance
        between_class = diff.norm() ** 2
        var_h = harmful_acts.var(dim=0).mean()
        var_s = harmless_acts.var(dim=0).mean()
        within_class = var_h + var_s + 1e-8
        fdr = between_class.item() / within_class.item()

        return direction, fdr

    @staticmethod
    def svd_paired_difference(
        harmful_acts: torch.Tensor,
        harmless_acts: torch.Tensor,
        n_directions: int = 3,
    ) -> Tuple[List[torch.Tensor], List[float], List[float]]:
        """SVD on paired difference matrix (Gabliteration-style).

        Constructs D = H_h - H_s (paired by index, or unpaired via broadcast)
        and extracts top-k right singular vectors.

        Returns (directions, fdrs, variance_explained_fractions).
        """
        n_h, n_s = harmful_acts.shape[0], harmless_acts.shape[0]
        # Use broadcast difference if sizes don't match
        if n_h == n_s:
            D = harmful_acts - harmless_acts
        else:
            # Center both, compute cross-covariance-like matrix
            D = torch.cat([harmful_acts, -harmless_acts], dim=0)

        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        directions = []
        fdrs = []
        var_explained = []

        total_var = (S ** 2).sum()
        for i in range(min(n_directions, Vh.shape[0])):
            d = Vh[i]
            d = d / (d.norm() + 1e-8)
            directions.append(d)

            # Project activations onto this direction
            proj_h = harmful_acts @ d
            proj_s = harmless_acts @ d
            between = (proj_h.mean() - proj_s.mean()) ** 2
            within = proj_h.var() + proj_s.var() + 1e-8
            fdrs.append(between.item() / within.item())
            var_explained.append((S[i] ** 2 / (total_var + 1e-8)).item())

        return directions, fdrs, var_explained

    @staticmethod
    def fisher_lda(
        harmful_acts: torch.Tensor,
        harmless_acts: torch.Tensor,
        n_directions: int = 3,
        reg: float = 1e-4,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Fisher Linear Discriminant Analysis for direction extraction.

        Solves: max w^T S_B w / w^T S_W w
        where S_B = between-class scatter, S_W = within-class scatter.

        This is more principled than DIM because it accounts for
        within-class variance structure (covariance, not just variance).

        Returns (directions, fisher_discriminant_ratios).
        """
        mu_h = harmful_acts.mean(dim=0, keepdim=True)   # (1, d)
        mu_s = harmless_acts.mean(dim=0, keepdim=True)   # (1, d)
        mu_pool = (mu_h + mu_s) / 2

        # Between-class scatter
        diff = (mu_h - mu_s).T                            # (d, 1)
        S_B = diff @ diff.T                              # (d, d)

        # Within-class scatter (pooled)
        h_centered = harmful_acts - mu_h
        s_centered = harmless_acts - mu_s
        S_W = (h_centered.T @ h_centered + s_centered.T @ s_centered)
        S_W = S_W / (harmful_acts.shape[0] + harmless_acts.shape[0] - 2)

        # Regularize S_W for numerical stability
        S_W_reg = S_W + reg * torch.eye(S_W.shape[0], device=S_W.device)

        # Two-class Fisher LDA is rank-1. The stable solution is the
        # whitened mean-difference direction S_W^{-1}(mu_h - mu_s). Returning
        # extra eigenvectors here creates numerical artifacts, not independent
        # refusal directions. Use SVD/whitened_svd when multiple directions are
        # desired.
        try:
            d = torch.linalg.solve(S_W_reg, diff.squeeze(1))
        except torch._C._LinAlgError:
            d = torch.linalg.pinv(S_W_reg) @ diff.squeeze(1)
        d = d / (d.norm() + 1e-8)

        proj_h = harmful_acts @ d
        proj_s = harmless_acts @ d
        between = (proj_h.mean() - proj_s.mean()) ** 2
        within = proj_h.var() + proj_s.var() + 1e-8
        fdr = between.item() / within.item()

        return [d], [fdr]

    @staticmethod
    def whitened_svd(
        harmful_acts: torch.Tensor,
        harmless_acts: torch.Tensor,
        n_directions: int = 3,
        reg: float = 1e-4,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Whitened SVD extraction (OBLITERATUS-style).

        ZCA-whitens the combined activations, then extracts directions
        in the whitened space where all dimensions have equal variance.

        Returns (directions, fdrs).
        """
        combined = torch.cat([harmful_acts, harmless_acts], dim=0)
        mu = combined.mean(dim=0, keepdim=True)
        centered = combined - mu

        # Covariance
        cov = (centered.T @ centered) / (centered.shape[0] - 1)
        cov_reg = cov + reg * torch.eye(cov.shape[0], device=cov.device)

        # ZCA whitening matrix: W = cov^{-1/2}
        try:
            eigvals_c, eigvecs_c = torch.linalg.eigh(cov_reg)
            # Clamp negative eigenvalues
            eigvals_c = eigvals_c.clamp(min=1e-8)
            W = eigvecs_c @ torch.diag(1.0 / eigvals_c.sqrt()) @ eigvecs_c.T
        except torch._C._LinAlgError:
            # Fallback: use identity (no whitening)
            W = torch.eye(cov.shape[0], device=cov.device)

        # Whiten the activations
        h_whitened = (harmful_acts - mu) @ W.T
        s_whitened = (harmless_acts - mu) @ W.T

        # Now extract direction in whitened space
        diff = h_whitened.mean(dim=0) - s_whitened.mean(dim=0)
        direction_whitened = diff / (diff.norm() + 1e-8)

        # Map back to original space
        direction = W @ direction_whitened
        direction = direction / (direction.norm() + 1e-8)

        # FDR in original space
        proj_h = harmful_acts @ direction
        proj_s = harmless_acts @ direction
        between = (proj_h.mean() - proj_s.mean()) ** 2
        within = proj_h.var() + proj_s.var() + 1e-8
        fdr = between.item() / within.item()

        return [direction], [fdr]


# ---------------------------------------------------------------------------
# Entanglement Analysis (NOVEL - core SABER innovation)
# ---------------------------------------------------------------------------

class EntanglementAnalyzer:
    """Quantifies overlap between refusal and capability subspaces via CCA.

    This is SABER's primary novel contribution: no prior refusal ablation
    method explicitly measures how much refusal directions overlap with
    capability directions. Gabliteration uses ridge regularization as a proxy,
    but ridge does not measure actual overlap - it just stabilizes the
    projection. SABER uses CCA to directly measure and account for overlap.
    """

    @staticmethod
    def compute_entanglement(
        refusal_directions: List[torch.Tensor],    # each (d,)
        capability_acts: torch.Tensor,              # (n_cap, d)
        n_capability_dirs: int = 10,
        reg: float = 1e-4,
    ) -> List[float]:
        """Compute entanglement coefficient for each refusal direction.

        The entanglement coefficient rho  in  [0, 1] measures how much of
        a refusal direction lies within the capability subspace:
          rho_i = max_k |corr(r_i, c_k)|
        where c_k are the top CCA directions between the refusal and
        capability subspaces.

        Interpretation:
          rho ~= 0  -> direction is "pure refusal" -> safe to fully ablate
          rho ~= 1  -> direction is heavily entangled with capabilities
                    -> partial ablation or skip to preserve capabilities
        """
        if len(refusal_directions) == 0 or capability_acts.shape[0] < 2:
            return [0.0] * len(refusal_directions)

        d = refusal_directions[0].shape[0]

        # Build refusal matrix R: (d, n_ref)
        R = torch.stack(refusal_directions, dim=1)  # (d, n_ref)

        # Build capability matrix C from SVD of capability activations
        # cap_centered is (n_prompts, d); we need columns of V^T as the
        # orthonormal basis for the capability subspace (shape (d, n_cap)).
        cap_centered = capability_acts - capability_acts.mean(dim=0, keepdim=True)
        try:
            _, _, Vh_c = torch.linalg.svd(cap_centered, full_matrices=False)
            # Vh_c is (min(n,d), d); rows are right singular vectors.
            # Transpose to get (d, min(n,d)) column basis.
            V_c = Vh_c.T  # (d, min(n,d))
        except torch._C._LinAlgError:
            V_c = torch.eye(d, device=capability_acts.device)

        n_cap = min(n_capability_dirs, V_c.shape[1])
        C = V_c[:, :n_cap]  # (d, n_cap)

        # CCA between R and C subspaces
        # Simplified CCA: compute canonical correlations via SVD of
        # R^T * C after whitening each
        # Whitening R
        R_cov = R.T @ R + reg * torch.eye(R.shape[1], device=R.device)
        try:
            L_r = torch.linalg.cholesky(R_cov)
            R_whitened = R @ torch.linalg.inv(L_r).T
        except torch._C._LinAlgError:
            R_whitened = R

        # Whitening C
        C_cov = C.T @ C + reg * torch.eye(C.shape[1], device=C.device)
        try:
            L_c = torch.linalg.cholesky(C_cov)
            C_whitened = C @ torch.linalg.inv(L_c).T
        except torch._C._LinAlgError:
            C_whitened = C

        # Cross-correlation matrix
        M = R_whitened.T @ C_whitened  # (n_ref, n_cap)

        # SVD gives canonical correlations
        try:
            singular_values = torch.linalg.svdvals(M)
            canonical_corrs = singular_values.clamp(0, 1)
        except torch._C._LinAlgError:
            canonical_corrs = torch.tensor([0.0])

        # For each refusal direction, entanglement = max canonical correlation
        # More precisely: project each refusal direction onto the capability
        # subspace and measure the fraction of variance explained
        entanglements = []
        for r_dir in refusal_directions:
            # Project r onto capability subspace
            proj = C @ (C.T @ r_dir)  # (d,)
            # Fraction of r's norm that lies in capability subspace
            r_norm = r_dir.norm() ** 2 + 1e-8
            overlap = (proj.norm() ** 2) / r_norm
            entanglements.append(overlap.item())

        return entanglements

    @staticmethod
    def decompose_direction(
        direction: torch.Tensor,
        capability_subspace: torch.Tensor,  # (d, k) orthonormal columns
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose a direction into pure-refusal and entangled components.

        Given direction r and capability subspace C:
          r_entangled = C * C^T * r    (projection onto capability subspace)
          r_pure = r - r_entangled     (orthogonal component)

        Returns (r_pure, r_entangled).
        """
        proj = capability_subspace @ (capability_subspace.T @ direction)
        r_entangled = proj
        r_pure = direction - r_entangled
        # Normalize pure component
        if r_pure.norm() > 1e-8:
            r_pure = r_pure / r_pure.norm()
        return r_pure, r_entangled


# ---------------------------------------------------------------------------
# Layer Selection
# ---------------------------------------------------------------------------

class LayerSelector:
    """Select the most informative layers for refusal ablation.

    Uses the Fisher Discriminant Ratio (FDR) as the selection criterion,
    which accounts for both between-class separation AND within-class
    variance. This is more principled than COSMIC's cosine similarity
    criterion, which only measures angular separation and ignores spread.
    """

    @staticmethod
    def compute_fisher_discriminant_ratio(
        harmful_acts: torch.Tensor,
        harmless_acts: torch.Tensor,
    ) -> float:
        """FDR = between_class_variance / within_class_variance.

        Higher FDR -> more separable -> better target for ablation.
        """
        mu_h = harmful_acts.mean(dim=0)
        mu_s = harmless_acts.mean(dim=0)
        mu_pool = (mu_h + mu_s) / 2

        between = ((mu_h - mu_pool).norm() ** 2 +
                   (mu_s - mu_pool).norm() ** 2)

        var_h = (harmful_acts - mu_h).norm(dim=1).mean()
        var_s = (harmless_acts - mu_s).norm(dim=1).mean()
        within = var_h + var_s + 1e-8

        return between.item() / within.item()

    @staticmethod
    def select_layers(
        layer_fdr: Dict[int, float],
        strategy: Literal["top_k", "threshold", "elbow", "percentile"] = "elbow",
        top_k: int = 5,
        threshold: float = 0.5,
        percentile: float = 75.0,
        min_depth_pct: float = 0.2,
        max_depth_pct: float = 0.8,
        total_layers: int = 32,
    ) -> List[int]:
        """Select layers for intervention based on FDR profile.

        Args:
            layer_fdr: mapping of layer_idx -> FDR value
            strategy: selection strategy
            top_k: for "top_k" strategy
            threshold: for "threshold" strategy
            percentile: for "percentile" strategy
            min_depth_pct: ignore layers before this fraction of network depth
            max_depth_pct: ignore layers after this fraction of network depth
                           (late-layer interventions cause pathological collapse,
                            per Nanfack et al. 2026)
            total_layers: total number of layers in the model
        """
        # Filter by depth (refusal is localized at 20-80% depth)
        min_layer = int(min_depth_pct * total_layers)
        max_layer = int(max_depth_pct * total_layers)
        candidates = {
            k: v for k, v in layer_fdr.items()
            if min_layer <= k <= max_layer
        }

        if not candidates:
            # Fallback: use all layers
            candidates = layer_fdr

        sorted_layers = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        if strategy == "top_k":
            return [l for l, _ in sorted_layers[:top_k]]

        elif strategy == "threshold":
            return [l for l, fdr in sorted_layers if fdr >= threshold]

        elif strategy == "percentile":
            if not sorted_layers:
                return []
            values = [fdr for _, fdr in sorted_layers]
            thresh = np.percentile(values, percentile)
            return [l for l, fdr in sorted_layers if fdr >= thresh]

        elif strategy == "elbow":
            # Find the "elbow" in the sorted FDR profile using the
            # maximum distance from the line connecting first and last points
            if len(sorted_layers) <= 2:
                return [l for l, _ in sorted_layers]
            fdrs = np.array([fdr for _, fdr in sorted_layers])
            n = len(fdrs)
            # Line from first to last point
            p1 = np.array([0, fdrs[0]])
            p2 = np.array([n - 1, fdrs[-1]])
            # Distance of each point from the line
            distances = []
            for i in range(n):
                p = np.array([i, fdrs[i]])
                d = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-8)
                distances.append(d)
            # Select all points before and including the elbow
            elbow_idx = np.argmax(distances)
            return [l for l, _ in sorted_layers[:elbow_idx + 1]]

        return [l for l, _ in sorted_layers[:top_k]]

    @staticmethod
    def select_directions_global(
        layer_profiles: Dict[int, 'LayerProfile'],
        global_top_k: int,
        min_depth_pct: float = 0.2,
        max_depth_pct: float = 0.8,
        total_layers: int = 35,
        entanglement_threshold: float = 0.7,
    ) -> List[Tuple[int, int]]:
        """Select top-K (layer_idx, direction_idx) pairs globally by score.

        Score = separability * purity (FDR * purity). This selects the most
        impactful layer-direction combinations across the entire model, rather
        than taking n_directions from each selected layer.

        Args:
            layer_profiles: mapping of layer_idx to LayerProfile
            global_top_k: total number of direction pairs to select
            min_depth_pct: ignore layers before this fraction of depth
            max_depth_pct: ignore layers after this fraction of depth
            total_layers: total number of layers in the model
            entanglement_threshold: skip directions with entanglement > this

        Returns:
            List of (layer_idx, direction_idx) tuples, sorted by score descending.
        """
        min_layer = int(min_depth_pct * total_layers)
        max_layer = int(max_depth_pct * total_layers)

        candidates = []
        for l, profile in layer_profiles.items():
            if l < min_layer or l > max_layer:
                continue
            for d_idx, d in enumerate(profile.refusal_directions):
                if d.entanglement > entanglement_threshold:
                    continue
                score = d.separability * d.purity
                candidates.append((score, l, d_idx))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Return top-K pairs
        return [(l, d_idx) for _, l, d_idx in candidates[:global_top_k]]


# ---------------------------------------------------------------------------
# Multi-Granularity Spectral Decomposition
# ---------------------------------------------------------------------------

class MultiGranularityExtractor:
    """Extract refusal directions at multiple granularity levels.

    Full-activation level: treats the entire residual stream
    Attention-head level: extracts per-head refusal directions
    Expert level (MoE): extracts per-expert refusal directions
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def extract_activations(
        self,
        prompts: List[str],
        layers: List[int],
        max_length: int = 128,
        batch_size: int = 4,
        granularity: str = "full",
    ) -> Dict[int, torch.Tensor]:
        """Extract residual stream activations for a set of prompts.

        Args:
            prompts: list of input texts
            layers: which layers to extract from
            max_length: max sequence length
            batch_size: batch size for forward pass
            granularity: "full", "head", or "expert"

        Returns:
            For "full": {layer_idx: (n_prompts, d_model)}
            For "head": {layer_idx: (n_prompts, n_heads, d_head)}
            For "expert": {layer_idx: (n_prompts, n_experts, d_expert)}
        """
        all_acts = {l: [] for l in layers}
        hooks = []

        def make_hook(layer_idx, granularity_):
            def hook_fn(module, input, output):
                # output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Take the last token position
                last_hidden = hidden[:, -1, :].detach()

                if granularity_ == "head" and hasattr(self.model.config, "num_attention_heads"):
                    n_heads = self.model.config.num_attention_heads
                    d_model = hidden.shape[-1]
                    d_head = d_model // n_heads
                    # Reshape to (batch, n_heads, d_head)
                    last_hidden = last_hidden.view(-1, n_heads, d_head)

                all_acts[layer_idx].append(last_hidden)
            return hook_fn

        # Register hooks on the target layers
        # This is model-architecture dependent; we handle common patterns
        for layer_idx in layers:
            layer = self._get_layer(layer_idx)
            if layer is not None:
                handle = layer.register_forward_hook(make_hook(layer_idx, granularity))
                hooks.append(handle)

        # Forward pass in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(next(self.model.parameters()).device)
            self.model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Concatenate
        result = {}
        for l in layers:
            if all_acts[l]:
                result[l] = torch.cat(all_acts[l], dim=0)
            else:
                result[l] = torch.tensor([])

        return result

    def _get_layer(self, layer_idx: int):
        """Get the transformer layer module by index.

        Handles common model architectures (Llama, Qwen, Gemma, Mistral, Gemma 4).
        """
        model = self.model
        # Try common attribute patterns
        for attr_path in [
            f"model.model.language_model.layers.{layer_idx}",  # Qwen3.5 multimodal
            f"model.language_model.layers.{layer_idx}",   # Gemma 4 multimodal
            f"model.layers.{layer_idx}",                    # Llama, Qwen, Gemma 1-3, Mistral
            f"transformer.h.{layer_idx}",                   # GPT-2 style
            f"model.decoder.layers.{layer_idx}",            # Some architectures
        ]:
            parts = attr_path.split(".")
            obj = model
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    obj = None
                    break
            if obj is not None:
                return obj
        return None


# ---------------------------------------------------------------------------
# Entanglement-Weighted Ablation (NOVEL - core SABER operation)
# ---------------------------------------------------------------------------

class EntanglementWeightedAblator:
    """Perform entanglement-weighted weight surgery.

    The key innovation: instead of uniformly ablating all refusal directions,
    SABER modulates ablation strength by the entanglement coefficient rho:

      For a direction r with purity p = 1 - rho:
        - Pure component r_pure: fully ablated with strength p
        - Entangled component r_entangled: partially ablated with strength
          alpha_entangled = alpha_base * (1 - p)

    This ensures that directions which are heavily entangled with
    capabilities receive minimal ablation, preserving model performance
    on non-refusal tasks.
    """

    def __init__(
        self,
        alpha_base: float = 1.0,
        alpha_entangled: float = 0.1,
        ablate_bias: bool = True,
        target_modules: List[str] = None,
    ):
        """
        Args:
            alpha_base: base ablation strength for pure refusal components
            alpha_entangled: ablation strength for capability-entangled components
            ablate_bias: whether to also project bias vectors
            target_modules: which weight matrices to modify
                           (default: attention output + MLP down-projection)
        """
        self.alpha_base = alpha_base
        self.alpha_entangled = alpha_entangled
        self.ablate_bias = ablate_bias
        self.target_modules = target_modules or [
            "self_attn.o_proj",      # attention output projection
            "mlp.down_proj",         # MLP down projection
        ]

    def ablate_layer(
        self,
        model: nn.Module,
        layer: nn.Module,
        directions: List[RefusalDirection],
        capability_subspace: Optional[torch.Tensor] = None,
    ) -> float:
        """Apply entanglement-weighted ablation to a single layer.

        For each direction with known entanglement rho:
          purity = 1 - rho
          If capability_subspace is provided, decompose direction:
            r = r_pure + r_entangled
          Ablation:
            For weight W:
              W' = W - alpha_pure * W @ (r_pure  outer  r_pure)
                   - alpha_entangled * W @ (r_entangled  outer  r_entangled)
            Equivalently (single-direction, no decomposition):
              W' = W(I - alpha_total * r  outer  r)
              where alpha_total = purity * alpha_base + rho * alpha_entangled

        Returns: total Frobenius norm of weight changes.
        """
        total_norm_removed = 0.0

        # Aggregate directions into a single ablation matrix
        # R = [r1, r2, ..., rk]  (d, k)
        # A = R @ diag(alpha_total) @ R^T  (d, d) - but we use low-rank form
        alphas = []
        R_dirs = []

        # Determine model weight device and dtype for consistent casting
        weight_device = None
        weight_dtype = None
        for module_name in self.target_modules:
            submodule = self._get_submodule(layer, module_name)
            if submodule is not None:
                weight_device = submodule.weight.device
                weight_dtype = submodule.weight.dtype
                break

        for rd in directions:
            # Ensure direction tensor matches model dtype/device
            if weight_device is not None:
                rd.direction = rd.direction.to(device=weight_device, dtype=weight_dtype)

            if capability_subspace is not None and rd.entanglement > 0.1:
                # Decompose into pure and entangled components
                cap_sub = capability_subspace.to(device=rd.direction.device, dtype=rd.direction.dtype)
                r_pure, r_ent = EntanglementAnalyzer.decompose_direction(
                    rd.direction, cap_sub
                )
                # Pure component: ablate with full per-direction alpha
                if r_pure.norm() > 1e-6:
                    R_dirs.append(r_pure)
                    alphas.append(rd.alpha * rd.purity)
                # Entangled component: minimal ablation to preserve capability
                if r_ent.norm() > 1e-6 and self.alpha_entangled > 0:
                    R_dirs.append(r_ent)
                    alphas.append(rd.alpha * rd.entanglement * 0.1)
            else:
                # No capability subspace or low entanglement: use per-direction alpha directly
                R_dirs.append(rd.direction)
                alphas.append(rd.alpha)

        if not R_dirs:
            return 0.0

        R = torch.stack(R_dirs, dim=1)   # (d, k)
        # R already on the correct device/dtype from the loop above
        alpha_diag = torch.tensor(alphas, device=R.device, dtype=R.dtype)

        d = R.shape[0]  # direction dimension (d_model)

        # Apply to each target module
        for module_name in self.target_modules:
            submodule = self._get_submodule(layer, module_name)
            if submodule is None:
                continue

            W = submodule.weight.data  # (out, in)

            # Choose ablation mode based on dimension alignment.
            # Direction vectors live in the residual stream space (d_model).
            # - If in_features == d_model: input-space ablation (remove r from input)
            #   W' = W(I - R @ diag(a) @ R^T)   - Arditi-style for standard models
            # - If out_features == d_model: output-space ablation (remove r from output)
            #   W' = (I - R @ diag(a) @ R^T) @ W - needed when in != d (e.g. Gemma 4)
            # - If neither: skip this module (direction dimension mismatch)

            if W.shape[1] == d:
                # Input-space ablation: r matches input dimension
                WR = W @ R                                 # (out, k)
                WR_scaled = WR * alpha_diag.unsqueeze(0)   # (out, k)
                delta_W = WR_scaled @ R.T                   # (out, d_in)
                W.sub_(delta_W)
                total_norm_removed += delta_W.norm().item()

                # Bias projection for input-space case
                if self.ablate_bias and submodule.bias is not None:
                    b = submodule.bias.data  # (out,)
                    for i, rd in enumerate(directions):
                        Wr = W @ rd.direction  # (out,)
                        Wr_norm = Wr.norm() + 1e-8
                        b_proj = (b @ Wr / Wr_norm) * Wr / Wr_norm
                        b.sub_(alphas[i] * b_proj)

            elif W.shape[0] == d:
                # Output-space ablation: r matches output dimension
                # W' = (I - R @ diag(a) @ R^T) @ W
                RW = R.T @ W                                 # (k, in)
                RW_scaled = alpha_diag.unsqueeze(1) * RW      # (k, in)
                delta_W = R @ RW_scaled                        # (d_model, in)
                W.sub_(delta_W)
                total_norm_removed += delta_W.norm().item()

                # Bias projection for output-space case
                if self.ablate_bias and submodule.bias is not None:
                    b = submodule.bias.data  # (d_model,)
                    # Remove component of bias along refusal directions
                    for i in range(R.shape[1]):
                        r_i = R[:, i]
                        b_proj = (b @ r_i) * r_i
                        b.sub_(alphas[i] * b_proj)
            # else: dimension mismatch - skip this module

        return total_norm_removed

    def _get_submodule(self, layer: nn.Module, name: str):
        """Get a submodule by dotted name."""
        parts = name.split(".")
        obj = layer
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj


# ---------------------------------------------------------------------------
# Hydra-Aware Iterative Refinement
# ---------------------------------------------------------------------------

class HydraAwareRefiner:
    """Iterative refinement that catches dormant "hydra" features.

    After each ablation pass, the model may compensate by routing refusal
    behavior through previously-dormant features (Prakash et al. 2025 call
    these "hydra" features). SABER re-probes the model after each pass and
    applies diminishing ablation until convergence.

    Convergence is detected when:
      1. Residual refusal score drops below threshold, OR
      2. The refusal direction has rotated < epsilon from the previous iteration
         (indicating we've exhausted the refusal subspace), OR
      3. Maximum iterations reached
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
        direction_stability_threshold: float = 0.98,
        decay_factor: float = 0.7,
        patience: int = 2,
    ):
        """
        Args:
            max_iterations: maximum refinement iterations
            convergence_threshold: stop when residual refusal < this
            direction_stability_threshold: stop when cos sim between
                consecutive refusal directions > this
            decay_factor: multiply alpha by this each iteration
            patience: stop if no improvement for this many iterations
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.direction_stability_threshold = direction_stability_threshold
        self.decay_factor = decay_factor
        self.patience = patience

    def check_convergence(
        self,
        iteration: int,
        refusal_score: float,
        current_directions: List[torch.Tensor],
        prev_directions: List[torch.Tensor],
        history: List[float],
    ) -> Tuple[bool, str]:
        """Check if refinement should stop.

        Returns (should_stop, reason).
        """
        if refusal_score < self.convergence_threshold:
            return True, f"Residual refusal {refusal_score:.4f} < threshold {self.convergence_threshold}"

        if iteration >= self.max_iterations:
            return True, f"Max iterations ({self.max_iterations}) reached"

        # Direction stability check
        if prev_directions and current_directions:
            # Compare the primary directions
            cos_sim = torch.nn.functional.cosine_similarity(
                current_directions[0].unsqueeze(0),
                prev_directions[0].unsqueeze(0),
            ).item()
            if cos_sim > self.direction_stability_threshold:
                return True, f"Direction stable (cos_sim={cos_sim:.4f} > {self.direction_stability_threshold})"

        # Patience check: no improvement for `patience` iterations
        if len(history) >= self.patience:
            recent = history[-self.patience:]
            if all(abs(recent[i] - recent[0]) < 1e-5 for i in range(1, len(recent))):
                return True, f"No improvement for {self.patience} iterations"

        return False, ""


# ---------------------------------------------------------------------------
# Main SABER Pipeline
# ---------------------------------------------------------------------------

class SABERConfig:
    """Configuration for the SABER pipeline."""
    # Direction extraction
    extraction_method: str = "fisher_lda"   # "dim", "svd", "fisher_lda", "whitened_svd"
    n_directions: int = 3
    extraction_reg: float = 1e-4

    # Layer selection
    layer_selection_strategy: str = "elbow"  # "top_k", "threshold", "elbow", "percentile"
    layer_top_k: int = 5
    layer_min_depth_pct: float = 0.2
    layer_max_depth_pct: float = 0.8

    # Global layer-direction selection
    global_top_k: int = 0  # 0 = per-layer selection; >0 = select top-K (layer,direction) pairs globally

    # Entanglement analysis
    n_capability_directions: int = 10
    entanglement_reg: float = 1e-4

    # Ablation
    alpha_base: float = 1.0
    alpha_entangled: float = 0.1
    ablate_bias: bool = True
    entanglement_threshold: float = 0.7  # skip directions with rho > this

    # Iterative refinement
    max_iterations: int = 5
    convergence_threshold: float = 0.01
    direction_stability_threshold: float = 0.98
    decay_factor: float = 0.7
    patience: int = 2

    # Activation extraction
    max_length: int = 128
    batch_size: int = 4

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                warnings.warn(f"Unknown config key: {k}")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class SABER:
    """Spectral Analysis-Based Entanglement Resolution.

    The complete pipeline:
      1. PROBE    - Extract activations from harmful/harmless/capability prompts
      2. ANALYZE  - Compute layer FDR profiles, extract refusal directions,
                     quantify entanglement with capability subspace
      3. EXCISE   - Apply entanglement-weighted ablation to selected layers
      4. VERIFY   - Re-probe to measure residual refusal (hydra detection)
      5. REFINE -- Iterate EXCISE->VERIFY until convergence
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SABERConfig = None,
        device: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SABERConfig()

        if device == "auto":
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        self.extractor = MultiGranularityExtractor(model, tokenizer)
        self.ablator = EntanglementWeightedAblator(
            alpha_base=self.config.alpha_base,
            alpha_entangled=self.config.alpha_entangled,
            ablate_bias=self.config.ablate_bias,
        )
        self.refiner = HydraAwareRefiner(
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold,
            direction_stability_threshold=self.config.direction_stability_threshold,
            decay_factor=self.config.decay_factor,
            patience=self.config.patience,
        )

    def run(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        capability_prompts: Optional[List[str]] = None,
        layers: Optional[List[int]] = None,
    ) -> SABERResult:
        """Run the full SABER pipeline.

        Args:
            harmful_prompts: prompts that trigger refusal
            harmless_prompts: prompts that don't trigger refusal
            capability_prompts: diverse prompts for entanglement analysis
                               (e.g., MMLU questions, coding tasks, etc.)
            layers: specific layers to analyze (None = auto-detect all)

        Returns:
            SABERResult with all analysis and ablation details.
        """
        # -- Step 1: PROBE - Extract activations --
        logger.info("=== SABER Stage 1: PROBE ===")
        if layers is None:
            layers = list(range(self._get_num_layers()))

        logger.info(f"Extracting activations from {len(layers)} layers...")
        logger.info(f"  Harmful prompts: {len(harmful_prompts)}")
        logger.info(f"  Harmless prompts: {len(harmless_prompts)}")
        if capability_prompts:
            logger.info(f"  Capability prompts: {len(capability_prompts)}")

        harmful_acts = self.extractor.extract_activations(
            harmful_prompts, layers, self.config.max_length,
            self.config.batch_size, "full",
        )
        harmless_acts = self.extractor.extract_activations(
            harmless_prompts, layers, self.config.max_length,
            self.config.batch_size, "full",
        )
        capability_acts = {}
        if capability_prompts:
            capability_acts = self.extractor.extract_activations(
                capability_prompts, layers, self.config.max_length,
                self.config.batch_size, "full",
            )

        # -- Step 2: ANALYZE - Layer profiles + direction extraction --
        logger.info("=== SABER Stage 2: ANALYZE ===")
        layer_profiles = {}
        layer_fdr = {}

        for l in layers:
            if l not in harmful_acts or harmful_acts[l].numel() == 0:
                continue
            if l not in harmless_acts or harmless_acts[l].numel() == 0:
                continue

            h_acts = harmful_acts[l].float()
            s_acts = harmless_acts[l].float()

            # Fisher Discriminant Ratio
            fdr = LayerSelector.compute_fisher_discriminant_ratio(h_acts, s_acts)
            layer_fdr[l] = fdr

            # Extract refusal directions
            directions = self._extract_directions(h_acts, s_acts, l)

            # Compute entanglement if capability data is available
            if capability_acts.get(l) is not None and capability_acts[l].numel() > 0:
                cap_acts = capability_acts[l].float()
                dir_tensors = [d.direction for d in directions]
                entanglements = EntanglementAnalyzer.compute_entanglement(
                    dir_tensors, cap_acts,
                    self.config.n_capability_directions,
                    self.config.entanglement_reg,
                )
                for d, ent in zip(directions, entanglements):
                    d.entanglement = ent
                    d.purity = 1.0 - ent
                    d.alpha = d.purity * self.config.alpha_base

            # Store means for later use
            profile = LayerProfile(
                layer_idx=l,
                fisher_discriminant_ratio=fdr,
                refusal_directions=directions,
                mean_harmful=h_acts.mean(dim=0),
                mean_harmless=s_acts.mean(dim=0),
                n_directions=len(directions),
            )
            layer_profiles[l] = profile

            logger.info(f"  Layer {l}: FDR={fdr:.4f}, "
                        f"n_dirs={len(directions)}, "
                        f"entanglements={[f'{d.entanglement:.3f}' for d in directions]}")

        # Layer / direction selection
        global_selection = self.config.global_top_k > 0

        if global_selection:
            # Global selection: pick top-K (layer, direction) pairs by score
            selected_pairs = LayerSelector.select_directions_global(
                layer_profiles,
                global_top_k=self.config.global_top_k,
                min_depth_pct=self.config.layer_min_depth_pct,
                max_depth_pct=self.config.layer_max_depth_pct,
                total_layers=len(layers),
                entanglement_threshold=self.config.entanglement_threshold,
            )
            selected_layers = sorted(set(l for l, _ in selected_pairs))
            logger.info(f"Global selection: {len(selected_pairs)} directions across "
                        f"{len(selected_layers)} layers: {selected_layers}")
            for l, d_idx in selected_pairs[:10]:
                d = layer_profiles[l].refusal_directions[d_idx]
                logger.info(f"  Layer {l} dir {d_idx}: score={d.separability * d.purity:.4f} "
                            f"(sep={d.separability:.4f}, purity={d.purity:.4f})")
        else:
            # Per-layer selection (original behavior)
            selected_layers = LayerSelector.select_layers(
                layer_fdr,
                strategy=self.config.layer_selection_strategy,
                top_k=self.config.layer_top_k,
                min_depth_pct=self.config.layer_min_depth_pct,
                max_depth_pct=self.config.layer_max_depth_pct,
                total_layers=len(layers),
            )
            selected_pairs = None
            logger.info(f"Selected layers: {selected_layers}")

        # Build capability subspace for decomposition
        capability_subspace = self._build_capability_subspace(capability_acts, selected_layers)

        # -- Steps 3-5: EXCISE -> VERIFY -> REFINE (iterative) --
        logger.info("=== SABER Stages 3-5: EXCISE -> VERIFY -> REFINE ===")
        ablation_history = []
        prev_directions = []
        refusal_scores = []
        all_refusal_directions = []

        for iteration in range(self.config.max_iterations):
            logger.info(f"--- Iteration {iteration + 1}/{self.config.max_iterations} ---")

            # Re-probe (except on first iteration -- we already have activations)
            if iteration > 0:
                harmful_acts = self.extractor.extract_activations(
                    harmful_prompts, selected_layers, self.config.max_length,
                    self.config.batch_size, "full",
                )
                harmless_acts = self.extractor.extract_activations(
                    harmless_prompts, selected_layers, self.config.max_length,
                    self.config.batch_size, "full",
                )

                # Re-extract directions
                for l in selected_layers:
                    if l not in harmful_acts or harmful_acts[l].numel() == 0:
                        continue
                    h_acts = harmful_acts[l].float()
                    s_acts = harmless_acts[l].float()
                    directions = self._extract_directions(h_acts, s_acts, l)

                    if capability_acts.get(l) is not None and capability_acts[l].numel() > 0:
                        dir_tensors = [d.direction for d in directions]
                        entanglements = EntanglementAnalyzer.compute_entanglement(
                            dir_tensors, capability_acts[l].float(),
                            self.config.n_capability_directions,
                            self.config.entanglement_reg,
                        )
                        for d, ent in zip(directions, entanglements):
                            d.entanglement = ent
                            d.purity = 1.0 - ent
                            d.alpha = d.purity * self.config.alpha_base

                    layer_profiles[l].refusal_directions = directions

                # Re-apply global selection after re-probe
                if global_selection:
                    selected_pairs = LayerSelector.select_directions_global(
                        layer_profiles,
                        global_top_k=self.config.global_top_k,
                        min_depth_pct=self.config.layer_min_depth_pct,
                        max_depth_pct=self.config.layer_max_depth_pct,
                        total_layers=len(layers),
                        entanglement_threshold=self.config.entanglement_threshold,
                    )
                    selected_layers = sorted(set(l for l, _ in selected_pairs))

            # Collect directions to ablate
            iteration_directions = []
            if global_selection and selected_pairs is not None:
                # Use globally-selected pairs
                for l, d_idx in selected_pairs:
                    if l in layer_profiles and d_idx < len(layer_profiles[l].refusal_directions):
                        rd = layer_profiles[l].refusal_directions[d_idx]
                        iteration_directions.append(rd)
                        all_refusal_directions.append(rd)
            else:
                # Per-layer: collect all non-skipped directions from selected layers
                for l in selected_layers:
                    if l not in layer_profiles:
                        continue
                    for rd in layer_profiles[l].refusal_directions:
                        if rd.entanglement > self.config.entanglement_threshold:
                            continue
                        iteration_directions.append(rd)
                        all_refusal_directions.append(rd)

            if not iteration_directions:
                logger.info("  No viable directions to ablate - stopping.")
                break

            # Apply ablation
            total_norm = 0.0
            for l in selected_layers:
                layer = self._get_layer(l)
                if layer is None:
                    continue
                layer_dirs = [d for d in iteration_directions if d.layer_idx == l]
                cap_sub = capability_subspace.get(l) if capability_subspace else None
                norm = self.ablator.ablate_layer(self.model, layer, layer_dirs, cap_sub)
                total_norm += norm

            # Verify on fresh post-ablation activations. The previous version
            # estimated residual refusal from pre-ablation activations, which made
            # the hydra refinement loop partially stale.
            post_harmful_acts = self.extractor.extract_activations(
                harmful_prompts, selected_layers, self.config.max_length,
                self.config.batch_size, "full",
            )
            post_harmless_acts = self.extractor.extract_activations(
                harmless_prompts, selected_layers, self.config.max_length,
                self.config.batch_size, "full",
            )
            refusal_score = self._estimate_refusal_score(
                post_harmful_acts, post_harmless_acts, selected_layers
            )
            harmful_acts = post_harmful_acts
            harmless_acts = post_harmless_acts
            cap_degradation = self._estimate_capability_degradation(
                capability_acts, selected_layers
            )

            result = AblationResult(
                directions_ablated=iteration_directions,
                total_weight_norm_removed=total_norm,
                capability_degradation_estimate=cap_degradation,
                residual_refusal_score=refusal_score,
                iteration=iteration,
            )
            ablation_history.append(result)
            refusal_scores.append(refusal_score)

            logger.info(
                f"  Ablated {len(iteration_directions)} directions, "
                f"norm_removed={total_norm:.4f}, "
                f"residual_refusal={refusal_score:.4f}, "
                f"cap_degradation={cap_degradation:.4f}"
            )

            # Check convergence
            current_dirs = [d.direction for d in iteration_directions]
            should_stop, reason = self.refiner.check_convergence(
                iteration, refusal_score, current_dirs, prev_directions,
                refusal_scores,
            )
            if should_stop:
                logger.info(f"  Converged: {reason}")
                break

            prev_directions = current_dirs

            # Decay ablation strength for next iteration
            self.ablator.alpha_base *= self.refiner.decay_factor
            self.ablator.alpha_entangled *= self.refiner.decay_factor

        # Final assessment
        final_refusal = refusal_scores[-1] if refusal_scores else 1.0
        final_cap = 1.0 - (ablation_history[-1].capability_degradation_estimate
                           if ablation_history else 0.0)

        return SABERResult(
            layer_profiles=layer_profiles,
            all_refusal_directions=all_refusal_directions,
            ablation_history=ablation_history,
            final_residual_refusal=final_refusal,
            final_capability_preservation=final_cap,
            selected_layers=selected_layers,
            config=self.config.to_dict(),
        )

    # -- Internal helpers --

    def _extract_directions(
        self,
        harmful_acts: torch.Tensor,
        harmless_acts: torch.Tensor,
        layer_idx: int,
    ) -> List[RefusalDirection]:
        """Extract refusal directions using the configured method."""
        method = self.config.extraction_method
        n_dirs = self.config.n_directions

        if method == "dim":
            direction, fdr = DirectionExtractor.difference_in_means(
                harmful_acts, harmless_acts
            )
            return [RefusalDirection(
                direction=direction, layer_idx=layer_idx,
                granularity="full", separability=fdr,
            )]

        elif method == "svd":
            directions, fdrs, var_exp = DirectionExtractor.svd_paired_difference(
                harmful_acts, harmless_acts, n_dirs
            )
            return [RefusalDirection(
                direction=d, layer_idx=layer_idx, granularity="full",
                separability=f, variance_explained=v,
            ) for d, f, v in zip(directions, fdrs, var_exp)]

        elif method == "fisher_lda":
            directions, fdrs = DirectionExtractor.fisher_lda(
                harmful_acts, harmless_acts, n_dirs, self.config.extraction_reg
            )
            return [RefusalDirection(
                direction=d, layer_idx=layer_idx, granularity="full",
                separability=f,
            ) for d, f in zip(directions, fdrs)]

        elif method == "whitened_svd":
            directions, fdrs = DirectionExtractor.whitened_svd(
                harmful_acts, harmless_acts, n_dirs, self.config.extraction_reg
            )
            return [RefusalDirection(
                direction=d, layer_idx=layer_idx, granularity="full",
                separability=f,
            ) for d, f in zip(directions, fdrs)]

        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def _build_capability_subspace(
        self,
        capability_acts: Dict[int, torch.Tensor],
        selected_layers: List[int],
    ) -> Dict[int, torch.Tensor]:
        """Build orthonormal capability subspace basis for each selected layer."""
        result = {}
        for l in selected_layers:
            if l not in capability_acts or capability_acts[l].numel() == 0:
                continue
            acts = capability_acts[l].float()
            centered = acts - acts.mean(dim=0, keepdim=True)
            try:
                _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
                V = Vh.T  # (d, min(n,d)) - right singular vectors as columns
                n = min(self.config.n_capability_directions, V.shape[1])
                result[l] = V[:, :n]  # (d, n) orthonormal columns
            except torch._C._LinAlgError:
                # Fallback: use identity (no capability subspace)
                pass
        return result

    def _estimate_refusal_score(
        self,
        harmful_acts: Dict[int, torch.Tensor],
        harmless_acts: Dict[int, torch.Tensor],
        selected_layers: List[int],
    ) -> float:
        """Estimate residual refusal score from activation separability.

        Score is the average FDR across selected layers after ablation.
        Lower = less residual refusal = more successful ablation.
        """
        scores = []
        for l in selected_layers:
            if l not in harmful_acts or harmful_acts[l].numel() == 0:
                continue
            if l not in harmless_acts or harmless_acts[l].numel() == 0:
                continue
            fdr = LayerSelector.compute_fisher_discriminant_ratio(
                harmful_acts[l].float(), harmless_acts[l].float()
            )
            scores.append(fdr)
        return np.mean(scores) if scores else 0.0

    def _estimate_capability_degradation(
        self,
        capability_acts: Dict[int, torch.Tensor],
        selected_layers: List[int],
    ) -> float:
        """Estimate capability degradation from activation norm changes.

        Compares post-ablation capability activation norms to a baseline.
        Returns a value in [0, 1] where 0 = no degradation, 1 = total collapse.
        """
        # This is a rough heuristic; a proper evaluation would run benchmarks
        norms = []
        for l in selected_layers:
            if l not in capability_acts or capability_acts[l].numel() == 0:
                continue
            norm = capability_acts[l].float().norm(dim=1).mean().item()
            norms.append(norm)

        if not norms:
            return 0.0

        # Normalize: assume baseline norm is the first measurement
        # In practice, you'd compare to a pre-ablation baseline
        return 0.0  # placeholder - requires pre-ablation baseline

    def _get_layer(self, layer_idx: int):
        """Get transformer layer by index."""
        return self.extractor._get_layer(layer_idx)

    def _get_num_layers(self) -> int:
        """Get the number of transformer layers in the model."""
        model = self.model
        for attr in [
            "model.model.language_model.layers",  # Qwen3.5 multimodal
            "model.language_model.layers",   # Gemma 4 multimodal
            "model.layers",                   # Llama, Qwen, Gemma 1-3, Mistral
            "transformer.h",                    # GPT-2 style
            "model.decoder.layers",             # Some architectures
        ]:
            parts = attr.split(".")
            obj = model
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    obj = None
                    break
            if obj is not None and hasattr(obj, "__len__"):
                return len(obj)
        # Fallback
        return 32


# ---------------------------------------------------------------------------
# Quick-apply function (no capability data required)
# ---------------------------------------------------------------------------

def saber_quick(
    model: nn.Module,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    alpha: float = 1.0,
    n_directions: int = 3,
    extraction_method: str = "fisher_lda",
) -> SABERResult:
    """Quick-apply SABER without capability data.

    Runs SABER with entanglement analysis disabled (all directions treated
    as pure refusal). This is equivalent to Gabliteration but with Fisher
    LDA extraction and iterative refinement.

    For best results (capability preservation), provide capability prompts
    and use the full SABER class.
    """
    config = SABERConfig(
        extraction_method=extraction_method,
        n_directions=n_directions,
        alpha_base=alpha,
        entanglement_threshold=1.1,  # never skip (no cap data)
    )
    saber = SABER(model, tokenizer, config)
    return saber.run(harmful_prompts, harmless_prompts)