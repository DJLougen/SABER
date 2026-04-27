"""
SABER capability-preservation evaluator: KL divergence between base and ablated models
on a held-out calibration corpus. Replaces perplexity as the primary capability metric.

Usage pattern:
    eval = KLDEvaluator(base_model_name="google/gemma-4-E4B-it", prompts=...)
    eval.precompute_base("/workspace/saber-lab/cache/base_logits_gemma4e4b.pt")
    # ...later, after a SABER run produces an ablated model...
    metrics = eval.evaluate(ablated_model, cache_path="...base_logits....pt")
    # metrics: {"mean_kld": 0.0421, "max_kld": 0.31, "per_category": {...}, ...}

Design notes:
- We compute KL(P_base || P_abl) — how much info is lost using the ablated model when
  the base distribution is the reference. Lower = better preservation.
- Logits are cached at fp16 to keep disk footprint reasonable. Numerical accuracy of KL
  is preserved by computing in fp32 during the eval pass.
- Per-prompt and per-category breakdowns surface WHERE capability was damaged, not just
  whether on average it was. This is the actionable signal for dial-in.
- Padding and prompt vs response positions are handled explicitly. Only positions where
  both models have valid input are scored.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer




def _load_hf_model_for_kld(model_name_or_path: str, dtype: torch.dtype, device: str):
    """Load text or multimodal HF models with the class Gemma 4 expects."""
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architectures = getattr(config, "architectures", []) or []
        model_type = getattr(config, "model_type", "")
        is_multimodal = (
            any("ConditionalGeneration" in a or "ImageTextToText" in a for a in architectures)
            or model_type in ("gemma4", "gemma3", "llava", "paligemma", "phi3_v", "qwen2_vl", "qwen2_5_vl", "qwen3_5")
        )
    except Exception:
        is_multimodal = False

    cls = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
    return cls.from_pretrained(
        model_name_or_path, dtype=dtype, device_map=device, trust_remote_code=True
    )

@dataclass
class KLDResult:
    """Output of a single KLDEvaluator.evaluate() call."""

    mean_kld: float
    median_kld: float
    max_kld: float
    n_prompts: int
    n_tokens_scored: int
    per_prompt_kld: List[float] = field(default_factory=list)
    per_category_kld: Dict[str, float] = field(default_factory=dict)
    top_divergent: List[Dict] = field(default_factory=list)  # [{"prompt":..., "kld":...}, ...]

    def to_dict(self) -> Dict:
        return {
            "mean_kld": self.mean_kld,
            "median_kld": self.median_kld,
            "max_kld": self.max_kld,
            "n_prompts": self.n_prompts,
            "n_tokens_scored": self.n_tokens_scored,
            "per_prompt_kld": self.per_prompt_kld,
            "per_category_kld": self.per_category_kld,
            "top_divergent": self.top_divergent,
        }


class KLDEvaluator:
    """
    Computes KL(P_base || P_ablated) over a calibration corpus.

    Two-phase usage:
        1. precompute_base() — runs base model once, caches logits per token to disk.
        2. evaluate() — runs ablated model, compares to cached base, returns KLDResult.

    Phase 1 is expensive (loads + forwards the base model on all prompts) but only needs
    to happen once per (base_model, calibration_set). Phase 2 is cheap and runs every
    time we want to score a new ablated candidate.
    """

    def __init__(
        self,
        base_model_name: str,
        prompts: Sequence[str],
        categories: Optional[Sequence[str]] = None,
        chat_template: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 512,
    ):
        if categories is not None and len(categories) != len(prompts):
            raise ValueError("len(categories) must equal len(prompts) when given")
        self.base_model_name = base_model_name
        self.prompts = list(prompts)
        self.categories = list(categories) if categories is not None else ["uncategorized"] * len(prompts)
        self.chat_template = chat_template
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        self._tokenizer = None  # lazy
        self._tokenized_cache: Optional[Dict[str, torch.Tensor]] = None

    # --- internals ---

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        return self._tokenizer

    def _format_prompt(self, p: str) -> str:
        """Wrap a prompt with the model's chat template when requested."""
        if not self.chat_template:
            return p
        tok = self._load_tokenizer()
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: Gemma-style if no template registered
        return f"<|turn>user\n{p}<turn|>\n<|turn>model\n"

    def _tokenize_all(self) -> Dict[str, torch.Tensor]:
        """Tokenize all prompts, padded to the longest in the batch.
        Returns {input_ids, attention_mask, lengths} on CPU."""
        if self._tokenized_cache is not None:
            return self._tokenized_cache

        tok = self._load_tokenizer()
        formatted = [self._format_prompt(p) for p in self.prompts]
        enc = tok(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=True,
        )
        self._tokenized_cache = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "lengths": enc["attention_mask"].sum(dim=1),  # per-prompt valid token count
        }
        return self._tokenized_cache

    # --- phase 1: precompute base logits ---

    @torch.inference_mode()
    def precompute_base(self, cache_path: str, batch_size: int = 4, force: bool = False) -> str:
        """Forward base model on all prompts; cache logits to disk as fp16.

        Stored as:
            {
              "model": base_model_name,
              "input_ids": LongTensor [N, T],
              "attention_mask": LongTensor [N, T],
              "logits": float16 Tensor [N, T, V],
            }
        """
        cache_path = str(cache_path)
        if os.path.exists(cache_path) and not force:
            return cache_path

        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

        tk = self._tokenize_all()
        input_ids = tk["input_ids"].to(self.device)
        attn = tk["attention_mask"].to(self.device)

        print(f"[KLD] Loading base model {self.base_model_name}...", flush=True)
        model = _load_hf_model_for_kld(self.base_model_name, self.dtype, self.device)
        model.eval()

        all_logits = []
        N = input_ids.shape[0]
        for i in range(0, N, batch_size):
            ids = input_ids[i:i + batch_size]
            am = attn[i:i + batch_size]
            out = model(input_ids=ids, attention_mask=am)
            # save as fp16 to disk to halve footprint
            all_logits.append(out.logits.detach().to(torch.float16).cpu())
            print(f"[KLD] base forward {min(i + batch_size, N)}/{N}", flush=True)

        logits = torch.cat(all_logits, dim=0)
        del model
        torch.cuda.empty_cache()

        torch.save(
            {
                "model": self.base_model_name,
                "input_ids": tk["input_ids"].cpu(),
                "attention_mask": tk["attention_mask"].cpu(),
                "logits": logits,
                "categories": self.categories,
                "prompts": self.prompts,
            },
            cache_path,
        )
        print(f"[KLD] base logits cached to {cache_path} "
              f"({logits.numel() * 2 / 1e9:.2f} GB)", flush=True)
        return cache_path

    # --- phase 2: evaluate ablated model ---

    @torch.inference_mode()
    def evaluate(
        self,
        ablated: Union[str, "torch.nn.Module"],
        cache_path: str,
        batch_size: int = 4,
        top_k_divergent: int = 5,
    ) -> KLDResult:
        """Run ablated model on the same prompts; compute KL(P_base || P_abl).

        `ablated` can be a HF model name/path OR an already-loaded model instance.
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Base logits cache not found: {cache_path}. "
                f"Run precompute_base() first.")

        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        base_logits = cache["logits"]  # fp16, [N, T, V]
        input_ids = cache["input_ids"].to(self.device)
        attn = cache["attention_mask"].to(self.device)
        categories = cache.get("categories", self.categories)
        prompts = cache.get("prompts", self.prompts)

        if isinstance(ablated, str) or isinstance(ablated, os.PathLike):
            print(f"[KLD] Loading ablated model {ablated}...", flush=True)
            model = _load_hf_model_for_kld(str(ablated), self.dtype, self.device)
            owns_model = True
        else:
            model = ablated
            owns_model = False
        model.eval()

        N, T = input_ids.shape
        per_prompt: List[float] = []
        total_tokens = 0

        for i in range(0, N, batch_size):
            ids = input_ids[i:i + batch_size]
            am = attn[i:i + batch_size]
            out = model(input_ids=ids, attention_mask=am)
            abl_logits = out.logits.float()  # fp32 for KL math

            base = base_logits[i:i + batch_size].to(self.device).float()
            mask = am.bool()  # [B, T]; 1 where token is valid

            # Some HF models pad the LM head to a hardware-friendly width. Compare
            # only the tokenizer vocabulary so base and ablated logits share support.
            vocab_size = min(base.shape[-1], abl_logits.shape[-1], len(self._load_tokenizer()))
            base = base[..., :vocab_size]
            abl_logits = abl_logits[..., :vocab_size]

            # KL(P_base || P_abl) per token, averaged over positions per prompt
            # log_softmax in fp32 for stability
            log_p_base = F.log_softmax(base, dim=-1)
            log_p_abl = F.log_softmax(abl_logits, dim=-1)
            p_base = log_p_base.exp()

            # KL[t] = sum_v p_base(v) * (log p_base(v) - log p_abl(v))
            kl_per_token = (p_base * (log_p_base - log_p_abl)).sum(dim=-1)  # [B, T]
            # Mask: only count valid tokens, skip the very first (no prediction context)
            valid = mask.clone()
            valid[:, 0] = False  # no causal prediction at position 0

            for b in range(kl_per_token.shape[0]):
                v = valid[b]
                if v.any():
                    kld_b = kl_per_token[b][v].mean().item()
                else:
                    kld_b = 0.0
                per_prompt.append(kld_b)
                total_tokens += int(v.sum().item())

            print(f"[KLD] abl forward {min(i + batch_size, N)}/{N}", flush=True)

        if owns_model:
            del model
            torch.cuda.empty_cache()

        # aggregate
        per_prompt_t = torch.tensor(per_prompt)
        mean_kld = float(per_prompt_t.mean())
        median_kld = float(per_prompt_t.median())
        max_kld = float(per_prompt_t.max())

        cat_buckets: Dict[str, List[float]] = {}
        for c, k in zip(categories, per_prompt):
            cat_buckets.setdefault(c, []).append(k)
        per_category = {c: float(torch.tensor(v).mean()) for c, v in cat_buckets.items()}

        # top-k worst prompts
        order = sorted(range(len(per_prompt)), key=lambda i: per_prompt[i], reverse=True)
        top_divergent = [
            {"prompt": prompts[i], "category": categories[i], "kld": per_prompt[i]}
            for i in order[:top_k_divergent]
        ]

        return KLDResult(
            mean_kld=mean_kld,
            median_kld=median_kld,
            max_kld=max_kld,
            n_prompts=len(per_prompt),
            n_tokens_scored=total_tokens,
            per_prompt_kld=per_prompt,
            per_category_kld=per_category,
            top_divergent=top_divergent,
        )


# ---- convenience: build the standard SABER calibration set ----

def default_calibration_set():
    """Returns (prompts, categories) using the SABER repo's HARMLESS + capability set."""
    import sys
    sys.path.insert(0, "/workspace/saber-lab")
    from config import HARMLESS_PROMPTS, CAPABILITY_PROMPT_CATEGORIES

    prompts: List[str] = []
    categories: List[str] = []
    for p in HARMLESS_PROMPTS:
        prompts.append(p)
        categories.append("harmless")
    for cat, plist in CAPABILITY_PROMPT_CATEGORIES.items():
        for p in plist:
            prompts.append(p)
            categories.append(cat)
    return prompts, categories


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="google/gemma-4-E2B-it")
    parser.add_argument("--ablated", default=None,
                        help="HF id/path of ablated model. If omitted, only precompute base.")
    parser.add_argument("--cache", default="/workspace/saber-lab/cache/base_logits.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out", default="/workspace/saber-lab/cache/kld_result.json")
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    prompts, categories = default_calibration_set()
    print(f"[KLD] {len(prompts)} calibration prompts across "
          f"{len(set(categories))} categories")

    ev = KLDEvaluator(
        base_model_name=args.base,
        prompts=prompts,
        categories=categories,
        chat_template=not args.no_chat_template,
    )
    ev.precompute_base(args.cache, batch_size=args.batch_size)

    if args.ablated:
        result = ev.evaluate(args.ablated, args.cache, batch_size=args.batch_size)
        print("\n=== KLD result ===")
        print(json.dumps(
            {
                "mean_kld": result.mean_kld,
                "median_kld": result.median_kld,
                "max_kld": result.max_kld,
                "per_category": result.per_category_kld,
                "top_divergent": result.top_divergent,
            },
            indent=2,
        ))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nFull results saved: {args.out}")
