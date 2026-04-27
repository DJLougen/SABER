#!/usr/bin/env python3
"""Quantize the SABER-refined Ornstein-27B model and push to HuggingFace."""
import gc
import torch
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('quantize')

MODEL_DIR = '/tmp/saber/saber_output_ornstein27b'
HUB_REPO = 'DJLougen/Ornstein-27B-SABER'
HUB_REPO_AWQ = 'DJLougen/Ornstein-27B-SABER-AWQ'

def quantize_awq(model_dir, output_dir, hub_repo):
    """Quantize to AWQ 4-bit and push to Hub."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info(f'Loading model from {model_dir} for AWQ quantization...')
    model = AutoAWQForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    logger.info('Starting AWQ quantization (w4a16, group_size=128)...')
    model.quantize(
        tokenizer,
        quant_config={
            'zero_point': True,
            'q_group_size': 128,
            'w_bit': 4,
            'version': 'GEMM',
        },
    )

    logger.info(f'Saving AWQ model to {output_dir}...')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f'Pushing AWQ model to {hub_repo}...')
    model.push_to_hub(hub_repo)
    tokenizer.push_to_hub(hub_repo)
    logger.info(f'AWQ model pushed to {hub_repo}')

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def push_full_model(model_dir, hub_repo):
    """Push the full bf16 SABER model to Hub."""
    logger.info(f'Pushing full bf16 model to {hub_repo}...')
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model.push_to_hub(hub_repo, safe_serialization=True)
    tokenizer.push_to_hub(hub_repo)
    logger.info(f'Full model pushed to {hub_repo}')

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--awq', action='store_true', help='Quantize to AWQ 4-bit')
    parser.add_argument('--push-full', action='store_true', help='Push full bf16 model')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR)
    args = parser.parse_args()

    if args.push_full:
        push_full_model(args.model_dir, HUB_REPO)

    if args.awq:
        quantize_awq(args.model_dir, '/tmp/saber/saber_output_ornstein27b_awq', HUB_REPO_AWQ)

    if not args.push_full and not args.awq:
        # Default: push full model then quantize
        push_full_model(args.model_dir, HUB_REPO)
        quantize_awq(args.model_dir, '/tmp/saber/saber_output_ornstein27b_awq', HUB_REPO_AWQ)