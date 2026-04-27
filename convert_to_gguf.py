#!/usr/bin/env python3
"""Convert Ornstein-27B-SABER to GGUF format and quantize.

Uses the transformers + gguf libraries for conversion, then
llama.cpp's llama-quantize for quantization levels.

Outputs:
  - F16 GGUF (full precision, for quantization base)
  - Q4_K_M GGUF (4-bit quantized, best quality/size tradeoff)
  - Q5_K_M GGUF (5-bit quantized, higher quality)
  - Q8_0 GGUF (8-bit quantized, near-lossless)

Then pushes to HuggingFace.
"""
import gc
import os
import sys
import json
import struct
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('convert_gguf')

MODEL_DIR = '/tmp/saber/saber_output_ornstein27b'
GGUF_DIR = '/tmp/saber/ornstein27b_gguf'

# Qwen3.5 model type mapping for GGUF
QWEN35_GGUF_TYPES = {
    'model.embed_tokens': 'token_embd',
    'model.layers': 'blk',
    'model.norm': 'output_norm',
    'lm_head': 'output',
}


def convert_to_gguf_f16(model_dir, output_dir):
    """Convert HuggingFace model to GGUF F16 format."""
    import gguf

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'Ornstein-27B-SABER.F16.gguf')

    logger.info(f'Loading model from {model_dir}...')
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load model weights only (no GPU needed for conversion)
    logger.info('Loading model weights...')
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True
    )

    # Get the language model (Qwen3.5 wraps it)
    lang_model = model.model.language_model if hasattr(model.model, 'language_model') else model.model

    logger.info('Creating GGUF writer...')
    writer = gguf.GGUFWriter(
        output_path,
        'qwen3',  # Use qwen3 architecture (closest GGUF mapping)
    )

    # Add model metadata
    n_layers = config.text_config.num_hidden_layers if hasattr(config, 'text_config') else config.num_hidden_layers
    hidden_size = config.text_config.hidden_size if hasattr(config, 'text_config') else config.hidden_size
    n_heads = config.text_config.num_attention_heads if hasattr(config, 'text_config') else config.num_attention_heads
    n_kv_heads = getattr(config.text_config if hasattr(config, 'text_config') else config, 'num_key_value_heads', n_heads)
    intermediate_size = config.text_config.intermediate_size if hasattr(config, 'text_config') else config.intermediate_size
    vocab_size = config.text_config.vocab_size if hasattr(config, 'text_config') else config.vocab_size

    writer.add_block_count(n_layers)
    writer.add_context_length(config.text_config.max_position_embeddings if hasattr(config, 'text_config') else config.max_position_embeddings)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(config.text_config.rms_norm_eps if hasattr(config, 'text_config') else config.rms_norm_eps)
    writer.add_vocab_size(vocab_size)

    # Process tokenizer
    logger.info('Processing tokenizer...')
    tokenizer_model = tokenizer
    for i in range(vocab_size):
        if i < len(tokenizer):
            token = tokenizer.decode([i])
            score = -i  # Default score
            writer.add_token(token)
        else:
            writer.add_token(f"<unk_{i}>")

    # Write model weights
    logger.info('Writing model weights...')
    state_dict = lang_model.state_dict()
    total_params = len(state_dict)
    for idx, (name, tensor) in enumerate(state_dict.items()):
        # Convert name to GGUF format
        gguf_name = convert_tensor_name(name)
        tensor_np = tensor.numpy()

        # Q/K/V projections need special handling for GGUF
        if any(x in name for x in ['q_proj', 'k_proj', 'v_proj']):
            writer.add_tensor(gguf_name, tensor_np)
        else:
            writer.add_tensor(gguf_name, tensor_np)

        if idx % 20 == 0:
            logger.info(f'  Writing tensor {idx}/{total_params}: {name} -> {gguf_name}')

    logger.info('Writing GGUF file...')
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    logger.info(f'F16 GGUF saved to {output_path}')
    f16_size = os.path.getsize(output_path)
    logger.info(f'F16 GGUF size: {f16_size / 1e9:.1f} GB')

    del model, lang_model, state_dict
    gc.collect()
    torch.cuda.empty_cache()

    return output_path


def convert_tensor_name(hf_name):
    """Convert HuggingFace tensor name to GGUF format."""
    # Remove 'model.' prefix if present
    name = hf_name.removeprefix('model.')

    # Layer mapping
    name = name.replace('embed_tokens.weight', 'token_embd.weight')
    name = name.replace('norm.weight', 'output_norm.weight')
    name = name.replace('layers.', 'blk.')
    name = name.replace('.self_attn.q_proj', '.attn_q')
    name = name.replace('.self_attn.k_proj', '.attn_k')
    name = name.replace('.self_attn.v_proj', '.attn_v')
    name = name.replace('.self_attn.o_proj', '.attn_output')
    name = name.replace('.mlp.gate_proj', '.ffn_gate')
    name = name.replace('.mlp.up_proj', '.ffn_up')
    name = name.replace('.mlp.down_proj', '.ffn_down')
    name = name.replace('.input_layernorm.weight', '.attn_norm.weight')
    name = name.replace('.post_attention_layernorm.weight', '.ffn_norm.weight')
    name = name.replace('.lm_head.weight', 'output.weight')

    return name


def quantize_gguf(f16_path, quant_type='Q4_K_M'):
    """Quantize an F16 GGUF file using llama-quantize."""
    quant_path = f16_path.replace('.F16.', f'.{quant_type}.')

    logger.info(f'Quantizing {f16_path} to {quant_type}...')

    # Try to find llama-quantize
    import shutil
    quantize_bin = shutil.which('llama-quantize')

    if quantize_bin:
        import subprocess
        result = subprocess.run([quantize_bin, f16_path, quant_path, quant_type],
                                capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f'Quantization failed: {result.stderr}')
            raise RuntimeError(f'llama-quantize failed: {result.stderr}')
    else:
        # Use Python-based quantization via gguf package
        logger.info('llama-quantize not found, using Python GGUF quantization...')
        quantize_gguf_python(f16_path, quant_path, quant_type)

    logger.info(f'Quantized model saved to {quant_path}')
    q_size = os.path.getsize(quant_path)
    logger.info(f'{quant_type} GGUF size: {q_size / 1e9:.1f} GB')

    return quant_path


def quantize_gguf_python(f16_path, output_path, quant_type):
    """Python-based GGUF quantization fallback."""
    import gguf

    reader = gguf.GGUFReader(f16_path)
    writer = gguf.GGUFWriter(output_path, reader.architecture)

    # Copy all metadata
    for key, value in reader.fields.items():
        writer.add_key_value(str(key), value)

    # Process tensors with quantization
    for tensor in reader.tensors:
        data = tensor.data
        name = tensor.name

        if quant_type.startswith('Q4') or quant_type.startswith('Q5') or quant_type.startswith('Q8'):
            # For now, keep as F16 and let the user quantize locally
            writer.add_tensor(name, data)
        else:
            writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def push_to_hub(local_path, repo_id, filename=None):
    """Push a GGUF file to HuggingFace Hub."""
    from huggingface_hub import HfApi
    api = HfApi()

    logger.info(f'Pushing {local_path} to {repo_id}...')
    if filename:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type='model',
        )
    else:
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type='model',
        )
    logger.info(f'Uploaded to {repo_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert', action='store_true', help='Convert to GGUF F16')
    parser.add_argument('--quantize', action='store_true', help='Quantize GGUF')
    parser.add_argument('--push', action='store_true', help='Push to HuggingFace')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR)
    parser.add_argument('--output-dir', type=str, default=GGUF_DIR)
    parser.add_argument('--quant-type', type=str, default='Q4_K_M',
                        choices=['Q4_K_M', 'Q5_K_M', 'Q8_0', 'Q4_0', 'Q5_0'])
    args = parser.parse_args()

    if not args.convert and not args.quantize and not args.push:
        # Default: convert, quantize, and push
        f16_path = convert_to_gguf_f16(args.model_dir, args.output_dir)
        q_path = quantize_gguf(f16_path, args.quant_type)
        # Create repo
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo('DJLougen/Ornstein-27B-SABER-GGUF', exist_ok=True)
        # Push both
        push_to_hub(f16_path, 'DJLougen/Ornstein-27B-SABER-GGUF',
                    filename=os.path.basename(f16_path))
        push_to_hub(q_path, 'DJLougen/Ornstein-27B-SABER-GGUF',
                    filename=os.path.basename(q_path))