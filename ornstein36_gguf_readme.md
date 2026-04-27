---
license: apache-2.0
base_model: DJLougen/Ornstein3.6-35B-A3B-SABER
base_model_relation: quantized
library_name: gguf
language:
- en
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- quantized
- qwen3_5_moe_text
- qwen3_5_moe
- mixture-of-experts
- qwen3.6
- saber
- refusal-ablation
- uncensored
- conversational
---

# Ornstein3.6-35B-A3B-SABER — GGUF

![Ornstein3.6 SABER](ornstein3.6SABER.jpeg)

GGUF quantizations of
[`DJLougen/Ornstein3.6-35B-A3B-SABER`](https://huggingface.co/DJLougen/Ornstein3.6-35B-A3B-SABER)
for use with `llama.cpp`, `ollama`, `LM Studio`, and compatible runtimes.

Source model is the **SABER**-ablated variant of Ornstein3.6-35B-A3B
(Qwen3.5 MoE, 35B total / ~3B active). See the source model card for a
description of SABER.

## Support This Work

This model card is a research note for an experimental open-weight model release. It intentionally avoids personal contact details and private infrastructure notes; use the repository issue tracker or linked model discussion page for public follow-up.

**[Support on Ko-fi](https://ko-fi.com/djlougen)**

---

## Quantization suite (8-bit and under)

All variants derived from the bf16 SABER safetensors via `llama.cpp`
`convert_hf_to_gguf.py` → `llama-quantize`. Non-Q8_0 K-quants are derived from
the Q8_0 file with `--allow-requantize`.

| File | Bits | Size (approx) | Notes |
|---|---|---|---|
| `…-Q8_0.gguf`    | 8.5 | ~36 GB | Highest fidelity, near-lossless |
| `…-Q6_K.gguf`    | 6.6 | ~29 GB | Very close to Q8_0 quality |
| `…-Q5_K_M.gguf`  | 5.7 | ~25 GB | Recommended for high-quality inference |
| `…-Q5_K_S.gguf`  | 5.5 | ~24 GB | |
| `…-Q4_K_M.gguf`  | 4.8 | ~22 GB | Recommended default |
| `…-Q4_K_S.gguf`  | 4.6 | ~20 GB | |
| `…-Q3_K_M.gguf`  | 3.9 | ~17 GB | Fits most 24 GB VRAM setups |
| `…-Q3_K_S.gguf`  | 3.5 | ~15 GB | |
| `…-Q2_K.gguf`    | ~3  | ~13 GB | Emergency size — expect quality loss |

Active parameters per token are ~3B regardless of file size; the table reflects
*total* weights on disk.

## Usage (llama.cpp)

```bash
./llama-cli -m Ornstein3.6-35B-A3B-SABER-Q4_K_M.gguf \
    -p "You are a helpful assistant." \
    -cnv --temp 0.7 --top-p 0.9
```

## Intended use

Research and red-teaming. The SABER-ablated model complies with requests its
parent model refused. Deploy behind your own policy/logging layer.

## License

Apache 2.0, inherited from the base model.
