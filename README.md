# SLM CPU Build

This is a **small language model project** that runs on CPU, with LoRA adapters for training and instruction tuning.

## Features
- CPU-only (no GPU required)
- Works offline (after downloading base models once)
- LoRA fine-tuning for plain text (`train_lora_causal.py`)
- LoRA instruction-tuning for JSONL tasks (`train_sft_lora.py`)
- Generate with adapters (`generate.py`)
- Merge adapter → standalone model (`merge_lora.py`)
- Simple rule-based router (`router.py`)

## Quickstart

```bash
# create venv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools
pip install transformers datasets peft torch sentencepiece

# train LoRA (English)
python scripts/train_lora_causal.py --model_path ~/models/distilgpt2   --train_file data/en.txt --output_dir adapters/en_lora

# generate
python scripts/generate.py --base ~/models/distilgpt2   --adapter adapters/en_lora --prompt "Write a haiku about teamwork"
```

## Dataset formats
- Plain text (`data/en.txt`, `data/ja.txt`)
- Instruction JSONL (`data/instruct.jsonl`)

## Notes
- Keep all secrets/configs **outside** training data
- This repo is generic; you can adapt to your own domain
