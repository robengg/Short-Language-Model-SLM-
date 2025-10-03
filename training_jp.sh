#!/bin/bash
python scripts/train_lora_causal.py \
  --model_path "/c/Users//Desktop/____projects/AI Model/slm-cpu/models/rinna-jgpt2-small" \
  --train_file "/c/Users//Desktop/____projects/AI Model/slm-cpu/data/ja.txt" \
  --output_dir "/c/Users//Desktop/____projects/AI Model/slm-cpu/outputs/jp_model" \
  --epochs 3 \
  --block_size 128 \
  --batch_size 8 \
  --grad_accum 4 \
  --lr 5e-5
