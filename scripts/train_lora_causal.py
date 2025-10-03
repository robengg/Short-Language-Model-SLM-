#!/usr/bin/env python3
import os, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings("ignore")

from transformers.utils import logging
logging.set_verbosity_error()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)

    peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["c_attn","c_proj","c_fc"],
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)

    ds = load_dataset("text", data_files={"train": args.train_file})
    ds_tok = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.block_size),
                    batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_total_limit=1,
        report_to=[]
    )
    Trainer(model=model, args=targs, train_dataset=ds_tok["train"], data_collator=collator).train()
    model.save_pretrained(args.output_dir); tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
