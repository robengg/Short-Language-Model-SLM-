#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter)

    x = tok(args.prompt, return_tensors="pt")
    y = model.generate(**x, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)
    print(tok.decode(y[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
