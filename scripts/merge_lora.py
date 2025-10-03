#!/usr/bin/env python3
import os, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    os.makedirs(args.output, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)
    print("Merged model saved to", args.output)

if __name__ == "__main__":
    main()
