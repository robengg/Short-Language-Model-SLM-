#!/usr/bin/env python3
import re, json, argparse

def simple_router(text: str):
    m = re.search(r'(?:pkg|package)\s*([0-9]+(?:\.[0-9]+)*)', text, re.I)
    if "download" in text.lower(): action = "DOWNLOAD"
    elif "link" in text.lower(): action = "LINK"
    else: action = "ANSWER"
    return {"action":action, "version": m.group(1) if m else "", "reason":"rule-based routing"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    args = ap.parse_args()
    print(json.dumps(simple_router(args.message), indent=2))

if __name__ == "__main__":
    main()
