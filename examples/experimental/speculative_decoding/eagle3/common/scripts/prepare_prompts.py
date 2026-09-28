#!/usr/bin/env python3
# Build a user-prompt pool from a public instruction dataset.
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=150000)
    parser.add_argument("--out", default="data/prompts.jsonl")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    count = 0
    with open(args.out, "w") as f:
        for example in dataset:
            conversations = example.get("messages") or example.get("conversations")
            user = None
            if conversations:
                for message in conversations:
                    role = message.get("role") or message.get("from")
                    if role in ("user", "human"):
                        user = message.get("content") or message.get("value")
                        break
            elif example.get("prompt"):
                user = example["prompt"]
            if not user:
                continue
            f.write(json.dumps({"conversations": [{"role": "user", "content": user}]}, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.n:
                break
    print(f"wrote {count} prompts -> {args.out}")


if __name__ == "__main__":
    main()
    # PyArrow may leave a helper thread alive after streaming. Output is closed,
    # so bypass interpreter teardown after a successful data-only invocation.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
