#!/usr/bin/env python3
# Measure EAGLE-3 speedup: baseline serve vs spec serve on the target model.
import argparse
import json
import re
import statistics
import time
import urllib.request


def post(url, payload, timeout=600):
    r = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def scrape_al(root):
    with urllib.request.urlopen(root + "/metrics", timeout=30) as resp:
        txt = resp.read().decode()
    metrics = {}
    for match in re.finditer(r"^(vllm:spec_decode_[a-z_]+)\{[^}]*\}\s+([0-9eE.+-]+)", txt, re.M):
        metrics[match.group(1)] = metrics.get(match.group(1), 0.0) + float(match.group(2))
    drafts = metrics.get("vllm:spec_decode_num_drafts_total", 0)
    accepted = metrics.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    return (1.0 + accepted / drafts) if drafts else None


def load_prompts(path, n):
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            conversations = json.loads(line).get("conversations", [])
            users = [
                {"role": "user", "content": message.get("content") or message.get("value")}
                for message in conversations
                if (message.get("role") or message.get("from")) == "user"
            ]
            if users:
                out.append([users[0]])
            if len(out) >= n:
                break
    return out


def warmup(port, model, prompt):
    endpoint = f"http://localhost:{port}/v1"
    post(endpoint + "/chat/completions", {"model": model, "messages": prompt, "max_tokens": 64, "temperature": 0})


def run(port, model, prompts, max_tokens):
    endpoint = f"http://localhost:{port}/v1"
    started = time.time()
    total = 0
    for messages in prompts:
        response = post(
            endpoint + "/chat/completions",
            {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
        )
        total += response.get("usage", {}).get("completion_tokens", 0)
    elapsed = time.time() - started
    return {"tokens": total, "seconds": elapsed, "tokens_per_second": total / elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model", default="target", help="served-model-name on both serves")
    parser.add_argument("--baseline-port", type=int, default=8000)
    parser.add_argument("--spec-port", type=int, default=8100)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--min-served-al", type=float, default=2.35)
    parser.add_argument("--draft-path", default="")
    parser.add_argument("--json-out", default="report.json")
    parser.add_argument("--mode", choices=("combined", "baseline", "speculative"), default="combined")
    parser.add_argument("--baseline-json")
    args = parser.parse_args()

    prompts = load_prompts(args.eval, args.n)
    if not prompts:
        raise SystemExit(f"no benchmark prompts found in {args.eval}")
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")

    print(
        f"Benchmarking {len(prompts)} prompts, {args.rounds} rounds, "
        f"concurrency=1, max_tokens={args.max_tokens}, temp=0"
    )
    if args.mode == "baseline":
        warmup(args.baseline_port, args.model, prompts[0])
        baseline_rounds = [run(args.baseline_port, args.model, prompts, args.max_tokens) for _ in range(args.rounds)]
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "prompt_count": len(prompts),
                    "round_count": args.rounds,
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "concurrency": 1,
                    "baseline_rounds": baseline_rounds,
                },
                f,
                indent=2,
            )
            f.write("\n")
        print(f"baseline median: {statistics.median(row['tokens_per_second'] for row in baseline_rounds):.1f} tok/s")
        return

    results = []
    if args.mode == "speculative":
        if not args.baseline_json:
            raise SystemExit("--baseline-json is required with --mode speculative")
        with open(args.baseline_json, encoding="utf-8") as f:
            baseline_report = json.load(f)
        baseline_rounds = baseline_report.get("baseline_rounds", [])
        if len(baseline_rounds) != args.rounds:
            raise SystemExit("baseline report round count does not match --rounds")
        warmup(args.spec_port, args.model, prompts[0])
        for index, baseline in enumerate(baseline_rounds):
            speculative = run(args.spec_port, args.model, prompts, args.max_tokens)
            speedup = speculative["tokens_per_second"] / baseline["tokens_per_second"]
            results.append({"round": index + 1, "baseline": baseline, "speculative": speculative, "speedup": speedup})
    else:
        warmup(args.baseline_port, args.model, prompts[0])
        warmup(args.spec_port, args.model, prompts[0])
        for index in range(args.rounds):
            if index % 2 == 0:
                baseline = run(args.baseline_port, args.model, prompts, args.max_tokens)
                speculative = run(args.spec_port, args.model, prompts, args.max_tokens)
            else:
                speculative = run(args.spec_port, args.model, prompts, args.max_tokens)
                baseline = run(args.baseline_port, args.model, prompts, args.max_tokens)
            speedup = speculative["tokens_per_second"] / baseline["tokens_per_second"]
            results.append({"round": index + 1, "baseline": baseline, "speculative": speculative, "speedup": speedup})

    for row in results:
        print(
            f"round {row['round']}: baseline={row['baseline']['tokens_per_second']:.1f} tok/s "
            f"eagle3={row['speculative']['tokens_per_second']:.1f} tok/s speedup={row['speedup']:.3f}x"
        )

    served_al = scrape_al(f"http://localhost:{args.spec_port}")
    speedup = statistics.median(row["speedup"] for row in results)
    baseline_tps = statistics.median(row["baseline"]["tokens_per_second"] for row in results)
    speculative_tps = statistics.median(row["speculative"]["tokens_per_second"] for row in results)
    passed = served_al is not None and speedup >= args.min_speedup and served_al >= args.min_served_al
    report = {
        "status": "passed" if passed else "failed",
        "model": args.model,
        "draft_path": args.draft_path,
        "prompt_count": len(prompts),
        "round_count": args.rounds,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "concurrency": 1,
        "median_baseline_tokens_per_second": baseline_tps,
        "median_speculative_tokens_per_second": speculative_tps,
        "median_speedup": speedup,
        "served_al": served_al,
        "min_speedup": args.min_speedup,
        "min_served_al": args.min_served_al,
        "rounds": results,
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"baseline median: {baseline_tps:.1f} tok/s")
    print(
        f"eagle3 median  : {speculative_tps:.1f} tok/s"
        + (f"  (served AL={served_al:.3f})" if served_al is not None else "")
    )
    print(f"MEDIAN SPEEDUP : {speedup:.3f}x")
    print(
        f"GATE            : {'PASS' if passed else 'FAIL'} "
        f"(requires speedup >= {args.min_speedup:.2f}x and served AL >= {args.min_served_al:.2f})"
    )
    if not passed:
        hints = []
        if served_al is None:
            hints.append("served AL unavailable from /metrics")
        elif served_al < args.min_served_al:
            hints.append(f"served AL {served_al:.3f} < {args.min_served_al:.2f}")
        if speedup < args.min_speedup:
            hints.append(f"median speedup {speedup:.3f}x < {args.min_speedup:.2f}x")
        if hints:
            print("GATE HINT       : " + "; ".join(hints))
            print(
                "GATE HINT       : the runner already exports TorchSpec's best marked checkpoint; "
                "use more on-policy data or training epochs if higher quality is required"
            )
        print("GATE ACTION     : report recorded with status=failed; continuing without a process error")


if __name__ == "__main__":
    main()
