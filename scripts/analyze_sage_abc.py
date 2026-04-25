#!/usr/bin/env python3
"""Analyze COVER-GRPO (SAGE) A/B/D/C results. Strict: fails on missing data unless --allow-incomplete."""
import argparse, json, os, sys
from pathlib import Path
from statistics import mean, stdev

VARIANTS = [
    ("A_legacy", "A_legacy_spo_replay"),
    ("B_tasa_only", "B_tasa_only"),
    ("D_positive_ce_only", "D_positive_ce_only"),
    ("C_sage_full", "C_sage_full"),
]


def load_evals(base: str, allow_incomplete: bool = False) -> dict:
    results = {}
    missing = []
    for variant_dir, tag_prefix in VARIANTS:
        results[variant_dir] = {}
        evals_dir = Path(base) / variant_dir / "evals"
        if not evals_dir.exists():
            missing.append(f"{variant_dir}/evals/ directory missing")
            continue
        for f in sorted(evals_dir.glob("eval_*.json")):
            if "manifest" in f.name:
                continue
            try:
                d = json.load(open(f))
                if "accuracy" in d:
                    seed = int(f.stem.split("seed")[-1])
                    results[variant_dir][seed] = d["accuracy"]
            except json.JSONDecodeError as e:
                missing.append(f"{f}: JSON parse error: {e}")
    if missing and not allow_incomplete:
        print("MISSING DATA (use --allow-incomplete to proceed):")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    return results


def summarize(accs: dict) -> dict:
    if not accs:
        return {"n": 0, "status": "NO_DATA"}
    vals = list(accs.values())
    return {
        "n": len(vals), "seeds": sorted(accs.keys()),
        "mean": round(mean(vals), 4),
        "std": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
        "per_seed": {s: round(v, 4) for s, v in sorted(accs.items())},
    }


def verdict(a, b, d, c):
    if any(x is None for x in [a, b, d, c]):
        return "INCOMPLETE"
    thresh = 0.02
    if c > a and c > b and c > d:
        return "CONTINUE_TO_MORE_SEEDS"
    if c > b + thresh and c > d + thresh and c < a:
        return "WEAK_SIGNAL_NEEDS_MORE_SEEDS_AND_A_AUDIT"
    if abs(c - b) < thresh / 2:
        return "MECHANISM_NO_CONTRIBUTION"
    if abs(c - d) < thresh / 2:
        return "CONTRASTIVE_REPLAY_NO_CONTRIBUTION"
    if c < b or c < d:
        return "NEW_MECHANISM_HURTS"
    return "MIXED"


def main():
    p = argparse.ArgumentParser(description="Analyze COVER-GRPO A/B/D/C results")
    p.add_argument("--base", default="results/sage_minimal_abc")
    p.add_argument("--out", default="reports/CORE_COMPARISON.md")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--bootstrap", action="store_true")
    args = p.parse_args()

    if not Path(args.base).exists():
        print(f"Result directory {args.base} does not exist. No results to analyze.")
        sys.exit(0)

    evals = load_evals(args.base, args.allow_incomplete)
    summaries = {v: summarize(accs) for v, accs in evals.items()}

    def _mean(v):
        s = summaries.get(v, {})
        return s.get("mean") if s.get("n") else None

    a, b, d, c = _mean("A_legacy"), _mean("B_tasa_only"), _mean("D_positive_ce_only"), _mean("C_sage_full")
    v = verdict(a, b, d, c)

    if args.bootstrap:
        print("BOOTSTRAP_NOT_IMPLEMENTED — statistical CI not available yet")

    report = {"method": "COVER-GRPO", "base": args.base, "variants": summaries, "verdict": v}

    # Write JSON
    json_out = args.out.replace(".md", ".json")
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)

    # Write Markdown
    with open(args.out, "w") as f:
        f.write("# Core Comparison — COVER-GRPO A/B/D/C\n\n")
        f.write(f"Method: COVER-GRPO (source files: sage_*)\n\n")
        f.write("| Variant | Seeds | Mean | Std | Per-seed |\n")
        f.write("|---------|:-----:|:----:|:---:|----------|\n")
        for vname, s in summaries.items():
            if s.get("n", 0) == 0:
                f.write(f"| {vname} | 0 | N/A | N/A | NO_DATA |\n")
            else:
                ps = ", ".join(f"{k}={v}" for k, v in s["per_seed"].items())
                f.write(f"| {vname} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | {ps} |\n")
        f.write(f"\n**Verdict:** {v}\n")

    print(json.dumps(report, indent=2))
    print(f"\nVERDICT: {v}")
    print(f"Saved: {json_out} and {args.out}")


if __name__ == "__main__":
    main()
