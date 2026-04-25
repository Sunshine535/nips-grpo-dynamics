#!/usr/bin/env python3
"""Analyze SAGE A/B/D/C results (GPT-5.5 Task 5)."""
import argparse, json, os, sys
from pathlib import Path
from statistics import mean, stdev


def load_evals(base: str) -> dict:
    results = {}
    for variant_dir, tag_prefix in [
        ("A_legacy", "A_legacy_spo_replay"),
        ("B_tasa_only", "B_tasa_only"),
        ("D_positive_ce_only", "D_positive_ce_only"),
        ("C_sage_full", "C_sage_full"),
    ]:
        results[variant_dir] = {}
        evals_dir = Path(base) / variant_dir / "evals"
        if not evals_dir.exists():
            continue
        for f in sorted(evals_dir.glob("eval_*.json")):
            if "manifest" in f.name:
                continue
            try:
                d = json.load(open(f))
                if "accuracy" in d:
                    seed = int(f.stem.split("seed")[-1])
                    results[variant_dir][seed] = d["accuracy"]
            except Exception:
                pass
    return results


def summarize(accs: dict) -> dict:
    if not accs:
        return {"n": 0}
    vals = list(accs.values())
    return {
        "n": len(vals), "seeds": sorted(accs.keys()),
        "mean": round(mean(vals), 4),
        "std": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
        "per_seed": {s: round(v, 4) for s, v in sorted(accs.items())},
    }


def verdict(a, b, d, c):
    if any(x is None for x in [a, b, d, c]):
        return "INCOMPLETE — not all variants have results"
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
    return "MIXED — inconclusive"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="results/sage_minimal_abc")
    p.add_argument("--out", default="reports/CORE_COMPARISON.md")
    args = p.parse_args()

    evals = load_evals(args.base)
    summaries = {v: summarize(accs) for v, accs in evals.items()}

    def _mean(v):
        s = summaries.get(v, {})
        return s.get("mean") if s.get("n") else None

    a, b, d, c = _mean("A_legacy"), _mean("B_tasa_only"), _mean("D_positive_ce_only"), _mean("C_sage_full")
    v = verdict(a, b, d, c)

    report = {"base": args.base, "variants": summaries, "verdict": v}
    print(json.dumps(report, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nVERDICT: {v}")


if __name__ == "__main__":
    main()
