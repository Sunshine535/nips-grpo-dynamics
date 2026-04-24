#!/usr/bin/env python3
"""Analyze TRACE A/B/C results per GPT-5.5 decision tree.

Produces:
- Mean/std/CI for each variant
- Pairwise deltas (C-A, C-B, B-A)
- Verdict per GPT-5.5 protocol:
  - C > A and C > B: trust gate adds value
  - C ≈ A: old fragment suffices
  - C ≈ B: infrastructure alone suffices (no trust gate signal)
  - C < A: mechanism hurts
- Mechanism activation summary (lambda_eff trajectory)
"""
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev


def load_evals(base: str):
    """Returns {variant: {seed: accuracy}}."""
    results = {}
    for variant_dir, variant_tag in [
        ("A_legacy", "A_legacy_spo_replay"),
        ("B_constant", "B_trace_constant_gate"),
        ("C_full", "C_trace_full"),
    ]:
        results[variant_dir] = {}
        evals_dir = Path(base) / variant_dir / "evals"
        if not evals_dir.exists():
            continue
        for f in evals_dir.glob(f"eval_{variant_tag}_seed*.json"):
            seed = int(f.stem.split("seed")[-1])
            d = json.load(open(f))
            results[variant_dir][seed] = d["accuracy"]
    return results


def load_mechanism_logs(base: str):
    """Returns per-variant list of step-level stats."""
    logs = {}
    for variant_dir, subdir_pat in [
        ("A_legacy", "A_legacy_spo_replay_seed*/aser_step_stats.json"),
        ("B_constant", "B_trace_constant_gate_seed*/trace_step_stats.json"),
        ("C_full", "C_trace_full_seed*/trace_step_stats.json"),
    ]:
        logs[variant_dir] = []
        for path in (Path(base) / variant_dir).glob(subdir_pat):
            try:
                d = json.load(open(path))
                logs[variant_dir].append((path.parent.name, d))
            except Exception:
                pass
    return logs


def summarize(accs: dict) -> dict:
    if len(accs) == 0:
        return {"n": 0}
    vals = list(accs.values())
    return {
        "n": len(vals),
        "seeds": sorted(accs.keys()),
        "mean": round(mean(vals), 4),
        "std": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "per_seed": {s: round(accs[s], 4) for s in sorted(accs.keys())},
    }


def verdict(a_mean, b_mean, c_mean):
    if c_mean is None or a_mean is None or b_mean is None:
        return "INCOMPLETE"
    d_ca = c_mean - a_mean
    d_cb = c_mean - b_mean
    d_ba = b_mean - a_mean
    msg = (f"C-A={d_ca:+.4f}, C-B={d_cb:+.4f}, B-A={d_ba:+.4f}\n")
    thresh = 0.02
    if d_ca > thresh and d_cb > thresh:
        msg += "VERDICT: C > A and C > B — trust gate adds value. Proceed."
    elif abs(d_ca) < thresh / 2:
        msg += "VERDICT: C ≈ A — new method may reduce to old fragment."
    elif abs(d_cb) < thresh / 2:
        msg += "VERDICT: C ≈ B — infrastructure alone matters, trust gate inactive."
    elif d_ca < -thresh:
        msg += "VERDICT: C < A — new method hurts. Check implementation or diagnosis."
    else:
        msg += "VERDICT: Mixed — inconclusive, need more seeds."
    return msg


def mechanism_summary(variant_logs):
    """Extract lambda_eff and frontier trajectory summary."""
    out = {}
    for run_name, stats in variant_logs:
        lambda_effs = [s.get("lambda_eff", 0.0) for s in stats]
        frontiers = [s.get("mean_frontier", 0.0) for s in stats]
        bank_sizes = [s.get("bank_size", 0) for s in stats]
        replay_ratios = [s.get("replay_token_ratio", 0.0) for s in stats]
        if lambda_effs:
            out[run_name] = {
                "n_steps": len(lambda_effs),
                "lambda_eff_mean": round(mean(lambda_effs), 6),
                "lambda_eff_max": round(max(lambda_effs), 6),
                "lambda_eff_nonzero_steps": sum(1 for x in lambda_effs if x > 1e-8),
                "mean_frontier_final": round(frontiers[-1], 4) if frontiers else 0,
                "bank_size_final": bank_sizes[-1] if bank_sizes else 0,
                "replay_token_ratio_final": round(replay_ratios[-1], 4) if replay_ratios else 0,
            }
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="results/trace_abc")
    p.add_argument("--out", default="results/trace_abc/analysis_abc.json")
    args = p.parse_args()

    evals = load_evals(args.base)
    logs = load_mechanism_logs(args.base)

    report = {
        "base": args.base,
        "variants": {v: summarize(accs) for v, accs in evals.items()},
        "mechanism": {v: mechanism_summary(l) for v, l in logs.items()},
    }

    def _mean(v):
        s = report["variants"].get(v, {})
        return s.get("mean") if s.get("n") else None

    a, b, c = _mean("A_legacy"), _mean("B_constant"), _mean("C_full")
    report["verdict"] = verdict(a, b, c)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\n" + "=" * 60)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
