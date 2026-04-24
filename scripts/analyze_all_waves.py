#!/usr/bin/env python3
"""Aggregate all wave results into a unified table for the paper."""
import json
import os
import glob
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent / "results"


def load_eval(path):
    with open(path) as f:
        d = json.load(f)
    return {"n": d["n"], "correct": d["correct"], "accuracy": d["accuracy"]}


def scan_wave14_500step():
    rows = []
    for f in sorted(glob.glob(str(BASE / "wave14_500step/evals/eval_seed*.json"))):
        seed = int(f.split("seed")[1].split(".")[0])
        r = load_eval(f)
        rows.append({"method": "SPO+Replay 500step", "seed": seed, **r})
    return rows


def scan_wave14_phase():
    rows = []
    for f in sorted(glob.glob(str(BASE / "wave14_phase_diagram/evals/eval_a*_b*.json"))):
        name = os.path.basename(f).replace("eval_", "").replace(".json", "")
        parts = name.split("_")
        alpha = float(parts[0][1:])
        beta = float(parts[1][1:])
        r = load_eval(f)
        rows.append({"method": f"Phase(a={alpha},b={beta})", "alpha": alpha, "beta": beta, **r})
    return rows


def scan_wave15_halluzero():
    rows = []
    for f in sorted(glob.glob(str(BASE / "wave15_halluzero/evals/eval_hz_*.json"))):
        name = os.path.basename(f).replace("eval_hz_", "").replace(".json", "")
        parts = name.rsplit("_seed", 1)
        strategy = parts[0]
        seed = int(parts[1])
        r = load_eval(f)
        rows.append({"method": f"HalluZero-{strategy}", "seed": seed, **r})
    return rows


def scan_previous_waves():
    """Scan eval JSONs from previous waves (10-13)."""
    rows = []
    patterns = [
        ("wave10_aser", "SPO+Replay"),
        ("wave11_rho070_boost", "FixedRho070"),
        ("wave12_controls", "Controls"),
        ("wave13_controls", "Controls"),
    ]
    for dirname, label in patterns:
        d = BASE / dirname
        if not d.exists():
            continue
        for f in sorted(glob.glob(str(d / "**/*.json"), recursive=True)):
            if "eval" not in os.path.basename(f).lower():
                continue
            try:
                r = load_eval(f)
                rows.append({"method": label, "source": os.path.basename(f), **r})
            except (json.JSONDecodeError, KeyError):
                pass
    return rows


def main():
    all_rows = []
    for scanner, name in [
        (scan_wave14_500step, "Wave14-500step"),
        (scan_wave14_phase, "Wave14-Phase"),
        (scan_wave15_halluzero, "Wave15-HalluZero"),
    ]:
        rows = scanner()
        if rows:
            print(f"\n{'='*60}")
            print(f" {name} ({len(rows)} results)")
            print(f"{'='*60}")
            for r in rows:
                print(f"  {r['method']:30s}  acc={r['accuracy']:.4f}  ({r['correct']}/{r['n']})")
            all_rows.extend(rows)

    if not all_rows:
        print("\n[warn] No results found yet.")
        return

    # Group by method and compute stats
    print(f"\n{'='*60}")
    print(" AGGREGATE BY METHOD")
    print(f"{'='*60}")
    groups = defaultdict(list)
    for r in all_rows:
        groups[r["method"]].append(r["accuracy"])
    for method, accs in sorted(groups.items(), key=lambda x: -max(x[1])):
        import numpy as np
        a = np.array(accs)
        if len(a) > 1:
            print(f"  {method:30s}  n={len(a):2d}  mean={a.mean():.4f} +/- {a.std():.4f}  range=[{a.min():.4f}, {a.max():.4f}]")
        else:
            print(f"  {method:30s}  n={len(a):2d}  acc={a[0]:.4f}")

    # Phase diagram table (if available)
    phase_rows = [r for r in all_rows if "Phase" in r.get("method", "")]
    if phase_rows:
        print(f"\n{'='*60}")
        print(" PHASE DIAGRAM (alpha x beta)")
        print(f"{'='*60}")
        alphas = sorted(set(r["alpha"] for r in phase_rows))
        betas = sorted(set(r["beta"] for r in phase_rows))
        header = f"{'alpha/beta':>12s}" + "".join(f"{b:>8.1f}" for b in betas)
        print(header)
        for a in alphas:
            vals = []
            for b in betas:
                match = [r for r in phase_rows if r["alpha"] == a and r["beta"] == b]
                if match:
                    vals.append(f"{match[0]['accuracy']:.3f}")
                else:
                    vals.append("  ---  ")
            print(f"{a:>12.1f}" + "".join(f"{v:>8s}" for v in vals))


if __name__ == "__main__":
    main()
