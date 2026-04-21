#!/usr/bin/env python3
"""Round 2 consolidated analysis — Wave 10 + Wave 11 + Wave 12 combined."""
import json
import numpy as np
from pathlib import Path
from statistics import mean, stdev


def acc(p):
    try:
        return json.load(open(p))["accuracy"]
    except Exception:
        return None


def welch(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    mA, mB = a.mean(), b.mean()
    vA, vB = a.var(ddof=1), b.var(ddof=1)
    nA, nB = len(a), len(b)
    t = (mA - mB) / np.sqrt(vA / nA + vB / nB + 1e-12)
    df = (vA / nA + vB / nB) ** 2 / (
        (vA / nA) ** 2 / (nA - 1) + (vB / nB) ** 2 / (nB - 1) + 1e-12
    )
    return float(t), float(df)


PROJ = Path("/home/tarkoy/nips/nips-grpo-dynamics")

sources = {
    "spo_replay (Wave 10, n=9)": [
        PROJ / f"results/stratified_eval_aser/spo_full_seed{s}.json"
        for s in [42, 43, 44, 46, 47, 48, 49, 50, 51]
    ],
    "spo_only (Wave 10, n=3)": [
        PROJ / f"results/stratified_eval_aser/spo_only_seed{s}.json"
        for s in [42, 43, 44]
    ],
    # Seed 45 appears in both w11 and w12 (two independent reruns on different boxes).
    # Keep only w12 to avoid double-counting the same seed. The two reruns are
    # reported separately as a sensitivity check below.
    "rft_only": (
        [PROJ / f"results/stratified_eval_wave11/rft_seed{s}.json" for s in [42, 43, 44]]
        + [PROJ / f"results/stratified_eval_wave11_12/w12_rft_seed{s}.json" for s in [45, 46, 47]]
    ),
    "fixed_rho_0.70 (matched)": (
        [PROJ / f"results/stratified_eval/rho0.70_seed{s}.json" for s in [42, 43, 44]]
        + [PROJ / f"results/stratified_eval_wave11/rho0.70_seed{s}.json" for s in [46, 47, 48, 49, 50, 51]]
    ),
    "fixed_sampler_asr": (
        [PROJ / f"results/stratified_eval_wave11_12/w11_fixed_sampler_s{s}.json" for s in [42, 43, 44]]
        + [PROJ / f"results/stratified_eval_wave11_12/w12_fixed_sampler_s{s}.json" for s in [45, 46, 47]]
    ),
    "lambda_rep_0.02": [
        PROJ / f"results/stratified_eval_wave11_12/w12_lambda002_s{s}.json"
        for s in [42, 43]
    ],
}

print("## Round-2 consolidated accuracy (GSM8K test n=200)\n")
print(f"| Arm                                     | n seeds | mean ± std     | per-seed (%) |")
print(f"| --------------------------------------- | ------- | -------------- | ------------- |")
per_arm = {}
per_arm_seeds = {}
for name, paths in sources.items():
    accs = []
    seeds = []
    for p in paths:
        a = acc(p)
        if a is not None:
            accs.append(a)
            seeds.append(p.stem)
    per_arm[name] = accs
    per_arm_seeds[name] = seeds
    n = len(accs)
    if n >= 2:
        m = mean(accs) * 100
        s = stdev(accs) * 100
        vals = " ".join(f"{a*100:.1f}" for a in accs)
        print(f"| {name:<39s} | {n:^7} | {m:5.1f} ± {s:4.1f}%   | {vals} |")
    elif n == 1:
        print(f"| {name:<39s} | {n:^7} | {accs[0]*100:5.1f}          | {accs[0]*100:.1f} |")
    else:
        print(f"| {name:<39s} | {n:^7} | NO DATA        | |")


def run_test(a_name, b_name):
    a = per_arm.get(a_name, [])
    b = per_arm.get(b_name, [])
    if not a or not b:
        return
    t, df = welch(a, b)
    ma, mb = mean(a) * 100, mean(b) * 100
    print(f"\n**{a_name}** vs **{b_name}**")
    print(f"  Δ = {ma - mb:+.1f}pp ({ma:.1f}% vs {mb:.1f}%)  |  Welch's t = {t:.2f}, df = {df:.1f}")


print("\n\n## Critical tests")
run_test("spo_replay (Wave 10, n=9)", "fixed_rho_0.70 (matched)")
run_test("spo_replay (Wave 10, n=9)", "rft_only")
run_test("spo_replay (Wave 10, n=9)", "spo_only (Wave 10, n=3)")
run_test("spo_replay (Wave 10, n=9)", "fixed_sampler_asr")
run_test("spo_replay (Wave 10, n=9)", "lambda_rep_0.02")
run_test("fixed_sampler_asr", "rft_only")
run_test("fixed_sampler_asr", "fixed_rho_0.70 (matched)")

# Paired same-seed test where possible (Wave 10 spo_full vs Wave 11 fixed-ρ=0.70)
print("\n\n## Paired same-seed tests\n")
def paired(a_name, b_name):
    a_paths = sources[a_name]
    b_paths = sources[b_name]
    a_map = {p.stem.split("_seed")[-1].replace("seed", ""): acc(p) for p in a_paths}
    b_map = {}
    for p in b_paths:
        key = None
        for sfx in ["rho0.70_seed", "spo_full_seed", "spo_only_seed", "w11_rft_seed", "w12_rft_seed", "w11_fixed_sampler_s", "w12_fixed_sampler_s", "w12_lambda002_s", "rft_seed"]:
            if sfx in p.stem:
                key = p.stem.split(sfx)[-1]
                break
        if key:
            b_map[key] = acc(p)
    shared = sorted(set(a_map.keys()) & set(b_map.keys()), key=lambda s: int(s) if s.isdigit() else 999)
    if not shared:
        print(f"  {a_name} vs {b_name}: no shared seeds")
        return
    deltas = [a_map[s] - b_map[s] for s in shared if a_map[s] and b_map[s]]
    if len(deltas) < 2:
        return
    md = mean(deltas) * 100
    sd = stdev(deltas) * 100
    t = md / (sd / np.sqrt(len(deltas)))
    print(f"**{a_name}** vs **{b_name}** (paired, n={len(deltas)} shared seeds):")
    print(f"  shared: {shared}")
    print(f"  per-seed Δ (%): {[f'{d*100:+.1f}' for d in deltas]}")
    print(f"  mean Δ = {md:+.1f}pp ± {sd:.1f}, paired t = {t:.2f}, df = {len(deltas)-1}")

paired("spo_replay (Wave 10, n=9)", "fixed_rho_0.70 (matched)")
paired("spo_replay (Wave 10, n=9)", "rft_only")
paired("spo_replay (Wave 10, n=9)", "fixed_sampler_asr")

# Save JSON summary
summary = {
    "timestamp": "2026-04-21T07:35 UTC",
    "per_arm": {name: {"n": len(v), "accs": v, "seeds": per_arm_seeds[name]}
                for name, v in per_arm.items()},
}
import json as _json
with open(PROJ / "results/round2_analysis.json", "w") as f:
    _json.dump(summary, f, indent=2)
print(f"\n[saved] {PROJ}/results/round2_analysis.json")
