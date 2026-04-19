#!/usr/bin/env python3
"""Aggregate Gate 1+2 final results (training reward + GSM8K test accuracy) across all 12 runs."""
import glob
import json
import os
import re
import numpy as np

BASE = "results/gates_1_2"


def main():
    results = {}

    # 1) From eval_run.log (lines like: "  eval NAME @ checkpoint-final (n=100) ..." then "    acc=0.470 (47/100)")
    with open(os.path.join(BASE, "eval_run.log")) as f:
        log = f.read()
    for m in re.finditer(r"eval (\S+) @ checkpoint-final.*?acc=(0\.\d+)", log, re.DOTALL):
        results[m.group(1)] = float(m.group(2))

    # 2) From parallel-eval JSON outputs
    for fp in glob.glob(os.path.join(BASE, "*", "gsm8k_eval.json")):
        run = os.path.basename(os.path.dirname(fp))
        with open(fp) as f:
            results[run] = json.load(f)["accuracy"]

    # 3) Training rewards
    rewards = {}
    for fp in glob.glob(os.path.join(BASE, "*", "pilot_results.json")):
        run = os.path.basename(os.path.dirname(fp))
        with open(fp) as f:
            data = json.load(f)
        if "final_reward_mean" in data:
            rewards[run] = data["final_reward_mean"]

    # Aggregate
    header = "{0:40s} {1:>12s} {2:>10s}".format("run", "train_reward", "test_acc")
    print(header)
    print("-" * 64)

    def sort_key(r):
        parts = r.split("_seed")
        if len(parts) < 2:
            return (999, r)
        rho_part = parts[0].replace("rho", "")
        return (float(rho_part), int(parts[1].split("_")[0]), r)

    for run in sorted(results.keys(), key=sort_key):
        tr = rewards.get(run)
        tr_s = "{:.3f}".format(tr) if tr is not None else "?"
        print("{0:40s} {1:>12s} {2:>10.3f}".format(run, tr_s, results[run]))

    print()
    print("=== Grouped mean ± std (ddof=1) ===")
    groups = {}
    for run, acc in results.items():
        if run.endswith("_adq"):
            key = "ADQ (init rho=1.0)"
        else:
            rho = run.split("_seed")[0].replace("rho", "")
            key = "fixed rho=" + rho
        groups.setdefault(key, []).append(acc)

    for k in sorted(groups.keys()):
        xs = np.array(groups[k])
        sd = float(xs.std(ddof=1)) if len(xs) >= 2 else 0.0
        print("  {0:25s}: n={1}  mean={2:.3f}  std={3:.3f}  runs={4}".format(
            k, len(xs), float(xs.mean()), sd, list(np.round(xs, 3))))

    with open(os.path.join(BASE, "final_acc_table.json"), "w") as f:
        json.dump({
            "per_run_acc": results,
            "per_run_reward": rewards,
            "grouped": {k: list(np.round(v, 4).tolist()) for k, v in groups.items()},
            "grouped_stats": {k: {"n": len(v), "mean": float(np.mean(v)),
                                   "std": float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0}
                              for k, v in groups.items()},
        }, f, indent=2)
    print()
    print("Saved to " + os.path.join(BASE, "final_acc_table.json"))


if __name__ == "__main__":
    main()
