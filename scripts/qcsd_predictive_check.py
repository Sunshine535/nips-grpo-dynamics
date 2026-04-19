#!/usr/bin/env python3
"""Quick check: does batch-mean Q_CSD predict final accuracy across the 12 runs?"""
import glob
import json
import os
import re

import numpy as np

BASE = "results/gates_1_2"

def main():
    # Load per-run test accuracy
    acc = {}
    with open(os.path.join(BASE, "eval_run.log")) as f:
        log = f.read()
    for m in re.finditer(r"eval (\S+) @ checkpoint-final.*?acc=(0\.\d+)", log, re.DOTALL):
        acc[m.group(1)] = float(m.group(2))
    for fp in glob.glob(os.path.join(BASE, "*", "gsm8k_eval.json")):
        run = os.path.basename(os.path.dirname(fp))
        with open(fp) as f:
            acc[run] = json.load(f)["accuracy"]

    # Load per-step Q_CSD and compute time-averaged Q_CSD + std
    rows = []
    for run, a in acc.items():
        qcsd_path = os.path.join(BASE, run, "rho_grpo_logs.json")
        if not os.path.isfile(qcsd_path):
            print("[skip] " + run + " (no rho_grpo_logs.json)")
            continue
        with open(qcsd_path) as f:
            logs = json.load(f)
        q_vals = [row.get("q_csd", 0.0) for row in logs if "q_csd" in row]
        p_vals = []
        for row in logs:
            n_pos = row.get("n_positive", 0)
            n_neg = row.get("n_negative", 0)
            if n_pos + n_neg > 0:
                p_vals.append(n_pos / (n_pos + n_neg))
        pilot = json.load(open(os.path.join(BASE, run, "pilot_results.json")))
        rows.append({
            "run": run,
            "acc": a,
            "train_reward": pilot.get("final_reward_mean", np.nan),
            "q_csd_mean": float(np.mean(q_vals)) if q_vals else 0.0,
            "q_csd_std": float(np.std(q_vals)) if q_vals else 0.0,
            "q_csd_final": q_vals[-1] if q_vals else 0.0,
            "p_mean": float(np.mean(p_vals)) if p_vals else 0.0,
            "is_adq": "adq" in run,
        })

    if len(rows) < 3:
        print("too few rows")
        return

    keys = ["q_csd_mean", "q_csd_std", "q_csd_final", "p_mean", "train_reward"]
    accs = np.array([r["acc"] for r in rows])

    print("{0:40s} {1:>8s} {2:>10s} {3:>10s} {4:>10s} {5:>10s} {6:>10s}".format(
        "run", "acc", "qcsd_mean", "qcsd_std", "qcsd_fin", "p_mean", "train_r"))
    for r in sorted(rows, key=lambda r: -r["acc"]):
        print("{0:40s} {1:>8.3f} {2:>10.4f} {3:>10.4f} {4:>10.4f} {5:>10.4f} {6:>10.3f}".format(
            r["run"], r["acc"], r["q_csd_mean"], r["q_csd_std"], r["q_csd_final"],
            r["p_mean"], r["train_reward"]))

    print()
    print("=== Pearson correlations with test_acc ===")
    for k in keys:
        xs = np.array([r[k] for r in rows])
        if np.std(xs) < 1e-8:
            print("  {0:15s}: constant (can't correlate)".format(k))
            continue
        r = float(np.corrcoef(xs, accs)[0, 1])
        print("  {0:15s}: r = {1:+.3f}".format(k, r))

    # Partial correlation of q_csd_mean | p_mean (does Q_CSD add info beyond p?)
    # Simple partialling: regress each on p_mean, correlate residuals.
    p_arr = np.array([r["p_mean"] for r in rows])
    q_arr = np.array([r["q_csd_mean"] for r in rows])
    def residual(y, x):
        A = np.column_stack([x, np.ones_like(x)])
        beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta
    acc_res = residual(accs, p_arr)
    q_res = residual(q_arr, p_arr)
    if np.std(q_res) > 1e-8 and np.std(acc_res) > 1e-8:
        part_r = float(np.corrcoef(q_res, acc_res)[0, 1])
        print("  partial corr(Q_CSD | p_mean, acc): r = {0:+.3f}".format(part_r))
    else:
        print("  partial correlation undefined (residuals constant)")

    out = {
        "rows": rows,
        "correlations": {k: float(np.corrcoef(np.array([r[k] for r in rows]), accs)[0, 1])
                         if np.std([r[k] for r in rows]) > 1e-8 else None
                         for k in keys},
    }
    with open(os.path.join(BASE, "qcsd_predictive.json"), "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved to " + os.path.join(BASE, "qcsd_predictive.json"))


if __name__ == "__main__":
    main()
