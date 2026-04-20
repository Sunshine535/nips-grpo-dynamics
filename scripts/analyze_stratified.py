#!/usr/bin/env python3
"""Stratified accuracy analysis — split GSM8K test by base-model difficulty."""
import json
import os
import glob
import numpy as np

STRAT_DIR = "results/stratified_eval"


def main():
    # Load base per-question correctness for difficulty stratification
    base = json.load(open(os.path.join(STRAT_DIR, "base.json")))
    n = base["n"]
    easy_mask = np.array([q["correct"] for q in base["per_q"]], dtype=bool)
    hard_mask = ~easy_mask
    n_easy, n_hard = int(easy_mask.sum()), int(hard_mask.sum())
    print(f"base n={n}: easy(base correct)={n_easy}  hard(base wrong)={n_hard}\n")

    groups = {}
    for fp in sorted(glob.glob(os.path.join(STRAT_DIR, "*.json"))):
        run = os.path.basename(fp).replace(".json", "")
        if run == "base":
            continue
        data = json.load(open(fp))
        if "per_q" not in data:
            continue
        corr = np.array([q["correct"] for q in data["per_q"]], dtype=bool)
        overall = float(corr.mean())
        easy_acc = float(corr[easy_mask].mean()) if n_easy else float("nan")
        hard_acc = float(corr[hard_mask].mean()) if n_hard else float("nan")
        groups[run] = {
            "overall": overall,
            "easy": easy_acc,
            "hard": hard_acc,
            "n_q": int(corr.sum()),
        }

    # Print per-run
    print(f"{'run':35s} {'overall':>8s} {'easy(base✓)':>12s} {'hard(base✗)':>12s}")
    for run, r in sorted(groups.items()):
        print(f"{run:35s} {r['overall']:>8.3f} {r['easy']:>12.3f} {r['hard']:>12.3f}")

    # Aggregate by method (bucket into fixed ρ, ADQ, bandit)
    def bucket(run):
        if "bandit" in run:
            return "bandit"
        if "_adq" in run:
            return "ADQ (init rho=1.0)"
        for rho in ("0.70", "1.00", "3.00"):
            if run.startswith(f"rho{rho}_"):
                return f"fixed rho={rho}"
        return "other"

    agg = {}
    for run, r in groups.items():
        b = bucket(run)
        agg.setdefault(b, []).append(r)

    print("\n=== grouped stratified means ===")
    print(f"{'group':25s} {'n':>3s} {'overall':>12s} {'easy':>14s} {'hard':>14s}")
    for g in sorted(agg.keys()):
        rows = agg[g]
        ov = np.array([r["overall"] for r in rows])
        ez = np.array([r["easy"] for r in rows])
        hd = np.array([r["hard"] for r in rows])
        print(f"{g:25s} {len(rows):>3d}  "
              f"{ov.mean():>4.3f}±{ov.std(ddof=1) if len(ov)>1 else 0:>4.3f}  "
              f"{ez.mean():>4.3f}±{ez.std(ddof=1) if len(ez)>1 else 0:>5.3f}  "
              f"{hd.mean():>4.3f}±{hd.std(ddof=1) if len(hd)>1 else 0:>5.3f}")

    # Key test: does best-ρ depend on difficulty?
    print("\n=== ρ ordering by difficulty ===")
    print("(If the best ρ on easy differs from the best on hard, that supports CSD theory's")
    print(" prediction that ρ* varies with task success rate p.)")
    for subset in ["overall", "easy", "hard"]:
        best = max(agg.items(), key=lambda kv: np.mean([r[subset] for r in kv[1]]))
        print(f"  best {subset:8s}: {best[0]:30s} mean={np.mean([r[subset] for r in best[1]]):.3f}")

    out = {"per_run": groups, "grouped": {k: [r for r in v] for k, v in agg.items()},
           "n_easy": n_easy, "n_hard": n_hard}
    with open(os.path.join(STRAT_DIR, "stratified_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {STRAT_DIR}/stratified_analysis.json")


if __name__ == "__main__":
    main()
