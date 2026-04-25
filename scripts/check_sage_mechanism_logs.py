#!/usr/bin/env python3
"""Check SAGE/COVER mechanism activation from sage_step_stats.json."""
import argparse, json, os, sys


def check(run_dir: str, warmup: int = 50, allow_short_smoke: bool = False) -> dict:
    stats_path = os.path.join(run_dir, "sage_step_stats.json")
    if not os.path.exists(stats_path):
        return {"ok": False, "error": f"Missing {stats_path}"}
    with open(stats_path) as f:
        stats = json.load(f)
    if not stats:
        return {"ok": False, "error": "Empty stats"}

    post_warmup = [s for s in stats if s.get("step", 0) >= warmup]
    if not post_warmup:
        if allow_short_smoke:
            return {"ok": False, "status": "SMOKE_TOO_SHORT_FOR_MECHANISM",
                    "total_steps": len(stats), "warmup": warmup}
        return {"ok": False, "error": f"No steps >= warmup={warmup} (total={len(stats)})"}

    bank_pos = max(s.get("bank_pos_total", 0) for s in stats)
    bank_neg = max(s.get("bank_neg_total", 0) for s in stats)
    n_pairs = [s.get("n_pairs_sampled", 0) for s in post_warmup]
    loss_pair = [s.get("loss_pair", 0.0) for s in post_warmup]
    active = any(s.get("mechanism_active", False) for s in post_warmup)

    result = {
        "ok": True, "bank_pos_total": bank_pos, "bank_neg_total": bank_neg,
        "n_pairs_mean": sum(n_pairs) / max(len(n_pairs), 1),
        "loss_pair_mean": sum(loss_pair) / max(len(loss_pair), 1),
        "mechanism_active_any": active,
        "total_steps": len(stats), "post_warmup_steps": len(post_warmup),
        "warmup": warmup,
    }

    failures = []
    if bank_pos == 0: failures.append("bank_pos_total == 0")
    if bank_neg == 0: failures.append("bank_neg_total == 0")
    if result["n_pairs_mean"] == 0: failures.append("n_pairs_sampled mean == 0")
    if result["loss_pair_mean"] == 0: failures.append("loss_pair mean == 0")
    if not active: failures.append("mechanism_active never true")

    if failures:
        result["ok"] = False
        result["failures"] = failures
    return result


def main():
    p = argparse.ArgumentParser(description="Check SAGE/COVER mechanism activation")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--allow-short-smoke", action="store_true")
    p.add_argument("--report-dir", default="reports")
    args = p.parse_args()

    result = check(args.run_dir, args.warmup, args.allow_short_smoke)
    print(json.dumps(result, indent=2))

    os.makedirs(args.report_dir, exist_ok=True)
    if result["ok"]:
        with open(os.path.join(args.report_dir, "MECHANISM_LOG_SUMMARY.md"), "w") as f:
            f.write(f"# Mechanism Log Summary\n\nRun: {args.run_dir}\nStatus: ACTIVE\n")
            for k, v in result.items():
                f.write(f"- {k}: {v}\n")
        print("MECHANISM ACTIVE")
    else:
        with open(os.path.join(args.report_dir, "MECHANISM_INACTIVE.md"), "w") as f:
            f.write(f"# Mechanism Inactive\n\nRun: {args.run_dir}\n")
            for fail in result.get("failures", [result.get("error", result.get("status", "unknown"))]):
                f.write(f"- {fail}\n")
        print("MECHANISM INACTIVE")
        sys.exit(1)


if __name__ == "__main__":
    main()
