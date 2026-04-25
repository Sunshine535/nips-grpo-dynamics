#!/usr/bin/env python3
"""GPT-5.5 review Task 2: audit A_legacy seed42 anomaly (83.55% on full GSM8K).

Investigates whether the result is a real reproducible signal or an artifact
(stale checkpoint, leakage, config mismatch, generation noise).

Checks:
1. Adapter hash for seed42 vs seed43 (must differ).
2. Sample 20 correct + 20 incorrect predictions for seed42; print question + pred + gold.
3. Count overlap between training prompts and eval question ids.
4. Check determinism by re-counting from per_q list and verifying acc matches JSON acc.
5. Compare which subset of questions seed42 vs seed43 got right.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def adapter_dir_hash(adapter_dir: str) -> dict:
    p = Path(adapter_dir)
    out = {}
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".json", ".yaml"):
            try:
                out[str(f.relative_to(p))] = file_sha1(str(f))
            except Exception:
                pass
    return out


def audit_one_seed(eval_json_path: str, adapter_dir: str, label: str):
    print(f"\n========== {label} ==========")
    with open(eval_json_path) as f:
        d = json.load(f)
    print(f"acc = {d['accuracy']:.4f}  ({d['correct']}/{d['n']})")
    print(f"selection = {d.get('selection', 'unknown')}")
    print(f"n_eval_question_ids = {len(d.get('eval_question_ids', []))}")

    if "per_q" in d:
        per_q = d["per_q"]
        # Determinism: recompute acc from per_q
        recomp = sum(1 for q in per_q if q["correct"])
        match = recomp == d["correct"]
        print(f"recomputed correct from per_q = {recomp} {'OK' if match else 'MISMATCH'}")

        correct_q = [q for q in per_q if q["correct"]]
        wrong_q = [q for q in per_q if not q["correct"]]
        print(f"n_correct={len(correct_q)}  n_wrong={len(wrong_q)}")
        # Sample 5 correct and 5 wrong
        print(f"\n--- 5 sample CORRECT predictions ---")
        for q in correct_q[:5]:
            print(f"  i={q['i']}  gold={q['gold']!r}  pred={q['pred']!r}")
        print(f"\n--- 5 sample WRONG predictions ---")
        for q in wrong_q[:5]:
            print(f"  i={q['i']}  gold={q['gold']!r}  pred={q['pred']!r}")

    if Path(adapter_dir).is_dir():
        h = adapter_dir_hash(adapter_dir)
        print(f"\nAdapter dir: {adapter_dir}")
        print(f"  files = {len(h)}")
        for fname, fh in list(h.items())[:5]:
            print(f"  {fname}: {fh}")
    else:
        print(f"\n(adapter dir missing: {adapter_dir})")

    return d


def compare_seeds(seed42_d, seed43_d):
    print("\n========== SEED COMPARISON ==========")
    if "per_q" not in seed42_d or "per_q" not in seed43_d:
        print("per_q missing; cannot diff")
        return
    p42 = {q["i"]: q for q in seed42_d["per_q"]}
    p43 = {q["i"]: q for q in seed43_d["per_q"]}
    common_ids = sorted(set(p42) & set(p43))
    only42_correct = sum(1 for i in common_ids if p42[i]["correct"] and not p43[i]["correct"])
    only43_correct = sum(1 for i in common_ids if p43[i]["correct"] and not p42[i]["correct"])
    both_correct = sum(1 for i in common_ids if p42[i]["correct"] and p43[i]["correct"])
    both_wrong = sum(1 for i in common_ids if not p42[i]["correct"] and not p43[i]["correct"])
    print(f"common eval ids: {len(common_ids)}")
    print(f"  both correct:   {both_correct}")
    print(f"  only seed42:    {only42_correct}")
    print(f"  only seed43:    {only43_correct}")
    print(f"  both wrong:     {both_wrong}")

    overlap = set(seed42_d.get("eval_question_ids", [])) & set(seed43_d.get("eval_question_ids", []))
    p42_ids = set(seed42_d.get("eval_question_ids", []))
    p43_ids = set(seed43_d.get("eval_question_ids", []))
    if p42_ids and p43_ids:
        print(f"eval_question_ids: same set? {p42_ids == p43_ids}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed42-eval", required=True)
    p.add_argument("--seed43-eval", required=True)
    p.add_argument("--seed42-adapter", required=True)
    p.add_argument("--seed43-adapter", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    s42 = audit_one_seed(args.seed42_eval, args.seed42_adapter, "SEED 42")
    s43 = audit_one_seed(args.seed43_eval, args.seed43_adapter, "SEED 43")
    compare_seeds(s42, s43)

    summary = {
        "seed42": {"acc": s42["accuracy"], "correct": s42["correct"], "n": s42["n"]},
        "seed43": {"acc": s43["accuracy"], "correct": s43["correct"], "n": s43["n"]},
        "seed42_adapter_files": list(adapter_dir_hash(args.seed42_adapter).keys()),
        "seed43_adapter_files": list(adapter_dir_hash(args.seed43_adapter).keys()),
        "seed42_adapter_hash": adapter_dir_hash(args.seed42_adapter),
        "seed43_adapter_hash": adapter_dir_hash(args.seed43_adapter),
    }
    s42_ids = set(s42.get("eval_question_ids", []))
    s43_ids = set(s43.get("eval_question_ids", []))
    summary["eval_ids_identical"] = s42_ids == s43_ids if s42_ids and s43_ids else None

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
