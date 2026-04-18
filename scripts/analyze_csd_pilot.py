#!/usr/bin/env python3
"""
CSD Pilot 结果分析脚本。

输入: results/csd_pilot/pilot2_adq_collapse/ 下的 8 个（或 4 个）run 目录
输出: 详细的 per-run 指标 + 跨 run 对比 + CSD 理论验证
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path


def load_run(run_dir):
    """Load all JSON outputs from a run directory."""
    data = {'run_dir': run_dir}
    for name in ['pilot_results', 'csd_logs', 'rho_grpo_logs', 'step_logs', 'stability_telemetry', 'ada_telemetry']:
        path = os.path.join(run_dir, f'{name}.json')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data[name] = json.load(f)
            except Exception as e:
                print(f"  [warn] failed to load {path}: {e}")
    return data


def extract_metrics(run_data):
    """Extract key metrics from a run."""
    metrics = {
        'rho': run_data.get('pilot_results', {}).get('rho'),
        'seed': run_data.get('pilot_results', {}).get('seed'),
        'use_adq': run_data.get('pilot_results', {}).get('use_adq', False),
    }

    # Reward trajectory (from step_logs TRL log_history)
    step_logs = run_data.get('step_logs', [])
    reward_history = [
        l.get('reward/mean', l.get('reward_mean'))
        for l in step_logs if 'reward/mean' in l or 'reward_mean' in l
    ]
    if reward_history:
        metrics['n_reward_entries'] = len(reward_history)
        metrics['final_reward'] = reward_history[-1] if reward_history else 0
        metrics['max_reward'] = max(reward_history) if reward_history else 0
        metrics['mean_reward_last20%'] = sum(reward_history[-len(reward_history)//5:]) / max(1, len(reward_history)//5)
    else:
        metrics['n_reward_entries'] = 0
        metrics['final_reward'] = 0
        metrics['max_reward'] = 0
        metrics['mean_reward_last20%'] = 0

    # CSD components (from our callback)
    csd = run_data.get('csd_logs', [])
    if csd:
        metrics['csd_n_steps'] = len(csd)
        metrics['total_n_pos'] = sum(l.get('n_pos', 0) for l in csd)
        metrics['total_n_neg'] = sum(l.get('n_neg', 0) for l in csd)
        metrics['total_n_degen'] = sum(l.get('n_degenerate', 0) for l in csd)
        metrics['avg_signal_strength'] = sum(l.get('csd_signal_strength', 0) for l in csd) / len(csd)
        metrics['final_is_collapsed'] = csd[-1].get('is_collapsed', False) if csd else False

    # Rho trajectory (for ADQ)
    if run_data.get('use_adq') and 'ada_telemetry' in run_data:
        rho_traj = [t.get('rho') for t in run_data['ada_telemetry'] if t.get('rho') is not None]
        if rho_traj:
            metrics['rho_range'] = [round(min(rho_traj), 4), round(max(rho_traj), 4)]
            metrics['final_rho'] = round(rho_traj[-1], 4)

    # Collapse classification
    metrics['collapsed'] = (
        metrics.get('final_is_collapsed', False) or
        metrics['mean_reward_last20%'] < 0.1
    )

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot_dir', default='results/csd_pilot/pilot2_adq_collapse')
    args = p.parse_args()

    run_dirs = [d for d in glob.glob(os.path.join(args.pilot_dir, 'rho*')) if os.path.isdir(d)]
    print(f"Found {len(run_dirs)} run directories in {args.pilot_dir}")

    const_runs = []
    adq_runs = []
    for rd in sorted(run_dirs):
        data = load_run(rd)
        m = extract_metrics(data)
        tag = 'ADQ' if m['use_adq'] else 'CONST'
        print(f"\n[{tag}] {os.path.basename(rd)}:")
        for k, v in m.items():
            if k != 'use_adq':
                print(f"  {k}: {v}")
        (adq_runs if m['use_adq'] else const_runs).append(m)

    # Summary comparison
    print("\n" + "="*60)
    print("CROSS-RUN COMPARISON")
    print("="*60)
    for label, runs in [('CONST', const_runs), ('ADQ', adq_runs)]:
        if not runs:
            continue
        collapse = sum(1 for r in runs if r.get('collapsed'))
        avg_final = sum(r.get('final_reward', 0) for r in runs) / len(runs)
        avg_max = sum(r.get('max_reward', 0) for r in runs) / len(runs)
        total_pos = sum(r.get('total_n_pos', 0) for r in runs)
        total_neg = sum(r.get('total_n_neg', 0) for r in runs)
        print(f"\n{label} (n={len(runs)}):")
        print(f"  Collapse rate:   {collapse}/{len(runs)} = {100*collapse/len(runs):.0f}%")
        print(f"  Mean final:      {avg_final:.4f}")
        print(f"  Mean max reward: {avg_max:.4f}")
        print(f"  Total n_pos:     {total_pos}")
        print(f"  Total n_neg:     {total_neg}")

    if const_runs and adq_runs:
        const_collapse_rate = sum(1 for r in const_runs if r.get('collapsed')) / len(const_runs)
        adq_collapse_rate = sum(1 for r in adq_runs if r.get('collapsed')) / len(adq_runs)
        print("\n" + "="*60)
        print("KILLER RESULT CHECK")
        print("="*60)
        print(f"Constant collapse rate: {100*const_collapse_rate:.0f}%")
        print(f"ADQ collapse rate:      {100*adq_collapse_rate:.0f}%")
        if const_collapse_rate > 0 and adq_collapse_rate == 0:
            print("KILLER: ADQ eliminates collapse that const has!")
        elif const_collapse_rate > adq_collapse_rate:
            print(f"POSITIVE: ADQ reduces collapse from {100*const_collapse_rate:.0f}% to {100*adq_collapse_rate:.0f}%")
        elif const_collapse_rate == adq_collapse_rate == 0:
            print("INCONCLUSIVE: No collapse in either — need stronger perturbation / more steps")
        else:
            print("NEGATIVE: ADQ did not improve")


if __name__ == '__main__':
    main()
