"""Provenance manifest for every train/eval run.

Addresses GPT-5.5 Task 2: every result must be traceable to a command,
git commit, resolved config, seed, dataset split, and package versions.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5)
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=5)
        return bool(out.decode().strip())
    except Exception:
        return None


def _package_versions() -> dict:
    mods = ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "numpy"]
    versions = {}
    for m in mods:
        try:
            mod = __import__(m)
            versions[m] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[m] = None
    return versions


def _file_hash(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _adapter_hash(adapter_dir: str) -> Optional[str]:
    if not adapter_dir or not Path(adapter_dir).is_dir():
        return None
    p = Path(adapter_dir)
    m = hashlib.sha1()
    for f in sorted(p.rglob("*.safetensors")) + sorted(p.rglob("*.bin")):
        if f.is_file():
            fh = _file_hash(str(f))
            if fh:
                m.update(f.name.encode())
                m.update(fh.encode())
    digest = m.hexdigest()
    return digest if digest != hashlib.sha1().hexdigest() else None


def write_manifest(
    output_dir: str,
    kind: str,
    config: Optional[dict] = None,
    config_path: Optional[str] = None,
    seed: Optional[int] = None,
    model: Optional[str] = None,
    dataset: Optional[dict] = None,
    adapter: Optional[str] = None,
    eval_question_ids: Optional[list] = None,
    generation_args: Optional[dict] = None,
    extra: Optional[dict] = None,
    manifest_name: str = "run_manifest.json",
) -> Path:
    """Write a run_manifest.json to output_dir with full provenance.

    kind: 'train' | 'eval' | 'smoke' | 'ablation'
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": kind,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python_version": sys.version.split()[0],
        "git_commit": _git_sha(),
        "git_dirty": _git_dirty(),
        "packages": _package_versions(),
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        },
        "seed": seed,
        "model": model,
        "dataset": dataset,
        "config": config,
        "config_path": config_path,
        "config_hash": _file_hash(config_path) if config_path else None,
        "adapter": adapter,
        "adapter_hash": _adapter_hash(adapter) if adapter else None,
        "generation_args": generation_args,
        "eval_question_ids_count": len(eval_question_ids) if eval_question_ids else None,
        "extra": extra or {},
    }
    manifest_path = out / manifest_name
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path


def check_manifest(manifest_path: str) -> dict:
    with open(manifest_path) as f:
        m = json.load(f)
    required = ["kind", "timestamp_utc", "command", "git_commit", "packages", "seed"]
    missing = [k for k in required if m.get(k) is None]
    return {
        "manifest_path": manifest_path,
        "kind": m.get("kind"),
        "seed": m.get("seed"),
        "git_commit": m.get("git_commit"),
        "git_dirty": m.get("git_dirty"),
        "missing_fields": missing,
        "ok": not missing,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    chk = sub.add_parser("check")
    chk.add_argument("--manifest", required=True)
    args = p.parse_args()
    if args.cmd == "check":
        result = check_manifest(args.manifest)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)
