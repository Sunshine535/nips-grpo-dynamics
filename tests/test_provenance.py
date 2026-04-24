"""Tests for provenance manifest (GPT-5.5 Task 2)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.provenance import write_manifest, check_manifest


def test_manifest_contains_required_fields():
    with tempfile.TemporaryDirectory() as tmp:
        p = write_manifest(
            tmp, kind="smoke",
            config={"lr": 2e-5}, seed=42, model="Qwen/Qwen3.5-9B",
            dataset={"name": "gsm8k"})
        with open(p) as f:
            m = json.load(f)
        for k in ("kind", "timestamp_utc", "command", "packages",
                  "seed", "model", "dataset"):
            assert k in m, f"missing {k}"
        assert m["seed"] == 42
        assert "torch" in m["packages"]


def test_check_manifest_flags_missing_fields():
    with tempfile.TemporaryDirectory() as tmp:
        bad_path = os.path.join(tmp, "bad.json")
        with open(bad_path, "w") as f:
            json.dump({"kind": "smoke", "seed": 1}, f)
        r = check_manifest(bad_path)
        assert not r["ok"]
        assert "command" in r["missing_fields"]


def test_git_commit_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        p = write_manifest(tmp, kind="smoke", seed=1)
        with open(p) as f:
            m = json.load(f)
    # Should have git commit since we're in a git repo
    assert m["git_commit"] is not None
    assert len(m["git_commit"]) == 40


if __name__ == "__main__":
    test_manifest_contains_required_fields()
    test_check_manifest_flags_missing_fields()
    test_git_commit_recorded()
    print("ALL provenance TESTS PASSED")
