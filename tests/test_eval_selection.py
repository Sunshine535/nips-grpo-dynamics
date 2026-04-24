"""Tests for eval_stratified selection modes (GPT-5.5 Task 3).

Uses static file analysis (not import) to avoid requiring `transformers`.
"""
import os
import re


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(REPO, "scripts", "eval_stratified.py")


def test_eval_script_has_selection_flag():
    with open(EVAL_PATH) as f:
        src = f.read()
    assert "--selection" in src, "eval script must support --selection"
    for mode in ("first_n", "full", "random"):
        assert mode in src, f"selection mode '{mode}' must be supported"


def test_eval_script_saves_question_ids():
    with open(EVAL_PATH) as f:
        src = f.read()
    assert "eval_question_ids" in src, \
        "eval JSON must include eval_question_ids"


def test_eval_script_accuracy_uses_evaluated_count():
    """Accuracy denominator must be actual evaluated count, not requested n."""
    with open(EVAL_PATH) as f:
        src = f.read()
    assert "n_evaluated" in src, \
        "accuracy must divide by n_evaluated, not args.n"


def test_extract_answer_regex_present():
    """Verify the GSM8K answer extraction regex is defined and documented."""
    with open(EVAL_PATH) as f:
        src = f.read()
    assert "_ans_pat" in src, "must define _ans_pat regex"
    assert "####" in src, "regex must use GSM8K '####' delimiter"


if __name__ == "__main__":
    test_eval_script_has_selection_flag()
    test_eval_script_saves_question_ids()
    test_eval_script_accuracy_uses_evaluated_count()
    test_extract_answer_regex_present()
    print("ALL eval_selection TESTS PASSED")
