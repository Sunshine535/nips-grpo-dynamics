"""MATH partial-credit reward function for TASA-GRPO experiments."""
import re

_BOXED_PAT = re.compile(r"\\boxed\{([^}]+)\}")
_ANS_PAT = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_THINK_PAT = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUM_PAT = re.compile(r"(-?\d+(?:\.\d+)?)")


def _extract_answer(text: str) -> str:
    text = _THINK_PAT.sub("", text).strip()
    m = _BOXED_PAT.search(text)
    if m:
        return m.group(1).strip()
    m = _ANS_PAT.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = _NUM_PAT.findall(text)
    return nums[-1] if nums else ""


def _try_numeric(s: str):
    try:
        s = s.replace(",", "").strip()
        return float(s)
    except (ValueError, TypeError):
        return None


def _numeric_close(pred_str: str, gold_str: str, rtol: float = 0.01) -> bool:
    pv = _try_numeric(pred_str)
    gv = _try_numeric(gold_str)
    if pv is None or gv is None:
        return False
    if gv == 0:
        return abs(pv) < 1e-6
    return abs(pv - gv) / max(abs(gv), 1e-8) <= rtol


def math_partial_credit_reward(pred_text: str, gold_answer: str) -> float:
    pred = _extract_answer(pred_text)
    if not pred:
        return 0.0
    gold = gold_answer.strip()
    if pred.strip() == gold:
        return 1.0
    if _numeric_close(pred, gold, rtol=0.01):
        return 0.7
    return 0.0


def build_math_partial_credit_reward_function():
    def reward_fn(prompts, completions, **kwargs):
        answers = kwargs.get("answer", [])
        rewards = []
        for comp, ans in zip(completions, answers):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            gold = _ANS_PAT.search(ans).group(1).replace(",", "") if _ANS_PAT.search(ans) else ans
            rewards.append(math_partial_credit_reward(text, gold))
        return rewards
    reward_fn.__name__ = "math_partial_credit"
    return reward_fn
