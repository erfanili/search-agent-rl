"""
Answer metrics and trajectory parsing for multi-hop QA (P0.8; moved to agent/ in P1.3).

Standard library only, on purpose: the same functions run in the eval scripts (.venv) and,
later, as the GRPO reward inside verl (.venv-verl). Training reward and evaluation must
score answers with one shared piece of code, or a score difference between them can hide
a formatting drift.

Metrics compare one predicted answer against a LIST of gold answers and keep the best match,
after SQuAD-style normalization: lowercase, drop punctuation, drop a/an/the, squeeze spaces.

  exact_match   1 if the normalized answer equals some normalized gold answer
  token_f1      best token-overlap F1 over the gold answers (partial credit)
  cover_match   1 if some normalized gold answer is a substring of the normalized answer
                (lenient; Search-R1 calls it subEM, 2609.28765 calls it ACC)

Trajectory parsing:
  strip_environment_text  remove text the environment inserted: <information> blocks and
                          Search-R1's "invalid action" message
  extract_answer          contents of the LAST complete <answer>...</answer> block, or None
  count_searches          number of <search>...</search> blocks

score_output strips environment text before parsing. This matters: Search-R1's invalid-action
message itself contains "<answer> and </answer>", so without stripping, a trajectory that ends
after an invalid turn is scored as answering "and" (an abort counted as an answer). A retrieved
passage could also contain tag-like text.

Pass a trajectory without the prompt. Search-R1's prompt contains an example <answer> tag,
which is why its own scorer skips the first match; scoring without the prompt avoids that.
"""

import re
import string
from collections import Counter

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
INFO_RE = re.compile(r"<information>.*?</information>", re.DOTALL)

# Search-R1's message for a turn with neither a <search> nor an <answer> block. One copy, used by
# the rollout (agent/search_r1_rollout.py) and by strip_environment_text, so they cannot drift.
SEARCH_R1_INVALID = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _as_list(golds):
    return [golds] if isinstance(golds, str) else list(golds)


def exact_match(pred, golds):
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in _as_list(golds)))


def _f1(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred_tokens), same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred, golds):
    p = normalize_answer(pred).split()
    return max(_f1(p, normalize_answer(g).split()) for g in _as_list(golds))


def cover_match(pred, golds):
    p = normalize_answer(pred)
    for gold in _as_list(golds):
        g = normalize_answer(gold)
        if g and g in p:
            return 1.0
    return 0.0


def strip_environment_text(text):
    """Model-written text only: drop <information> blocks and invalid-action messages."""
    text = (text or "").replace(SEARCH_R1_INVALID.strip(), "")
    return INFO_RE.sub("", text)


def extract_answer(text):
    found = ANSWER_RE.findall(text or "")
    return found[-1].strip() if found else None


def count_searches(text):
    return len(SEARCH_RE.findall(text or ""))


def _scores(answer, golds):
    if answer is None:
        return {"em": 0.0, "f1": 0.0, "acc": 0.0}
    return {"em": exact_match(answer, golds), "f1": token_f1(answer, golds), "acc": cover_match(answer, golds)}


def score_output(output, golds):
    """Score an agent trajectory (environment text is stripped first).
    No complete <answer> block = abort = 0 on every metric."""
    model_text = strip_environment_text(output)
    answer = extract_answer(model_text)
    return {"answer": answer, "answered": answer is not None, "n_search": count_searches(model_text), **_scores(answer, golds)}


def score_prediction(prediction, golds):
    """Score a bare answer string (no-retrieval and naive-RAG baselines). None = no answer."""
    return {"answer": prediction, "answered": prediction is not None, "n_search": 0, **_scores(prediction, golds)}


METRIC_KEYS = ("em", "f1", "acc", "answer_rate", "abort_rate", "mean_searches")


def aggregate(rows):
    """rows: per-example dicts with dataset, em, f1, acc, answered, n_search.
    Returns per-dataset means plus macro_avg: the unweighted mean over datasets, so a large
    dev set (HotpotQA 7,405) does not drown a small one (MuSiQue 2,417)."""
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    out = {}
    for name, rs in sorted(by.items()):
        n = len(rs)
        answered = sum(r["answered"] for r in rs) / n
        out[name] = {
            "n": n,
            "em": sum(r["em"] for r in rs) / n,
            "f1": sum(r["f1"] for r in rs) / n,
            "acc": sum(r["acc"] for r in rs) / n,
            "answer_rate": answered,
            "abort_rate": 1 - answered,
            "mean_searches": sum(r["n_search"] for r in rs) / n,
        }
    if out:
        out["macro_avg"] = {k: sum(v[k] for v in out.values()) / len(by) for k in METRIC_KEYS}
        out["macro_avg"]["n_datasets"] = len(by)
    return out