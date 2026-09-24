"""Per-task metrics for LongBench, ported from the official evaluation.

LongBench is a mixed benchmark: QA tasks use token-overlap F1, summarization
uses ROUGE-L, classification uses label matching, retrieval/counting use exact
number matching, and code tasks use edit-similarity. Scoring every task with F1
(as the previous implementation did) is only valid for the QA subset and yields
meaningless numbers elsewhere.

Reference: THUDM/LongBench `metrics.py` and `eval.py`.

Optional dependencies are imported lazily so English QA tasks run with no extra
packages; ROUGE / Chinese tasks require `rouge` and `jieba` respectively.
"""

import difflib
import re
import string
from collections import Counter


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


_CN_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】"
    "〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
)


def normalize_zh_answer(s: str) -> str:
    all_punctuation = set(string.punctuation + _CN_PUNCTUATION)
    text = "".join(ch for ch in s.lower() if ch not in all_punctuation)
    return "".join(text.split())


def _f1(prediction_tokens, ground_truth_tokens) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


# --------------------------------------------------------------------------- #
# Task metrics
# --------------------------------------------------------------------------- #
def qa_f1_score(prediction, ground_truth, **kwargs) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    return _f1(p, g)


def qa_f1_zh_score(prediction, ground_truth, **kwargs) -> float:
    import jieba
    p = [normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    g = [normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    p = [t for t in p if len(t) > 0]
    g = [t for t in g if len(t) > 0]
    return _f1(p, g)


def rouge_score(prediction, ground_truth, **kwargs) -> float:
    from rouge import Rouge
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs) -> float:
    import jieba
    prediction = " ".join(jieba.cut(prediction, cut_all=False))
    ground_truth = " ".join(jieba.cut(ground_truth, cut_all=False))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction, ground_truth, all_classes=None, **kwargs) -> float:
    all_classes = all_classes or []
    em_match_list = [c for c in all_classes if c in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction, ground_truth, **kwargs) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return right / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return right / len(numbers)


def count_score(prediction, ground_truth, **kwargs) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if str(n) == str(ground_truth))
    return right / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs) -> float:
    # Official uses fuzzywuzzy.fuzz.ratio; difflib's SequenceMatcher is the same
    # underlying algorithm and needs no extra dependency.
    all_lines = prediction.lstrip('\n').split('\n')
    line = ""
    for candidate in all_lines:
        if ('`' not in candidate) and ('#' not in candidate) and ('//' not in candidate):
            line = candidate
            break
    return difflib.SequenceMatcher(None, line, ground_truth).ratio()


# Dataset -> metric function (base names; LongBench-E variants share the metric).
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Tasks whose prediction should be truncated to the first line before scoring.
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def score_prediction(dataset: str, prediction: str, answers: list, all_classes=None) -> float:
    """Score one prediction against its reference answers (max over answers)."""
    metric = dataset2metric.get(dataset)
    if metric is None:
        # Unknown dataset: fall back to QA F1 but make the ambiguity visible.
        print(f"[LongBench] Warning: no official metric registered for '{dataset}', "
              f"falling back to qa_f1_score.")
        metric = qa_f1_score

    if dataset in FIRST_LINE_TASKS:
        prediction = prediction.lstrip('\n').split('\n')[0]

    best = 0.0
    for gt in answers:
        best = max(best, metric(prediction, str(gt), all_classes=all_classes))
    return best
