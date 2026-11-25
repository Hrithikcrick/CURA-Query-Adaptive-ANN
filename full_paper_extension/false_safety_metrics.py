"""Pure metrics for frozen-ranking judgment interventions; no file I/O.

Returned-depth-ten alternatives do not reconstruct deeper ANN results. Bpref
uses grade>=2; judgment thinning still removes grade>0 records, preserving the
historical intervention. Inapplicable bpref is NaN, never automatically safe.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

EFS = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256)
CONDITIONS = ('FULL', '50_PERCENT_POSITIVES', '25_PERCENT_POSITIVES', 'ONE_POSITIVE_PER_QUERY')
METRICS = ('standard_ndcg_at10', 'condensed_returned10_ndcg_at10', 'bpref_returned10')
TOLERANCE = 0.025


def dcg(grades: Sequence[int]) -> float:
    return sum((2 ** int(g) - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg(documents: Sequence[str], qrels: Mapping[str, int], *, condensed=False) -> float:
    denominator = dcg(sorted((g for g in qrels.values() if g > 0), reverse=True)[:10])
    if denominator == 0:
        return 0.0
    returned = list(documents[:10])
    if condensed:
        returned = [d for d in returned if d in qrels]
    return dcg([qrels.get(d, 0) for d in returned]) / denominator


def bpref(documents: Sequence[str], qrels: Mapping[str, int]) -> float:
    """Current trec_eval formula, grade>=2, on the submitted top10 list.

    Both binary classes must exist. This deliberately reports NaN for
    degenerate preference comparisons, rather than inheriting software defaults.
    """
    relevant = sum(g >= 2 for g in qrels.values())
    nonrelevant = sum(0 <= g < 2 for g in qrels.values())
    if relevant == 0 or nonrelevant == 0:
        return float('nan')
    before = 0
    total = 0.0
    for doc in documents[:10]:
        if doc not in qrels:
            continue
        grade = qrels[doc]
        if grade >= 2:
            total += 1.0 - min(before, relevant) / min(relevant, nonrelevant)
        elif grade >= 0:
            before += 1
    return total / relevant


def thin_query(qrels: Mapping[str, int], raw_qid: str, seed: int) -> dict[str, dict[str, int]]:
    positives = sorted(d for d, grade in qrels.items() if grade > 0)
    if not positives:
        raise ValueError(f'Query {raw_qid} has no positive FULL judgments')
    key = hashlib.sha256(f'{seed}\0{raw_qid}'.encode('utf-8')).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(key[:8], 'big')))
    permutation = [positives[i] for i in rng.permutation(len(positives))]
    result = {}
    for condition, size in zip(CONDITIONS[1:], [math.ceil(.5 * len(positives)), math.ceil(.25 * len(positives)), 1]):
        retained = set(permutation[:max(1, size)])
        result[condition] = {d: g for d, g in qrels.items() if g <= 0 or d in retained}
    return result


def safe_labels(exact: np.ndarray, approximate: np.ndarray) -> np.ndarray:
    """Float labels permit NaN=metric inapplicable, 0=unsafe and 1=safe."""
    loss = np.maximum(0, exact[..., None] - approximate)
    return np.where(np.isfinite(loss), (loss <= TOLERANCE).astype(float), np.nan)


def safe_minimum(labels: np.ndarray, grid=EFS) -> np.ndarray:
    """Smallest safe grid element; no-safe and all-NA both remain NaN."""
    return np.min(np.where(labels == 1, np.asarray(grid, float), np.inf), axis=-1).clip(max=np.inf)


def finite_minimum(labels: np.ndarray, grid=EFS) -> np.ndarray:
    values = safe_minimum(labels, grid)
    return np.where(np.isfinite(values), values, np.nan)


def optimism(full_minimum: np.ndarray, thin_minimum: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(full_minimum) & np.isfinite(thin_minimum), full_minimum - thin_minimum, np.nan)


def confusion_counts(full: np.ndarray, thin: np.ndarray) -> dict[str, int]:
    """Pairs lacking either valid metric are visible in planned-minus-valid."""
    full, thin = np.broadcast_arrays(full, thin)
    valid = np.isfinite(full) & np.isfinite(thin)
    return dict(
        planned_pairs=int(full.size), valid_pairs=int(valid.sum()),
        missing_pairs=int((~valid).sum()),
        false_safe=int(((full == 0) & (thin == 1) & valid).sum()),
        false_unsafe=int(((full == 1) & (thin == 0) & valid).sum()),
        both_safe=int(((full == 1) & (thin == 1) & valid).sum()),
        both_unsafe=int(((full == 0) & (thin == 0) & valid).sum()),
        thin_safe=int(((thin == 1) & valid).sum()),
        full_unsafe=int(((full == 0) & valid).sum()),
        full_unsafe_planned=int((full == 0).sum()),
    )


def rates(counts: Mapping[str, float]) -> dict[str, float]:
    def ratio(a, b):
        return float(a / b) if b else float('nan')
    sensitivity = ratio(counts['both_unsafe'], counts['full_unsafe'])
    specificity = ratio(counts['both_safe'], counts['both_safe'] + counts['false_unsafe'])
    return dict(
        fsr_all=ratio(counts['false_safe'], counts['planned_pairs']) if counts['missing_pairs'] == 0 else float('nan'),
        fsr_all_observed_lower_bound=ratio(counts['false_safe'], counts['planned_pairs']),
        fsr_all_possible_upper_bound=ratio(counts['false_safe'] + counts['missing_pairs'], counts['planned_pairs']),
        fsr_valid=ratio(counts['false_safe'], counts['valid_pairs']),
        fsr_given_thin_safe=ratio(counts['false_safe'], counts['thin_safe']),
        reverse_disagreement_rate=ratio(counts['false_unsafe'], counts['valid_pairs']),
        agreement=ratio(counts['both_safe'] + counts['both_unsafe'], counts['valid_pairs']),
        full_unsafe_sensitivity=sensitivity,
        full_unsafe_sensitivity_all_reference_pairs=ratio(counts['both_unsafe'], counts['full_unsafe_planned']),
        balanced_accuracy=(sensitivity + specificity) / 2 if np.isfinite(sensitivity + specificity) else float('nan'),
        full_unsafe_prevalence=ratio(counts['full_unsafe'], counts['valid_pairs']),
        applicability_fraction=ratio(counts['valid_pairs'], counts['planned_pairs']),
    )


@dataclass
class Evaluation:
    exact: np.ndarray       # metric, query
    approximate: np.ndarray # metric, query, ef
    labels: np.ndarray      # metric, query, ef
    minimum: np.ndarray     # metric, query
    judged_exact: np.ndarray
    judged_hnsw: np.ndarray


def evaluate(exact: Sequence[Sequence[str]], hnsw: Sequence[Sequence[Sequence[str]]],
             qrels: Sequence[Mapping[str, int]]) -> Evaluation:
    n = len(exact)
    if len(hnsw) != n or len(qrels) != n:
        raise ValueError('Unequal exact/HNSW/qrel query coverage')
    reference = np.empty((len(METRICS), n), float)
    approximate = np.empty((len(METRICS), n, len(EFS)), float)
    judged_exact = np.empty(n, int)
    judged_hnsw = np.empty((n, len(EFS)), int)
    for qi in range(n):
        documents = list(exact[qi][:10])
        if len(documents) != 10 or len(set(documents)) != 10 or len(hnsw[qi]) != len(EFS):
            raise ValueError('Expected ten unique exact results and the full ef grid')
        judgments = qrels[qi]
        judged_exact[qi] = sum(d in judgments for d in documents)
        scorers = [lambda d: ndcg(d, judgments), lambda d: ndcg(d, judgments, condensed=True), lambda d: bpref(d, judgments)]
        for mi, scorer in enumerate(scorers):
            reference[mi, qi] = scorer(documents)
        for ei, ranking in enumerate(hnsw[qi]):
            if len(ranking) < 10 or len(set(ranking[:10])) != 10:
                raise ValueError('Every ANN list must have ten unique documents')
            judged_hnsw[qi, ei] = sum(d in judgments for d in ranking[:10])
            for mi, scorer in enumerate(scorers):
                approximate[mi, qi, ei] = scorer(ranking)
    labels = safe_labels(reference, approximate)
    return Evaluation(reference, approximate, labels, finite_minimum(labels), judged_exact, judged_hnsw)


def spearman_finite(a, b) -> tuple[float, int]:
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        return float('nan'), len(a)
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]), len(a)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid], kind='stable')]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1)))
    return result
