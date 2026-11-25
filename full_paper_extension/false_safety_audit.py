"""Verify completed TREC2019 findings from saved artifacts, without retrieval.

This is a verification utility written on 2026-11-26, not a recovered original
extension script. It independently scores the saved rankings using the saved
thinned judgments. The existing scientific artifacts are never overwritten.
By default JSON goes to stdout; only an explicit --output writes a result file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EFS = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256)
THIN = ('50_PERCENT_POSITIVES', '25_PERCENT_POSITIVES', 'ONE_POSITIVE_PER_QUERY')
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ndcg_at10(ranking, judgments):
    """Historical exponential gain, global ideal top ten, unjudged=zero."""
    ideal = sorted((int(g) for g in judgments.values() if g > 0), reverse=True)[:10]
    denominator = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if not denominator:
        return 0.0
    return sum((2 ** int(judgments.get(d, 0)) - 1) / math.log2(i + 2)
               for i, d in enumerate(ranking[:10])) / denominator


def audit_losses(full_loss, thin_loss, thin_exact, epsilon):
    """Count disagreements; conditional denominator is FULL-unsafe instances."""
    full_loss, thin_loss, thin_exact = map(lambda x: np.asarray(x, dtype=float),
                                          (full_loss, thin_loss, thin_exact))
    if full_loss.shape != thin_loss.shape or full_loss.shape != thin_exact.shape:
        raise ValueError('Loss and reference arrays must have identical shapes')
    if not all(np.isfinite(x).all() for x in (full_loss, thin_loss, thin_exact)):
        raise ValueError('Nonfinite observations cannot be silently classified')
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    full_unsafe = np.maximum(0, full_loss) > epsilon
    thin_safe = np.maximum(0, thin_loss) <= epsilon
    false_safe = full_unsafe & thin_safe
    total, denominator, count = int(full_loss.size), int(full_unsafe.sum()), int(false_safe.sum())
    positive = int((false_safe & (thin_exact > 0)).sum())
    return dict(
        evaluated_pair_instances=total, full_unsafe_pair_instances=denominator,
        false_safe_count=count,
        conditional_false_safe_rate=count / denominator if denominator else None,
        all_pair_false_safe_rate=count / total if total else None,
        false_safe_positive_exact_count=positive,
        false_safe_zero_exact_count=int((false_safe & (thin_exact == 0)).sum()),
        false_safe_positive_exact_fraction=positive / count if count else None,
        false_unsafe_count=int((~full_unsafe & ~thin_safe).sum()),
    )


def validate_grid(frame):
    conditions = {'FULL', *THIN}
    if set(frame.condition) != conditions or len(frame) != 64930:
        raise ValueError('Expected all 64,930 saved query-condition-budget rows')
    keys = set(frame.query_key)
    if len(keys) != 43 or set(frame.efSearch) != set(EFS):
        raise ValueError('Expected 43 queries and the ten original efSearch values')
    if frame.duplicated(['condition', 'seed', 'query_key', 'efSearch']).any():
        raise ValueError('Duplicate query-condition-seed-budget records')
    grid = {(key, ef) for key in keys for ef in EFS}
    for condition in conditions:
        part = frame[frame.condition.eq(condition)]
        expected_seeds = {-1} if condition == 'FULL' else set(range(50))
        if set(part.seed) != expected_seeds:
            raise ValueError(f'Incomplete or unexpected seeds: {condition}')
        for seed, group in part.groupby('seed'):
            if set(zip(group.query_key, group.efSearch)) != grid:
                raise ValueError(f'Incomplete grid: {condition}, seed {seed}')
    return sorted(keys)


def verify(root=ROOT):
    root = Path(root).resolve()
    relative = dict(
        saved_rows='sigir_analysis/trec_judgment_density_per_query.csv',
        exact='sigir_analysis/trec_exact_rankings.csv',
        hnsw='sigir_analysis/trec_hnsw_rankings.csv',
        thinned='sigir_analysis/trec_thinned_positive_judgments.jsonl.gz',
        qrels='full_paper_extension/data/2019qrels-pass.txt',
        provenance='sigir_analysis/trec_ranking_provenance.json',
        qrels_provenance='sigir_analysis/qrels_recovery.json',
        fixed='sigir_analysis/trec_fixed_risk_work_by_density.csv',
        cura='sigir_analysis/trec_cura_by_density.csv',
        original_work='msmarco_large/msmarco_1000000_q1000/research_grade_validation/top10_query_budget_results.csv',
    )
    paths = {key: root / value for key, value in relative.items()}
    before = {relative[key]: sha256(path) for key, path in paths.items()}
    provenance = json.loads(paths['provenance'].read_text(encoding='utf-8'))
    for key in ('exact', 'hnsw'):
        if before[relative[key]] != provenance[f'{key}_rankings_sha256']:
            raise ValueError(f'Original ranking checksum mismatch: {key}')
    qrel_provenance = json.loads(paths['qrels_provenance'].read_text(encoding='utf-8'))
    if before[relative['qrels']] != qrel_provenance['sha256']:
        raise ValueError('Qrels differ from checksum-verified original pooled qrels')
    rows = pd.read_csv(paths['saved_rows'])
    keys = validate_grid(rows)
    qrels = {}
    for line in paths['qrels'].read_text(encoding='utf-8').splitlines():
        qid, _, doc, grade = line.split()
        if doc in qrels.setdefault(qid, {}):
            raise ValueError(f'Duplicate qrel: {qid}/{doc}')
        qrels[qid][doc] = int(grade)
    if set(qrels) != {key.split('::', 1)[1] for key in keys}:
        raise ValueError('Original raw query identities differ from saved rows')
    rankings = {}
    for name, group_columns in (('exact', ['query_key']), ('hnsw', ['query_key', 'efSearch'])):
        frame = pd.read_csv(paths[name], dtype={'doc_id': str, 'query_id': str})
        for group_key, group in frame.groupby(group_columns):
            group = group.sort_values('rank')
            if list(group['rank']) != list(range(1, len(group) + 1)) or group.doc_id.duplicated().any():
                raise ValueError(f'Invalid ordered ranking: {name}/{group_key}')
            if len(group) < 10:
                raise ValueError('Scoring requires all ten ranked document IDs')
            rankings[name, *group_key] = group.doc_id.tolist()[:10]
    if {k[1] for k in rankings if k[0] == 'exact'} != set(keys):
        raise ValueError('Exact ranking query coverage differs from the saved grid')
    if {(k[1], k[2]) for k in rankings if k[0] == 'hnsw'} != {(k, ef) for k in keys for ef in EFS}:
        raise ValueError('HNSW ranking coverage differs from the saved grid')
    retained = {}
    with gzip.open(paths['thinned'], 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            identity = (record['condition'], int(record['seed']), record['query_key'])
            if identity in retained or record['raw_query_id'] != record['query_key'].split('::', 1)[1]:
                raise ValueError('Duplicate or conflicting saved thinning identity')
            retained[identity] = record['retained_positive_judgments']
    expected = {(condition, seed, key) for condition in THIN for seed in range(50) for key in keys}
    if set(retained) != expected:
        raise ValueError('Saved positive-judgment selections do not cover all seeds/queries')
    score_error, recall_error = 0.0, 0.0
    for (condition, seed, key), group in rows.groupby(['condition', 'seed', 'query_key']):
        original = qrels[key.split('::', 1)[1]]
        judgments = original
        if condition != 'FULL':
            selected = retained[condition, int(seed), key]
            if not all(doc in original and original[doc] == grade and grade > 0
                       for doc, grade in selected.items()):
                raise ValueError('Thinned selections changed original grades or added judgments')
            size = sum(g > 0 for g in original.values())
            expected_count = {'50_PERCENT_POSITIVES': math.ceil(.5 * size),
                              '25_PERCENT_POSITIVES': math.ceil(.25 * size),
                              'ONE_POSITIVE_PER_QUERY': 1}[condition]
            if len(selected) != expected_count:
                raise ValueError('Incorrect number of retained positive judgments')
            positives = sorted(doc for doc, grade in original.items() if grade > 0)
            seed_digest = hashlib.sha256(f'{int(seed)}\0{key.split("::", 1)[1]}'.encode()).digest()
            generator = np.random.Generator(np.random.PCG64(int.from_bytes(seed_digest[:8], 'big')))
            ordered = [positives[i] for i in generator.permutation(len(positives))]
            if set(selected) != set(ordered[:expected_count]):
                raise ValueError('Saved selections differ from the paired deterministic PCG64 intervention')
            judgments = {doc: grade for doc, grade in original.items() if grade <= 0} | selected
        exact = ndcg_at10(rankings['exact', key], judgments)
        for row in group.itertuples(index=False):
            ranking = rankings['hnsw', key, row.efSearch]
            approximate = ndcg_at10(ranking, judgments)
            loss = max(0, exact - approximate)
            score_error = max(score_error, abs(exact - row.exact_nDCG_at_10),
                              abs(approximate - row.HNSW_nDCG_at_10), abs(loss - row.nDCG_loss))
            recall = len(set(rankings['exact', key]) & set(ranking)) / 10
            recall_error = max(recall_error, abs(recall - row.ANN_Recall_at_10))
    if score_error > 1e-12 or recall_error > 1e-12:
        raise ValueError(f'Saved score reproduction failed: {score_error=}, {recall_error=}')
    if rows.ranking_sha256.nunique() != 1:
        raise ValueError('Ranking fingerprints changed across judgment conditions')
    if rows.groupby(['query_key', 'efSearch']).ANN_Recall_at_10.nunique().max() != 1:
        raise ValueError('ANN Recall changed across judgment conditions')
    fixed = pd.read_csv(paths['fixed'])
    work = pd.read_csv(paths['original_work'])
    work = work[work.query_key.isin(keys)]
    if work.duplicated(['query_key', 'efSearch']).any() or len(work) != 430:
        raise ValueError('Saved original work does not have one row per query-budget pair')
    for ef in EFS:
        group = fixed[fixed.efSearch.eq(ef)]
        if set(zip(group.condition, group.seed)) != {('FULL', -1)} | {(c, s) for c in THIN for s in range(50)}:
            raise ValueError('Fixed work frontier has incomplete condition/seed coverage')
        for column in ('mean_distance_computations', 'mean_latency_ms', 'p95_latency_ms', 'ANN_Recall_at_10'):
            if group[column].nunique() != 1:
                raise ValueError(f'Frozen fixed work/ANN invariant failed: {column}')
        if not np.allclose(group.mean_distance_computations, work[work.efSearch.eq(ef)].distance_computations.mean(), atol=1e-9, rtol=0):
            raise ValueError('Fixed mean work differs from the original saved measurements')
    cura = pd.read_csv(paths['cura'])
    for column in ('selection_sha256', 'mean_work', 'mean_latency_ms', 'p95_latency_ms', 'ANN_Recall_at_10'):
        if cura[column].nunique() != 1:
            raise ValueError(f'Frozen CURA choice/work invariant failed: {column}')
    full = rows[rows.condition.eq('FULL')]
    full_losses = full[['query_key', 'efSearch', 'nDCG_loss']].rename(columns={'nDCG_loss': 'full_loss'})
    sensitivity = []
    for epsilon in (.01, .025, .05):
        conditions = {}
        for condition in THIN:
            thin = rows[rows.condition.eq(condition)].merge(full_losses, on=['query_key', 'efSearch'], validate='many_to_one')
            conditions[condition] = audit_losses(thin.full_loss, thin.nDCG_loss, thin.exact_nDCG_at_10, epsilon)
        sensitivity.append(dict(tolerance=epsilon, full_unsafe_pairs=int(full.nDCG_loss.gt(epsilon).sum()), conditions=conditions))
    after = {relative[key]: sha256(path) for key, path in paths.items()}
    if before != after:
        raise ValueError('A scientific input changed during verification')
    return dict(
        status='PASS', utility_origin='New verification utility, written 2026-11-26; not recovered original extension code',
        collection='TREC DL 2019', query_count=43, efSearch=list(EFS), full_query_budget_pairs=430,
        paired_thinning_seeds=50, saved_rows=len(rows), score_reproduction_max_absolute_error=score_error,
        saved_selections_match_paired_deterministic_PCG64_intervention=True,
        recall_reproduction_max_absolute_error=recall_error, independently_rescored_rows=len(rows),
        rankings_unchanged=True, ANN_recall_unchanged=True, fixed_work_unchanged=True,
        frozen_CURA_selections_and_work_unchanged=True, inputs_unchanged_during_verification=True,
        tolerance_sensitivity=sensitivity, source_sha256=before,
        limitations=[
            'FULL is the original pooled judgment set, not exhaustive relevance truth.',
            'Conditional percentages use FULL-unsafe pairs repeated across paired seeds, not all 430 pairs.',
            'Budgets within queries and seeds are dependent; counts are descriptive, not independent trials.',
            'This verifies the saved TREC2019 intervention and does not establish cross-domain controller transfer.',
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='Explicit new JSON destination; existing files are never overwritten')
    args = parser.parse_args()
    result = verify(args.root)
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
