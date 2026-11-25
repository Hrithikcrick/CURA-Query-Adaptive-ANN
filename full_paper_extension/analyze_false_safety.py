"""Analyze fixed-ranking qrel interventions; never retrieve or train models.

Run from the repository root after the Phase 0 audit and protocol are frozen:
    python full_paper_extension/analyze_false_safety.py --collections 2019
    python full_paper_extension/analyze_false_safety.py --collections 2019 2020

All writes stay in full_paper_extension/. Missing collection inputs fail loudly.
Every run records protocol and source hashes, software versions, and arguments.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import scipy

from false_safety_metrics import (
    CONDITIONS, EFS, METRICS, TOLERANCE, confusion_counts, evaluate,
    finite_minimum, holm, optimism, rates, spearman_finite, thin_query,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'full_paper_extension'
BASE = ROOT / 'msmarco_large/msmarco_1000000_q1000'
COUNT_COLUMNS = list(confusion_counts(np.array([1.]), np.array([1.])))
RNG_SEED = 20261118


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(a, b):
    return float(a / b) if b else float('nan')


def mean_finite(values):
    values = np.asarray(values, float)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float('nan')


def read_qrels(path):
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        qid, _, docid, grade = line.split()
        grade = int(grade)
        if grade < 0:
            raise ValueError(f'Unsupported negative qrel grade at {path}:{number}')
        query = result.setdefault(qid, {})
        if docid in query:
            raise ValueError(f'Duplicate qrel {qid}/{docid}')
        query[docid] = grade
    return result


@dataclass
class Collection:
    year: str
    keys: list[str]
    raw_ids: list[str]
    exact: list
    hnsw: list
    qrels: list
    work: np.ndarray
    latency: np.ndarray
    recall: np.ndarray
    ann_minimum: np.ndarray
    source_paths: list[Path]
    validation: dict
    full: object


def load_collection(year):
    if year == '2019':
        exact_path = ROOT / 'sigir_analysis/trec_exact_rankings.csv'
        hnsw_path = ROOT / 'sigir_analysis/trec_hnsw_rankings.csv'
        work_path = BASE / 'research_grade_validation/top10_query_budget_results.csv'
        qrels_path = OUT / 'data/2019qrels-pass.txt'
        if not qrels_path.exists():
            qrels_path = ROOT / 'local_data/trec2019/2019qrels-pass.txt'
    else:
        exact_path = OUT / f'trec{year}_exact_rankings.csv'
        hnsw_path = OUT / f'trec{year}_hnsw_rankings.csv'
        work_path = OUT / f'trec{year}_search_work.csv'
        qrels_path = OUT / 'data/2020qrels-pass.txt'
    for path in (exact_path, hnsw_path, work_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f'{year} blocked: required input {path.relative_to(ROOT)} is absent')
    exact_frame = pd.read_csv(exact_path, dtype={'query_key': str, 'query_id': str, 'doc_id': str})
    hnsw_frame = pd.read_csv(hnsw_path, dtype={'query_key': str, 'query_id': str, 'doc_id': str})
    work_frame = pd.read_csv(work_path, dtype={'query_key': str})
    qrel_map = read_qrels(qrels_path)
    if exact_frame.query_key.isna().any() or exact_frame.doc_id.isna().any():
        raise ValueError('Missing exact query/document identity')
    keys = sorted(exact_frame.query_key.unique())
    raw_ids = [key.split('::', 1)[1] for key in keys]
    if len(set(raw_ids)) != len(keys) or set(raw_ids) != set(qrel_map):
        raise ValueError('Exact query IDs must cover the entire collection qrel query set')
    if set(hnsw_frame.query_key) != set(keys):
        raise ValueError('HNSW query set differs from exact query set')
    expected = {(key, ef) for key in keys for ef in EFS}
    if set(zip(hnsw_frame.query_key, hnsw_frame.efSearch)) != expected:
        raise ValueError('Incomplete or extra HNSW query/ef grid')
    rankings = {}
    for frame, columns, name in ((exact_frame, ['query_key'], 'exact'), (hnsw_frame, ['query_key', 'efSearch'], 'hnsw')):
        for groupkey, group in frame.groupby(columns, sort=False):
            group = group.sort_values('rank')
            if list(group['rank']) != list(range(1, len(group) + 1)) or len(group) < 10 or group.doc_id.duplicated().any():
                raise ValueError(f'{name} list is incomplete or duplicated: {groupkey}')
            if 'query_id' in group and set(group.query_id.astype(str)) != {str(group.iloc[0].query_key).split('::', 1)[1]}:
                raise ValueError('Raw query ID conflicts with prefixed query key')
            normalized_key = groupkey[0] if name == 'exact' else groupkey
            rankings[name, normalized_key] = group.doc_id.astype(str).tolist()[:10]
    exact = [rankings['exact', key] for key in keys]
    hnsw = [[rankings['hnsw', (key, ef)] for ef in EFS] for key in keys]
    query_qrels = [qrel_map[qid] for qid in raw_ids]
    full = evaluate(exact, hnsw, query_qrels)
    work_frame = work_frame[work_frame.query_key.isin(keys)].copy()
    if work_frame.duplicated(['query_key', 'efSearch']).any() or set(zip(work_frame.query_key, work_frame.efSearch)) != expected:
        raise ValueError('Saved work does not cover exactly one complete query/ef grid')
    indexed_work = work_frame.set_index(['query_key', 'efSearch']).loc[pd.MultiIndex.from_product([keys, EFS])]
    work = indexed_work.distance_computations.to_numpy(float).reshape(len(keys), len(EFS))
    latency = indexed_work.mean_latency_ms.to_numpy(float).reshape(len(keys), len(EFS))
    if not np.isfinite(work).all() or (work < 0).any() or not np.isfinite(latency).all() or (latency < 0).any():
        raise ValueError('Work/latency must be finite and nonnegative')
    recall = np.array([[len(set(exact[qi]) & set(ranking)) / 10 for ranking in hnsw[qi]] for qi in range(len(keys))])
    ann_minimum = finite_minimum((recall >= .9).astype(float))
    validation = dict(query_count=len(keys), query_budget_pairs=len(expected), complete_grid=True,
                      raw_query_ids_preserved=True, no_queries_dropped=True, full_reproduction=False)
    # FULL reproduction is a mandatory gate before any thinning outcomes.
    if year == '2019':
        expected_score = indexed_work['nDCG@10'].to_numpy(float).reshape(len(keys), len(EFS))
        expected_recall = indexed_work['ANN_Recall@10'].to_numpy(float).reshape(len(keys), len(EFS))
        expected_loss = indexed_work['nDCG_loss'].to_numpy(float).reshape(len(keys), len(EFS))
        score_error = float(np.max(np.abs(full.approximate[0] - expected_score)))
        recall_error = float(np.max(np.abs(recall - expected_recall)))
        loss_error = float(np.max(np.abs(np.maximum(0, full.exact[0, :, None] - full.approximate[0]) - expected_loss)))
        if score_error > 1e-6 or loss_error > 1e-6 or recall_error > 1e-9:
            raise ValueError(f'FULL 2019 reproduction failed: {score_error=}, {loss_error=}, {recall_error=}')
        validation.update(full_reproduction=True, full_nDCG_max_error=score_error, full_loss_max_error=loss_error,
                          full_ANN_recall_max_error=recall_error)
    else:
        gate_path = OUT / f'trec{year}_retrieval_validation.json'
        if not gate_path.exists():
            gate_path = OUT / f'trec{year}_reproduction_validation.json'
        if not gate_path.exists():
            raise FileNotFoundError('TREC2020 requires a separate retrieval identity/FULL validation JSON')
        gate = json.loads(gate_path.read_text(encoding='utf-8'))
        if gate.get('status') != 'PASS':
            raise ValueError(f'2020 source retrieval gate has not passed: {gate_path.name}')
        validation.update(full_reproduction=True, independent_historical_reproduction=False,
                          reproduction_scope='new rankings: separate retrieval validation plus explicit FULL scoring, not a historical reproduction')
    return Collection(year, keys, raw_ids, exact, hnsw, query_qrels, work, latency,
                      recall, ann_minimum, [exact_path, hnsw_path, work_path, qrels_path], validation, full)


def stratum_groups(collection):
    scores = collection.full.exact[0]
    cutpoints = np.quantile(scores, [.25, .5, .75], method='linear')
    quartiles = np.searchsorted(cutpoints, scores, side='left') + 1
    groups = [('overall', 'all', np.arange(len(scores)), np.arange(len(EFS)))]
    groups += [('efSearch', str(ef), np.arange(len(scores)), np.array([ei])) for ei, ef in enumerate(EFS)]
    groups += [('exact_FULL_nDCG_quartile', str(q), np.flatnonzero(quartiles == q), np.arange(len(EFS))) for q in range(1, 5)]
    groups += [('exact_FULL_zero', str(zero), np.flatnonzero((scores == 0) == zero), np.arange(len(EFS))) for zero in (False, True)]
    bins = ['ef10', 'ef16-32', 'ef48-96', 'ef128-256', 'ANN-grid-infeasible']
    difficulty = np.array([bins[4] if not np.isfinite(v) else bins[0] if v == 10 else bins[1] if v <= 32 else bins[2] if v <= 96 else bins[3] for v in collection.ann_minimum])
    groups += [('ANN_difficulty', label, np.flatnonzero(difficulty == label), np.arange(len(EFS))) for label in bins]
    return groups, dict(zip(['Q25', 'Q50', 'Q75'], map(float, cutpoints)))


def check_policy_scores(frame, collection, choices, name):
    if 'nDCG@10' in frame:
        expected = collection.full.approximate[0, np.arange(len(choices)), choices]
        if np.max(np.abs(expected - frame['nDCG@10'].to_numpy(float))) > 1e-6:
            raise ValueError(f'Frozen policy {name} ranking outcomes do not reproduce')


def load_policies(collection):
    """Returns original choices and their real charged work; no refitting."""
    policies = {}
    source_info = []
    for ei, ef in enumerate(EFS):
        policies[f'Fixed ef={ef}'] = (np.full(len(collection.keys), ei), collection.work[:, ei], collection.latency[:, ei], 'frozen single search')
    if collection.year != '2019':
        return policies, source_info
    candidates = [
        ('CURA', BASE / 'cura_controller/selected_cura_full_trec_ood.csv', 'effective_distance_computations', 'effective_latency_ms'),
        ('ANN-risk proxy (local)', BASE / 'cura_controller/selected_ann_risk_trec_ood.csv', 'effective_distance_computations', 'effective_latency_ms'),
        ('Score-Gap-Budget', BASE / 'score_threshold_baseline/selected_score_gap_trec_ood.csv', 'effective_distance_computations', 'effective_latency_ms'),
        ('Patience-inspired (local)', BASE / 'patience_style_baseline/patience_style_trec_ood.csv', 'realized_search_work', 'realized_latency_ms'),
    ]
    for name, path, workcol, timecol in candidates:
        original_path = path
        if name == 'Score-Gap-Budget' and not path.exists():
            path = OUT / 'data/selected_score_gap_trec_ood.csv'
        if not path.exists():
            source_info.append(dict(policy=name, status='BLOCKED', reason='saved selections absent', path=str(path.relative_to(ROOT))))
            continue
        if name == 'Score-Gap-Budget':
            # This source was local and untracked; preserve a portable copy with
            # explicit lineage rather than pretending it was historical Git data.
            target = OUT / 'data/selected_score_gap_trec_ood.csv'
            if target.exists() and sha(target) != sha(path):
                raise ValueError('Portable Score-Gap copy conflicts with its source')
            if not target.exists():
                shutil.copyfile(path, target)
            original_protocol = path.parent / 'protocol.json'
            portable_protocol = OUT / 'data/score_gap_protocol.json'
            if original_protocol.exists():
                if portable_protocol.exists() and sha(portable_protocol) != sha(original_protocol):
                    raise ValueError('Portable Score-Gap protocol conflicts with original')
                if not portable_protocol.exists():
                    shutil.copyfile(original_protocol, portable_protocol)
            path_used = target
        else:
            path_used = path
        frame = pd.read_csv(path_used, dtype={'query_key': str})
        if frame.query_key.duplicated().any() or set(frame.query_key) != set(collection.keys):
            raise ValueError(f'Policy {name} has missing/extra/duplicate query IDs')
        frame = frame.set_index('query_key').loc[collection.keys]
        choices = np.array([EFS.index(int(ef)) for ef in frame.efSearch], int)
        work, latency = frame[workcol].to_numpy(float), frame[timecol].to_numpy(float)
        if not np.isfinite(work).all() or (work < 0).any() or not np.isfinite(latency).all() or (latency < 0).any():
            raise ValueError(f'Invalid effective measurements for {name}')
        check_policy_scores(frame, collection, choices, name)
        policies[name] = (choices, work, latency, 'frozen original choices; saved effective work including probes/repeated searches')
        source_info.append(dict(policy=name, status='AVAILABLE', original_path=str(original_path.relative_to(ROOT)),
                                portable_input=str(path_used.relative_to(ROOT)), sha256=sha(path_used),
                                source_was_untracked=(name == 'Score-Gap-Budget')))
    return policies, source_info


def per_seed(collection, seeds):
    groups, cutpoints = stratum_groups(collection)
    policies, policy_sources = load_policies(collection)
    false_rows, budget_rows, robustness_rows, policy_rows, density_rows, evidence = [], [], [], [], [], []
    labels_by_condition = {condition: [] for condition in CONDITIONS}
    optimism_by_condition = {condition: [] for condition in CONDITIONS}
    condition_sets = [('FULL', -1, collection.qrels)]
    for seed in range(seeds):
        thinned = [thin_query(judgments, qid, seed) for judgments, qid in zip(collection.qrels, collection.raw_ids)]
        condition_sets.extend((condition, seed, [q[condition] for q in thinned]) for condition in CONDITIONS[1:])
    for condition, seed, judgments in condition_sets:
        result = collection.full if condition == 'FULL' else evaluate(collection.exact, collection.hnsw, judgments)
        labels_by_condition[condition].append(result.labels.copy())
        optimism_by_condition[condition].append(optimism(collection.full.minimum, result.minimum))
        std_joint = np.isfinite(collection.ann_minimum) & np.isfinite(result.minimum[0])
        density_rows.append(dict(collection=collection.year, condition=condition, seed=seed, query_count=len(collection.keys),
            jointly_feasible_query_count=int(std_joint.sum()),
            mean_ANN_safe_ef=mean_finite(collection.ann_minimum[std_joint]),
            mean_IR_safe_ef=mean_finite(result.minimum[0, std_joint]),
            ANN_minus_IR_gap=mean_finite(collection.ann_minimum[std_joint] - result.minimum[0, std_joint]),
            IR_allows_lower_fraction=mean_finite((result.minimum[0, std_joint] < collection.ann_minimum[std_joint]).astype(float)),
            exact_zero_fraction=float(np.mean(result.exact[0] == 0)),
            ANN_grid_infeasible_count=int((~np.isfinite(collection.ann_minimum)).sum()),
            IR_grid_infeasible_count=int((~np.isfinite(result.minimum[0])).sum()),
            mean_fixed_work=float(collection.work.mean()), mean_ANN_recall=float(collection.recall.mean()),
            retained_positive_judgments=sum(sum(g > 0 for g in q.values()) for q in judgments),
            work_scope='mean over all fixed query-budget pairs; see per-policy frontier for each fixed ef'))
        for mi, metric in enumerate(METRICS):
            full_labels, thin_labels = collection.full.labels[mi], result.labels[mi]
            common_counts = confusion_counts(collection.full.labels[0], thin_labels)
            for stratum, value, qi, ei in groups:
                full_subset = full_labels[np.ix_(qi, ei)]
                thin_subset = thin_labels[np.ix_(qi, ei)]
                counts = confusion_counts(full_subset, thin_subset)
                reference_zero = (result.exact[mi, qi] == 0)[:, None]
                false_safe = (full_subset == 0) & (thin_subset == 1)
                zero_component = int((false_safe & reference_zero).sum())
                positive_component = int((false_safe & ~reference_zero).sum())
                if (reference_zero & np.isfinite(thin_subset) & (thin_subset != 1)).any():
                    raise AssertionError('Nonnegative metric zero reference must be loss-safe')
                false_rows.append(dict(collection=collection.year, metric=metric, condition=condition, seed=seed,
                    stratum=stratum, stratum_value=value, query_count=len(qi), budget_count=len(ei),
                    false_safe_zero_reference_count=zero_component,
                    false_safe_positive_reference_count=positive_component,
                    fsr_thin_exact_zero_component=ratio(zero_component, counts['planned_pairs']),
                    fsr_thin_exact_positive_component=ratio(positive_component, counts['planned_pairs']),
                    **counts, **rates(counts)))
            full_minimum, thin_minimum = collection.full.minimum[mi], result.minimum[mi]
            difference = optimism(full_minimum, thin_minimum)
            paired_feasible = np.isfinite(difference)
            full_applicable = np.isfinite(collection.full.exact[mi])
            thin_applicable = np.isfinite(result.exact[mi])
            rho, rho_count = spearman_finite(full_minimum, thin_minimum)
            robustness_rows.append(dict(collection=collection.year, metric=metric, condition=condition, seed=seed,
                **{f'common_{k}': v for k, v in common_counts.items()},
                **{f'common_{k}': v for k, v in rates(common_counts).items()},
                query_count=len(collection.keys), full_applicable_query_count=int(full_applicable.sum()),
                thin_applicable_query_count=int(thin_applicable.sum()),
                full_exact_zero_fraction=ratio(int(((collection.full.exact[mi] == 0) & full_applicable).sum()), int(full_applicable.sum())),
                thin_exact_zero_fraction=ratio(int(((result.exact[mi] == 0) & thin_applicable).sum()), int(thin_applicable.sum())),
                full_grid_infeasible_count=int((full_applicable & ~np.isfinite(full_minimum)).sum()),
                thin_grid_infeasible_count=int((thin_applicable & ~np.isfinite(thin_minimum)).sum()),
                pairwise_feasible_queries=int(paired_feasible.sum()),
                mean_budget_optimism=mean_finite(difference), median_budget_optimism=float(np.median(difference[paired_feasible])) if paired_feasible.any() else np.nan,
                positive_optimism_fraction=mean_finite((difference[paired_feasible] > 0).astype(float)),
                negative_optimism_fraction=mean_finite((difference[paired_feasible] < 0).astype(float)),
                minimum_spearman=rho, minimum_spearman_pair_count=rho_count,
                mean_loss=mean_finite(np.maximum(0, result.exact[mi, :, None] - result.approximate[mi])),
                mean_judged_retained_exact=float(np.mean(result.judged_exact)),
                mean_judged_retained_hnsw=float(np.mean(result.judged_hnsw))))
            for qi, key in enumerate(collection.keys):
                full_found, thin_found = np.isfinite(full_minimum[qi]), np.isfinite(thin_minimum[qi])
                budget_rows.append(dict(collection=collection.year, metric=metric, condition=condition, seed=seed,
                    query_key=key, query_id=collection.raw_ids[qi], FULL_safe_budget=full_minimum[qi], thin_safe_budget=thin_minimum[qi],
                    FULL_metric_applicable=bool(full_applicable[qi]), thin_metric_applicable=bool(thin_applicable[qi]),
                    FULL_safe_found=bool(full_found), thin_safe_found=bool(thin_found),
                    feasibility_pattern=f'FULL_{int(full_found)}_THIN_{int(thin_found)}',
                    ANN_safe_budget=collection.ann_minimum[qi], budget_optimism=difference[qi],
                    headroom_inflation=difference[qi] if np.isfinite(collection.ann_minimum[qi]) else np.nan,
                    exact_FULL_score=collection.full.exact[mi, qi], exact_thin_score=result.exact[mi, qi]))
            # Per-pair evidence is compressed, preserving exact query IDs and NA.
            for qi, key in enumerate(collection.keys):
                for ei, ef in enumerate(EFS):
                    evidence.append(dict(collection=collection.year, metric=metric, condition=condition, seed=seed,
                        query_key=key, query_id=collection.raw_ids[qi], efSearch=ef,
                        FULL_safe=full_labels[qi, ei], thin_safe=thin_labels[qi, ei], standard_FULL_safe=collection.full.labels[0, qi, ei],
                        FULL_exact_score=collection.full.exact[mi, qi], thin_exact_score=result.exact[mi, qi],
                        FULL_approximate_score=collection.full.approximate[mi, qi, ei], thin_approximate_score=result.approximate[mi, qi, ei],
                        ANN_Recall_at10=collection.recall[qi, ei], distance_computations=collection.work[qi, ei]))
        # Every fixed/frozen policy is evaluated at its original choices. The
        # oracle is explicitly retrospective and changes choices with the qrels.
        full_oracle = collection.full.minimum[0]
        current_policies = dict(policies)
        oracle_choices = np.array([EFS.index(int(v)) if np.isfinite(v) else -1 for v in result.minimum[0]], int)
        oracle_work = np.array([collection.work[q, e] if e >= 0 else np.nan for q, e in enumerate(oracle_choices)])
        oracle_time = np.array([collection.latency[q, e] if e >= 0 else np.nan for q, e in enumerate(oracle_choices)])
        current_policies['Oracle IR-safe (retrospective)'] = (oracle_choices, oracle_work, oracle_time, 'thinned-qrel first-safe oracle; actual single-search work; no prospective policy')
        for name, (choices, charged_work, charged_latency, scope) in current_policies.items():
            full_selected = np.array([collection.full.labels[0, q, e] if e >= 0 else np.nan for q, e in enumerate(choices)])
            thin_selected = np.array([result.labels[0, q, e] if e >= 0 else np.nan for q, e in enumerate(choices)])
            counts = confusion_counts(full_selected, thin_selected)
            valid = choices >= 0
            chosen_efs = np.array([EFS[e] if e >= 0 else np.nan for e in choices])
            changed = int(np.sum(chosen_efs[np.isfinite(chosen_efs) & np.isfinite(full_oracle)] != full_oracle[np.isfinite(chosen_efs) & np.isfinite(full_oracle)])) if name.startswith('Oracle') else 0
            policy_rows.append(dict(collection=collection.year, policy=name, condition=condition, seed=seed,
                **counts, **rates(counts), mean_work=mean_finite(charged_work), p95_work=float(np.nanquantile(charged_work, .95)) if valid.any() else np.nan,
                mean_latency_ms=mean_finite(charged_latency), mean_selected_ef=mean_finite(chosen_efs),
                FULL_violation_count=int((full_selected == 0).sum()), thin_violation_count=int((thin_selected == 0).sum()),
                FULL_violation_rate=ratio(int((full_selected == 0).sum()), int(valid.sum())),
                thin_violation_rate=ratio(int((thin_selected == 0).sum()), int(valid.sum())),
                missing_decisions=int((~valid).sum()), changed_decision_count=changed,
                prospective=False if name.startswith('Oracle') else True, work_scope=scope,
                latency_scope='saved search-component means; excludes feature/model overhead'))
    return dict(false=pd.DataFrame(false_rows), budgets=pd.DataFrame(budget_rows), robustness=pd.DataFrame(robustness_rows),
                policies=pd.DataFrame(policy_rows), density=pd.DataFrame(density_rows), evidence=pd.DataFrame(evidence),
                labels={k: np.asarray(v) for k, v in labels_by_condition.items()},
                optimism={k: np.asarray(v) for k, v in optimism_by_condition.items()},
                cutpoints=cutpoints, policy_sources=policy_sources)


def percentile(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return tuple(np.quantile(values, [.025, .975])) if len(values) else (np.nan, np.nan)


def bootstrap_rows(frame, grouping, resamples, *, prefixes=('',), numeric=()):
    """Aggregate paired-seed rows and ratio-of-count rates, with seed CIs."""
    rows = []
    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    weight_cache = {}
    for key, group in frame.groupby(grouping, sort=False, dropna=False):
        group = group.sort_values('seed')
        n = len(group)
        if n not in weight_cache:
            weight_cache[n] = rng.multinomial(n, np.full(n, 1 / n), size=resamples).astype(float) / n
        weights = weight_cache[n]
        row = dict(zip(grouping, key if isinstance(key, tuple) else (key,)))
        row.update(seed_count=n, inference='paired thinning-seed bootstrap conditional on fixed collection; pointwise 95% CI')
        for prefix in prefixes:
            columns = [prefix + c for c in COUNT_COLUMNS]
            values = group[columns].to_numpy(float)
            totals = dict(zip(COUNT_COLUMNS, values.sum(axis=0)))
            means = dict(zip(COUNT_COLUMNS, values.mean(axis=0)))
            row.update({prefix + k + '_total': int(v) for k, v in totals.items()})
            row.update({prefix + k + '_per_seed': float(v) for k, v in means.items()})
            row.update({prefix + k: v for k, v in rates(totals).items()})
            boot = weights @ values
            ci_ratios = [('fsr_all', 'false_safe', 'planned_pairs'), ('fsr_valid', 'false_safe', 'valid_pairs'),
                         ('fsr_given_thin_safe', 'false_safe', 'thin_safe'),
                         ('full_unsafe_sensitivity', 'both_unsafe', 'full_unsafe'),
                         ('full_unsafe_sensitivity_all_reference_pairs', 'both_unsafe', 'full_unsafe_planned')]
            for endpoint, numerator, denominator in ci_ratios:
                a, b = boot[:, COUNT_COLUMNS.index(numerator)], boot[:, COUNT_COLUMNS.index(denominator)]
                samples = np.divide(a, b, out=np.full(resamples, np.nan), where=b > 0)
                if endpoint == 'fsr_all' and totals['missing_pairs']:
                    samples[:] = np.nan
                low, high = percentile(samples)
                row.update({prefix + endpoint + '_ci_low': low, prefix + endpoint + '_ci_high': high})
        for column in numeric:
            values = group[column].to_numpy(float)
            valid = np.isfinite(values).astype(float)
            numerator = weights @ np.nan_to_num(values, nan=0)
            denominator = weights @ valid
            samples = np.divide(numerator, denominator, out=np.full(resamples, np.nan), where=denominator > 0)
            low, high = percentile(samples)
            row.update({column: mean_finite(values), column + '_ci_low': low, column + '_ci_high': high,
                        column + '_finite_seed_count': int(valid.sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def crossed_bootstrap(collection, results, resamples):
    """Crossed query/seed resampling preserves every within-query budget cell."""
    rng = np.random.Generator(np.random.PCG64(RNG_SEED + int(collection.year)))
    rows = []
    for condition in CONDITIONS[1:]:
        labels = results['labels'][condition] # seed, metric, query, ef
        s, _, q, _ = labels.shape
        seed_weights = rng.multinomial(s, np.full(s, 1 / s), size=resamples).astype(float)
        query_weights = rng.multinomial(q, np.full(q, 1 / q), size=resamples).astype(float)
        for mi, metric in enumerate(METRICS):
            full = collection.full.labels[mi]
            thin = labels[:, mi]
            valid = np.isfinite(full)[None] & np.isfinite(thin)
            false = ((full[None] == 0) & (thin == 1) & valid).sum(axis=-1).astype(float)
            thin_safe = ((thin == 1) & valid).sum(axis=-1).astype(float)
            false_boot = np.einsum('rs,sq,rq->r', seed_weights, false, query_weights, optimize=True)
            safe_boot = np.einsum('rs,sq,rq->r', seed_weights, thin_safe, query_weights, optimize=True)
            for endpoint, samples in [('fsr_all', false_boot / (s * q * len(EFS))),
                                      ('fsr_given_thin_safe', np.divide(false_boot, safe_boot, out=np.full(resamples, np.nan), where=safe_boot > 0))]:
                if endpoint == 'fsr_all' and not valid.all():
                    samples[:] = np.nan
                low, high = percentile(samples)
                rows.append(dict(collection=collection.year, metric=metric, condition=condition, endpoint=endpoint,
                    ci_low=low, ci_high=high, inference='crossed query-and-seed bootstrap sensitivity; conditional on fixed rankings',
                    query_count=q, seed_count=s, bootstrap_resamples=resamples))
    return pd.DataFrame(rows)


def paired_tests(robustness, false):
    """Signed metric differences are diagnostic, never FSR-vs-structural-zero."""
    rng = np.random.Generator(np.random.PCG64(RNG_SEED + 2))
    rows = []
    overall = false[(false.stratum == 'overall') & (false.condition != 'FULL')]
    for collection, group in robustness[robustness.condition != 'FULL'].groupby('collection', sort=False):
        for metric in METRICS:
            for condition in CONDITIONS[1:]:
                difference = group[(group.metric == metric) & (group.condition == condition)].sort_values('seed').mean_budget_optimism.to_numpy(float)
                estimate, interval, pvalue = diagnostic_difference(difference, rng)
                rows.append(dict(collection=collection, metric=metric, condition=condition, endpoint='mean_budget_optimism',
                    hypothesis_family=f'{collection}:{metric}:budget_optimism:three_density_comparisons',
                    estimate=estimate, ci_low=interval[0], ci_high=interval[1], raw_p=pvalue,
                    test='paired seed sign-flip diagnostic; symmetry/exchangeability assumption',
                    comparison='FULL minus THIN first-safe ef; pairwise feasible subset; signed differences',
                    seed_count=len(difference), signflip_resamples=20000, bootstrap_resamples=20000))
        for metric in METRICS[1:]:
            for endpoint in ('fsr_all', 'common_fsr_all'):
                for condition in CONDITIONS[1:]:
                    if endpoint == 'fsr_all':
                        a = overall[(overall.collection == collection) & (overall.metric == metric) & (overall.condition == condition)].sort_values('seed')[endpoint].to_numpy(float)
                        b = overall[(overall.collection == collection) & (overall.metric == METRICS[0]) & (overall.condition == condition)].sort_values('seed')[endpoint].to_numpy(float)
                    else:
                        a = group[(group.metric == metric) & (group.condition == condition)].sort_values('seed')[endpoint].to_numpy(float)
                        b = group[(group.metric == METRICS[0]) & (group.condition == condition)].sort_values('seed')[endpoint].to_numpy(float)
                    if len(a) != len(b):
                        raise ValueError('Unpaired metric seed outcomes')
                    difference = a - b
                    estimate, interval, pvalue = diagnostic_difference(difference, rng)
                    rows.append(dict(collection=collection, metric=metric, condition=condition, endpoint=endpoint,
                        hypothesis_family=f'{collection}:{metric}:{endpoint}:three_density_comparisons',
                        estimate=estimate, ci_low=interval[0], ci_high=interval[1], raw_p=pvalue,
                        test='paired seed sign-flip diagnostic; symmetry/exchangeability assumption',
                        comparison='alternative minus standard nDCG; identical numeric tolerance is not utility equivalence',
                        seed_count=len(a), signflip_resamples=20000, bootstrap_resamples=20000))
    frame = pd.DataFrame(rows)
    if len(frame):
        frame['holm_p'] = np.nan
        for _, group in frame.groupby('hypothesis_family', sort=False):
            frame.loc[group.index, 'holm_p'] = holm(group.raw_p)
            frame.loc[group.index, 'family_tests_performed'] = int(np.isfinite(group.raw_p).sum())
    return frame


def diagnostic_difference(difference, rng):
    if not len(difference) or not np.isfinite(difference).all():
        return np.nan, (np.nan, np.nan), np.nan
    weights = rng.multinomial(len(difference), np.full(len(difference), 1 / len(difference)), size=20000) / len(difference)
    interval = percentile(weights @ difference)
    signs = rng.choice([-1, 1], size=(20000, len(difference)))
    null = (signs * difference).mean(axis=1)
    estimate = float(difference.mean())
    pvalue = (1 + int(np.sum(np.abs(null) >= abs(estimate) - 1e-15))) / 20001
    return estimate, interval, pvalue


def write_csv(frame, name):
    target = OUT / name
    if target.suffix == '.gz':
        frame.to_csv(target, index=False, compression={'method': 'gzip', 'mtime': 0})
    else:
        frame.to_csv(target, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collections', nargs='+', choices=['2019', '2020'], default=['2019'])
    parser.add_argument('--seeds', type=int, default=50)
    parser.add_argument('--bootstrap-resamples', type=int, default=20000)
    args = parser.parse_args()
    if args.seeds < 50 or args.bootstrap_resamples < 2000:
        parser.error('Require >=50 thinning seeds and >=2000 bootstrap resamples')
    if len(set(args.collections)) != len(args.collections):
        parser.error('Duplicate collections')
    protocol_path, audit_path = OUT / 'METRIC_PROTOCOL.md', OUT / 'NOVELTY_AUDIT.md'
    if not protocol_path.exists() or not audit_path.exists():
        raise FileNotFoundError('Phase 0 audit and metric protocol must exist before outcomes')
    protocol_sha = sha(protocol_path)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [protocol_path, audit_path, Path(__file__), OUT / 'false_safety_metrics.py']}
    collected, validations, cutpoints, policy_sources, crossed = [], {}, {}, {}, []
    for year in args.collections:
        collection = load_collection(year)
        for path in collection.source_paths:
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
        print(f'{year}: FULL gate PASS; {len(collection.keys)} queries. Evaluating frozen rankings.', flush=True)
        result = per_seed(collection, args.seeds)
        for source in result['policy_sources']:
            if source.get('status') == 'AVAILABLE':
                for field in ['original_path', 'portable_input']:
                    relative = source[field]
                    if (ROOT / relative).exists():
                        source_hashes[relative] = source['sha256']
        if year == '2019':
            original_protocol = BASE / 'score_threshold_baseline/protocol.json'
            if original_protocol.exists():
                source_hashes[str(original_protocol.relative_to(ROOT))] = sha(original_protocol)
                source_hashes['full_paper_extension/data/score_gap_protocol.json'] = sha(OUT / 'data/score_gap_protocol.json')
        archived_path = ROOT / 'sigir_analysis/trec_judgment_density_per_seed.csv' if year == '2019' else OUT / 'trec2020_judgment_density_per_seed.csv'
        if archived_path.exists():
            archived = pd.read_csv(archived_path)
            archived = archived[['condition', 'seed', 'mean_ANN_minus_IR_budget_gap']]
            reproduced = result['density'].merge(archived, on=['condition', 'seed'], validate='one_to_one')
            if len(reproduced) != 1 + 3 * args.seeds:
                raise ValueError('Density reproduction dropped seed conditions')
            gap_error = float(np.max(np.abs(reproduced.ANN_minus_IR_gap - reproduced.mean_ANN_minus_IR_budget_gap)))
            if gap_error > 1e-9:
                raise ValueError(f'{year} saved density gap reproduction failed: {gap_error}')
            collection.validation['density_mean_gap_max_error'] = gap_error
        collected.append(result)
        validations[year] = collection.validation
        cutpoints[year] = result['cutpoints']
        policy_sources[year] = result['policy_sources']
        crossed.append(crossed_bootstrap(collection, result, args.bootstrap_resamples))
    if sha(protocol_path) != protocol_sha:
        raise RuntimeError('Metric protocol changed during outcome execution')
    frames = {key: pd.concat([r[key] for r in collected], ignore_index=True) for key in ['false', 'budgets', 'robustness', 'policies', 'density', 'evidence']}
    fsummary = bootstrap_rows(frames['false'], ['collection', 'metric', 'condition', 'stratum', 'stratum_value'], args.bootstrap_resamples,
        numeric=['false_safe_zero_reference_count', 'false_safe_positive_reference_count', 'fsr_thin_exact_zero_component', 'fsr_thin_exact_positive_component'])
    robustness_numeric = ['query_count', 'full_applicable_query_count', 'thin_applicable_query_count', 'full_exact_zero_fraction', 'thin_exact_zero_fraction',
        'full_grid_infeasible_count', 'thin_grid_infeasible_count', 'pairwise_feasible_queries', 'mean_budget_optimism', 'median_budget_optimism',
        'positive_optimism_fraction', 'negative_optimism_fraction', 'minimum_spearman', 'minimum_spearman_pair_count', 'mean_loss', 'mean_judged_retained_exact', 'mean_judged_retained_hnsw']
    robustness = bootstrap_rows(frames['robustness'], ['collection', 'metric', 'condition'], args.bootstrap_resamples,
                                prefixes=('common_',), numeric=robustness_numeric)
    overall = fsummary[fsummary.stratum == 'overall'].drop(columns=['stratum', 'stratum_value'])
    robustness = robustness.merge(overall, on=['collection', 'metric', 'condition'], validate='one_to_one', suffixes=('', '_own_metric'))
    robustness['metric_scope'] = np.where(robustness.metric == METRICS[0], 'historical exponential-gain standard nDCG@10', 'submitted returned10 only; no deeper ANN ranks inferred')
    robustness['threshold_comparability'] = 'same 0.025 numeric tolerance; not equivalent utility scales; inspect common-reference sensitivity and applicability'
    policy_numeric = ['mean_work', 'p95_work', 'mean_latency_ms', 'mean_selected_ef', 'FULL_violation_count', 'thin_violation_count', 'FULL_violation_rate', 'thin_violation_rate', 'missing_decisions', 'changed_decision_count']
    policies = bootstrap_rows(frames['policies'], ['collection', 'policy', 'condition'], args.bootstrap_resamples, numeric=policy_numeric)
    scope = frames['policies'].drop_duplicates(['collection', 'policy'])[['collection', 'policy', 'work_scope', 'latency_scope', 'prospective']]
    policies = policies.merge(scope, on=['collection', 'policy'], validate='many_to_one')
    density_numeric = [c for c in frames['density'].columns if c not in ['collection', 'condition', 'seed', 'work_scope']]
    density = bootstrap_rows(frames['density'], ['collection', 'condition'], args.bootstrap_resamples, prefixes=(), numeric=density_numeric)
    tests = paired_tests(frames['robustness'], frames['false'])
    for key, name in [('false', 'false_safety_per_seed.csv'), ('budgets', 'budget_optimism_analysis.csv'),
                      ('robustness', 'metric_robustness_per_seed.csv'), ('policies', 'policy_false_safety_per_seed.csv'),
                      ('density', 'collection_density_per_seed.csv'), ('evidence', 'false_safety_pair_evidence.csv.gz')]:
        write_csv(frames[key], name)
    for frame, name in [(fsummary, 'false_safety_analysis.csv'), (robustness, 'metric_robustness_under_thinning.csv'),
                        (policies, 'policy_false_safety_analysis.csv'), (density, 'collection_density_summary.csv'),
                        (pd.concat(crossed, ignore_index=True), 'crossed_query_seed_bootstrap.csv'), (tests, 'stats_summary.csv')]:
        write_csv(frame, name)
    budget_summary = robustness[['collection', 'metric', 'condition', 'query_count', 'pairwise_feasible_queries', 'mean_budget_optimism',
        'mean_budget_optimism_ci_low', 'mean_budget_optimism_ci_high', 'median_budget_optimism', 'positive_optimism_fraction', 'negative_optimism_fraction',
        'full_grid_infeasible_count', 'thin_grid_infeasible_count', 'full_applicable_query_count', 'thin_applicable_query_count']]
    write_csv(budget_summary, 'budget_optimism_summary.csv')
    patterns = frames['budgets'].groupby(['collection', 'metric', 'condition', 'feasibility_pattern'], sort=False).agg(
        query_seed_count=('query_key', 'size'), FULL_applicable_count=('FULL_metric_applicable', 'sum'),
        thin_applicable_count=('thin_metric_applicable', 'sum')).reset_index()
    write_csv(patterns, 'budget_feasibility_patterns.csv')
    # Recheck every historical input, including local policy files, before PASS.
    for relative, expected in source_hashes.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f'Input changed during analysis: {relative}')
    try:
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_head = 'UNAVAILABLE'
    manifest = dict(command_line=' '.join(sys.argv), git_head_at_analysis=git_head, collections=args.collections,
        thinning_seeds=list(range(args.seeds)), bootstrap_resamples=args.bootstrap_resamples, resampling_seed=RNG_SEED,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, input_sha256=source_hashes,
        source_policy_lineage=policy_sources, full_nDCG_quartile_cutpoints=cutpoints,
        protocol_frozen_before_new_outcomes=True, protocol_sha256=protocol_sha,
        metric_loss_tolerance=TOLERANCE, bpref_relevance_threshold=2,
        metric_scope='standard nDCG@10 plus returned-depth10 condensed nDCG and bpref; not deeper-run evaluation',
        no_retrieval_training_or_historical_writes=True)
    (OUT / 'false_safety_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    validation = dict(status='PASS', collection_validation=validations, source_inputs_unchanged=True,
        all_queries_retained=True, rankings_work_recall_fixed=True, no_maxgrid_safety_imputation=True,
        undefined_bpref_visible=True, protocol_sha256=protocol_sha,
        output_rows={key: len(frame) for key, frame in frames.items()},
        interpretation='FULL disagreements, not true relevance errors; alternative metrics have restricted output scope')
    (OUT / 'false_safety_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'collections': args.collections, 'output_rows': validation['output_rows']}), flush=True)


if __name__ == '__main__':
    main()
