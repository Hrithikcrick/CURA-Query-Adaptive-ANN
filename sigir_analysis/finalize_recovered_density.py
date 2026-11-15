"""Promote verified density results and complete only new density endpoints."""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'sigir_analysis'
EVIDENCE = OUT / 'restored_ranking_evaluation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_checks():
    protocol = json.loads((OUT/'analysis_protocol.json').read_text())
    for name, digest in protocol['input_hashes'].items():
        assert sha(ROOT/name) == digest, name
    preserved = json.loads((OUT/'preserved_analysis_hashes.json').read_text())
    for name, digest in preserved.items():
        assert sha(OUT/name) == digest, name
    return preserved


def promote():
    preserved = frozen_checks()
    validation = json.loads((EVIDENCE/'validation.json').read_text())
    assert validation['overall_analysis_complete'] and validation['ranking_seeds_evaluated'] == 50
    old = pd.read_csv(OUT/'trec_judgment_density_per_seed.csv')
    new = pd.read_csv(EVIDENCE/'trec_judgment_density_per_seed.csv')
    assert len(old) == len(new) == 151
    keys = ['condition', 'seed']
    comparison = old.merge(new, on=keys, validate='one_to_one', suffixes=('_prepared','_scored'))
    assert (comparison.qrels_sha256_prepared == comparison.qrels_sha256_scored).all()
    assert new.evaluation_completed.all()
    per_query = pd.read_csv(EVIDENCE/'trec_judgment_density_per_query.csv')
    assert len(per_query) == 64930 and per_query.ranking_sha256.nunique() == 1
    assert not per_query.duplicated(['condition','seed','query_key','efSearch']).any()
    assert (per_query.groupby(['query_key','efSearch']).ANN_Recall_at_10.nunique() == 1).all()
    assert (per_query.groupby('query_key').min_ANN_safe_ef.nunique() == 1).all()
    front = pd.read_csv(EVIDENCE/'trec_fixed_risk_work_by_density.csv')
    for col in ['ANN_Recall_at_10','mean_distance_computations','mean_latency_ms','p95_latency_ms']:
        assert (front.groupby('efSearch')[col].nunique() == 1).all()
    files = ['trec_judgment_density_per_query.csv','trec_judgment_density_per_seed.csv',
             'trec_judgment_density_summary.csv','trec_fixed_risk_work_by_density.csv','trec_cura_by_density.csv',
             'judgment_density_vs_budget_gap.png','judgment_density_vs_zero_ndcg.png',
             'judgment_density_vs_ir_safe_budget.png','judgment_density_vs_risk_work.png']
    for name in files:
        (OUT/name).write_bytes((EVIDENCE/name).read_bytes())
    (OUT/'trec_judgment_density_frontier.csv').write_bytes((EVIDENCE/'trec_fixed_risk_work_by_density.csv').read_bytes())
    result = dict(status='PASS', prepared_seed_hashes_matched=151, stochastic_seeds_per_condition=50,
                  query_seed_condition_budget_rows=64930, rankings_constant=True,
                  ANN_recall_invariant=True, ANN_safe_budget_invariant=True, work_and_timing_invariant=True,
                  frozen_analysis_hashes=preserved, original_analyses_recomputed=False,
                  promoted_hashes={name:sha(OUT/name) for name in files},
                  frontier_sha256=sha(OUT/'trec_judgment_density_frontier.csv'))
    (OUT/'density_promotion_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def complete_statistics():
    frozen_checks()
    sys.path.insert(0, str(OUT))
    from density_uncertainty import _percentile_mean_ci, _signflip
    source_path = OUT/'statistical_tests.csv'
    with source_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); original_columns = reader.fieldnames; old = list(reader)
    if any(r.get('primary_test_id') == 'P7' for r in old):
        raise ValueError('New density statistics already exist; do not recompute them.')
    protected = [r for r in old if r.get('primary_test_id') not in {'P5','P6'}]
    protected_sha = hashlib.sha256(json.dumps(protected,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with (EVIDENCE/'density_statistical_tests.csv').open(newline='',encoding='utf-8-sig') as stream:
        density = list(csv.DictReader(stream))
    assert len(density) == 6 and all(r['test_status']=='COMPLETED_DENSITY_SEED_ANALYSIS' for r in density)
    seeds = pd.read_csv(EVIDENCE/'trec_judgment_density_per_seed.csv')
    assert (seeds.jointly_feasible_query_count == 43).all()
    full = seeds[seeds.condition=='FULL'].iloc[0]
    for i, condition in enumerate(['50_PERCENT_POSITIVES','25_PERCENT_POSITIVES','ONE_POSITIVE_PER_QUERY']):
        frame = seeds[seeds.condition==condition].sort_values('seed')
        assert len(frame) == 50
        p5 = next(r for r in density if r['primary_test_id']=='P5' and r['comparison']==condition+' minus FULL')
        ir_effect = frame.mean_min_IR_safe_ef.to_numpy()-full.mean_min_IR_safe_ef
        gap_effect = frame.mean_ANN_minus_IR_budget_gap.to_numpy()-full.mean_ANN_minus_IR_budget_gap
        assert np.allclose(ir_effect, -gap_effect, atol=1e-12, rtol=0)
        derived = dict(p5)
        derived.update(primary_test_id='P7', metric='thinned minus FULL IR-safe budget', units='efSearch',
                       seed_matched_FULL_gap_mean='', paired_gap_query_count_mean='', paired_gap_query_count_min='', paired_gap_query_count_max='',
                       mean_difference=repr(-float(p5['mean_difference'])),
                       ci95_low=repr(-float(p5['ci95_high'])), ci95_high=repr(-float(p5['ci95_low'])),
                       estimator='exact sign reversal of P5 on the same 43 jointly feasible queries',
                       test='Paired sign-flip result inherited from exact P5 sign reversal',
                       effect_direction='thinned minus FULL', FULL_reference_value=repr(float(full.mean_min_IR_safe_ef)),
                       FULL_global_reference_value=repr(float(full.mean_min_IR_safe_ef)),
                       interpretation='ANN-safe budgets and the common 43-query feasible cohort are invariant; each IR-safe budget effect is exactly the negative gap effect. P5 bootstrap draws and significance are reused rather than recomputed. This endpoint is algebraically redundant, not independent confirming evidence.')
        density.append(derived)
        effect = frame.IR_allows_lower_rate.to_numpy()-full.IR_allows_lower_rate
        random_seed = 20261116+i*2
        low, high = _percentile_mean_ci(effect,20000,random_seed)
        p, test, nonzero, draws = _signflip(effect,20000,random_seed+1)
        extra = dict(p5)
        extra.update(primary_test_id='P8',metric='thinned minus FULL IR-allows-lower rate',units='fraction of queries',
                     seed_matched_FULL_gap_mean='', paired_gap_query_count_mean='', paired_gap_query_count_min='', paired_gap_query_count_max='',
                     mean_difference=repr(float(effect.mean())),ci95_low=repr(low),ci95_high=repr(high),
                     p_value=repr(p),nonzero_pairs=str(nonzero),test=test,
                     estimator='mean paired seed-level rate difference on 43 fixed queries',
                     random_seed=str(random_seed),signflip_random_seed=str(random_seed+1),
                     signflip_resamples=str(draws),bootstrap_resamples='20000',
                     FULL_reference_value=repr(float(full.IR_allows_lower_rate)),
                     FULL_global_reference_value=repr(float(full.IR_allows_lower_rate)),
                     interpretation='Intervals and exploratory paired sign-flip tests resample thinning seeds, conditional on these 43 queries and fixed rankings. Query-by-seed cells are not independent observations; no query-population or deployment-risk guarantee is inferred.')
        density.append(extra)
    assert len(density) == 12
    # Holm adjustment is valid despite correlated/redundant endpoints.
    order = sorted(range(12),key=lambda i:float(density[i]['p_value']))
    previous = 0.0
    for j,i in enumerate(order):
        previous = min(1.0,max(previous,(12-j)*float(density[i]['p_value'])))
        density[i]['holm_p_value_12_density_comparisons'] = repr(previous)
        density[i]['multiplicity'] = '12 density comparisons; Holm-adjusted p-value supplied; endpoints are dependent'
    merged = protected+density
    columns = list(dict.fromkeys([*original_columns,*(key for r in density for key in r)]))
    with source_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(merged)
    with source_path.open(newline='',encoding='utf-8') as stream:
        reread=list(csv.DictReader(stream))
    restored_protected=[{k:r[k] for k in original_columns} for r in reread if r.get('primary_test_id') not in {'P5','P6','P7','P8'}]
    assert restored_protected == protected
    with (OUT/'density_statistical_tests_completed.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(density)
    result=dict(status='PASS', density_comparisons_completed=12, preserved_completed_test_rows=len(protected),
                preserved_completed_rows_sha256=protected_sha, P1_P4_recomputed=False,
                P5_P6_recomputed=False, P7_derived_from_exact_sign_reversal=True,
                P8_new_paired_seed_tests=3, thinning_seeds=50, bootstrap_resamples=20000,
                statistics_sha256=sha(source_path), all_new_tests_completed=True)
    (OUT/'density_statistics_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    frozen_checks()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--promote',action='store_true');parser.add_argument('--statistics',action='store_true')
    args=parser.parse_args()
    if args.promote: print(json.dumps(promote(),indent=2))
    if args.statistics: print(json.dumps(complete_statistics(),indent=2))
