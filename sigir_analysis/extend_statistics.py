"""Complete available primary comparisons and expose unavailable density tests."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import analyze_saved_artifacts as analysis

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
BASE=ROOT/'msmarco_large/msmarco_1000000_q1000'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    saved=OUT/'statistical_tests_saved.csv'
    preserved=json.loads((OUT/'preserved_analysis_hashes.json').read_text())
    if not saved.exists():
        current=OUT/'statistical_tests.csv'
        assert sha(current)==preserved['statistical_tests.csv'],'Unexpected prior statistical file; refusing to replace preservation evidence'
        saved.write_bytes(current.read_bytes())
    if 'statistical_tests.csv' in preserved:
        assert sha(saved)==preserved.pop('statistical_tests.csv')
        preserved['statistical_tests_saved.csv']=sha(saved)
        (OUT/'preserved_analysis_hashes.json').write_text(json.dumps(preserved,indent=2))
    base=pd.read_csv(saved)
    base['test_status']='COMPLETED_SAVED_DATA'
    base['analysis_role']='preserved exploratory comparison'
    oracle=pd.read_csv(OUT/'oracle_budget_gap_per_query.csv')
    summary=pd.read_csv(OUT/'oracle_budget_gap_analysis.csv')
    primary=[]
    seed=analysis.SEED+500
    # Tests on both the legacy capped statistic and the joint-feasibility
    # sensitivity preserve censoring instead of calling a failed ef256 safe.
    for pid,positive in [('P1',False),('P2',True)]:
        frame=oracle[oracle.source=='msmarco-dev-small']
        if positive:frame=frame[frame['exact_nDCG@10']>0]
        group='exact_positive' if positive else 'all'
        for mode in ['saved_grid_censored_fallback','both_criteria_feasible']:
            g=frame if mode=='saved_grid_censored_fallback' else frame[(frame.ANN_safe_found==1)&(frame.IR_safe_found==1)]
            row=summary[(summary.cohort=='MSMARCO_ALL')&(summary.exact_group==group)&(summary.estimator==mode)].iloc[0]
            difference=g.min_ANN_safe_ef-g.min_IR_safe_ef
            p,test,nonzero=analysis.signflip(difference,seed)
            primary.append(dict(primary_test_id=pid,cohort='MSMARCO_ALL '+group,comparison='ANN-safe minus IR-safe',metric='oracle efSearch gap',units='efSearch',estimator=mode,queries=len(g),nonzero_pairs=nonzero,mean_difference=difference.mean(),ci95_low=row.gap_ci95_low,ci95_high=row.gap_ci95_high,p_value=p,test=test,test_status='COMPLETED_SAVED_DATA',analysis_role='requested primary comparison',random_seed=seed,bootstrap_resamples=analysis.B,relative_reduction_pct=row.relative_oracle_budget_reduction_pct,relative_reduction_ci95_low_pct=row.reduction_ci95_low_pct,relative_reduction_ci95_high_pct=row.reduction_ci95_high_pct,IR_allows_lower_rate=row.IR_allows_lower_rate,interpretation='Paired-query percentile CI; exploratory sign-flip assumes sign exchangeability; capped ef256 is not a verified safe threshold' if mode=='saved_grid_censored_fallback' else 'Conditional on both criteria feasible; not an estimate of thresholds beyond the tested grid'))
            seed+=1
    fixed=pd.read_csv(BASE/'cura_controller/selected_fixed_ir_msmarco_test.csv')
    cura=pd.read_csv(BASE/'cura_controller/selected_cura_full_msmarco_test.csv')
    paired=fixed[['query_key','effective_distance_computations','IR_unsafe']].merge(cura[['query_key','effective_distance_computations','IR_unsafe']],on='query_key',suffixes=('_fixed','_cura'),validate='one_to_one')
    assert len(paired)==200
    matched=pd.read_csv(OUT/'risk_matched_fixed_per_query.csv')
    for pid,left,right,comparison,bootstrap_seed in [
        ('P3',paired.effective_distance_computations_fixed,paired.effective_distance_computations_cura,'original fixed ef96 minus CURA-Full',analysis.SEED+510),
        ('P4',matched.effective_distance_computations,matched.distance_computations,'CURA-Full minus post-hoc fixed ef48',analysis.SEED+80)]:
        estimate=analysis.paired_boot(left,right,bootstrap_seed)
        p,test,nonzero=analysis.signflip(np.asarray(left)-np.asarray(right),bootstrap_seed+1)
        primary.append(dict(primary_test_id=pid,cohort='MSMARCO_TEST',comparison=comparison,metric='effective distance computations',units='distance computations/query',estimator='paired mean difference',queries=200,nonzero_pairs=nonzero,mean_difference=estimate['difference'],ci95_low=estimate['difference_ci'][0],ci95_high=estimate['difference_ci'][1],p_value=p,test=test,test_status='COMPLETED_SAVED_DATA',analysis_role='requested primary comparison',random_seed=bootstrap_seed,bootstrap_resamples=analysis.B,relative_reduction_pct=estimate['saving_pct'],relative_reduction_ci95_low_pct=estimate['saving_pct_ci'][0],relative_reduction_ci95_high_pct=estimate['saving_pct_ci'][1],interpretation='Saving is 1 - mean(right)/mean(left); paired query CI; frozen policy; sign-flip assumes sign exchangeability'+('; conditional on post-hoc ef48 choice, excluding comparator-selection uncertainty' if pid=='P4' else '; original ef96 selected using calibration only')))
    for name,left,right in [('original ef96',paired.IR_unsafe_fixed,paired.IR_unsafe_cura),('post-hoc ef48',matched.fixed_unsafe,matched.cura_unsafe)]:
        delta=np.asarray(right)-np.asarray(left)
        rng=np.random.default_rng(seed)
        ci=analysis.ci(delta[rng.integers(0,len(delta),(analysis.B,len(delta)))].mean(axis=1))
        cura_only=int(((np.asarray(right)==1)&(np.asarray(left)==0)).sum())
        fixed_only=int(((np.asarray(left)==1)&(np.asarray(right)==0)).sum())
        discordant=cura_only+fixed_only
        primary.append(dict(primary_test_id='SUPPLEMENT_VIOLATIONS',cohort='MSMARCO_TEST',comparison='CURA-Full minus '+name,metric='IR violation indicator',units='fraction of queries',queries=200,nonzero_pairs=discordant,mean_difference=float(delta.mean()),ci95_low=ci[0],ci95_high=ci[1],p_value=binomtest(cura_only,discordant,.5).pvalue if discordant else 1.,test='exact McNemar with paired-query bootstrap effect CI',test_status='COMPLETED_SAVED_DATA',analysis_role='paired risk uncertainty',cura_only_failures=cura_only,fixed_only_failures=fixed_only,interpretation='Few discordances; does not establish risk equivalence'))
        seed+=1
    for pid,metric,units in [('P5','thinned minus FULL ANN-minus-IR budget gap','efSearch'),('P6','thinned minus FULL exact-zero fraction','fraction of queries')]:
        for condition in ['50_PERCENT_POSITIVES','25_PERCENT_POSITIVES','ONE_POSITIVE_PER_QUERY']:
            primary.append(dict(primary_test_id=pid,cohort='TREC_DL_2019',comparison=condition+' minus FULL',metric=metric,units=units,queries=43,thinning_seeds_prepared=50,thinning_seeds_evaluated=0,test_status='BLOCKED_MISSING_RANKINGS',test='NOT_RUN',analysis_role='requested primary comparison',interpretation='Original exact and per-ef HNSW ranked document IDs are missing; no effect, CI, p-value or density outcome is estimable'))
    results=pd.DataFrame(primary)
    results['multiplicity']='Exploratory, unadjusted; blocked tests have no p-value'
    results.to_csv(OUT/'primary_statistical_results.csv',index=False)
    combined=pd.concat([base,results],ignore_index=True,sort=False)
    combined.to_csv(OUT/'statistical_tests.csv',index=False)
    protocol=dict(saved_data_primary_tests_completed=['P1','P2','P3','P4'],density_primary_tests_completed=[],blocked_primary_tests=['P5','P6'],statistical_analysis_complete=False,bootstrap='20000 paired-query percentile resamples; original method choices frozen',signflip='two-sided, 20000 Monte Carlo draws or exact enumeration for <=18 nonzero pairs; sign exchangeability assumed',density_seed_analysis='NOT RUN: zero scored seeds; prepared seeds are not completed experiments',censoring='Both legacy grid-capped and jointly feasible paired oracle estimators reported',multiplicity='Exploratory and unadjusted',source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in [saved,OUT/'oracle_budget_gap_per_query.csv',OUT/'oracle_budget_gap_analysis.csv',BASE/'cura_controller/selected_fixed_ir_msmarco_test.csv',BASE/'cura_controller/selected_cura_full_msmarco_test.csv',OUT/'risk_matched_fixed_per_query.csv']})
    (OUT/'statistical_analysis_status.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (OUT/'statistical_analysis_summary.md').write_text('''# Statistical analysis scope

P1 (all 1,000 MS MARCO oracle pairs), P2 (exact-positive oracle pairs), P3 (CURA versus original calibrated ef96 on 200 test queries), and P4 (CURA versus post-hoc matched ef48) are complete from saved paired data. Both grid-capped and jointly feasible oracle summaries are retained. Source hashes, bootstrap seeds, effect directions, units, effect sizes and CIs are stored in primary_statistical_results.csv and statistical_analysis_status.json.

P5 and P6 (FULL versus each thinned TREC condition: oracle gap and exact-zero fraction) are blocked, with blank effect/CI/p-value fields. Original qrels and prepared seeds cannot replace missing rankings. No outcome A/B/C or seed-level significance claim can be made.

All p-values are exploratory and unadjusted. Paired sign-flip tests assume sign exchangeability; they do not reflect randomized assignment of algorithms. Bootstrap CIs preserve query pairing and condition on the frozen policy; post-hoc comparator intervals also condition on ef48's test-set selection. Exact McNemar comparisons have very few discordant failures and cannot demonstrate equivalent risk. Oracle ef256 fallbacks are grid-capped descriptive values, not verified safe thresholds; the joint-feasibility sensitivity targets a smaller, selected population. Seed uncertainty and query uncertainty are distinct and have not been conflated.

The pre-extension statistical table is preserved byte-for-byte as statistical_tests_saved.csv. statistical_tests.csv combines those rows with the explicitly identified primary comparisons and blocked tests. The analysis as a whole is INCOMPLETE until the density comparisons can be performed.
''',encoding='utf-8')
    print(results[['primary_test_id','cohort','mean_difference','ci95_low','ci95_high','test_status']].to_string(index=False))
    return protocol


if __name__=='__main__':main()
