"""Validate preserved results and real judgment preparation, without retrieval."""
from pathlib import Path
from collections import defaultdict
import gzip
import hashlib
import json
import re
import py_compile
import sys
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'


def main():
    protocol=json.loads((OUT/'analysis_protocol.json').read_text())
    for path,digest in protocol['input_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    preserved=json.loads((OUT/'preserved_analysis_hashes.json').read_text())
    for name,digest in preserved.items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
    recovery=json.loads((OUT/'qrels_recovery.json').read_text())
    original=defaultdict(dict)
    raw=(ROOT/recovery['local_path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==recovery['sha256']
    for line in raw.decode().splitlines():
        qid,_,docid,rel=line.split();original[qid][docid]=int(rel)
    assert len(original)==43
    records={}
    with gzip.open(OUT/'trec_thinned_positive_judgments.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);key=(r['condition'],r['seed'],r['raw_query_id'])
            assert key not in records
            kept=r['retained_positive_judgments'];full=original[r['raw_query_id']]
            assert kept and all(full[d]==grade and grade>0 for d,grade in kept.items())
            records[key]=kept
    assert len(records)==43*50*3
    for seed in range(50):
        for qid,full in original.items():
            half=records[('50_PERCENT_POSITIVES',seed,qid)]
            quarter=records[('25_PERCENT_POSITIVES',seed,qid)]
            one=records[('ONE_POSITIVE_PER_QUERY',seed,qid)]
            assert set(one)<=set(quarter)<=set(half)<=set(full)
            count=sum(grade>0 for grade in full.values())
            assert len(half)==(count+1)//2
            assert len(quarter)==(count+3)//4
            assert len(one)==1
    per_seed=pd.read_csv(OUT/'trec_judgment_density_per_seed.csv')
    assert len(per_seed)==151 and not per_seed.evaluation_completed.any()
    for row in per_seed.itertuples():
        if row.condition=='FULL':judgments=dict(original)
        else:
            judgments={qid:{docid:grade for docid,grade in full.items() if grade<=0 or docid in records[(row.condition,row.seed,qid)]} for qid,full in original.items()}
        digest=hashlib.sha256(json.dumps(judgments,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert digest==row.qrels_sha256
        assert sum(sum(grade>0 for grade in q.values()) for q in judgments.values())==row.retained_positive_judgments
        assert sum(sum(grade<=0 for grade in q.values()) for q in judgments.values())==row.preserved_nonpositive_judgments
    metrics=['exact_ndcg_zero_query_fraction','mean_exact_ndcg_at_10','mean_min_IR_safe_ef','mean_min_ANN_safe_ef','mean_ANN_minus_IR_budget_gap','relative_IR_safe_budget_reduction','IR_allows_lower_rate','same_budget_rate','IR_requires_higher_rate','ANN_Recall_at_10','CURA_IR_violation_rate','CURA_mean_distance_computations']
    assert per_seed[metrics].isna().all().all()
    summary=pd.read_csv(OUT/'trec_judgment_density_summary.csv')
    assert len(summary)==4 and summary.ranking_seeds_evaluated.sum()==0
    assert summary[metrics].isna().all().all()
    assert len(pd.read_csv(OUT/'trec_judgment_density_frontier.csv'))==0
    for name in ['trec_fixed_risk_work_by_density.csv','trec_judgment_density_per_query.csv','trec_cura_by_density.csv']:
        assert len(pd.read_csv(OUT/name))==0
    strat=pd.read_csv(OUT/'exact_ndcg_stratified_results.csv')
    assert len(strat)==4 and strat.query_count.sum()==400
    assert len(pd.read_csv(OUT/'fixed_risk_work_frontier.csv'))==11
    assert len(pd.read_csv(OUT/'oracle_budget_gap_analysis.csv'))==18
    audit=json.loads((OUT/'local_retrieval_audit.json').read_text())
    assert not audit['errors']
    # No pickle, tensor loader, model or index search is invoked by this checker.
    for path in OUT.glob('*.py'):py_compile.compile(str(path),doraise=True)
    images={}
    for path in OUT.glob('*.png'):
        with Image.open(path) as img:img.verify()
        with Image.open(path) as img:images[path.name]=list(img.size)
    unavailable=['judgment_density_vs_budget_gap.png','judgment_density_vs_zero_ndcg.png','judgment_density_vs_ir_safe_budget.png','judgment_density_vs_risk_work.png']
    assert not any((OUT/name).exists() for name in unavailable)
    missing=[]
    docs=list(OUT.glob('*.md'))+[ROOT/'reports/SIGIR_RESEARCH_ANALYSIS.md',ROOT/'paper/SIGIR_REVISED_DIRECTION.md']
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#') and not (path.parent/target.split('#')[0]).exists():missing.append((str(path),target))
    assert not missing,missing
    for path in OUT.glob('*.json'):json.loads(path.read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
    banned_directory='professor'+'_'+'feedback'
    banned_report='PROFESSOR'+'_'+'SIGIR_FEEDBACK_RESPONSE.md'
    assert not (ROOT/banned_directory).exists()
    assert not (ROOT/'reports'/banned_report).exists()
    recovery_gate=json.loads((OUT/'trec_ranking_recovery_validation.json').read_text())
    assert recovery_gate['qrels_recovered'] and recovery_gate['metric_reproduction_pass'] is None
    assert not any(recovery_gate[k] for k in ['exact_rankings_recovered','hnsw_rankings_recovered','retrieval_only_rerun','embeddings_regenerated','index_rebuilt','controller_retrained'])
    split=json.loads((ROOT/'msmarco_large/msmarco_1000000_q1000/cura_controller/query_split.json').read_text())
    query_table=pd.read_csv(ROOT/'msmarco_large/msmarco_1000000_q1000/results/queries.csv',dtype=str)
    fold={key:name for name,keys in split.items() if isinstance(keys,list) for key in keys}
    assert len(fold)==1043 and len(set().union(*[set(v) for v in split.values() if isinstance(v,list)]))==1043
    assert sum(len(v) for v in split.values() if isinstance(v,list))==1043
    query_table['fold']=query_table.query_key.map(fold)
    assert not query_table.fold.isna().any()
    query_table['normalized_text']=query_table.query_text.str.lower().str.split().str.join(' ')
    assert (query_table.groupby('raw_query_id').fold.nunique()<=1).all()
    assert (query_table.groupby('normalized_text').fold.nunique()<=1).all()
    tq=query_table[query_table.source=='trec-dl-2019']
    assert len(tq)==43 and set(tq.raw_query_id)==set(original)
    ef_values=[10,16,24,32,48,64,96,128,192,256]
    budget=pd.read_csv(ROOT/'msmarco_large/msmarco_1000000_q1000/research_grade_validation/top10_query_budget_results.csv')
    tb=budget[budget.source=='trec-dl-2019']
    assert len(tb)==430 and not tb.duplicated(['query_key','efSearch']).any()
    assert all(sorted(g.efSearch.tolist())==ef_values for _,g in tb.groupby('query_key'))
    assert not (OUT/'trec_exact_rankings.csv').exists() and not (OUT/'trec_hnsw_rankings.csv').exists()
    paper_gate=json.loads((OUT/'short_paper_evidence_gate.json').read_text())
    assert hashlib.sha256((ROOT/'paper/sigir_short.tex').read_bytes()).hexdigest()==paper_gate['current_paper_sha256']
    primary=pd.read_csv(OUT/'primary_statistical_results.csv')
    complete=primary[primary.test_status=='COMPLETED_SAVED_DATA']
    blocked=primary[primary.test_status=='BLOCKED_MISSING_RANKINGS']
    assert set(complete.primary_test_id)>={'P1','P2','P3','P4'}
    assert len(blocked)==6 and blocked[['mean_difference','ci95_low','ci95_high','p_value']].isna().all().all()
    stats_status=json.loads((OUT/'statistical_analysis_status.json').read_text())
    for path,digest in stats_status['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for path,digest in recovery_gate['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    assert complete[['mean_difference','ci95_low','ci95_high','p_value']].notna().all().all()
    assert (complete.ci95_low<=complete.mean_difference).all() and (complete.mean_difference<=complete.ci95_high).all()
    q_oracle=pd.read_csv(OUT/'oracle_budget_gap_per_query.csv')
    for row in complete[complete.primary_test_id.isin(['P1','P2'])].itertuples():
        cohort=q_oracle[q_oracle.source=='msmarco-dev-small']
        if row.primary_test_id=='P2':cohort=cohort[cohort['exact_nDCG@10']>0]
        if row.estimator=='both_criteria_feasible':cohort=cohort[(cohort.ANN_safe_found==1)&(cohort.IR_safe_found==1)]
        assert len(cohort)==row.queries
        assert abs((cohort.min_ANN_safe_ef-cohort.min_IR_safe_ef).mean()-row.mean_difference)<1e-10
    controller=ROOT/'msmarco_large/msmarco_1000000_q1000/cura_controller'
    fixed_source=pd.read_csv(controller/'selected_fixed_ir_msmarco_test.csv')
    cura_source=pd.read_csv(controller/'selected_cura_full_msmarco_test.csv')
    assert set(fixed_source.query_key)==set(cura_source.query_key)==set(split['test_queries'])
    p3=complete[complete.primary_test_id=='P3'].iloc[0]
    assert abs(fixed_source.effective_distance_computations.mean()-cura_source.effective_distance_computations.mean()-p3.mean_difference)<1e-9
    p4=complete[complete.primary_test_id=='P4'].iloc[0]
    fixed48=budget[(budget.efSearch==48)&budget.query_key.isin(split['test_queries'])]
    assert abs(cura_source.effective_distance_computations.mean()-fixed48.distance_computations.mean()-p4.mean_difference)<1e-9
    requirements={
        '43_original_queries_retained':'PASS',
        'full_original_qrels_present':'PASS',
        'all_10_ef_values_in_archived_scalar_runs':'PASS',
        'all_10_ef_values_in_verified_rankings':'BLOCKED_MISSING_RANKINGS',
        'same_rankings_across_qrel_conditions':'NOT_VERIFIED_MISSING_RANKINGS',
        'ANN_recall_invariant_across_conditions':'NOT_VERIFIED_MISSING_RANKINGS',
        '50_thinning_seeds_scored_per_condition':'FAIL_ZERO_COMPLETED_EVALUATIONS',
        'query_ID_and_exact_text_leakage':'PASS',
        'duplicate_query_budget_rows_in_archived_metrics':'PASS',
        'no_duplicate_query_rank_doc_rows':'NOT_VERIFIED_MISSING_RANKINGS',
        'FULL_recomputed_metric_reproduction':'NOT_RUN_MISSING_RANKINGS',
        'source_hashes_saved_and_unchanged':'PASS',
        'no_embedding_generation_or_index_rebuild':'PASS',
        'no_controller_retraining':'PASS',
        'short_paper_unchanged_until_evidence_complete':'PASS'}
    result=dict(checks='input and saved-data integrity checks passed; requested full-study requirements unmet',validation_pass=False,overall_validation='FAIL',input_validation_passed=True,saved_data_validation_passed=True,density_experiment_completed=False,required_experiment_checks=requirements,input_hashes_unchanged=len(protocol['input_hashes']),preserved_numerical_CSV_hashes=len(preserved),qrels_recovery='passed: original checksum, exact query identities and all positive counts',original_qrel_rows=sum(len(v) for v in original.values()),prepared_qrel_records=len(records),stochastic_qrel_seeds_prepared_per_condition=50,prepared_seed_rows=len(per_seed),ranking_seed_evaluations=0,condition_checks='passed: nested sets, original grades, retained counts, preserved nonpositive judgments and canonical hashes',ranking_metrics_unavailable_and_blank=True,ANN_recall_invariance='not verified: rankings absent',local_report_links='passed',python_compile_check='passed',audit_errors=0,figures=images,unproduced_density_plots=unavailable,short_paper_unchanged=True,statistical_analysis_complete=False,completed_primary_tests=['P1','P2','P3','P4'],blocked_primary_tests=['P5','P6'],scientific_judgment_density_outcome='UNRESOLVED; cannot select A/B/C',manuscript_compilation='not run; evidence gate remains closed',retrieval_executed=False,embedding_generation=False,index_construction=False,controller_retraining=False)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':sys.exit(0 if main()['validation_pass'] else 1)
