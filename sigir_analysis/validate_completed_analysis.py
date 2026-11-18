"""Audit completed evidence without retrieval, scoring or statistical resampling."""
import ast
import csv
import gzip
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_completed(root):
    out=root/'sigir_analysis'
    bundle=out/'restored_ranking_evaluation'
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    protocol=read('analysis_protocol.json')
    preserved=read('preserved_analysis_hashes.json')
    gate=read('recovered_ranking_validation.json')
    run=read('restored_ranking_evaluation/validation.json')
    promotion=read('density_promotion_validation.json')
    stats=read('density_statistics_validation.json')
    for name in ['ranking_recovery_final_audit.json','trec_retrieval_state_final_audit.json']:
        audit=read(name)
        assert audit['status']=='RECOVERED'
        for target,digest in audit.get('evidence_hashes',audit.get('hashes',{})).items():
            assert sha(out/target)==digest,(name,target)
    checkout=read('checkout_validation.json')
    checkout_files=len(read('checkout_byte_integrity.json')['policies'])
    assert checkout['status']=='PASS' and checkout['files_checked']==checkout_files and not checkout['mismatches']
    for name,digest in protocol['input_hashes'].items():
        assert sha(root/name)==digest,name
    for name,digest in preserved.items():
        assert sha(out/name)==digest,name
    for name,digest in gate['source_hashes'].items():
        # The original map is intentionally not uploaded; local recovery verifies it.
        # Published ranking IDs and recorded gate remain independently auditable.
        if '/cache/' in name and not (root/name).exists():
            continue
        assert sha(root/name)==digest,name
    g=gate['validation']
    assert g['status']=='PASS' and g['full_metric_reproduction_verified']
    assert g['queries_found']==43 and g['hnsw_query_budget_count']==430
    efs=[10,16,24,32,48,64,96,128,192,256]
    assert g['ef_values_found']==efs
    assert run['overall_analysis_complete'] and run['ranking_seeds_evaluated']==50
    assert not any(run[k] for k in ['retrieval_executed','embeddings_regenerated','index_rebuilt','controller_retrained'])
    reproduction=pd.read_csv(out/'trec_retrieval_reproduction.csv')
    assert (out/'trec_retrieval_reproduction.csv').read_bytes()==(bundle/'trec_retrieval_reproduction.csv').read_bytes()
    assert (out/'trec_retrieval_reproduction.csv').read_bytes()==(out/'trec_retrieval_reproduction_per_query.csv').read_bytes()
    assert len(reproduction)==473 and reproduction.comparison_completed.all() and reproduction.comparison_passed.all()
    assert not reproduction.duplicated(['method','query_key','efSearch']).any()
    assert reproduction.nDCG_absolute_difference.max()<=1e-6
    assert reproduction.loss_absolute_difference.max()<=1e-6
    assert reproduction.recall_absolute_difference.max()<=1e-9
    for archived,recomputed,difference,tolerance in [
        ('archived_nDCG_at_10','recomputed_nDCG_at_10','nDCG_absolute_difference',1e-6),
        ('archived_nDCG_loss','recomputed_nDCG_loss','loss_absolute_difference',1e-6),
        ('archived_ANN_Recall_at_10','recomputed_ANN_Recall_at_10','recall_absolute_difference',1e-9)]:
        part=reproduction.dropna(subset=[archived,recomputed])
        actual=(part[archived]-part[recomputed]).abs()
        assert np.allclose(actual,part[difference],atol=1e-15,rtol=0)
        assert (actual<=tolerance).all()
    approx=reproduction[reproduction.method!='Exact']
    assert (approx.archived_IR_violation==approx.recomputed_IR_violation).all()
    for name in ['trec_exact_rankings.csv','trec_hnsw_rankings.csv']:
        frame=pd.read_csv(out/name,dtype={'doc_id':str})
        assert len(frame)==4300
        groups=['query_key']+(['efSearch'] if 'hnsw' in name else [])
        assert not frame.duplicated(groups+['rank']).any()
        assert not frame.duplicated(groups+['doc_id']).any()
        assert all(sorted(f['rank'])==list(range(1,len(f)+1)) for _,f in frame.groupby(groups))
    for name,digest in promotion['promoted_hashes'].items():
        assert sha(out/name)==sha(bundle/name)==digest,name
    assert sha(out/'trec_judgment_density_frontier.csv')==promotion['frontier_sha256']
    assert (out/'trec_judgment_density_frontier.csv').read_bytes()==(out/'trec_fixed_risk_work_by_density.csv').read_bytes()
    seed=pd.read_csv(out/'trec_judgment_density_per_seed.csv')
    query=pd.read_csv(out/'trec_judgment_density_per_query.csv')
    summary=pd.read_csv(out/'trec_judgment_density_summary.csv')
    assert len(seed)==151 and seed.evaluation_completed.all() and len(summary)==4
    assert not seed.duplicated(['condition','seed']).any()
    assert len(query)==64930 and query.ranking_sha256.nunique()==1
    assert not query.duplicated(['condition','seed','query_key','efSearch']).any()
    assert (query.groupby(['condition','seed']).size()==430).all()
    assert query.query_key.nunique()==43 and sorted(query.efSearch.unique())==efs
    assert (query.groupby(['query_key','efSearch']).ANN_Recall_at_10.nunique()==1).all()
    assert (query.groupby('query_key').min_ANN_safe_ef.nunique()==1).all()
    assert query.ANN_safe_found.all() and query.IR_safe_found.all()
    assert (seed.jointly_feasible_query_count==43).all()
    assert np.allclose(query.min_ANN_safe_ef-query.min_IR_safe_ef,query.ANN_minus_IR_budget_gap)
    base=root/'msmarco_large/msmarco_1000000_q1000'
    split=json.loads((base/'cura_controller/query_split.json').read_text())
    fold={key:name for name,keys in split.items() if isinstance(keys,list) for key in keys}
    assert len(fold)==1043 and sum(len(v) for v in split.values() if isinstance(v,list))==1043
    identities=pd.read_csv(base/'results/queries.csv',dtype=str)
    identities['fold']=identities.query_key.map(fold)
    identities['normalized_text']=identities.query_text.str.lower().str.split().str.join(' ')
    assert not identities.fold.isna().any()
    assert (identities.groupby('raw_query_id').fold.nunique()<=1).all()
    assert (identities.groupby('normalized_text').fold.nunique()<=1).all()
    assert set(identities[identities.source=='trec-dl-2019'].query_key)==set(query.query_key)
    oracle=pd.read_csv(base/'research_grade_validation/top10_oracle_query_results.csv')
    oracle=oracle[(oracle.source=='trec-dl-2019') & (oracle.IR_tolerance==.025) & (oracle.analysis_source=='ALL')]
    full_query=query[query.condition=='FULL'].drop_duplicates('query_key').set_index('query_key')
    oracle=oracle.set_index('query_key').reindex(full_query.index)
    assert len(oracle)==43 and oracle.min_IR_safe_ef.notna().all()
    for col in ['min_ANN_safe_ef','min_IR_safe_ef','IR_allows_lower','same_budget','IR_requires_higher']:
        assert np.allclose(oracle[col],full_query[col]),col
    full=seed[seed.condition=='FULL'].iloc[0]
    for cond in ['50_PERCENT_POSITIVES','25_PERCENT_POSITIVES','ONE_POSITIVE_PER_QUERY']:
        sub=seed[seed.condition==cond]
        assert sorted(sub.seed)==list(range(50))
        assert np.allclose(sub.mean_min_IR_safe_ef-full.mean_min_IR_safe_ef,
                           -(sub.mean_ANN_minus_IR_budget_gap-full.mean_ANN_minus_IR_budget_gap))
    # Verify already-prepared qrels and hashes; do not regenerate permutations.
    recovery=read('qrels_recovery.json')
    raw=(root/recovery['local_path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==recovery['sha256']
    original=defaultdict(dict)
    for line in raw.decode().splitlines():
        qid,_,doc,rel=line.split();original[qid][doc]=int(rel)
    assert len(original)==43 and sum(map(len,original.values()))==9260
    records={}
    with gzip.open(out/'trec_thinned_positive_judgments.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line);key=(item['condition'],item['seed'],item['raw_query_id'])
            assert key not in records
            records[key]=item['retained_positive_judgments']
    assert len(records)==6450
    for s in range(50):
        for qid,labels in original.items():
            half=records[('50_PERCENT_POSITIVES',s,qid)]
            quarter=records[('25_PERCENT_POSITIVES',s,qid)]
            one=records[('ONE_POSITIVE_PER_QUERY',s,qid)]
            assert set(one)<=set(quarter)<=set(half)<=set(labels)
            n=sum(v>0 for v in labels.values())
            assert (len(half),len(quarter),len(one))==((n+1)//2,(n+3)//4,1)
            assert all(labels[d]==v and v>0 for kept in [half,quarter,one] for d,v in kept.items())
    for row in seed.itertuples():
        judgments=original if row.condition=='FULL' else {
            qid:{d:v for d,v in labels.items() if v<=0 or d in records[(row.condition,row.seed,qid)]}
            for qid,labels in original.items()}
        digest=hashlib.sha256(json.dumps(judgments,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert digest==row.qrels_sha256
    front=pd.read_csv(out/'trec_fixed_risk_work_by_density.csv')
    cura=pd.read_csv(out/'trec_cura_by_density.csv')
    assert len(front)==1510 and len(cura)==151
    for col in ['ANN_Recall_at_10','mean_distance_computations','mean_latency_ms','p95_latency_ms']:
        assert (front.groupby('efSearch')[col].nunique()==1).all(),col
    archived=pd.read_csv(base/'research_grade_validation/top10_query_budget_results.csv')
    archived=archived[archived.source=='trec-dl-2019']
    for ef,part in archived.groupby('efSearch'):
        point=front[(front.condition=='FULL') & (front.efSearch==ef)].iloc[0]
        assert abs(point.mean_distance_computations-part.distance_computations.mean())<1e-9
        assert abs(point.mean_latency_ms-part.mean_latency_ms.mean())<1e-9
        assert abs(point.p95_latency_ms-np.percentile(part.mean_latency_ms,95))<1e-9
    for col in ['selection_sha256','mean_work','ANN_Recall_at_10','mean_latency_ms','p95_latency_ms','mean_efSearch']:
        assert cura[col].nunique()==1,col
    assert sha(out/'statistical_tests.csv')==stats['statistics_sha256']
    with (out/'statistical_tests.csv').open(newline='',encoding='utf-8-sig') as stream:
        allstats=list(csv.DictReader(stream))
    # Original header is retained in the pre-extension snapshot recorded by the finalizer.
    cols=stats['preserved_completed_original_columns']
    old=[{k:r[k] for k in cols} for r in allstats if r['primary_test_id'] not in {'P5','P6','P7','P8'}]
    assert len(old)==42
    assert hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()==stats['preserved_completed_rows_sha256']
    density=[r for r in allstats if r['primary_test_id'] in {'P5','P6','P7','P8'}]
    assert len(allstats)==54 and len(density)==12
    for r in density:
        assert r['test_status']=='COMPLETED_DENSITY_SEED_ANALYSIS'
        for col in ['mean_difference','ci95_low','ci95_high','p_value','holm_p_value_12_density_comparisons']:
            assert np.isfinite(float(r[col]))
        assert float(r['ci95_low'])<=float(r['mean_difference'])<=float(r['ci95_high'])
        assert 0<=float(r['p_value'])<=float(r['holm_p_value_12_density_comparisons'])<=1
    review=json.loads((root/'reports/50_CASE_FAILURE_REVIEW_STATUS.json').read_text())
    for key,digest in [('source_worksheet','source_worksheet_sha256'),('checkpoint','checkpoint_sha256'),('human_decision_log','decision_log_sha256')]:
        assert sha(root/review[key])==review[digest]
    with (root/review['checkpoint']).open(newline='',encoding='utf-8-sig') as stream:
        human=list(csv.DictReader(stream))
    log=json.loads((root/review['human_decision_log']).read_text())
    done=[r for r in human if r['human_verified']=='YES']
    assert len(human)==50 and len({r['query_id'] for r in human})==50
    assert len(done)==review['human_verified']==review['completed']
    assert review['remaining']==50-len(done)
    for row in done:
        events=[e for e in log['events'] if e.get('query_id')==row['query_id'] and e['action'] in {'legacy_human_review_preserved','human_review_completed'}]
        assert events and events[-1]['human_category']==row['human_category'] and events[-1]['human_note']==row['human_note']
    pending=pd.read_csv(root/review['assistance_csv'])
    assert len(pending)==review['remaining']==review['model_assisted']
    assert (pending.suggestion_source=='MODEL_ASSISTED').all()
    assert (pending.human_verified_snapshot=='NO').all()
    if review['status']=='COMPLETE':
        assert len(done)==50 and review['remaining']==0
        assert all(r['review_status']=='COMPLETE' and r['human_category'] and r['human_note'].strip() for r in done)
        taxonomy=pd.read_csv(root/'reports/HUMAN_50_FAILURE_TAXONOMY.csv')
        counts=pd.Series([r['human_category'] for r in done]).value_counts()
        assert taxonomy.failure_count.sum()==50 and abs(taxonomy.share_percent.sum()-100)<1e-9
        for r in taxonomy.itertuples():assert r.failure_count==counts.get(r.category,0)
        human_audit=json.loads((root/'reports/HUMAN_50_REVIEW_VALIDATION.json').read_text())
        assert human_audit['status']=='PASS' and human_audit['human_verified_total']==50
        assert human_audit['checkpoint_sha256']==sha(root/review['checkpoint'])
        assert human_audit['decision_log_sha256']==sha(root/review['human_decision_log'])
        assert human_audit['taxonomy_sha256']==sha(root/'reports/HUMAN_50_FAILURE_TAXONOMY.csv')
        assert human_audit['analysis_sha256']==sha(root/'reports/HUMAN_50_FAILURE_ANALYSIS.md')
    images={}
    for name in ['budget_gap','zero_ndcg','ir_safe_budget','risk_work']:
        path=out/f'judgment_density_vs_{name}.png'
        with Image.open(path) as im:im.verify()
        with Image.open(path) as im:images[path.name]=list(im.size)
    for path in out.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    paper=root/'paper/sigir_short.tex'
    text=paper.read_text(encoding='utf-8')
    assert sha(paper)==read('short_paper_evidence_gate.json')['current_paper_sha256']
    # Static source validation does not substitute for compiling the manuscript.
    stack=[]
    for token,env in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
        if token=='begin':stack.append(env)
        else:assert stack and stack.pop()==env
    assert not stack and r'\end{document}' in text
    for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
        assert (paper.parent/target).exists(),target
    docs=[out/'README.md',out/'RESTORED_RANKING_EVALUATION.md',root/'reports/SIGIR_RESEARCH_ANALYSIS.md',root/'paper/SIGIR_REVISED_DIRECTION.md']
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent/target.split('#')[0]).exists(),(path,target)
    assert len(pd.read_csv(out/'exact_ndcg_stratified_results.csv'))==4
    assert len(pd.read_csv(out/'fixed_risk_work_frontier.csv'))==11
    assert len(pd.read_csv(out/'oracle_budget_gap_analysis.csv'))==18
    return dict(validation_pass=True,overall_validation='PASS',input_validation_passed=True,
                saved_data_validation_passed=True,density_experiment_completed=True,
                statistical_analysis_complete=True,scientific_judgment_density_outcome='A',
                input_hashes_unchanged=len(protocol['input_hashes']),preserved_numerical_CSV_hashes=len(preserved),
                original_qrel_rows=9260,prepared_qrel_records=6450,prepared_seed_rows=151,
                ranking_seed_evaluations=50,query_seed_condition_budget_rows=64930,
                FULL_reproduction_comparisons=473,ANN_recall_invariance='PASS',
                FULL_oracle_budgets_match_archived='PASS',query_ID_and_exact_text_leakage='PASS',
                original_corpus_identity_and_order='PASS_WITH_DOCUMENTED_ARCHIVE_LINEAGE_LIMITATION',
                no_unintended_corpus_change=True,source_hashes_recorded=True,
                final_recovery_audits='PASS',portable_checkout_checksum_files=checkout_files,
                search_work_and_saved_timing_invariance='PASS',jointly_feasible_queries_every_condition=43,
                completed_saved_statistical_rows_preserved=42,completed_density_followups=12,
                human_verified=len(done),human_reviews_remaining=review['remaining'],pending_model_assisted=len(pending),
                local_report_links='PASS',python_syntax_check='PASS',figures=images,
                manuscript_compilation='NOT RUN; static environment and figure checks only',
                previous_completed_analyses_redone=False,retrieval_executed=False,
                embedding_generation=False,index_construction=False,controller_retraining=False,
                archive_identity_limitation=g['corpus_identity_limitation'])
