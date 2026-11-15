"""Evaluate restored original rankings after a strict FULL-reproduction gate.

No retrieval, embeddings, models or indexes are loaded. With missing inputs the
command reports BLOCKED and writes no experiment results. Successful runs go to
a separate evidence directory for validation before the main report is revised.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
EFS=[10,16,24,32,48,64,96,128,192,256]
CONDITIONS=['FULL','50_PERCENT_POSITIVES','25_PERCENT_POSITIVES','ONE_POSITIVE_PER_QUERY']
sys.path.insert(0,str(OUT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qrel_hash(judgments):
    return hashlib.sha256(json.dumps(judgments,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def thinned_qrels(qrels,seed):
    """The exact nested PCG64 protocol used by the saved judgment preparation."""
    result={condition:{} for condition in CONDITIONS[1:]}
    for qid,original in qrels.items():
        positives=sorted(d for d,grade in original.items() if grade>0)
        if not positives:raise ValueError('Every original TREC query must have a positive judgment')
        key=hashlib.sha256(f'{seed}\0{qid}'.encode()).digest()
        rng=np.random.Generator(np.random.PCG64(int.from_bytes(key[:8],'big')))
        perm=[positives[i] for i in rng.permutation(len(positives))]
        for condition,size in zip(CONDITIONS[1:],[math.ceil(.5*len(perm)),math.ceil(.25*len(perm)),1]):
            retained=set(perm[:max(1,size)])
            result[condition][qid]={d:g for d,g in original.items() if g<=0 or d in retained}
        assert set(result[CONDITIONS[3]][qid])<=set(result[CONDITIONS[2]][qid])<=set(result[CONDITIONS[1]][qid])<=set(original)
    return result


def _json_default(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=_json_default,allow_nan=False),encoding='utf-8')


def plot_results(summary,frontier,curas,directory):
    sys.path.insert(0,str(OUT/'.plot_deps'))
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['FULL','50%','25%','1/query']
    for metric,name,ylabel in [
        ('mean_ANN_minus_IR_budget_gap','judgment_density_vs_budget_gap.png','Mean ANN-minus-IR safe ef'),
        ('exact_ndcg_zero_query_fraction','judgment_density_vs_zero_ndcg.png','Exact nDCG@10 zero-query fraction'),
        ('mean_min_IR_safe_ef','judgment_density_vs_ir_safe_budget.png','Mean minimum IR-safe ef')]:
        frame=summary.set_index('condition').loc[CONDITIONS]
        x=np.arange(4);y=frame[metric].to_numpy(float)
        lo=frame[metric+'_ci95_low'].to_numpy(float);hi=frame[metric+'_ci95_high'].to_numpy(float)
        # Percentile intervals need not contain the point estimate. Plot their
        # endpoints directly instead of constructing possibly negative yerr.
        fig,ax=plt.subplots(figsize=(7,4.5))
        ax.plot(x,y,'o-',color='#245c91');ax.vlines(x,lo,hi,color='#245c91',linewidth=1.7)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,xlabel='Retained positive judgments')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(directory/name,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for condition,label in zip(CONDITIONS,labels):
        g=frontier[frontier.condition==condition].groupby('efSearch',sort=True)[['mean_distance_computations','IR_violation_rate']].mean()
        line,=ax.plot(g.mean_distance_computations,g.IR_violation_rate,'o-',label=label)
        c=curas[curas.condition==condition]
        ax.scatter(c.mean_work.mean(),c.CURA_IR_violation_rate.mean(),marker='*',s=150,color=line.get_color())
    ax.set(xlabel='Mean recorded distance work per query',ylabel='IR violation fraction')
    ax.grid(alpha=.2);ax.legend(title='* = unchanged CURA policy');fig.tight_layout()
    fig.savefig(directory/'judgment_density_vs_risk_work.png',dpi=180);plt.close(fig)


def evaluate_restored(root,exact_path,hnsw_path,provenance_path,output_dir,*,seeds=50,bootstrap_resamples=20000):
    """Return a completed evidence bundle only after every input gate passes."""
    from ranking_input_gate import load_and_validate_rankings
    from density_metrics import evaluate_query_rankings,summarize_queries,aggregate_fixed,aggregate_frozen_cura
    from density_uncertainty import add_seed_confidence_intervals,density_paired_tests
    if seeds<50:raise ValueError('At least 50 deterministic thinning seeds are required')
    if bootstrap_resamples<20000:raise ValueError('At least 20,000 bootstrap resamples are required')
    root=Path(root).resolve();output_dir=Path(output_dir).resolve()
    # Never replace the source directory or write outside the research area.
    output_dir.relative_to(root/'sigir_analysis')
    if output_dir==root/'sigir_analysis':raise ValueError('Use a separate evidence directory; do not overwrite the current blocked report with an unreviewed run')
    if output_dir.exists():raise ValueError('Evidence directory already exists; choose a new empty output path')
    gate=load_and_validate_rankings(root,Path(exact_path),Path(hnsw_path),Path(provenance_path))
    base=root/'msmarco_large/msmarco_1000000_q1000'
    recovery=json.loads((root/'sigir_analysis/qrels_recovery.json').read_text())
    qrels={}
    for line in (root/recovery['local_path']).read_text().splitlines():
        qid,_,docid,grade=line.split();qrels.setdefault(qid,{})[docid]=int(grade)
    queries=pd.read_csv(base/'results/queries.csv',dtype=str)
    queries=queries[queries.source=='trec-dl-2019']
    key_to_id=dict(zip(queries.query_key,queries.raw_query_id))
    raw_budget_path=base/'research_grade_validation/top10_query_budget_results.csv'
    selection_path=base/'cura_controller/selected_cura_full_trec_ood.csv'
    protocol_path=root/'sigir_analysis/analysis_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    for source in [raw_budget_path,selection_path]:
        if protocol['input_hashes'].get(source.relative_to(root).as_posix())!=sha(source):
            raise ValueError('Frozen archived source differs from preserved analysis protocol: '+str(source))
    work=pd.read_csv(raw_budget_path);work=work[work.source=='trec-dl-2019'].copy()
    selections=pd.read_csv(selection_path)
    if set(selections.query_key)!=set(key_to_id) or not selections.query_key.is_unique:
        raise ValueError('Frozen CURA choices must contain exactly the original 43 queries')
    if not selections.efSearch.isin(EFS).all():raise ValueError('Frozen CURA choice outside original ef grid')
    paired=selections.merge(work[['query_key','efSearch','distance_computations','mean_latency_ms']],
        on=['query_key','efSearch'],validate='one_to_one',suffixes=('_selected','_archived'))
    if len(paired)!=43:raise ValueError('Frozen CURA choice missing from archived budget grid')
    for field in ['distance_computations','mean_latency_ms']:
        if not np.allclose(paired[field+'_selected'],paired[field+'_archived'],rtol=0,atol=1e-9):
            raise ValueError('Selected search measurement differs from archived sweep: '+field)
    expected_work=np.where(selections.efSearch==10,selections.probe_ndis,
        selections.probe_ndis+selections.distance_computations)
    expected_latency=np.where(selections.efSearch==10,selections.probe_latency_ms,
        selections.probe_latency_ms+selections.mean_latency_ms)
    if not np.allclose(expected_work,selections.effective_distance_computations,rtol=0,atol=1e-9):
        raise ValueError('Frozen effective work does not match original probe/selected-search accounting')
    if not np.allclose(expected_latency,selections.effective_latency_ms,rtol=0,atol=1e-9):
        raise ValueError('Frozen effective timing does not match original search-component accounting')
    source_hashes={**gate['source_hashes'],
        protocol_path.relative_to(root).as_posix():sha(protocol_path),
        raw_budget_path.relative_to(root).as_posix():sha(raw_budget_path),
        selection_path.relative_to(root).as_posix():sha(selection_path)}
    ranking_hash=hashlib.sha256(json.dumps(gate['ranking_hashes'],sort_keys=True).encode()).hexdigest()
    selection_hash=sha(selection_path)
    seed_rows=[];query_rows=[];fixed_rows=[];cura_rows=[];reference_ann=None;reference_safe=None;full_oracles=None
    full_positive=sum(sum(g>0 for g in q.values()) for q in qrels.values())
    nonpositive=sum(sum(g<=0 for g in q.values()) for q in qrels.values())
    condition_sets=[('FULL',-1,qrels)]
    for seed in range(seeds):
        condition_sets.extend((condition,seed,judgments) for condition,judgments in thinned_qrels(qrels,seed).items())
    # Computation happens in memory; a failed condition leaves all study output
    # files untouched. Frozen source hashes are checked again before writing.
    for condition,seed,judgments in condition_sets:
        keyed_qrels={key:judgments[qid] for key,qid in key_to_id.items()}
        per_query=evaluate_query_rankings(gate['exact'],gate['hnsw'],keyed_qrels,efs=EFS)
        per_query['query_id']=per_query.query_key.map(key_to_id)
        ann=per_query.sort_values(['query_key','efSearch']).ANN_Recall_at_10.to_numpy()
        safe=per_query.sort_values(['query_key','efSearch']).min_ANN_safe_ef.to_numpy()
        if reference_ann is None:reference_ann=ann.copy();reference_safe=safe.copy()
        else:
            if not np.array_equal(ann,reference_ann):raise ValueError('ANN Recall changed under qrel thinning')
            if not np.array_equal(safe,reference_safe,equal_nan=True):raise ValueError('ANN-safe ef changed under qrel thinning')
        front=aggregate_fixed(per_query,work)
        cura=aggregate_frozen_cura(per_query,selections,fixed_frontier=front)
        metrics=summarize_queries(per_query)
        if metrics.pop('query_count')!=43:raise ValueError('Condition lost an original TREC query')
        oracles=per_query.drop_duplicates('query_key').set_index('query_key')
        if full_oracles is None:
            full_oracles=oracles.copy()
            if not (full_oracles.ANN_safe_found & full_oracles.IR_safe_found).all():
                raise ValueError('FULL oracle feasibility disagrees with the saved 43-query reference')
        paired_keys=oracles.index[oracles.ANN_safe_found & oracles.IR_safe_found]
        metrics['paired_gap_query_count']=len(paired_keys)
        metrics['paired_FULL_mean_ANN_minus_IR_budget_gap']=float(full_oracles.loc[paired_keys,'ANN_minus_IR_budget_gap'].mean()) if len(paired_keys) else np.nan
        metrics.update(CURA_IR_violation_rate=cura['CURA_IR_violation_rate'],
            CURA_mean_distance_computations=cura['mean_work'])
        seed_rows.append(dict(condition=condition,seed=seed,status='COMPLETED_FROM_VERIFIED_ORIGINAL_RANKINGS',
            evaluation_completed=True,query_count=43,retained_positive_judgments=sum(sum(g>0 for g in q.values()) for q in judgments.values()),
            retained_positive_fraction=sum(sum(g>0 for g in q.values()) for q in judgments.values())/full_positive,
            preserved_nonpositive_judgments=nonpositive,qrels_sha256=qrel_hash(judgments),**metrics))
        for frame,collection in [(per_query,query_rows),(front,fixed_rows)]:
            frame=frame.copy();frame['condition']=condition;frame['seed']=seed;frame['ranking_sha256']=ranking_hash;collection.append(frame)
        cura_rows.append(dict(condition=condition,seed=seed,selection_sha256=selection_hash,**cura))
    per_seed=pd.DataFrame(seed_rows);per_query=pd.concat(query_rows,ignore_index=True)
    frontier=pd.concat(fixed_rows,ignore_index=True);curas=pd.DataFrame(cura_rows)
    for column in ['ANN_Recall_at_10','mean_distance_computations','mean_latency_ms','p95_latency_ms']:
        if (frontier.groupby('efSearch')[column].nunique(dropna=False)>1).any():
            raise ValueError('Frozen ANN/work/timing invariant failed: '+column)
    for column in ['mean_work','mean_latency_ms','p95_latency_ms']:
        if curas[column].nunique(dropna=False)!=1:raise ValueError('Frozen CURA work/timing invariant failed: '+column)
    if per_query.duplicated(['condition','seed','query_key','efSearch']).any():raise ValueError('Duplicate query/condition/seed/ef rows')
    if len(per_query)!=43*10*(1+3*seeds):raise ValueError('Incomplete per-query evaluation coverage')
    from density_uncertainty import METRIC_COLUMNS
    sensitivity_metrics=[c for c in per_seed if pd.api.types.is_numeric_dtype(per_seed[c]) and (c.startswith('legacy_grid_capped_') or c in [
        'jointly_feasible_query_count','ANN_not_safe_in_grid','IR_not_safe_in_grid',
        'mean_available_ANN_safe_ef','mean_available_IR_safe_ef','paired_gap_query_count'])]
    summary=add_seed_confidence_intervals(per_seed,bootstrap_resamples=bootstrap_resamples,
        metric_columns=[*METRIC_COLUMNS,*sensitivity_metrics])
    tests=density_paired_tests(per_seed,bootstrap_resamples=bootstrap_resamples,signflip_resamples=bootstrap_resamples)
    statistics_complete=bool((tests.test_status=='COMPLETED_DENSITY_SEED_ANALYSIS').all())
    summaries_complete=bool(summary.required_metrics_completed.all())
    for path,digest in source_hashes.items():
        if sha(root/path)!=digest:raise ValueError('Source input changed during evaluation: '+path)
    # Stage the complete bundle on the same filesystem and rename atomically.
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.density_stage_',dir=output_dir.parent) as temporary:
        stage=Path(temporary)/'evidence';stage.mkdir()
        for name,frame in [
            ('trec_judgment_density_per_query.csv',per_query),
            ('trec_judgment_density_per_seed.csv',per_seed),
            ('trec_judgment_density_summary.csv',summary),
            ('trec_fixed_risk_work_by_density.csv',frontier),
            ('trec_cura_by_density.csv',curas),
            ('trec_retrieval_reproduction.csv',gate['reproduction']),
            ('density_statistical_tests.csv',tests)]:frame.to_csv(stage/name,index=False)
        plot_results(summary,frontier,curas,stage)
        validation=dict(status='PASS',experiment_completed=True,scoring_completed=True,
            statistical_analysis_complete=statistics_complete,
            requested_finite_metric_summaries_complete=summaries_complete,
            overall_analysis_complete=statistics_complete and summaries_complete,
            inference_completion_note='PASS covers ranking/scoring integrity. Censored oracle cohorts may leave some requested estimates or tests unavailable; inspect the separate statistical completion flags.',query_count=43,
            ef_values=EFS,ranking_seeds_evaluated=seeds,query_seed_condition_budget_rows=len(per_query),
            ANN_recall_invariant=True,ANN_safe_budget_invariant=True,search_work_invariant=True,
            CURA_selections_and_work_invariant=True,FULL_metric_reproduction_pass=True,
            ranking_validation=gate['validation'],ranking_hashes=gate['ranking_hashes'],
            source_hashes=source_hashes,retrieval_executed=False,embeddings_regenerated=False,
            index_rebuilt=False,controller_retrained=False,
            uncertainty_scope='Seed intervals condition on the 43 fixed queries and supplied original rankings; no query-population inference or risk-equivalence claim.',
            density_outcome='Requires interpretation of actual paired effects; non-significance alone does not establish Outcome B.',
            main_report_updated=False)
        _write_json(stage/'validation.json',validation)
        _write_json(stage/'run_manifest.json',dict(schema_version=1,seeds=seeds,
            bootstrap_resamples=bootstrap_resamples,conditions=CONDITIONS,
            main_report_promotion='Validate this evidence bundle before incorporating its results into the main research report or manuscript.',
            source_hashes=source_hashes,ranking_hashes=gate['ranking_hashes']))
        stage.rename(output_dir)
    return validation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exact',type=Path,default=OUT/'trec_exact_rankings.csv')
    parser.add_argument('--hnsw',type=Path,default=OUT/'trec_hnsw_rankings.csv')
    parser.add_argument('--provenance',type=Path,default=OUT/'trec_ranking_provenance.json')
    parser.add_argument('--output-dir',type=Path,default=OUT/'restored_ranking_evaluation')
    parser.add_argument('--preflight',action='store_true',help='Check restored inputs and FULL reproduction without writing any density output')
    parser.add_argument('--seeds',type=int,default=50)
    args=parser.parse_args(argv)
    missing=[str(p) for p in [args.exact,args.hnsw,args.provenance] if not p.is_file()]
    if missing:
        print(json.dumps(dict(status='BLOCKED',reason='Original rankings/provenance unavailable',missing=missing,
            experiment_completed=False,ranking_seeds_evaluated=0,experiment_outputs_written=False),indent=2))
        return 2
    try:
        if args.preflight:
            from ranking_input_gate import load_and_validate_rankings
            gate=load_and_validate_rankings(ROOT,args.exact,args.hnsw,args.provenance)
            print(json.dumps(dict(status='PREFLIGHT_PASS',density_evaluation_performed=False,
                ranking_validation=gate['validation']),indent=2,default=_json_default))
        else:
            result=evaluate_restored(ROOT,args.exact,args.hnsw,args.provenance,args.output_dir,seeds=args.seeds)
            print(json.dumps(result,indent=2,default=_json_default))
    except (ValueError,FileNotFoundError,AssertionError) as exc:
        print(json.dumps(dict(status='BLOCKED_OR_REJECTED',reason=str(exc),experiment_completed=False,
            experiment_outputs_written=False),indent=2))
        return 2
    return 0


if __name__=='__main__':sys.exit(main())
