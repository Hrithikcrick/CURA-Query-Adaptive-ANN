"""Prepare real 50-seed qrel conditions; stop when original rankings are absent.

This performs judgment preparation, not a retrieval or ranking reevaluation.
Unavailable ranking-dependent results stay blank. No illustrative density
curves or fabricated results are produced.
"""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
BASE=ROOT/'msmarco_large/msmarco_1000000_q1000'
EFS=[10,16,24,32,48,64,96,128,192,256]
CONDITIONS=['FULL','50_PERCENT_POSITIVES','25_PERCENT_POSITIVES','ONE_POSITIVE_PER_QUERY']


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def main():
    recovery=json.loads((OUT/'qrels_recovery.json').read_text())
    path=ROOT/recovery['local_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==recovery['sha256']
    queries=pd.read_csv(BASE/'results/queries.csv',dtype=str)
    queries=queries[queries.source=='trec-dl-2019'].sort_values('query_key')
    assert len(queries)==43 and queries.query_key.is_unique
    qrels=defaultdict(dict)
    for line in path.read_text().splitlines():
        qid,_,docid,grade=line.split();qrels[qid][docid]=int(grade)
    assert set(queries.raw_query_id)==set(qrels)
    source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,BASE/'results/queries.csv',BASE/'cura_controller/selected_cura_full_trec_ood.csv',BASE/'research_grade_validation/top10_query_budget_results.csv']}
    # Fifty paired seed permutations; no query or condition gets a separate
    # pseudo-independent draw. Nested prefixes preserve paired conditions.
    rows=[];records=[];full_positive=sum(sum(v>0 for v in q.values()) for q in qrels.values())
    nonpositive=sum(sum(v<=0 for v in q.values()) for q in qrels.values())
    for seed in range(50):
        conditions={condition:{} for condition in CONDITIONS[1:]}
        for row in queries.itertuples():
            original=qrels[row.raw_query_id]
            positives=sorted(docid for docid,grade in original.items() if grade>0)
            assert positives
            key=hashlib.sha256(f'{seed}\0{row.raw_query_id}'.encode('utf-8')).digest()
            rng=np.random.Generator(np.random.PCG64(int.from_bytes(key[:8],'big')))
            perm=[positives[i] for i in rng.permutation(len(positives))]
            sizes=[math.ceil(.5*len(perm)),math.ceil(.25*len(perm)),1]
            sets=[set(perm[:max(1,n)]) for n in sizes]
            assert sets[2]<=sets[1]<=sets[0]<=set(positives)
            for condition,retained in zip(CONDITIONS[1:],sets):
                thinned={docid:grade for docid,grade in original.items() if grade<=0 or docid in retained}
                assert all(thinned.get(d)==g for d,g in original.items() if g<=0)
                assert all(original[d]==g for d,g in thinned.items())
                assert sum(g>0 for g in thinned.values())==len(retained)
                conditions[condition][row.raw_query_id]=thinned
                records.append(dict(condition=condition,seed=seed,query_key=row.query_key,raw_query_id=row.raw_query_id,retained_positive_judgments={d:original[d] for d in sorted(retained)}))
        for condition,judgments in conditions.items():
            count=sum(sum(v>0 for v in q.values()) for q in judgments.values())
            rows.append(dict(condition=condition,seed=seed,status='QRELS_PREPARED_RANKINGS_MISSING',evaluation_completed=False,query_count=43,retained_positive_judgments=count,retained_positive_fraction=count/full_positive,preserved_nonpositive_judgments=nonpositive,qrels_sha256=canonical_hash(judgments)))
    with (OUT/'trec_thinned_positive_judgments.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
            for record in records:gz.write((json.dumps(record,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8'))
    full=dict(condition='FULL',seed=-1,status='ORIGINAL_QRELS_RECOVERED_RANKINGS_MISSING',evaluation_completed=False,query_count=43,retained_positive_judgments=full_positive,retained_positive_fraction=1.,preserved_nonpositive_judgments=nonpositive,qrels_sha256=canonical_hash(dict(qrels)))
    per_seed=pd.DataFrame([full]+rows)
    metrics=['exact_ndcg_zero_query_fraction','mean_exact_ndcg_at_10','mean_min_IR_safe_ef','mean_min_ANN_safe_ef','mean_ANN_minus_IR_budget_gap','relative_IR_safe_budget_reduction','IR_allows_lower_rate','same_budget_rate','IR_requires_higher_rate','ANN_Recall_at_10','CURA_IR_violation_rate','CURA_mean_distance_computations']
    for name in metrics:per_seed[name]=np.nan
    per_seed.to_csv(OUT/'trec_judgment_density_per_seed.csv',index=False)
    summary=[]
    for condition in CONDITIONS:
        g=per_seed[per_seed.condition==condition]
        row=dict(condition=condition,status='BLOCKED_MISSING_RANKINGS_AND_RECOVERY_CACHES',queries=43,number_of_evaluated_seeds=0,qrel_seeds_prepared=1 if condition=='FULL' else 50,ranking_seeds_evaluated=0,retained_positive_judgments=int(g.retained_positive_judgments.iloc[0]),retained_positive_fraction=float(g.retained_positive_fraction.iloc[0]),preserved_nonpositive_judgments=nonpositive)
        for metric in metrics:row[metric]=np.nan
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'trec_judgment_density_summary.csv',index=False)
    pd.DataFrame(columns=['condition','seed','method','efSearch','query_count','IR_violations','IR_violation_rate','wilson95_low','wilson95_upper','ANN_Recall_at_10','mean_nDCG_at_10','mean_distance_computations','mean_latency_ms','p95_latency_ms','ranking_sha256']).to_csv(OUT/'trec_judgment_density_frontier.csv',index=False)
    pd.DataFrame(columns=['condition','seed','efSearch','query_count','IR_violations','IR_violation_rate','wilson95_low','wilson95_upper','mean_nDCG_at_10','mean_nDCG_loss','ANN_Recall_at_10','mean_distance_computations','mean_latency_ms','p95_latency_ms','ranking_sha256']).to_csv(OUT/'trec_fixed_risk_work_by_density.csv',index=False)
    pd.DataFrame(columns=['condition','seed','query_key','query_id','efSearch','exact_nDCG_at_10','HNSW_nDCG_at_10','nDCG_loss','IR_unsafe','ANN_Recall_at_10','min_ANN_safe_ef','min_IR_safe_ef','ANN_safe_found','IR_safe_found','ANN_minus_IR_budget_gap','IR_allows_lower','same_budget','IR_requires_higher','ranking_sha256']).to_csv(OUT/'trec_judgment_density_per_query.csv',index=False)
    pd.DataFrame(columns=['condition','seed','policy_label','query_count','CURA_IR_violation_rate','mean_nDCG_at_10','mean_nDCG_loss','mean_work','risk_matched_fixed_ef','fixed_observed_violation_rate','CURA_minus_fixed_work','selection_sha256']).to_csv(OUT/'trec_cura_by_density.csv',index=False)
    # Explicit recovery requirements. A future evaluator must validate original
    # ranking hashes and FULL metrics before filling any density-study results.
    cache=BASE/'cache'
    requirements=[dict(artifact='Original TREC-DL-2019 passage qrels',status='RECOVERED',path=recovery['local_path']),dict(artifact='Exact original-document top10 rankings for all 43 queries',status='MISSING',path='cache/exact_top100_indices.npy plus original docids.txt, or archived original-document rankings'),dict(artifact='HNSW original-document top10 rankings for all 43 queries at all 10 ef values',status='MISSING',path='No saved ranking file; Phase-05 only saved scalar metrics'),dict(artifact='Original FAISS row-to-document-ID mapping',status='MISSING',path=(cache/'docids.txt').relative_to(ROOT).as_posix()),dict(artifact='Original query vectors',status='MISSING',path=(cache/'query_embeddings.npy').relative_to(ROOT).as_posix()),dict(artifact='Original corpus vectors',status='MISSING',path=(cache/'corpus_embeddings.f32').relative_to(ROOT).as_posix()),dict(artifact='Original built HNSW index',status='MISSING',path=(cache/'hnsw_M32_efC200.faiss').relative_to(ROOT).as_posix())]
    for r in requirements[3:]:assert not (ROOT/r['path']).exists(),'Recovery state changed; review recovered cache before marking blocked'
    pd.DataFrame(requirements).to_csv(OUT/'ranking_recovery_requirements.csv',index=False)
    status=dict(status='BLOCKED',experiment_completed=False,input_validation_passed=True,reason='Original exact/HNSW ranked document IDs are absent; original query/corpus embeddings, document map and HNSW index are also absent. Qrels have been recovered.',query_count=43,required_ef_values=EFS,qrels_status='RECOVERED',qrels_sha256=recovery['sha256'],qrel_seeds_prepared_per_stochastic_condition=50,ranking_seeds_evaluated=0,qrel_record_count=len(records),qrel_checks='passed: nested sets, preserved grades/nonpositive judgments, original query identities',retrieval_executed=False,embedding_generation=False,index_construction=False,controller_retraining=False,source_hashes=source_hashes,ANN_recall_invariance='Required per-query/per-ef invariant for future reevaluation; not newly verified without rankings',unavailable_plots=['judgment_density_vs_budget_gap.png','judgment_density_vs_zero_ndcg.png','judgment_density_vs_ir_safe_budget.png','judgment_density_vs_risk_work.png'])
    (OUT/'ranking_recovery_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    # This compact pointer supersedes the earlier filename-level audit.
    (OUT/'artifact_audit.json').write_text(json.dumps(dict(status='BLOCKED',qrels_status='RECOVERED',comprehensive_audit='local_retrieval_audit.json',requirements='ranking_recovery_requirements.csv',details='ranking_recovery_status.json'),indent=2),encoding='utf-8')
    text='''# Same-ranking TREC judgment-density study: BLOCKED

The original qrels are recovered and validated. The remaining blocker is the missing exact/HNSW document rankings and original retrieval caches. No ranking-dependent density result or plot has been fabricated.

## Qrels recovered

The original [NIST passage judgments](https://trec.nist.gov/data/deep/2019qrels-pass.txt) were recovered via the [ir-datasets checksum mirror](https://mirror.ir-datasets.com/2f4be390198da108f6845c822e5ada14) after the NIST endpoint failed certificate validation. TLS verification was retained. The 187,092-byte file has MD5 `2f4be390198da108f6845c822e5ada14`, 9,260 judgments and exactly the original 43 query IDs. Every query's grade>0 count matches the saved exact-reference metadata. See qrels_recovery.json and trec_qrels_query_audit.csv. This does not yet validate FULL ranking metrics because ranking identities are unavailable.

## Exhaustive local audit

local_retrieval_audit.json and local_artifact_inventory.csv document recursive enumeration of the entire workspace, including hidden/ignored files. CSV/TSV contents, JSON/JSONL structures, text qrel/run formats, NumPy headers, archive members and all local Git blobs (including unreachable objects) were inspected. Archive content was inspected, not just filenames. No search or model generation was run. Pickles, if present, are inspected as opcodes without executing them.

The only workspace vector cache is data/beir/fiqa_embeddings.npz: 57,638 document vectors and 648 query vectors, dimension 384, with FiQA IDs. This is a different corpus and is not usable as the original million-scale MS MARCO cache. Git candidates are parameter/configuration IDs, completion flags or manual FiQA evidence, not ranked-document arrays. Source sweeps retain first_indices only in memory and export scalar metrics.

## Prepared conditions and exact deterministic protocol

FULL uses the complete supplied original qrels once. For seeds 0..49 and each original raw query ID, sort its grade>0 original document IDs lexicographically. Seed PCG64 with int.from_bytes(SHA256(f'{seed}\\0{raw_qid}'.encode('utf-8')).digest()[:8], 'big'), and permute that sorted list. Retain nested prefixes of max(1,ceil(0.5*n)), max(1,ceil(0.25*n)), and 1 positive for 50_PERCENT_POSITIVES, 25_PERCENT_POSITIVES and ONE_POSITIVE_PER_QUERY. Retain original grades and every nonpositive judgment; drop the other positive judgments. Removed positives become unjudged, with relevance 0 under the original protocol. FULL means all available pooled judgments, not complete relevance ground truth. Grade 1 counts as positive because the original graded nDCG code gives it gain 1; no binary-relevance threshold is substituted.

trec_thinned_positive_judgments.jsonl.gz stores the actual retained original positive judgments for every query/condition/seed; nonpositive judgments are referenced from the unchanged original qrels. The per-seed CSV has 150 prepared stochastic seed rows plus one FULL row, all explicitly evaluation_completed=False. Its qrel counts/hashes are real preparation results. All retrieval-dependent fields are blank. No ranking seeds have been evaluated, so no seed CIs or paired density tests exist.

## Evaluation gate once original rankings are restored

Freeze the query set, exact and all ten HNSW top10 original-document rankings, work observations, thresholds and existing CURA selections. Require 43 exact rankings and 430 query/budget HNSW rankings, no duplicate documents within top10, valid original IDs and immutable hashes. Validate FULL nDCG and ANN Recall against archived per-query metrics before thinning. Unjudged=0; gain=2**grade-1; discount=log2(rank+2); IDCG is recomputed from all retained positive judgments for the query, not only retrieved documents. IDCG<=0 gives nDCG0. Loss=max(0,exact-approximate nDCG), violation iff loss>0.025.

Recompute zero-reference fraction, exact/approximate scores, losses, smallest actually safe tested ef and feasibility flags, ANN-minus-IR gap, lower-budget rate, all ten fixed points and frozen CURA risk/work. ANN Recall, ANN-safe ef and work must remain invariant across all conditions and seeds. Do not assume IR safety is monotone in ef or judgment density. Flag no-safe-in-grid outcomes; do not silently treat ef256 as safe.

Use seed bootstrap intervals for seed-average effects conditional on these 43 queries; pair each thinned query/seed with its FULL query. Add a crossed query-and-seed bootstrap for population sensitivity while preserving all paired conditions. FULL is one observation, not 50 independent replications. Pointwise Wilson intervals and post-hoc fixed matching do not confer new risk guarantees.

## Recovery boundary

Restore original exact/HNSW ranked document IDs to evaluate without retrieval. If HNSW rankings were not persisted, retrieval alone requires the original cached query vectors, built index and original row/document map. Exact rankings additionally require the original exact arrays or original corpus vectors for exhaustive search without constructing an index. These caches are missing. No embeddings, index construction or controller training were attempted. See ranking_recovery_requirements.csv for exact paths.

The four requested judgment-density plots (budget gap, exact-zero fraction, IR-safe budget, and risk/work) are intentionally absent because there are no ranking reevaluation observations. The canonical trec_fixed_risk_work_by_density.csv, per-query and unchanged-policy CURA tables are schemas with zero observations. Historical FULL results remain in fixed_risk_work_frontier_trec_saved_full.csv and oracle_budget_gap_analysis.csv and must not be confused with a completed density experiment.
'''
    (OUT/'trec_judgment_density_status.md').write_text(text,encoding='utf-8')
    for path,digest in source_hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    print(json.dumps({k:v for k,v in status.items() if k!='source_hashes'},indent=2))
    return status


if __name__=='__main__':main()
