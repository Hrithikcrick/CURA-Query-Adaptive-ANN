"""Record archived TREC comparison targets without inventing a retrieval rerun.

This missing-input fallback deliberately leaves reproduced metrics blank.
Restored runs require a separate provenance check before this gate can open.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
BASE=ROOT/'msmarco_large/msmarco_1000000_q1000'
EFS=[10,16,24,32,48,64,96,128,192,256]


def main():
    recovery=json.loads((OUT/'trec_ranking_recovery_validation.json').read_text())
    state=json.loads((OUT/'trec_retrieval_state_audit.json').read_text())
    assert not recovery['exact_rankings_recovered'] and not recovery['hnsw_rankings_recovered']
    assert not state['retrieval_only_rerun_possible'] and not recovery['retrieval_only_rerun']
    source=BASE/'research_grade_validation/top10_query_budget_results.csv'
    reference=BASE/'research_grade_validation/exact_query_reference.csv'
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,reference]}
    budget=pd.read_csv(source)
    budget=budget[budget.source=='trec-dl-2019'].copy()
    exact=pd.read_csv(reference)
    exact=exact[exact.source=='trec-dl-2019']
    assert len(exact)==43 and len(budget)==430
    assert not budget.duplicated(['query_key','efSearch']).any()
    assert all(sorted(g.efSearch.tolist())==EFS for _,g in budget.groupby('query_key'))
    rows=[]

    def append(method,ef,metric,value,path,tolerance,notes):
        rp=path.relative_to(ROOT).as_posix()
        rows.append(dict(method=method,efSearch=ef,query_count=43,metric=metric,
            archived_value=float(value),reproduced_value=np.nan,absolute_difference=np.nan,
            tolerance=tolerance,comparison_completed=False,
            status='NOT_RUN_MISSING_ORIGINAL_RANKINGS_AND_RETRIEVAL_STATE',
            reference_source=rp,reference_sha256=hashes[rp],notes=notes))

    append('Exact',np.nan,'mean_nDCG_at_10',exact['exact_nDCG@10'].mean(),reference,1e-6,
        'Saved reference only; FULL nDCG has not been reproduced from document IDs.')
    for ef,g in budget.groupby('efSearch',sort=True):
        entries=[('mean_ANN_Recall_at_10',g['ANN_Recall@10'].mean(),1e-9),
                 ('mean_nDCG_at_10',g['nDCG@10'].mean(),1e-6),
                 ('IR_violation_rate',(g.nDCG_loss>.025).mean(),0.),
                 ('mean_nDCG_loss',g.nDCG_loss.mean(),1e-6),
                 ('mean_distance_computations',g.distance_computations.mean(),np.nan),
                 ('mean_latency_ms',g.mean_latency_ms.mean(),np.nan)]
        for metric,value,tolerance in entries:
            note='Saved metric only; no recreated observation exists.'
            if metric=='mean_distance_computations':
                note+=' Original graph and measurement environment are absent; work reproduction cannot be assessed.'
            if metric=='mean_latency_ms':
                note+=' Archived five-repeat per-query search means; no comparable timing rerun was performed.'
            append('HNSW',ef,metric,value,source,tolerance,note)
    pd.DataFrame(rows).to_csv(OUT/'trec_retrieval_reproduction.csv',index=False)
    text='''# TREC retrieval reproduction: NOT RUN

Original qrels and the 43 query identities are recovered. Exact and all ten
HNSW ranked-document outputs remain missing. The original controlled corpus
mapping/order, document vectors, query vectors and built HNSW index are also
missing, so a faithful retrieval-only rerun is impossible from available inputs.

[trec_retrieval_reproduction.csv](trec_retrieval_reproduction.csv) contains 61
archived comparison targets: exact mean nDCG and six metrics for each of ten
HNSW budgets. Reproduced values and differences are blank; every comparison is
explicitly incomplete. These are saved references, not reconstructed results or
a FULL-qrel ranking validation. Numerical comparison tolerances are prospective
(nDCG/loss 1e-6, ANN Recall 1e-9, identical violation rate); no tolerance was
applied to missing observations. Work and timing tolerances are unset because
their reproducibility conditions cannot be checked.

No retrieval, embedding generation, index rebuild or controller retraining was
performed. No alternative corpus was used. Configured encoder, normalization,
similarity and HNSW parameters can be read from source; that does not verify
the contents of an unavailable index or vector cache.

Restore original exact/HNSW document rankings for all 43 queries (and HNSW at
efSearch 10,16,24,32,48,64,96,128,192,256), or restore original cached vectors,
document map and built index. Verify corpus membership/order and query mapping,
then compare FULL per-query nDCG, ANN Recall and fixed violation rates before
thinning. Stop if recreated metrics disagree materially with the saved runs.
Latency is comparable only if hardware and timing conditions are established.

See [ranking validation](trec_ranking_recovery_validation.json),
[retrieval-state audit](trec_retrieval_state_audit.json) and
[detailed original settings](trec_retrieval_state.json).
'''
    (OUT/'trec_retrieval_reproduction_status.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='NOT_RUN',archived_comparison_targets=len(rows),
        completed_comparisons=0,retrieval_executed=False,source_hashes=hashes),indent=2))


if __name__=='__main__':main()
