"""Verify local ranking candidates and record the retrieval-only recovery gate.

This script reads artifacts and Git blobs. It never performs retrieval, builds
an index, loads an encoder, unpickles data, or claims unavailable reproduction.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import re
import subprocess
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
BASE=ROOT/'msmarco_large/msmarco_1000000_q1000'
EFS=[10,16,24,32,48,64,96,128,192,256]
DOC_FIELD=re.compile(r'^(doc_?id|docids|document_?ids?|retrieved_ids|topk_ids|candidate_ids)$',re.I)
RANK_FIELD=re.compile(r'^ranks?$',re.I)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def review_table(text,fields):
    first=text.splitlines()[0] if text.splitlines() else ''
    delimiter='\t' if '\t' in first else ','
    reader=csv.DictReader(io.StringIO(text),delimiter=delimiter)
    names=reader.fieldnames or []
    qcols=[n for n in names if n in ['query_key','query_id','raw_query_id','qid']]
    doccols=[n for n in names if DOC_FIELD.fullmatch(n)]
    rankcols=[n for n in names if RANK_FIELD.fullmatch(n)]
    efcols=[n for n in names if n.lower() in ['efsearch','ef','method','budget']]
    nonempty=Counter();seen=set();examples={};count=0
    for row in reader:
        count+=1
        for n in fields:
            value=row.get(n,'')
            if value and value.strip():
                nonempty[n]+=1
                if n in ['candidate_id','human_evidence_doc_ids_or_ranks'] and n not in examples:examples[n]=value[:100]
        for n in qcols:
            qid=row.get(n,'').split('::')[-1]
            if qid in TREC_IDS:seen.add(qid)
    plausible=len(seen)==43 and bool(doccols) and bool(rankcols) and bool(efcols)
    if 'human_evidence_doc_ids_or_ranks' in names:
        reason='Manual free-text evidence; lacks full 43-query exact/ef-labelled document-rank coverage'
    elif any(n.startswith('candidate_ef') for n in names) or 'candidate_id' in names:
        reason='Policy candidate/configuration fields; no document-ID/rank retrieval schema'
    elif not doccols or not rankcols:
        reason='No structured original-document-ID plus rank schema'
    elif len(seen)!=43:reason='Does not cover the original 43 TREC query IDs'
    elif not efcols:reason='Cannot identify exact versus individual HNSW ef settings'
    else:reason='Requires corpus-membership and saved-metric reproduction before acceptance'
    return dict(rows=count,fields=fields,nonempty_candidate_cells=dict(nonempty),candidate_examples=examples,TREC_query_count=len(seen),has_document_id_column=bool(doccols),has_rank_column=bool(rankcols),has_method_or_budget_column=bool(efcols),plausible_complete_ranking_candidate=plausible,accepted=False,reason=reason)


def main():
    if (OUT/'recovered_ranking_validation.json').exists():
        raise SystemExit('Original ranking recovery already recorded. Do not overwrite it with the historical missing-input audit; use validate_analysis.py.')
    global TREC_IDS
    queries=pd.read_csv(BASE/'results/queries.csv',dtype=str)
    tq=queries[queries.source=='trec-dl-2019']
    TREC_IDS=set(tq.raw_query_id);assert len(TREC_IDS)==43
    audit=json.loads((OUT/'local_retrieval_audit.json').read_text())
    review=[]
    for candidate in audit['git_candidates']:
        entry=dict(source='git_blob',oid=candidate['oid'],kind=candidate['kind'])
        if candidate['kind']=='tabular':
            blob=subprocess.run(['git','cat-file','blob',candidate['oid']],cwd=ROOT,capture_output=True,check=True).stdout.decode('utf-8-sig')
            entry.update(review_table(blob,candidate['candidate_id_fields']))
        else:
            entry.update(accepted=False,reason='Binary/archive/run candidate requires explicit original-corpus ranking verification',details=candidate,plausible_complete_ranking_candidate=candidate.get('plausible_complete_ranking_candidate',False))
        review.append(entry)
    # Archive members were already parsed in full by the audit. Identify any
    # table schema that could hold the required query/document/rank records.
    working_candidates=[]
    for candidate in audit['content_inspections']:
        columns=candidate.get('columns',[])
        if candidate.get('candidate_id_fields') or candidate.get('plausible_complete_ranking_candidate'):
            seen=candidate.get('TREC_query_ids_seen',[])
            plausible=candidate.get('plausible_complete_ranking_candidate',False) or (len(seen)==43 and any(DOC_FIELD.fullmatch(n) for n in columns) and any(RANK_FIELD.fullmatch(n) for n in columns))
            working_candidates.append(dict(path=candidate['path'],candidate_fields=candidate.get('candidate_id_fields',[]),TREC_query_count=len(seen),has_document_id_column=any(DOC_FIELD.fullmatch(n) for n in columns),has_rank_column=any(RANK_FIELD.fullmatch(n) for n in columns),plausible_complete_ranking_candidate=plausible,accepted=False,reason='No original 43-query document/rank retrieval schema' if not plausible else 'Requires corpus-membership and metric reproduction checks'))
    (OUT/'git_candidate_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    (OUT/'ranking_candidate_review.json').write_text(json.dumps(working_candidates,indent=2),encoding='utf-8')
    candidates=[x for x in review+working_candidates if x.get('plausible_complete_ranking_candidate')]
    if candidates:
        raise RuntimeError('Potential new rankings require manual provenance verification; refusing to label missing or start density evaluation.')
    normalized=[OUT/'trec_exact_rankings.csv',OUT/'trec_hnsw_rankings.csv']
    assert not any(p.exists() for p in normalized),'Normalized ranking files appeared; evaluate their provenance before proceeding'
    cache=BASE/'cache'
    paths=dict(query_embeddings=cache/'query_embeddings.npy',corpus_embeddings=cache/'corpus_embeddings.f32',HNSW_index=cache/'hnsw_M32_efC200.faiss',docid_map=cache/'docids.txt',exact_ranked_indices=cache/'exact_top100_indices.npy',exact_scores=cache/'exact_top100_scores.npy',corpus_meta=cache/'corpus_meta.json')
    found={k:p.exists() for k,p in paths.items()}
    if any(found.values()):raise RuntimeError('Original cache changed: inspect it for retrieval-only recovery before marking blocked')
    metadata=dict(encoder='sentence-transformers/msmarco-MiniLM-L6-cos-v5',encoder_provenance='04_msmarco_large_pilot.py MODEL_NAME',dimension=384,normalization='normalize_embeddings=True for query and document encoding',similarity='inner product on normalized float32 vectors',exact_reference='IndexFlatIP; exact top100 arrays saved by Phase04; original exact index not serialized',hnsw_type='IndexHNSWFlat',HNSW_M=32,efConstruction=200,search_k_for_this_study=10,exact_cache_depth=100,efSearch_values=EFS,search_threads=1,timing_repeats=5,original_faiss_version='1.15.1',reported_original_corpus_documents=1000629,corpus_size_provenance='results/large_pilot_summary.txt',original_subset_membership_verifiable=False,query_key_mapping=tq[['query_key','raw_query_id']].to_dict('records'),cache_files={k:dict(path=p.relative_to(ROOT).as_posix(),present=found[k]) for k,p in paths.items()},retrieval_only_fallback_possible=False,reason='Original index, query vectors, corpus vectors, exact arrays and document mapping are absent; FiQA cache is for a different corpus')
    metadata['source_hashes']={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'04_msmarco_large_pilot.py',ROOT/'05_research_grade_validation.py',BASE/'research_grade_validation/experiment_protocol.json',BASE/'results/large_pilot_summary.txt',BASE/'results/queries.csv',BASE/'cura_controller/selected_cura_full_trec_ood.csv']}
    (OUT/'trec_retrieval_state.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    # Configured parameters can be recovered from code; missing vectors and
    # indexes prevent verification of the state that was actually searched.
    state_audit=dict(
        corpus_mapping_available=False,document_vectors_available=False,
        query_vectors_available=False,exact_index_available=False,
        hnsw_index_available=False,encoder_identity_available=True,
        normalization_verified=False,metric_verified=False,
        corpus_identity_verified=False,query_identity_verified=True,
        retrieval_only_rerun_possible=False,
        missing_components=[p.relative_to(ROOT).as_posix() for p in paths.values()],
        original_corpus_order_available=False,original_query_order_available=True,
        configured_encoder=metadata['encoder'],encoder_revision_available=False,
        configured_normalization=metadata['normalization'],
        configured_similarity=metadata['similarity'],configured_seed=42,
        HNSW_M=32,efConstruction=200,search_k=10,efSearch_values=EFS,
        exact_index_note='The original exact index was not serialized; original exact arrays or original corpus vectors are also absent.',
        verification_scope='Query identities/order verified from saved query table and qrels. Normalization and metric are configured in source, but cannot be verified on missing original vectors/index. Encoder name is available; immutable model revision is not.',
        no_unintended_corpus_change=True,retrieval_executed=False,
        embeddings_regenerated=False,index_rebuilt=False,controller_retrained=False,
        source_paths=list(metadata['source_hashes']),source_hashes=metadata['source_hashes'])
    (OUT/'trec_retrieval_state_audit.json').write_text(json.dumps(state_audit,indent=2),encoding='utf-8')
    recovery=json.loads((OUT/'qrels_recovery.json').read_text())
    assert sha(ROOT/recovery['local_path'])==recovery['sha256']
    status=dict(qrels_recovered=True,exact_rankings_recovered=False,hnsw_rankings_recovered=False,retrieval_only_rerun=False,embeddings_regenerated=False,index_rebuilt=False,controller_retrained=False,metric_reproduction_pass=None,status='BLOCKED',original_query_count=43,required_efSearch=EFS,exact_query_rankings_verified=0,hnsw_query_budget_rankings_verified=0,expected_exact_query_rankings=43,expected_hnsw_query_budget_rankings=430,candidate_verification='No valid original-corpus ranking candidate',git_blobs_checked=audit['git_blob_count'],git_candidate_blobs_reviewed=len(review),working_tree_archive_candidate_schemas_reviewed=len(working_candidates),audit_errors=audit['errors'],source_hashes=metadata['source_hashes'],notes=['Qrels have 9,260 original judgments. The 6,450 prepared records are 43 queries x 50 seeds x 3 thinning conditions, not original qrels or retrieval rankings.','All normalized ranking files remain absent; no empty/fabricated ranking files were created.','Query IDs and frozen per-query CURA ef selections exist, but scalar metrics do not determine ranked document IDs.','Original encoder, M, efConstruction, k, normalization and similarity are recoverable from source/protocol; the actual original retrieval state is not.','Metric reproduction was NOT RUN because rankings are missing; null is not a passing result.','No retrieval-only fallback is possible from the available FiQA-only vector cache.','Restore original exact/HNSW document rankings, or original corpus/query vectors, built index and ID map. No expensive reconstruction was attempted.'])
    status.update(queries_expected=43,queries_found=0,query_identities_available=43,
        queries_found_definition='Queries with recovered and provenance-verified document rankings; saved query identities alone do not count.',
        ef_values_expected=EFS,ef_values_found=[],doc_id_mapping_verified=False,
        saved_metric_reproduction_verified=None,
        source_paths=list(metadata['source_hashes'])+[recovery['local_path']],
        source_hashes={**metadata['source_hashes'],recovery['local_path']:recovery['sha256']})
    (OUT/'trec_ranking_recovery_validation.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    gate_path=OUT/'short_paper_evidence_gate.json'
    gate=json.loads(gate_path.read_text())
    assert gate['current_paper_sha256']==sha(ROOT/'paper/sigir_short.tex'),'Short paper changed without a recorded maintenance check'
    assert not gate['evidence_complete'],'Missing-ranking fallback cannot mark the density evidence complete'
    print(json.dumps({k:v for k,v in status.items() if k not in ['source_hashes','notes']},indent=2))
    return status


if __name__=='__main__':main()
