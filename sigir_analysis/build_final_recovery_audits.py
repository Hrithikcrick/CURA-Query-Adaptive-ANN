"""Consolidate existing recovery evidence; no new artifact scan or retrieval."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def portable(value):
    if isinstance(value,str):
        return re.sub(r'C:[\\/]+Users[\\/]+(?:HRITHIK SHARMA|HRITHI~1)', '~', value).replace('pytest-of-HRITHIK SHARMA','pytest-of-user')
    if isinstance(value,list):return [portable(v) for v in value]
    if isinstance(value,dict):return {portable(k):portable(v) for k,v in value.items()}
    return value


def main():
    read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
    # Publish portable metadata, not machine-specific absolute home paths.
    for name in ['recovery_delta_external_inventory.json','recovery_delta_20261116.json']:
        (OUT/name).write_text(json.dumps(portable(read(name)),indent=2)+'\n',encoding='utf-8')
    old=read('local_retrieval_audit.json');delta=read('recovery_delta_20261116.json')
    provenance=read('trec_ranking_provenance.json');gate=read('recovered_ranking_validation.json')
    records=[]
    inspections=old['content_inspections']+old['git_content_inspections']
    for item in inspections:
        source=item.get('path',item.get('inspection_path','UNKNOWN'))
        oid=item.get('oid')
        if source.startswith('git:') and not oid:oid=source.split(':')[1].split('!')[0]
        path=ROOT/source
        digest=sha(path) if path.is_file() and path.stat().st_size<100_000_000 else None
        doccols=[c for c in item.get('columns',[]) if c.lower() in {'doc_id','docid','document_id','retrieved_ids','candidate_ids'}]
        records.append(dict(source_path_or_git_object=source,
            artifact_hash={'sha256_current_file':digest,'git_object_sha1':oid},
            hash_scope='Current file bytes if available; Git object identity otherwise. Historic inspection is linked separately.',
            query_coverage=item.get('TREC_query_ids_seen',[]),
            ef_coverage=item.get('ef_values',[]),
            document_ID_format='Possible columns: '+', '.join(doccols) if doccols else 'No verified original ranking document-ID field',
            rank_ordering_available=False,exact_HNSW_identity='UNVERIFIED_OR_NOT_A_RANKING',
            validation_outcome='REJECTED_AS_ORIGINAL_TREC_RANKING',
            rejection_reason='Historical schema/content inspection did not verify original-corpus IDs plus query, rank, exact/HNSW and ef association.',
            inspection_evidence=item))
    for kind in ['new_git_blobs','archive_reviews','temporary_parquet_reviews']:
        for item in delta[kind]:
            records.append(dict(source_path_or_git_object=item.get('name',item.get('oid','UNKNOWN')),
                artifact_hash={'sha256':item.get('sha256'),'git_object_sha1':item.get('oid')},
                query_coverage=[],ef_coverage=[],document_ID_format='UNVERIFIED',rank_ordering_available=False,
                exact_HNSW_identity='UNVERIFIED',validation_outcome='INITIAL_SCAN_NOT_ACCEPTED',
                rejection_reason='Generic initial scan did not verify original ranking axes; accepted original arrays are separately verified below. Some unrelated parquet streams could not be inspected.',
                inspection_evidence=item))
    for member,info in provenance['archive_member_checks'].items():
        exact=member.endswith('exact_top100_indices.npy')
        ann=member.endswith('stability_top10_searches.npz')
        records.append(dict(source_path_or_git_object='~/Downloads/IR_PROJECT.zip::IR_PROJECT/PILOT/'+member,
                artifact_hash={'sha256':info['sha256'],'zip_crc32':info['zip_crc32']},
                query_coverage=43 if exact or ann else 'Component rather than ranked query output',
                ef_coverage=gate['validation']['ef_values_found'] if ann else [],
                document_ID_format='Original integer row indices translated through checksum-recorded ordered MS MARCO passage-ID map' if exact or ann else 'See component provenance',
                rank_ordering_available=exact or ann,
                exact_HNSW_identity='EXACT_TOP100' if exact else 'HNSW_TOP10' if ann else 'ORIGINAL_RUN_COMPONENT',
                validation_outcome='ACCEPTED_ORIGINAL_RANKINGS_FULL_REPRODUCTION_PASS' if exact or ann else 'VERIFIED_COMPONENT',
                rejection_reason=None,inspection_evidence=info))
    audit=dict(status='RECOVERED',date='2026-11-17',
        scope='Consolidates all recorded workspace/Git inspection entries and external delta candidates; no repeated scientific analysis or whole-disk completeness claim.',
        candidates=records,candidate_records=len(records),
        query_count=43,ef_values=gate['validation']['ef_values_found'],
        accepted_ranking_arrays=2,rankings_inferred_from_aggregate_metrics=False,
        full_qrel_reproduction='PASS',
        limitations=[gate['validation']['corpus_identity_limitation'],delta['limitations'],
                     'Some rejected historical entries lack a recorded SHA256 or ef coverage; null means unrecorded, never inferred. Container/member identities and original inspection evidence are retained.'],
        evidence_hashes={n:sha(OUT/n) for n in ['local_retrieval_audit.json','git_candidate_review.json','recovery_delta_20261116.json','recovery_delta_external_inventory.json','trec_ranking_provenance.json','recovered_ranking_validation.json']})
    (OUT/'ranking_recovery_final_audit.json').write_text(json.dumps(portable(audit),indent=2)+'\n',encoding='utf-8')
    state=dict(status='RECOVERED',date='2026-11-17',corpus_subset_manifest_found=True,
        corpus_subset_deterministically_reconstructable='NOT NEEDED: original ordered map recovered; source seed42 alone is not an independent subset checksum',
        document_id_mapping_verified=True,corpus_order_verified=True,document_embeddings_found=True,
        query_embeddings_found=True,exact_index_found=False,exact_ranked_arrays_found=True,hnsw_index_found=True,
        encoder_verified=True,encoder_revision_verified=False,normalization_verified=True,
        similarity_metric_verified=True,hnsw_parameters_verified=True,
        retrieval_only_rerun_possible=True,retrieval_only_rerun_executed=False,
        full_reconstruction_possible=False,
        full_reconstruction_note='Original vectors/index permit retrieval-only reuse, which was unnecessary. Fresh encoder regeneration cannot be certified as identical because the historical immutable revision was not pinned.',
        original_corpus_documents=1000629,encoder='sentence-transformers/msmarco-MiniLM-L6-cos-v5',
        normalization='unit-length float32 original vectors',similarity='inner product',M=32,efConstruction=200,
        threads=1,configured_selection_seed=42,ef_values=gate['validation']['ef_values_found'],
        missing_components=['Serialized exact index was not saved; original exact arrays suffice','Independent historical map checksum unavailable','Historical immutable encoder revision unpinned'],
        scientific_blocker=None,identity_limitation=gate['validation']['corpus_identity_limitation'],
        evidence_paths=['trec_ranking_provenance.json','recovered_ranking_validation.json','trec_retrieval_state.json','trec_retrieval_state_audit.json'],
        hashes={n:sha(OUT/n) for n in ['trec_ranking_provenance.json','recovered_ranking_validation.json','trec_retrieval_state.json','trec_retrieval_state_audit.json']},
        archive_hash=provenance['archive']['sha256'],archive_members=provenance['archive_member_checks'])
    (OUT/'trec_retrieval_state_final_audit.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(f'Final audits consolidated: {len(records)} candidate/inspection records, 2 accepted original ranking arrays; no retrieval.')


if __name__=='__main__':main()
