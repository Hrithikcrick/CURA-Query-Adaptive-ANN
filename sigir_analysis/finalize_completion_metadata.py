"""Reconcile current completion metadata with the already-validated evidence.

No scientific estimates, rankings, qrels or human decisions are regenerated.
"""
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gate=read('recovered_ranking_validation.json')
    g=gate['validation'];assert g['status']=='PASS' and g['full_metric_reproduction_verified']
    density=read('restored_ranking_evaluation/validation.json')
    assert density['overall_analysis_complete'] and density['ranking_seeds_evaluated']==50
    stats=read('density_statistics_validation.json');assert stats['status']=='PASS'
    # Compare historical completed cells, without rerunning P1-P4 or resampling.
    historical=subprocess.run(['git','show','7968d6c:sigir_analysis/statistical_tests.csv'],cwd=ROOT,
                              check=True,stdout=subprocess.PIPE).stdout.decode('utf-8-sig')
    reader=csv.DictReader(io.StringIO(historical));cols=reader.fieldnames
    old=[r for r in reader if r['primary_test_id'] not in {'P5','P6'}]
    with (OUT/'statistical_tests.csv').open(newline='',encoding='utf-8-sig') as stream:
        now=[{k:r[k] for k in cols} for r in csv.DictReader(stream) if r['primary_test_id'] not in {'P5','P6','P7','P8'}]
    assert old==now and len(now)==42
    assert hashlib.sha256(json.dumps(now,sort_keys=True,separators=(',',':')).encode()).hexdigest()==stats['preserved_completed_rows_sha256']
    stats.update(preserved_completed_original_columns=cols,compared_source_commit='7968d6c',
                 preserved_cells_compared=len(cols)*42,preserved_cell_mismatches=0)
    write('density_statistics_validation.json',stats)
    recovery=read('trec_ranking_recovery_validation.json')
    recovery.update(status='PASS',exact_rankings_recovered=True,hnsw_rankings_recovered=True,
                    metric_reproduction_pass=True,saved_metric_reproduction_verified=True,
                    queries_found=43,exact_query_rankings_verified=43,hnsw_query_budget_rankings_verified=430,
                    ef_values_found=g['ef_values_found'],doc_id_mapping_verified=True,
                    candidate_verification='Original exact and HNSW cached ranked indices recovered directly from IR_PROJECT.zip; FULL reproduction passed.',
                    source_hashes=gate['source_hashes'],source_paths=list(gate['source_hashes']),
                    notes=[g['original_run_provenance'],g['corpus_identity_limitation'],
                           'ANN scores reconstructed from original cached vector inner products without changing rank order; original ANN scores were not cached.',
                           'Historical workspace/Git-only missing-input audits are superseded by external archive recovery; no retrieval or model training was performed.'])
    write('trec_ranking_recovery_validation.json',recovery)
    state=read('trec_retrieval_state_audit.json')
    state.update(corpus_mapping_available=True,document_vectors_available=True,query_vectors_available=True,
                 exact_index_available=False,hnsw_index_available=True,normalization_verified=True,
                 metric_verified=True,corpus_identity_verified=True,original_corpus_order_available=True,
                 retrieval_only_rerun_possible=True,missing_components=[],
                 exact_index_note='IndexFlatIP was not serialized; original exact top100 indices and scores recovered directly.',
                 verification_scope='Original cache members in external archive; normalized original vectors verified during streaming recovery. HNSW index is archived but was not loaded or searched. Seven source/config/query/run files match byte-for-byte.',
                 artifact_location='~/Downloads/IR_PROJECT.zip::IR_PROJECT/PILOT/',
                 archive_identity_limitation=g['corpus_identity_limitation'],
                 encoder_revision_available=False,source_hashes=gate['source_hashes'],source_paths=list(gate['source_hashes']))
    write('trec_retrieval_state_audit.json',state)
    details=read('trec_retrieval_state.json')
    for item in details['cache_files'].values():
        item.update(present=(ROOT/item['path']).exists(),available_in_external_archive=True,
                    archive_member='IR_PROJECT/PILOT/'+item['path'])
    details.update(original_subset_membership_verifiable=True,retrieval_only_fallback_possible=True,
                   reason='Original ranked indices, ordered map and original vector/index caches recovered in external archive; direct ID recovery made retrieval unnecessary.',
                   original_independent_corpus_checksum_available=False,
                   identity_limitation=g['corpus_identity_limitation'],encoder_revision_available=False)
    write('trec_retrieval_state.json',details)
    status=read('ranking_recovery_status.json')
    status.update(status='COMPLETE',experiment_completed=True,ranking_seeds_evaluated=50,
                  reason='Direct original ranking recovery and FULL reproduction passed before all prepared seeds were scored.',
                  ANN_recall_invariance='PASS: same original rankings at every condition and seed',
                  unavailable_plots=[],density_outcome='A',query_seed_condition_budget_rows=64930)
    write('ranking_recovery_status.json',status)
    conclusion=dict(density_outcome='A',outcome_A_supported=True,outcome_B_supported=False,outcome_C_supported=False,
                    reason='Observed condition means show increasing measured ANN-minus-IR safe-budget gap under positive-judgment thinning with fixed rankings and work.',
                    scope='50 thinning seeds conditional on 43 fixed queries; FULL means full available pooled qrels, not complete relevance truth.',
                    interpretation='Apparent evaluation headroom, not retrieval improvement or actual relevance preservation. One-positive exact-zero fraction is 82.88%.',
                    universal_per_query_monotonicity_claimed=False,risk_matched_fixed_beats_current_cura=True,
                    post_hoc_diagnostic=True,adaptive_advantage_over_observed_fixed_frontier_demonstrated=False)
    write('scientific_conclusion.json',conclusion)
    results=read('analysis_results.json')
    results.update(judgment_density_status='COMPLETE',density_outcome='A',
                   density_evidence='trec_judgment_density_summary.csv',historical_estimates_recomputed=False)
    write('analysis_results.json',results)
    delta=read('recovery_delta_20261116.json')
    if 'initial_metadata_scan_result' not in delta:
        keys=['exact_rankings_recovered','hnsw_rankings_recovered','faithful_reconstruction_gate','full_qrel_reproduction','density_experiment']
        delta['initial_metadata_scan_result']={k:delta.get(k) for k in keys if k in delta}
    delta['initial_metadata_scan_result'].setdefault('HNSW_rankings_recovered',False)
    delta['initial_metadata_scan_result'].setdefault('density_seeds_scored',0)
    delta.update(current_status='RECOVERED; FULL reproduction PASS; 50-seed study COMPLETE',
                 exact_rankings_recovered=True,hnsw_rankings_recovered=True,HNSW_rankings_recovered=True,density_seeds_scored=50,
                 subsequent_verified_recovery='trec_ranking_provenance.json',
                 scan_interpretation='Initial generic table-schema scan did not accept ranked arrays. Dedicated provenance verification subsequently recovered original exact NPY and HNSW NPZ indices in IR_PROJECT.zip.',
                 limitations='Some Temp directories inaccessible; 19 unrelated parquet streams could not be inspected by the legacy non-seekable reader. No whole-disk completeness claim; these limitations do not affect the recovered archive gate.')
    for k in list(delta):
        if k in {'faithful_reconstruction_gate','full_qrel_reproduction','density_experiment'}:
            delta[k]='SUPERSEDED: see current_status and subsequent_verified_recovery'
    write('recovery_delta_20261116.json',delta)
    pipeline=read('evaluation_pipeline_validation.json')
    pipeline.update(record_scope='Historical synthetic implementation preflight before external archive recovery; not current real-study status.',
                    current_real_evidence='restored_ranking_evaluation/validation.json')
    write('evaluation_pipeline_validation.json',pipeline)
    papergate=read('short_paper_evidence_gate.json')
    papergate.setdefault('pre_density_paper_sha256',papergate['current_paper_sha256'])
    papergate.update(current_paper_sha256=sha(ROOT/'paper/sigir_short.tex'),evidence_complete=True,
                     scientific_claims_changed=True,density_claims_permitted=True,
                     policy='Density claims permitted only after original-ranking FULL reproduction and completed 50-seed same-ranking study.',
                     maintenance_change='Updated only evidence-dependent density, provenance, uncertainty and completion wording.',
                     manuscript_scope='Preserved prior saved-run findings and negative fixed-frontier evidence; added conditional thinning Outcome A.',
                     manuscript_compilation='NOT RUN; no LaTeX compiler available')
    write('short_paper_evidence_gate.json',papergate)
    statstatus=read('statistical_analysis_status.json')
    statstatus['multiplicity']='Historical comparisons retain original exploratory scope; new 12 density comparisons have Holm-adjusted p-values and unadjusted CIs.'
    write('statistical_analysis_status.json',statstatus)
    print('Completion metadata reconciled; 42 historical statistical rows unchanged; no experiments rerun.')


if __name__=='__main__':main()
