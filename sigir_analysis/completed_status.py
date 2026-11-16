"""Render current status from completed evidence, without recomputing analyses."""
import json


def build_completed(root):
    out=root/'sigir_analysis'
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    validation=read('validation.json')
    assert validation['validation_pass'] and validation['density_experiment_completed']
    assert read('trec_ranking_recovery_validation.json')['metric_reproduction_pass']
    assert read('statistical_analysis_status.json')['statistical_analysis_complete']
    review=json.loads((root/'reports/50_CASE_FAILURE_REVIEW_STATUS.json').read_text())
    fields={
        'TREC qrels':'RECOVERED','Exact rankings':'RECOVERED','HNSW rankings':'RECOVERED',
        'Retrieval-only rerun':'NO','Embeddings regenerated':'NO','Index rebuilt':'NO','Controller retrained':'NO',
        'Previous completed analyses redone':'NO','Exact-nDCG stratified analysis':'COMPLETE',
        'Fixed risk/work frontier':'COMPLETE','Full-qrel reproduction':'PASS',
        'Same-ranking judgment-density study':'COMPLETE','50-seed judgment thinning':'COMPLETE',
        'Statistical analysis':'COMPLETE','Validation':'PASS','Risk-matched fixed beats current CURA':'YES',
        'Judgment sparsity materially changes ANN-vs-IR budget gap':'YES (Outcome A: apparent evaluation headroom)',
        'Scientific density outcome':'A',
        'Strongest positive result':'CURA saves 20.51% recorded distance work versus calibration-selected ef96; exact-positive queries contribute 92.59% of that saving. Fixed-ranking thinning increases measured budget headroom.',
        'Strongest negative result':'Post-hoc ef48 matches CURA observed 2/200 violations with 28.29% less work; CURA does not dominate the observed fixed frontier. Matched observed risk is not a population-risk guarantee.',
        'Uncertainty scope':'50 paired thinning seeds conditional on 43 queries; 12 follow-up Holm-adjusted p-values, unadjusted CIs. P7 is an algebraic reversal of P5. No query-population guarantee.',
        'Human failure reviews':f"{review['human_verified']}/50; {review['remaining']} require human confirmation",
        'Pending model-assisted suggestions':str(review['model_assisted']),
        'Remaining scientific blocker':'NONE','SIGIR-short readiness':'READY FOR DRAFT',
        'Remaining non-scientific work':'Human confirmation of case37 note and cases38-50; LaTeX compilation and official submission-format verification.'}
    text='SIGIR RESEARCH COMPLETION STATUS\n\n'+'\n\n'.join(f'{k}:\n{v}' for k,v in fields.items())+'\n'
    status=dict(fields)
    status.update(original_qrel_rows=9260,prepared_query_seed_condition_records=6450,
                  prepared_stochastic_seeds_per_condition=50,scored_stochastic_seeds_per_condition=50,
                  input_and_saved_data_checks='PASS',outcome_A_B_C='A',human_review_status=review['status'],
                  validation_explanation='Current evidence, preserved-input hashes, FULL reproduction, all 50 seeds and report sources validated; no retrieval or statistical resampling in this status command.',
                  git_synchronization='Verify using git fetch and compare HEAD with origin/main; status artifacts do not self-certify future pushes.')
    for name in ['COMPLETION_STATUS.txt','SIGIR_COMPLETION_STATUS.txt']:
        (out/name).write_text(text,encoding='utf-8')
    (out/'completion_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    return text
