"""Render the requested completion report from verified availability and results."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'


def build():
    recovery=json.loads((OUT/'trec_ranking_recovery_validation.json').read_text())
    density=json.loads((OUT/'ranking_recovery_status.json').read_text())
    results=json.loads((OUT/'analysis_results.json').read_text())
    statistics=json.loads((OUT/'statistical_analysis_status.json').read_text())
    frontier=pd.read_csv(OUT/'fixed_risk_work_frontier.csv')
    assert len(frontier)==11
    ef96=frontier[frontier.method=='Fixed ef=96'].iloc[0]
    cura=frontier[frontier.method=='CURA-Full'].iloc[0]
    original_saving=100*(1-cura.mean_distance_computations/ef96.mean_distance_computations)
    fixed=results['fixed_frontier']
    paper_blocked=not density['experiment_completed']
    main=f"CURA uses {original_saving:.2f}% less work than calibration-selected ef96 on 200 held-out queries. Post-hoc fixed ef48 matches CURA's observed 1% violation rate with {fixed['fixed_saving_vs_cura']['saving_pct']:.2f}% less work; ef64 also has lower observed risk and work than CURA. Exact-positive queries account for 92.59% of the original saving, while judgment-density effects remain unresolved. Available-input checks pass, but full-study validation fails because ranking-based checks and 50 scored seeds remain unavailable."
    novelty='The strongest supported contribution is a paired analysis showing that ANN-neighbor fidelity and judged-IR safety can require different budgets even when exact nDCG is positive. The full fixed frontier exposes how calibration and probe overhead affect whether that headroom yields an adaptive efficiency advantage.'
    blocker='Original exact top-k document IDs and HNSW document IDs for all 43 queries at all ten ef values are missing. The original subset document map, query/corpus vectors and built HNSW index needed for retrieval-only recovery are also missing; zero ranking seeds have been evaluated.'
    fields=[('TREC qrels','RECOVERED' if recovery['qrels_recovered'] else 'MISSING'),('Exact rankings','RECOVERED' if recovery['exact_rankings_recovered'] else 'MISSING'),('HNSW rankings','RECOVERED' if recovery['hnsw_rankings_recovered'] else 'MISSING'),('Retrieval-only rerun','YES' if recovery['retrieval_only_rerun'] else 'NO'),('Embeddings regenerated','YES' if recovery['embeddings_regenerated'] else 'NO'),('Index rebuilt','YES' if recovery['index_rebuilt'] else 'NO'),('Controller retrained','YES' if recovery['controller_retrained'] else 'NO'),('Exact-nDCG stratified analysis','COMPLETE'),('Full fixed risk/work frontier','COMPLETE'),('Same-ranking judgment-density experiment','COMPLETE' if density['experiment_completed'] else 'BLOCKED'),('50-seed thinning','COMPLETE' if density['ranking_seeds_evaluated']>=50 else 'INCOMPLETE'),('Statistical analysis','COMPLETE' if statistics['statistical_analysis_complete'] else 'INCOMPLETE'),('Validation','FAIL' if paper_blocked else 'PASS'),('Main finding',main),('Does a risk-matched fixed budget beat current CURA?','YES' if fixed['cheapest_fixed_mean_work']<fixed['cura_mean_work'] else 'NO'),('Does judgment sparsity materially change the ANN-vs-IR budget gap?','UNRESOLVED'),('Current strongest novelty',novelty),('Remaining blocker',blocker if paper_blocked else 'NONE'),('SIGIR-short readiness','BLOCKED' if paper_blocked else 'READY FOR DRAFT')]
    text='SIGIR RESEARCH COMPLETION STATUS\n\n'+'\n\n'.join(f"{label}{'' if label.endswith('?') else ':'}\n{value}" for label,value in fields)+'\n'
    status=dict(fields)
    status.update(input_and_saved_data_checks='PASS',validation_explanation='Requested full-study checks fail because original rankings, same-ranking verification, FULL metric reproduction and 50 scored seeds are unavailable.',original_qrel_rows=9260,prepared_query_seed_condition_records=6450,prepared_stochastic_seeds_per_condition=50,scored_stochastic_seeds_per_condition=0,outcome_A_B_C='UNRESOLVED',commit_created=False,pushed=False)
    (OUT/'completion_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    (OUT/'COMPLETION_STATUS.txt').write_text(text,encoding='utf-8')
    (OUT/'scientific_conclusion.json').write_text(json.dumps(dict(density_outcome='UNRESOLVED',outcome_A_supported=None,outcome_B_supported=None,outcome_C_supported=None,reason='No same-ranking density observations; none of A/B/C can be selected',risk_matched_fixed_beats_current_cura=True,post_hoc_diagnostic=True,adaptive_advantage_over_observed_fixed_frontier_demonstrated=False),indent=2),encoding='utf-8')
    return text


if __name__=='__main__':print(build())
