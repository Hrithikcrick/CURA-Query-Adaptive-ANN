"""Validate evaluator implementation using isolated synthetic fixtures only.

Passing this script is not completion of any TREC retrieval or density study.
The real-input preflight is read-only and must leave all research outputs intact.
"""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    # Names deliberately limited to implementation tests. Fixtures live in
    # temporary directories; none is accepted as a real retrieval observation.
    loader=unittest.TestLoader()
    suite=unittest.TestSuite([
        loader.discover(str(OUT),pattern='test_density*.py'),
        loader.discover(str(OUT),pattern='test_ranking_input_gate.py')])
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    research_names=[
        'trec_exact_rankings.csv','trec_hnsw_rankings.csv',
        'trec_judgment_density_per_query.csv','trec_judgment_density_per_seed.csv',
        'trec_judgment_density_summary.csv','trec_fixed_risk_work_by_density.csv',
        'trec_cura_by_density.csv','trec_retrieval_reproduction.csv',
        'statistical_tests.csv','primary_statistical_results.csv',
        'judgment_density_vs_budget_gap.png','judgment_density_vs_zero_ndcg.png',
        'judgment_density_vs_ir_safe_budget.png','judgment_density_vs_risk_work.png']
    before={name:sha(OUT/name) for name in research_names}
    preflight=subprocess.run([sys.executable,str(OUT/'run_trec_judgment_density.py'),'--preflight'],
        cwd=ROOT,capture_output=True,text=True)
    after={name:sha(OUT/name) for name in research_names}
    preflight_output=json.loads(preflight.stdout)
    safe_preflight=(before==after and preflight.returncode in [0,2]
                    and preflight_output.get('status') in ['BLOCKED','BLOCKED_OR_REJECTED','PREFLIGHT_PASS'])
    source_names=[
        'ranking_input_gate.py','density_metrics.py','density_uncertainty.py',
        'run_trec_judgment_density.py','test_ranking_input_gate.py',
        'test_density_metrics.py','test_density_uncertainty.py','test_density_runner.py',
        'validate_evaluation_pipeline.py']
    status=dict(
        implementation_validation='PASS' if result.wasSuccessful() and safe_preflight else 'FAIL',
        synthetic_tests_run=result.testsRun,synthetic_test_failures=len(result.failures),
        synthetic_test_errors=len(result.errors),synthetic_test_skips=len(result.skipped),
        real_input_preflight_status=preflight_output['status'],real_input_preflight_exit_code=preflight.returncode,
        preflight_did_not_modify_research_outputs=before==after,
        actual_TREC_ranking_evaluations_completed=0,
        actual_thinning_seed_evaluations_completed=0,
        synthetic_fixture_results_are_research_observations=False,
        scientific_study_completion='UNCHANGED; consult validation.json and SIGIR_COMPLETION_STATUS.txt',
        no_retrieval_or_embedding_generation_or_index_rebuild_or_training=True,
        source_hashes={f'sigir_analysis/{name}':sha(OUT/name) for name in source_names},
        unittest_log='evaluation_pipeline_test_log.txt')
    (OUT/'evaluation_pipeline_test_log.txt').write_text(stream.getvalue(),encoding='utf-8')
    (OUT/'evaluation_pipeline_validation.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))
    return 0 if status['implementation_validation']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
