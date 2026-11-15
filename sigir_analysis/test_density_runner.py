"""Synthetic protocol and blocked-run checks; these are not TREC observations."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

try:
    from . import run_trec_judgment_density as runner
except ImportError:
    import run_trec_judgment_density as runner


class DensityRunnerTests(unittest.TestCase):
    def test_nested_graded_thinning_is_deterministic(self):
        qrels={'synthetic-q':{'a':3,'b':1,'c':2,'d':1,'e':2,'zero':0}}
        first=runner.thinned_qrels(qrels,17)
        self.assertEqual(first,runner.thinned_qrels(qrels,17))
        sizes=[];previous=set(qrels['synthetic-q'])
        for condition in runner.CONDITIONS[1:]:
            current=first[condition]['synthetic-q']
            self.assertLessEqual(set(current),previous)
            self.assertEqual(current['zero'],0)
            for doc,grade in current.items():self.assertEqual(grade,qrels['synthetic-q'][doc])
            sizes.append(sum(g>0 for g in current.values()));previous=set(current)
        self.assertEqual(sizes,[3,2,1])
        self.assertEqual(sum(g>0 for g in qrels['synthetic-q'].values()),5)

    def test_missing_inputs_write_no_experiment_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            sentinel=directory/'saved_result.csv';sentinel.write_text('preserve\n',encoding='utf-8')
            digest=hashlib.sha256(sentinel.read_bytes()).hexdigest()
            args=['--exact',str(directory/'missing_exact.csv'),'--hnsw',str(directory/'missing_hnsw.csv'),
                  '--provenance',str(directory/'missing_provenance.json'),'--output-dir',str(directory/'out'),'--preflight']
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):code=runner.main(args)
            result=json.loads(stream.getvalue())
            self.assertEqual(code,2);self.assertEqual(result['status'],'BLOCKED')
            self.assertFalse(result['experiment_completed']);self.assertFalse(result['experiment_outputs_written'])
            self.assertEqual(len(result['missing']),3)
            self.assertEqual(list(directory.iterdir()),[sentinel])
            self.assertEqual(hashlib.sha256(sentinel.read_bytes()).hexdigest(),digest)

    def test_invalid_provenance_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for name in ['exact.csv','hnsw.csv','provenance.json']:(directory/name).write_text('{}',encoding='utf-8')
            # Isolated failure fixture exercises the CLI rejection path without
            # constructing or accepting synthetic rankings as a real run.
            import sys
            import types
            module=types.ModuleType('ranking_input_gate')
            def reject(*args):raise ValueError('synthetic provenance rejection')
            module.load_and_validate_rankings=reject
            stream=io.StringIO()
            args=['--exact',str(directory/'exact.csv'),'--hnsw',str(directory/'hnsw.csv'),
                  '--provenance',str(directory/'provenance.json'),'--output-dir',str(directory/'out'),'--preflight']
            with patch.dict(sys.modules,{'ranking_input_gate':module}),contextlib.redirect_stdout(stream):
                code=runner.main(args)
            result=json.loads(stream.getvalue())
            self.assertEqual(code,2);self.assertEqual(result['status'],'BLOCKED_OR_REJECTED')
            self.assertFalse((directory/'out').exists())

    def test_isolated_synthetic_bundle_runs_all_fifty_seeds(self):
        # This tests orchestration against an explicit gate double. The real
        # provenance gate is tested separately. Every output lives in a deleted
        # temporary synthetic workspace, never in the CURA research artifacts.
        import ranking_input_gate
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='synthetic_density_test_') as temporary:
            root=Path(temporary);out=root/'sigir_analysis';out.mkdir()
            base=root/'msmarco_large/msmarco_1000000_q1000'
            for sub in ['results','research_grade_validation','cura_controller']:(base/sub).mkdir(parents=True)
            keys=[f'synthetic::{i}' for i in range(43)]
            documents=[f'synthetic-doc-{i}' for i in range(10)]
            queries=pd.DataFrame([dict(query_key=key,raw_query_id=f'q{i}',source='trec-dl-2019') for i,key in enumerate(keys)])
            queries.to_csv(base/'results/queries.csv',index=False)
            qrels_path=root/'synthetic_qrels.txt'
            qrels_path.write_text(''.join(f'q{i} 0 {d} {3 if j<5 else 0}\n' for i in range(43) for j,d in enumerate(documents)),encoding='utf-8')
            (out/'qrels_recovery.json').write_text(json.dumps({'local_path':'synthetic_qrels.txt'}),encoding='utf-8')
            work=pd.DataFrame([dict(query_key=key,source='trec-dl-2019',efSearch=ef,distance_computations=100+ef,mean_latency_ms=.01*ef) for key in keys for ef in runner.EFS])
            work.to_csv(base/'research_grade_validation/top10_query_budget_results.csv',index=False)
            choices=pd.DataFrame([dict(query_key=key,efSearch=16,distance_computations=116.,mean_latency_ms=.16,
                probe_ndis=184.,probe_latency_ms=.14,effective_distance_computations=300.,effective_latency_ms=.3) for key in keys])
            selection_path=base/'cura_controller/selected_cura_full_trec_ood.csv'
            choices.to_csv(selection_path,index=False)
            archived_paths=[selection_path,base/'research_grade_validation/top10_query_budget_results.csv']
            (out/'analysis_protocol.json').write_text(json.dumps(dict(input_hashes={
                p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in archived_paths})),encoding='utf-8')
            gate=dict(exact={key:documents[:] for key in keys},hnsw={(key,ef):documents[:] for key in keys for ef in runner.EFS},
                source_hashes={},ranking_hashes={'synthetic_fixture':'NOT_AN_EXPERIMENT'},
                reproduction=pd.DataFrame({'synthetic_fixture':['gate double only']}),
                validation={'synthetic_fixture_only':True})
            evidence=out/'synthetic_evidence'
            with patch.object(ranking_input_gate,'load_and_validate_rankings',return_value=gate):
                status=runner.evaluate_restored(root,out/'synthetic_exact.csv',out/'synthetic_hnsw.csv',out/'synthetic_provenance.json',evidence)
            self.assertTrue(status['experiment_completed'])
            self.assertTrue(status['statistical_analysis_complete'])
            per_seed=pd.read_csv(evidence/'trec_judgment_density_per_seed.csv')
            self.assertEqual(len(per_seed),151);self.assertTrue(per_seed.evaluation_completed.all())
            self.assertEqual(len(pd.read_csv(evidence/'trec_judgment_density_per_query.csv')),43*10*151)
            self.assertEqual(len(pd.read_csv(evidence/'trec_fixed_risk_work_by_density.csv')),10*151)
            self.assertEqual(set(pd.read_csv(evidence/'trec_cura_by_density.csv').mean_work),{300.})
            tests=pd.read_csv(evidence/'density_statistical_tests.csv')
            self.assertEqual(len(tests),6);self.assertTrue((tests.mean_difference==0).all())
            self.assertTrue((tests.thinning_seeds_evaluated==50).all())
            plots=list(evidence.glob('*.png'));self.assertEqual(len(plots),4)
            for path in plots:
                with Image.open(path) as img:img.verify()


if __name__=='__main__':unittest.main()
