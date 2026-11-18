"""Audit genuine FiQA review completion without rerunning scientific analyses."""
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
BASELINE='82ed57f54976131064f2606cc13ad41b81aa5cb6'
SOURCE_SHA='2936cf021358632d3117ad1c1907d21e2c975468120b635559677251492113e7'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def rows(raw):return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline='')))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def audit():
    spec=importlib.util.spec_from_file_location('human_review',ROOT/'24_human_course_review.py')
    review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
    fields,current,log=review.load()
    assert len(current)==50 and len({r['query_id'] for r in current})==50
    assert all(r['human_verified']=='YES' and r['review_status']=='COMPLETE'
               and r['human_category'] in review.CATEGORIES and r['human_note'].strip() for r in current)
    assert sha(review.SOURCE.read_bytes())==SOURCE_SHA
    source=rows(review.SOURCE.read_bytes())
    assert all(all(r[k]==v for k,v in original.items()) for r,original in zip(current,source))
    checkpoint=review.TARGET.relative_to(ROOT).as_posix()
    decision_log=review.LOG.relative_to(ROOT).as_posix()
    before=rows(git('show',BASELINE+':'+checkpoint))
    before_log=json.loads(git('show',BASELINE+':'+decision_log))
    done_before=[r for r in before if r['human_verified']=='YES']
    assert len(done_before)==36
    by_id={r['query_id']:r for r in current}
    assert all(by_id[r['query_id']]==r for r in done_before)
    assert log['events'][:len(before_log['events'])]==before_log['events']
    old_ids={r['query_id'] for r in done_before}
    new_events=[e for e in log['events'][len(before_log['events']):] if e['action']=='human_review_completed']
    new_ids={r['query_id'] for r in current if r['query_id'] not in old_ids}
    assert len(new_ids)==len(new_events)==14 and {e['query_id'] for e in new_events}==new_ids
    assert [r['case_id'] for r in current if r['query_id'] in new_ids]==[str(i) for i in range(37,51)]
    for r in current:
        evidence=[e for e in log['events'] if e.get('query_id')==r['query_id']
                  and e['action'] in {'human_review_completed','legacy_human_review_preserved'}]
        assert len(evidence)==1
        event=evidence[0]
        assert event['human_category']==r['human_category'] and event['human_note']==r['human_note']
        assert event['review_timestamp_or_sequence']==r['review_timestamp_or_sequence']
        if r['query_id'] in new_ids:
            assert event['source']=='researcher_chat' and event['source_worksheet_sha256']==SOURCE_SHA
            assert event['note_answer'] in {'ACCEPT_SUGGESTED_NOTE','REPLACEMENT_NOTE','RETAIN_EVIDENCE_NOTE_SINGLE_ACCEPTANCE'}
    taxonomy_path=ROOT/'reports/HUMAN_50_FAILURE_TAXONOMY.csv'
    analysis_path=ROOT/'reports/HUMAN_50_FAILURE_ANALYSIS.md'
    taxonomy=rows(taxonomy_path.read_bytes())
    counts=Counter(r['human_category'] for r in current)
    assert len(taxonomy)==9 and {r['category'] for r in taxonomy}==set(review.CATEGORIES)
    assert sum(int(r['failure_count']) for r in taxonomy)==50
    assert sum(Decimal(r['share_percent']) for r in taxonomy)==Decimal('100')
    for r in taxonomy:
        assert int(r['failure_count'])==counts[r['category']]
        assert Decimal(r['share_percent'])==Decimal(2*counts[r['category']])
    assert analysis_path.stat().st_size>0
    status=json.loads((ROOT/'reports/50_CASE_FAILURE_REVIEW_STATUS.json').read_text(encoding='utf-8'))
    assert (status['human_verified'],status['completed'],status['remaining'])==(50,50,0)
    assert status['already_completed_before_this_run']==36 and status['newly_completed']==14
    assert status['first_unfinished_case'] is None and status['status']=='COMPLETE'
    assert status['checkpoint_sha256']==sha(review.TARGET.read_bytes())
    assert status['decision_log_sha256']==sha(review.LOG.read_bytes())
    assert rows((ROOT/status['assistance_csv']).read_bytes())==[]
    # Compare stored scientific tables/figures with the pre-review commit.
    # Text normalization accounts only for Git's documented checkout conversion.
    paths=git('ls-tree','-r','--name-only',BASELINE,'--','sigir_analysis').decode().splitlines()
    protected={}
    for name in paths:
        if Path(name).suffix not in {'.csv','.png','.gz'}:continue
        current_bytes=(ROOT/name).read_bytes();previous_bytes=git('show',BASELINE+':'+name)
        normalize=lambda b:b.replace(b'\r\n',b'\n') if name.endswith('.csv') else b
        assert normalize(current_bytes)==normalize(previous_bytes),name
        protected[name]=sha(current_bytes)
    # Preserve the separate model-assisted MS MARCO/TREC set as model assisted.
    separate=ROOT/'msmarco_large/msmarco_1000000_q1000/actual_50_failure_analysis/actual_50_ir_failures_final.csv'
    separate_rows=rows(separate.read_bytes())
    assert len(separate_rows)==50 and not any(r.get('human_verified','').upper()=='YES' for r in separate_rows)
    result=dict(status='PASS',total_cases=50,previously_human_verified=36,newly_human_verified=14,
                human_verified_total=50,remaining=0,cases_processed_in_this_run='37-50',
                final_taxonomy_generated=True,final_analysis_generated=True,
                decision_log_validation='PASS',checkpoint_validation='PASS',
                original_evidence_unchanged='PASS',previous_completed_cases_preserved='1-36',
                original_log_events_preserved=len(before_log['events']),
                new_explicit_researcher_completion_events=len(new_events),
                original_source_sha256=SOURCE_SHA,checkpoint_sha256=sha(review.TARGET.read_bytes()),
                decision_log_sha256=sha(review.LOG.read_bytes()),
                taxonomy_sha256=sha(taxonomy_path.read_bytes()),analysis_sha256=sha(analysis_path.read_bytes()),
                taxonomy_counts=dict(counts),taxonomy_total=50,taxonomy_percentage_total='100.00',
                preserved_baseline_commit=BASELINE,scientific_SIGIR_results_recomputed=False,
                scientific_artifacts_unchanged=protected,separate_MS_MARCO_TREC_human_verified=0,
                methodology='Model-prepared evidence and initial suggestions; explicit researcher category/note confirmations, preserving earlier genuine decisions. No independent second annotator or causal search-path identification.',
                final_status='COMPLETE')
    (ROOT/'reports/HUMAN_50_REVIEW_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','previously_human_verified','newly_human_verified','human_verified_total','remaining','taxonomy_counts','scientific_SIGIR_results_recomputed']},indent=2))
    return result


if __name__=='__main__':audit()
