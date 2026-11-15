"""Prepare only unfinished FiQA suggestions without modifying human decisions.

During a paused rebase, --snapshot-archive reads the verified pre-rebase copy.
After integration, omit that option to use the live checkpoint and decision log.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = 'beir_fiqa/manual_failure_review/'
CHECKPOINT = FOLDER + 'fiqa_50_actual_failures_HUMAN_VERIFIED.csv'
DECISIONS = FOLDER + 'fiqa_50_human_review_decisions.json'
SOURCE = FOLDER + 'fiqa_50_actual_ef10_failures_TO_REVIEW.csv'
MODEL_CASES = FOLDER + 'fiqa_pending_cases_MODEL_ASSISTED.csv'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def rows_from(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))


def prepare(snapshot=None):
    def read(name):
        if snapshot:
            with zipfile.ZipFile(snapshot) as archive:
                return archive.read(name)
        return (ROOT / name).read_bytes()

    raw_source = (ROOT / SOURCE).read_bytes()
    raw_checkpoint, raw_log = read(CHECKPOINT), read(DECISIONS)
    source, rows = rows_from(raw_source), rows_from(raw_checkpoint)
    log = json.loads(raw_log)
    assert len(source) == len(rows) == 50
    assert len({r['query_id'] for r in rows}) == 50
    assert log['source_worksheet_sha256'] == digest(raw_source)
    completed, pending = [], []
    for original, row in zip(source, rows):
        assert all(row[k] == v for k, v in original.items())
        assert row['source_worksheet_sha256'] == digest(raw_source)
        assert Decimal(row['saved_nDCG_loss']) > Decimal('0.025')
        if row['human_verified'] == 'YES':
            events = [e for e in log['events'] if e.get('query_id') == row['query_id']
                      and e['action'] in {'legacy_human_review_preserved', 'human_review_completed'}]
            assert events and events[-1]['human_category'] == row['human_category']
            assert events[-1]['human_note'] == row['human_note'] and row['human_note'].strip()
            if events[-1]['action'] == 'human_review_completed':
                assert events[-1]['source'] == 'researcher_chat'
                assert events[-1]['source_worksheet_sha256'] == digest(raw_source)
            completed.append(row)
        else:
            assert row['human_verified'] == 'NO'
            pending.append(row)

    assistance = []
    for row in pending:
        exact = re.findall(r'#(\d+) \[(\d+)\] relevance=([\d.]+)', row['exact_top5_passages'])
        approx = re.findall(r'#(\d+) \[(\d+)\] relevance=([\d.]+)', row['HNSW_ef10_top5_passages'])
        assert len(exact) == len(approx) == 5
        approx_ids = {doc for _, doc, _ in approx}
        missing = [(rank, doc) for rank, doc, rel in exact if Decimal(rel) > 0 and doc not in approx_ids]
        recall = Decimal(row['saved_ANN_Recall10'])
        category = ('HIGH_ANN_RECALL_BUT_IR_FAILURE' if recall >= Decimal('0.90')
                    else 'RELEVANT_DOCUMENT_DROPPED' if missing
                    else 'LOW_ANN_RECALL_AND_IR_FAILURE')
        note = (f"Recorded nDCG@10 decreases from exact {row['saved_exact_nDCG10']} to "
                f"approximate {row['saved_HNSW_nDCG10']} (loss {row['saved_nDCG_loss']}); "
                f"ANN Recall@10 is {row['saved_ANN_Recall10']}. ")
        if missing:
            note += '; '.join(f'Relevant document {doc} is at exact rank {rank} but absent from the displayed approximate top five'
                              for rank, doc in missing) + '. '
        else:
            note += 'The displayed top-five excerpts do not locate the lost or demoted judged-relevant result. '
        note += 'The underlying search-path cause is unverified.'
        carried = bool(row['suggested_category'])
        if carried:
            category, note = row['suggested_category'], row['suggested_note']
        assistance.append(dict(
            case_id=row['case_id'], query_id=row['query_id'], query_text=row['query_text'],
            source='FiQA', failed_method=row['failed_method'], efSearch='10',
            saved_ANN_Recall10=row['saved_ANN_Recall10'], saved_exact_nDCG10=row['saved_exact_nDCG10'],
            saved_HNSW_nDCG10=row['saved_HNSW_nDCG10'], saved_nDCG_loss=row['saved_nDCG_loss'],
            existing_human_category=row['human_category'],
            human_review_status_snapshot=row['review_status'], human_verified_snapshot='NO',
            model_suggested_category=category, model_suggested_evidence_note=note,
            suggestion_source='MODEL_ASSISTED',
            suggestion_action='PRESERVED_EXISTING_SUGGESTION' if carried else 'NEW_PENDING_CASE_SUGGESTION',
            needs_human_confirmation='YES', exact_top5_document_ids=' '.join(d for _, d, _ in exact),
            approximate_top5_document_ids=' '.join(d for _, d, _ in approx),
            boundary_gap='NOT_RECORDED', source_worksheet_sha256=digest(raw_source),
            checkpoint_sha256=digest(raw_checkpoint), decision_log_sha256=digest(raw_log)))
    if assistance:
        with (ROOT / MODEL_CASES).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(assistance[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(assistance)
        assert rows_from((ROOT / MODEL_CASES).read_bytes()) == assistance

    ms_path = 'msmarco_large/msmarco_1000000_q1000/actual_50_failure_analysis/actual_50_ir_failures_final.csv'
    ms_rows = rows_from(read(ms_path))
    assert len(ms_rows) == 50 and not any(r.get('human_verified', '').upper() == 'YES' for r in ms_rows)
    summary = dict(
        total_cases=50, already_completed_before_this_run=len(completed), newly_completed=0,
        completed=len(completed), remaining=len(pending), human_verified=len(completed),
        model_assisted=len(assistance), model_assisted_count_scope='pending FiQA suggestions; not completed human reviews',
        newly_prepared_suggestions=sum(r['suggestion_action'] == 'NEW_PENDING_CASE_SUGGESTION' for r in assistance),
        carried_forward_suggestions=sum(r['suggestion_action'] == 'PRESERVED_EXISTING_SUGGESTION' for r in assistance),
        existing_assistant_draft_cases_available=50, existing_assistant_draft_revalidated=False,
        first_unfinished_case=pending[0]['case_id'] if pending else None,
        first_unfinished_query_id=pending[0]['query_id'] if pending else None,
        first_unfinished_stage=pending[0]['review_status'] if pending else None,
        pending_human_stage_counts=dict(Counter(r['review_status'] for r in pending)),
        completed_case_numbers=[r['case_id'] for r in completed], completed_query_ids=[r['query_id'] for r in completed],
        status='NEEDS_HUMAN_CONFIRMATION' if pending else 'COMPLETE',
        separate_MS_MARCO_TREC_analysis=dict(total_cases=50, human_verified=0, model_assisted=50,
                                           verification_sources=dict(Counter(r.get('verification_source', '') for r in ms_rows))),
        source_worksheet=SOURCE, checkpoint=CHECKPOINT, human_decision_log=DECISIONS,
        source_worksheet_sha256=digest(raw_source), checkpoint_sha256=digest(raw_checkpoint),
        decision_log_sha256=digest(raw_log), assistance_csv=MODEL_CASES,
        checkpoint_read_from='pre-rebase safety archive' if snapshot else 'live workspace',
        checkpoint_preserved_commit='0193a05', human_decisions_modified=False,
        validation='PASS', interpretation=f'{len(completed)} completed human decisions and {len(pending)} pending model suggestions do not equal 50 human reviews.' if pending else 'All 50 human decisions have matching evidence.')
    out = ROOT / 'reports/50_CASE_FAILURE_REVIEW_STATUS.json'
    out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    lines = ['# Fifty-case failure review status', '',
             f"Required human reviews: **{len(completed)}/50 complete**, **{len(pending)} pending**.", '',
             f"First unfinished case: **{summary['first_unfinished_case']}**, query **{summary['first_unfinished_query_id']}**, stage **{summary['first_unfinished_stage']}**.",
             'Case 37 already has the researcher\'s category 2. Its displayed evidence note still needs confirmation. No note acceptance was inferred.' if pending and pending[0]['case_id']=='37' else 'No new human response was inferred.', '',
             '| Field | Value |', '|---|---:|', *[f'| {k} | {summary[k]} |' for k in
              ['total_cases', 'already_completed_before_this_run', 'newly_completed', 'remaining', 'human_verified', 'model_assisted', 'newly_prepared_suggestions', 'carried_forward_suggestions']], '',
             f"**Status: {summary['status']}.** Model-assisted counts refer to pending suggestions, not additional completed reviews.", '',
             f"Completed case numbers: {', '.join(summary['completed_case_numbers'])}.",
             f"Completed query IDs: {', '.join(summary['completed_query_ids'])}.", '',
             'The companion CSV prepares only cases 37–50: one suggestion carried forward exactly and 13 new suggestions from saved metrics and displayed document/rank evidence. It contains no new human approval. The original worksheet, checkpoint and proof log were not written.', '',
             'The older assistant draft remains available for all 50 cases; its content was not reanalyzed. The separate MS MARCO/TREC taxonomy remains 50 model-assisted labels and zero human verifications.', '',
             ('The checkpoint was read from the byte-verified pre-rebase archive. Commit 0193a05 preserves the same review records. Rerun against the live checkpoint after integration to verify these hashes.' if snapshot else 'The checkpoint was read and validated directly from the live workspace after integration. Its original human decisions and notes remain unchanged.'), '',
             f"Source worksheet SHA256: `{summary['source_worksheet_sha256']}`.",
             f"Checkpoint SHA256: `{summary['checkpoint_sha256']}`.",
             f"Decision log SHA256: `{summary['decision_log_sha256']}`.", '',
             'Evidence label 0 means no positive qrel was recorded in the displayed evidence; it does not establish that every passage was explicitly judged nonrelevant. Missing top-five entries do not by themselves establish absence at all ranks. No causal graph-search explanation was inferred.']
    (ROOT / 'reports/50_CASE_FAILURE_REVIEW_STATUS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert read(CHECKPOINT) == raw_checkpoint and read(DECISIONS) == raw_log
    assert (ROOT / SOURCE).read_bytes() == raw_source
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-archive', type=Path)
    args = parser.parse_args()
    result = prepare(args.snapshot_archive)
    print(json.dumps({k: result[k] for k in ['completed', 'remaining', 'human_verified', 'model_assisted', 'first_unfinished_case', 'first_unfinished_stage', 'status', 'validation']}, indent=2))
