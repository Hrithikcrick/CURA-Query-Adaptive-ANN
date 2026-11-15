"""Evidence-preserving FiQA manual-review checkpoint, never automatic verification.

Initialize/prepare the next case: python 24_human_course_review.py --prepare-next
After an actual category answer: --record-category QUERY_ID --answer Y|1..9
After an actual note answer: --accept-note QUERY_ID, or --replace-note QUERY_ID
--note-file PATH. With --single-answer, the researcher's category response also
retains the displayed evidence note, as explicitly requested in the latest chat.
Without that flag the older separate note-response workflow remains available. Existing
complete researcher reviews are preserved with their original provenance.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'beir_fiqa/manual_failure_review'
SOURCE = FOLDER / 'fiqa_50_actual_ef10_failures_TO_REVIEW.csv'
TARGET = FOLDER / 'fiqa_50_actual_failures_HUMAN_VERIFIED.csv'
LOG = FOLDER / 'fiqa_50_human_review_decisions.json'
CATEGORIES = ('HIGH_ANN_RECALL_BUT_IR_FAILURE', 'LOW_ANN_RECALL_AND_IR_FAILURE',
              'RELEVANT_DOCUMENT_DROPPED', 'RANKING_BOUNDARY_INSTABILITY', 'AMBIGUOUS_QUERY',
              'RARE_OR_SPECIALIZED_ENTITY', 'LONG_OR_COMPLEX_QUERY', 'DISTRIBUTION_OR_DOMAIN_EFFECT', 'OTHER')
EXTRA = ('measured_failure','suggested_category','human_category','suggested_note','human_note',
         'human_verified','review_timestamp_or_sequence','suggestion_confidence','review_status',
         'source_worksheet_sha256')
LEGACY_FIELDS = ('human_failure_category','human_analysis_of_what_went_wrong',
                 'human_evidence_doc_ids_or_ranks','human_suggested_remedy','reviewer')
LEGACY_MAP = {
    'Judged-relevant document missed by approximate search': CATEGORIES[2],
    'Relevant document ranked too low by approximate search': CATEGORIES[2],
    'Ambiguous query or specialized financial terminology': CATEGORIES[8],
    'Near-duplicate or competing passages': CATEGORIES[8],
    'Relevance-judgment or evaluation limitation': CATEGORIES[8],
    'Approximate index/search-path failure': CATEGORIES[1],
    'Other (describe in analysis)': CATEGORIES[8],
}
CASE2_NOTE = ('Exact search ranks judged-relevant document 538005 first, but it is absent from the displayed HNSW ef=10 top five, '
              'whose displayed relevance labels are all 0 (no positive qrel recorded). Recorded nDCG@10 falls from 1.0 to 0.0; ANN Recall@10 is 0.2. '
              'The evidence shows lost relevant retrieval but does not isolate its search-path cause.')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if not fields or len(fields) != len(set(fields)) or any(None in r or None in r.values() for r in rows):
        raise ValueError(f'Malformed CSV: {path}')
    return fields, rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path, content):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(content.encode('utf-8'))
    temporary.replace(path)


def save(fields, rows, log):
    # Commit evidence first: a completed CSV row can never precede its log event.
    atomic_text(LOG, json.dumps(log, indent=2) + '\n')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(TARGET, buffer.getvalue())
    if read_csv(TARGET) != (list(fields), rows):
        raise ValueError('Checkpoint round-trip mismatch')


def load():
    source_fields, source = read_csv(SOURCE)
    source_hash = sha(SOURCE)
    if len(source) != 50 or len({r['query_id'] for r in source}) != 50:
        raise ValueError('Expected exactly 50 unique source failures')
    if not all(Decimal(r['saved_nDCG_loss']) > Decimal('0.025') for r in source):
        raise ValueError('Source contains a nonfailure')
    fields = [*source_fields, *EXTRA]
    if TARGET.exists():
        actual_fields, rows = read_csv(TARGET)
        log = json.loads(LOG.read_text(encoding='utf-8'))
        if actual_fields != fields or len(rows) != 50 or log['source_worksheet_sha256'] != source_hash:
            raise ValueError('Checkpoint source/schema changed')
        if any(any(a[f] != b[f] for f in source_fields) for a, b in zip(source, rows)):
            raise ValueError('Source evidence or numeric strings changed')
    else:
        if LOG.exists():
            raise ValueError('Log exists without checkpoint; inspect before recovery')
        rows, events = [], []
        for original in source:
            row = dict(original, **dict.fromkeys(EXTRA, ''))
            row.update(measured_failure='YES', human_verified='NO', review_status='WAITING_CATEGORY', source_worksheet_sha256=source_hash)
            if original['human_review_complete_YES_NO'].strip().upper() == 'YES':
                if not all(original[f].strip() for f in LEGACY_FIELDS) or original['human_failure_category'] not in LEGACY_MAP:
                    raise ValueError('Incomplete/unmapped legacy review requires inspection')
                category = LEGACY_MAP[original['human_failure_category']]
                sequence = 'LEGACY_SOURCE_CASE_' + original['case_id']
                row.update(human_category=category, human_note=original['human_analysis_of_what_went_wrong'],
                           human_verified='YES', review_timestamp_or_sequence=sequence, review_status='COMPLETE')
                events.append(dict(action='legacy_human_review_preserved', query_id=original['query_id'], human_category=category,
                                   human_note=row['human_note'], human_verified='YES', review_timestamp_or_sequence=sequence,
                                   source_worksheet_sha256=source_hash, original_human_failure_category=original['human_failure_category'],
                                   reviewer=original['reviewer'], original_evidence=original['human_evidence_doc_ids_or_ranks']))
            rows.append(row)
        log = dict(source_worksheet=SOURCE.relative_to(ROOT).as_posix(), source_worksheet_sha256=source_hash,
                   important='Actual fixed-HNSW ef=10 failures, not deployed CURA failures. Suggestions are model-assisted; only explicit researcher decisions count.',
                   category_mapping_note='Existing human category text is preserved; equivalent compact codes are recorded alongside it. No new decision is inferred.',
                   events=events)
        save(fields, rows, log)
    for row in rows:
        if row['human_verified'] == 'YES':
            matches = [e for e in log['events'] if e['query_id'] == row['query_id'] and e['action'] in {'legacy_human_review_preserved','human_review_completed'}]
            if not matches or matches[-1]['human_category'] != row['human_category'] or matches[-1]['human_note'] != row['human_note']:
                raise ValueError('Verified row lacks matching explicit decision evidence')
    return fields, rows, log


def pending(rows):
    return next((r for r in rows if r['human_verified'] != 'YES'), None)


def prepare(row):
    if row['suggested_category']:
        return
    if row['query_id'] == '1451':
        row.update(suggested_category=CATEGORIES[2], suggested_note=CASE2_NOTE, suggestion_confidence='HIGH')
        return
    recall = Decimal(row['saved_ANN_Recall10'])
    relevant = re.findall(r'\[(\d+)\] relevance=([1-9][0-9.]*)', row['exact_top5_passages'])
    approx_ids = set(re.findall(r'\[(\d+)\]', row['HNSW_ef10_top5_passages']))
    missing = [doc for doc, _ in relevant if doc not in approx_ids]
    if recall >= Decimal('0.90'):
        category = CATEGORIES[0]
    elif missing:
        category = CATEGORIES[2]
    else:
        category = CATEGORIES[1] if recall < Decimal('0.90') else CATEGORIES[8]
    note = (f"Fixed HNSW ef=10 records ANN Recall@10={row['saved_ANN_Recall10']} and nDCG@10={row['saved_HNSW_nDCG10']}, "
            f"compared with exact nDCG@10={row['saved_exact_nDCG10']} (loss {row['saved_nDCG_loss']}). ")
    if missing:
        note += 'Judged-relevant document(s) ' + ', '.join(missing) + ' appear in the exact top five but not the displayed approximate top five. '
    note += 'These observations do not isolate the search-path cause.'
    row.update(suggested_category=category, suggested_note=note, suggestion_confidence='HIGH' if missing or recall >= Decimal('0.90') else 'MEDIUM')


def finalize(rows, log):
    if not all(r['human_verified'] == 'YES' and r['human_category'] in CATEGORIES and r['human_note'].strip() for r in rows):
        raise ValueError('Human taxonomy is gated: all 50 category and note decisions must be complete')
    counts = Counter(r['human_category'] for r in rows)
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('category','failure_count','share_percent'))
    for category in sorted(CATEGORIES, key=lambda c: (-counts[c], CATEGORIES.index(c))):
        writer.writerow((category, counts[category], f'{counts[category]*2:.2f}'))
    atomic_text(ROOT/'reports/HUMAN_50_FAILURE_TAXONOMY.csv', output.getvalue())
    lines = ['# Human-verified FiQA failure analysis', '', '50 actual fixed-HNSW low-budget IR failures were manually analyzed.',
             'These are fixed efSearch=10 outcomes, not 50 deployed CURA-policy failures. Explicit category/note decisions and preserved prior reviews are recorded in the decision log.', '', '| Category | Count | Share |', '|---|---:|---:|']
    lines += [f'| {c} | {counts[c]} | {counts[c]*2:.2f}% |' for c in sorted(CATEGORIES, key=lambda c: (-counts[c], CATEGORIES.index(c)))]
    lines += ['', '## Representative evidence', '']
    for category in sorted(counts, key=lambda c: (-counts[c], CATEGORIES.index(c))):
        row = next(r for r in rows if r['human_category']==category)
        lines.append(f"- {category}: query {row['query_id']} ({row['query_text']}). Researcher note: {row['human_note']}")
    high = sum(Decimal(r['saved_ANN_Recall10']) >= Decimal('0.90') for r in rows)
    lines += ['', f'{high}/50 selected failures have ANN Recall@10 >=0.90; {50-high}/50 have lower recall. These deliberately selected cases show coexistence patterns, not population prevalence or policy failure rates.',
              'Coding is qualitative and may admit competing labels. Truncated passage evidence and sparse relevance judgments limit causal interpretation; no inter-annotator agreement or causal effect is inferred.']
    atomic_text(ROOT/'reports/HUMAN_50_FAILURE_ANALYSIS.md', '\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-next', action='store_true')
    parser.add_argument('--record-category', metavar='QUERY_ID')
    parser.add_argument('--answer', choices=('Y','1','2','3','4','5','6','7','8','9'))
    parser.add_argument('--single-answer', action='store_true', help='After an actual researcher category response, retain the displayed evidence note and complete this case immediately')
    parser.add_argument('--accept-note', metavar='QUERY_ID')
    parser.add_argument('--replace-note', metavar='QUERY_ID')
    parser.add_argument('--note-file', type=Path)
    parser.add_argument('--finalize', action='store_true')
    args=parser.parse_args()
    if args.single_answer and not args.record_category:
        parser.error('--single-answer requires an actual --record-category response')
    fields, rows, log=load()
    row=pending(rows)
    if args.prepare_next and row:
        prepare(row)
        save(fields, rows, log)
    if args.record_category:
        if row is None or row['query_id']!=args.record_category or not args.answer or not row['suggested_category']:
            raise ValueError('An explicit answer to the currently displayed case is required')
        category=row['suggested_category'] if args.answer=='Y' else CATEGORIES[int(args.answer)-1]
        row.update(human_category=category, review_status='WAITING_NOTE')
        log['events'].append(dict(action='human_category_decision', query_id=row['query_id'], category_answer=args.answer,
                                  human_category=category, human_verified='NO', source='researcher_chat'))
        save(fields, rows, log)
    note_query=args.accept_note or args.replace_note or (args.record_category if args.single_answer else None)
    if note_query:
        if row is None or row['query_id']!=note_query or row['review_status']!='WAITING_NOTE':
            raise ValueError('A category decision must precede the researcher note response')
        note=row['suggested_note'] if args.accept_note or args.single_answer else args.note_file.read_text(encoding='utf-8-sig').strip()
        if not note:
            raise ValueError('Human note must be nonempty')
        previous=[e for e in log['events'] if e['query_id']==row['query_id'] and e['action']=='human_category_decision'][-1]
        sequence='CHAT_REVIEW_' + str(1+sum(e['action']=='human_review_completed' for e in log['events']))
        row.update(human_note=note, human_verified='YES', review_timestamp_or_sequence=sequence, review_status='COMPLETE')
        log['events'].append(dict(action='human_review_completed', query_id=row['query_id'], human_category=row['human_category'],
                                  human_note=note, human_verified='YES', review_timestamp_or_sequence=sequence,
                                  category_answer=previous['category_answer'],
                                  note_answer='RETAIN_EVIDENCE_NOTE_SINGLE_ACCEPTANCE' if args.single_answer else ('ACCEPT_SUGGESTED_NOTE' if args.accept_note else 'REPLACEMENT_NOTE'),
                                  review_workflow='SINGLE_ACCEPTANCE' if args.single_answer else 'CATEGORY_AND_NOTE_RESPONSES',
                                  source='researcher_chat', source_worksheet_sha256=sha(SOURCE)))
        save(fields, rows, log)
    if args.finalize:
        finalize(rows, log)
    print(f"Human verified: {sum(r['human_verified']=='YES' for r in rows)} / 50")
    row=pending(rows)
    if row:
        print(json.dumps(row, indent=2))
    else:
        print('All 50 human decisions are recorded.')


if __name__=='__main__':
    main()
