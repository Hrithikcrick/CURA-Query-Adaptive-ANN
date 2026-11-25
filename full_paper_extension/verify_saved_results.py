"""Verify the published saved-data summary; no retrieval or historical writes.

New verification wrapper, 2026-11-26. An explicit --output exclusively creates
a new summary; the default compares a fresh computation with the saved summary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from false_safety_audit import verify as verify_trec
from fiqa_oracle_robustness import verify as verify_fiqa

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def compute():
    trec = verify_trec(ROOT)
    fiqa = verify_fiqa(ROOT/'beir_fiqa/fiqa_query_budget_results.csv',
                       ROOT/'beir_fiqa/fiqa_oracle_budgets.csv')
    fiqa['inputs'] = {(Path(k).relative_to(ROOT) if Path(k).is_absolute() else Path(k)).as_posix(): v
                      for k, v in fiqa['inputs'].items()}
    # Historical FiQA files have no pinned checkout EOL rule. Preserve raw hash
    # provenance while checking text identity across legitimate LF/CRLF checkouts.
    fiqa['input_canonical_LF_sha256'] = {
        key: hashlib.sha256((ROOT/key).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for key in fiqa['inputs']}
    if trec['status'] != 'PASS' or fiqa['status'] != 'PASS':
        raise ValueError('A requested saved-data result failed reproduction')
    primary = next(x for x in trec['tolerance_sensitivity'] if x['tolerance'] == .025)
    for name, expected in [('50_PERCENT_POSITIVES', 27.43), ('25_PERCENT_POSITIVES', 50.14),
                           ('ONE_POSITIVE_PER_QUERY', 86.86)]:
        assert round(primary['conditions'][name]['conditional_false_safe_rate'] * 100, 2) == expected
    assert round(primary['conditions']['50_PERCENT_POSITIVES']['false_safe_positive_exact_fraction']*100, 1) == 95.3
    return {'status': 'PASS', 'verification_date': '2026-11-26',
            'provenance': 'Existing local research inputs; no named ZIP or original push-kit summary was found. This summary and focused verification utilities were created retrospectively on 2026-11-26.',
            'trec_false_safety': trec, 'fiqa_oracle_replication': fiqa}


def compare(actual, expected, where='summary'):
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'Key mismatch: {where}')
        if where == 'summary.fiqa_oracle_replication.inputs':
            # Canonical text hashes below are strict; raw byte hashes remain
            # measured provenance. No historical file is normalized or rewritten.
            return
        for key in actual:
            compare(actual[key], expected[key], f'{where}.{key}')
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError(f'Length mismatch: {where}')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{where}[{i}]')
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError(f'Numeric mismatch: {where}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'Value mismatch: {where}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Create new JSON summary; refuses overwrite')
    args = parser.parse_args()
    actual = compute()
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(actual, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        expected = json.loads((HERE/'VERIFIED_SAVED_DATA_RESULTS.json').read_text(encoding='utf-8'))
        # Software versions are recorded provenance, not a restriction on another environment.
        actual['fiqa_oracle_replication']['software'] = expected['fiqa_oracle_replication']['software']
        compare(actual, expected)
    print('PASS: TREC false-safety and FiQA oracle summary independently reproduced')


if __name__ == '__main__':
    main()
