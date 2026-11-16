"""Inspect only new Git blobs and additional local recovery scopes.

Reuses the existing validated artifact parser and the saved external metadata
inventory. Does not rerun any scientific analysis, load models, or retrieve.
"""
import hashlib
import io
import json
import re
import runpy
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'sigir_analysis'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def run():
    inventory_path = OUT / 'recovery_delta_external_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    prior_path = OUT / 'local_retrieval_audit.json'
    prior = json.loads(prior_path.read_text())
    helper = runpy.run_path(str(OUT / 'audit_local_retrieval.py'))
    inspect = helper['inspect_payload']
    inspections, members, containers, issues = [], [], [], []

    def inspect_bytes(name, raw, label):
        before = len(inspections)
        try:
            inspect(name, io.BytesIO(raw), len(raw), label, inspections, members, containers)
        except Exception as exc:
            issues.append(dict(path=label, error=str(exc), type=type(exc).__name__))
        return inspections[before:]

    old_oids = {r['oid'] for r in prior['git_content_inspections']}
    history = {}
    for line in git('rev-list', '--objects', '--all', '--reflog').decode().splitlines():
        oid, _, name = line.partition(' ')
        if name:
            history.setdefault(oid, []).append(name)
    new_git = []
    for line in git('cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype) %(objectsize)').decode().splitlines():
        oid, kind, size = line.split()
        if kind != 'blob' or oid in old_oids:
            continue
        names = history.get(oid, ['unnamed_git_blob'])
        details = inspect_bytes(names[0], git('cat-file', 'blob', oid), 'git:' + oid)
        new_git.append(dict(oid=oid, bytes=int(size), paths=names,
                            formats=sorted({r['format'] for r in details}),
                            plausible_ranking=any(r.get('plausible_complete_ranking_candidate') for r in details)))

    archives = []
    selected = {Path(x['path']) for x in inventory['download_archive_path_hits']}
    selected.update(Path(x['path']) for x in inventory['candidate_files'] if Path(x['path']).suffix.lower() == '.zip')
    seen_archive_hashes = {}
    for path in sorted(selected):
        digest = sha(path)
        if digest in seen_archive_hashes:
            archives.append(dict(name=path.name, sha256=digest, duplicate_of=seen_archive_hashes[digest]))
            continue
        seen_archive_hashes[digest] = path.name
        start = len(inspections)
        try:
            with zipfile.ZipFile(path) as archive:
                entries = [x for x in archive.infolist() if not x.is_dir()]
                project = bool(re.search(r'cura|ir.project|ir.mid', path.name, re.I))
                candidates = [x for x in entries if project or re.search(
                    r'exact_top|query_embeddings|corpus_embeddings|docids\.txt|selected_corpus|\.faiss$|\.run$|\.trec$', x.filename, re.I)]
                for entry in candidates:
                    inspect_bytes(entry.filename, archive.read(entry), path.name + '::' + entry.filename)
            details = inspections[start:]
            archives.append(dict(name=path.name, bytes=path.stat().st_size, sha256=digest,
                                 members=len(entries), members_inspected=len(candidates),
                                 plausible_ranking=any(r.get('plausible_complete_ranking_candidate') for r in details)))
        except Exception as exc:
            issues.append(dict(path=path.name, error=str(exc), type=type(exc).__name__))

    parquet = []
    for item in inventory['candidate_files']:
        path = Path(item['path'])
        if path.suffix.lower() == '.parquet':
            details = inspect_bytes(path.name, path.read_bytes(), 'external_temp:' + path.parent.name + '/' + path.name)
            parquet.append(dict(name=path.name, parent=path.parent.name, bytes=item['bytes'],
                                sha256=sha(path), inspections=details))

    home = Path.home()
    model_dir = home / '.cache/huggingface/hub/models--sentence-transformers--msmarco-MiniLM-L6-cos-v5'
    cached_revision = (model_dir / 'refs/main').read_text().strip()
    snapshot = model_dir / 'snapshots' / cached_revision
    weights = snapshot / 'model.safetensors'
    raw_corpus = home / '.ir_datasets/msmarco-passage/collection.tsv'
    qrels = home / '.ir_datasets/msmarco-passage/trec-dl-2019/qrels'
    original_qrels = json.loads((OUT / 'qrels_recovery.json').read_text())
    raw_candidates = dict(
        full_corpus=dict(path='~/.ir_datasets/msmarco-passage/collection.tsv', present=raw_corpus.exists(),
                         bytes=raw_corpus.stat().st_size, original_subset_mapping_recovered=False,
                         historical_membership_order_checksum_available=False),
        model=dict(name='sentence-transformers/msmarco-MiniLM-L6-cos-v5', cached_revision=cached_revision,
                   weights_bytes=weights.stat().st_size, weights_sha256=sha(weights),
                   original_run_revision_verified=False,
                   reason='Current cache snapshot is identified, but the historical run did not record an immutable revision.'),
        cached_trec_qrels=dict(sha256=sha(qrels), matches_recovered_original=sha(qrels) == original_qrels['sha256']))
    plausible = [r['path'] for r in inspections if r.get('plausible_complete_ranking_candidate')]
    result = dict(
        date='2026-11-16', prior_audit_sha256=sha(prior_path),
        prior_audit_scope=prior['scope'], prior_git_blobs_inspected=prior['git_blob_count'],
        completed_analyses_recomputed=False, external_metadata_scope_file_counts=inventory['scope_file_counts'],
        external_metadata_candidates=len(inventory['candidate_files']),
        external_inaccessible_entries=len(inventory['errors']),
        search_limitations='Some operating-system temporary directories were inaccessible; software environments and the safety archive were excluded. No whole-disk completeness claim.',
        new_git_blobs=new_git, archive_reviews=archives, temporary_parquet_reviews=parquet,
        inspected_format_counts=dict(Counter(r['format'] for r in inspections)),
        plausible_ranking_candidates=plausible, inspection_issues=issues,
        raw_reconstruction_candidates=raw_candidates,
        exact_rankings_recovered=False, HNSW_rankings_recovered=False,
        faithful_reconstruction_gate='BLOCKED_UNVERIFIED_ORIGINAL_SUBSET_AND_MODEL_AND_GRAPH',
        full_qrel_reproduction='BLOCKED', density_experiment='BLOCKED', density_seeds_scored=0,
        retrieval_executed=False, embeddings_generated=False, index_built=False, controller_retrained=False)
    (OUT / 'recovery_delta_20261116.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(new_git_blobs=len(new_git), unique_archives=len(seen_archive_hashes),
                          parquet_files=len(parquet), plausible_candidates=plausible,
                          inspection_issues=issues, raw_candidates=raw_candidates), indent=2))
    return result


if __name__ == '__main__':
    run()
