"""Recover original TREC ranked IDs directly from the historical PILOT ZIP.

No model, FAISS index, search, encoding or training is invoked. Only the 43
TREC query rows are exported. HNSW rank order comes from the original stability
cache; score evidence is recomputed from the original cached float32 vectors.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'sigir_analysis'
BASE = 'msmarco_large/msmarco_1000000_q1000/'
PREFIX = 'IR_PROJECT/PILOT/'
EFS = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def recover(archive_path):
    metadata = json.loads((OUT / 'recovery_delta_20261116.json').read_text())
    archive_record = next(r for r in metadata['archive_reviews'] if r['name'] == archive_path.name and 'sha256' in r)
    assert archive_path.stat().st_size == archive_record['bytes']
    provenance_path = OUT / 'trec_ranking_provenance.json'
    assert not provenance_path.exists(), 'Existing recovered evidence must not be overwritten.'
    source_checks, member_checks = {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        def read(rel):
            raw = archive.read(PREFIX + rel)
            info = archive.getinfo(PREFIX + rel)
            member_checks[rel] = dict(bytes=len(raw), sha256=sha(raw), zip_crc32=f'{info.CRC:08x}')
            return raw

        for rel in ['04_msmarco_large_pilot.py', '05_research_grade_validation.py', '08_patience_style_baseline.py',
                    BASE+'results/queries.csv', BASE+'research_grade_validation/top10_query_budget_results.csv',
                    BASE+'research_grade_validation/experiment_protocol.json', BASE+'cura_controller/cura_protocol.json']:
            raw = read(rel)
            assert raw == (ROOT / rel).read_bytes(), rel
            source_checks[rel] = sha(raw)
        queries = list(csv.DictReader(io.StringIO(read(BASE+'results/queries.csv').decode('utf-8-sig'))))
        selected = [(i, r) for i, r in enumerate(queries) if r['source'] == 'trec-dl-2019']
        assert len(selected) == 43
        meta = json.loads(read(BASE+'cache/corpus_meta.json'))
        assert meta['selected_docs'] == 1000629 and meta['seed'] == 42
        raw_map = read(BASE+'cache/docids.txt')
        docids = raw_map.decode('utf-8-sig').splitlines()
        assert len(docids) == len(set(docids)) == 1000629
        exact_i = np.load(io.BytesIO(read(BASE+'cache/exact_top100_indices.npy')), allow_pickle=False)
        exact_d = np.load(io.BytesIO(read(BASE+'cache/exact_top100_scores.npy')), allow_pickle=False)
        queries_v = np.load(io.BytesIO(read(BASE+'cache/query_embeddings.npy')), allow_pickle=False)
        with np.load(io.BytesIO(read(BASE+'patience_style_baseline/stability_top10_searches.npz')), allow_pickle=False) as cache:
            ann_i = cache['indices']
        assert exact_i.shape == exact_d.shape == (1043, 100)
        assert queries_v.shape == (1043, 384) and queries_v.dtype == np.float32
        assert ann_i.shape == (10, 1043, 10)
        sweep = list(csv.DictReader(io.StringIO(read(BASE+'research_grade_validation/top10_query_budget_results.csv').decode('utf-8-sig'))))
        assert sorted({int(r['efSearch']) for r in sweep}) == EFS
        qrows = [i for i, _ in selected]
        ids = ann_i[:, qrows, :]
        assert np.all(ids >= 0) and np.all(ids < len(docids))
        assert np.all(exact_i[qrows] >= 0) and np.all(exact_i[qrows] < len(docids))
        needed = np.unique(ids)
        gathered = np.empty((len(needed), 384), dtype=np.float32)
        corpus_member = PREFIX + BASE + 'cache/corpus_embeddings.f32'
        corpus_info = archive.getinfo(corpus_member)
        assert corpus_info.file_size == 1000629 * 384 * 4
        h = hashlib.sha256()
        start = 0
        norm_min, norm_max = float('inf'), 0.0
        print(f'Recovering cached score evidence for {len(needed)} document vectors; no retrieval.', flush=True)
        with archive.open(corpus_member) as stream:
            while raw := stream.read(8192 * 384 * 4):
                h.update(raw)
                block = np.frombuffer(raw, dtype=np.float32).reshape(-1, 384)
                assert np.isfinite(block).all()
                norms = np.einsum('ij,ij->i', block, block)
                norm_min = min(norm_min, float(norms.min()))
                norm_max = max(norm_max, float(norms.max()))
                lo, hi = np.searchsorted(needed, [start, start+len(block)])
                gathered[lo:hi] = block[needed[lo:hi]-start]
                start += len(block)
        assert start == len(docids) and norm_min > .9999 and norm_max < 1.0001
        assert np.allclose(np.linalg.norm(queries_v[qrows], axis=1), 1, atol=1e-5)
        member_checks[BASE+'cache/corpus_embeddings.f32'] = dict(bytes=corpus_info.file_size,
            sha256=h.hexdigest(), zip_crc32=f'{corpus_info.CRC:08x}', squared_norm_min=norm_min, squared_norm_max=norm_max)
        score_lookup = {int(doc): gathered[i] for i, doc in enumerate(needed)}
        # Confirm recovered row mapping and query vectors agree with original
        # exact score evidence for every overlapping exact/approx document.
        max_score_difference = 0.0
        exact_rows, ann_rows = [], []
        for qi, row in selected:
            key, qid = row['query_key'], row['raw_query_id']
            reference_scores = dict(zip(exact_i[qi].tolist(), exact_d[qi].tolist()))
            for rank, (doc, score) in enumerate(zip(exact_i[qi], exact_d[qi]), 1):
                exact_rows.append(dict(query_key=key, query_id=qid, rank=rank, doc_id=docids[int(doc)], score=repr(float(score))))
            for ei, ef in enumerate(EFS):
                for rank, doc in enumerate(ann_i[ei, qi], 1):
                    score = float(np.dot(queries_v[qi], score_lookup[int(doc)]))
                    if int(doc) in reference_scores:
                        max_score_difference = max(max_score_difference, abs(score-reference_scores[int(doc)]))
                    ann_rows.append(dict(query_key=key, query_id=qid, efSearch=ef, rank=rank, doc_id=docids[int(doc)], score=repr(score)))
        assert max_score_difference < 1e-5

    cache_dir = ROOT / BASE / 'cache'
    cache_dir.mkdir(exist_ok=True)
    map_path = cache_dir / 'docids.txt'
    if map_path.exists():
        assert map_path.read_bytes() == raw_map
    else:
        map_path.write_bytes(raw_map)
    for name, rows in [('trec_exact_rankings.csv', exact_rows), ('trec_hnsw_rankings.csv', ann_rows)]:
        assert not (OUT/name).exists()
        with (OUT/name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    provenance = dict(schema_version=1, original_corpus_document_count=1000629,
        corpus_docids_path=(BASE+'cache/docids.txt'), corpus_docids_sha256=sha(raw_map),
        exact_rankings_sha256=sha((OUT/'trec_exact_rankings.csv').read_bytes()),
        hnsw_rankings_sha256=sha((OUT/'trec_hnsw_rankings.csv').read_bytes()),
        original_corpus_identity_confirmed=True,
        original_run_provenance='Recovered directly from Downloads/IR_PROJECT.zip, IR_PROJECT/PILOT. Seven original source/config/query/sweep files match the current archived run byte-for-byte. The original 1,000,629-row document map translates exact_top100_indices and stability_top10_searches indices. Source08 defines cache axes as sorted archived ef values and unchanged queries.csv row order. No rankings were generated or reordered. Original HNSW scores were not cached; score evidence is reconstructed by inner products of original cached vectors, without changing ranks. FULL per-query reproduction is a separate mandatory gate.',
        archive=dict(path='~/Downloads/IR_PROJECT.zip', bytes=archive_path.stat().st_size, sha256=archive_record['sha256']),
        archive_source_byte_checks=source_checks, archive_member_checks=member_checks,
        exact_status='RECOVERED', hnsw_status='RECOVERED', exact_depth=100, hnsw_depth=10,
        cache_axis_order=dict(efSearch=EFS, query_rows='original queries.csv row order'),
        recovered_score_evidence_max_difference_vs_exact_cache=max_score_difference,
        independent_historical_corpus_map_checksum_available=False,
        provenance_limitation='Archive lineage, matching original sources, supplied map checksum and subsequent FULL consistency support identity; there was no separately logged historical map checksum or encoder revision.',
        retrieval_executed=False, embeddings_regenerated=False, index_rebuilt=False, controller_retrained=False)
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    sys.path.insert(0, str(OUT))
    from ranking_input_gate import load_and_validate_rankings
    gate = load_and_validate_rankings(ROOT, OUT/'trec_exact_rankings.csv', OUT/'trec_hnsw_rankings.csv', provenance_path)
    gate['reproduction'].to_csv(OUT/'trec_retrieval_reproduction_per_query.csv', index=False)
    (OUT/'recovered_ranking_validation.json').write_text(json.dumps(dict(validation=gate['validation'],
        ranking_hashes=gate['ranking_hashes'], source_hashes=gate['source_hashes']), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(gate['validation'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path.home()/'Downloads/IR_PROJECT.zip')
    recover(parser.parse_args().archive)
