"""Create a 50-case, human-review worksheet from *actual* FiQA HNSW ef=10 failures.

IMPORTANT: These are low-budget HNSW failures, NOT 50 failures of deployed CURA.
Generates evidence for manual inspection; it NEVER generates human judgments.
Does NOT fit models, optimize thresholds, or modify CURA evaluation results.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import faiss
import numpy as np
from beir.datasets.data_loader import GenericDataLoader

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "beir_fiqa"
SWEEP = BASE / "fiqa_query_budget_results.csv"
DATASET = ROOT / "data" / "beir" / "fiqa"
CACHE = ROOT / "data" / "beir" / "fiqa_embeddings.npz"
OUT = BASE / "manual_failure_review"
DELTA = 0.025
EF = 10
N_CASES = 50
K = 10


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def nDCG(doc_idxs, qid, doc_ids, qrels):
    judgments = qrels[qid]
    gains = [max(0.0, float(judgments.get(doc_ids[int(i)], 0))) for i in doc_idxs if i >= 0]
    best = sorted((max(0.0, float(g)) for g in judgments.values()), reverse=True)[:K]
    dcg = lambda vals: sum((2.0 ** g - 1.0) / math.log2(pos + 2) for pos, g in enumerate(vals))
    ideal = dcg(best)
    return dcg(gains) / ideal if ideal else 0.0


def passages(indices, doc_ids, corpus, qrels, qid, max_items=5):
    out = []
    for rank, i in enumerate(indices[:max_items], 1):
        if i < 0:
            continue
        did = doc_ids[int(i)]
        doc = corpus[did]
        text = (str(doc.get("title") or "") + " " + str(doc.get("text") or "")).strip()
        text = " ".join(text.split())[:350]
        relevance = qrels[qid].get(did, 0)
        out.append(f"#{rank} [{did}] relevance={relevance}: {text}")
    return "\n".join(out)


def main():
    if not SWEEP.is_file():
        raise FileNotFoundError(f"FiQA sweep is missing: {SWEEP}")
    if not DATASET.is_dir() or not CACHE.is_file():
        raise FileNotFoundError("FiQA data/cache missing. Run 12_beir_fiqa_pilot.py first (do not edit older saved results).")

    candidates = [r for r in read_csv(SWEEP)
                  if int(float(r["efSearch"])) == EF and int(float(r["IR_unsafe"])) == 1]
    candidates.sort(key=lambda r: (-float(r["nDCG_loss"]), str(r["query_id"])))
    unique = {}
    for r in candidates:
        unique.setdefault(str(r["query_id"]), r)
    if len(unique) < N_CASES:
        raise RuntimeError(f"Only {len(unique)} unique FiQA ef={EF} failures; need {N_CASES}.")
    selected = list(unique.values())[:N_CASES]

    corpus, queries, qrels = GenericDataLoader(data_folder=str(DATASET)).load(split="test")
    doc_ids = sorted(corpus)
    query_ids = sorted(set(queries) & set(qrels))
    with np.load(CACHE, allow_pickle=False) as saved:
        if not np.array_equal(saved["dids"], np.asarray(doc_ids)):
            raise ValueError("Cached document IDs do not match the FiQA dataset")
        if not np.array_equal(saved["qids"], np.asarray(query_ids)):
            raise ValueError("Cached query IDs do not match the FiQA dataset")
        doc_vectors = np.ascontiguousarray(saved["dv"], dtype="float32")
        query_vectors = np.ascontiguousarray(saved["qv"], dtype="float32")
    qloc = {qid: i for i, qid in enumerate(query_ids)}

    # Reconstruct the earlier index solely to attach retrieved-document evidence.
    # HNSW may not reproduce byte-for-byte on a different FAISS build. Check below.
    faiss.omp_set_num_threads(1)
    print(f"Building evidence-only HNSW index for {len(doc_ids):,} FiQA documents...")
    exact = faiss.IndexFlatIP(doc_vectors.shape[1]); exact.add(doc_vectors)
    hnsw = faiss.IndexHNSWFlat(doc_vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(doc_vectors)
    hnsw.hnsw.efSearch = EF

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    mismatches = []
    for pos, row in enumerate(selected, 1):
        qid = str(row["query_id"])
        if qid not in qloc:
            raise KeyError(f"Missing FiQA query: {qid}")
        vector = query_vectors[qloc[qid]:qloc[qid] + 1]
        _, exact_idx = exact.search(vector, K)
        _, hnsw_idx = hnsw.search(vector, K)
        original_exact = float(row["exact_nDCG10"])
        original_approx = float(row["nDCG10"])
        reproduced_exact = nDCG(exact_idx[0], qid, doc_ids, qrels)
        reproduced_approx = nDCG(hnsw_idx[0], qid, doc_ids, qrels)
        matches = abs(reproduced_exact - original_exact) < 1e-5 and abs(reproduced_approx - original_approx) < 1e-5
        if not matches:
            mismatches.append(qid)
        relevant_doc_ids = sorted(qrels[qid], key=lambda d: (-float(qrels[qid][d]), str(d)))
        relevant = []
        for did in relevant_doc_ids[:3]:
            if did in corpus:
                doc = corpus[did]
                txt = " ".join((str(doc.get("title") or "") + " " + str(doc.get("text") or "")).split())[:350]
                relevant.append(f"[{did}] relevance={qrels[qid][did]}: {txt}")
        rows.append({
            "case_id": pos,
            "query_id": qid,
            "query_text": queries[qid],
            "failed_method": "Fixed HNSW efSearch=10 (NOT deployed CURA)",
            "saved_exact_nDCG10": original_exact,
            "saved_HNSW_nDCG10": original_approx,
            "saved_nDCG_loss": float(row["nDCG_loss"]),
            "saved_ANN_Recall10": float(row["ANN_Recall10"]),
            "saved_distance_computations": float(row["distance_computations"]),
            "reproduced_exact_nDCG10": reproduced_exact,
            "reproduced_HNSW_nDCG10": reproduced_approx,
            "reconstruction_matches_saved_metrics": "YES" if matches else "NO: investigate index reproducibility",
            "judged_relevant_passages": "\n".join(relevant),
            "exact_top5_passages": passages(exact_idx[0], doc_ids, corpus, qrels, qid),
            "HNSW_ef10_top5_passages": passages(hnsw_idx[0], doc_ids, corpus, qrels, qid),
            "human_failure_category": "",
            "human_analysis_of_what_went_wrong": "",
            "human_evidence_doc_ids_or_ranks": "",
            "human_suggested_remedy": "",
            "reviewer": "",
            "human_review_complete_YES_NO": "NO",
        })
    outfile = OUT / "fiqa_50_actual_ef10_failures_TO_REVIEW.csv"
    with outfile.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {
        "dataset": "BEIR FiQA 2018", "source": "Saved fiqa_query_budget_results.csv",
        "selection": "50 unique queries with actual IR_unsafe=1 at fixed HNSW efSearch=10; sorted by descending nDCG loss",
        "n_unique_available_failures_at_ef10": len(unique),
        "n_selected": len(rows), "n_human_reviews_completed": 0,
        "n_reconstruction_matches_saved_metrics": len(rows) - len(mismatches),
        "queries_with_reconstruction_mismatch": mismatches,
        "do_not_claim": [
            "These are NOT 50 failures of deployed CURA-FiQA.",
            "The automatically populated worksheet is NOT 50 completed human analyses.",
            "Do not tune CURA on these failure cases or change already viewed held-out conclusions.",
        ],
    }
    (OUT / "fiqa_50_failure_review_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    guide = """# FiQA — manual review of 50 real low-budget HNSW failures

These are **true nDCG-loss violations at ef=10** in the saved FiQA sweep, NOT deployed CURA-FiQA failures.
The review form has **no human judgments filled in**. Inspect each query, the judged-relevant
passages, and the exact/HNSW ranked passages. Record your own evidence and explanation.

## Suggested categories (choose based on evidence, do not auto-fill)
- Relevant passage missed by ANN search
- Wrong ranking among retrieved passages
- Ambiguity, rare entity or abbreviation
- Several near-equivalent candidate passages
- Qrel/annotation or evaluation limitation
- Index/search approximation failure
- Other (describe specifically)

For each row, fill `human_failure_category`, `human_analysis_of_what_went_wrong`,
`human_evidence_doc_ids_or_ranks`, `human_suggested_remedy`, `reviewer`, and
`human_review_complete_YES_NO=YES` only after reviewing it.

If `reconstruction_matches_saved_metrics` is NO, investigate FAISS index reproducibility
before using the reconstructed top-5 examples as evidence. Original metric rows remain unchanged.

Do not reuse this analysis for model selection on the existing FiQA held-out partition.
"""
    (OUT / "HOW_TO_REVIEW.md").write_text(guide, encoding="utf-8")

    print(f"Original ef=10 IR failures across unique queries: {len(unique)}")
    print(f"Generated {len(rows)} human-review candidates, with original query and passage evidence")
    print(f"Reconstructed result mismatches: {len(mismatches)}")
    print("Human reviews completed: 0 (complete the worksheet manually)")
    print(f"Saved: {outfile}")
    if mismatches:
        print("WARNING: Reconstructed HNSW ranking differs for some cases. Inspect flags; do not conflate reconstructions with original results.")


if __name__ == "__main__":
    main()
