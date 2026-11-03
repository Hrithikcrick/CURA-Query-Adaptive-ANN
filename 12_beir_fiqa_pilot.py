"""Independent BEIR FiQA HNSW baseline/oracle pilot; this does not run CURA."""
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "beir"
OUT = ROOT / "beir_fiqa"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
EF_VALUES = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256]
K = 10
DELTA = 0.025
MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"

print("Loading BEIR FiQA test split...")
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
folder = util.download_and_unzip(url, str(DATA))
corpus, queries, qrels = GenericDataLoader(data_folder=folder).load(split="test")
dids = sorted(corpus)
qids = sorted(set(queries) & set(qrels))
print(f"Documents: {len(dids)}; test queries: {len(qids)}")

cache_path = DATA / "fiqa_embeddings.npz"
if cache_path.exists():
    with np.load(cache_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["dids"], np.array(dids)) or not np.array_equal(saved["qids"], np.array(qids)):
            raise RuntimeError("Embedding cache IDs differ from current FiQA data. Remove cache and rerun.")
        dv = saved["dv"].astype("float32")
        qv = saved["qv"].astype("float32")
    print("Loaded cached embeddings.")
else:
    model = SentenceTransformer(MODEL_NAME)
    texts = [((corpus[d].get("title") or "") + " " + (corpus[d].get("text") or "")).strip() for d in dids]
    dv = model.encode(texts, batch_size=64, show_progress_bar=True,
                      convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    qv = model.encode([queries[q] for q in qids], batch_size=64,
                      show_progress_bar=True, convert_to_numpy=True,
                      normalize_embeddings=True).astype("float32")
    np.savez(cache_path, dv=dv, qv=qv, dids=np.array(dids), qids=np.array(qids))

faiss.omp_set_num_threads(1)
d = dv.shape[1]
exact = faiss.IndexFlatIP(d)
exact.add(dv)
_, exact_ids = exact.search(qv, K)
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 200
hnsw.add(dv)


def ndcg(indices, qid):
    judgments = qrels[qid]
    gains = [max(0., float(judgments.get(dids[int(i)], 0))) for i in indices if i >= 0]
    best = sorted((max(0., float(v)) for v in judgments.values()), reverse=True)[:K]
    def dcg(vals):
        return sum((2.0 ** r - 1.0) / np.log2(pos + 2) for pos, r in enumerate(vals))
    ideal = dcg(best)
    return dcg(gains) / ideal if ideal else 0.


exact_utility = [ndcg(ids, qid) for ids, qid in zip(exact_ids, qids)]
rows = []
for ef in EF_VALUES:
    hnsw.hnsw.efSearch = ef
    for i, qid in enumerate(qids):
        if hasattr(faiss.cvar, "hnsw_stats"):
            faiss.cvar.hnsw_stats.reset()
        start = time.perf_counter()
        _, ids = hnsw.search(qv[i:i+1], K)
        latency_ms = 1000 * (time.perf_counter() - start)
        approx_ndcg = ndcg(ids[0], qid)
        loss = exact_utility[i] - approx_ndcg
        work = float(faiss.cvar.hnsw_stats.ndis) if hasattr(faiss.cvar, "hnsw_stats") else float("nan")
        overlap = len(set(map(int, exact_ids[i])) & set(map(int, ids[0]))) / K
        rows.append(dict(query_id=qid, efSearch=ef, exact_nDCG10=exact_utility[i],
                         nDCG10=approx_ndcg, nDCG_loss=loss,
                         IR_unsafe=int(loss > DELTA), ANN_Recall10=overlap,
                         ANN_unsafe=int(overlap < .90),
                         distance_computations=work, latency_ms=latency_ms))
    print(f"Completed efSearch={ef}")

results = pd.DataFrame(rows)
results.to_csv(OUT / "fiqa_query_budget_results.csv", index=False)
summary = results.groupby("efSearch", as_index=False).agg(
    queries=("query_id", "nunique"), mean_nDCG10=("nDCG10", "mean"),
    IR_violation_rate=("IR_unsafe", "mean"),
    ANN_violation_rate=("ANN_unsafe", "mean"),
    ANN_recall10=("ANN_Recall10", "mean"),
    mean_work=("distance_computations", "mean"),
    mean_latency_ms=("latency_ms", "mean"),
    p95_latency_ms=("latency_ms", lambda s: s.quantile(.95)))
summary.to_csv(OUT / "fiqa_budget_summary.csv", index=False)

# Diagnostic oracles can see outcomes and are NOT deployable controllers.
oracle = []
for qid, group in results.groupby("query_id"):
    ann_safe = group.loc[group.ANN_Recall10 >= .90, "efSearch"]
    ir_safe = group.loc[group.nDCG_loss <= DELTA, "efSearch"]
    oracle.append(dict(query_id=qid, ann_safe_ef=ann_safe.min() if len(ann_safe) else None,
                       ir_safe_ef=ir_safe.min() if len(ir_safe) else None))
pd.DataFrame(oracle).to_csv(OUT / "fiqa_oracle_budgets.csv", index=False)
(OUT / "fiqa_protocol.json").write_text(json.dumps({
    "dataset": "BEIR FiQA-2018 test", "encoder": MODEL_NAME, "seed": SEED,
    "n_docs": len(dids), "n_queries": len(qids), "ef_values": EF_VALUES,
    "hnsw_M": 32, "efConstruction": 200,
    "nDCG_loss_tolerance": DELTA, "ANN_recall_target": .90,
    "warning": "Baseline and diagnostic oracles only: NOT a CURA transfer experiment"
}, indent=2), encoding="utf-8")
print("\nBEIR FiQA BASELINE/ORACLE RESULTS")
print(summary.to_string(index=False))
print("\nSaved results under:", OUT)
print("This is NOT yet a CURA generalization claim.")
