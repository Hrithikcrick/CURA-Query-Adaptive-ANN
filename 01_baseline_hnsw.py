import os
import time
import random
import numpy as np
import pandas as pd
import faiss
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from collections import Counter
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

faiss.omp_set_num_threads(4)

print("FAISS threads:", faiss.omp_get_max_threads())


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET = "scifact"

MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"

ANN_K = 10
IR_K = 100

TARGET_RECALL = 0.90

EF_VALUES = [16, 32, 64, 96, 128, 192, 256]

HNSW_M = 32
EF_CONSTRUCTION = 200

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE) # change to "cpu" if GPU unavailable

print("Dataset:", DATASET)
print("Encoder:", MODEL_NAME)
print("efSearch values:", EF_VALUES)


# ============================================================
# 3. DOWNLOAD BEIR DATASET
# ============================================================

url = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)

out_dir = "./datasets"

data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="test")

print("\nCorpus size:", len(corpus))
print("Queries:", len(queries))
print("Queries with qrels:", len(qrels))


# ============================================================
# 4. PREPARE DOCUMENTS AND QUERIES
# ============================================================

doc_ids = list(corpus.keys())

documents = []

for doc_id in doc_ids:

    title = corpus[doc_id].get("title", "")
    text = corpus[doc_id].get("text", "")

    documents.append(
        (title + " " + text).strip()
    )


# Keep only queries for which qrels exist

query_ids = [
    qid for qid in queries
    if qid in qrels
]

query_texts = [
    queries[qid]
    for qid in query_ids
]

print("Evaluation queries:", len(query_ids))


# ============================================================
# 5. LOAD DENSE ENCODER
# ============================================================

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print("\nEncoding corpus...")


# ============================================================
# 6. ENCODE CORPUS
# ============================================================

corpus_embeddings = model.encode(
    documents,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

corpus_embeddings = np.asarray(
    corpus_embeddings,
    dtype=np.float32
)


# ============================================================
# 7. ENCODE QUERIES
# ============================================================

print("\nEncoding queries...")

query_embeddings = model.encode(
    query_texts,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

query_embeddings = np.asarray(
    query_embeddings,
    dtype=np.float32
)

print("Corpus embedding shape:", corpus_embeddings.shape)
print("Query embedding shape:", query_embeddings.shape)


# ============================================================
# 8. BUILD EXACT FAISS INDEX
# ============================================================

dimension = corpus_embeddings.shape[1]

exact_index = faiss.IndexFlatIP(dimension)

exact_index.add(corpus_embeddings)

print("\nExact index contains:", exact_index.ntotal, "vectors")


# ============================================================
# 9. EXACT SEARCH
# ============================================================

exact_scores, exact_indices = exact_index.search(
    query_embeddings,
    IR_K
)

print("Exact search complete")


# ============================================================
# 10. BUILD HNSW INDEX
# ============================================================

hnsw_index = faiss.IndexHNSWFlat(
    dimension,
    HNSW_M,
    faiss.METRIC_INNER_PRODUCT
)

hnsw_index.hnsw.efConstruction = EF_CONSTRUCTION

print("\nBuilding HNSW index...")

start = time.perf_counter()

hnsw_index.add(corpus_embeddings)

end = time.perf_counter()

print(
    "HNSW build time:",
    round(end - start, 3),
    "seconds"
)


# ============================================================
# 11. ANN RECALL FUNCTION
# ============================================================

def ann_recall_at_k(exact_result, approx_result, k=10):

    exact_set = set(exact_result[:k])
    approx_set = set(approx_result[:k])

    return len(
        exact_set.intersection(approx_set)
    ) / k


# ============================================================
# 12. IR METRICS
# ============================================================

def ndcg_at_k(result_indices, qid, k=10):

    gains = []

    relevant_docs = qrels.get(qid, {})

    for rank, idx in enumerate(result_indices[:k]):

        doc_id = doc_ids[idx]

        rel = relevant_docs.get(doc_id, 0)

        gain = (
            (2 ** rel - 1)
            /
            np.log2(rank + 2)
        )

        gains.append(gain)

    dcg = sum(gains)

    ideal_rels = sorted(
        relevant_docs.values(),
        reverse=True
    )[:k]

    idcg = sum(
        (2 ** rel - 1)
        /
        np.log2(rank + 2)

        for rank, rel in enumerate(ideal_rels)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def recall_at_k(result_indices, qid, k=100):

    relevant_docs = {
        doc_id
        for doc_id, rel in qrels.get(qid, {}).items()
        if rel > 0
    }

    if len(relevant_docs) == 0:
        return 0.0

    retrieved_docs = {
        doc_ids[idx]
        for idx in result_indices[:k]
    }

    return len(
        relevant_docs.intersection(retrieved_docs)
    ) / len(relevant_docs)


# ============================================================
# 13. EXACT IR QUALITY
# ============================================================

exact_ndcg = []

exact_ir_recall = []

for i, qid in enumerate(query_ids):

    exact_ndcg.append(
        ndcg_at_k(
            exact_indices[i],
            qid,
            10
        )
    )

    exact_ir_recall.append(
        recall_at_k(
            exact_indices[i],
            qid,
            100
        )
    )

print("\nExact retrieval:")
print("nDCG@10:", np.mean(exact_ndcg))
print("Recall@100:", np.mean(exact_ir_recall))


# ============================================================
# 14. SWEEP efSearch
# ============================================================

results_by_ef = {}

summary_rows = []

print("\nStarting efSearch sweep...\n")

for ef in EF_VALUES:

    hnsw_index.hnsw.efSearch = ef


    # warm up
    _ = hnsw_index.search(
        query_embeddings[:min(20, len(query_embeddings))],
        IR_K
    )


    start = time.perf_counter()

    scores, indices = hnsw_index.search(
        query_embeddings,
        IR_K
    )

    elapsed = time.perf_counter() - start

    latency_per_query_ms = (
        elapsed
        /
        len(query_embeddings)
        *
        1000
    )


    recalls = []

    ndcgs = []

    ir_recalls = []


    for i, qid in enumerate(query_ids):

        recalls.append(
            ann_recall_at_k(
                exact_indices[i],
                indices[i],
                ANN_K
            )
        )

        ndcgs.append(
            ndcg_at_k(
                indices[i],
                qid,
                10
            )
        )

        ir_recalls.append(
            recall_at_k(
                indices[i],
                qid,
                100
            )
        )


    recalls = np.array(recalls)

    ndcgs = np.array(ndcgs)

    ir_recalls = np.array(ir_recalls)


    violation_rate = np.mean(
        recalls < TARGET_RECALL
    )


    results_by_ef[ef] = {
        "indices": indices,
        "scores": scores,
        "ann_recall": recalls,
        "ndcg": ndcgs,
        "ir_recall": ir_recalls,
        "latency": latency_per_query_ms
    }


    summary_rows.append({

        "efSearch": ef,

        "ANN_Recall@10":
            np.mean(recalls),

        "IR_Recall@100":
            np.mean(ir_recalls),

        "nDCG@10":
            np.mean(ndcgs),

        "Violation_Rate":
            violation_rate,

        "Latency_ms":
            latency_per_query_ms
    })


summary = pd.DataFrame(summary_rows)

print("\n================ BASELINE SWEEP ================")

print(summary.to_string(index=False))


# ============================================================
# 15. FIND MINIMUM REQUIRED BUDGET FOR EACH QUERY
# ============================================================

minimum_budget = []

for i in range(len(query_ids)):

    required_ef = EF_VALUES[-1]

    for ef in EF_VALUES:

        recall = results_by_ef[ef][
            "ann_recall"
        ][i]

        if recall >= TARGET_RECALL:

            required_ef = ef
            break

    minimum_budget.append(required_ef)


minimum_budget = np.array(minimum_budget)


print("\nMinimum required efSearch distribution:")

for ef in EF_VALUES:

    count = np.sum(
        minimum_budget == ef
    )

    print(
        f"ef={ef:<4} : {count} queries"
    )


# ============================================================
# 16. THIS IS THE FIRST IMPORTANT RESEARCH CHECK
# ============================================================

print("\nBudget heterogeneity:")

print(
    "Number of distinct optimal budgets:",
    len(np.unique(minimum_budget))
)

print(
    "Mean required budget:",
    np.mean(minimum_budget)
)

print(
    "Median required budget:",
    np.median(minimum_budget)
)


# ============================================================
# 17. TRAIN / TEST SPLIT
# ============================================================

indices_all = np.arange(
    len(query_embeddings)
)

train_idx, test_idx = train_test_split(

    indices_all,

    test_size=0.30,

    random_state=SEED
)


# ============================================================
# 18. CREATE CHEAP QUERY FEATURES
# ============================================================

def create_query_features(embeddings):

    features = []

    for emb in embeddings:

        stats = [

            np.linalg.norm(emb),

            np.mean(emb),

            np.std(emb),

            np.min(emb),

            np.max(emb),

            np.mean(np.abs(emb)),

            np.sum(emb > 0)
            /
            len(emb),

            np.max(np.abs(emb))
        ]

        # Embedding itself is already available after encoding.
        # This costs almost nothing extra at retrieval time.

        feature = np.concatenate([
            emb,
            np.array(stats)
        ])

        features.append(feature)

    return np.asarray(
        features,
        dtype=np.float32
    )


X = create_query_features(
    query_embeddings
)

y = minimum_budget


X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


print("\nFeature shape:", X.shape)


# ============================================================
# 19. LIGHTWEIGHT BUDGET CLASSIFIER
# ============================================================

policy = DecisionTreeClassifier(

    max_depth=6,

    min_samples_leaf=5,

    class_weight="balanced",

    random_state=SEED
)


policy.fit(
    X_train,
    y_train
)


predicted_budget = policy.predict(
    X_test
)


print(
    "\nExact budget classification accuracy:",
    accuracy_score(
        y_test,
        predicted_budget
    )
)


# ============================================================
# 20. CONSERVATIVE SAFETY MARGIN
# ============================================================
#
# Instead of trusting the exact prediction,
# increase it by one budget level.
#
# Later we replace this with real calibration/conformal risk.
#

def increase_budget_once(pred):

    pos = EF_VALUES.index(
        int(pred)
    )

    new_pos = min(
        pos + 1,
        len(EF_VALUES) - 1
    )

    return EF_VALUES[new_pos]


safe_predicted_budget = np.array([

    increase_budget_once(x)

    for x in predicted_budget
])


print(
    "\nPredicted budgets:",
    Counter(predicted_budget)
)

print(
    "Safe predicted budgets:",
    Counter(safe_predicted_budget)
)


# ============================================================
# 21. FIND FIXED-BUDGET BASELINE USING TRAINING DATA
# ============================================================
#
# Choose the smallest global efSearch such that
# at least 95% of training queries satisfy target recall.
#

FIXED_MAX_VIOLATION = 0.05

fixed_budget = EF_VALUES[-1]


for ef in EF_VALUES:

    recalls = results_by_ef[ef][
        "ann_recall"
    ][train_idx]

    violation = np.mean(
        recalls < TARGET_RECALL
    )

    if violation <= FIXED_MAX_VIOLATION:

        fixed_budget = ef
        break


print(
    "\nSelected fixed baseline efSearch:",
    fixed_budget
)


# ============================================================
# 22. RUN ADAPTIVE SEARCH
# ============================================================

adaptive_results = []

adaptive_latencies = []

adaptive_recalls = []

adaptive_ndcg = []

adaptive_ir_recall = []


print("\nRunning adaptive policy...")


# Warm up

hnsw_index.hnsw.efSearch = EF_VALUES[-1]

_ = hnsw_index.search(
    query_embeddings[:10],
    IR_K
)


for pos, query_index in enumerate(test_idx):

    ef = int(
        safe_predicted_budget[pos]
    )

    hnsw_index.hnsw.efSearch = ef


    q = query_embeddings[
        query_index:
        query_index + 1
    ]


    start = time.perf_counter_ns()

    scores, idx = hnsw_index.search(
        q,
        IR_K
    )

    end = time.perf_counter_ns()


    latency_ms = (
        end - start
    ) / 1_000_000


    result = idx[0]


    adaptive_results.append(result)

    adaptive_latencies.append(
        latency_ms
    )


    ann_r = ann_recall_at_k(

        exact_indices[query_index],

        result,

        ANN_K
    )


    adaptive_recalls.append(
        ann_r
    )


    qid = query_ids[
        query_index
    ]


    adaptive_ndcg.append(

        ndcg_at_k(
            result,
            qid,
            10
        )
    )


    adaptive_ir_recall.append(

        recall_at_k(
            result,
            qid,
            100
        )
    )


adaptive_recalls = np.array(
    adaptive_recalls
)

adaptive_latencies = np.array(
    adaptive_latencies
)


# ============================================================
# 23. RUN FIXED-BUDGET BASELINE ON SAME TEST QUERIES
# ============================================================

hnsw_index.hnsw.efSearch = fixed_budget


fixed_latencies = []

fixed_recalls = []

fixed_ndcg = []

fixed_ir_recall = []


for query_index in test_idx:

    q = query_embeddings[
        query_index:
        query_index + 1
    ]


    start = time.perf_counter_ns()

    scores, idx = hnsw_index.search(
        q,
        IR_K
    )

    end = time.perf_counter_ns()


    latency_ms = (
        end - start
    ) / 1_000_000


    result = idx[0]


    fixed_latencies.append(
        latency_ms
    )


    fixed_recalls.append(

        ann_recall_at_k(

            exact_indices[query_index],

            result,

            ANN_K
        )
    )


    qid = query_ids[
        query_index
    ]


    fixed_ndcg.append(

        ndcg_at_k(
            result,
            qid,
            10
        )
    )


    fixed_ir_recall.append(

        recall_at_k(
            result,
            qid,
            100
        )
    )


fixed_recalls = np.array(
    fixed_recalls
)

fixed_latencies = np.array(
    fixed_latencies
)


# ============================================================
# 24. POLICY INFERENCE OVERHEAD
# ============================================================

start = time.perf_counter_ns()

_ = policy.predict(
    X_test
)

end = time.perf_counter_ns()


policy_overhead_ms = (
    (end - start)
    /
    1_000_000
    /
    len(X_test)
)


# ============================================================
# 25. FINAL COMPARISON
# ============================================================

fixed_violation = np.mean(
    fixed_recalls < TARGET_RECALL
)

adaptive_violation = np.mean(
    adaptive_recalls < TARGET_RECALL
)


comparison = pd.DataFrame([

    {

        "Policy":
            f"Fixed ef={fixed_budget}",

        "Mean efSearch":
            fixed_budget,

        "ANN Recall@10":
            np.mean(fixed_recalls),

        "IR Recall@100":
            np.mean(fixed_ir_recall),

        "nDCG@10":
            np.mean(fixed_ndcg),

        "Violation Rate":
            fixed_violation,

        "Mean Latency ms":
            np.mean(fixed_latencies),

        "P95 Latency ms":
            np.percentile(
                fixed_latencies,
                95
            )
    },

    {

        "Policy":
            "Query-specific",

        "Mean efSearch":
            np.mean(
                safe_predicted_budget
            ),

        "ANN Recall@10":
            np.mean(
                adaptive_recalls
            ),

        "IR Recall@100":
            np.mean(
                adaptive_ir_recall
            ),

        "nDCG@10":
            np.mean(
                adaptive_ndcg
            ),

        "Violation Rate":
            adaptive_violation,

        "Mean Latency ms":
            np.mean(
                adaptive_latencies
            ),

        "P95 Latency ms":
            np.percentile(
                adaptive_latencies,
                95
            )
    }
])


print("\n")
print("=" * 90)

print("FINAL RESULTS")

print("=" * 90)

print(
    comparison.to_string(
        index=False
    )
)


print(
    "\nPolicy inference overhead:",
    round(
        policy_overhead_ms,
        6
    ),
    "ms/query"
)


# ============================================================
# 26. HOW OFTEN DOES POLICY SAVE SEARCH EFFORT?
# ============================================================

saving = (
    safe_predicted_budget
    <
    fixed_budget
)

same = (
    safe_predicted_budget
    ==
    fixed_budget
)

higher = (
    safe_predicted_budget
    >
    fixed_budget
)


print("\nPolicy behaviour:")

print(
    "Queries using LESS effort:",
    np.mean(saving)
)

print(
    "Queries using SAME effort:",
    np.mean(same)
)

print(
    "Queries using MORE effort:",
    np.mean(higher)
)


# ============================================================
# 27. PLOT QUALITY VS SEARCH BUDGET
# ============================================================

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(8, 5))

plt.plot(
    summary["efSearch"],
    summary["ANN_Recall@10"],
    marker="o"
)

plt.axhline(
    TARGET_RECALL,
    linestyle="--",
    label="Target recall"
)

plt.xlabel("HNSW efSearch")
plt.ylabel("Mean ANN Recall@10")

plt.title(
    "ANN Quality vs HNSW Search Budget"
)

plt.grid(True)

plt.legend()

plt.tight_layout()
plt.savefig(
    "results/ann_quality_vs_budget.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved: results/ann_quality_vs_budget.png")


# ============================================================
# 28. PLOT VIOLATION RATE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    summary["efSearch"],
    summary["Violation_Rate"],
    marker="o"
)

plt.xlabel("HNSW efSearch")
plt.ylabel("Constraint Violation Rate")

plt.title(
    "Recall Constraint Violations vs Search Budget"
)

plt.grid(True)

plt.tight_layout()
plt.savefig(
    "results/violation_rate_vs_budget.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved: results/violation_rate_vs_budget.png")


# ============================================================
# 29. BUDGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

values, counts = np.unique(
    minimum_budget,
    return_counts=True
)

plt.bar(
    values.astype(str),
    counts
)

plt.xlabel(
    "Minimum efSearch required"
)

plt.ylabel(
    "Number of Queries"
)

plt.title(
    "Query-Specific ANN Budget Distribution"
)

plt.tight_layout()
plt.savefig(
    "results/budget_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved: results/budget_distribution.png")


# ============================================================
# 30. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

summary.to_csv(
    "results/efsearch_sweep.csv",
    index=False
)

comparison.to_csv(
    "results/policy_comparison.csv",
    index=False
)


query_results = pd.DataFrame({

    "query_id":
        np.array(query_ids)[test_idx],

    "required_budget":
        minimum_budget[test_idx],

    "predicted_budget":
        predicted_budget,

    "safe_budget":
        safe_predicted_budget,

    "adaptive_ann_recall":
        adaptive_recalls,

    "adaptive_ndcg":
        adaptive_ndcg,

    "adaptive_ir_recall":
        adaptive_ir_recall,

    "adaptive_latency_ms":
        adaptive_latencies
})


query_results.to_csv(

    "results/query_level_results.csv",

    index=False
)


print("\nSaved:")

print(
    "results/efsearch_sweep.csv"
)

print(
    "results/policy_comparison.csv"
)

print(
    "results/query_level_results.csv"
)