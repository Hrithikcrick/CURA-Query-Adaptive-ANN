# ============================================================
# LEARNING QUERY-SPECIFIC ANN SEARCH BUDGETS
# COL764 - INITIAL PILOT + PHASE 1
#
# Dataset: BEIR SciFact
# Encoder: msmarco-MiniLM-L6-cos-v5
#
# This script:
# 1. Loads SciFact
# 2. Encodes corpus + queries
# 3. Builds exact FAISS index
# 4. Builds HNSW index
# 5. Sweeps efSearch
# 6. Measures ANN Recall@10, Recall@100, nDCG@10
# 7. Builds query-specific ANN-budget baseline
# 8. Creates ANN-safe and IR-safe labels
# 9. Studies ANN-recall vs ranking-utility disagreement
# 10. Computes ANN-based and IR-based oracle budgets
# 11. Saves plots + CSV files
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import time
import random

import numpy as np
import pandas as pd
import faiss

import matplotlib

# Prevent matplotlib windows from blocking execution
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from collections import Counter

import torch

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

print("=" * 90)
print("QUERY-SPECIFIC ANN SEARCH BUDGET PILOT")
print("=" * 90)

print("FAISS threads:", faiss.omp_get_max_threads())


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET = "scifact"

MODEL_NAME = (
    "sentence-transformers/"
    "msmarco-MiniLM-L6-cos-v5"
)

ANN_K = 10

IR_K = 100


# ------------------------------------------------------------
# ANN constraint
# ------------------------------------------------------------

TARGET_RECALL = 0.90


# ------------------------------------------------------------
# IR-UTILITY CONSTRAINTS
#
# These are exploratory Phase-1 values.
# Later we will sweep them as an ablation.
# ------------------------------------------------------------

NDCG_DROP_TOLERANCE = 0.05

IR_RECALL_DROP_TOLERANCE = 0.10


# ------------------------------------------------------------
# HNSW budgets
# ------------------------------------------------------------

EF_VALUES = [
    16,
    32,
    64,
    96,
    128,
    192,
    256
]

HNSW_M = 32

EF_CONSTRUCTION = 200


# ------------------------------------------------------------
# CPU / GPU
# ------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Using device:", DEVICE)

print("Dataset:", DATASET)

print("Encoder:", MODEL_NAME)

print("ANN Recall target:", TARGET_RECALL)

print(
    "nDCG drop tolerance:",
    NDCG_DROP_TOLERANCE
)

print(
    "IR Recall drop tolerance:",
    IR_RECALL_DROP_TOLERANCE
)

print(
    "efSearch values:",
    EF_VALUES
)


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES FIRST
# ============================================================

os.makedirs(
    "results_phase1",
    exist_ok=True
)

os.makedirs(
    "datasets",
    exist_ok=True
)


# ============================================================
# 4. DOWNLOAD / LOAD BEIR SCIFACT
# ============================================================

url = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)

out_dir = "./datasets"

print("\nLoading SciFact...")

data_path = util.download_and_unzip(
    url,
    out_dir
)

corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(
    split="test"
)


print("\nCorpus size:", len(corpus))

print(
    "Total queries:",
    len(queries)
)

print(
    "Queries with qrels:",
    len(qrels)
)


# ============================================================
# 5. PREPARE DOCUMENTS
# ============================================================

doc_ids = list(
    corpus.keys()
)

documents = []


for doc_id in doc_ids:

    title = corpus[
        doc_id
    ].get(
        "title",
        ""
    )

    text = corpus[
        doc_id
    ].get(
        "text",
        ""
    )

    document = (
        title
        + " "
        + text
    ).strip()

    documents.append(
        document
    )


# ============================================================
# 6. PREPARE QUERIES
# ============================================================

query_ids = [
    qid
    for qid in queries
    if qid in qrels
]


query_texts = [
    queries[qid]
    for qid in query_ids
]


print(
    "Evaluation queries:",
    len(query_ids)
)


# ============================================================
# 7. LOAD DENSE ENCODER
# ============================================================

print(
    "\nLoading dense encoder..."
)

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)


# ============================================================
# 8. ENCODE CORPUS
# ============================================================

print(
    "\nEncoding corpus..."
)

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
# 9. ENCODE QUERIES
# ============================================================

print(
    "\nEncoding queries..."
)

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


print(
    "Corpus embedding shape:",
    corpus_embeddings.shape
)

print(
    "Query embedding shape:",
    query_embeddings.shape
)


# ============================================================
# 10. BUILD EXACT FAISS INDEX
# ============================================================

dimension = (
    corpus_embeddings.shape[1]
)

exact_index = faiss.IndexFlatIP(
    dimension
)

exact_index.add(
    corpus_embeddings
)


print(
    "\nExact index vectors:",
    exact_index.ntotal
)


# ============================================================
# 11. EXACT SEARCH
# ============================================================

print(
    "\nRunning exact search..."
)

exact_scores, exact_indices = (
    exact_index.search(
        query_embeddings,
        IR_K
    )
)

print(
    "Exact search complete"
)


# ============================================================
# 12. BUILD HNSW INDEX
# ============================================================

hnsw_index = faiss.IndexHNSWFlat(
    dimension,
    HNSW_M,
    faiss.METRIC_INNER_PRODUCT
)

hnsw_index.hnsw.efConstruction = (
    EF_CONSTRUCTION
)


print(
    "\nBuilding HNSW index..."
)

start = time.perf_counter()

hnsw_index.add(
    corpus_embeddings
)

end = time.perf_counter()


print(
    "HNSW build time:",
    round(
        end - start,
        3
    ),
    "seconds"
)


# ============================================================
# 13. ANN RECALL FUNCTION
# ============================================================

def ann_recall_at_k(
    exact_result,
    approx_result,
    k=10
):

    exact_set = set(
        exact_result[:k]
    )

    approx_set = set(
        approx_result[:k]
    )

    overlap = (
        exact_set
        .intersection(
            approx_set
        )
    )

    return (
        len(overlap)
        / k
    )


# ============================================================
# 14. nDCG FUNCTION
# ============================================================

def ndcg_at_k(
    result_indices,
    qid,
    k=10
):

    gains = []

    relevant_docs = qrels.get(
        qid,
        {}
    )


    for rank, idx in enumerate(
        result_indices[:k]
    ):

        doc_id = doc_ids[
            idx
        ]

        rel = (
            relevant_docs
            .get(
                doc_id,
                0
            )
        )

        gain = (
            (2 ** rel - 1)
            /
            np.log2(
                rank + 2
            )
        )

        gains.append(
            gain
        )


    dcg = sum(
        gains
    )


    ideal_rels = sorted(
        relevant_docs.values(),
        reverse=True
    )[:k]


    idcg = sum(

        (
            (2 ** rel - 1)
            /
            np.log2(
                rank + 2
            )
        )

        for rank, rel
        in enumerate(
            ideal_rels
        )
    )


    if idcg == 0:

        return 0.0


    return (
        dcg / idcg
    )


# ============================================================
# 15. RECALL@K FUNCTION
# ============================================================

def recall_at_k(
    result_indices,
    qid,
    k=100
):

    relevant_docs = {

        doc_id

        for doc_id, rel
        in qrels.get(
            qid,
            {}
        ).items()

        if rel > 0
    }


    if len(
        relevant_docs
    ) == 0:

        return 0.0


    retrieved_docs = {

        doc_ids[idx]

        for idx
        in result_indices[:k]
    }


    found = (
        relevant_docs
        .intersection(
            retrieved_docs
        )
    )


    return (
        len(found)
        /
        len(relevant_docs)
    )


# ============================================================
# 16. EXACT IR QUALITY
# ============================================================

exact_ndcg = []

exact_ir_recall = []


for i, qid in enumerate(
    query_ids
):

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


exact_ndcg = np.asarray(
    exact_ndcg,
    dtype=np.float64
)

exact_ir_recall = np.asarray(
    exact_ir_recall,
    dtype=np.float64
)


print(
    "\nExact retrieval quality:"
)

print(
    "Exact nDCG@10:",
    np.mean(
        exact_ndcg
    )
)

print(
    "Exact Recall@100:",
    np.mean(
        exact_ir_recall
    )
)


# ============================================================
# 17. SWEEP HNSW efSearch
# ============================================================

results_by_ef = {}

summary_rows = []


print(
    "\nStarting efSearch sweep..."
)


for ef in EF_VALUES:

    hnsw_index.hnsw.efSearch = ef


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    _ = hnsw_index.search(

        query_embeddings[
            :min(
                20,
                len(
                    query_embeddings
                )
            )
        ],

        IR_K
    )


    # --------------------------------------------------------
    # Search all queries
    # --------------------------------------------------------

    start = (
        time.perf_counter()
    )

    scores, indices = (
        hnsw_index.search(
            query_embeddings,
            IR_K
        )
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )


    latency_per_query_ms = (

        elapsed

        /

        len(
            query_embeddings
        )

        *

        1000
    )


    recalls = []

    ndcgs = []

    ir_recalls = []


    for i, qid in enumerate(
        query_ids
    ):

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


    recalls = np.asarray(
        recalls,
        dtype=np.float64
    )

    ndcgs = np.asarray(
        ndcgs,
        dtype=np.float64
    )

    ir_recalls = np.asarray(
        ir_recalls,
        dtype=np.float64
    )


    violation_rate = np.mean(
        recalls
        <
        TARGET_RECALL
    )


    results_by_ef[
        ef
    ] = {

        "indices":
            indices,

        "scores":
            scores,

        "ann_recall":
            recalls,

        "ndcg":
            ndcgs,

        "ir_recall":
            ir_recalls,

        "latency":
            latency_per_query_ms
    }


    summary_rows.append({

        "efSearch":
            ef,

        "ANN_Recall@10":
            np.mean(
                recalls
            ),

        "IR_Recall@100":
            np.mean(
                ir_recalls
            ),

        "nDCG@10":
            np.mean(
                ndcgs
            ),

        "Violation_Rate":
            violation_rate,

        "Latency_ms":
            latency_per_query_ms
    })


summary = pd.DataFrame(
    summary_rows
)


print(
    "\n"
    + "=" * 90
)

print(
    "BASELINE HNSW SWEEP"
)

print(
    "=" * 90
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 18. PHASE 1:
# QUERY Ã— BUDGET SAFETY LABEL DATASET
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PHASE 1: CREATING QUERY-BUDGET SAFETY LABELS"
)

print(
    "=" * 90
)


query_budget_rows = []


for ef in EF_VALUES:

    ann_recalls = (
        results_by_ef[
            ef
        ][
            "ann_recall"
        ]
    )

    ann_ndcgs = (
        results_by_ef[
            ef
        ][
            "ndcg"
        ]
    )

    ann_ir_recalls = (
        results_by_ef[
            ef
        ][
            "ir_recall"
        ]
    )


    for i, qid in enumerate(
        query_ids
    ):

        # ----------------------------------------------------
        # Measured ANN values
        # ----------------------------------------------------

        ann_recall_value = float(
            ann_recalls[i]
        )

        ndcg_value = float(
            ann_ndcgs[i]
        )

        ir_recall_value = float(
            ann_ir_recalls[i]
        )


        # ----------------------------------------------------
        # Exact retrieval reference values
        # ----------------------------------------------------

        exact_ndcg_value = float(
            exact_ndcg[i]
        )

        exact_ir_recall_value = float(
            exact_ir_recall[i]
        )


        # ----------------------------------------------------
        # Per-query quality losses
        # ----------------------------------------------------

        raw_ndcg_loss = (

            exact_ndcg_value

            -

            ndcg_value
        )


        raw_ir_recall_loss = (

            exact_ir_recall_value

            -

            ir_recall_value
        )


        # ----------------------------------------------------
        # ANN can occasionally improve qrel quality
        # relative to exact embedding-neighbour search.
        #
        # Such cases have zero quality "loss".
        # ----------------------------------------------------

        ndcg_loss = max(
            0.0,
            raw_ndcg_loss
        )


        ir_recall_loss = max(
            0.0,
            raw_ir_recall_loss
        )


        # ----------------------------------------------------
        # ANN-SAFE LABEL
        # ----------------------------------------------------

        ann_safe = int(

            ann_recall_value

            >=

            TARGET_RECALL
        )


        # ----------------------------------------------------
        # nDCG-SAFE LABEL
        # ----------------------------------------------------

        ndcg_safe = int(

            ndcg_loss

            <=

            NDCG_DROP_TOLERANCE
        )


        # ----------------------------------------------------
        # Recall@100-SAFE LABEL
        # ----------------------------------------------------

        ir_recall_safe = int(

            ir_recall_loss

            <=

            IR_RECALL_DROP_TOLERANCE
        )


        # ----------------------------------------------------
        # COMBINED IR-SAFE LABEL
        # ----------------------------------------------------

        ir_safe = int(

            ndcg_safe == 1

            and

            ir_recall_safe == 1
        )


        # ----------------------------------------------------
        # DISAGREEMENT LABELS
        # ----------------------------------------------------

        ann_unsafe_but_ir_safe = int(

            ann_safe == 0

            and

            ir_safe == 1
        )


        ann_safe_but_ir_unsafe = int(

            ann_safe == 1

            and

            ir_safe == 0
        )


        query_budget_rows.append({

            "query_id":
                qid,

            "query_text":
                query_texts[i],

            "efSearch":
                int(
                    ef
                ),

            "exact_nDCG@10":
                exact_ndcg_value,

            "ANN_nDCG@10":
                ndcg_value,

            "nDCG_loss":
                ndcg_loss,

            "exact_IR_Recall@100":
                exact_ir_recall_value,

            "ANN_IR_Recall@100":
                ir_recall_value,

            "IR_Recall_loss":
                ir_recall_loss,

            "ANN_Recall@10":
                ann_recall_value,

            "ANN_safe":
                ann_safe,

            "nDCG_safe":
                ndcg_safe,

            "IR_Recall_safe":
                ir_recall_safe,

            "IR_safe":
                ir_safe,

            "ANN_unsafe_but_IR_safe":
                ann_unsafe_but_ir_safe,

            "ANN_safe_but_IR_unsafe":
                ann_safe_but_ir_unsafe
        })


query_budget_df = pd.DataFrame(
    query_budget_rows
)


print(
    "\nQuery-budget dataset shape:",
    query_budget_df.shape
)


print(
    "\nExpected rows:",
    len(query_ids)
    *
    len(EF_VALUES)
)


print(
    "\nFirst 10 rows:"
)

print(

    query_budget_df
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. ANN RECALL VS IR QUALITY DISAGREEMENT
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "ANN RECALL VS IR QUALITY DISAGREEMENT"
)

print(
    "=" * 90
)


total_pairs = len(
    query_budget_df
)


ann_unsafe_ir_safe_count = int(

    query_budget_df[
        "ANN_unsafe_but_IR_safe"
    ].sum()
)


ann_safe_ir_unsafe_count = int(

    query_budget_df[
        "ANN_safe_but_IR_unsafe"
    ].sum()
)


ann_unsafe_ir_safe_rate = (

    ann_unsafe_ir_safe_count

    /

    total_pairs
)


ann_safe_ir_unsafe_rate = (

    ann_safe_ir_unsafe_count

    /

    total_pairs
)


print(
    "\nTotal query-budget pairs:",
    total_pairs
)


print(
    "ANN unsafe but IR safe:",
    ann_unsafe_ir_safe_count
)


print(
    "ANN safe but IR unsafe:",
    ann_safe_ir_unsafe_count
)


print(
    "\nANN unsafe but IR safe rate:",
    ann_unsafe_ir_safe_rate
)


print(
    "ANN safe but IR unsafe rate:",
    ann_safe_ir_unsafe_rate
)


# ============================================================
# 20. SAFETY SUMMARY BY efSearch
# ============================================================

safety_summary = (

    query_budget_df

    .groupby(
        "efSearch"
    )

    .agg({

        "ANN_Recall@10":
            "mean",

        "nDCG_loss":
            "mean",

        "IR_Recall_loss":
            "mean",

        "ANN_safe":
            "mean",

        "nDCG_safe":
            "mean",

        "IR_Recall_safe":
            "mean",

        "IR_safe":
            "mean",

        "ANN_unsafe_but_IR_safe":
            "mean",

        "ANN_safe_but_IR_unsafe":
            "mean"
    })

    .reset_index()
)


print(
    "\n"
    + "=" * 90
)

print(
    "QUALITY-SAFETY SUMMARY BY efSearch"
)

print(
    "=" * 90
)


print(

    safety_summary.to_string(
        index=False
    )
)


# ============================================================
# 21. COMPUTE ORACLE MINIMUM BUDGETS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "ORACLE QUERY-SPECIFIC BUDGETS"
)

print(
    "=" * 90
)


oracle_rows = []


for qid in query_ids:

    qdf = (

        query_budget_df[
            query_budget_df[
                "query_id"
            ]
            ==
            qid
        ]

        .sort_values(
            "efSearch"
        )
    )


    # --------------------------------------------------------
    # ANN-safe oracle
    # --------------------------------------------------------

    ann_safe_rows = qdf[
        qdf[
            "ANN_safe"
        ]
        ==
        1
    ]


    if len(
        ann_safe_rows
    ) > 0:

        min_ann_budget = int(

            ann_safe_rows
            .iloc[0][
                "efSearch"
            ]
        )

    else:

        min_ann_budget = (
            EF_VALUES[-1]
        )


    # --------------------------------------------------------
    # nDCG-safe oracle
    # --------------------------------------------------------

    ndcg_safe_rows = qdf[
        qdf[
            "nDCG_safe"
        ]
        ==
        1
    ]


    if len(
        ndcg_safe_rows
    ) > 0:

        min_ndcg_budget = int(

            ndcg_safe_rows
            .iloc[0][
                "efSearch"
            ]
        )

    else:

        min_ndcg_budget = (
            EF_VALUES[-1]
        )


    # --------------------------------------------------------
    # Recall@100-safe oracle
    # --------------------------------------------------------

    recall_safe_rows = qdf[
        qdf[
            "IR_Recall_safe"
        ]
        ==
        1
    ]


    if len(
        recall_safe_rows
    ) > 0:

        min_recall_budget = int(

            recall_safe_rows
            .iloc[0][
                "efSearch"
            ]
        )

    else:

        min_recall_budget = (
            EF_VALUES[-1]
        )


    # --------------------------------------------------------
    # Combined IR-safe oracle
    # --------------------------------------------------------

    ir_safe_rows = qdf[
        qdf[
            "IR_safe"
        ]
        ==
        1
    ]


    if len(
        ir_safe_rows
    ) > 0:

        min_ir_budget = int(

            ir_safe_rows
            .iloc[0][
                "efSearch"
            ]
        )

    else:

        min_ir_budget = (
            EF_VALUES[-1]
        )


    oracle_rows.append({

        "query_id":
            qid,

        "query_text":
            queries[qid],

        "min_ANN_safe_ef":
            min_ann_budget,

        "min_nDCG_safe_ef":
            min_ndcg_budget,

        "min_Recall100_safe_ef":
            min_recall_budget,

        "min_IR_safe_ef":
            min_ir_budget,

        "IR_minus_ANN_budget":
            (
                min_ir_budget
                -
                min_ann_budget
            )
    })


oracle_budget_df = pd.DataFrame(
    oracle_rows
)


print(
    "\nMean minimum ANN-safe budget:",
    oracle_budget_df[
        "min_ANN_safe_ef"
    ].mean()
)


print(
    "Mean minimum nDCG-safe budget:",
    oracle_budget_df[
        "min_nDCG_safe_ef"
    ].mean()
)


print(
    "Mean minimum Recall@100-safe budget:",
    oracle_budget_df[
        "min_Recall100_safe_ef"
    ].mean()
)


print(
    "Mean minimum IR-safe budget:",
    oracle_budget_df[
        "min_IR_safe_ef"
    ].mean()
)


print(
    "\nMedian minimum ANN-safe budget:",
    oracle_budget_df[
        "min_ANN_safe_ef"
    ].median()
)


print(
    "Median minimum nDCG-safe budget:",
    oracle_budget_df[
        "min_nDCG_safe_ef"
    ].median()
)


print(
    "Median minimum IR-safe budget:",
    oracle_budget_df[
        "min_IR_safe_ef"
    ].median()
)


# ============================================================
# 22. ANN-BASED VS IR-BASED ORACLE COMPARISON
# ============================================================

same_budget = (

    oracle_budget_df[
        "min_ANN_safe_ef"
    ]

    ==

    oracle_budget_df[
        "min_IR_safe_ef"
    ]
)


ir_needs_less = (

    oracle_budget_df[
        "min_IR_safe_ef"
    ]

    <

    oracle_budget_df[
        "min_ANN_safe_ef"
    ]
)


ir_needs_more = (

    oracle_budget_df[
        "min_IR_safe_ef"
    ]

    >

    oracle_budget_df[
        "min_ANN_safe_ef"
    ]
)


same_budget_rate = float(
    same_budget.mean()
)

ir_needs_less_rate = float(
    ir_needs_less.mean()
)

ir_needs_more_rate = float(
    ir_needs_more.mean()
)


print(
    "\n"
    + "=" * 90
)

print(
    "ANN-BASED VS IR-UTILITY-BASED ORACLE BUDGETS"
)

print(
    "=" * 90
)


print(
    "\nSame budget:",
    same_budget_rate
)


print(
    "IR quality allows LOWER budget:",
    ir_needs_less_rate
)


print(
    "IR quality requires HIGHER budget:",
    ir_needs_more_rate
)


# ============================================================
# 23. ORIGINAL ANN MINIMUM BUDGET
# ============================================================

minimum_budget = []


for i in range(
    len(
        query_ids
    )
):

    required_ef = (
        EF_VALUES[-1]
    )


    for ef in EF_VALUES:

        recall = (

            results_by_ef[
                ef
            ][
                "ann_recall"
            ][i]
        )


        if (
            recall
            >=
            TARGET_RECALL
        ):

            required_ef = ef

            break


    minimum_budget.append(
        required_ef
    )


minimum_budget = np.asarray(
    minimum_budget
)


print(
    "\nMinimum ANN-safe efSearch distribution:"
)


for ef in EF_VALUES:

    count = int(
        np.sum(
            minimum_budget
            ==
            ef
        )
    )

    print(
        f"ef={ef:<4}: {count} queries"
    )


print(
    "\nBudget heterogeneity:"
)


print(
    "Number of distinct optimal ANN budgets:",
    len(
        np.unique(
            minimum_budget
        )
    )
)


print(
    "Mean required ANN budget:",
    np.mean(
        minimum_budget
    )
)


print(
    "Median required ANN budget:",
    np.median(
        minimum_budget
    )
)


# ============================================================
# 24. TRAIN / TEST SPLIT FOR OLD BASELINE POLICY
# ============================================================

indices_all = np.arange(
    len(
        query_embeddings
    )
)


train_idx, test_idx = train_test_split(

    indices_all,

    test_size=0.30,

    random_state=SEED
)


# ============================================================
# 25. CREATE SIMPLE QUERY FEATURES
#
# This remains the ORIGINAL baseline policy.
#
# Later we will replace this objective with:
# P(IR-safe | query, budget)
# ============================================================

def create_query_features(
    embeddings
):

    features = []


    for emb in embeddings:

        stats = [

            np.linalg.norm(
                emb
            ),

            np.mean(
                emb
            ),

            np.std(
                emb
            ),

            np.min(
                emb
            ),

            np.max(
                emb
            ),

            np.mean(
                np.abs(
                    emb
                )
            ),

            (
                np.sum(
                    emb > 0
                )
                /
                len(
                    emb
                )
            ),

            np.max(
                np.abs(
                    emb
                )
            )
        ]


        feature = np.concatenate([

            emb,

            np.asarray(
                stats
            )
        ])


        features.append(
            feature
        )


    return np.asarray(
        features,
        dtype=np.float32
    )


X = create_query_features(
    query_embeddings
)

y = minimum_budget


X_train = X[
    train_idx
]

X_test = X[
    test_idx
]

y_train = y[
    train_idx
]

y_test = y[
    test_idx
]


print(
    "\nBaseline feature shape:",
    X.shape
)


# ============================================================
# 26. ORIGINAL DECISION-TREE BUDGET CLASSIFIER
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


classification_accuracy = accuracy_score(
    y_test,
    predicted_budget
)


print(
    "\nExact budget classification accuracy:",
    classification_accuracy
)


print(
    "Predicted budgets:",
    Counter(
        predicted_budget
    )
)


# ============================================================
# 27. CONSERVATIVE ONE-LEVEL SAFETY MARGIN
# ============================================================

def increase_budget_once(
    pred
):

    pos = EF_VALUES.index(
        int(
            pred
        )
    )

    new_pos = min(
        pos + 1,
        len(
            EF_VALUES
        ) - 1
    )

    return EF_VALUES[
        new_pos
    ]


safe_predicted_budget = np.asarray([

    increase_budget_once(
        x
    )

    for x
    in predicted_budget
])


print(
    "Safe predicted budgets:",
    Counter(
        safe_predicted_budget
    )
)


# ============================================================
# 28. SELECT FIXED HNSW BASELINE
# ============================================================

FIXED_MAX_VIOLATION = 0.05

fixed_budget = (
    EF_VALUES[-1]
)


for ef in EF_VALUES:

    recalls = (

        results_by_ef[
            ef
        ][
            "ann_recall"
        ][
            train_idx
        ]
    )


    violation = np.mean(
        recalls
        <
        TARGET_RECALL
    )


    if (
        violation
        <=
        FIXED_MAX_VIOLATION
    ):

        fixed_budget = ef

        break


print(
    "\nSelected fixed baseline efSearch:",
    fixed_budget
)


# ============================================================
# 29. RUN QUERY-SPECIFIC BASELINE POLICY
# ============================================================

adaptive_results = []

adaptive_latencies = []

adaptive_recalls = []

adaptive_ndcg = []

adaptive_ir_recall = []


print(
    "\nRunning adaptive baseline policy..."
)


# Warm-up
hnsw_index.hnsw.efSearch = (
    EF_VALUES[-1]
)

_ = hnsw_index.search(
    query_embeddings[:10],
    IR_K
)


for pos, query_index in enumerate(
    test_idx
):

    ef = int(
        safe_predicted_budget[
            pos
        ]
    )


    hnsw_index.hnsw.efSearch = (
        ef
    )


    q = query_embeddings[
        query_index:
        query_index + 1
    ]


    start = (
        time.perf_counter_ns()
    )


    _, idx = (
        hnsw_index.search(
            q,
            IR_K
        )
    )


    end = (
        time.perf_counter_ns()
    )


    latency_ms = (

        end - start

    ) / 1_000_000


    result = idx[0]


    adaptive_results.append(
        result
    )


    adaptive_latencies.append(
        latency_ms
    )


    adaptive_recalls.append(

        ann_recall_at_k(
            exact_indices[
                query_index
            ],
            result,
            ANN_K
        )
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


adaptive_recalls = np.asarray(
    adaptive_recalls,
    dtype=np.float64
)

adaptive_latencies = np.asarray(
    adaptive_latencies,
    dtype=np.float64
)

adaptive_ndcg = np.asarray(
    adaptive_ndcg,
    dtype=np.float64
)

adaptive_ir_recall = np.asarray(
    adaptive_ir_recall,
    dtype=np.float64
)


# ============================================================
# 30. RUN FIXED BASELINE ON SAME TEST QUERIES
# ============================================================

hnsw_index.hnsw.efSearch = (
    fixed_budget
)


fixed_latencies = []

fixed_recalls = []

fixed_ndcg = []

fixed_ir_recall = []


for query_index in test_idx:

    q = query_embeddings[
        query_index:
        query_index + 1
    ]


    start = (
        time.perf_counter_ns()
    )


    _, idx = (
        hnsw_index.search(
            q,
            IR_K
        )
    )


    end = (
        time.perf_counter_ns()
    )


    latency_ms = (

        end - start

    ) / 1_000_000


    result = idx[0]


    fixed_latencies.append(
        latency_ms
    )


    fixed_recalls.append(

        ann_recall_at_k(
            exact_indices[
                query_index
            ],
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


fixed_recalls = np.asarray(
    fixed_recalls,
    dtype=np.float64
)

fixed_latencies = np.asarray(
    fixed_latencies,
    dtype=np.float64
)

fixed_ndcg = np.asarray(
    fixed_ndcg,
    dtype=np.float64
)

fixed_ir_recall = np.asarray(
    fixed_ir_recall,
    dtype=np.float64
)


# ============================================================
# 31. POLICY INFERENCE OVERHEAD
# ============================================================

start = (
    time.perf_counter_ns()
)

_ = policy.predict(
    X_test
)

end = (
    time.perf_counter_ns()
)


policy_overhead_ms = (

    (
        end - start
    )

    /

    1_000_000

    /

    len(
        X_test
    )
)


# ============================================================
# 32. FINAL OLD-BASELINE COMPARISON
# ============================================================

fixed_violation = np.mean(
    fixed_recalls
    <
    TARGET_RECALL
)


adaptive_violation = np.mean(
    adaptive_recalls
    <
    TARGET_RECALL
)


comparison = pd.DataFrame([

    {

        "Policy":
            f"Fixed ef={fixed_budget}",

        "Mean efSearch":
            float(
                fixed_budget
            ),

        "ANN Recall@10":
            np.mean(
                fixed_recalls
            ),

        "IR Recall@100":
            np.mean(
                fixed_ir_recall
            ),

        "nDCG@10":
            np.mean(
                fixed_ndcg
            ),

        "Violation Rate":
            fixed_violation,

        "Mean Latency ms":
            np.mean(
                fixed_latencies
            ),

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


print(
    "\n"
    + "=" * 90
)

print(
    "ORIGINAL BASELINE POLICY RESULTS"
)

print(
    "=" * 90
)


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
# 33. POLICY BEHAVIOUR
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


print(
    "\nPolicy behaviour:"
)


print(
    "Queries using LESS effort:",
    np.mean(
        saving
    )
)


print(
    "Queries using SAME effort:",
    np.mean(
        same
    )
)


print(
    "Queries using MORE effort:",
    np.mean(
        higher
    )
)


# ============================================================
# 34. PLOT 1:
# ANN QUALITY VS SEARCH BUDGET
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    summary[
        "efSearch"
    ],
    summary[
        "ANN_Recall@10"
    ],
    marker="o"
)


plt.axhline(
    TARGET_RECALL,
    linestyle="--",
    label="Target recall"
)


plt.xlabel(
    "HNSW efSearch"
)

plt.ylabel(
    "Mean ANN Recall@10"
)

plt.title(
    "ANN Quality vs HNSW Search Budget"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "results_phase1/ann_quality_vs_budget.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nSaved: "
    "results_phase1/ann_quality_vs_budget.png"
)


# ============================================================
# 35. PLOT 2:
# ANN CONSTRAINT VIOLATION RATE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    summary[
        "efSearch"
    ],
    summary[
        "Violation_Rate"
    ],
    marker="o"
)


plt.xlabel(
    "HNSW efSearch"
)

plt.ylabel(
    "Constraint Violation Rate"
)

plt.title(
    "Recall Constraint Violations vs Search Budget"
)

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "results_phase1/violation_rate_vs_budget.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved: "
    "results_phase1/violation_rate_vs_budget.png"
)


# ============================================================
# 36. PLOT 3:
# ORIGINAL ANN ORACLE BUDGET DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)


values, counts = np.unique(
    minimum_budget,
    return_counts=True
)


plt.bar(
    values.astype(
        str
    ),
    counts
)


plt.xlabel(
    "Minimum ANN-safe efSearch"
)

plt.ylabel(
    "Number of Queries"
)

plt.title(
    "Query-Specific ANN Budget Distribution"
)

plt.tight_layout()


plt.savefig(
    "results_phase1/budget_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved: "
    "results_phase1/budget_distribution.png"
)


# ============================================================
# 37. PLOT 4:
# ANN ORACLE VS IR ORACLE MEAN BUDGET
# ============================================================

oracle_names = [
    "ANN-safe",
    "nDCG-safe",
    "IR-safe"
]


oracle_means = [

    oracle_budget_df[
        "min_ANN_safe_ef"
    ].mean(),

    oracle_budget_df[
        "min_nDCG_safe_ef"
    ].mean(),

    oracle_budget_df[
        "min_IR_safe_ef"
    ].mean()
]


plt.figure(
    figsize=(8, 5)
)


plt.bar(
    oracle_names,
    oracle_means
)


plt.ylabel(
    "Mean Minimum efSearch"
)

plt.xlabel(
    "Oracle Quality Objective"
)

plt.title(
    "ANN-Based vs Ranking-Utility-Based Oracle Budgets"
)

plt.tight_layout()


plt.savefig(
    "results_phase1/oracle_budget_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved: "
    "results_phase1/oracle_budget_comparison.png"
)


# ============================================================
# 38. PLOT 5:
# DISAGREEMENT RATE BY efSearch
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    safety_summary[
        "efSearch"
    ],
    safety_summary[
        "ANN_unsafe_but_IR_safe"
    ],
    marker="o",
    label="ANN unsafe / IR safe"
)


plt.plot(
    safety_summary[
        "efSearch"
    ],
    safety_summary[
        "ANN_safe_but_IR_unsafe"
    ],
    marker="o",
    label="ANN safe / IR unsafe"
)


plt.xlabel(
    "HNSW efSearch"
)

plt.ylabel(
    "Fraction of Queries"
)

plt.title(
    "ANN Recall vs IR Utility Disagreement"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "results_phase1/ann_ir_disagreement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved: "
    "results_phase1/ann_ir_disagreement.png"
)


# ============================================================
# 39. SAVE BASELINE CSV FILES
# ============================================================

summary.to_csv(
    "results_phase1/efsearch_sweep.csv",
    index=False
)


comparison.to_csv(
    "results_phase1/policy_comparison.csv",
    index=False
)


query_results = pd.DataFrame({

    "query_id":
        np.asarray(
            query_ids
        )[
            test_idx
        ],

    "required_budget":
        minimum_budget[
            test_idx
        ],

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
    "results_phase1/query_level_results.csv",
    index=False
)


# ============================================================
# 40. SAVE PHASE-1 RESEARCH FILES
# ============================================================

query_budget_df.to_csv(
    "results_phase1/query_budget_safety_labels.csv",
    index=False
)


oracle_budget_df.to_csv(
    "results_phase1/oracle_budget_comparison.csv",
    index=False
)


safety_summary.to_csv(
    "results_phase1/safety_summary_by_ef.csv",
    index=False
)


# ============================================================
# 41. SAVE ONE SUMMARY TEXT FILE
# ============================================================

with open(
    "results_phase1/phase1_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "QUERY-SPECIFIC ANN BUDGET "
        "PHASE-1 SUMMARY\n"
    )

    f.write(
        "=" * 70
        + "\n"
    )

    f.write(
        f"Dataset: {DATASET}\n"
    )

    f.write(
        f"Queries: {len(query_ids)}\n"
    )

    f.write(
        f"Query-budget pairs: "
        f"{len(query_budget_df)}\n"
    )

    f.write(
        f"Target ANN Recall@10: "
        f"{TARGET_RECALL}\n"
    )

    f.write(
        f"nDCG drop tolerance: "
        f"{NDCG_DROP_TOLERANCE}\n"
    )

    f.write(
        f"Recall@100 drop tolerance: "
        f"{IR_RECALL_DROP_TOLERANCE}\n\n"
    )

    f.write(
        "DISAGREEMENT\n"
    )

    f.write(
        f"ANN unsafe but IR safe: "
        f"{ann_unsafe_ir_safe_count} "
        f"({ann_unsafe_ir_safe_rate:.4f})\n"
    )

    f.write(
        f"ANN safe but IR unsafe: "
        f"{ann_safe_ir_unsafe_count} "
        f"({ann_safe_ir_unsafe_rate:.4f})\n\n"
    )

    f.write(
        "ORACLE BUDGETS\n"
    )

    f.write(
        "Mean ANN-safe ef: "
        f"{oracle_budget_df['min_ANN_safe_ef'].mean():.4f}\n"
    )

    f.write(
        "Mean nDCG-safe ef: "
        f"{oracle_budget_df['min_nDCG_safe_ef'].mean():.4f}\n"
    )

    f.write(
        "Mean IR-safe ef: "
        f"{oracle_budget_df['min_IR_safe_ef'].mean():.4f}\n\n"
    )

    f.write(
        "ANN VS IR ORACLE\n"
    )

    f.write(
        f"Same budget: "
        f"{same_budget_rate:.4f}\n"
    )

    f.write(
        f"IR allows lower budget: "
        f"{ir_needs_less_rate:.4f}\n"
    )

    f.write(
        f"IR requires higher budget: "
        f"{ir_needs_more_rate:.4f}\n"
    )


# ============================================================
# 42. FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PHASE 1 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 90
)


print(
    "\nSaved baseline files:"
)

print(
    "  results/efsearch_sweep.csv"
)

print(
    "  results/policy_comparison.csv"
)

print(
    "  results/query_level_results.csv"
)


print(
    "\nSaved Phase-1 research files:"
)

print(
    "  results/query_budget_safety_labels.csv"
)

print(
    "  results/oracle_budget_comparison.csv"
)

print(
    "  results/safety_summary_by_ef.csv"
)

print(
    "  results/phase1_summary.txt"
)


print(
    "\nSaved figures:"
)

print(
    "  results/ann_quality_vs_budget.png"
)

print(
    "  results/violation_rate_vs_budget.png"
)

print(
    "  results/budget_distribution.png"
)

print(
    "  results/oracle_budget_comparison.png"
)

print(
    "  results/ann_ir_disagreement.png"
)


print(
    "\nImportant numbers to send me:"
)

print(
    "1. Query-budget dataset shape =",
    query_budget_df.shape
)

print(
    "2. ANN unsafe but IR safe =",
    ann_unsafe_ir_safe_count
)

print(
    "3. ANN safe but IR unsafe =",
    ann_safe_ir_unsafe_count
)

print(
    "4. Mean minimum ANN-safe budget =",
    oracle_budget_df[
        "min_ANN_safe_ef"
    ].mean()
)

print(
    "5. Mean minimum nDCG-safe budget =",
    oracle_budget_df[
        "min_nDCG_safe_ef"
    ].mean()
)

print(
    "6. Mean minimum IR-safe budget =",
    oracle_budget_df[
        "min_IR_safe_ef"
    ].mean()
)

print(
    "7. Same ANN/IR oracle budget =",
    same_budget_rate
)

print(
    "8. IR allows LOWER budget =",
    ir_needs_less_rate
)

print(
    "9. IR requires HIGHER budget =",
    ir_needs_more_rate
)


print(
    "\nDONE."
)
