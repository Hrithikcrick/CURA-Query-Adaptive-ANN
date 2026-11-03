# ============================================================
# 07_cura_controller.py
#
# COL764 PROJECT
#
# CURA:
# Calibrated Utility-Risk Allocation
#
# ------------------------------------------------------------
# RESEARCH QUESTION
# ------------------------------------------------------------
#
# Instead of asking:
#
#   "Have we recovered enough exact ANN neighbours?"
#
# CURA asks:
#
#   "What is the probability that using this search budget
#    will cause downstream nDCG@10 loss > delta?"
#
# We choose the smallest efSearch whose predicted downstream
# failure risk is sufficiently low.
#
#
# IMPORTANT EXPERIMENTAL DESIGN
# ------------------------------------------------------------
#
# TRAIN:
#   60% MS MARCO dev queries
#
# CALIBRATION:
#   20% MS MARCO dev queries
#
# HELD-OUT TEST:
#   20% MS MARCO dev queries
#
# OOD TEST:
#   ALL TREC-DL-2019 queries
#
#
# NO LEAKAGE:
# ------------------------------------------------------------
#
# qrels / exact neighbours / nDCG / ANN recall are used
# ONLY as training labels and evaluation targets.
#
# They are NEVER used as inference features.
#
#
# DEPLOYABLE FEATURES:
# ------------------------------------------------------------
#
# Query-only features:
#   - query length
#   - token count
#   - average token length
#   - query embedding PCA components
#   - candidate efSearch
#
# Probe features:
#   - cheap efSearch=10 HNSW search
#   - top score statistics
#   - score margins
#   - HNSW distance computations
#   - HNSW hops
#
#
# METHODS:
# ------------------------------------------------------------
#
# 1. Fixed-IR
# 2. ANN-Risk Controller
# 3. CURA Query-Only
# 4. CURA Full (query + cheap probe)
# 5. IR Oracle
# 6. ANN Oracle
#
#
# NOTE:
# ------------------------------------------------------------
#
# ANN-Risk is an ANN-target adaptive baseline.
# It is NOT claimed to be DARTH.
#
# This script does NOT claim to reproduce Patience or DARTH.
# Those remain external baselines.
#
# ============================================================


# ============================================================
# 0. WINDOWS / PATH SETUP
# ============================================================

import os

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEMP_DIR = os.path.join(
    SCRIPT_DIR,
    "ir_datasets_tmp"
)

os.makedirs(
    TEMP_DIR,
    exist_ok=True
)

os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["PYTHONUTF8"] = "1"


# ============================================================
# 1. IMPORTS
# ============================================================

import sys
import json
import math
import time
import argparse
import platform
import warnings

warnings.filterwarnings(
    "ignore"
)

import numpy as np
import pandas as pd

import faiss

from tqdm import tqdm

from sklearn.decomposition import PCA

from sklearn.ensemble import (
    HistGradientBoostingClassifier
)

from sklearn.isotonic import (
    IsotonicRegression
)

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 2. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "CURA: Calibrated Utility-Risk Allocation"
    )
)

parser.add_argument(
    "--n_docs",
    type=int,
    default=1_000_000
)

parser.add_argument(
    "--n_dev_queries",
    type=int,
    default=1000
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

parser.add_argument(
    "--risk_alpha",
    type=float,
    default=0.05,
    help=(
        "Target query-level failure probability."
    )
)

parser.add_argument(
    "--ndcg_tolerance",
    type=float,
    default=0.025
)

parser.add_argument(
    "--ann_target",
    type=float,
    default=0.90
)

parser.add_argument(
    "--bootstrap",
    type=int,
    default=5000
)

parser.add_argument(
    "--permutations",
    type=int,
    default=10000
)

args = parser.parse_args()


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = args.seed

np.random.seed(
    SEED
)

RNG = np.random.default_rng(
    SEED
)


# ============================================================
# 4. MAIN CONSTANTS
# ============================================================

RISK_ALPHA = args.risk_alpha

NDCG_TOLERANCE = args.ndcg_tolerance

ANN_TARGET = args.ann_target

PROBE_EF = 10

PROBE_K = 10

PCA_COMPONENTS = 16

HNSW_M = 32

EF_CONSTRUCTION = 200


# ============================================================
# 5. PATHS
# ============================================================

RUN_NAME = (
    f"msmarco_{args.n_docs}"
    f"_q{args.n_dev_queries}"
)


BASE_DIR = os.path.join(
    SCRIPT_DIR,
    "msmarco_large",
    RUN_NAME
)


CACHE_DIR = os.path.join(
    BASE_DIR,
    "cache"
)


PHASE04_RESULTS = os.path.join(
    BASE_DIR,
    "results"
)


PHASE05_RESULTS = os.path.join(
    BASE_DIR,
    "research_grade_validation"
)


RESULTS_DIR = os.path.join(
    BASE_DIR,
    "cura_controller"
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


QUERY_FILE = os.path.join(
    PHASE04_RESULTS,
    "queries.csv"
)


QUERY_EMBEDDINGS_FILE = os.path.join(
    CACHE_DIR,
    "query_embeddings.npy"
)


HNSW_FILE = os.path.join(
    CACHE_DIR,
    (
        f"hnsw_M{HNSW_M}"
        f"_efC{EF_CONSTRUCTION}.faiss"
    )
)


TOP10_RESULTS_FILE = os.path.join(
    PHASE05_RESULTS,
    "top10_query_budget_results.csv"
)


PROBE_CACHE_FILE = os.path.join(
    RESULTS_DIR,
    "ef10_probe_features.csv"
)


# ============================================================
# 6. VERIFY INPUT FILES
# ============================================================

required_files = [
    QUERY_FILE,
    QUERY_EMBEDDINGS_FILE,
    HNSW_FILE,
    TOP10_RESULTS_FILE
]


missing_files = [

    path

    for path in required_files

    if not os.path.exists(
        path
    )
]


if missing_files:

    print(
        "\nERROR: Required cached files are missing."
    )

    print(
        "\nMissing:"
    )

    for path in missing_files:

        print(
            "  ",
            path
        )

    print(
        "\nYour Phase 04 and Phase 05 "
        "1M experiments must exist first."
    )

    sys.exit(
        1
    )


# ============================================================
# 7. HEADER
# ============================================================

print(
    "=" * 100
)

print(
    "CURA — CALIBRATED UTILITY-RISK ALLOCATION"
)

print(
    "=" * 100
)

print(
    "Experiment:",
    RUN_NAME
)

print(
    "nDCG loss tolerance:",
    NDCG_TOLERANCE
)

print(
    "ANN Recall@10 target:",
    ANN_TARGET
)

print(
    "Risk SLA alpha:",
    RISK_ALPHA
)

print(
    "Probe efSearch:",
    PROBE_EF
)

print(
    "Random seed:",
    SEED
)


# ============================================================
# 8. LOAD EXISTING DATA
# ============================================================

print(
    "\nLoading cached experiment data..."
)


query_df = pd.read_csv(
    QUERY_FILE,
    dtype={
        "query_key": str,
        "raw_query_id": str,
        "source": str,
        "query_text": str
    }
)


query_df[
    "query_key"
] = query_df[
    "query_key"
].astype(
    str
)


query_embeddings = np.load(
    QUERY_EMBEDDINGS_FILE
).astype(
    np.float32
)


budget_df = pd.read_csv(
    TOP10_RESULTS_FILE,
    dtype={
        "query_key": str,
        "source": str
    }
)


if len(
    query_embeddings
) != len(
    query_df
):

    raise RuntimeError(
        "Query embeddings and queries.csv "
        "do not have the same number of rows."
    )


# ============================================================
# 9. CANDIDATE BUDGETS
# ============================================================

EF_VALUES = sorted(

    int(x)

    for x in budget_df[
        "efSearch"
    ].unique()
)


print(
    "\nCandidate efSearch values:"
)

print(
    EF_VALUES
)


if PROBE_EF not in EF_VALUES:

    raise RuntimeError(
        f"Probe ef={PROBE_EF} not present "
        "in Phase-05 results."
    )


# ============================================================
# 10. CREATE LABELS
#
# IMPORTANT:
#
# These are labels only.
# They must never enter the feature matrix.
# ============================================================

budget_df[
    "IR_unsafe"
] = (

    budget_df[
        "nDCG_loss"
    ]

    >

    NDCG_TOLERANCE
).astype(
    int
)


budget_df[
    "ANN_unsafe"
] = (

    budget_df[
        "ANN_Recall@10"
    ]

    <

    ANN_TARGET
).astype(
    int
)


print(
    "\nOverall row-level IR unsafe rate:",
    budget_df[
        "IR_unsafe"
    ].mean()
)


print(
    "Overall row-level ANN unsafe rate:",
    budget_df[
        "ANN_unsafe"
    ].mean()
)


# ============================================================
# 11. QUERY-DISJOINT SPLIT
#
# MS MARCO:
#
#   60% train
#   20% calibration
#   20% final test
#
# TREC-DL:
#
#   100% OOD test only
# ============================================================

msmarco_query_keys = (

    query_df[

        query_df[
            "source"
        ]
        ==
        "msmarco-dev-small"

    ][
        "query_key"
    ]

    .astype(
        str
    )

    .tolist()
)


trec_query_keys = (

    query_df[

        query_df[
            "source"
        ]
        ==
        "trec-dl-2019"

    ][
        "query_key"
    ]

    .astype(
        str
    )

    .tolist()
)


split_rng = np.random.default_rng(
    SEED
)


msmarco_query_keys = np.array(
    msmarco_query_keys,
    dtype=object
)


split_rng.shuffle(
    msmarco_query_keys
)


n_total = len(
    msmarco_query_keys
)


n_train = int(
    0.60
    *
    n_total
)


n_calibration = int(
    0.20
    *
    n_total
)


train_query_keys = set(

    msmarco_query_keys[
        :n_train
    ]
)


calibration_query_keys = set(

    msmarco_query_keys[
        n_train:
        n_train + n_calibration
    ]
)


test_query_keys = set(

    msmarco_query_keys[
        n_train + n_calibration:
    ]
)


ood_query_keys = set(
    trec_query_keys
)


print(
    "\nQuery split:"
)

print(
    "Train:",
    len(
        train_query_keys
    )
)

print(
    "Calibration:",
    len(
        calibration_query_keys
    )
)

print(
    "Held-out MS MARCO test:",
    len(
        test_query_keys
    )
)

print(
    "OOD TREC-DL test:",
    len(
        ood_query_keys
    )
)


# ============================================================
# 12. SAVE SPLIT
# ============================================================

split_metadata = {

    "seed":
        SEED,

    "train_queries":
        sorted(
            train_query_keys
        ),

    "calibration_queries":
        sorted(
            calibration_query_keys
        ),

    "test_queries":
        sorted(
            test_query_keys
        ),

    "ood_queries":
        sorted(
            ood_query_keys
        )
}


with open(

    os.path.join(
        RESULTS_DIR,
        "query_split.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        split_metadata,
        f,
        indent=2
    )


# ============================================================
# 13. MAP QUERY KEY -> EMBEDDING ROW
# ============================================================

query_index_map = {

    str(
        row[
            "query_key"
        ]
    ):
    i

    for i, row
    in query_df.iterrows()
}


# ============================================================
# 14. QUERY TEXT FEATURES
# ============================================================

query_feature_rows = []


for _, row in query_df.iterrows():

    query_key = str(
        row[
            "query_key"
        ]
    )


    text = str(
        row[
            "query_text"
        ]
    )


    tokens = text.split()


    token_count = len(
        tokens
    )


    char_count = len(
        text
    )


    if token_count > 0:

        average_token_length = np.mean(
            [
                len(token)
                for token
                in tokens
            ]
        )

    else:

        average_token_length = 0.0


    digit_count = sum(
        character.isdigit()
        for character
        in text
    )


    question_mark = int(
        "?"
        in text
    )


    query_feature_rows.append({

        "query_key":
            query_key,

        "query_char_count":
            char_count,

        "query_token_count":
            token_count,

        "query_avg_token_length":
            average_token_length,

        "query_digit_count":
            digit_count,

        "query_has_question_mark":
            question_mark
    })


query_feature_df = pd.DataFrame(
    query_feature_rows
)


# ============================================================
# 15. PCA ON QUERY EMBEDDINGS
#
# IMPORTANT:
# Fit PCA ONLY on training queries.
# ============================================================

train_embedding_indices = [

    query_index_map[
        query_key
    ]

    for query_key
    in train_query_keys
]


n_pca = min(

    PCA_COMPONENTS,

    len(
        train_embedding_indices
    ),

    query_embeddings.shape[
        1
    ]
)


pca = PCA(
    n_components=n_pca,
    random_state=SEED
)


pca.fit(

    query_embeddings[
        train_embedding_indices
    ]
)


all_query_pca = pca.transform(
    query_embeddings
)


for component_index in range(
    n_pca
):

    query_feature_df[
        f"query_pca_{component_index}"
    ] = all_query_pca[
        :,
        component_index
    ]


# ============================================================
# 16. LOAD HNSW FOR CHEAP PROBE
# ============================================================

def hnsw_stats_available():

    try:

        _ = faiss.cvar.hnsw_stats.ndis

        return True

    except Exception:

        return False


HAS_HNSW_STATS = hnsw_stats_available()


def reset_hnsw_stats():

    if not HAS_HNSW_STATS:

        return

    try:

        faiss.cvar.hnsw_stats.reset()

    except Exception:

        pass


def read_hnsw_stats():

    if not HAS_HNSW_STATS:

        return (
            np.nan,
            np.nan
        )

    try:

        return (

            float(
                faiss.cvar.hnsw_stats.ndis
            ),

            float(
                faiss.cvar.hnsw_stats.nhops
            )
        )

    except Exception:

        return (
            np.nan,
            np.nan
        )


# ============================================================
# 17. CHEAP ef=10 PROBE FEATURES
#
# This is a deployable probe.
#
# It does NOT use:
#   - exact search
#   - qrels
#   - nDCG
#   - ANN recall
# ============================================================

if os.path.exists(
    PROBE_CACHE_FILE
):

    print(
        "\nLoading cached ef=10 probe features..."
    )

    probe_df = pd.read_csv(
        PROBE_CACHE_FILE,
        dtype={
            "query_key": str
        }
    )


else:

    print(
        "\nRunning cheap ef=10 probe search..."
    )


    hnsw_index = faiss.read_index(
        HNSW_FILE
    )


    faiss.omp_set_num_threads(
        1
    )


    hnsw_index.hnsw.efSearch = PROBE_EF


    # Warm-up

    for i in range(
        min(
            30,
            len(
                query_embeddings
            )
        )
    ):

        hnsw_index.search(

            np.ascontiguousarray(

                query_embeddings[
                    i:i + 1
                ],

                dtype=np.float32
            ),

            PROBE_K
        )


    probe_rows = []


    for i, row in tqdm(

        query_df.iterrows(),

        total=len(
            query_df
        ),

        desc="CURA ef=10 probe"
    ):

        query_vector = np.ascontiguousarray(

            query_embeddings[
                i:i + 1
            ],

            dtype=np.float32
        )


        reset_hnsw_stats()


        start_ns = time.perf_counter_ns()


        scores, indices = hnsw_index.search(

            query_vector,

            PROBE_K
        )


        end_ns = time.perf_counter_ns()


        scores = scores[
            0
        ].astype(
            np.float64
        )


        ndis, nhops = read_hnsw_stats()


        latency_ms = (

            end_ns
            -
            start_ns

        ) / 1_000_000.0


        top1 = float(
            scores[
                0
            ]
        )


        top2 = float(
            scores[
                1
            ]
        )


        top5 = float(
            scores[
                4
            ]
        )


        top9 = float(
            scores[
                8
            ]
        )


        top10 = float(
            scores[
                9
            ]
        )


        mean_score = float(
            np.mean(
                scores
            )
        )


        std_score = float(
            np.std(
                scores
            )
        )


        top5_mean = float(
            np.mean(
                scores[
                    :5
                ]
            )
        )


        bottom5_mean = float(
            np.mean(
                scores[
                    5:
                ]
            )
        )


        probe_rows.append({

            "query_key":
                str(
                    row[
                        "query_key"
                    ]
                ),

            "probe_top1":
                top1,

            "probe_top2":
                top2,

            "probe_top5":
                top5,

            "probe_top9":
                top9,

            "probe_top10":
                top10,

            "probe_mean":
                mean_score,

            "probe_std":
                std_score,

            "probe_gap_1_2":
                top1
                -
                top2,

            "probe_gap_1_10":
                top1
                -
                top10,

            "probe_gap_5_10":
                top5
                -
                top10,

            "probe_gap_9_10":
                top9
                -
                top10,

            "probe_top5_mean":
                top5_mean,

            "probe_bottom5_mean":
                bottom5_mean,

            "probe_ndis":
                ndis,

            "probe_nhops":
                nhops,

            "probe_latency_ms":
                latency_ms
        })


    probe_df = pd.DataFrame(
        probe_rows
    )


    probe_df.to_csv(

        PROBE_CACHE_FILE,

        index=False
    )


# ============================================================
# 18. BUILD MODEL DATASET
# ============================================================

model_df = (

    budget_df

    .merge(

        query_feature_df,

        on="query_key",

        how="left"
    )

    .merge(

        probe_df,

        on="query_key",

        how="left"
    )
)


# Candidate-budget features

model_df[
    "candidate_ef"
] = model_df[
    "efSearch"
].astype(
    float
)


model_df[
    "candidate_log_ef"
] = np.log1p(
    model_df[
        "candidate_ef"
    ]
)


model_df[
    "candidate_ef_over_probe"
] = (

    model_df[
        "candidate_ef"
    ]

    /

    float(
        PROBE_EF
    )
)


# ============================================================
# 19. STRICT FEATURE SETS
#
# NO target-derived values allowed here.
# ============================================================

PCA_COLUMNS = [

    f"query_pca_{i}"

    for i in range(
        n_pca
    )
]


QUERY_ONLY_FEATURES = [

    "query_char_count",
    "query_token_count",
    "query_avg_token_length",
    "query_digit_count",
    "query_has_question_mark",

    "candidate_ef",
    "candidate_log_ef",
    "candidate_ef_over_probe",

] + PCA_COLUMNS


PROBE_FEATURES = [

    "probe_top1",
    "probe_top2",
    "probe_top5",
    "probe_top9",
    "probe_top10",

    "probe_mean",
    "probe_std",

    "probe_gap_1_2",
    "probe_gap_1_10",
    "probe_gap_5_10",
    "probe_gap_9_10",

    "probe_top5_mean",
    "probe_bottom5_mean",

    "probe_ndis",
    "probe_nhops"
]


FULL_FEATURES = (

    QUERY_ONLY_FEATURES

    +

    PROBE_FEATURES
)


# ============================================================
# 20. VERIFY NO LABEL LEAKAGE
# ============================================================

FORBIDDEN_FEATURES = {

    "ANN_Recall@10",
    "nDCG@10",
    "MRR@10",

    "nDCG_loss",
    "MRR_loss",

    "IR_unsafe",
    "ANN_unsafe",

    "boundary_gap_10_11",
    "top1_top10_gap",
    "mean_top10_similarity",

    "positive_qrel_count"
}


used_features = set(

    QUERY_ONLY_FEATURES

    +

    FULL_FEATURES
)


leaked = (

    used_features

    &

    FORBIDDEN_FEATURES
)


if leaked:

    raise RuntimeError(
        "LEAKAGE DETECTED: "
        +
        str(
            leaked
        )
    )


print(
    "\nLeakage check: PASSED"
)


# ============================================================
# 21. SPLIT ROWS BY QUERY
# ============================================================

train_mask = model_df[
    "query_key"
].isin(
    train_query_keys
)


calibration_mask = model_df[
    "query_key"
].isin(
    calibration_query_keys
)


test_mask = model_df[
    "query_key"
].isin(
    test_query_keys
)


ood_mask = model_df[
    "query_key"
].isin(
    ood_query_keys
)


# ============================================================
# 22. CLASS-BALANCED SAMPLE WEIGHTS
# ============================================================

def balanced_weights(
    y
):

    y = np.asarray(
        y,
        dtype=int
    )


    positives = np.sum(
        y == 1
    )


    negatives = np.sum(
        y == 0
    )


    if positives == 0:

        return np.ones(
            len(
                y
            ),
            dtype=np.float64
        )


    positive_weight = (

        negatives

        /

        positives
    )


    weights = np.ones(
        len(
            y
        ),
        dtype=np.float64
    )


    weights[
        y == 1
    ] = positive_weight


    return weights


# ============================================================
# 23. MODEL TRAINING
# ============================================================

def train_risk_model(
    feature_columns,
    target_column,
    name
):

    X_train = model_df.loc[
        train_mask,
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )


    y_train = model_df.loc[
        train_mask,
        target_column
    ].to_numpy(
        dtype=int
    )


    print(
        f"\nTraining {name}..."
    )


    print(
        "Training rows:",
        len(
            y_train
        )
    )


    print(
        "Unsafe rate:",
        y_train.mean()
    )


    classifier = HistGradientBoostingClassifier(

        learning_rate=0.05,

        max_iter=250,

        max_leaf_nodes=15,

        min_samples_leaf=20,

        l2_regularization=1.0,

        random_state=SEED
    )


    classifier.fit(

        X_train,

        y_train,

        sample_weight=balanced_weights(
            y_train
        )
    )


    return classifier


# ============================================================
# 24. TRAIN THREE MODELS
# ============================================================

cura_query_model = train_risk_model(

    QUERY_ONLY_FEATURES,

    "IR_unsafe",

    "CURA Query-Only"
)


cura_full_model = train_risk_model(

    FULL_FEATURES,

    "IR_unsafe",

    "CURA Full"
)


ann_risk_model = train_risk_model(

    FULL_FEATURES,

    "ANN_unsafe",

    "ANN-Risk Controller"
)


# ============================================================
# 25. PROBABILITY CALIBRATION
#
# Isotonic calibration is fit ONLY on calibration queries.
# ============================================================

def fit_probability_calibrator(
    model,
    feature_columns,
    target_column,
    name
):

    X_cal = model_df.loc[
        calibration_mask,
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )


    y_cal = model_df.loc[
        calibration_mask,
        target_column
    ].to_numpy(
        dtype=int
    )


    raw_probability = model.predict_proba(
        X_cal
    )[
        :,
        1
    ]


    calibrator = IsotonicRegression(

        y_min=0.0,

        y_max=1.0,

        out_of_bounds="clip"
    )


    calibrator.fit(

        raw_probability,

        y_cal
    )


    calibrated_probability = calibrator.transform(
        raw_probability
    )


    brier = brier_score_loss(

        y_cal,

        calibrated_probability
    )


    try:

        auc = roc_auc_score(

            y_cal,

            calibrated_probability
        )

    except Exception:

        auc = np.nan


    print(
        f"\n{name} calibration:"
    )


    print(
        "Brier score:",
        brier
    )


    print(
        "Calibration AUC:",
        auc
    )


    return calibrator


query_calibrator = fit_probability_calibrator(

    cura_query_model,

    QUERY_ONLY_FEATURES,

    "IR_unsafe",

    "CURA Query-Only"
)


full_calibrator = fit_probability_calibrator(

    cura_full_model,

    FULL_FEATURES,

    "IR_unsafe",

    "CURA Full"
)


ann_calibrator = fit_probability_calibrator(

    ann_risk_model,

    FULL_FEATURES,

    "ANN_unsafe",

    "ANN-Risk"
)


# ============================================================
# 26. PREDICT CALIBRATED RISK FOR ALL QUERY-BUDGET PAIRS
# ============================================================

def calibrated_predictions(
    model,
    calibrator,
    feature_columns
):

    X_all = model_df[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )


    raw = model.predict_proba(
        X_all
    )[
        :,
        1
    ]


    return calibrator.transform(
        raw
    )


model_df[
    "CURA_query_risk"
] = calibrated_predictions(

    cura_query_model,

    query_calibrator,

    QUERY_ONLY_FEATURES
)


model_df[
    "CURA_full_risk"
] = calibrated_predictions(

    cura_full_model,

    full_calibrator,

    FULL_FEATURES
)


model_df[
    "ANN_risk"
] = calibrated_predictions(

    ann_risk_model,

    ann_calibrator,

    FULL_FEATURES
)


# ============================================================
# 27. WILSON UPPER CONFIDENCE BOUND
# ============================================================

def wilson_upper_bound(
    violations,
    n,
    z=1.96
):

    if n <= 0:

        return np.nan


    p = (

        violations

        /

        n
    )


    denominator = (

        1.0

        +

        (
            z * z
            /
            n
        )
    )


    centre = (

        p

        +

        (
            z * z
            /
            (
                2.0
                *
                n
            )
        )
    )


    margin = z * math.sqrt(

        (
            p
            *
            (
                1.0
                -
                p
            )
            /
            n
        )

        +

        (
            z * z
            /
            (
                4.0
                *
                n
                *
                n
            )
        )
    )


    return (

        centre
        +
        margin

    ) / denominator


# ============================================================
# 28. SELECT ONE ROW PER QUERY FROM RISK MODEL
# ============================================================

def choose_by_risk(
    dataframe,
    query_keys,
    risk_column,
    threshold
):

    subset = dataframe[

        dataframe[
            "query_key"
        ].isin(
            query_keys
        )

    ].copy()


    selected_rows = []


    for query_key, group in subset.groupby(
        "query_key"
    ):

        group = group.sort_values(
            "efSearch"
        )


        acceptable = group[

            group[
                risk_column
            ]

            <=

            threshold
        ]


        if len(
            acceptable
        ) > 0:

            selected = acceptable.iloc[
                0
            ]

        else:

            selected = group.iloc[
                -1
            ]


        selected_rows.append(
            selected
        )


    return pd.DataFrame(
        selected_rows
    ).reset_index(
        drop=True
    )


# ============================================================
# 29. SELECT CALIBRATED RISK THRESHOLD
#
# We choose the most efficient threshold whose empirical
# calibration-set Wilson upper bound satisfies alpha.
#
# If no threshold satisfies the conservative Wilson bound,
# we fall back to the safest available setting and report it.
# ============================================================

THRESHOLD_GRID = np.unique(

    np.concatenate(

        [

            np.linspace(
                0.001,
                0.05,
                50
            ),

            np.linspace(
                0.055,
                0.20,
                30
            ),

            np.linspace(
                0.225,
                0.50,
                12
            ),

            np.array(
                [
                    0.60,
                    0.75,
                    1.00
                ]
            )
        ]
    )
)


def tune_risk_threshold(
    risk_column,
    target_column,
    name
):

    candidates = []


    for threshold in THRESHOLD_GRID:

        selected = choose_by_risk(

            model_df,

            calibration_query_keys,

            risk_column,

            threshold
        )


        violations = int(

            selected[
                target_column
            ].sum()
        )


        n_queries = len(
            selected
        )


        violation_rate = (

            violations

            /

            n_queries
        )


        upper_bound = wilson_upper_bound(

            violations,

            n_queries
        )


        mean_ef = selected[
            "efSearch"
        ].mean()


        candidates.append({

            "threshold":
                float(
                    threshold
                ),

            "violations":
                violations,

            "queries":
                n_queries,

            "violation_rate":
                violation_rate,

            "wilson_upper":
                upper_bound,

            "mean_ef":
                mean_ef
        })


    candidates_df = pd.DataFrame(
        candidates
    )


    feasible = candidates_df[

        candidates_df[
            "wilson_upper"
        ]

        <=

        RISK_ALPHA
    ]


    if len(
        feasible
    ) > 0:

        chosen = (

            feasible

            .sort_values(

                [
                    "mean_ef",
                    "wilson_upper"
                ],

                ascending=[
                    True,
                    True
                ]
            )

            .iloc[
                0
            ]
        )


        mode = (
            "Wilson-SLA satisfied"
        )


    else:

        chosen = (

            candidates_df

            .sort_values(

                [
                    "wilson_upper",
                    "mean_ef"
                ],

                ascending=[
                    True,
                    True
                ]
            )

            .iloc[
                0
            ]
        )


        mode = (
            "No Wilson-feasible threshold; "
            "safest calibration setting used"
        )


    print(
        f"\n{name} threshold selection:"
    )


    print(
        "Threshold =",
        chosen[
            "threshold"
        ]
    )


    print(
        "Calibration violation rate =",
        chosen[
            "violation_rate"
        ]
    )


    print(
        "Wilson upper bound =",
        chosen[
            "wilson_upper"
        ]
    )


    print(
        "Mean calibration ef =",
        chosen[
            "mean_ef"
        ]
    )


    print(
        "Mode =",
        mode
    )


    return (
        float(
            chosen[
                "threshold"
            ]
        ),
        candidates_df,
        mode
    )


# ============================================================
# 30. TUNE CURA / ANN-RISK THRESHOLDS
# ============================================================

(
    cura_query_threshold,
    cura_query_threshold_df,
    cura_query_mode
) = tune_risk_threshold(

    "CURA_query_risk",

    "IR_unsafe",

    "CURA Query-Only"
)


(
    cura_full_threshold,
    cura_full_threshold_df,
    cura_full_mode
) = tune_risk_threshold(

    "CURA_full_risk",

    "IR_unsafe",

    "CURA Full"
)


(
    ann_risk_threshold,
    ann_threshold_df,
    ann_risk_mode
) = tune_risk_threshold(

    "ANN_risk",

    "ANN_unsafe",

    "ANN-Risk Controller"
)


cura_query_threshold_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "cura_query_threshold_sweep.csv"
    ),

    index=False
)


cura_full_threshold_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "cura_full_threshold_sweep.csv"
    ),

    index=False
)


ann_threshold_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "ann_risk_threshold_sweep.csv"
    ),

    index=False
)


# ============================================================
# 31. FIXED IR-SAFE BASELINE
#
# Select the smallest fixed efSearch satisfying the same
# calibration-set downstream IR SLA.
# ============================================================

fixed_rows = []


for ef in EF_VALUES:

    subset = model_df[

        (
            model_df[
                "query_key"
            ].isin(
                calibration_query_keys
            )
        )

        &

        (
            model_df[
                "efSearch"
            ]
            ==
            ef
        )

    ]


    violations = int(

        subset[
            "IR_unsafe"
        ].sum()
    )


    n_queries = len(
        subset
    )


    violation_rate = (

        violations

        /

        n_queries
    )


    upper = wilson_upper_bound(

        violations,

        n_queries
    )


    fixed_rows.append({

        "efSearch":
            ef,

        "violations":
            violations,

        "queries":
            n_queries,

        "violation_rate":
            violation_rate,

        "wilson_upper":
            upper
    })


fixed_calibration_df = pd.DataFrame(
    fixed_rows
)


fixed_feasible = fixed_calibration_df[

    fixed_calibration_df[
        "wilson_upper"
    ]

    <=

    RISK_ALPHA
]


if len(
    fixed_feasible
) > 0:

    FIXED_EF = int(

        fixed_feasible

        .sort_values(
            "efSearch"
        )

        .iloc[
            0
        ][
            "efSearch"
        ]
    )


    FIXED_MODE = (
        "Wilson-SLA satisfied"
    )


else:

    safest = (

        fixed_calibration_df

        .sort_values(

            [
                "wilson_upper",
                "efSearch"
            ]
        )

        .iloc[
            0
        ]
    )


    FIXED_EF = int(
        safest[
            "efSearch"
        ]
    )


    FIXED_MODE = (
        "No Wilson-feasible fixed budget; "
        "safest available ef used"
    )


print(
    "\nFixed-IR baseline:"
)

print(
    "Fixed efSearch =",
    FIXED_EF
)

print(
    "Mode =",
    FIXED_MODE
)


fixed_calibration_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "fixed_budget_calibration.csv"
    ),

    index=False
)


# ============================================================
# 32. ORACLE SELECTORS
# ============================================================

def choose_oracle(
    dataframe,
    query_keys,
    safe_column
):

    subset = dataframe[

        dataframe[
            "query_key"
        ].isin(
            query_keys
        )

    ].copy()


    selected_rows = []


    for query_key, group in subset.groupby(
        "query_key"
    ):

        group = group.sort_values(
            "efSearch"
        )


        safe = group[

            group[
                safe_column
            ]

            ==
            0
        ]


        if len(
            safe
        ) > 0:

            chosen = safe.iloc[
                0
            ]

        else:

            chosen = group.iloc[
                -1
            ]


        selected_rows.append(
            chosen
        )


    return pd.DataFrame(
        selected_rows
    ).reset_index(
        drop=True
    )


# ============================================================
# 33. FIXED SELECTOR
# ============================================================

def choose_fixed(
    dataframe,
    query_keys,
    ef
):

    return dataframe[

        (
            dataframe[
                "query_key"
            ].isin(
                query_keys
            )
        )

        &

        (
            dataframe[
                "efSearch"
            ]
            ==
            ef
        )

    ].copy().reset_index(
        drop=True
    )


# ============================================================
# 34. BOOTSTRAP CI
# ============================================================

def bootstrap_mean_ci(
    values,
    n_bootstrap,
    seed
):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(
        values
    ) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )


    rng = np.random.default_rng(
        seed
    )


    n = len(
        values
    )


    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=np.float64
    )


    for i in range(
        n_bootstrap
    ):

        indices = rng.integers(

            0,

            n,

            size=n
        )


        bootstrap_means[
            i
        ] = np.mean(

            values[
                indices
            ]
        )


    return (

        float(
            np.mean(
                values
            )
        ),

        float(
            np.quantile(
                bootstrap_means,
                0.025
            )
        ),

        float(
            np.quantile(
                bootstrap_means,
                0.975
            )
        )
    )


# ============================================================
# 35. PAIRED RANDOMIZATION TEST
# ============================================================

def paired_randomization_pvalue(
    differences,
    n_permutations,
    seed
):

    differences = np.asarray(
        differences,
        dtype=np.float64
    )


    differences = differences[
        np.isfinite(
            differences
        )
    ]


    if len(
        differences
    ) == 0:

        return np.nan


    observed = abs(
        np.mean(
            differences
        )
    )


    if observed == 0:

        return 1.0


    rng = np.random.default_rng(
        seed
    )


    extreme = 0


    for _ in range(
        n_permutations
    ):

        signs = rng.choice(

            [
                -1.0,
                1.0
            ],

            size=len(
                differences
            )
        )


        statistic = abs(

            np.mean(
                differences
                *
                signs
            )
        )


        if statistic >= observed:

            extreme += 1


    return (

        extreme
        +
        1

    ) / (

        n_permutations
        +
        1
    )


# ============================================================
# 36. EFFECTIVE COST
#
# Query-only CURA:
#   no HNSW probe required
#
# CURA Full / ANN-Risk:
#
#   selected ef=10:
#       only the probe search is needed
#
#   selected ef>10:
#       cost = ef10 probe + selected search
#
# This explicitly accounts for probe overhead.
# ============================================================

def add_effective_cost(
    selected_df,
    uses_probe
):

    result = selected_df.copy()


    if not uses_probe:

        result[
            "effective_distance_computations"
        ] = result[
            "distance_computations"
        ]


        result[
            "effective_latency_ms"
        ] = result[
            "mean_latency_ms"
        ]


        return result


    selected_work = result[
        "distance_computations"
    ].astype(
        float
    )


    selected_latency = result[
        "mean_latency_ms"
    ].astype(
        float
    )


    probe_work = result[
        "probe_ndis"
    ].astype(
        float
    )


    probe_latency = result[
        "probe_latency_ms"
    ].astype(
        float
    )


    is_probe_budget = (

        result[
            "efSearch"
        ]

        ==

        PROBE_EF
    )


    result[
        "effective_distance_computations"
    ] = np.where(

        is_probe_budget,

        probe_work,

        probe_work
        +
        selected_work
    )


    result[
        "effective_latency_ms"
    ] = np.where(

        is_probe_budget,

        probe_latency,

        probe_latency
        +
        selected_latency
    )


    return result


# ============================================================
# 37. EVALUATE ONE METHOD
# ============================================================

def evaluate_selected(
    selected_df,
    method_name,
    split_name,
    uses_probe
):

    selected_df = add_effective_cost(

        selected_df,

        uses_probe
    )


    n_queries = len(
        selected_df
    )


    ir_violations = int(

        selected_df[
            "IR_unsafe"
        ].sum()
    )


    ann_violations = int(

        selected_df[
            "ANN_unsafe"
        ].sum()
    )


    ir_violation_rate = (

        ir_violations

        /

        n_queries
    )


    ann_violation_rate = (

        ann_violations

        /

        n_queries
    )


    (
        mean_work,
        work_ci_low,
        work_ci_high
    ) = bootstrap_mean_ci(

        selected_df[
            "effective_distance_computations"
        ].values,

        args.bootstrap,

        SEED + 100
    )


    (
        mean_ef,
        ef_ci_low,
        ef_ci_high
    ) = bootstrap_mean_ci(

        selected_df[
            "efSearch"
        ].values,

        args.bootstrap,

        SEED + 200
    )


    (
        mean_ndcg_loss,
        ndcg_loss_ci_low,
        ndcg_loss_ci_high
    ) = bootstrap_mean_ci(

        selected_df[
            "nDCG_loss"
        ].values,

        args.bootstrap,

        SEED + 300
    )


    return {

        "split":
            split_name,

        "method":
            method_name,

        "queries":
            n_queries,

        "mean_ef":
            mean_ef,

        "mean_ef_CI_low":
            ef_ci_low,

        "mean_ef_CI_high":
            ef_ci_high,

        "median_ef":
            selected_df[
                "efSearch"
            ].median(),

        "IR_violation_rate":
            ir_violation_rate,

        "IR_violation_Wilson_upper":
            wilson_upper_bound(
                ir_violations,
                n_queries
            ),

        "ANN_violation_rate":
            ann_violation_rate,

        "mean_ANN_Recall@10":
            selected_df[
                "ANN_Recall@10"
            ].mean(),

        "mean_nDCG@10":
            selected_df[
                "nDCG@10"
            ].mean(),

        "mean_nDCG_loss":
            mean_ndcg_loss,

        "nDCG_loss_CI_low":
            ndcg_loss_ci_low,

        "nDCG_loss_CI_high":
            ndcg_loss_ci_high,

        "mean_search_work":
            mean_work,

        "search_work_CI_low":
            work_ci_low,

        "search_work_CI_high":
            work_ci_high,

        "mean_effective_latency_ms":
            selected_df[
                "effective_latency_ms"
            ].mean(),

        "p95_effective_latency_ms":
            selected_df[
                "effective_latency_ms"
            ].quantile(
                0.95
            )
    }


# ============================================================
# 38. RUN ALL METHODS ON ONE SPLIT
# ============================================================

def evaluate_split(
    query_keys,
    split_name
):

    selected_outputs = {}


    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    fixed_selected = choose_fixed(

        model_df,

        query_keys,

        FIXED_EF
    )


    selected_outputs[
        "Fixed-IR"
    ] = add_effective_cost(

        fixed_selected,

        uses_probe=False
    )


    # --------------------------------------------------------
    # ANN-RISK
    # --------------------------------------------------------

    ann_selected = choose_by_risk(

        model_df,

        query_keys,

        "ANN_risk",

        ann_risk_threshold
    )


    selected_outputs[
        "ANN-Risk"
    ] = add_effective_cost(

        ann_selected,

        uses_probe=True
    )


    # --------------------------------------------------------
    # CURA QUERY-ONLY
    # --------------------------------------------------------

    cura_query_selected = choose_by_risk(

        model_df,

        query_keys,

        "CURA_query_risk",

        cura_query_threshold
    )


    selected_outputs[
        "CURA-Query"
    ] = add_effective_cost(

        cura_query_selected,

        uses_probe=False
    )


    # --------------------------------------------------------
    # CURA FULL
    # --------------------------------------------------------

    cura_full_selected = choose_by_risk(

        model_df,

        query_keys,

        "CURA_full_risk",

        cura_full_threshold
    )


    selected_outputs[
        "CURA-Full"
    ] = add_effective_cost(

        cura_full_selected,

        uses_probe=True
    )


    # --------------------------------------------------------
    # IR ORACLE
    # --------------------------------------------------------

    ir_oracle = choose_oracle(

        model_df,

        query_keys,

        "IR_unsafe"
    )


    selected_outputs[
        "IR-Oracle"
    ] = add_effective_cost(

        ir_oracle,

        uses_probe=False
    )


    # --------------------------------------------------------
    # ANN ORACLE
    # --------------------------------------------------------

    ann_oracle = choose_oracle(

        model_df,

        query_keys,

        "ANN_unsafe"
    )


    selected_outputs[
        "ANN-Oracle"
    ] = add_effective_cost(

        ann_oracle,

        uses_probe=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summaries = []


    for method_name, selected_df in selected_outputs.items():

        uses_probe = method_name in {
            "ANN-Risk",
            "CURA-Full"
        }


        summaries.append(

            evaluate_selected(

                selected_df,

                method_name,

                split_name,

                uses_probe=False
                # Cost already added above.
            )
        )


    return (
        pd.DataFrame(
            summaries
        ),
        selected_outputs
    )


# ============================================================
# 39. HELD-OUT MS MARCO TEST
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "HELD-OUT MS MARCO TEST"
)

print(
    "=" * 100
)


(
    test_summary_df,
    test_selected
) = evaluate_split(

    test_query_keys,

    "MSMARCO_TEST"
)


# ============================================================
# 40. OOD TREC-DL TEST
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "OOD TREC-DL-2019 TEST"
)

print(
    "=" * 100
)


(
    ood_summary_df,
    ood_selected
) = evaluate_split(

    ood_query_keys,

    "TREC_DL_2019_OOD"
)


# ============================================================
# 41. COMBINE RESULTS
# ============================================================

summary_df = pd.concat(

    [
        test_summary_df,
        ood_summary_df
    ],

    ignore_index=True
)


# ============================================================
# 42. RELATIVE SAVINGS AGAINST FIXED
# ============================================================

for split_name in summary_df[
    "split"
].unique():

    split_mask = (

        summary_df[
            "split"
        ]

        ==

        split_name
    )


    split_data = summary_df[
        split_mask
    ]


    fixed_row = split_data[

        split_data[
            "method"
        ]
        ==
        "Fixed-IR"
    ]


    if len(
        fixed_row
    ) != 1:

        continue


    fixed_work = float(

        fixed_row.iloc[
            0
        ][
            "mean_search_work"
        ]
    )


    fixed_latency = float(

        fixed_row.iloc[
            0
        ][
            "mean_effective_latency_ms"
        ]
    )


    summary_df.loc[

        split_mask,

        "work_saving_vs_fixed_%"

    ] = (

        100.0

        *

        (
            fixed_work

            -

            summary_df.loc[
                split_mask,
                "mean_search_work"
            ]
        )

        /

        fixed_work
    )


    summary_df.loc[

        split_mask,

        "latency_saving_vs_fixed_%"

    ] = (

        100.0

        *

        (
            fixed_latency

            -

            summary_df.loc[
                split_mask,
                "mean_effective_latency_ms"
            ]
        )

        /

        fixed_latency
    )


# ============================================================
# 43. SAVE SELECTED QUERY RESULTS
# ============================================================

for method_name, selected_df in test_selected.items():

    selected_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            (
                "selected_"
                +
                method_name
                .lower()
                .replace(
                    "-",
                    "_"
                )
                +
                "_msmarco_test.csv"
            )
        ),

        index=False
    )


for method_name, selected_df in ood_selected.items():

    selected_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            (
                "selected_"
                +
                method_name
                .lower()
                .replace(
                    "-",
                    "_"
                )
                +
                "_trec_ood.csv"
            )
        ),

        index=False
    )


summary_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "final_method_comparison.csv"
    ),

    index=False
)


# ============================================================
# 44. PROBABILITY CALIBRATION METRICS
# ============================================================

def expected_calibration_error(
    probabilities,
    labels,
    bins=10
):

    probabilities = np.asarray(
        probabilities
    )


    labels = np.asarray(
        labels
    )


    edges = np.linspace(
        0.0,
        1.0,
        bins + 1
    )


    ece = 0.0


    for i in range(
        bins
    ):

        if i == bins - 1:

            mask = (

                (
                    probabilities
                    >=
                    edges[
                        i
                    ]
                )

                &

                (
                    probabilities
                    <=
                    edges[
                        i + 1
                    ]
                )
            )

        else:

            mask = (

                (
                    probabilities
                    >=
                    edges[
                        i
                    ]
                )

                &

                (
                    probabilities
                    <
                    edges[
                        i + 1
                    ]
                )
            )


        if not np.any(
            mask
        ):

            continue


        confidence = np.mean(
            probabilities[
                mask
            ]
        )


        accuracy = np.mean(
            labels[
                mask
            ]
        )


        ece += (

            np.mean(
                mask
            )

            *

            abs(
                confidence
                -
                accuracy
            )
        )


    return float(
        ece
    )


calibration_rows = []


for split_name, split_mask in [

    (
        "MSMARCO_TEST",
        test_mask
    ),

    (
        "TREC_DL_2019_OOD",
        ood_mask
    )

]:

    y = model_df.loc[
        split_mask,
        "IR_unsafe"
    ].to_numpy(
        dtype=int
    )


    for model_name, probability_column in [

        (
            "CURA-Query",
            "CURA_query_risk"
        ),

        (
            "CURA-Full",
            "CURA_full_risk"
        )

    ]:

        probability = model_df.loc[

            split_mask,

            probability_column

        ].to_numpy(
            dtype=np.float64
        )


        brier = brier_score_loss(

            y,

            probability
        )


        ece = expected_calibration_error(

            probability,

            y,

            bins=10
        )


        try:

            auc = roc_auc_score(

                y,

                probability
            )

        except Exception:

            auc = np.nan


        calibration_rows.append({

            "split":
                split_name,

            "model":
                model_name,

            "Brier":
                brier,

            "ECE":
                ece,

            "ROC_AUC":
                auc
        })


calibration_metrics_df = pd.DataFrame(

    calibration_rows
)


calibration_metrics_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "calibration_metrics.csv"
    ),

    index=False
)


# ============================================================
# 45. PAIRED SIGNIFICANCE:
# CURA FULL vs FIXED
#
# Search work difference on held-out test.
# ============================================================

fixed_test = (

    test_selected[
        "Fixed-IR"
    ]

    .set_index(
        "query_key"
    )
)


cura_test = (

    test_selected[
        "CURA-Full"
    ]

    .set_index(
        "query_key"
    )
)


common_test_queries = (

    fixed_test.index

    .intersection(
        cura_test.index
    )
)


paired_work_difference = (

    fixed_test.loc[

        common_test_queries,

        "effective_distance_computations"

    ].to_numpy()

    -

    cura_test.loc[

        common_test_queries,

        "effective_distance_computations"

    ].to_numpy()
)


work_pvalue = paired_randomization_pvalue(

    paired_work_difference,

    args.permutations,

    SEED + 999
)


(
    work_difference_mean,
    work_difference_ci_low,
    work_difference_ci_high
) = bootstrap_mean_ci(

    paired_work_difference,

    args.bootstrap,

    SEED + 998
)


# ============================================================
# 46. MANUAL REVIEW SET — 50 QUERIES
#
# We create a review queue.
#
# Actual CURA violations are prioritized.
# Remaining rows are difficult / near-miss cases.
#
# This is NOT pretending all 50 are failures.
# ============================================================

def build_manual_review_rows(
    selected_df,
    split_name
):

    working = selected_df.copy()


    query_text_map = (

        query_df

        .set_index(
            "query_key"
        )[
            "query_text"
        ]

        .to_dict()
    )


    working[
        "query_text"
    ] = working[
        "query_key"
    ].map(
        query_text_map
    )


    working[
        "review_priority"
    ] = (

        working[
            "IR_unsafe"
        ]
        *
        1000.0

        +

        working[
            "nDCG_loss"
        ]
        *
        100.0

        +

        working[
            "CURA_full_risk"
        ]
        *
        10.0
    )


    working[
        "evaluation_split"
    ] = split_name


    working[
        "review_reason"
    ] = np.where(

        working[
            "IR_unsafe"
        ]
        ==
        1,

        "ACTUAL_IR_VIOLATION",

        "HIGH_RISK_OR_NEAR_MISS"
    )


    return working


review_test = build_manual_review_rows(

    test_selected[
        "CURA-Full"
    ],

    "MSMARCO_TEST"
)


review_ood = build_manual_review_rows(

    ood_selected[
        "CURA-Full"
    ],

    "TREC_DL_2019_OOD"
)


manual_review_df = (

    pd.concat(

        [
            review_test,
            review_ood
        ],

        ignore_index=True
    )

    .sort_values(

        "review_priority",

        ascending=False
    )

    .head(
        50
    )

    .copy()
)


manual_review_df[
    "manual_error_category"
] = ""


manual_review_df[
    "manual_notes"
] = ""


manual_columns = [

    "evaluation_split",

    "query_key",

    "query_text",

    "efSearch",

    "CURA_full_risk",

    "IR_unsafe",

    "nDCG@10",

    "nDCG_loss",

    "ANN_Recall@10",

    "probe_gap_1_10",

    "probe_gap_9_10",

    "probe_std",

    "probe_ndis",

    "review_reason",

    "manual_error_category",

    "manual_notes"
]


manual_review_df[
    manual_columns
].to_csv(

    os.path.join(
        RESULTS_DIR,
        "manual_review_50.csv"
    ),

    index=False
)


# ============================================================
# 47. PAPER-READY TABLE
# ============================================================

paper_table = summary_df[

    [
        "split",
        "method",
        "queries",
        "mean_ef",
        "IR_violation_rate",
        "ANN_violation_rate",
        "mean_nDCG@10",
        "mean_nDCG_loss",
        "mean_search_work",
        "work_saving_vs_fixed_%",
        "mean_effective_latency_ms",
        "latency_saving_vs_fixed_%"
    ]

].copy()


paper_table.to_csv(

    os.path.join(
        RESULTS_DIR,
        "paper_ready_method_table.csv"
    ),

    index=False
)


# ============================================================
# 48. PLOT:
# SEARCH WORK VS IR VIOLATION
# ============================================================

for split_name in [

    "MSMARCO_TEST",
    "TREC_DL_2019_OOD"
]:

    plot_df = summary_df[

        summary_df[
            "split"
        ]
        ==
        split_name
    ].copy()


    plt.figure(
        figsize=(8, 5)
    )


    for _, row in plot_df.iterrows():

        plt.scatter(

            row[
                "mean_search_work"
            ],

            row[
                "IR_violation_rate"
            ],

            s=70
        )


        plt.annotate(

            row[
                "method"
            ],

            (
                row[
                    "mean_search_work"
                ],

                row[
                    "IR_violation_rate"
                ]
            ),

            xytext=(
                5,
                5
            ),

            textcoords="offset points"
        )


    plt.axhline(

        RISK_ALPHA,

        linestyle="--",

        label="Target IR violation rate"
    )


    plt.xlabel(
        "Mean HNSW distance computations"
    )


    plt.ylabel(
        "Downstream IR violation rate"
    )


    plt.title(
        (
            "Efficiency vs Downstream Risk — "
            +
            split_name
        )
    )


    plt.grid(
        True
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULTS_DIR,

            (
                "efficiency_vs_risk_"
                +
                split_name.lower()
                +
                ".png"
            )
        ),

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 49. PLOT:
# MEAN efSearch
# ============================================================

test_plot_df = summary_df[

    summary_df[
        "split"
    ]
    ==
    "MSMARCO_TEST"

].copy()


plt.figure(
    figsize=(9, 5)
)


plt.bar(

    test_plot_df[
        "method"
    ],

    test_plot_df[
        "mean_ef"
    ]
)


plt.ylabel(
    "Mean selected efSearch"
)


plt.title(
    "Held-Out MS MARCO: Selected Search Budget"
)


plt.xticks(
    rotation=25
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "heldout_mean_ef_comparison.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 50. PLOT:
# WORK SAVINGS
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.bar(

    test_plot_df[
        "method"
    ],

    test_plot_df[
        "work_saving_vs_fixed_%"
    ]
)


plt.ylabel(
    "Search-work saving vs fixed (%)"
)


plt.title(
    "Held-Out MS MARCO: Search-Work Savings"
)


plt.xticks(
    rotation=25
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "heldout_work_savings.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 51. PLOT:
# SIMPLE CALIBRATION CURVE FOR CURA FULL
# ============================================================

def calibration_curve_points(
    probabilities,
    labels,
    bins=10
):

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1
    )


    predicted = []

    observed = []


    for i in range(
        bins
    ):

        if i == bins - 1:

            mask = (

                (
                    probabilities
                    >=
                    edges[
                        i
                    ]
                )

                &

                (
                    probabilities
                    <=
                    edges[
                        i + 1
                    ]
                )
            )

        else:

            mask = (

                (
                    probabilities
                    >=
                    edges[
                        i
                    ]
                )

                &

                (
                    probabilities
                    <
                    edges[
                        i + 1
                    ]
                )
            )


        if not np.any(
            mask
        ):

            continue


        predicted.append(

            np.mean(
                probabilities[
                    mask
                ]
            )
        )


        observed.append(

            np.mean(
                labels[
                    mask
                ]
            )
        )


    return (

        np.asarray(
            predicted
        ),

        np.asarray(
            observed
        )
    )


for split_name, split_mask in [

    (
        "MSMARCO_TEST",
        test_mask
    ),

    (
        "TREC_DL_2019_OOD",
        ood_mask
    )

]:

    probabilities = model_df.loc[

        split_mask,

        "CURA_full_risk"

    ].to_numpy(
        dtype=np.float64
    )


    labels = model_df.loc[

        split_mask,

        "IR_unsafe"

    ].to_numpy(
        dtype=int
    )


    predicted, observed = calibration_curve_points(

        probabilities,

        labels,

        bins=10
    )


    plt.figure(
        figsize=(6, 6)
    )


    plt.plot(

        [
            0,
            1
        ],

        [
            0,
            1
        ],

        linestyle="--",

        label="Perfect calibration"
    )


    plt.plot(

        predicted,

        observed,

        marker="o",

        label="CURA-Full"
    )


    plt.xlabel(
        "Predicted downstream failure probability"
    )


    plt.ylabel(
        "Observed failure frequency"
    )


    plt.title(
        (
            "CURA Calibration — "
            +
            split_name
        )
    )


    plt.grid(
        True
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULTS_DIR,

            (
                "calibration_"
                +
                split_name.lower()
                +
                ".png"
            )
        ),

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 52. SAVE EXPERIMENT PROTOCOL
# ============================================================

protocol = {

    "method":
        "CURA",

    "full_name":
        "Calibrated Utility-Risk Allocation",

    "seed":
        SEED,

    "corpus_run":
        RUN_NAME,

    "nDCG_loss_tolerance":
        NDCG_TOLERANCE,

    "ANN_recall_target":
        ANN_TARGET,

    "risk_alpha":
        RISK_ALPHA,

    "probe_ef":
        PROBE_EF,

    "candidate_ef_values":
        EF_VALUES,

    "split":
        {
            "MSMARCO_train":
                len(
                    train_query_keys
                ),

            "MSMARCO_calibration":
                len(
                    calibration_query_keys
                ),

            "MSMARCO_test":
                len(
                    test_query_keys
                ),

            "TREC_OOD":
                len(
                    ood_query_keys
                )
        },

    "CURA_query_threshold":
        cura_query_threshold,

    "CURA_full_threshold":
        cura_full_threshold,

    "ANN_risk_threshold":
        ann_risk_threshold,

    "fixed_ef":
        FIXED_EF,

    "CURA_query_threshold_mode":
        cura_query_mode,

    "CURA_full_threshold_mode":
        cura_full_mode,

    "ANN_threshold_mode":
        ann_risk_mode,

    "fixed_mode":
        FIXED_MODE,

    "bootstrap":
        args.bootstrap,

    "permutations":
        args.permutations,

    "python":
        sys.version,

    "platform":
        platform.platform(),

    "faiss_version":
        getattr(
            faiss,
            "__version__",
            "unknown"
        )
}


with open(

    os.path.join(
        RESULTS_DIR,
        "cura_protocol.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        protocol,
        f,
        indent=2
    )


# ============================================================
# 53. SAVE SIGNIFICANCE RESULT
# ============================================================

significance_df = pd.DataFrame([

    {
        "comparison":
            "Fixed-IR minus CURA-Full search work",

        "mean_paired_work_reduction":
            work_difference_mean,

        "95CI_low":
            work_difference_ci_low,

        "95CI_high":
            work_difference_ci_high,

        "paired_randomization_p":
            work_pvalue
    }

])


significance_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "cura_vs_fixed_significance.csv"
    ),

    index=False
)


# ============================================================
# 54. OBJECTIVE INTERPRETATION
# ============================================================

test_cura = test_summary_df[

    test_summary_df[
        "method"
    ]
    ==
    "CURA-Full"

].iloc[
    0
]


test_fixed = test_summary_df[

    test_summary_df[
        "method"
    ]
    ==
    "Fixed-IR"

].iloc[
    0
]


ood_cura = ood_summary_df[

    ood_summary_df[
        "method"
    ]
    ==
    "CURA-Full"

].iloc[
    0
]


heldout_safe = (

    test_cura[
        "IR_violation_rate"
    ]

    <=

    RISK_ALPHA
)


heldout_work_better = (

    test_cura[
        "mean_search_work"
    ]

    <

    test_fixed[
        "mean_search_work"
    ]
)


work_statistically_supported = (

    work_difference_ci_low

    >

    0
)


ood_safe = (

    ood_cura[
        "IR_violation_rate"
    ]

    <=

    RISK_ALPHA
)


if (

    heldout_safe

    and

    heldout_work_better

    and

    work_statistically_supported

    and

    ood_safe

):

    RESEARCH_DECISION = (

        "STRONG GO: CURA reduces search work on held-out "
        "MS MARCO while meeting the downstream-risk target, "
        "the paired search-work reduction is statistically "
        "supported, and the risk target transfers to TREC-DL OOD."
    )


elif (

    heldout_safe

    and

    heldout_work_better

    and

    work_statistically_supported

):

    RESEARCH_DECISION = (

        "IN-DOMAIN GO / OOD LIMITATION: CURA provides a "
        "statistically supported efficiency gain on held-out "
        "MS MARCO at the target downstream-risk level, but "
        "the TREC-DL OOD safety target is not maintained. "
        "Distribution-shift robustness becomes a central finding."
    )


elif (

    heldout_safe

    and

    heldout_work_better

):

    RESEARCH_DECISION = (

        "PROMISING BUT NOT YET STATISTICALLY SECURE: "
        "CURA meets the held-out downstream-risk target and "
        "uses less search work than fixed search, but the "
        "paired efficiency difference is not yet statistically "
        "supported."
    )


else:

    RESEARCH_DECISION = (

        "PIVOT / NEGATIVE RESULT: under the current held-out "
        "evaluation, CURA does not simultaneously improve "
        "efficiency and satisfy the downstream-risk target. "
        "Do not claim an adaptive efficiency improvement."
    )


with open(

    os.path.join(
        RESULTS_DIR,
        "research_decision.txt"
    ),

    "w",

    encoding="utf-8"

) as f:

    f.write(
        RESEARCH_DECISION
        +
        "\n"
    )


# ============================================================
# 55. FINAL TERMINAL REPORT
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "CURA EXPERIMENT COMPLETED"
)

print(
    "=" * 100
)


print(
    "\nCALIBRATED OPERATING POINTS"
)

print(
    "-" * 100
)

print(
    "Fixed-IR efSearch =",
    FIXED_EF
)

print(
    "CURA Query threshold =",
    cura_query_threshold
)

print(
    "CURA Full threshold =",
    cura_full_threshold
)

print(
    "ANN-Risk threshold =",
    ann_risk_threshold
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "HELD-OUT MS MARCO RESULTS"
)

print(
    "=" * 100
)


print(

    test_summary_df[

        [
            "method",
            "queries",
            "mean_ef",
            "IR_violation_rate",
            "ANN_violation_rate",
            "mean_nDCG@10",
            "mean_search_work",
            "mean_effective_latency_ms"
        ]

    ].to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "OOD TREC-DL-2019 RESULTS"
)

print(
    "=" * 100
)


print(

    ood_summary_df[

        [
            "method",
            "queries",
            "mean_ef",
            "IR_violation_rate",
            "ANN_violation_rate",
            "mean_nDCG@10",
            "mean_search_work",
            "mean_effective_latency_ms"
        ]

    ].to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "CALIBRATION METRICS"
)

print(
    "=" * 100
)


print(
    calibration_metrics_df.to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "CURA-FULL VS FIXED — HELD-OUT SEARCH WORK"
)

print(
    "=" * 100
)


print(
    "Mean paired work reduction =",
    work_difference_mean
)


print(
    "95% CI = [",
    work_difference_ci_low,
    ",",
    work_difference_ci_high,
    "]"
)


print(
    "Paired randomization p-value =",
    work_pvalue
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "OBJECTIVE RESEARCH DECISION"
)

print(
    "=" * 100
)


print(
    RESEARCH_DECISION
)


print(
    "\nManual 50-query review sheet:"
)

print(
    os.path.join(
        RESULTS_DIR,
        "manual_review_50.csv"
    )
)


print(
    "\nFinal paper-ready table:"
)

print(
    os.path.join(
        RESULTS_DIR,
        "paper_ready_method_table.csv"
    )
)


print(
    "\nAll results:"
)

print(
    RESULTS_DIR
)


print(
    "\nDONE."
)