# ============================================================
# 08_patience_style_baseline.py
#
# COL764 - QUERY-SPECIFIC ANN SEARCH BUDGETS
#
# PATIENCE-INSPIRED RESULT-STABILITY BASELINE
#
# IMPORTANT:
# This is NOT claimed as the official ECIR 2025 Patience
# implementation.
#
# Idea:
# Search at increasing efSearch budgets.
# If the top-10 result set stops changing substantially for
# several consecutive budget increases, stop.
#
# Hyperparameters are selected ONLY on the calibration split.
# Final MS MARCO test and TREC-DL OOD remain untouched.
#
# Costs include every probe/search actually performed.
# ============================================================

import os
import sys
import json
import math
import argparse

import numpy as np
import pandas as pd
import faiss


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

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
    "--risk_alpha",
    type=float,
    default=0.05
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
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

SEED = args.seed


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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

PHASE04_DIR = os.path.join(
    BASE_DIR,
    "results"
)

PHASE05_DIR = os.path.join(
    BASE_DIR,
    "research_grade_validation"
)

CURA_DIR = os.path.join(
    BASE_DIR,
    "cura_controller"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "patience_style_baseline"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


QUERY_FILE = os.path.join(
    PHASE04_DIR,
    "queries.csv"
)

QUERY_EMBED_FILE = os.path.join(
    CACHE_DIR,
    "query_embeddings.npy"
)

HNSW_FILE = os.path.join(
    CACHE_DIR,
    "hnsw_M32_efC200.faiss"
)

TOP10_RESULTS_FILE = os.path.join(
    PHASE05_DIR,
    "top10_query_budget_results.csv"
)

SPLIT_FILE = os.path.join(
    CURA_DIR,
    "query_split.json"
)

CURA_TABLE_FILE = os.path.join(
    CURA_DIR,
    "paper_ready_method_table.csv"
)

CURA_TEST_FILE = os.path.join(
    CURA_DIR,
    "selected_cura_full_msmarco_test.csv"
)

CURA_OOD_FILE = os.path.join(
    CURA_DIR,
    "selected_cura_full_trec_ood.csv"
)

SEARCH_CACHE = os.path.join(
    RESULTS_DIR,
    "stability_top10_searches.npz"
)


required = [
    QUERY_FILE,
    QUERY_EMBED_FILE,
    HNSW_FILE,
    TOP10_RESULTS_FILE,
    SPLIT_FILE,
    CURA_TABLE_FILE,
    CURA_TEST_FILE,
    CURA_OOD_FILE
]

missing = [
    p
    for p in required
    if not os.path.exists(p)
]

if missing:

    print("\nERROR - missing required files:")

    for p in missing:
        print(p)

    sys.exit(1)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("PATIENCE-INSPIRED RESULT-STABILITY BASELINE")
print("=" * 100)

query_df = pd.read_csv(
    QUERY_FILE,
    dtype={
        "query_key": str,
        "source": str
    }
)

query_embeddings = np.load(
    QUERY_EMBED_FILE
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

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:

    split = json.load(f)


train_keys = set(
    split["train_queries"]
)

calibration_keys = set(
    split["calibration_queries"]
)

test_keys = set(
    split["test_queries"]
)

ood_keys = set(
    split["ood_queries"]
)


EF_VALUES = sorted(
    int(x)
    for x in budget_df[
        "efSearch"
    ].unique()
)


print("\nCandidate efSearch values:")
print(EF_VALUES)

print("\nCalibration queries:", len(calibration_keys))
print("Held-out MS MARCO queries:", len(test_keys))
print("TREC-DL OOD queries:", len(ood_keys))


# ============================================================
# SEARCH TOP-10 RESULTS AT EVERY BUDGET
# ============================================================

if os.path.exists(
    SEARCH_CACHE
):

    print(
        "\nLoading cached stability searches..."
    )

    cache = np.load(
        SEARCH_CACHE
    )

    all_indices = cache[
        "indices"
    ]

else:

    print(
        "\nRunning top-10 searches for stability baseline..."
    )

    index = faiss.read_index(
        HNSW_FILE
    )

    faiss.omp_set_num_threads(
        1
    )

    result_list = []

    for ef in EF_VALUES:

        print(
            "Searching efSearch =",
            ef
        )

        index.hnsw.efSearch = int(
            ef
        )

        _, ids = index.search(

            np.ascontiguousarray(
                query_embeddings,
                dtype=np.float32
            ),

            10
        )

        result_list.append(
            ids.astype(
                np.int64
            )
        )

    all_indices = np.stack(
        result_list,
        axis=0
    )

    np.savez_compressed(

        SEARCH_CACHE,

        indices=all_indices
    )


# Shape:
# [num_ef, num_queries, 10]

print(
    "\nSearch cache shape:",
    all_indices.shape
)


# ============================================================
# QUERY INDEX MAP
# ============================================================

query_to_row = {

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
# SUCCESSIVE RESULT STABILITY
#
# stability =
# |top10(previous) intersection top10(current)| / 10
# ============================================================

num_efs = len(
    EF_VALUES
)

num_queries = len(
    query_df
)

stability = np.full(

    (
        num_queries,
        num_efs
    ),

    np.nan,

    dtype=np.float64
)


for q in range(
    num_queries
):

    for j in range(
        1,
        num_efs
    ):

        previous = set(
            int(x)
            for x in all_indices[
                j - 1,
                q
            ]
            if x >= 0
        )

        current = set(
            int(x)
            for x in all_indices[
                j,
                q
            ]
            if x >= 0
        )

        stability[
            q,
            j
        ] = (

            len(
                previous.intersection(
                    current
                )
            )

            /

            10.0
        )


# ============================================================
# WILSON UPPER BOUND
# ============================================================

def wilson_upper(
    violations,
    n,
    z=1.96
):

    if n <= 0:
        return np.nan

    p = violations / n

    denominator = (
        1.0
        +
        z * z / n
    )

    centre = (
        p
        +
        z * z / (2.0 * n)
    )

    margin = z * math.sqrt(

        p * (1.0 - p) / n

        +

        z * z / (
            4.0 * n * n
        )
    )

    return (
        centre + margin
    ) / denominator


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_ci(
    values,
    n_boot,
    seed
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    rng = np.random.default_rng(
        seed
    )

    estimates = []

    n = len(values)

    for _ in range(
        n_boot
    ):

        sample = values[
            rng.integers(
                0,
                n,
                n
            )
        ]

        estimates.append(
            np.mean(sample)
        )

    estimates = np.asarray(
        estimates
    )

    return (

        float(
            np.mean(values)
        ),

        float(
            np.quantile(
                estimates,
                0.025
            )
        ),

        float(
            np.quantile(
                estimates,
                0.975
            )
        )
    )


# ============================================================
# PAIRED RANDOMIZATION TEST
# ============================================================

def paired_randomization(
    differences,
    permutations,
    seed
):

    differences = np.asarray(
        differences,
        dtype=np.float64
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:

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
        permutations
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

        value = abs(

            np.mean(
                differences
                *
                signs
            )
        )

        if value >= observed:

            extreme += 1

    return (

        extreme + 1

    ) / (

        permutations + 1
    )


# ============================================================
# PATIENCE-STYLE STOP RULE
# ============================================================

def select_budget(
    query_key,
    threshold,
    patience
):

    q = query_to_row[
        query_key
    ]

    streak = 0

    selected_index = (
        num_efs - 1
    )

    for j in range(
        1,
        num_efs
    ):

        if (
            stability[
                q,
                j
            ]
            >=
            threshold
        ):

            streak += 1

        else:

            streak = 0

        if streak >= patience:

            selected_index = j

            break

    return selected_index


# ============================================================
# GET PHASE-05 ROW FOR QUERY + ef
# ============================================================

budget_lookup = {

    (
        str(row["query_key"]),
        int(row["efSearch"])
    ):
    row

    for _, row
    in budget_df.iterrows()
}


# ============================================================
# BUILD SELECTED RESULTS
#
# Honest realized cost:
#
# We perform searches at every budget visited by the ladder.
# Therefore:
#
# realized_work =
# sum(distance computations of all visited ef values)
#
# We ALSO report the optimistic selected-search-only cost.
# ============================================================

def select_queries(
    query_keys,
    threshold,
    patience
):

    rows = []

    for query_key in query_keys:

        selected_index = select_budget(

            query_key,
            threshold,
            patience
        )

        selected_ef = EF_VALUES[
            selected_index
        ]

        selected_row = budget_lookup[
            (
                query_key,
                selected_ef
            )
        ]

        visited_efs = EF_VALUES[
            :selected_index + 1
        ]

        realized_work = 0.0
        realized_latency = 0.0

        for visited_ef in visited_efs:

            visited_row = budget_lookup[
                (
                    query_key,
                    visited_ef
                )
            ]

            realized_work += float(

                visited_row[
                    "distance_computations"
                ]
            )

            realized_latency += float(

                visited_row[
                    "mean_latency_ms"
                ]
            )

        rows.append({

            "query_key":
                query_key,

            "efSearch":
                selected_ef,

            "stability_threshold":
                threshold,

            "patience":
                patience,

            "visited_searches":
                selected_index + 1,

            "final_stability":
                (
                    stability[
                        query_to_row[
                            query_key
                        ],
                        selected_index
                    ]
                    if selected_index > 0
                    else np.nan
                ),

            "IR_unsafe":
                int(
                    selected_row[
                        "nDCG_loss"
                    ]
                    >
                    0.025
                ),

            "ANN_unsafe":
                int(
                    selected_row[
                        "ANN_Recall@10"
                    ]
                    <
                    0.90
                ),

            "nDCG@10":
                float(
                    selected_row[
                        "nDCG@10"
                    ]
                ),

            "nDCG_loss":
                float(
                    selected_row[
                        "nDCG_loss"
                    ]
                ),

            "ANN_Recall@10":
                float(
                    selected_row[
                        "ANN_Recall@10"
                    ]
                ),

            "selected_search_work":
                float(
                    selected_row[
                        "distance_computations"
                    ]
                ),

            "realized_search_work":
                realized_work,

            "selected_latency_ms":
                float(
                    selected_row[
                        "mean_latency_ms"
                    ]
                ),

            "realized_latency_ms":
                realized_latency
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# CALIBRATION GRID
# ============================================================

STABILITY_THRESHOLDS = [
    0.70,
    0.80,
    0.90,
    1.00
]

PATIENCE_VALUES = [
    1,
    2,
    3
]


tuning_rows = []


print(
    "\n"
    +
    "=" * 100
)

print(
    "CALIBRATING STABILITY BASELINE"
)

print(
    "=" * 100
)


for threshold in STABILITY_THRESHOLDS:

    for patience in PATIENCE_VALUES:

        selected = select_queries(

            calibration_keys,

            threshold,

            patience
        )

        violations = int(

            selected[
                "IR_unsafe"
            ].sum()
        )

        n = len(
            selected
        )

        violation_rate = (

            violations
            /
            n
        )

        upper = wilson_upper(

            violations,

            n
        )

        tuning_rows.append({

            "stability_threshold":
                threshold,

            "patience":
                patience,

            "queries":
                n,

            "IR_violation_rate":
                violation_rate,

            "Wilson_upper":
                upper,

            "mean_ef":
                selected[
                    "efSearch"
                ].mean(),

            "mean_selected_search_work":
                selected[
                    "selected_search_work"
                ].mean(),

            "mean_realized_search_work":
                selected[
                    "realized_search_work"
                ].mean(),

            "mean_realized_latency_ms":
                selected[
                    "realized_latency_ms"
                ].mean()
        })


tuning_df = pd.DataFrame(
    tuning_rows
)


tuning_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "patience_style_tuning.csv"
    ),

    index=False
)


feasible = tuning_df[

    tuning_df[
        "Wilson_upper"
    ]

    <=

    args.risk_alpha
]


if len(
    feasible
) > 0:

    chosen = (

        feasible

        .sort_values(

            [
                "mean_realized_search_work",
                "mean_ef"
            ]
        )

        .iloc[
            0
        ]
    )

    mode = "SLA satisfied"

else:

    chosen = (

        tuning_df

        .sort_values(

            [
                "Wilson_upper",
                "mean_realized_search_work"
            ]
        )

        .iloc[
            0
        ]
    )

    mode = (
        "No Wilson-feasible setting; "
        "safest setting selected"
    )


BEST_THRESHOLD = float(

    chosen[
        "stability_threshold"
    ]
)

BEST_PATIENCE = int(

    chosen[
        "patience"
    ]
)


print(
    "\nSelected stability threshold =",
    BEST_THRESHOLD
)

print(
    "Selected patience =",
    BEST_PATIENCE
)

print(
    "Calibration IR violation rate =",
    chosen[
        "IR_violation_rate"
    ]
)

print(
    "Calibration Wilson upper =",
    chosen[
        "Wilson_upper"
    ]
)

print(
    "Calibration mean realized work =",
    chosen[
        "mean_realized_search_work"
    ]
)

print(
    "Selection mode =",
    mode
)


# ============================================================
# FINAL TESTS
# ============================================================

test_selected = select_queries(

    test_keys,

    BEST_THRESHOLD,

    BEST_PATIENCE
)


ood_selected = select_queries(

    ood_keys,

    BEST_THRESHOLD,

    BEST_PATIENCE
)


test_selected.to_csv(

    os.path.join(
        RESULTS_DIR,
        "patience_style_msmarco_test.csv"
    ),

    index=False
)


ood_selected.to_csv(

    os.path.join(
        RESULTS_DIR,
        "patience_style_trec_ood.csv"
    ),

    index=False
)


# ============================================================
# SUMMARIZE
# ============================================================

def summarize(
    df,
    split_name
):

    violations = int(
        df[
            "IR_unsafe"
        ].sum()
    )

    n = len(
        df
    )

    return {

        "split":
            split_name,

        "method":
            "Patience-Inspired-Stability",

        "queries":
            n,

        "mean_ef":
            df[
                "efSearch"
            ].mean(),

        "median_ef":
            df[
                "efSearch"
            ].median(),

        "IR_violation_rate":
            df[
                "IR_unsafe"
            ].mean(),

        "IR_Wilson_upper":
            wilson_upper(
                violations,
                n
            ),

        "ANN_violation_rate":
            df[
                "ANN_unsafe"
            ].mean(),

        "mean_ANN_Recall@10":
            df[
                "ANN_Recall@10"
            ].mean(),

        "mean_nDCG@10":
            df[
                "nDCG@10"
            ].mean(),

        "mean_nDCG_loss":
            df[
                "nDCG_loss"
            ].mean(),

        "mean_selected_search_work":
            df[
                "selected_search_work"
            ].mean(),

        "mean_realized_search_work":
            df[
                "realized_search_work"
            ].mean(),

        "mean_realized_latency_ms":
            df[
                "realized_latency_ms"
            ].mean(),

        "p95_realized_latency_ms":
            df[
                "realized_latency_ms"
            ].quantile(
                0.95
            ),

        "mean_number_of_searches":
            df[
                "visited_searches"
            ].mean()
    }


summary_df = pd.DataFrame([

    summarize(
        test_selected,
        "MSMARCO_TEST"
    ),

    summarize(
        ood_selected,
        "TREC_DL_2019_OOD"
    )
])


summary_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "patience_style_summary.csv"
    ),

    index=False
)


# ============================================================
# LOAD CURA RESULTS
# ============================================================

cura_table = pd.read_csv(
    CURA_TABLE_FILE
)

cura_test = pd.read_csv(
    CURA_TEST_FILE,
    dtype={
        "query_key": str
    }
)

cura_ood = pd.read_csv(
    CURA_OOD_FILE,
    dtype={
        "query_key": str
    }
)


# ============================================================
# PAIRED CURA VS STABILITY BASELINE
#
# Positive difference:
# stability baseline uses MORE work than CURA.
# ============================================================

patience_test_indexed = (

    test_selected

    .set_index(
        "query_key"
    )
)


cura_test_indexed = (

    cura_test

    .set_index(
        "query_key"
    )
)


common = (

    patience_test_indexed.index

    .intersection(
        cura_test_indexed.index
    )
)


work_difference = (

    patience_test_indexed.loc[

        common,

        "realized_search_work"

    ].to_numpy(
        dtype=np.float64
    )

    -

    cura_test_indexed.loc[

        common,

        "effective_distance_computations"

    ].to_numpy(
        dtype=np.float64
    )
)


(
    mean_difference,
    ci_low,
    ci_high
) = bootstrap_ci(

    work_difference,

    args.bootstrap,

    SEED + 100
)


p_value = paired_randomization(

    work_difference,

    args.permutations,

    SEED + 200
)


significance_df = pd.DataFrame([{

    "comparison":
        (
            "Patience-inspired realized work "
            "minus CURA-Full realized work"
        ),

    "mean_work_difference":
        mean_difference,

    "95CI_low":
        ci_low,

    "95CI_high":
        ci_high,

    "paired_randomization_p":
        p_value
}])


significance_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "cura_vs_patience_style_significance.csv"
    ),

    index=False
)


# ============================================================
# COMBINED PAPER TABLE
# ============================================================

comparison_rows = []


for _, row in cura_table.iterrows():

    if row[
        "method"
    ] in [

        "Fixed-IR",
        "ANN-Risk",
        "CURA-Query",
        "CURA-Full",
        "IR-Oracle",
        "ANN-Oracle"
    ]:

        comparison_rows.append({

            "split":
                row[
                    "split"
                ],

            "method":
                row[
                    "method"
                ],

            "queries":
                row[
                    "queries"
                ],

            "mean_ef":
                row[
                    "mean_ef"
                ],

            "IR_violation_rate":
                row[
                    "IR_violation_rate"
                ],

            "ANN_violation_rate":
                row[
                    "ANN_violation_rate"
                ],

            "mean_nDCG@10":
                row[
                    "mean_nDCG@10"
                ],

            "mean_search_work":
                row[
                    "mean_search_work"
                ],

            "cost_definition":
                "effective realized cost"
        })


for _, row in summary_df.iterrows():

    comparison_rows.append({

        "split":
            row[
                "split"
            ],

        "method":
            row[
                "method"
            ],

        "queries":
            row[
                "queries"
            ],

        "mean_ef":
            row[
                "mean_ef"
            ],

        "IR_violation_rate":
            row[
                "IR_violation_rate"
            ],

        "ANN_violation_rate":
            row[
                "ANN_violation_rate"
            ],

        "mean_nDCG@10":
            row[
                "mean_nDCG@10"
            ],

        "mean_search_work":
            row[
                "mean_realized_search_work"
            ],

        "cost_definition":
            "sum of all stability-ladder searches"
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "comparison_with_cura.csv"
    ),

    index=False
)


# ============================================================
# OBJECTIVE INTERPRETATION
# ============================================================

test_ir_safe = (

    test_selected[
        "IR_unsafe"
    ].mean()

    <=

    args.risk_alpha
)


ood_ir_safe = (

    ood_selected[
        "IR_unsafe"
    ].mean()

    <=

    args.risk_alpha
)


cura_test_row = cura_table[

    (
        cura_table[
            "split"
        ]
        ==
        "MSMARCO_TEST"
    )

    &

    (
        cura_table[
            "method"
        ]
        ==
        "CURA-Full"
    )

].iloc[
    0
]


patience_mean_work = test_selected[
    "realized_search_work"
].mean()


cura_mean_work = float(

    cura_test_row[
        "mean_search_work"
    ]
)


if (
    test_ir_safe
    and
    cura_mean_work
    <
    patience_mean_work
    and
    ci_low
    >
    0
):

    decision = (

        "CURA STRONGER THAN STABILITY BASELINE: "
        "both are evaluated on the same held-out split, "
        "and CURA uses significantly less realized search "
        "work while satisfying the downstream IR-risk target."
    )

elif test_ir_safe:

    decision = (

        "STABILITY BASELINE IS COMPETITIVE: "
        "the simple stability rule meets the downstream SLA. "
        "CURA must be compared carefully on efficiency."
    )

else:

    decision = (

        "STABILITY BASELINE FAILS THE HELD-OUT IR SLA: "
        "simple top-k stability is not sufficient for "
        "safe downstream stopping under this protocol."
    )


with open(

    os.path.join(
        RESULTS_DIR,
        "baseline_decision.txt"
    ),

    "w",

    encoding="utf-8"
) as f:

    f.write(
        decision
        +
        "\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "PATIENCE-INSPIRED BASELINE COMPLETE"
)

print(
    "=" * 100
)


print(
    "\nSELECTED PARAMETERS"
)

print(
    "Stability threshold =",
    BEST_THRESHOLD
)

print(
    "Patience =",
    BEST_PATIENCE
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "FINAL STABILITY BASELINE"
)

print(
    "=" * 100
)


print(

    summary_df.to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "CURA VS PATIENCE-INSPIRED SEARCH WORK"
)

print(
    "=" * 100
)


print(
    "Mean baseline minus CURA work =",
    mean_difference
)

print(
    "95% CI = [",
    ci_low,
    ",",
    ci_high,
    "]"
)

print(
    "Paired randomization p =",
    p_value
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "OBJECTIVE BASELINE DECISION"
)

print(
    "=" * 100
)

print(
    decision
)


print(
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)


print(
    "\nIMPORTANT:"
)

print(
    "This experiment is a Patience-INSPIRED "
    "top-k stability baseline."
)

print(
    "Do NOT describe it as an official reproduction "
    "of the ECIR Patience implementation."
)

print(
    "\nDONE."
)
