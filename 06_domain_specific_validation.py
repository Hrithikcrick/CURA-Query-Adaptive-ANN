# ============================================================
# 06_domain_specific_validation.py
#
# COL764 PROJECT
# DOMAIN-SPECIFIC VALIDATION:
#
# MS MARCO DEV vs TREC-DL-2019
#
# PURPOSE
# -------
# Determine whether the ANN-recall vs downstream IR-utility
# budget gap observed in Phase 05 is:
#
#   1. present on MS MARCO dev queries,
#   2. present on deeply judged TREC-DL-2019 queries,
#   3. robust across utility tolerances,
#   4. concentrated among hard queries.
#
# IMPORTANT:
# No ANN search is rerun.
# No embeddings are recomputed.
#
# This script reads cached Phase-05 outputs only.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import sys
import json
import math
import argparse
import platform

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 2. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_docs",
    type=int,
    default=200000
)

parser.add_argument(
    "--n_dev_queries",
    type=int,
    default=300
)

parser.add_argument(
    "--bootstrap",
    type=int,
    default=10000
)

parser.add_argument(
    "--permutations",
    type=int,
    default=20000
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = args.seed

np.random.seed(
    SEED
)


# ============================================================
# 4. PATHS
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


PHASE05_DIR = os.path.join(
    BASE_DIR,
    "research_grade_validation"
)


RESULTS_DIR = os.path.join(
    BASE_DIR,
    "domain_specific_validation"
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 5. INPUT FILES
# ============================================================

TOP10_RESULTS_FILE = os.path.join(
    PHASE05_DIR,
    "top10_query_budget_results.csv"
)


TOP100_RESULTS_FILE = os.path.join(
    PHASE05_DIR,
    "top100_query_budget_results.csv"
)


EXACT_REFERENCE_FILE = os.path.join(
    PHASE05_DIR,
    "exact_query_reference.csv"
)


# ============================================================
# 6. VERIFY INPUTS
# ============================================================

required_files = [
    TOP10_RESULTS_FILE,
    TOP100_RESULTS_FILE,
    EXACT_REFERENCE_FILE
]


missing = [

    path

    for path in required_files

    if not os.path.exists(
        path
    )
]


if missing:

    print(
        "\nERROR: Phase-05 result files are missing."
    )

    print(
        "\nMissing:"
    )

    for path in missing:

        print(
            "  ",
            path
        )

    print(
        "\nRun 05_research_grade_validation.py first."
    )

    sys.exit(
        1
    )


# ============================================================
# 7. LOAD DATA
# ============================================================

print(
    "=" * 100
)

print(
    "DOMAIN-SPECIFIC VALIDATION"
)

print(
    "=" * 100
)


top10_df = pd.read_csv(
    TOP10_RESULTS_FILE
)


top100_df = pd.read_csv(
    TOP100_RESULTS_FILE
)


exact_df = pd.read_csv(
    EXACT_REFERENCE_FILE
)


print(
    "\nTop-10 rows:",
    len(
        top10_df
    )
)


print(
    "Top-100 rows:",
    len(
        top100_df
    )
)


print(
    "Unique queries:",
    exact_df[
        "query_key"
    ].nunique()
)


print(
    "\nQuery distribution:"
)


print(
    exact_df[
        "source"
    ].value_counts()
)


# ============================================================
# 8. EXPERIMENT CONSTANTS
# ============================================================

ANN_TARGET = 0.90


TOP10_EF_VALUES = sorted(

    top10_df[
        "efSearch"
    ]

    .unique()
)


TOP100_EF_VALUES = sorted(

    top100_df[
        "efSearch"
    ]

    .unique()
)


NDCG_TOLERANCES = [
    0.0,
    0.01,
    0.025,
    0.05
]


RECALL100_TOLERANCES = [
    0.0,
    0.05,
    0.10
]


PRIMARY_NDCG_TOLERANCE = 0.025

PRIMARY_RECALL100_TOLERANCE = 0.05


SOURCES = [

    "msmarco-dev-small",

    "trec-dl-2019"
]


# ============================================================
# 9. BOOTSTRAP CONFIDENCE INTERVAL
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


    estimates = np.empty(

        n_bootstrap,

        dtype=np.float64
    )


    for i in range(
        n_bootstrap
    ):

        sample_indices = rng.integers(

            0,

            n,

            size=n
        )


        estimates[
            i
        ] = np.mean(

            values[
                sample_indices
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
# 10. BOOTSTRAP PROPORTION CI
# ============================================================

def bootstrap_proportion_ci(
    binary_values,
    n_bootstrap,
    seed
):

    return bootstrap_mean_ci(

        binary_values,

        n_bootstrap,

        seed
    )


# ============================================================
# 11. PAIRED RANDOMIZATION TEST
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


    extreme_count = 0


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

            extreme_count += 1


    return (

        extreme_count
        +
        1

    ) / (

        n_permutations
        +
        1
    )


# ============================================================
# 12. ORACLE BUDGET CALCULATION
# ============================================================

def compute_oracle(

    dataframe,

    source,

    ef_values,

    ann_column,

    ir_loss_column,

    ir_tolerance
):

    working = dataframe[

        dataframe[
            "source"
        ]
        ==
        source
    ].copy()


    oracle_rows = []


    for query_key, group in working.groupby(
        "query_key"
    ):

        group = group.sort_values(
            "efSearch"
        )


        ann_safe_rows = group[

            group[
                ann_column
            ]

            >=

            ANN_TARGET
        ]


        ir_safe_rows = group[

            group[
                ir_loss_column
            ]

            <=

            ir_tolerance
        ]


        if len(
            ann_safe_rows
        ) > 0:

            min_ann_ef = int(

                ann_safe_rows.iloc[
                    0
                ][
                    "efSearch"
                ]
            )

            ann_found = 1

        else:

            min_ann_ef = int(
                ef_values[-1]
            )

            ann_found = 0


        if len(
            ir_safe_rows
        ) > 0:

            min_ir_ef = int(

                ir_safe_rows.iloc[
                    0
                ][
                    "efSearch"
                ]
            )

            ir_found = 1

        else:

            min_ir_ef = int(
                ef_values[-1]
            )

            ir_found = 0


        oracle_rows.append({

            "query_key":
                query_key,

            "source":
                source,

            "min_ANN_safe_ef":
                min_ann_ef,

            "min_IR_safe_ef":
                min_ir_ef,

            "ANN_minus_IR_gap":
                (
                    min_ann_ef

                    -

                    min_ir_ef
                ),

            "IR_allows_lower":
                int(
                    min_ir_ef
                    <
                    min_ann_ef
                ),

            "same_budget":
                int(
                    min_ir_ef
                    ==
                    min_ann_ef
                ),

            "IR_requires_higher":
                int(
                    min_ir_ef
                    >
                    min_ann_ef
                ),

            "ANN_safe_found":
                ann_found,

            "IR_safe_found":
                ir_found
        })


    return pd.DataFrame(
        oracle_rows
    )


# ============================================================
# 13. SUMMARIZE ONE DOMAIN
# ============================================================

def summarize_domain_oracle(

    oracle_df,

    task,

    source,

    tolerance,

    seed_offset
):

    gaps = oracle_df[
        "ANN_minus_IR_gap"
    ].to_numpy(
        dtype=np.float64
    )


    (
        mean_gap,
        gap_low,
        gap_high
    ) = bootstrap_mean_ci(

        gaps,

        args.bootstrap,

        SEED
        +
        seed_offset
    )


    (
        lower_rate,
        lower_low,
        lower_high
    ) = bootstrap_proportion_ci(

        oracle_df[
            "IR_allows_lower"
        ].values,

        args.bootstrap,

        SEED
        +
        seed_offset
        +
        1000
    )


    p_value = paired_randomization_pvalue(

        gaps,

        args.permutations,

        SEED
        +
        seed_offset
        +
        2000
    )


    mean_ann = float(

        oracle_df[
            "min_ANN_safe_ef"
        ].mean()
    )


    mean_ir = float(

        oracle_df[
            "min_IR_safe_ef"
        ].mean()
    )


    if mean_ann > 0:

        relative_reduction = (

            (
                mean_ann

                -

                mean_ir
            )

            /

            mean_ann

            *

            100.0
        )

    else:

        relative_reduction = 0.0


    return {

        "task":
            task,

        "source":
            source,

        "tolerance":
            tolerance,

        "queries":
            len(
                oracle_df
            ),

        "mean_ANN_safe_ef":
            mean_ann,

        "mean_IR_safe_ef":
            mean_ir,

        "mean_budget_gap":
            mean_gap,

        "gap_95CI_low":
            gap_low,

        "gap_95CI_high":
            gap_high,

        "relative_budget_reduction_%":
            relative_reduction,

        "IR_allows_lower_rate":
            lower_rate,

        "lower_rate_95CI_low":
            lower_low,

        "lower_rate_95CI_high":
            lower_high,

        "same_budget_rate":
            oracle_df[
                "same_budget"
            ].mean(),

        "IR_requires_higher_rate":
            oracle_df[
                "IR_requires_higher"
            ].mean(),

        "paired_randomization_p":
            p_value,

        "ANN_safe_found_rate":
            oracle_df[
                "ANN_safe_found"
            ].mean(),

        "IR_safe_found_rate":
            oracle_df[
                "IR_safe_found"
            ].mean()
    }


# ============================================================
# 14. TOP-10 DOMAIN ANALYSIS
# ============================================================

top10_summary_rows = []

top10_query_rows = []


counter = 0


for source in SOURCES:

    for tolerance in NDCG_TOLERANCES:

        counter += 1


        oracle_df = compute_oracle(

            dataframe=top10_df,

            source=source,

            ef_values=TOP10_EF_VALUES,

            ann_column="ANN_Recall@10",

            ir_loss_column="nDCG_loss",

            ir_tolerance=tolerance
        )


        oracle_df[
            "tolerance"
        ] = tolerance


        oracle_df[
            "task"
        ] = "TOP10_NDCG"


        top10_query_rows.append(
            oracle_df
        )


        summary = summarize_domain_oracle(

            oracle_df=oracle_df,

            task="TOP10_NDCG",

            source=source,

            tolerance=tolerance,

            seed_offset=counter * 100
        )


        top10_summary_rows.append(
            summary
        )


top10_domain_query_df = pd.concat(

    top10_query_rows,

    ignore_index=True
)


top10_domain_summary_df = pd.DataFrame(

    top10_summary_rows
)


# ============================================================
# 15. TOP-100 DOMAIN ANALYSIS
# ============================================================

top100_summary_rows = []

top100_query_rows = []


counter = 0


for source in SOURCES:

    for tolerance in RECALL100_TOLERANCES:

        counter += 1


        oracle_df = compute_oracle(

            dataframe=top100_df,

            source=source,

            ef_values=TOP100_EF_VALUES,

            ann_column="ANN_Recall@100",

            ir_loss_column="Recall100_loss",

            ir_tolerance=tolerance
        )


        oracle_df[
            "tolerance"
        ] = tolerance


        oracle_df[
            "task"
        ] = "TOP100_RECALL"


        top100_query_rows.append(
            oracle_df
        )


        summary = summarize_domain_oracle(

            oracle_df=oracle_df,

            task="TOP100_RECALL",

            source=source,

            tolerance=tolerance,

            seed_offset=5000
            +
            counter * 100
        )


        top100_summary_rows.append(
            summary
        )


top100_domain_query_df = pd.concat(

    top100_query_rows,

    ignore_index=True
)


top100_domain_summary_df = pd.DataFrame(

    top100_summary_rows
)


# ============================================================
# 16. SAVE DOMAIN TABLES
# ============================================================

top10_domain_query_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "top10_domain_query_oracles.csv"
    ),

    index=False
)


top10_domain_summary_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "top10_domain_statistics.csv"
    ),

    index=False
)


top100_domain_query_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "top100_domain_query_oracles.csv"
    ),

    index=False
)


top100_domain_summary_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "top100_domain_statistics.csv"
    ),

    index=False
)


# ============================================================
# 17. PRIMARY TOP-10 DOMAIN RESULT
# ============================================================

primary_top10 = top10_domain_summary_df[

    top10_domain_summary_df[
        "tolerance"
    ]
    ==
    PRIMARY_NDCG_TOLERANCE

].copy()


primary_top100 = top100_domain_summary_df[

    top100_domain_summary_df[
        "tolerance"
    ]
    ==
    PRIMARY_RECALL100_TOLERANCE

].copy()


# ============================================================
# 18. HARDNESS ANALYSIS WITHIN EACH DOMAIN
# ============================================================

hardness_results = []


primary_top10_query_oracle = (

    top10_domain_query_df[

        top10_domain_query_df[
            "tolerance"
        ]
        ==
        PRIMARY_NDCG_TOLERANCE

    ]

    .copy()
)


merged_hardness = (

    primary_top10_query_oracle

    .merge(

        exact_df[
            [
                "query_key",
                "boundary_gap_10_11",
                "top1_top10_gap",
                "mean_top10_similarity",
                "positive_qrel_count"
            ]
        ],

        on="query_key",

        how="left"
    )
)


for source in SOURCES:

    source_df = merged_hardness[

        merged_hardness[
            "source"
        ]
        ==
        source

    ].copy()


    if len(
        source_df
    ) < 8:

        continue


    try:

        source_df[
            "hardness_quartile"
        ] = pd.qcut(

            source_df[
                "boundary_gap_10_11"
            ],

            q=4,

            labels=[
                "Q1_small_gap",
                "Q2",
                "Q3",
                "Q4_large_gap"
            ],

            duplicates="drop"
        )


    except Exception:

        source_df[
            "hardness_quartile"
        ] = "ALL"


    grouped = (

        source_df

        .groupby(
            "hardness_quartile",
            observed=True
        )

        .agg(

            queries=(
                "query_key",
                "count"
            ),

            mean_boundary_gap=(
                "boundary_gap_10_11",
                "mean"
            ),

            mean_ANN_safe_ef=(
                "min_ANN_safe_ef",
                "mean"
            ),

            mean_IR_safe_ef=(
                "min_IR_safe_ef",
                "mean"
            ),

            mean_budget_gap=(
                "ANN_minus_IR_gap",
                "mean"
            ),

            IR_allows_lower_rate=(
                "IR_allows_lower",
                "mean"
            ),

            IR_requires_higher_rate=(
                "IR_requires_higher",
                "mean"
            )

        )

        .reset_index()
    )


    grouped[
        "source"
    ] = source


    hardness_results.append(
        grouped
    )


if hardness_results:

    hardness_summary_df = pd.concat(

        hardness_results,

        ignore_index=True
    )

else:

    hardness_summary_df = pd.DataFrame()


hardness_summary_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "domain_hardness_statistics.csv"
    ),

    index=False
)


# ============================================================
# 19. EFFECT-SIZE COMPARISON
# ============================================================

domain_effect_rows = []


for _, row in primary_top10.iterrows():

    domain_effect_rows.append({

        "source":
            row[
                "source"
            ],

        "queries":
            row[
                "queries"
            ],

        "top10_relative_budget_reduction_%":
            row[
                "relative_budget_reduction_%"
            ],

        "top10_IR_lower_rate":
            row[
                "IR_allows_lower_rate"
            ],

        "top10_gap":
            row[
                "mean_budget_gap"
            ],

        "top10_gap_CI_low":
            row[
                "gap_95CI_low"
            ],

        "top10_gap_CI_high":
            row[
                "gap_95CI_high"
            ],

        "top10_p":
            row[
                "paired_randomization_p"
            ]
    })


domain_effect_df = pd.DataFrame(

    domain_effect_rows
)


# Add top-100 fields

for i, row in domain_effect_df.iterrows():

    source = row[
        "source"
    ]


    top100_match = primary_top100[

        primary_top100[
            "source"
        ]
        ==
        source
    ]


    if len(
        top100_match
    ) == 1:

        matched = top100_match.iloc[
            0
        ]


        domain_effect_df.loc[
            i,
            "top100_relative_budget_reduction_%"
        ] = matched[
            "relative_budget_reduction_%"
        ]


        domain_effect_df.loc[
            i,
            "top100_IR_lower_rate"
        ] = matched[
            "IR_allows_lower_rate"
        ]


        domain_effect_df.loc[
            i,
            "top100_gap"
        ] = matched[
            "mean_budget_gap"
        ]


        domain_effect_df.loc[
            i,
            "top100_p"
        ] = matched[
            "paired_randomization_p"
        ]


domain_effect_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "domain_effect_comparison.csv"
    ),

    index=False
)


# ============================================================
# 20. INTERPRETATION FLAGS
# ============================================================

interpretation_rows = []


for _, row in primary_top10.iterrows():

    source = row[
        "source"
    ]


    significant = (

        row[
            "gap_95CI_low"
        ]
        >
        0

        and

        row[
            "paired_randomization_p"
        ]
        <
        0.05
    )


    meaningful_rate = (

        row[
            "IR_allows_lower_rate"
        ]
        >=
        0.10
    )


    if significant and meaningful_rate:

        interpretation = (
            "STRONG: statistically supported and "
            "substantively frequent utility gap."
        )


    elif significant:

        interpretation = (
            "STATISTICALLY SUPPORTED but effect "
            "is less frequent."
        )


    elif (
        row[
            "mean_budget_gap"
        ]
        >
        0
    ):

        interpretation = (
            "WEAK: positive mean gap, but current "
            "evidence is insufficient."
        )


    else:

        interpretation = (
            "NO SUPPORT for lower IR-safe search effort."
        )


    interpretation_rows.append({

        "source":
            source,

        "interpretation":
            interpretation
    })


interpretation_df = pd.DataFrame(

    interpretation_rows
)


interpretation_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "domain_interpretation.csv"
    ),

    index=False
)


# ============================================================
# 21. PLOT:
# DOMAIN TOP-10 ORACLE BUDGET
# ============================================================

plt.figure(
    figsize=(8, 5)
)


x = np.arange(
    len(
        primary_top10
    )
)


width = 0.35


plt.bar(

    x
    -
    width / 2,

    primary_top10[
        "mean_ANN_safe_ef"
    ],

    width,

    label="ANN-safe"
)


plt.bar(

    x
    +
    width / 2,

    primary_top10[
        "mean_IR_safe_ef"
    ],

    width,

    label="nDCG-safe"
)


plt.xticks(

    x,

    primary_top10[
        "source"
    ],

    rotation=15
)


plt.ylabel(
    "Mean minimum efSearch"
)


plt.title(
    "Top-10 Oracle Budget by Query Set"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "top10_oracle_budget_by_domain.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 22. PLOT:
# DOMAIN RELATIVE BUDGET REDUCTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.bar(

    primary_top10[
        "source"
    ],

    primary_top10[
        "relative_budget_reduction_%"
    ]
)


plt.ylabel(
    "Oracle budget reduction (%)"
)


plt.xlabel(
    "Query set"
)


plt.title(
    "Top-10 ANN-safe vs nDCG-safe Budget Gap"
)


plt.xticks(
    rotation=15
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "top10_relative_reduction_by_domain.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 23. PLOT:
# TOLERANCE SENSITIVITY BY DOMAIN
# ============================================================

plt.figure(
    figsize=(8, 5)
)


for source in SOURCES:

    subset = (

        top10_domain_summary_df[

            top10_domain_summary_df[
                "source"
            ]
            ==
            source

        ]

        .sort_values(
            "tolerance"
        )
    )


    plt.plot(

        subset[
            "tolerance"
        ],

        subset[
            "relative_budget_reduction_%"
        ],

        marker="o",

        label=source
    )


plt.xlabel(
    "Allowed nDCG@10 degradation"
)


plt.ylabel(
    "Oracle budget reduction (%)"
)


plt.title(
    "Tolerance Sensitivity by Query Set"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "domain_tolerance_sensitivity.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 24. PLOT:
# HARDNESS x DOMAIN
# ============================================================

if len(
    hardness_summary_df
) > 0:

    plt.figure(
        figsize=(9, 5)
    )


    for source in SOURCES:

        subset = hardness_summary_df[

            hardness_summary_df[
                "source"
            ]
            ==
            source
        ]


        if len(
            subset
        ) == 0:

            continue


        plt.plot(

            subset[
                "hardness_quartile"
            ].astype(
                str
            ),

            subset[
                "mean_budget_gap"
            ],

            marker="o",

            label=source
        )


    plt.xlabel(
        "Exact top-10 boundary-gap quartile"
    )


    plt.ylabel(
        "Mean ANN-safe ef - nDCG-safe ef"
    )


    plt.title(
        "Utility Gap by Query Hardness and Domain"
    )


    plt.xticks(
        rotation=20
    )


    plt.grid(
        True
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(
            RESULTS_DIR,
            "hardness_gap_by_domain.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 25. SAVE REPRODUCIBILITY METADATA
# ============================================================

metadata = {

    "seed":
        SEED,

    "corpus_experiment":
        RUN_NAME,

    "bootstrap_samples":
        args.bootstrap,

    "permutation_samples":
        args.permutations,

    "ANN_target":
        ANN_TARGET,

    "primary_nDCG_tolerance":
        PRIMARY_NDCG_TOLERANCE,

    "primary_Recall100_tolerance":
        PRIMARY_RECALL100_TOLERANCE,

    "top10_tolerances":
        NDCG_TOLERANCES,

    "top100_tolerances":
        RECALL100_TOLERANCES,

    "sources":
        SOURCES,

    "python":
        sys.version,

    "platform":
        platform.platform()
}


with open(

    os.path.join(
        RESULTS_DIR,
        "domain_validation_protocol.json"
    ),

    "w",

    encoding="utf-8"
) as f:

    json.dump(

        metadata,

        f,

        indent=2
    )


# ============================================================
# 26. PAPER-READY TABLE
# ============================================================

paper_rows = []


for _, row in primary_top10.iterrows():

    source = row[
        "source"
    ]


    paper_rows.append({

        "Query_Set":
            source,

        "Queries":
            int(
                row[
                    "queries"
                ]
            ),

        "ANN_safe_ef":
            row[
                "mean_ANN_safe_ef"
            ],

        "nDCG_safe_ef":
            row[
                "mean_IR_safe_ef"
            ],

        "Absolute_gap":
            row[
                "mean_budget_gap"
            ],

        "Gap_95CI":
            (
                f"["
                f"{row['gap_95CI_low']:.3f}, "
                f"{row['gap_95CI_high']:.3f}"
                f"]"
            ),

        "Relative_reduction_%":
            row[
                "relative_budget_reduction_%"
            ],

        "IR_allows_lower_%":
            (
                100.0
                *
                row[
                    "IR_allows_lower_rate"
                ]
            ),

        "p_value":
            row[
                "paired_randomization_p"
            ]
    })


paper_table_df = pd.DataFrame(

    paper_rows
)


paper_table_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "paper_ready_domain_table.csv"
    ),

    index=False
)


# ============================================================
# 27. TEXT SUMMARY
# ============================================================

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "domain_validation_summary.txt"
)


with open(

    SUMMARY_FILE,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "DOMAIN-SPECIFIC VALIDATION\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )


    f.write(
        "PRIMARY TOP-10 RESULT\n"
    )

    f.write(
        f"nDCG tolerance = "
        f"{PRIMARY_NDCG_TOLERANCE}\n\n"
    )


    for _, row in primary_top10.iterrows():

        f.write(
            f"Source: "
            f"{row['source']}\n"
        )


        f.write(
            f"Queries: "
            f"{int(row['queries'])}\n"
        )


        f.write(
            f"Mean ANN-safe ef: "
            f"{row['mean_ANN_safe_ef']:.4f}\n"
        )


        f.write(
            f"Mean nDCG-safe ef: "
            f"{row['mean_IR_safe_ef']:.4f}\n"
        )


        f.write(
            f"Relative reduction: "
            f"{row['relative_budget_reduction_%']:.2f}%\n"
        )


        f.write(
            f"IR allows lower rate: "
            f"{row['IR_allows_lower_rate']:.4f}\n"
        )


        f.write(
            f"Gap 95% CI: "
            f"[{row['gap_95CI_low']:.4f}, "
            f"{row['gap_95CI_high']:.4f}]\n"
        )


        f.write(
            f"p-value: "
            f"{row['paired_randomization_p']:.6g}\n\n"
        )


    f.write(
        "\nINTERPRETATION\n"
    )


    for _, row in interpretation_df.iterrows():

        f.write(
            f"{row['source']}: "
            f"{row['interpretation']}\n"
        )


# ============================================================
# 28. FINAL TERMINAL OUTPUT
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "DOMAIN-SPECIFIC VALIDATION COMPLETED"
)

print(
    "=" * 100
)


print(
    "\nPRIMARY TOP-10 RESULTS"
)

print(
    "nDCG tolerance =",
    PRIMARY_NDCG_TOLERANCE
)


for _, row in primary_top10.iterrows():

    print(
        "\n"
        +
        "-" * 80
    )


    print(
        "Query set:",
        row[
            "source"
        ]
    )


    print(
        "Queries =",
        int(
            row[
                "queries"
            ]
        )
    )


    print(
        "Mean ANN-safe ef =",
        float(
            row[
                "mean_ANN_safe_ef"
            ]
        )
    )


    print(
        "Mean nDCG-safe ef =",
        float(
            row[
                "mean_IR_safe_ef"
            ]
        )
    )


    print(
        "Mean absolute gap =",
        float(
            row[
                "mean_budget_gap"
            ]
        )
    )


    print(
        "95% CI = [",
        float(
            row[
                "gap_95CI_low"
            ]
        ),
        ",",
        float(
            row[
                "gap_95CI_high"
            ]
        ),
        "]"
    )


    print(
        "Relative budget reduction (%) =",
        float(
            row[
                "relative_budget_reduction_%"
            ]
        )
    )


    print(
        "IR allows lower budget rate =",
        float(
            row[
                "IR_allows_lower_rate"
            ]
        )
    )


    print(
        "Same budget rate =",
        float(
            row[
                "same_budget_rate"
            ]
        )
    )


    print(
        "IR requires higher rate =",
        float(
            row[
                "IR_requires_higher_rate"
            ]
        )
    )


    print(
        "Paired randomization p-value =",
        float(
            row[
                "paired_randomization_p"
            ]
        )
    )


print(
    "\n"
    +
    "=" * 100
)

print(
    "TOP-10 TOLERANCE SWEEP BY DOMAIN"
)

print(
    "=" * 100
)


print(

    top10_domain_summary_df[

        [
            "source",
            "tolerance",
            "queries",
            "mean_ANN_safe_ef",
            "mean_IR_safe_ef",
            "relative_budget_reduction_%",
            "IR_allows_lower_rate",
            "IR_requires_higher_rate",
            "gap_95CI_low",
            "gap_95CI_high",
            "paired_randomization_p"
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
    "HARDNESS ANALYSIS BY DOMAIN"
)

print(
    "=" * 100
)


print(
    hardness_summary_df.to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "INTERPRETATION"
)

print(
    "=" * 100
)


print(
    interpretation_df.to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)


print(
    "\nDONE."
)