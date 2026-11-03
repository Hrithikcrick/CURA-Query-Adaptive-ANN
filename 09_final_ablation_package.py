# ============================================================
# 09_final_ablation_package.py
#
# FINAL ABLATION + BASELINE PACKAGE
#
# Tests what actually makes CURA useful:
#
# 1. Fixed-IR        -> no adaptation
# 2. ANN-Risk        -> learned adaptation targeting ANN recall
# 3. CURA-Query      -> downstream risk, query features only
# 4. CURA-Full       -> downstream risk + search-state features
# 5. Patience-Inspired Stability
# 6. ANN Oracle
# 7. IR Oracle
#
# IMPORTANT:
# ANN-Risk is NOT an official DARTH implementation.
# Patience-Inspired is NOT an official Patience reproduction.
# ============================================================

import os
import sys
import math
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

CURA_DIR = os.path.join(
    BASE_DIR,
    "cura_controller"
)

PATIENCE_DIR = os.path.join(
    BASE_DIR,
    "patience_style_baseline"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "final_ablation_package"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILE MAP
# ============================================================

CURA_METHODS = {

    "Fixed-IR":
        "selected_fixed_ir",

    "ANN-Risk":
        "selected_ann_risk",

    "CURA-Query":
        "selected_cura_query",

    "CURA-Full":
        "selected_cura_full",

    "IR-Oracle":
        "selected_ir_oracle",

    "ANN-Oracle":
        "selected_ann_oracle"
}


# ============================================================
# HELPERS
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

    n = len(values)

    means = np.empty(
        n_boot,
        dtype=np.float64
    )

    for i in range(
        n_boot
    ):

        sample = values[
            rng.integers(
                0,
                n,
                size=n
            )
        ]

        means[i] = np.mean(
            sample
        )

    return (

        float(
            np.mean(values)
        ),

        float(
            np.quantile(
                means,
                0.025
            )
        ),

        float(
            np.quantile(
                means,
                0.975
            )
        )
    )


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

        extreme + 1

    ) / (

        permutations + 1
    )


def exact_mcnemar_p(
    comparator_unsafe,
    cura_unsafe
):

    comparator_unsafe = np.asarray(
        comparator_unsafe,
        dtype=int
    )

    cura_unsafe = np.asarray(
        cura_unsafe,
        dtype=int
    )

    # comparator wrong, CURA correct
    b = int(
        np.sum(
            (
                comparator_unsafe == 1
            )
            &
            (
                cura_unsafe == 0
            )
        )
    )

    # comparator correct, CURA wrong
    c = int(
        np.sum(
            (
                comparator_unsafe == 0
            )
            &
            (
                cura_unsafe == 1
            )
        )
    )

    n = b + c

    if n == 0:

        return 1.0, b, c

    k = min(
        b,
        c
    )

    probability = 0.0

    for i in range(
        k + 1
    ):

        probability += (
            math.comb(
                n,
                i
            )
            /
            (
                2 ** n
            )
        )

    p_value = min(
        1.0,
        2.0 * probability
    )

    return (
        p_value,
        b,
        c
    )


# ============================================================
# NORMALIZE CURA METHOD FILE
# ============================================================

def load_cura_method(
    method,
    split_suffix,
    split_name
):

    prefix = CURA_METHODS[
        method
    ]

    path = os.path.join(

        CURA_DIR,

        (
            prefix
            +
            "_"
            +
            split_suffix
            +
            ".csv"
        )
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        dtype={
            "query_key": str
        }
    )

    df[
        "method"
    ] = method

    df[
        "split"
    ] = split_name

    df[
        "normalized_work"
    ] = df[
        "effective_distance_computations"
    ]

    df[
        "normalized_latency_ms"
    ] = df[
        "effective_latency_ms"
    ]

    return df


# ============================================================
# NORMALIZE PATIENCE-INSPIRED FILE
# ============================================================

def load_patience(
    filename,
    split_name
):

    path = os.path.join(
        PATIENCE_DIR,
        filename
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        dtype={
            "query_key": str
        }
    )

    df[
        "method"
    ] = "Patience-Inspired"

    df[
        "split"
    ] = split_name

    df[
        "normalized_work"
    ] = df[
        "realized_search_work"
    ]

    df[
        "normalized_latency_ms"
    ] = df[
        "realized_latency_ms"
    ]

    return df


# ============================================================
# LOAD ALL METHODS
# ============================================================

print(
    "=" * 100
)

print(
    "FINAL CURA ABLATION + BASELINE PACKAGE"
)

print(
    "=" * 100
)


all_frames = []


for method in CURA_METHODS:

    all_frames.append(

        load_cura_method(

            method,

            "msmarco_test",

            "MSMARCO_TEST"
        )
    )


all_frames.append(

    load_patience(

        "patience_style_msmarco_test.csv",

        "MSMARCO_TEST"
    )
)


for method in CURA_METHODS:

    all_frames.append(

        load_cura_method(

            method,

            "trec_ood",

            "TREC_DL_2019_OOD"
        )
    )


all_frames.append(

    load_patience(

        "patience_style_trec_ood.csv",

        "TREC_DL_2019_OOD"
    )
)


all_df = pd.concat(
    all_frames,
    ignore_index=True
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []


for (
    split_name,
    method
), group in all_df.groupby(
    [
        "split",
        "method"
    ]
):

    summary_rows.append({

        "split":
            split_name,

        "method":
            method,

        "queries":
            len(
                group
            ),

        "mean_ef":
            group[
                "efSearch"
            ].mean(),

        "median_ef":
            group[
                "efSearch"
            ].median(),

        "IR_violation_rate":
            group[
                "IR_unsafe"
            ].mean(),

        "ANN_violation_rate":
            group[
                "ANN_unsafe"
            ].mean(),

        "mean_ANN_Recall@10":
            group[
                "ANN_Recall@10"
            ].mean(),

        "mean_nDCG@10":
            group[
                "nDCG@10"
            ].mean(),

        "mean_nDCG_loss":
            group[
                "nDCG_loss"
            ].mean(),

        "mean_search_work":
            group[
                "normalized_work"
            ].mean(),

        "p50_search_work":
            group[
                "normalized_work"
            ].median(),

        "p95_search_work":
            group[
                "normalized_work"
            ].quantile(
                0.95
            ),

        "mean_latency_ms":
            group[
                "normalized_latency_ms"
            ].mean(),

        "p95_latency_ms":
            group[
                "normalized_latency_ms"
            ].quantile(
                0.95
            )
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVINGS RELATIVE TO FIXED
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

    fixed = summary_df[

        split_mask

        &

        (
            summary_df[
                "method"
            ]
            ==
            "Fixed-IR"
        )

    ].iloc[
        0
    ]

    fixed_work = float(
        fixed[
            "mean_search_work"
        ]
    )

    fixed_latency = float(
        fixed[
            "mean_latency_ms"
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
                "mean_latency_ms"
            ]
        )

        /
        fixed_latency
    )


summary_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "final_ablation_table.csv"
    ),

    index=False
)


# ============================================================
# HELD-OUT PAIRWISE COMPARISONS AGAINST CURA-FULL
# ============================================================

heldout = all_df[

    all_df[
        "split"
    ]
    ==
    "MSMARCO_TEST"

].copy()


cura = (

    heldout[

        heldout[
            "method"
        ]
        ==
        "CURA-Full"

    ]

    .set_index(
        "query_key"
    )
)


comparators = [

    "Fixed-IR",
    "ANN-Risk",
    "CURA-Query",
    "Patience-Inspired",
    "ANN-Oracle",
    "IR-Oracle"
]


pairwise_rows = []


for index, comparator in enumerate(
    comparators
):

    other = (

        heldout[

            heldout[
                "method"
            ]
            ==
            comparator

        ]

        .set_index(
            "query_key"
        )
    )

    common = cura.index.intersection(
        other.index
    )

    cura_common = cura.loc[
        common
    ]

    other_common = other.loc[
        common
    ]


    # Positive = comparator uses MORE work,
    # therefore CURA saves work.

    work_difference = (

        other_common[
            "normalized_work"
        ].to_numpy(
            dtype=np.float64
        )

        -

        cura_common[
            "normalized_work"
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

        SEED
        +
        index * 100
    )


    p_work = paired_randomization(

        work_difference,

        args.permutations,

        SEED
        +
        index * 1000
    )


    (
        mcnemar_p,
        comparator_wrong_cura_correct,
        comparator_correct_cura_wrong
    ) = exact_mcnemar_p(

        other_common[
            "IR_unsafe"
        ].to_numpy(),

        cura_common[
            "IR_unsafe"
        ].to_numpy()
    )


    comparator_mean_work = other_common[
        "normalized_work"
    ].mean()


    cura_mean_work = cura_common[
        "normalized_work"
    ].mean()


    relative_saving = (

        100.0

        *
        (
            comparator_mean_work
            -
            cura_mean_work
        )

        /
        comparator_mean_work

        if comparator_mean_work > 0

        else np.nan
    )


    pairwise_rows.append({

        "comparison":
            (
                comparator
                +
                " minus CURA-Full"
            ),

        "queries":
            len(
                common
            ),

        "comparator_mean_work":
            comparator_mean_work,

        "CURA_mean_work":
            cura_mean_work,

        "CURA_work_saving_%":
            relative_saving,

        "mean_work_difference":
            mean_difference,

        "work_95CI_low":
            ci_low,

        "work_95CI_high":
            ci_high,

        "paired_work_p":
            p_work,

        "comparator_IR_violation_rate":
            other_common[
                "IR_unsafe"
            ].mean(),

        "CURA_IR_violation_rate":
            cura_common[
                "IR_unsafe"
            ].mean(),

        "McNemar_exact_p":
            mcnemar_p,

        "comparator_wrong_CURA_correct":
            comparator_wrong_cura_correct,

        "comparator_correct_CURA_wrong":
            comparator_correct_cura_wrong
    })


pairwise_df = pd.DataFrame(
    pairwise_rows
)


pairwise_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "pairwise_cura_tests.csv"
    ),

    index=False
)


# ============================================================
# ABLATION INTERPRETATION
# ============================================================

def get_summary(
    split,
    method
):

    return summary_df[

        (
            summary_df[
                "split"
            ]
            ==
            split
        )

        &

        (
            summary_df[
                "method"
            ]
            ==
            method
        )

    ].iloc[
        0
    ]


fixed = get_summary(
    "MSMARCO_TEST",
    "Fixed-IR"
)

annrisk = get_summary(
    "MSMARCO_TEST",
    "ANN-Risk"
)

query_only = get_summary(
    "MSMARCO_TEST",
    "CURA-Query"
)

full = get_summary(
    "MSMARCO_TEST",
    "CURA-Full"
)

patience = get_summary(
    "MSMARCO_TEST",
    "Patience-Inspired"
)

ir_oracle = get_summary(
    "MSMARCO_TEST",
    "IR-Oracle"
)


interpretation_lines = []


# Adaptivity value

if (
    full[
        "mean_search_work"
    ]
    <
    fixed[
        "mean_search_work"
    ]

    and

    full[
        "IR_violation_rate"
    ]
    <=
    0.05
):

    interpretation_lines.append(

        "A1 PASS: adaptive downstream-risk allocation "
        "reduces search work relative to fixed search "
        "while meeting the 5% IR-risk target."
    )

else:

    interpretation_lines.append(

        "A1 FAIL: CURA does not simultaneously beat "
        "fixed search and meet the IR-risk target."
    )


# Downstream objective vs ANN objective

if (
    full[
        "mean_search_work"
    ]
    <
    annrisk[
        "mean_search_work"
    ]

    and

    full[
        "IR_violation_rate"
    ]
    <=
    annrisk[
        "IR_violation_rate"
    ]
    +
    0.01
):

    interpretation_lines.append(

        "A2 PASS: targeting downstream IR risk is more "
        "efficient than targeting ANN-recall risk under "
        "comparable downstream safety."
    )

else:

    interpretation_lines.append(

        "A2 MIXED: ANN-risk and downstream-risk allocation "
        "must be reported without claiming clear superiority."
    )


# Search-state feature value

if (
    full[
        "mean_search_work"
    ]
    <
    query_only[
        "mean_search_work"
    ]
):

    interpretation_lines.append(

        "A3 PASS: cheap search-state features improve "
        "efficiency over query-only CURA."
    )

else:

    interpretation_lines.append(

        "A3 NO GAIN: query-only CURA is as efficient as "
        "or more efficient than CURA-Full."
    )


# Stability baseline

if (
    full[
        "mean_search_work"
    ]
    <
    patience[
        "mean_search_work"
    ]
):

    interpretation_lines.append(

        "A4 PASS: CURA uses less realized search work "
        "than the Patience-inspired stability baseline."
    )

else:

    interpretation_lines.append(

        "A4 FAIL: the stability baseline is more efficient."
    )


# Oracle headroom

oracle_gap = (

    full[
        "mean_search_work"
    ]

    -

    ir_oracle[
        "mean_search_work"
    ]
)


if oracle_gap > 0:

    interpretation_lines.append(

        "A5: positive oracle headroom remains; CURA "
        "does not exhaust all theoretically available "
        "downstream-safe efficiency."
    )


# ============================================================
# OOD SAFETY
# ============================================================

ood_full = get_summary(
    "TREC_DL_2019_OOD",
    "CURA-Full"
)


if ood_full[
    "IR_violation_rate"
] <= 0.05:

    interpretation_lines.append(

        "OOD PASS: CURA-Full remains within the 5% "
        "downstream IR-risk target on TREC-DL-2019."
    )

else:

    interpretation_lines.append(

        "OOD LIMITATION: CURA-Full exceeds the 5% "
        "downstream IR-risk target on TREC-DL-2019."
    )


with open(

    os.path.join(
        RESULTS_DIR,
        "ablation_interpretation.txt"
    ),

    "w",

    encoding="utf-8"

) as f:

    for line in interpretation_lines:

        f.write(
            line
            +
            "\n"
        )


# ============================================================
# PAPER-READY SHORT TABLE
# ============================================================

method_order = [

    "Fixed-IR",
    "ANN-Risk",
    "CURA-Query",
    "CURA-Full",
    "Patience-Inspired",
    "ANN-Oracle",
    "IR-Oracle"
]


paper_df = summary_df[

    summary_df[
        "split"
    ]
    ==
    "MSMARCO_TEST"

].copy()


paper_df[
    "method"
] = pd.Categorical(

    paper_df[
        "method"
    ],

    categories=method_order,

    ordered=True
)


paper_df = paper_df.sort_values(
    "method"
)


paper_df = paper_df[

    [
        "method",
        "mean_ef",
        "IR_violation_rate",
        "ANN_violation_rate",
        "mean_nDCG@10",
        "mean_search_work",
        "work_saving_vs_fixed_%",
        "mean_latency_ms",
        "latency_saving_vs_fixed_%"
    ]

]


paper_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "paper_ready_ablation_table.csv"
    ),

    index=False
)


# ============================================================
# PLOT 1:
# WORK VS DOWNSTREAM VIOLATION
# ============================================================

plt.figure(
    figsize=(9, 6)
)


heldout_summary = summary_df[

    summary_df[
        "split"
    ]
    ==
    "MSMARCO_TEST"

]


for _, row in heldout_summary.iterrows():

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

    0.05,

    linestyle="--",

    label="5% downstream-risk SLA"
)


plt.xlabel(
    "Mean realized HNSW distance computations"
)


plt.ylabel(
    "IR violation rate"
)


plt.title(
    "Held-Out MS MARCO: Efficiency vs Downstream Risk"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "final_efficiency_vs_risk.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT 2:
# METHOD SEARCH WORK
# ============================================================

plot_df = paper_df.copy()


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    plot_df[
        "method"
    ].astype(
        str
    ),

    plot_df[
        "mean_search_work"
    ]
)


plt.ylabel(
    "Mean realized distance computations"
)


plt.title(
    "Held-Out MS MARCO: Search Work by Method"
)


plt.xticks(
    rotation=25
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "final_search_work_comparison.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL CLAIM CHECK
# ============================================================

cura_pairwise = pairwise_df.set_index(
    "comparison"
)


fixed_test = cura_pairwise.loc[
    "Fixed-IR minus CURA-Full"
]


annrisk_test = cura_pairwise.loc[
    "ANN-Risk minus CURA-Full"
]


patience_test = cura_pairwise.loc[
    "Patience-Inspired minus CURA-Full"
]


strong_fixed = (

    fixed_test[
        "work_95CI_low"
    ]
    >
    0

    and

    fixed_test[
        "paired_work_p"
    ]
    <
    0.05
)


strong_patience = (

    patience_test[
        "work_95CI_low"
    ]
    >
    0

    and

    patience_test[
        "paired_work_p"
    ]
    <
    0.05
)


if (
    full[
        "IR_violation_rate"
    ]
    <=
    0.05

    and

    strong_fixed

    and

    strong_patience
):

    FINAL_DECISION = (

        "STRONG FINAL RESULT: CURA meets the held-out "
        "downstream-risk target and achieves statistically "
        "supported search-work reductions relative to both "
        "fixed search and the Patience-inspired stability "
        "baseline. ANN-Risk and query-only ablations quantify "
        "the contributions of the downstream objective and "
        "search-state features."
    )

else:

    FINAL_DECISION = (

        "MIXED FINAL RESULT: report all ablations and "
        "avoid claiming universal superiority."
    )


with open(

    os.path.join(
        RESULTS_DIR,
        "final_research_decision.txt"
    ),

    "w",

    encoding="utf-8"

) as f:

    f.write(
        FINAL_DECISION
        +
        "\n"
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "FINAL HELD-OUT ABLATION TABLE"
)

print(
    "=" * 100
)


print(
    paper_df.to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "PAIRWISE CURA-FULL TESTS"
)

print(
    "=" * 100
)


print(

    pairwise_df[

        [
            "comparison",
            "CURA_work_saving_%",
            "mean_work_difference",
            "work_95CI_low",
            "work_95CI_high",
            "paired_work_p",
            "comparator_IR_violation_rate",
            "CURA_IR_violation_rate",
            "McNemar_exact_p"
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
    "ABLATION INTERPRETATION"
)

print(
    "=" * 100
)


for line in interpretation_lines:

    print(
        line
    )


print(
    "\n"
    +
    "=" * 100
)

print(
    "FINAL RESEARCH DECISION"
)

print(
    "=" * 100
)


print(
    FINAL_DECISION
)


print(
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)


print(
    "\nIMPORTANT CLAIM RULES:"
)

print(
    "1. ANN-Risk is an ANN-target learned baseline, "
    "NOT an official DARTH reproduction."
)

print(
    "2. Patience-Inspired is a stability baseline, "
    "NOT an official Patience reproduction."
)

print(
    "3. Report realized search work including probe/"
    "ladder overhead."
)

print(
    "\nDONE."
)
