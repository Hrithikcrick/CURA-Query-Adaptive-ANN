# ============================================================
# 10_manual_taxonomy_summary.py
#
# Summarize the 50-query manual review into:
# - category counts
# - actual IR failures by category
# - MS MARCO vs TREC-DL breakdown
# - paper-ready table
# - paper-ready textual summary
# ============================================================

import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "msmarco_large",
    "msmarco_1000000_q1000",
    "cura_controller",
    "manual_review_50.csv"
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "msmarco_large",
    "msmarco_1000000_q1000",
    "manual_error_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if not os.path.exists(INPUT_FILE):

    print("ERROR: manual_review_50.csv not found.")
    print(INPUT_FILE)
    sys.exit(1)

df = pd.read_csv(
    INPUT_FILE
)

required = [
    "evaluation_split",
    "query_key",
    "query_text",
    "efSearch",
    "CURA_full_risk",
    "IR_unsafe",
    "nDCG@10",
    "nDCG_loss",
    "ANN_Recall@10",
    "review_reason",
    "manual_error_category",
    "manual_notes"
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:

    print("ERROR: Missing columns:")
    print(missing)
    sys.exit(1)


# ============================================================
# CLEAN MANUAL FIELDS
# ============================================================

df["manual_error_category"] = (
    df["manual_error_category"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["manual_notes"] = (
    df["manual_notes"]
    .fillna("")
    .astype(str)
    .str.strip()
)

filled_category = (
    df["manual_error_category"] != ""
)

filled_notes = (
    df["manual_notes"] != ""
)

print("=" * 100)
print("MANUAL REVIEW COMPLETENESS")
print("=" * 100)

print("Rows:", len(df))
print(
    "Categories filled:",
    int(filled_category.sum())
)
print(
    "Notes filled:",
    int(filled_notes.sum())
)

if filled_category.sum() < 50:

    print()
    print(
        "STOP: not all 50 manual_error_category cells are filled."
    )

    print(
        "Please complete the CSV first."
    )

    sys.exit(1)


# ============================================================
# BASIC COUNTS
# ============================================================

category_counts = (
    df
    .groupby(
        "manual_error_category"
    )
    .agg(
        queries=(
            "query_key",
            "count"
        ),
        actual_IR_failures=(
            "IR_unsafe",
            "sum"
        ),
        mean_CURA_risk=(
            "CURA_full_risk",
            "mean"
        ),
        mean_nDCG_loss=(
            "nDCG_loss",
            "mean"
        ),
        mean_ANN_recall=(
            "ANN_Recall@10",
            "mean"
        ),
        mean_ef=(
            "efSearch",
            "mean"
        )
    )
    .reset_index()
)

category_counts[
    "share_of_review_%"
] = (
    100.0
    *
    category_counts[
        "queries"
    ]
    /
    len(df)
)

category_counts[
    "failure_rate_within_category_%"
] = np.where(
    category_counts[
        "queries"
    ] > 0,
    100.0
    *
    category_counts[
        "actual_IR_failures"
    ]
    /
    category_counts[
        "queries"
    ],
    0.0
)

category_counts = category_counts.sort_values(
    [
        "actual_IR_failures",
        "queries"
    ],
    ascending=[
        False,
        False
    ]
)

category_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "manual_taxonomy_summary.csv"
    ),
    index=False
)


# ============================================================
# ACTUAL FAILURE SUBSET
# ============================================================

failures = df[
    df[
        "IR_unsafe"
    ] == 1
].copy()

failure_category_counts = (
    failures[
        "manual_error_category"
    ]
    .value_counts()
    .rename_axis(
        "manual_error_category"
    )
    .reset_index(
        name="actual_failure_count"
    )
)

if len(failures) > 0:

    failure_category_counts[
        "share_of_actual_failures_%"
    ] = (
        100.0
        *
        failure_category_counts[
            "actual_failure_count"
        ]
        /
        len(failures)
    )

else:

    failure_category_counts[
        "share_of_actual_failures_%"
    ] = []

failure_category_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "actual_failure_taxonomy.csv"
    ),
    index=False
)


# ============================================================
# DOMAIN BREAKDOWN
# ============================================================

domain_category = (
    df
    .groupby(
        [
            "evaluation_split",
            "manual_error_category"
        ]
    )
    .agg(
        queries=(
            "query_key",
            "count"
        ),
        actual_IR_failures=(
            "IR_unsafe",
            "sum"
        ),
        mean_nDCG_loss=(
            "nDCG_loss",
            "mean"
        ),
        mean_CURA_risk=(
            "CURA_full_risk",
            "mean"
        )
    )
    .reset_index()
)

domain_category.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "taxonomy_by_domain.csv"
    ),
    index=False
)


# ============================================================
# RISK GROUPS
# ============================================================

def risk_band(x):

    if x < 0.01:
        return "VERY_LOW_<1%"

    if x < 0.05:
        return "LOW_1-5%"

    if x < 0.10:
        return "MEDIUM_5-10%"

    return "HIGH_>=10%"


df[
    "predicted_risk_band"
] = df[
    "CURA_full_risk"
].apply(
    risk_band
)

risk_summary = (
    df
    .groupby(
        "predicted_risk_band"
    )
    .agg(
        queries=(
            "query_key",
            "count"
        ),
        actual_IR_failures=(
            "IR_unsafe",
            "sum"
        ),
        mean_predicted_risk=(
            "CURA_full_risk",
            "mean"
        ),
        mean_nDCG_loss=(
            "nDCG_loss",
            "mean"
        )
    )
    .reset_index()
)

risk_summary[
    "observed_failure_rate"
] = (
    risk_summary[
        "actual_IR_failures"
    ]
    /
    risk_summary[
        "queries"
    ]
)

risk_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "manual_risk_band_summary.csv"
    ),
    index=False
)


# ============================================================
# PAPER-READY TABLE
# ============================================================

paper_table = category_counts[
    [
        "manual_error_category",
        "queries",
        "share_of_review_%",
        "actual_IR_failures",
        "failure_rate_within_category_%",
        "mean_CURA_risk",
        "mean_nDCG_loss",
        "mean_ANN_recall"
    ]
].copy()

paper_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "paper_ready_error_taxonomy.csv"
    ),
    index=False
)


# ============================================================
# PAPER-READY TEXT
# ============================================================

total = len(df)

actual_failures = int(
    df[
        "IR_unsafe"
    ].sum()
)

msmarco_count = int(
    (
        df[
            "evaluation_split"
        ]
        ==
        "MSMARCO_TEST"
    ).sum()
)

trec_count = int(
    (
        df[
            "evaluation_split"
        ]
        ==
        "TREC_DL_2019_OOD"
    ).sum()
)

top_categories = (
    category_counts
    .head(3)
)

category_text = []

for _, row in top_categories.iterrows():

    category_text.append(
        (
            f"{row['manual_error_category']} "
            f"({int(row['queries'])}/{total}, "
            f"{row['share_of_review_%']:.1f}%)"
        )
    )

top_category_sentence = ", ".join(
    category_text
)

if actual_failures > 0:

    top_failure_categories = (
        failure_category_counts
        .head(3)
    )

    failure_text = []

    for _, row in top_failure_categories.iterrows():

        failure_text.append(
            (
                f"{row['manual_error_category']} "
                f"({int(row['actual_failure_count'])})"
            )
        )

    failure_sentence = ", ".join(
        failure_text
    )

else:

    failure_sentence = (
        "No actual IR-unsafe cases were present "
        "in the manually reviewed subset."
    )


paragraph = (
    f"We manually reviewed {total} high-priority queries "
    f"selected from CURA's held-out evaluation, including "
    f"{msmarco_count} MS MARCO test queries and "
    f"{trec_count} TREC-DL-2019 out-of-distribution queries. "
    f"The most common manually assigned categories were "
    f"{top_category_sentence}. "
    f"Among the reviewed cases, {actual_failures} queries "
    f"were actual downstream IR violations under the "
    f"predefined nDCG-loss criterion. "
    f"The dominant categories among these actual failures were "
    f"{failure_sentence}. "
    f"This analysis indicates that CURA's residual errors are "
    f"not uniformly distributed across queries, and helps "
    f"identify the query types for which downstream-risk "
    f"estimation remains most difficult."
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "paper_ready_error_analysis.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        paragraph
        +
        "\n"
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print()
print("=" * 100)
print("MANUAL ERROR TAXONOMY")
print("=" * 100)

print(
    category_counts[
        [
            "manual_error_category",
            "queries",
            "share_of_review_%",
            "actual_IR_failures",
            "failure_rate_within_category_%",
            "mean_CURA_risk",
            "mean_nDCG_loss",
            "mean_ANN_recall"
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 100)
print("ACTUAL CURA IR FAILURES")
print("=" * 100)

print(
    "Total actual IR failures in review set:",
    actual_failures
)

if len(
    failure_category_counts
) > 0:

    print(
        failure_category_counts.to_string(
            index=False
        )
    )

print()
print("=" * 100)
print("PAPER-READY ERROR ANALYSIS")
print("=" * 100)

print(
    paragraph
)

print()
print("Results saved to:")
print(
    OUTPUT_DIR
)

print()
print("DONE.")
