import os
import pandas as pd

BASE = r"C:\Users\HRITHIK SHARMA\Downloads\IR_PROJECT\PILOT"

CSV = os.path.join(
    BASE,
    "msmarco_large",
    "msmarco_1000000_q1000",
    "cura_controller",
    "manual_review_50.csv"
)

df = pd.read_csv(CSV)

df["manual_error_category"] = (
    df["manual_error_category"]
    .fillna("")
    .astype(str)
)

df["manual_notes"] = (
    df["manual_notes"]
    .fillna("")
    .astype(str)
)

categories = {
    "1": "AMBIGUOUS_QUERY",
    "2": "NEAR_TIE_TOP_RESULTS",
    "3": "RARE_ENTITY",
    "4": "LONG_OR_COMPLEX_QUERY",
    "5": "EARLY_STOP_TOO_AGGRESSIVE",
    "6": "ANN_MISMATCH_BUT_IR_SAFE",
    "7": "IR_FAILURE_DESPITE_HIGH_ANN_RECALL",
    "8": "OOD_DISTRIBUTION_SHIFT",
    "9": "OTHER",
}

print("=" * 100)
print("MANUAL 50-QUERY REVIEW")
print("=" * 100)

print("""
CATEGORY GUIDE

1 = AMBIGUOUS_QUERY
    Query can reasonably mean more than one thing.

2 = NEAR_TIE_TOP_RESULTS
    Candidate results are extremely close / unstable.

3 = RARE_ENTITY
    Obscure person, place, product, technical term, etc.

4 = LONG_OR_COMPLEX_QUERY
    Long, multi-part, or unusually complicated query.

5 = EARLY_STOP_TOO_AGGRESSIVE
    CURA stopped too early and downstream quality suffered.

6 = ANN_MISMATCH_BUT_IR_SAFE
    ANN recall is weak, but downstream nDCG is still safe.

7 = IR_FAILURE_DESPITE_HIGH_ANN_RECALL
    ANN recall looks good, yet downstream ranking failed.

8 = OOD_DISTRIBUTION_SHIFT
    TREC/OOD behavior appears different from training domain.

9 = OTHER
    None of the above.
""")

for i in range(len(df)):

    current = str(df.loc[i, "manual_error_category"]).strip()

    if current:
        continue

    row = df.loc[i]

    print("\n" + "=" * 100)
    print(f"QUERY {i + 1} / {len(df)}")
    print("=" * 100)

    print("Split       :", row.get("evaluation_split", ""))
    print("Query ID    :", row.get("query_key", ""))
    print("Query       :", row.get("query_text", ""))
    print()
    print("efSearch    :", row.get("efSearch", ""))
    print("CURA risk   :", row.get("CURA_full_risk", ""))
    print("IR unsafe   :", row.get("IR_unsafe", ""))
    print("nDCG@10     :", row.get("nDCG@10", ""))
    print("nDCG loss   :", row.get("nDCG_loss", ""))
    print("ANN Recall  :", row.get("ANN_Recall@10", ""))
    print("Review type :", row.get("review_reason", ""))

    print("""
Choose:
1 AMBIGUOUS_QUERY
2 NEAR_TIE_TOP_RESULTS
3 RARE_ENTITY
4 LONG_OR_COMPLEX_QUERY
5 EARLY_STOP_TOO_AGGRESSIVE
6 ANN_MISMATCH_BUT_IR_SAFE
7 IR_FAILURE_DESPITE_HIGH_ANN_RECALL
8 OOD_DISTRIBUTION_SHIFT
9 OTHER
""")

    while True:
        choice = input("Category [1-9]: ").strip()

        if choice in categories:
            break

        print("Please enter a number from 1 to 9.")

    category = categories[choice]

    note = input(
        "Short manual note (press Enter to use default): "
    ).strip()

    if not note:

        if category == "EARLY_STOP_TOO_AGGRESSIVE":
            note = "CURA stopped before downstream ranking quality was sufficiently stable."

        elif category == "ANN_MISMATCH_BUT_IR_SAFE":
            note = "ANN recall was below target, but downstream ranking quality remained acceptable."

        elif category == "IR_FAILURE_DESPITE_HIGH_ANN_RECALL":
            note = "High ANN recall did not prevent a downstream ranking-quality failure."

        elif category == "AMBIGUOUS_QUERY":
            note = "The query admits multiple plausible interpretations."

        elif category == "NEAR_TIE_TOP_RESULTS":
            note = "Top candidates appear closely matched, making the ranking sensitive to search effort."

        elif category == "RARE_ENTITY":
            note = "The query contains an uncommon or specialized entity."

        elif category == "LONG_OR_COMPLEX_QUERY":
            note = "The query is structurally complex or contains multiple information needs."

        elif category == "OOD_DISTRIBUTION_SHIFT":
            note = "The query comes from the out-of-distribution TREC-DL evaluation regime."

        else:
            note = "The case does not clearly fit the predefined error categories."

    df.loc[i, "manual_error_category"] = category
    df.loc[i, "manual_notes"] = note

    # Save after EVERY query.
    df.to_csv(
        CSV,
        index=False
    )

    print()
    print("SAVED:", category)

print("\n" + "=" * 100)
print("MANUAL REVIEW COMPLETE")
print("=" * 100)

print(
    "Categories filled:",
    (
        df["manual_error_category"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
)

print("Saved to:")
print(CSV)
