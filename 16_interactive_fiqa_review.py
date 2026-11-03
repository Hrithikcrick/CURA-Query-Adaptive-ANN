"""Human-in-the-loop review for 50 FiQA low-budget HNSW failures.

No automatic annotation or test-set tuning. This edits only the existing
manual-review worksheet and a progress summary. Do not rerun Step 15 after
completing annotations: it regenerates the worksheet from scratch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "beir_fiqa" / "manual_failure_review"
WORKSHEET = FOLDER / "fiqa_50_actual_ef10_failures_TO_REVIEW.csv"
STATUS = FOLDER / "human_review_progress.json"

CATEGORIES = {
    "1": "Judged-relevant document missed by approximate search",
    "2": "Relevant document ranked too low by approximate search",
    "3": "Ambiguous query or specialized financial terminology",
    "4": "Near-duplicate or competing passages",
    "5": "Relevance-judgment or evaluation limitation",
    "6": "Approximate index/search-path failure",
    "7": "Other (describe in analysis)",
}
FIELDS = [
    "human_failure_category",
    "human_analysis_of_what_went_wrong",
    "human_evidence_doc_ids_or_ranks",
    "human_suggested_remedy",
    "reviewer",
]


def load_rows():
    if not WORKSHEET.is_file():
        raise FileNotFoundError(
            "Step 15 worksheet not found: " + str(WORKSHEET)
        )
    with WORKSHEET.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    missing = set(FIELDS + ["human_review_complete_YES_NO"]) - set(columns)
    if missing:
        raise ValueError("Missing worksheet columns: " + str(sorted(missing)))
    if len(rows) != 50:
        raise ValueError("Expected exactly 50 source failure cases")
    return rows, columns


def complete(row):
    return (
        row["human_review_complete_YES_NO"].strip().upper() == "YES"
        and all((row.get(col) or "").strip() for col in FIELDS)
    )


def save(rows, columns):
    tmp = WORKSHEET.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(tmp, WORKSHEET)
    except PermissionError as exc:
        raise PermissionError(
            "Close the worksheet in Excel/another program and rerun. "
            "The temporary file preserves the updated data: " + str(tmp)
        ) from exc


def save_progress(rows):
    reviewed = [r for r in rows if complete(r)]
    counts = Counter(r["human_failure_category"] for r in reviewed)
    report = {
        "review_type": "Human-authored analysis of fixed HNSW efSearch=10 failures",
        "important": "NOT failures of deployed CURA-FiQA. Not a model-selection set.",
        "total_selected": len(rows),
        "completed_human_reviews": len(reviewed),
        "remaining": len(rows) - len(reviewed),
        "category_counts_among_completed": dict(sorted(counts.items())),
        "ready_for_final_taxonomy": len(reviewed) == len(rows),
    }
    STATUS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def required_input(prompt, minimum=4):
    while True:
        value = input(prompt).strip()
        if len(value) >= minimum:
            return value
        print("Please enter a substantive answer (at least "
              + str(minimum) + " characters).")


def review(row, reviewer):
    print("\n" + "=" * 78)
    print("CASE " + row["case_id"] + " | FiQA query_id=" + row["query_id"])
    print("QUERY: " + row["query_text"])
    print("Failed policy: " + row["failed_method"])
    print("Exact nDCG@10: " + row["saved_exact_nDCG10"])
    print("HNSW nDCG@10:  " + row["saved_HNSW_nDCG10"])
    print("Loss: " + row["saved_nDCG_loss"])
    print("ANN Recall@10: " + row["saved_ANN_Recall10"])
    print("Reconstruction verified: " + row["reconstruction_matches_saved_metrics"])
    print("\n--- Judged-relevant passages ---")
    print(row["judged_relevant_passages"] or "(not available)")
    print("\n--- Exact-search top 5 ---")
    print(row["exact_top5_passages"] or "(not available)")
    print("\n--- HNSW ef=10 top 5 ---")
    print(row["HNSW_ef10_top5_passages"] or "(not available)")
    print("\nSelect a category only after comparing passages:")
    for key, name in CATEGORIES.items():
        print("  " + key + ". " + name)
    while True:
        choice = input("Category number (1-7; S=skip): ").strip().lower()
        if choice == "s":
            return False
        if choice in CATEGORIES:
            break
        print("Select 1-7, or S to skip.")

    print("\nWrite your OWN evidence-based explanation.")
    print("Do not invent a cause from the metric alone.")
    analysis = required_input("What went wrong? ", 15)
    evidence = required_input("Document IDs/ranks showing the problem: ", 4)
    remedy = required_input("Reasonable remedy or investigation: ", 10)
    confirm = input("Save as a completed HUMAN review? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Not saved.")
        return False

    row["human_failure_category"] = CATEGORIES[choice]
    row["human_analysis_of_what_went_wrong"] = analysis
    row["human_evidence_doc_ids_or_ranks"] = evidence
    row["human_suggested_remedy"] = remedy
    row["reviewer"] = reviewer
    row["human_review_complete_YES_NO"] = "YES"
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum new human reviews this session (default 5)")
    parser.add_argument("--progress", action="store_true",
                        help="Print current progress; no interactive reviewing")
    args = parser.parse_args()
    rows, columns = load_rows()
    report = save_progress(rows)
    print("FiQA manual review: " + str(report["completed_human_reviews"])
          + "/50 completed.")
    if args.progress:
        print(json.dumps(report, indent=2))
        return
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    reviewer = required_input("Reviewer name (for audit trail): ", 2)
    done_this_session = 0
    for row in rows:
        if complete(row):
            continue
        mode = input("\nNext case " + row["case_id"]
                     + ": Enter=review, S=skip, Q=quit: ").strip().lower()
        if mode == "q":
            break
        if mode == "s":
            continue
        if review(row, reviewer):
            save(rows, columns)
            report = save_progress(rows)
            done_this_session += 1
            print("Saved: " + str(report["completed_human_reviews"]) + "/50")
            if done_this_session >= args.limit:
                break
    print("\nSession finished. Human reviews saved: "
          + str(save_progress(rows)["completed_human_reviews"]) + "/50")
    print("IMPORTANT: This worksheet covers fixed HNSW ef=10, NOT CURA.")


if __name__ == "__main__":
    main()
