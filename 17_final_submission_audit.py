"""Read-only COL764 CURA final-submission audit of committed and local artifacts.

Checks saved numeric results, case review progress, and expected report files.
Does NOT train, select policies, inspect held-out labels for tuning, alter CSVs,
or claim human review from assistant-generated notes.
Writes only reports/final_submission_audit.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MS = ROOT / "msmarco_large/msmarco_1000000_q1000/final_ablation_package"
FQ = ROOT / "beir_fiqa/fiqa_cura_controller"
REVIEW = ROOT / "beir_fiqa/manual_failure_review"
OUTPUT = REPORTS / "final_submission_audit.json"
FIELD_NAMES = [
    "human_failure_category",
    "human_analysis_of_what_went_wrong",
    "human_evidence_doc_ids_or_ranks",
    "human_suggested_remedy",
    "reviewer",
]


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(row, name):
    return float(row[name])


def approx(actual, expected, tolerance=0.01):
    return abs(actual - expected) <= tolerance


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    required = {
        "msmarco_ablation": MS / "final_ablation_table.csv",
        "msmarco_pairwise": MS / "pairwise_cura_tests.csv",
        "fiqa_heldout": FQ / "heldout_comparison.csv",
        "fiqa_protocol": FQ / "protocol_and_findings.json",
        "fiqa_failure_diagnostics": FQ / "fiqa_failure_diagnostics_summary.json",
        "review_worksheet": REVIEW / "fiqa_50_actual_ef10_failures_TO_REVIEW.csv",
        "final_paper_draft": REPORTS / "CURA_SIGIR_STYLE_FINAL_DRAFT.md",
        "research_results_report": REPORTS / "CURA_RESULTS_AND_LIMITATIONS.md",
        "model_source": ROOT / "07_cura_controller.py",
        "fiqa_controller_source": ROOT / "13_fiqa_cura_controller.py",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required artifacts missing: " + ", ".join(missing))

    ab = rows(required["msmarco_ablation"])
    test = [r for r in ab if r["split"] == "MSMARCO_TEST"]
    fixed = next(r for r in test if r["method"] == "Fixed-IR")
    cura = next(r for r in test if r["method"] == "CURA-Full")
    fixed_work = number(fixed, "mean_search_work")
    cura_work = number(cura, "mean_search_work")
    saving = 100 * (fixed_work - cura_work) / fixed_work
    risk_ms = number(cura, "IR_violation_rate")
    n_ms = int(cura["queries"])

    pair = next(r for r in rows(required["msmarco_pairwise"])
                if r["comparison"] == "Fixed-IR minus CURA-Full")
    paired_diff = number(pair, "mean_work_difference")
    paired_p = number(pair, "paired_work_p")
    ms_ok = (
        n_ms == 200 and approx(fixed_work, 3431.77)
        and approx(cura_work, 2727.965)
        and approx(saving, 20.5085)
        and approx(risk_ms, 0.01, 1e-6)
        and approx(paired_diff, fixed_work - cura_work, 1e-5)
        and paired_p < 0.05
    )

    fq = rows(required["fiqa_heldout"])
    ff = next(r for r in fq if r["method"] == "Fixed-HNSW")
    fc = next(r for r in fq if r["method"] == "CURA-FiQA")
    fq_fixed_work = number(ff, "mean_work")
    fq_cura_work = number(fc, "mean_work")
    fq_saving = 100 * (fq_fixed_work - fq_cura_work) / fq_fixed_work
    fq_ok = (
        int(ff["query_count"]) == 131
        and int(fc["query_count"]) == 131
        and approx(fq_fixed_work, 1097.08397)
        and approx(fq_cura_work, 1922.87023)
        and approx(fq_saving, -75.2710)
        and approx(number(ff, "IR_violation_rate"), 4 / 131, 1e-8)
        and approx(number(fc, "IR_violation_rate"), 4 / 131, 1e-8)
    )

    manual_rows = rows(required["review_worksheet"])
    valid_reviews = [
        r for r in manual_rows
        if (r.get("human_review_complete_YES_NO") or "").strip().upper() == "YES"
        and all((r.get(field) or "").strip() for field in FIELD_NAMES)
    ]
    inconsistent = [
        str(r.get("case_id", "?")) for r in manual_rows
        if (r.get("human_review_complete_YES_NO") or "").strip().upper() == "YES"
        and r not in valid_reviews
    ]
    reconstruction_match = sum(
        (r.get("reconstruction_matches_saved_metrics") or "").strip() == "YES"
        for r in manual_rows
    )

    REPORTS.mkdir(exist_ok=True)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "COL764 CURA",
        "git_commit": get_git_sha(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "source": "Saved artifacts; no fresh experiment rerun",
        "msmarco": {
            "queries": n_ms,
            "fixed_work": fixed_work,
            "cura_work": cura_work,
            "work_saving_percent": saving,
            "cura_ir_violation_rate": risk_ms,
            "paired_work_difference": paired_diff,
            "paired_randomization_p": paired_p,
            "saved_numbers_consistent": ms_ok,
        },
        "fiqa": {
            "heldout_queries": int(fc["query_count"]),
            "fixed_work": fq_fixed_work,
            "cura_work": fq_cura_work,
            "work_saving_percent": fq_saving,
            "fixed_violation_rate": number(ff, "IR_violation_rate"),
            "cura_violation_rate": number(fc, "IR_violation_rate"),
            "saved_numbers_consistent": fq_ok,
            "zero_shot_msmarco_transfer": False,
            "official_test_untouched": False,
        },
        "manual_review": {
            "selected_fixed_ef10_cases": len(manual_rows),
            "reconstruction_matching_saved_metrics": reconstruction_match,
            "human_reviews_with_all_required_fields": len(valid_reviews),
            "human_reviews_remaining": 50 - len(valid_reviews),
            "YES_rows_missing_required_fields": inconsistent,
            "assistant_drafts_count_as_human_reviews": False,
        },
        "files": {name: {"path": path.relative_to(ROOT).as_posix(),
                          "sha256": sha256(path)}
                  for name, path in required.items()},
        "submission_ready": (
            ms_ok and fq_ok and len(manual_rows) == 50
            and reconstruction_match == 50
            and len(valid_reviews) == 50 and not inconsistent
        ),
        "note": (
            "submission_ready is a partial artifact check only: even True "
            "does not verify independent clean-environment reproducibility, "
            "official recent baselines, final manuscript references, "
            "AI-use approval, or viva readiness."
        ),
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("CURA FINAL SUBMISSION EVIDENCE AUDIT")
    print("MS MARCO saved-result checks:", "PASS" if ms_ok else "FAIL")
    print("FiQA saved-result checks:", "PASS" if fq_ok else "FAIL")
    print("MS MARCO realized work saving: %.2f%%" % saving)
    print("FiQA realized work saving: %.2f%%" % fq_saving)
    print("Human reviews with required fields: %d/50" % len(valid_reviews))
    print("Reconstructed review cases matching saved metrics: %d/50"
          % reconstruction_match)
    if inconsistent:
        print("Incomplete rows marked YES:", ", ".join(inconsistent))
    print("Report:", OUTPUT)
    print("NOTE: AI-generated case drafts are NOT counted as human reviews.")
    print("NOTE: This audit does NOT rerun training or indexing.")
    if not (ms_ok and fq_ok and len(manual_rows) == 50 and reconstruction_match == 50):
        raise SystemExit("AUDIT ERROR: Saved evidence inconsistent or missing")


if __name__ == "__main__":
    main()
