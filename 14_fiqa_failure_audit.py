"""Diagnose FiQA CURA negative result using frozen held-out artifacts; no retraining."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "beir_fiqa"
CTRL = BASE / "fiqa_cura_controller"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


fixed = {r["query_id"]: r for r in read(CTRL / "heldout_fixed_per_query.csv")}
cura = {r["query_id"]: r for r in read(CTRL / "heldout_cura_per_query.csv")}
sweep = {(r["query_id"], r["efSearch"]): r for r in read(BASE / "fiqa_query_budget_results.csv")}

if fixed.keys() != cura.keys():
    raise ValueError("Fixed and CURA held-out query sets differ")
if len(fixed) != 131:
    raise ValueError(f"Unexpected held-out query count: {len(fixed)}")

rows = []
for qid in sorted(fixed):
    f, c = fixed[qid], cura[qid]
    ef = c["efSearch"]
    raw = sweep.get((qid, ef))
    if raw is None:
        raise ValueError(f"Missing source sweep row for query {qid}, ef={ef}")

    f_work = float(f["realized_work"])
    c_work = float(c["realized_work"])
    raw_work = float(raw["distance_computations"])
    probe = c_work - raw_work
    f_fail, c_fail = int(float(f["IR_unsafe"])), int(float(c["IR_unsafe"]))
    if probe < -1e-6:
        raise ValueError(f"Negative probe work for query {qid}")
    outcome = ("BOTH_FAIL" if f_fail and c_fail else
               "CURA_ONLY_FAIL" if c_fail else
               "FIXED_ONLY_FAIL" if f_fail else "BOTH_SAFE")
    rows.append({
        "query_id": qid, "fixed_ef": int(f["efSearch"]), "cura_ef": int(ef),
        "predicted_risk": c["predicted_risk"],
        "fixed_work": f_work, "cura_raw_search_work": raw_work,
        "cura_probe_overhead": probe, "cura_total_work": c_work,
        "extra_work_vs_fixed": c_work - f_work,
        "fixed_nDCG10": float(f["nDCG10"]),
        "cura_nDCG10": float(c["nDCG10"]),
        "fixed_IR_unsafe": f_fail, "cura_IR_unsafe": c_fail,
        "failure_outcome": outcome,
    })

n = len(rows)
avg = lambda field: sum(float(r[field]) for r in rows) / n
counts = Counter(r["cura_ef"] for r in rows)
outcomes = Counter(r["failure_outcome"] for r in rows)
summary = {
    "heldout_queries": n,
    "fixed_mean_work": avg("fixed_work"),
    "cura_mean_raw_work": avg("cura_raw_search_work"),
    "cura_mean_probe_overhead": avg("cura_probe_overhead"),
    "cura_mean_total_work": avg("cura_total_work"),
    "cura_work_saving_pct": 100 * (avg("fixed_work") - avg("cura_total_work")) / avg("fixed_work"),
    "cura_query_count_ef_above_32": sum(r["cura_ef"] > 32 for r in rows),
    "cura_query_count_with_second_search": sum(r["cura_ef"] > 10 for r in rows),
    "selected_ef_histogram": {str(k): counts[k] for k in sorted(counts)},
    "failure_outcomes": dict(outcomes),
    "top_extra_work_queries": sorted(rows, key=lambda r: r["extra_work_vs_fixed"], reverse=True)[:10],
    "warning": "Diagnostic on frozen held-out data only. Do not tune model or select risk threshold from these labels.",
}
CTRL.mkdir(parents=True, exist_ok=True)
with (CTRL / "fiqa_failure_diagnostics.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(CTRL / "fiqa_failure_diagnostics_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")

print("FiQA CURA FAILURE AUDIT — frozen 131-query held-out set")
for name in ["fixed_mean_work", "cura_mean_raw_work", "cura_mean_probe_overhead", "cura_mean_total_work", "cura_work_saving_pct"]:
    print(f"  {name}: {summary[name]:.2f}")
print(f"  Assigned ef>32: {summary['cura_query_count_ef_above_32']}/{n}")
print(f"  Required a second search: {summary['cura_query_count_with_second_search']}/{n}")
print("  Failure overlap:", dict(outcomes))
print("  ef histogram:", summary["selected_ef_histogram"])
print("  Worst extra-work query IDs:", [r["query_id"] for r in summary["top_extra_work_queries"]])
print("Saved: fiqa_failure_diagnostics.csv and fiqa_failure_diagnostics_summary.json")
print("NO RETRAINING OR HELD-OUT THRESHOLD OPTIMIZATION PERFORMED")
