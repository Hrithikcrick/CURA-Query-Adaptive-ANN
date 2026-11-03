import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "msmarco_large/msmarco_1000000_q1000"

def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def get(rows, method, split="MSMARCO_TEST"):
    return next(r for r in rows
                if r["method"] == method and r["split"] == split)

print("=" * 55)
print("CURA RESEARCH RESULTS AUDIT")
print("=" * 55)

# 1. Read final ablation results
rows = read_csv(BASE / "final_ablation_package/final_ablation_table.csv")

fixed = get(rows, "Fixed-IR")
cura = get(rows, "CURA-Full")
ann = get(rows, "ANN-Risk")

fixed_work = float(fixed["mean_search_work"])
cura_work = float(cura["mean_search_work"])
ann_work = float(ann["mean_search_work"])

saving = 100 * (fixed_work - cura_work) / fixed_work
ann_saving = 100 * (ann_work - cura_work) / ann_work
risk = float(cura["IR_violation_rate"])

print("\n1. MAIN RESULTS")
print("Fixed search work:", fixed_work)
print("CURA search work:", cura_work)
print("CURA work saving: %.2f%%" % saving)
print("Saving vs ANN-Risk: %.2f%%" % ann_saving)
print("CURA IR violation rate: %.2f%%" % (risk * 100))

assert abs(saving - 20.5085) < 0.01
assert risk <= 0.05
assert int(cura["queries"]) == 200

# 2. Check whether probe overhead was counted
raw = read_csv(BASE / "cura_controller/final_method_comparison.csv")
raw_cura = get(raw, "CURA-Full")
raw_work = float(raw_cura["mean_search_work"])

print("\n2. PROBE OVERHEAD")
print("Search without full overhead:", raw_work)
print("Total realized search work:", cura_work)
print("Additional work accounted:", round(cura_work - raw_work, 2))

assert cura_work >= raw_work

# 3. Statistical tests
tests = read_csv(BASE / "final_ablation_package/pairwise_cura_tests.csv")
test = next(r for r in tests
            if r["comparison"] == "Fixed-IR minus CURA-Full")

print("\n3. STATISTICAL SIGNIFICANCE")
print("Paired work reduction:", test["mean_work_difference"])
print("95% CI:", test["work_95CI_low"], "to", test["work_95CI_high"])
print("p-value:", test["paired_work_p"])

# 4. Manual evaluation audit
manual = read_csv(BASE / "cura_controller/manual_review_50.csv")
failures = sum(int(float(r["IR_unsafe"])) for r in manual)

print("\n4. MANUAL REVIEW")
print("Reviewed cases:", len(manual))
print("Actual IR violations:", failures)
print("Other diagnostic cases:", len(manual) - failures)

assert len(manual) == 50

print("\n" + "=" * 55)
print("AUDIT PASSED: SAVED RESULTS ARE CONSISTENT")
print("=" * 55)
print("NOTE: This checks saved files, not a fresh experiment run.")
