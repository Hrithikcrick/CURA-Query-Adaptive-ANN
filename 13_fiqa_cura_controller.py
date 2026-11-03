"""FiQA query-disjoint CURA-style controller, using previously saved HNSW sweep.

This is an independently trained FiQA controller, NOT zero-shot transfer of the
MS MARCO model. It uses only deployable query-text and ef=10 probe features.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "beir_fiqa" / "fiqa_query_budget_results.csv"
OUTPUT = ROOT / "beir_fiqa" / "fiqa_cura_controller"
DATA_DIR = ROOT / "data" / "beir"
SEED = 42
TARGET = 0.05
PROBE_EF = 10
FEATURES = ["query_chars", "query_words", "query_digit", "query_question",
            "probe_work", "probe_latency_ms", "candidate_ef", "candidate_log_ef"]


def wilson_upper(flags, z=1.959963984540054):
    a = np.asarray(flags, dtype=float)
    n = len(a)
    if not n:
        return 1.0
    p = float(a.mean())
    return (p + z*z/(2*n) + z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)


def read_queries():
    candidates = list(DATA_DIR.glob("**/queries.jsonl"))
    if not candidates:
        raise FileNotFoundError("Cannot locate BEIR queries.jsonl in data/beir. Run 12_beir_fiqa_pilot.py first.")
    # Prefer the FiQA dataset folder.
    candidates.sort(key=lambda p: ("fiqa" not in str(p).lower(), len(str(p))))
    result = {}
    with open(candidates[0], encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            result[str(item["_id"])] = str(item.get("text", ""))
    return result


def selected_rows(table, query_ids, prediction_col=None, threshold=None, fixed_ef=None):
    local = table.loc[table.query_id.isin(query_ids)]
    if fixed_ef is not None:
        sel = local.loc[local.efSearch == fixed_ef].copy()
        sel["predicted_risk"] = np.nan
    else:
        good = (local.loc[local[prediction_col] <= threshold]
                .sort_values(["query_id", "efSearch"])
                .drop_duplicates("query_id", keep="first"))
        remaining = (local.loc[~local.query_id.isin(good.query_id)]
                     .sort_values(["query_id", "efSearch"])
                     .drop_duplicates("query_id", keep="last"))
        sel = pd.concat([good, remaining], ignore_index=True).copy()
        sel["predicted_risk"] = sel[prediction_col]
    # Stable query order is vital for paired comparisons.
    sel = sel.set_index("query_id").loc[list(query_ids)].reset_index()
    overhead = (sel.efSearch > PROBE_EF).astype(float) if fixed_ef is None else 0.
    sel["realized_work"] = sel.distance_computations + overhead * sel.probe_work
    sel["realized_latency_ms"] = sel.latency_ms + overhead * sel.probe_latency_ms
    return sel


def metrics(rows):
    return {"query_count": len(rows), "mean_ef": float(rows.efSearch.mean()),
            "mean_work": float(rows.realized_work.mean()),
            "mean_latency_ms": float(rows.realized_latency_ms.mean()),
            "p95_latency_ms": float(rows.realized_latency_ms.quantile(.95)),
            "IR_violation_rate": float(rows.IR_unsafe.mean()),
            "Wilson_95_upper": wilson_upper(rows.IR_unsafe),
            "mean_nDCG10": float(rows.nDCG10.mean()),
            "mean_ANN_Recall10": float(rows.ANN_Recall10.mean())}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}. First run 12_beir_fiqa_pilot.py")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, dtype={"query_id": str})
    required = {"query_id", "efSearch", "IR_unsafe", "distance_computations",
                "latency_ms", "nDCG10", "ANN_Recall10"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df.duplicated(["query_id", "efSearch"]).any():
        raise ValueError("Duplicate query/budget results found")
    budgets = sorted(df.efSearch.unique())
    if budgets[0] != PROBE_EF or df.groupby("query_id").size().nunique() != 1:
        raise ValueError("Expected complete budget grid with ef=10 probe for every query")
    if df.distance_computations.isna().any():
        raise RuntimeError("FAISS distance counters are absent; cannot compare realized work")

    queries = read_queries()
    ids = sorted(df.query_id.unique())
    missing = [q for q in ids if q not in queries]
    if missing:
        raise ValueError(f"FiQA query text missing for {len(missing)} IDs, first {missing[:3]}")
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(ids)
    ntrain = int(.6*len(ids))
    ncal = int(.2*len(ids))
    train_ids = list(shuffled[:ntrain])
    cal_ids = list(shuffled[ntrain:ntrain+ncal])
    test_ids = list(shuffled[ntrain+ncal:])
    assert len(set(train_ids) & set(cal_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(cal_ids) & set(test_ids)) == 0

    probe = df.loc[df.efSearch == PROBE_EF, ["query_id", "distance_computations", "latency_ms"]]
    probe = probe.rename(columns={"distance_computations": "probe_work", "latency_ms": "probe_latency_ms"})
    df = df.merge(probe, on="query_id", how="left", validate="many_to_one")
    df["query_chars"] = df.query_id.map(lambda q: len(queries[q]))
    df["query_words"] = df.query_id.map(lambda q: len(queries[q].split()))
    df["query_digit"] = df.query_id.map(lambda q: int(any(c.isdigit() for c in queries[q])))
    df["query_question"] = df.query_id.map(lambda q: int("?" in queries[q]))
    df["candidate_ef"] = df.efSearch.astype(float)
    df["candidate_log_ef"] = np.log(df.candidate_ef)
    tr = df[df.query_id.isin(train_ids)]
    ca = df[df.query_id.isin(cal_ids)]
    clf = HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15, min_samples_leaf=15,
                                         l2_regularization=1., random_state=SEED)
    y = tr.IR_unsafe.astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("Training labels have only one class")
    # Weight class imbalance using only TRAIN labels.
    counts = np.bincount(y, minlength=2)
    weights = np.where(y == 1, len(y)/(2*counts[1]), len(y)/(2*counts[0]))
    clf.fit(tr[FEATURES].to_numpy(), y, sample_weight=weights)
    cal_scores = clf.predict_proba(ca[FEATURES].to_numpy())[:,1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_scores, ca.IR_unsafe.astype(float).to_numpy())
    df["risk"] = iso.predict(clf.predict_proba(df[FEATURES].to_numpy())[:,1])

    # Fix both operating points using CALIBRATION queries only.
    options = []
    for ef in budgets:
        sample = selected_rows(df, cal_ids, fixed_ef=int(ef))
        options.append({"method":"Fixed-HNSW", "parameter":float(ef), **metrics(sample)})
    thresholds = sorted(set([0., 1.] + list(np.linspace(0.001,0.50,150)) +
                            list(df.loc[df.query_id.isin(cal_ids), "risk"].unique())))
    for threshold in thresholds:
        sample = selected_rows(df, cal_ids, prediction_col="risk", threshold=float(threshold))
        options.append({"method":"CURA-FiQA", "parameter":float(threshold), **metrics(sample)})
    sweep = pd.DataFrame(options)
    sweep.to_csv(OUTPUT / "calibration_operating_points.csv", index=False)
    chosen = {}
    for method in ["Fixed-HNSW", "CURA-FiQA"]:
        candidates = sweep[(sweep.method == method) & (sweep.Wilson_95_upper <= TARGET)]
        if len(candidates):
            winner = candidates.sort_values(["mean_work", "IR_violation_rate"]).iloc[0]
            chosen[method] = {"parameter": float(winner.parameter), "met_calibration_Wilson":True}
        else:
            # Transparent fallback: no observed calibration-feasible operating point.
            winner = sweep[sweep.method == method].sort_values(["Wilson_95_upper", "mean_work"]).iloc[0]
            chosen[method] = {"parameter": float(winner.parameter), "met_calibration_Wilson":False}

    fixed = selected_rows(df, test_ids, fixed_ef=int(chosen["Fixed-HNSW"]["parameter"]))
    cura = selected_rows(df, test_ids, prediction_col="risk", threshold=chosen["CURA-FiQA"]["parameter"])
    result = pd.DataFrame([{ "method":"Fixed-HNSW", "calibration_parameter":chosen["Fixed-HNSW"]["parameter"],
                            "calibration_met_wilson_target":chosen["Fixed-HNSW"]["met_calibration_Wilson"], **metrics(fixed)},
                           {"method":"CURA-FiQA", "calibration_parameter":chosen["CURA-FiQA"]["parameter"],
                            "calibration_met_wilson_target":chosen["CURA-FiQA"]["met_calibration_Wilson"], **metrics(cura)}])
    result.to_csv(OUTPUT / "heldout_comparison.csv", index=False)
    cols = ["query_id", "efSearch", "predicted_risk", "nDCG10", "IR_unsafe", "ANN_Recall10", "realized_work", "realized_latency_ms"]
    fixed[cols].to_csv(OUTPUT / "heldout_fixed_per_query.csv", index=False)
    cura[cols].to_csv(OUTPUT / "heldout_cura_per_query.csv", index=False)

    work_fixed = float(fixed.realized_work.mean())
    work_cura = float(cura.realized_work.mean())
    saving = 100*(work_fixed-work_cura)/work_fixed
    # Paired bootstrap uncertainty on realized computation differences.
    diff = (fixed.set_index("query_id").loc[test_ids, "realized_work"].to_numpy() -
            cura.set_index("query_id").loc[test_ids, "realized_work"].to_numpy())
    idx = np.random.default_rng(SEED+1).integers(0,len(diff),size=(3000,len(diff)))
    ci = np.quantile(diff[idx].mean(axis=1),[.025,.975]).tolist()
    report = {"experiment":"FiQA within-domain query-disjoint CURA-style risk controller",
              "NOT_zero_shot_MSMARCO_transfer":True,
              "limitations":"Uses query text and ef=10 probe work/latency only, not MS MARCO CURA's full partial-score features. Calibration and policy selection reuse the calibration query split.",
              "seed":SEED,"split":{"train_queries":len(train_ids),"cal_queries":len(cal_ids),"heldout_queries":len(test_ids)},
              "target_risk":TARGET,"calibration_choice":chosen,
              "heldout_work_saving_percent":saving,"paired_work_difference_CI_95":ci,
              "heldout_summary":result.to_dict(orient="records")}
    (OUTPUT/"protocol_and_findings.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\nFiQA within-domain CURA experiment (not zero-shot MS MARCO transfer)")
    print("Train / calibration / heldout:",len(train_ids),len(cal_ids),len(test_ids))
    print(result.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(f"CURA work saving vs tuned fixed: {saving:+.2f}%")
    print(f"Paired mean work difference 95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")
    print("Saved:",OUTPUT)
    if not all(v["met_calibration_Wilson"] for v in chosen.values()):
        print("WARNING: At least one policy could not satisfy the 5% calibration Wilson bound.")
    if result.loc[result.method=="CURA-FiQA","IR_violation_rate"].iloc[0] > TARGET:
        print("WARNING: CURA exceeded the target on heldout data; report this failure honestly.")

if __name__ == "__main__":
    main()
