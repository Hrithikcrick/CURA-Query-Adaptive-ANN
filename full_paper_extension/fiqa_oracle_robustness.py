"""Verify saved FiQA oracles without retrieval or historical-file changes.

New verification utility, 2026-11-26; this is not a recovered original script.
The declared paired-query bootstrap reproduces the supplied rounded interval;
the original interval's generating script was not recovered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EF_VALUES = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256)
ANN_TARGET = 0.90
IR_TOLERANCE = 0.025
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10000
EXPECTED = {
    "total_queries": 648, "exact_positive_queries": 305,
    "jointly_feasible_exact_positive_queries": 303,
    "mean_ANN_safe_ef": 24.086, "mean_IR_safe_ef": 15.525,
    "mean_ANN_minus_IR_gap": 8.561, "gap_ci_low": 5.498,
    "gap_ci_high": 11.624, "IR_safe_lower": 120,
    "equal_budgets": 166, "IR_safe_higher": 17,
}


def paired_query_ci(gaps, *, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    """Percentile CI, resampling each query's paired ANN-minus-IR difference."""
    gaps = np.asarray(gaps, dtype=float)
    if gaps.ndim != 1 or not len(gaps) or not np.isfinite(gaps).all():
        raise ValueError("Bootstrap requires a nonempty finite vector")
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    # Batching bounds memory without changing the seeded index-draw sequence.
    means = []
    for start in range(0, resamples, 1000):
        indices = rng.integers(0, len(gaps), size=(min(1000, resamples-start), len(gaps)))
        means.append(gaps[indices].mean(axis=1))
    return np.quantile(np.concatenate(means), [0.025, 0.975]).tolist()


def recompute_oracles(source, saved, *, ef_values=EF_VALUES):
    """Scan all tested budgets; never assume that safety is monotone."""
    source, saved = source.copy(), saved.copy()
    required = {"query_id", "efSearch", "exact_nDCG10", "nDCG10", "nDCG_loss",
                "ANN_Recall10", "ANN_unsafe", "IR_unsafe"}
    if not required.issubset(source.columns):
        raise ValueError(f"Missing source columns: {sorted(required-set(source.columns))}")
    if not {"query_id", "ann_safe_ef", "ir_safe_ef"}.issubset(saved.columns):
        raise ValueError("Saved oracle lacks required columns")
    for table in (source, saved):
        if table.query_id.isna().any():
            raise ValueError("Null query IDs")
        table["query_id"] = table.query_id.astype(str)
    if source.duplicated(["query_id", "efSearch"]).any():
        raise ValueError("Duplicate query-budget records")
    if saved.query_id.duplicated().any():
        raise ValueError("Duplicate saved oracle query IDs")
    if set(source.query_id) != set(saved.query_id):
        raise ValueError("Source and saved-oracle query IDs differ")
    for name in required-{"query_id"}:
        source[name] = pd.to_numeric(source[name], errors="raise")
        if not np.isfinite(source[name]).all():
            raise ValueError(f"Nonfinite source values in {name}")
    if not source.groupby("query_id").efSearch.agg(lambda x: set(x)==set(ef_values)).all():
        raise ValueError("Each query must contain the complete tested efSearch grid")
    if not source.groupby("query_id").exact_nDCG10.nunique().eq(1).all():
        raise ValueError("Exact-reference nDCG differs across budgets")
    for name in ("exact_nDCG10", "nDCG10", "ANN_Recall10"):
        if not source[name].between(0, 1).all():
            raise ValueError(f"Out-of-range values in {name}")
    # Original pilot stores signed exact-minus-approximate loss, including gains.
    if not np.allclose(source.nDCG_loss, source.exact_nDCG10-source.nDCG10,
                       rtol=0, atol=1e-12):
        raise ValueError("Saved nDCG loss disagrees with signed utility difference")
    ann_safe = source.ANN_Recall10 >= ANN_TARGET
    ir_safe = source.nDCG_loss <= IR_TOLERANCE
    if not (source.ANN_unsafe.to_numpy() == (~ann_safe).to_numpy()).all():
        raise ValueError("ANN unsafe flags disagree with recall threshold")
    if not (source.IR_unsafe.to_numpy() == (~ir_safe).to_numpy()).all():
        raise ValueError("IR unsafe flags disagree with loss tolerance")
    # The original pilot sorts string IDs; preserve that canonical order for CI.
    query_ids = sorted(source.query_id.unique())
    result = pd.DataFrame(index=pd.Index(query_ids, name="query_id"))
    result["ann_safe_ef"] = source.loc[ann_safe].groupby("query_id").efSearch.min()
    result["ir_safe_ef"] = source.loc[ir_safe].groupby("query_id").efSearch.min()
    result["exact_nDCG10"] = source.groupby("query_id").exact_nDCG10.first()
    comparison = saved.set_index("query_id").reindex(query_ids)
    for name in ("ann_safe_ef", "ir_safe_ef"):
        old = pd.to_numeric(comparison[name], errors="raise")
        new = result[name]
        equal = old.eq(new) | (old.isna() & new.isna())
        if not equal.all():
            raise ValueError(f"Saved oracle mismatch in {name}: {list(equal.index[~equal])}")
    return result


def summarize(source, saved, *, ef_values=EF_VALUES):
    oracle = recompute_oracles(source, saved, ef_values=ef_values)
    positive = oracle.exact_nDCG10 > 0
    joint = oracle.ann_safe_ef.notna() & oracle.ir_safe_ef.notna()
    cohort = oracle.loc[positive & joint]
    gaps = cohort.ann_safe_ef-cohort.ir_safe_ef
    low, high = paired_query_ci(gaps)
    sensitivity = paired_query_ci(gaps, seed=20261126, resamples=50000)
    return {
        "source_rows": len(source), "total_queries": len(oracle),
        "exact_positive_queries": int(positive.sum()),
        "jointly_feasible_exact_positive_queries": len(cohort),
        "saved_oracle_rows_compared": len(oracle), "saved_oracle_mismatches": 0,
        "mean_ANN_safe_ef": float(cohort.ann_safe_ef.mean()),
        "mean_IR_safe_ef": float(cohort.ir_safe_ef.mean()),
        "mean_ANN_minus_IR_gap": float(gaps.mean()),
        "gap_ci_low": low, "gap_ci_high": high,
        "independent_CI_sensitivity": {
            "seed": 20261126, "resamples": 50000,
            "gap_ci_low": sensitivity[0], "gap_ci_high": sensitivity[1],
            "method": "Same paired-query percentile bootstrap with independently declared verification-date seed",
        },
        "IR_safe_lower": int((gaps > 0).sum()),
        "equal_budgets": int((gaps == 0).sum()),
        "IR_safe_higher": int((gaps < 0).sum()),
        "excluded_exact_positive_query_ids": oracle.index[positive & ~joint].tolist(),
    }


def verify(source_path, oracle_path):
    source_path, oracle_path = Path(source_path), Path(oracle_path)
    source = pd.read_csv(source_path, dtype={"query_id": str})
    saved = pd.read_csv(oracle_path, dtype={"query_id": str})
    results = summarize(source, saved)
    matches = {key: round(results[key], 3) == value for key, value in EXPECTED.items()}
    def input_name(path):
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)
    return {
        "status": "PASS" if all(matches.values()) else "FAIL",
        "utility_origin": "New read-only verification utility created 2026-11-26; not a recovered original script",
        "results": results, "supplied_claim_checks": matches,
        "protocol": {
            "ANN_safe": "ANN_Recall10 >= 0.90", "IR_safe": "signed nDCG_loss <= 0.025",
            "candidate_efSearch": list(EF_VALUES),
            "minimum": "Smallest safe tested budget, no monotonicity assumption; infeasible stays missing",
            "cohort": "Exact-reference nDCG@10 > 0 and both minima finite",
            "CI": "Paired-query nonparametric bootstrap; percentile 2.5 and 97.5; numpy default_rng/PCG64",
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_order": "Ascending string FiQA query ID, matching original pilot and saved-oracle row order",
            "CI_provenance": "New explicitly declared verification estimator reproduces supplied rounded CI; original CI-generating script not recovered",
            "analysis_status": "Retrospective verification; no preregistration claim",
        },
        "inputs": {input_name(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source_path, oracle_path)},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "limitations": [
            "Diagnostic outcome-aware oracle comparison, not a deployable policy or CURA controller transfer experiment.",
            "Does not replicate TREC judgment-density false safety; FiQA judgments are unchanged.",
            "The CI resamples the 303 finite paired query gaps conditional on saved retrieval and exact-positive/cohort selection.",
            "Two exact-positive queries have no IR-safe tested budget; no grid-maximum substitution is used.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"beir_fiqa/fiqa_query_budget_results.csv")
    parser.add_argument("--oracle", type=Path, default=ROOT/"beir_fiqa/fiqa_oracle_budgets.csv")
    parser.add_argument("--output", type=Path, help="Optional JSON path; default prints JSON only")
    args = parser.parse_args()
    if args.output and args.output.resolve() in (args.source.resolve(), args.oracle.resolve()):
        parser.error("Output must not overwrite a historical input")
    try:
        report = verify(args.source, args.oracle)
    except (ValueError, OSError, KeyError) as exc:
        report = {"status": "FAIL", "error": str(exc)}
    rendered = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
