"""Seed-level uncertainty for a future *scored* TREC judgment-density study.

This module never reads rankings, prepares judgments, or writes result files.
Prepared seeds are not observations. Confidence intervals describe variation
over independent thinning seeds conditional on the same fixed query set and
rankings; they do not estimate uncertainty over queries or a query population.
FULL is a fixed reference, so its displayed intervals are degenerate, not query
confidence intervals. P5/P6 compare each seed-level mean with that reference.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd


CONDITIONS = (
    "FULL", "50_PERCENT_POSITIVES", "25_PERCENT_POSITIVES",
    "ONE_POSITIVE_PER_QUERY",
)
METRIC_COLUMNS = (
    "exact_ndcg_zero_query_fraction", "mean_exact_ndcg_at_10",
    "mean_min_IR_safe_ef", "mean_min_ANN_safe_ef",
    "mean_ANN_minus_IR_budget_gap", "relative_IR_safe_budget_reduction",
    "IR_allows_lower_rate", "same_budget_rate", "IR_requires_higher_rate",
    "ANN_Recall_at_10", "CURA_IR_violation_rate",
    "CURA_mean_distance_computations",
)
CORE_METRIC_COLUMNS = tuple(name for name in METRIC_COLUMNS if not name.startswith("CURA_"))
QREL_COLUMNS = (
    "retained_positive_judgments", "retained_positive_fraction",
    "preserved_nonpositive_judgments",
)
PRIMARY_COLUMNS = (
    "primary_test_id", "cohort", "comparison", "metric", "units",
    "estimator", "queries", "nonzero_pairs", "mean_difference",
    "ci95_low", "ci95_high", "p_value", "test", "test_status",
    "analysis_role", "random_seed", "bootstrap_resamples",
    "relative_reduction_pct", "relative_reduction_ci95_low_pct",
    "relative_reduction_ci95_high_pct", "IR_allows_lower_rate",
    "interpretation", "cura_only_failures", "fixed_only_failures",
    "thinning_seeds_prepared", "thinning_seeds_evaluated", "multiplicity",
)
CI_SCOPE = "thinning-seed uncertainty conditional on the fixed query set and rankings"
REQUIRED_THINNING_SEEDS = 50


def _completed(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Invalid evaluation_completed value: {value!r}")


def _frame(per_seed: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "seed", "query_count", "evaluation_completed"}
    missing = required.difference(per_seed.columns)
    if missing:
        raise ValueError(f"Missing per-seed fields: {sorted(missing)}")
    frame = per_seed.copy()
    if frame.condition.isna().any() or not set(frame.condition).issubset(CONDITIONS):
        raise ValueError("Unknown or missing density condition")
    frame["seed"] = pd.to_numeric(frame.seed, errors="raise")
    if frame.seed.isna().any() or not np.isfinite(frame.seed).all() or (frame.seed % 1 != 0).any():
        raise ValueError("Seed IDs must be finite integers")
    frame["seed"] = frame.seed.astype("int64")
    if frame.duplicated(["condition", "seed"]).any():
        raise ValueError("Duplicate condition/seed rows would pseudoreplicate observations")
    full = frame[frame.condition == "FULL"]
    if len(full) > 1 or (len(full) == 1 and full.iloc[0].seed != -1):
        raise ValueError("FULL must contain at most one fixed-reference row with seed=-1")
    if (frame.loc[frame.condition != "FULL", "seed"] < 0).any():
        raise ValueError("Thinning seed IDs must be nonnegative")
    frame["query_count"] = pd.to_numeric(frame.query_count, errors="raise")
    if (frame.query_count.isna().any() or not np.isfinite(frame.query_count).all()
            or (frame.query_count <= 0).any() or (frame.query_count % 1 != 0).any()
            or frame.query_count.nunique() > 1):
        raise ValueError("Every row must refer to the same nonempty fixed query set")
    frame["evaluation_completed"] = frame.evaluation_completed.map(_completed)
    for column in (*METRIC_COLUMNS, *QREL_COLUMNS,
                   "paired_FULL_mean_ANN_minus_IR_budget_gap", "paired_gap_query_count"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"Infinite metric: {column}")
    if "paired_gap_query_count" in frame and len(frame):
        counts = frame.paired_gap_query_count.dropna()
        if ((counts < 0).any() or (counts % 1 != 0).any()
                or (counts > frame.query_count.iloc[0]).any()):
            raise ValueError("Matched gap query counts must be integers within the fixed query count")
    # This is only a necessary aggregate check. The ranking evaluator must also
    # verify per-query Recall invariance using unchanged ranked document IDs.
    scored = frame[frame.evaluation_completed]
    if "ANN_Recall_at_10" in scored:
        recall = scored.ANN_Recall_at_10.dropna().to_numpy(dtype=float)
        if len(recall) and not np.allclose(recall, recall[0], rtol=0, atol=1e-12):
            raise ValueError("ANN Recall changed across qrel-thinning conditions/seeds")
    return frame.sort_values(["condition", "seed"]).reset_index(drop=True)


def _resample_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _percentile_mean_ci(values: np.ndarray, count: int, seed: int) -> tuple[float, float]:
    """Bootstrap seed means with bounded memory; seed is the resampling unit."""
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(count, dtype=float)
    # Restrict temporary draws even if restored input contains many seeds.
    batch = max(1, min(2048, 1_000_000 // len(values)))
    for start in range(0, count, batch):
        size = min(batch, count - start)
        means[start:start + size] = values[rng.integers(0, len(values), (size, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def add_seed_confidence_intervals(
    per_seed: pd.DataFrame, *, bootstrap_resamples: int = 20_000,
    random_seed: int = 20261115, metric_columns: Iterable[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Return four condition summaries without treating query/seed rows as IID.

    Means use genuinely evaluated seeds only. Partial scored conditions retain
    their observed estimates and are explicitly incomplete; qrel preparation
    counts can be reported before ranking evaluation. Each metric has its own
    scored-seed count, allowing genuinely unavailable optional CURA metrics.
    """
    count = _resample_count(bootstrap_resamples, "bootstrap_resamples")
    frame = _frame(per_seed)
    columns = tuple(metric_columns)
    records = []
    for ci, condition in enumerate(CONDITIONS):
        prepared = frame[frame.condition == condition]
        scored = prepared[prepared.evaluation_completed]
        n = len(scored)
        target = 1 if condition == "FULL" else REQUIRED_THINNING_SEEDS
        record = dict(
            condition=condition, queries=int(frame.query_count.iloc[0]) if len(frame) else np.nan,
            number_of_evaluated_seeds=n, qrel_seeds_prepared=len(prepared),
            ranking_seeds_evaluated=n, evaluation_completed=n >= target,
            experiment_completed=n >= target,
            status="COMPLETE" if n >= target else ("INCOMPLETE_SEED_COUNT" if n else "BLOCKED_NO_SCORED_SEEDS"),
            required_evaluated_seeds=target, ci_scope=CI_SCOPE,
            ci_kind="fixed reference; degenerate interval, not query uncertainty" if condition == "FULL" else "percentile bootstrap of seed-level means",
            resampling_unit="none: fixed FULL reference" if condition == "FULL" else "independent thinning seed",
            random_seed=int(random_seed + ci * 100),
            bootstrap_resamples=0 if condition == "FULL" else count,
            thinning_seed_ids=json.dumps(scored.seed.tolist()),
        )
        for column in QREL_COLUMNS:
            values = prepared[column].dropna() if column in prepared else pd.Series(dtype=float)
            record[column] = float(values.mean()) if len(values) else np.nan
        for mi, column in enumerate(columns):
            values = scored[column].dropna().to_numpy(dtype=float) if column in scored else np.array([], dtype=float)
            record[column] = float(values.mean()) if len(values) else np.nan
            record[column + "_evaluated_seeds"] = len(values)
            if condition == "FULL" and len(values):
                low = high = float(values[0])
            else:
                low, high = _percentile_mean_ci(values, count, random_seed + ci * 100 + mi)
            record[column + "_ci95_low"] = low
            record[column + "_ci95_high"] = high
        core_complete = all(
            column in scored and scored[column].notna().sum() >= target
            for column in CORE_METRIC_COLUMNS
        )
        record["required_metrics_completed"] = bool(core_complete)
        record["evaluation_completed"] = bool(n >= target and core_complete)
        record["experiment_completed"] = record["evaluation_completed"]
        if n >= target and not core_complete:
            record["status"] = "INCOMPLETE_METRICS"
        records.append(record)
    return pd.DataFrame.from_records(records)


def _signflip(values: np.ndarray, resamples: int, seed: int) -> tuple[float, str, int, int]:
    """Two-sided conditional sign-flip; zero effects remain in the mean's N."""
    nonzero = values[values != 0]
    n = len(nonzero)
    if not n:
        return 1.0, "exact seed-level paired sign-flip", 0, 1
    observed = abs(float(nonzero.sum()))
    tolerance = max(1.0, float(np.abs(nonzero).sum())) * 1e-12
    if n <= 18:
        total = 1 << n
        extreme = 0
        for start in range(0, total, 32768):
            code = np.arange(start, min(start + 32768, total), dtype=np.uint64)
            signs = 2 * ((code[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float) - 1
            extreme += int((np.abs(signs @ nonzero) >= observed - tolerance).sum())
        return extreme / total, "exact seed-level paired sign-flip", n, total
    rng = np.random.default_rng(seed)
    extreme = 0
    batch = max(1, min(2048, 1_000_000 // n))
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        signs = 2 * rng.integers(0, 2, (size, n)) - 1
        extreme += int((np.abs(signs @ nonzero) >= observed - tolerance).sum())
    return (extreme + 1) / (resamples + 1), "Monte Carlo seed-level paired sign-flip", n, resamples


def density_paired_tests(
    per_seed: pd.DataFrame, *, bootstrap_resamples: int = 20_000,
    signflip_resamples: int = 20_000, random_seed: int = 20262115,
) -> pd.DataFrame:
    """Return P5/P6 comparisons; inferential completion requires >=50 scores.

    Effects are thinned-minus-FULL, with one effect per independent thinning
    seed. If supplied, P5 uses each seed's FULL gap on the identical matched
    query cohort; supplied-but-missing matched references never fall back to
    global FULL. P6 always uses the fixed FULL exact-zero fraction.
    No query-by-seed pooling, test of query-population generalization, or
    equivalence conclusion is made. Sign flipping is exploratory: it requires
    symmetry/sign exchangeability of seed effects under the null; thinning
    conditions are not randomized treatment assignments.
    """
    boot_count = _resample_count(bootstrap_resamples, "bootstrap_resamples")
    flip_count = _resample_count(signflip_resamples, "signflip_resamples")
    frame = _frame(per_seed)
    full = frame[(frame.condition == "FULL") & frame.evaluation_completed]
    records = []
    specs = (
        ("P5", "mean_ANN_minus_IR_budget_gap", "thinned minus FULL ANN-minus-IR budget gap", "efSearch"),
        ("P6", "exact_ndcg_zero_query_fraction", "thinned minus FULL exact-zero fraction", "fraction of queries"),
    )
    for pi, (pid, column, metric, units) in enumerate(specs):
        for ci, condition in enumerate(CONDITIONS[1:]):
            seed = int(random_seed + pi * 100 + ci * 2)
            prepared = frame[frame.condition == condition]
            scored = prepared[prepared.evaluation_completed]
            full_value = float(full.iloc[0][column]) if len(full) and column in full and pd.notna(full.iloc[0][column]) else np.nan
            matched_reference = pid == "P5" and any(
                name in frame for name in (
                    "paired_FULL_mean_ANN_minus_IR_budget_gap", "paired_gap_query_count"
                )
            )
            valid = scored[column].notna() if column in scored else pd.Series(False, index=scored.index)
            if matched_reference:
                if all(name in scored for name in (
                    "paired_FULL_mean_ANN_minus_IR_budget_gap", "paired_gap_query_count"
                )):
                    valid &= (scored.paired_FULL_mean_ANN_minus_IR_budget_gap.notna()
                              & scored.paired_gap_query_count.notna()
                              & (scored.paired_gap_query_count > 0))
                else:
                    valid &= False
            paired = scored.loc[valid]
            values = paired[column].to_numpy(dtype=float) if column in paired else np.array([], dtype=float)
            if matched_reference:
                references = paired.paired_FULL_mean_ANN_minus_IR_budget_gap.to_numpy(dtype=float) if "paired_FULL_mean_ANN_minus_IR_budget_gap" in paired else np.array([], dtype=float)
                reference_value = float(references.mean()) if len(references) else np.nan
                gap_counts = paired.paired_gap_query_count.to_numpy(dtype=float) if "paired_gap_query_count" in paired else np.array([], dtype=float)
                reference_scope = "FULL evaluated on each seed's identical jointly feasible query cohort"
                estimator = "query-paired thinned-minus-FULL gap effect averaged over thinning seeds"
            else:
                references = np.full(len(values), full_value, dtype=float)
                reference_value = full_value
                gap_counts = np.array([], dtype=float)
                reference_scope = "fixed FULL value on the entire fixed query set"
                estimator = "mean seed-level thinned-minus-fixed-FULL effect"
            complete = len(values) >= REQUIRED_THINNING_SEEDS and len(full) == 1 and np.isfinite(references).all()
            delta = values - references
            record = {name: np.nan for name in PRIMARY_COLUMNS}
            record.update(
                primary_test_id=pid, cohort="TREC_DL_2019", comparison=condition + " minus FULL",
                metric=metric, units=units, estimator=estimator,
                queries=int(frame.query_count.iloc[0]) if len(frame) else np.nan,
                test="NOT_RUN", test_status="COMPLETED_DENSITY_SEED_ANALYSIS" if complete else "INCOMPLETE_DENSITY_SEED_ANALYSIS",
                analysis_role="requested primary comparison", random_seed=seed,
                bootstrap_resamples=boot_count if complete else 0,
                thinning_seeds_prepared=len(prepared), thinning_seeds_evaluated=len(values),
                multiplicity="Exploratory, unadjusted; incomplete tests have no p-value",
                ci_scope=CI_SCOPE, resampling_unit="independent thinning seed",
                effect_direction="thinned minus FULL", FULL_reference_value=reference_value,
                FULL_global_reference_value=full_value, reference_scope=reference_scope,
                seed_matched_FULL_gap_mean=reference_value if matched_reference else np.nan,
                paired_gap_query_count_mean=float(gap_counts.mean()) if len(gap_counts) else np.nan,
                paired_gap_query_count_min=float(gap_counts.min()) if len(gap_counts) else np.nan,
                paired_gap_query_count_max=float(gap_counts.max()) if len(gap_counts) else np.nan,
                signflip_random_seed=seed + 1, signflip_resamples=0,
                thinning_seed_ids=json.dumps(paired.seed.tolist()),
                interpretation="Seed effects condition on the fixed query set and unchanged rankings; nonzero_pairs counts seeds, not queries. Sign-flip inference assumes sign exchangeability. Nonsignificance does not establish equivalence or no material effect.",
            )
            if matched_reference:
                record["interpretation"] += " Each gap effect pairs FULL and thinned values on that seed's identical jointly feasible queries; the selected query cohort can differ across seeds."
            if complete:
                low, high = _percentile_mean_ci(delta, boot_count, seed)
                p, test, nonzero, draws = _signflip(delta, flip_count, seed + 1)
                record.update(mean_difference=float(delta.mean()), ci95_low=low, ci95_high=high,
                              p_value=p, test=test, nonzero_pairs=nonzero, signflip_resamples=draws)
            else:
                record["interpretation"] += " At least 50 evaluated seed effects and a scored FULL reference are required; provided gap-pairing fields also require finite seed-matched FULL values and positive matched query counts. No inferential result was generated."
            records.append(record)
    return pd.DataFrame.from_records(records)
