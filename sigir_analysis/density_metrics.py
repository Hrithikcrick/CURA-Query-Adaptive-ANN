"""Pure metrics for a future same-ranking judgment-density evaluation.

This module reads and writes no files and performs no retrieval. Callers must
verify ranking provenance, query/document identities and FULL reproduction
before treating its return values as experimental results. Synthetic unit
fixtures exercise the implementation without creating research observations.

Primary oracle comparisons use actually feasible minima. ``legacy_grid_capped``
fields explicitly reproduce the old maximum-ef substitution for sensitivity
analysis; that substitution never changes a ``*_safe_found`` flag.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_EFS = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256)
IR_TOLERANCE = 0.025
ANN_RECALL_THRESHOLD = 0.9
POLICY_LABEL = "unchanged CURA policy under alternate relevance-judgment evaluation"
_PRIMARY_QUERY_FIELDS = (
    "exact_nDCG_at_10", "min_ANN_safe_ef", "min_IR_safe_ef",
    "ANN_safe_found", "IR_safe_found", "ANN_minus_IR_budget_gap",
    "relative_IR_safe_budget_reduction", "IR_allows_lower", "same_budget",
    "IR_requires_higher", "legacy_grid_capped_min_ANN_safe_ef",
    "legacy_grid_capped_min_IR_safe_ef", "legacy_grid_capped_ANN_minus_IR_budget_gap",
    "legacy_grid_capped_relative_IR_safe_budget_reduction",
    "legacy_grid_capped_IR_allows_lower", "legacy_grid_capped_same_budget",
    "legacy_grid_capped_IR_requires_higher",
)


def _grid(efs: Sequence[int]) -> list[int]:
    if not efs or any(isinstance(e, bool) or not isinstance(e, Integral) or e <= 0 for e in efs):
        raise ValueError("efSearch grid must contain positive integers")
    values = [int(e) for e in efs]
    if len(set(values)) != len(values):
        raise ValueError("efSearch grid contains duplicates")
    return sorted(values)


def _top10(documents: Sequence[str], label: str) -> list[str]:
    if isinstance(documents, (str, bytes)) or len(documents) < 10:
        raise ValueError(f"{label} must contain at least ten document IDs")
    top = list(documents[:10])
    if any(not isinstance(d, str) or not d for d in top):
        raise ValueError(f"{label} requires nonempty original string document IDs")
    if len(set(top)) != 10:
        raise ValueError(f"{label} contains duplicate top10 document IDs")
    return top


def ndcg_at_10(documents: Sequence[str], judgments: Mapping[str, int]) -> float:
    """Original exponential-gain nDCG; IDCG uses all positive query qrels."""
    def dcg(grades):
        return sum((2 ** int(g) - 1) / math.log2(rank + 2) for rank, g in enumerate(grades))

    ideal = sorted((g for g in judgments.values() if g > 0), reverse=True)[:10]
    denominator = dcg(ideal)
    if denominator <= 0:
        return 0.0
    return dcg(judgments.get(doc, 0) for doc in documents[:10]) / denominator


def evaluate_query_rankings(
    exact: dict[str, list[str]],
    hnsw: dict[tuple[str, int], list[str]],
    qrels: dict[str, dict[str, int]],
    efs: Sequence[int] = DEFAULT_EFS,
) -> pd.DataFrame:
    """Evaluate frozen rankings under one judgment condition.

    Query keys are preserved exactly. All query/ef combinations are required.
    First-safe minima scan the tested grid without assuming monotone safety.
    A missing criterion minimum is NaN. Paired gaps, reductions and comparison
    indicators are NaN whenever either criterion has no safe tested setting.
    """
    grid = _grid(efs)
    if not exact or any(not isinstance(q, str) or not q for q in exact):
        raise ValueError("Exact rankings require nonempty string query keys")
    if not set(exact).issubset(qrels):
        raise ValueError("Qrels are missing an exact-ranking query")
    required = {(query, ef) for query in exact for ef in grid}
    if set(hnsw) != required:
        raise ValueError("HNSW rankings must cover exactly every query and tested ef")
    for query in exact:
        if any(not isinstance(doc, str) or not doc for doc in qrels[query]):
            raise ValueError("Qrels require nonempty original string document IDs")
        if any(isinstance(g, bool) or not isinstance(g, Integral) or g < 0 for g in qrels[query].values()):
            raise ValueError("Qrel grades must be nonnegative integers")

    rows = []
    for query, documents in exact.items():
        reference = _top10(documents, f"Exact ranking for {query}")
        exact_score = ndcg_at_10(reference, qrels[query])
        reference_set = set(reference)
        query_rows = []
        for ef in grid:
            approximate = _top10(hnsw[query, ef], f"HNSW ranking for {query} at ef{ef}")
            approximate_score = ndcg_at_10(approximate, qrels[query])
            loss = max(0.0, exact_score - approximate_score)
            recall = len(reference_set.intersection(approximate)) / 10.0
            query_rows.append(dict(
                query_key=query, efSearch=ef, exact_nDCG_at_10=exact_score,
                HNSW_nDCG_at_10=approximate_score, nDCG_loss=loss,
                IR_unsafe=bool(loss > IR_TOLERANCE), ANN_Recall_at_10=recall,
            ))
        ann = next((row["efSearch"] for row in query_rows if row["ANN_Recall_at_10"] >= ANN_RECALL_THRESHOLD), None)
        ir = next((row["efSearch"] for row in query_rows if not row["IR_unsafe"]), None)
        joint = ann is not None and ir is not None
        capped_ann, capped_ir = ann if ann is not None else grid[-1], ir if ir is not None else grid[-1]
        gap = ann - ir if joint else np.nan
        oracle = dict(
            min_ANN_safe_ef=ann if ann is not None else np.nan,
            min_IR_safe_ef=ir if ir is not None else np.nan,
            ANN_safe_found=ann is not None, IR_safe_found=ir is not None,
            ANN_minus_IR_budget_gap=gap,
            relative_IR_safe_budget_reduction=gap / ann if joint else np.nan,
            IR_allows_lower=float(ir < ann) if joint else np.nan,
            same_budget=float(ir == ann) if joint else np.nan,
            IR_requires_higher=float(ir > ann) if joint else np.nan,
            legacy_grid_capped_min_ANN_safe_ef=capped_ann,
            legacy_grid_capped_min_IR_safe_ef=capped_ir,
            legacy_grid_capped_ANN_minus_IR_budget_gap=capped_ann - capped_ir,
            legacy_grid_capped_relative_IR_safe_budget_reduction=(capped_ann - capped_ir) / capped_ann,
            legacy_grid_capped_IR_allows_lower=float(capped_ir < capped_ann),
            legacy_grid_capped_same_budget=float(capped_ir == capped_ann),
            legacy_grid_capped_IR_requires_higher=float(capped_ir > capped_ann),
        )
        rows.extend({**row, **oracle} for row in query_rows)
    return pd.DataFrame(rows)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} cannot be empty")


def _validate_grid_frame(frame: pd.DataFrame) -> None:
    _require_columns(frame, ["query_key", "efSearch"], "Ranking metrics")
    if frame[["query_key", "efSearch"]].isna().any().any() or frame.duplicated(["query_key", "efSearch"]).any():
        raise ValueError("Ranking metrics require unique, nonmissing query/ef rows")
    grid = set(frame.efSearch)
    if any(set(group.efSearch) != grid for _, group in frame.groupby("query_key", sort=False)):
        raise ValueError("Ranking metrics have incomplete query/ef coverage")


def _mean(series: pd.Series) -> float:
    return float(series.mean()) if series.notna().any() else float("nan")


def summarize_queries(frame: pd.DataFrame) -> dict:
    """Summarize one condition/seed, deduplicating repeated query oracle values.

    Primary paired minima and gap means use jointly feasible queries. Separate
    available-criterion means expose differing denominators. Relative aggregate
    reduction is ratio of means, matching archived oracle summaries; the mean
    of per-query relative reductions is reported separately.
    """
    _validate_grid_frame(frame)
    _require_columns(frame, [*_PRIMARY_QUERY_FIELDS, "ANN_Recall_at_10"], "Ranking metrics")
    for column in _PRIMARY_QUERY_FIELDS:
        if (frame.groupby("query_key", sort=False)[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"Query-level field varies across ef settings: {column}")
    queries = frame.drop_duplicates("query_key")
    joint = queries[queries.ANN_safe_found & queries.IR_safe_found]
    ann_mean = _mean(joint.min_ANN_safe_ef)
    ir_mean = _mean(joint.min_IR_safe_ef)
    result = dict(
        query_count=len(queries), jointly_feasible_query_count=len(joint),
        ANN_not_safe_in_grid=int((~queries.ANN_safe_found).sum()),
        IR_not_safe_in_grid=int((~queries.IR_safe_found).sum()),
        exact_ndcg_zero_query_fraction=float((queries.exact_nDCG_at_10 == 0).mean()),
        mean_exact_ndcg_at_10=_mean(queries.exact_nDCG_at_10),
        mean_min_ANN_safe_ef=ann_mean, mean_min_IR_safe_ef=ir_mean,
        mean_available_ANN_safe_ef=_mean(queries.min_ANN_safe_ef),
        mean_available_IR_safe_ef=_mean(queries.min_IR_safe_ef),
        mean_ANN_minus_IR_budget_gap=_mean(joint.ANN_minus_IR_budget_gap),
        relative_IR_safe_budget_reduction=(ann_mean - ir_mean) / ann_mean if len(joint) else np.nan,
        mean_query_relative_IR_safe_budget_reduction=_mean(joint.relative_IR_safe_budget_reduction),
        IR_allows_lower_rate=_mean(joint.IR_allows_lower),
        same_budget_rate=_mean(joint.same_budget),
        IR_requires_higher_rate=_mean(joint.IR_requires_higher),
        ANN_Recall_at_10=_mean(frame.ANN_Recall_at_10),
        oracle_estimator="both_criteria_feasible",
        ANN_Recall_scope="mean over the complete query-by-ef grid",
    )
    capped_ann = _mean(queries.legacy_grid_capped_min_ANN_safe_ef)
    capped_ir = _mean(queries.legacy_grid_capped_min_IR_safe_ef)
    result.update(
        legacy_grid_capped_mean_min_ANN_safe_ef=capped_ann,
        legacy_grid_capped_mean_min_IR_safe_ef=capped_ir,
        legacy_grid_capped_mean_ANN_minus_IR_budget_gap=_mean(queries.legacy_grid_capped_ANN_minus_IR_budget_gap),
        legacy_grid_capped_relative_IR_safe_budget_reduction=(capped_ann - capped_ir) / capped_ann,
        legacy_grid_capped_IR_allows_lower_rate=_mean(queries.legacy_grid_capped_IR_allows_lower),
        legacy_grid_capped_same_budget_rate=_mean(queries.legacy_grid_capped_same_budget),
        legacy_grid_capped_IR_requires_higher_rate=_mean(queries.legacy_grid_capped_IR_requires_higher),
        primary_minimum_scope="jointly feasible queries; no-safe minima are not imputed",
        legacy_grid_capped_scope="all queries; no-safe criterion substitutes largest tested ef without claiming safety",
    )
    return result


def wilson_interval(violations: int, query_count: int) -> tuple[float, float]:
    """Pointwise two-sided Wilson interval, z=1.96 as in the saved protocol."""
    if query_count <= 0 or violations < 0 or violations > query_count:
        raise ValueError("Invalid binomial counts")
    z = 1.96
    proportion = violations / query_count
    denominator = 1 + z * z / query_count
    center = (proportion + z * z / (2 * query_count)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / query_count + z * z / (4 * query_count ** 2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _validate_measurements(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Saved measurement {column} must be finite and nonnegative")


def aggregate_fixed(frame: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
    """All fixed ef points with frozen saved work and per-query mean timing."""
    _validate_grid_frame(frame)
    _require_columns(frame, ["IR_unsafe", "HNSW_nDCG_at_10", "nDCG_loss", "ANN_Recall_at_10"], "Ranking metrics")
    columns = ["query_key", "efSearch", "distance_computations", "mean_latency_ms"]
    _require_columns(work, columns, "Fixed work")
    if work.duplicated(["query_key", "efSearch"]).any():
        raise ValueError("Fixed work contains duplicate query/ef rows")
    _validate_measurements(work, columns[2:])
    keys = ["query_key", "efSearch"]
    if set(map(tuple, frame[keys].to_numpy())) != set(map(tuple, work[keys].to_numpy())):
        raise ValueError("Fixed work must cover exactly the ranking query/ef grid")
    merged = frame.drop(columns=columns[2:], errors="ignore").merge(work[columns], on=keys, validate="one_to_one")
    rows = []
    for ef, group in merged.groupby("efSearch", sort=True):
        violations = int(group.IR_unsafe.sum())
        low, high = wilson_interval(violations, len(group))
        rows.append(dict(
            method=f"Fixed ef={int(ef)}", efSearch=int(ef), query_count=len(group),
            IR_violations=violations, IR_violation_rate=violations / len(group),
            wilson95_low=low, wilson95_upper=high,
            ANN_Recall_at_10=_mean(group.ANN_Recall_at_10),
            mean_nDCG_at_10=_mean(group.HNSW_nDCG_at_10),
            mean_nDCG_loss=_mean(group.nDCG_loss),
            mean_distance_computations=_mean(group.distance_computations),
            mean_latency_ms=_mean(group.mean_latency_ms),
            p95_latency_ms=float(group.mean_latency_ms.quantile(0.95)),
            latency_scope="saved per-query search means; p95 across query means, not request latency",
            work_scope="single frozen fixed search", diagnostic_only=True,
        ))
    return pd.DataFrame(rows)


def aggregate_frozen_cura(
    frame: pd.DataFrame,
    selections: pd.DataFrame,
    fixed_frontier: pd.DataFrame | None = None,
) -> dict:
    """Score unchanged choices while charging original effective work/timing.

    Pass ``aggregate_fixed(...)`` as the optional third argument to calculate a
    retrospective matched-observed-risk diagnostic. No selection is refitted.
    """
    _validate_grid_frame(frame)
    metrics = ["query_key", "efSearch", "IR_unsafe", "HNSW_nDCG_at_10", "nDCG_loss", "ANN_Recall_at_10"]
    _require_columns(frame, metrics, "Ranking metrics")
    chosen = ["query_key", "efSearch", "effective_distance_computations", "effective_latency_ms"]
    _require_columns(selections, chosen, "Frozen selections")
    if selections.query_key.isna().any() or selections.query_key.duplicated().any():
        raise ValueError("Frozen selections must contain exactly one choice per query")
    if set(selections.query_key) != set(frame.query_key):
        raise ValueError("Frozen selections do not match the ranking query set")
    _validate_measurements(selections, chosen[2:])
    merged = selections[chosen].merge(frame[metrics], on=["query_key", "efSearch"], validate="one_to_one")
    if len(merged) != len(selections):
        raise ValueError("A frozen selection has no matching ranked ef")
    violations = int(merged.IR_unsafe.sum())
    low, high = wilson_interval(violations, len(merged))
    risk = violations / len(merged)
    mean_work = _mean(merged.effective_distance_computations)
    result = dict(
        policy_label=POLICY_LABEL, query_count=len(merged), CURA_IR_violations=violations,
        CURA_IR_violation_rate=risk, wilson95_low=low, wilson95_upper=high,
        mean_nDCG_at_10=_mean(merged.HNSW_nDCG_at_10),
        mean_nDCG_loss=_mean(merged.nDCG_loss), ANN_Recall_at_10=_mean(merged.ANN_Recall_at_10),
        mean_work=mean_work, CURA_mean_distance_computations=mean_work,
        mean_latency_ms=_mean(merged.effective_latency_ms),
        p95_latency_ms=float(merged.effective_latency_ms.quantile(0.95)),
        mean_efSearch=_mean(merged.efSearch),
        work_scope="original effective work: probe only at ef10, otherwise probe plus selected fresh search",
        latency_scope="original effective search-component means; excludes model/feature overhead; p95 across query means",
        risk_matched_fixed_ef=np.nan, fixed_observed_violation_rate=np.nan,
        risk_matched_fixed_mean_work=np.nan, CURA_minus_fixed_work=np.nan,
        fixed_work_saving_vs_CURA_pct=np.nan,
        risk_matching_scope="post-hoc matched observed risk only; no prospective calibration or population-risk equivalence",
    )
    if fixed_frontier is None and {"distance_computations", "mean_latency_ms"}.issubset(frame):
        fixed_frontier = aggregate_fixed(frame, frame[["query_key", "efSearch", "distance_computations", "mean_latency_ms"]])
    if fixed_frontier is not None:
        _require_columns(fixed_frontier, ["efSearch", "query_count", "IR_violation_rate", "mean_distance_computations"], "Fixed frontier")
        if fixed_frontier.efSearch.duplicated().any() or set(fixed_frontier.efSearch) != set(frame.efSearch):
            raise ValueError("Fixed frontier must contain exactly one row for every tested ef")
        if not (fixed_frontier.query_count == len(merged)).all():
            raise ValueError("Fixed frontier and CURA query counts differ")
        _validate_measurements(fixed_frontier, ["mean_distance_computations", "IR_violation_rate"])
        if (fixed_frontier.IR_violation_rate > 1).any():
            raise ValueError("Fixed violation rates exceed one")
        eligible = fixed_frontier[fixed_frontier.IR_violation_rate <= risk]
        if not eligible.empty:
            match = eligible.sort_values(["mean_distance_computations", "efSearch"]).iloc[0]
            result.update(
                risk_matched_fixed_ef=int(match.efSearch),
                fixed_observed_violation_rate=float(match.IR_violation_rate),
                risk_matched_fixed_mean_work=float(match.mean_distance_computations),
                CURA_minus_fixed_work=mean_work - float(match.mean_distance_computations),
                fixed_work_saving_vs_CURA_pct=100 * (mean_work - float(match.mean_distance_computations)) / mean_work if mean_work else np.nan,
            )
    return result
