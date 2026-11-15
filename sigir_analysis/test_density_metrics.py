"""Synthetic, in-memory checks; these are not CURA experimental observations."""
import math
import unittest

import numpy as np
import pandas as pd

try:
    from .density_metrics import (
        POLICY_LABEL, aggregate_fixed, aggregate_frozen_cura,
        evaluate_query_rankings, ndcg_at_10, summarize_queries, wilson_interval,
    )
except ImportError:
    from density_metrics import (
        POLICY_LABEL, aggregate_fixed, aggregate_frozen_cura,
        evaluate_query_rankings, ndcg_at_10, summarize_queries, wilson_interval,
    )


class DensityMetricsTests(unittest.TestCase):
    def setUp(self):
        self.reference = [f"d{i}" for i in range(10)]
        self.efs = [10, 16, 24]

    def evaluate(self, rankings, judgments=None):
        return evaluate_query_rankings(
            {"raw::000123": self.reference},
            {("raw::000123", ef): docs for ef, docs in zip(self.efs, rankings)},
            {"raw::000123": {"d0": 3} if judgments is None else judgments}, self.efs,
        )

    def test_graded_idcg_uses_all_judgments_and_unjudged_zero(self):
        judgments = {"d0": 3, "outside": 2, "another": 1, "d1": 0}
        expected = 7 / (7 + 3 / math.log2(3) + 1 / math.log2(4))
        self.assertAlmostEqual(ndcg_at_10(self.reference, judgments), expected)
        self.assertEqual(ndcg_at_10(self.reference, {}), 0)

    def test_first_safe_does_not_assume_monotonicity(self):
        unsafe = ["unjudged", *self.reference[1:]]
        frame = self.evaluate([unsafe, self.reference, unsafe])
        self.assertEqual(frame.IR_unsafe.tolist(), [True, False, True])
        self.assertTrue((frame.min_IR_safe_ef == 16).all())
        self.assertTrue((frame.min_ANN_safe_ef == 10).all())
        self.assertTrue((frame.ANN_minus_IR_budget_gap == -6).all())
        self.assertEqual(set(frame.query_key), {"raw::000123"})

    def test_missing_safety_is_not_silently_maximum_ef(self):
        low_recall = ["d0", *[f"unjudged{i}" for i in range(9)]]
        frame = self.evaluate([low_recall] * 3)
        self.assertFalse(frame.ANN_safe_found.any())
        self.assertTrue(frame.IR_safe_found.all())
        self.assertTrue(frame.min_ANN_safe_ef.isna().all())
        self.assertTrue((frame.min_IR_safe_ef == 10).all())
        for column in ["ANN_minus_IR_budget_gap", "relative_IR_safe_budget_reduction", "IR_allows_lower", "same_budget", "IR_requires_higher"]:
            self.assertTrue(frame[column].isna().all(), column)
        summary = summarize_queries(frame)
        self.assertEqual(summary["jointly_feasible_query_count"], 0)
        self.assertTrue(math.isnan(summary["mean_min_ANN_safe_ef"]))
        self.assertTrue(math.isnan(summary["mean_min_IR_safe_ef"]))
        self.assertEqual(summary["mean_available_IR_safe_ef"], 10)
        self.assertEqual(summary["legacy_grid_capped_mean_ANN_minus_IR_budget_gap"], 14)
        self.assertEqual(summary["ANN_not_safe_in_grid"], 1)

    def test_ir_can_be_infeasible_despite_ann_safety(self):
        unsafe = ["unjudged", *self.reference[1:]]
        frame = self.evaluate([unsafe] * 3)
        self.assertTrue(frame.ANN_safe_found.all())
        self.assertFalse(frame.IR_safe_found.any())
        self.assertTrue(frame.min_IR_safe_ef.isna().all())
        self.assertTrue((frame.legacy_grid_capped_min_IR_safe_ef == 24).all())
        self.assertTrue(frame.ANN_minus_IR_budget_gap.isna().all())

    def test_recall_and_ann_minima_are_invariant_to_judgment_changes(self):
        unsafe = ["unjudged", *self.reference[1:]]
        full = self.evaluate([unsafe, self.reference, unsafe])
        empty = self.evaluate([unsafe, self.reference, unsafe], {})
        np.testing.assert_array_equal(full.ANN_Recall_at_10, empty.ANN_Recall_at_10)
        np.testing.assert_array_equal(full.min_ANN_safe_ef, empty.min_ANN_safe_ef)
        self.assertEqual(empty.exact_nDCG_at_10.sum(), 0)
        self.assertFalse(empty.IR_unsafe.any())
        self.assertEqual(summarize_queries(empty)["exact_ndcg_zero_query_fraction"], 1)

    def test_summary_deduplicates_queries_and_uses_ratio_of_means(self):
        references = {"first": self.reference, "second": self.reference}
        judgments = {"first": {"d0": 3}, "second": {"d0": 3}}
        low_recall = ["d0", *[f"x{i}" for i in range(9)]]
        hnsw = {
            ("first", 10): self.reference, ("first", 16): self.reference,
            ("first", 24): self.reference,
            ("second", 10): low_recall, ("second", 16): low_recall,
            ("second", 24): self.reference,
        }
        frame = evaluate_query_rankings(references, hnsw, judgments, self.efs)
        summary = summarize_queries(frame)
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["mean_min_ANN_safe_ef"], 17)
        self.assertEqual(summary["mean_min_IR_safe_ef"], 10)
        self.assertAlmostEqual(summary["relative_IR_safe_budget_reduction"], 7 / 17)
        self.assertAlmostEqual(summary["mean_query_relative_IR_safe_budget_reduction"], (14 / 24) / 2)
        changed = frame.copy()
        changed.loc[0, "min_IR_safe_ef"] = 16
        with self.assertRaises(ValueError):
            summarize_queries(changed)

    def test_fixed_and_frozen_cura_preserve_saved_effective_costs(self):
        unsafe = ["unjudged", *self.reference[1:]]
        frame = self.evaluate([unsafe, self.reference, self.reference])
        work = pd.DataFrame({
            "query_key": ["raw::000123"] * 3, "efSearch": self.efs,
            "distance_computations": [50.0, 100.0, 200.0],
            "mean_latency_ms": [0.1, 0.2, 0.3],
        })
        frontier = aggregate_fixed(frame, work)
        np.testing.assert_array_equal(frontier.IR_violation_rate, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(frontier.iloc[1].wilson95_upper, wilson_interval(0, 1)[1])
        selections = pd.DataFrame({
            "query_key": ["raw::000123"], "efSearch": [24],
            "effective_distance_computations": [250.0], "effective_latency_ms": [0.4],
            # Deliberately stale original metrics must not leak into alternate judgments.
            "IR_unsafe": [True], "nDCG@10": [-100.0],
        })
        before = selections.copy(deep=True)
        result = aggregate_frozen_cura(frame, selections, frontier)
        self.assertEqual(result["policy_label"], POLICY_LABEL)
        self.assertEqual(result["CURA_IR_violation_rate"], 0)
        self.assertEqual(result["mean_work"], 250)
        self.assertEqual(result["mean_latency_ms"], 0.4)
        self.assertEqual(result["risk_matched_fixed_ef"], 16)
        self.assertEqual(result["CURA_minus_fixed_work"], 150)
        self.assertEqual(result["fixed_work_saving_vs_CURA_pct"], 60)
        pd.testing.assert_frame_equal(selections, before)
        pd.testing.assert_frame_equal(work, pd.DataFrame({
            "query_key": ["raw::000123"] * 3, "efSearch": self.efs,
            "distance_computations": [50.0, 100.0, 200.0],
            "mean_latency_ms": [0.1, 0.2, 0.3],
        }))

    def test_reject_duplicate_incomplete_or_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.evaluate([["duplicate"] * 10] * 3)
        with self.assertRaises(ValueError):
            evaluate_query_rankings({"q": self.reference}, {}, {"q": {}}, self.efs)
        with self.assertRaises(ValueError):
            self.evaluate([self.reference] * 3, {"d0": -1})
        frame = self.evaluate([self.reference] * 3)
        with self.assertRaises(ValueError):
            summarize_queries(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        work = pd.DataFrame({
            "query_key": ["raw::000123"], "efSearch": [10],
            "distance_computations": [1.0], "mean_latency_ms": [0.1],
        })
        with self.assertRaises(ValueError):
            aggregate_fixed(frame, work)


if __name__ == "__main__":
    unittest.main()
