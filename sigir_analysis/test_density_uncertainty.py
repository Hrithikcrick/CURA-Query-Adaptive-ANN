"""Synthetic in-memory checks; these are not observations from the TREC study."""
import unittest

import numpy as np
import pandas as pd

from density_uncertainty import (
    CONDITIONS, METRIC_COLUMNS, add_seed_confidence_intervals, density_paired_tests,
)


def synthetic_frame(n=50, completed=True):
    rows = []
    for condition in CONDITIONS:
        seeds = [-1] if condition == "FULL" else range(n)
        for seed in seeds:
            row = dict(condition=condition, seed=seed, query_count=43,
                       evaluation_completed=completed, retained_positive_judgments=43,
                       retained_positive_fraction=.1, preserved_nonpositive_judgments=10)
            row.update({name: 1.0 if completed else np.nan for name in METRIC_COLUMNS})
            if completed:
                row["ANN_Recall_at_10"] = .95
                row["mean_ANN_minus_IR_budget_gap"] = 3 if seed == -1 else 3 + (seed % 3)
                row["exact_ndcg_zero_query_fraction"] = .1 if seed == -1 else .2
            rows.append(row)
    return pd.DataFrame(rows)


class DensityUncertaintyTests(unittest.TestCase):
    def test_prepared_seeds_are_not_observations(self):
        frame = synthetic_frame(completed=False)
        frame["evaluation_completed"] = "False"  # CSV round-trip semantics.
        summary = add_seed_confidence_intervals(frame, bootstrap_resamples=20)
        tests = density_paired_tests(frame, bootstrap_resamples=20, signflip_resamples=20)
        self.assertEqual(len(summary), 4)
        self.assertEqual(summary.number_of_evaluated_seeds.sum(), 0)
        self.assertFalse(summary.experiment_completed.any())
        self.assertTrue(summary.mean_ANN_minus_IR_budget_gap.isna().all())
        self.assertEqual(len(tests), 6)
        self.assertTrue(tests.mean_difference.isna().all())
        self.assertTrue(tests.p_value.isna().all())

    def test_summary_and_tests_are_reproducible_and_seed_level(self):
        frame = synthetic_frame()
        kwargs = dict(bootstrap_resamples=100, random_seed=7)
        left = add_seed_confidence_intervals(frame, **kwargs)
        right = add_seed_confidence_intervals(frame.sample(frac=1, random_state=1), **kwargs)
        pd.testing.assert_frame_equal(left, right)
        full = left[left.condition == "FULL"].iloc[0]
        self.assertEqual(full.mean_ANN_minus_IR_budget_gap_ci95_low, 3)
        self.assertEqual(full.mean_ANN_minus_IR_budget_gap_ci95_high, 3)
        self.assertIn("not query uncertainty", full.ci_kind)
        tests = density_paired_tests(frame, bootstrap_resamples=100, signflip_resamples=100, random_seed=7)
        p5 = tests[tests.primary_test_id == "P5"]
        self.assertTrue(np.allclose(p5.mean_difference, np.mean(np.arange(50) % 3)))
        self.assertTrue((tests.thinning_seeds_evaluated == 50).all())
        self.assertTrue((tests.queries == 43).all())
        self.assertTrue((tests.nonzero_pairs <= 50).all())
        self.assertTrue(tests.test_status.eq("COMPLETED_DENSITY_SEED_ANALYSIS").all())
        self.assertTrue(tests.interpretation.str.contains("does not establish equivalence").all())
        pd.testing.assert_frame_equal(tests, density_paired_tests(frame, bootstrap_resamples=100, signflip_resamples=100, random_seed=7))

    def test_exact_signflip_and_zero_effect(self):
        frame = synthetic_frame()
        frame["mean_ANN_minus_IR_budget_gap"] = 3.
        frame.loc[(frame.condition != "FULL") & frame.seed.isin([0, 1]), "mean_ANN_minus_IR_budget_gap"] = 4.
        frame["exact_ndcg_zero_query_fraction"] = .1
        tests = density_paired_tests(frame, bootstrap_resamples=100, signflip_resamples=100)
        p5 = tests[tests.primary_test_id == "P5"]
        self.assertTrue((p5.nonzero_pairs == 2).all())
        self.assertTrue((p5.p_value == .5).all())
        self.assertTrue((p5.signflip_resamples == 4).all())
        p6 = tests[tests.primary_test_id == "P6"]
        self.assertTrue((p6.p_value == 1).all())
        self.assertTrue((p6.ci95_low == 0).all())
        self.assertTrue((p6.ci95_high == 0).all())

    def test_partial_scored_conditions_stay_incomplete(self):
        frame = synthetic_frame(n=10)
        summary = add_seed_confidence_intervals(frame, bootstrap_resamples=20)
        self.assertFalse(summary[summary.condition != "FULL"].experiment_completed.any())
        tests = density_paired_tests(frame, bootstrap_resamples=20, signflip_resamples=20)
        self.assertTrue(tests.p_value.isna().all())
        self.assertTrue((tests.thinning_seeds_evaluated == 10).all())

    def test_invalid_pairing_and_ann_invariance_are_rejected(self):
        frame = synthetic_frame()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            add_seed_confidence_intervals(pd.concat([frame, frame.iloc[[1]]]))
        frame.loc[frame.seed == 0, "ANN_Recall_at_10"] = .9
        with self.assertRaisesRegex(ValueError, "ANN Recall changed"):
            density_paired_tests(frame)

    def test_completion_flags_do_not_replace_missing_metrics_or_full(self):
        frame = synthetic_frame()
        frame["mean_exact_ndcg_at_10"] = np.nan
        summary = add_seed_confidence_intervals(frame, bootstrap_resamples=20)
        self.assertFalse(summary.experiment_completed.any())
        self.assertTrue(summary.status.eq("INCOMPLETE_METRICS").all())
        tests = density_paired_tests(frame[frame.condition != "FULL"], bootstrap_resamples=20, signflip_resamples=20)
        self.assertTrue(tests.mean_difference.isna().all())
        self.assertTrue(tests.p_value.isna().all())

    def test_gap_comparison_uses_seed_matched_full_query_cohorts(self):
        frame = synthetic_frame()
        frame["paired_FULL_mean_ANN_minus_IR_budget_gap"] = 100.
        frame["paired_gap_query_count"] = 20
        frame.loc[frame.condition != "FULL", "mean_ANN_minus_IR_budget_gap"] = 102.
        tests = density_paired_tests(frame, bootstrap_resamples=100, signflip_resamples=100)
        p5 = tests[tests.primary_test_id == "P5"]
        self.assertTrue((p5.mean_difference == 2).all())
        self.assertTrue((p5.FULL_reference_value == 100).all())
        self.assertTrue((p5.FULL_global_reference_value == 3).all())
        self.assertTrue((p5.paired_gap_query_count_min == 20).all())
        self.assertTrue(p5.estimator.str.contains("query-paired").all())
        # P6 retains the fixed FULL zero-query fraction, independently of P5.
        p6 = tests[tests.primary_test_id == "P6"]
        self.assertTrue(np.allclose(p6.mean_difference, .1))
        # A supplied but missing per-seed reference cannot silently switch to
        # the global FULL mean, even though 50 gap scores remain available.
        frame.loc[frame.seed == 0, "paired_FULL_mean_ANN_minus_IR_budget_gap"] = np.nan
        missing = density_paired_tests(frame, bootstrap_resamples=20, signflip_resamples=20)
        self.assertTrue(missing[missing.primary_test_id == "P5"].p_value.isna().all())
        self.assertTrue((missing[missing.primary_test_id == "P5"].thinning_seeds_evaluated == 49).all())
        frame.loc[frame.seed == 0, "paired_FULL_mean_ANN_minus_IR_budget_gap"] = 100.
        frame.loc[frame.seed == 0, "paired_gap_query_count"] = 0
        missing = density_paired_tests(frame, bootstrap_resamples=20, signflip_resamples=20)
        self.assertTrue(missing[missing.primary_test_id == "P5"].p_value.isna().all())
        self.assertTrue((missing[missing.primary_test_id == "P5"].thinning_seeds_evaluated == 49).all())
        # Presence of only one matching field is also insufficient evidence.
        missing = density_paired_tests(frame.drop(columns="paired_gap_query_count"), bootstrap_resamples=20, signflip_resamples=20)
        self.assertTrue(missing[missing.primary_test_id == "P5"].mean_difference.isna().all())


if __name__ == "__main__":
    unittest.main()
