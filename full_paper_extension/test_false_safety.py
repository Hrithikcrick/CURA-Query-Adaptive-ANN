"""Synthetic fixtures: no research observations and no artifact writes."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from false_safety_metrics import (
    EFS, bpref, confusion_counts, evaluate, finite_minimum, holm, ndcg,
    optimism, rates, safe_labels, thin_query,
)
from analyze_false_safety import bootstrap_rows, crossed_bootstrap


class IncompleteJudgmentMetrics(unittest.TestCase):
    def test_condensation_removes_unjudged_but_keeps_judged_zero(self):
        self.assertAlmostEqual(ndcg(['u', 'r'], {'r': 2}, condensed=True), 1)
        self.assertLess(ndcg(['u', 'r'], {'u': 0, 'r': 2}, condensed=True), 1)
        # An available eleventh result must never be promoted in returned10.
        self.assertEqual(ndcg([f'u{i}' for i in range(10)] + ['r'], {'r': 2}, condensed=True), 0)

    def test_global_idcg_and_exponential_gain(self):
        self.assertAlmostEqual(ndcg(['low'], {'low': 1, 'high': 3}), 1 / (7 + 1 / np.log2(3)))

    def test_bpref_matches_formula_and_unreturned_relevant_denominator(self):
        qrels = {'r1': 2, 'r2': 3, 'n1': 0, 'n2': 1, 'n3': 0}
        self.assertAlmostEqual(bpref(['u', 'n1', 'r1', 'r2'], qrels), .5)
        self.assertAlmostEqual(bpref(['r1'], qrels), .5)
        # R>N: denominator is N, not R.
        self.assertAlmostEqual(bpref(['n1', 'r1'], {'n1': 0, 'r1': 2, 'r2': 2, 'r3': 2}), 0)
        self.assertTrue(np.isnan(bpref(['r'], {'r': 1, 'n': 0})))
        self.assertTrue(np.isnan(bpref(['r'], {'r': 2})))

    def test_thinning_is_nested_deterministic_deletion_with_original_grades(self):
        qrels = {str(i): (i % 3) + 1 for i in range(13)} | {'n': 0}
        a, b = thin_query(qrels, '001', 7), thin_query(qrels, '001', 7)
        self.assertEqual(a, b)
        sets = [set(a[c]) for c in a]
        self.assertTrue(sets[2] <= sets[1] <= sets[0])
        self.assertEqual([sum(g > 0 for g in a[c].values()) for c in a], [7, 4, 1])
        for judgments in a.values():
            self.assertEqual(judgments['n'], 0)
            self.assertTrue(all(qrels[d] == g for d, g in judgments.items()))
            self.assertTrue(all(g > 0 for d, g in judgments.items() if d != 'n'))

    def test_false_safety_both_denominators_and_abstentions(self):
        counts = confusion_counts(np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]))
        result = rates(counts)
        self.assertEqual(result['fsr_all'], .25)
        self.assertEqual(result['fsr_given_thin_safe'], .5)
        self.assertEqual(result['full_unsafe_sensitivity'], .5)
        missing = rates(confusion_counts(np.array([0, 1]), np.array([np.nan, 1])))
        self.assertTrue(np.isnan(missing['fsr_all']))
        self.assertEqual(missing['applicability_fraction'], .5)
        self.assertEqual(missing['fsr_all_possible_upper_bound'], .5)

    def test_nonmonotone_safety_negative_optimism_and_no_maxgrid_imputation(self):
        labels = np.zeros((3, len(EFS)))
        labels[0, 0] = 1
        labels[0, -1] = 0
        labels[1, 3] = 1
        minima = finite_minimum(labels)
        np.testing.assert_array_equal(minima[:2], [10, 32])
        self.assertTrue(np.isnan(minima[2]))
        differences = optimism(np.array([10, 32, np.nan]), np.array([32, 10, 256]))
        np.testing.assert_array_equal(differences[:2], [-22, 22])
        self.assertTrue(np.isnan(differences[2]))

    def test_zero_reference_is_structurally_safe_and_nan_is_not_safe(self):
        labels = safe_labels(np.array([0, np.nan]), np.array([[0, .3], [0, .3]]))
        np.testing.assert_array_equal(labels[0], [1, 1])
        self.assertTrue(np.isnan(labels[1]).all())

    def test_complete_grid_evaluation_and_binary_bpref_zero_positive(self):
        ranking = ['r', 'n'] + [f'u{i}' for i in range(8)]
        result = evaluate([ranking], [[ranking] * len(EFS)], [{'r': 1, 'n': 0}])
        self.assertEqual(result.labels.shape, (3, 1, 10))
        self.assertTrue((result.labels[:2] == 1).all())
        self.assertTrue(np.isnan(result.labels[2]).all())
        self.assertTrue(np.isnan(result.minimum[2]).all())

    def test_holm_keeps_missing_and_adjusts_actual_tests(self):
        adjusted = holm([.01, .04, np.nan, .03])
        np.testing.assert_allclose(adjusted[[0, 1, 3]], [.03, .06, .06])
        self.assertTrue(np.isnan(adjusted[2]))

    def test_seed_bootstrap_uses_ratio_of_counts_and_paired_condition_draws(self):
        rows = []
        # Seed0: one safe decision, one false-safe. Seed1: nine safe
        # decisions, none false-safe. Conditional FSR is1/10, not mean(1,0).
        for condition in ['A', 'B']:
            for seed, (full, thin) in enumerate([
                (np.array([0] + [1] * 9), np.array([1] + [0] * 9)),
                (np.array([0] + [1] * 9), np.array([0] + [1] * 9)),
            ]):
                rows.append(dict(condition=condition, seed=seed, **confusion_counts(full, thin)))
        summary = bootstrap_rows(pd.DataFrame(rows), ['condition'], 2000)
        np.testing.assert_allclose(summary.fsr_given_thin_safe, [.1, .1])
        np.testing.assert_allclose(summary.fsr_all, [.05, .05])
        # Identical outcomes under paired conditions must have identical CIs.
        self.assertEqual(summary.iloc[0].fsr_given_thin_safe_ci_low, summary.iloc[1].fsr_given_thin_safe_ci_low)
        self.assertEqual(summary.iloc[0].fsr_given_thin_safe_ci_high, summary.iloc[1].fsr_given_thin_safe_ci_high)

    def test_crossed_bootstrap_preserves_query_budget_clusters(self):
        # Two perfectly dependent budget blocks: one query is unsafe at all
        # ten budgets; the other is safe at all ten. Resampling query blocks
        # has support0,.5,1, not a narrow binomial CI from1000 pseudo-iid cells.
        full = np.tile(np.array([[0.] * 10, [1.] * 10])[None], (3, 1, 1))
        labels = {c: np.ones((50, 3, 2, 10)) for c in ['50_PERCENT_POSITIVES', '25_PERCENT_POSITIVES', 'ONE_POSITIVE_PER_QUERY']}
        collection = SimpleNamespace(year='2019', full=SimpleNamespace(labels=full))
        summary = crossed_bootstrap(collection, {'labels': labels}, 2000)
        selected = summary[(summary.metric == 'standard_ndcg_at10') & (summary.endpoint == 'fsr_all')]
        np.testing.assert_allclose(selected.ci_low, 0)
        np.testing.assert_allclose(selected.ci_high, 1)


if __name__ == '__main__':
    unittest.main()
