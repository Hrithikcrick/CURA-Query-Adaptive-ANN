"""Synthetic tests for the new saved-data verifier; no artifact writes."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from false_safety_audit import audit_losses, ndcg_at10


class SavedFalseSafetyAuditTests(unittest.TestCase):
    def test_full_unsafe_is_the_conditional_denominator(self):
        result = audit_losses([.1, .1, 0, 0], [0, .1, 0, 0], [1, 1, 1, 1], .025)
        self.assertEqual(result['full_unsafe_pair_instances'], 2)
        self.assertEqual(result['false_safe_count'], 1)
        self.assertEqual(result['conditional_false_safe_rate'], .5)
        self.assertEqual(result['all_pair_false_safe_rate'], .25)

    def test_threshold_equality_is_safe_and_loss_is_nonnegative(self):
        result = audit_losses([.025, .026, .03], [.1, .025, -.2], [.1, .2, 0], .025)
        self.assertEqual(result['false_safe_count'], 2)
        self.assertEqual(result['false_unsafe_count'], 1)
        self.assertEqual(result['false_safe_positive_exact_count'], 1)
        self.assertEqual(result['false_safe_zero_exact_count'], 1)
        self.assertEqual(result['false_safe_positive_exact_fraction'], .5)

    def test_no_unsafe_denominator_is_missing_not_zero(self):
        result = audit_losses([0, 0], [0, 0], [0, 0], .025)
        self.assertIsNone(result['conditional_false_safe_rate'])
        self.assertIsNone(result['false_safe_positive_exact_fraction'])
        self.assertEqual(result['all_pair_false_safe_rate'], 0)

    def test_no_silent_missing_or_broadcasting(self):
        with self.assertRaises(ValueError):
            audit_losses([0], [0, 0], [0, 0], .025)
        with self.assertRaises(ValueError):
            audit_losses([np.nan], [0], [0], .025)
        with self.assertRaises(ValueError):
            audit_losses([0], [0], [0], -.1)

    def test_ndcg_uses_original_graded_global_ideal(self):
        self.assertAlmostEqual(ndcg_at10(['low'], {'low': 1, 'high': 3}),
                               1 / (7 + 1 / np.log2(3)))
        self.assertEqual(ndcg_at10(['unknown'], {'rel': 3}), 0)
        self.assertEqual(ndcg_at10(['rel'], {'rel': 3}), 1)

    def test_depth_ten_does_not_promote_eleventh_document(self):
        self.assertEqual(ndcg_at10([f'u{i}' for i in range(10)] + ['rel'], {'rel': 3}), 0)

    def test_empty_positive_reference_is_zero(self):
        self.assertEqual(ndcg_at10(['n'], {'n': 0}), 0)


if __name__ == '__main__':
    unittest.main()
