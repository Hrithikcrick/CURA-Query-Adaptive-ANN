"""Nov 26 verification tests: first-safe semantics and saved FiQA integration."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1]/"fiqa_oracle_robustness.py"
SPEC = importlib.util.spec_from_file_location("fiqa_oracle_robustness", SCRIPT)
fiqa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fiqa)


def fixture():
    # q1 is safe, unsafe, safe: scanning all budgets must keep the first safe.
    # q2 has an approximate gain (negative signed loss) and no ANN-safe budget.
    source = pd.DataFrame([
        ["1", 10, .5, .5, 0, .9, 0, 0],
        ["1", 16, .5, .4, .1, .8, 1, 1],
        ["1", 24, .5, .5, 0, 1., 0, 0],
        ["2", 10, .5, .6, -.1, .8, 1, 0],
        ["2", 16, .5, .4, .1, .8, 1, 1],
        ["2", 24, .5, .4, .1, .8, 1, 1],
    ], columns=["query_id", "efSearch", "exact_nDCG10", "nDCG10", "nDCG_loss", "ANN_Recall10", "ANN_unsafe", "IR_unsafe"])
    saved = pd.DataFrame([["1", 10., 10.], ["2", np.nan, 10.]],
                         columns=["query_id", "ann_safe_ef", "ir_safe_ef"])
    return source, saved


class OracleSemanticsTests(unittest.TestCase):
    def test_nonmonotone_first_safe_signed_gain_and_infeasible(self):
        source, saved = fixture()
        out = fiqa.recompute_oracles(source, saved, ef_values=(10, 16, 24))
        self.assertEqual(out.loc["1", "ann_safe_ef"], 10)
        self.assertEqual(out.loc["1", "ir_safe_ef"], 10)
        self.assertTrue(np.isnan(out.loc["2", "ann_safe_ef"]))
        self.assertEqual(out.loc["2", "ir_safe_ef"], 10)

    def test_missing_budget_rejected(self):
        source, saved = fixture()
        with self.assertRaisesRegex(ValueError, "complete tested"):
            fiqa.recompute_oracles(source.iloc[1:], saved, ef_values=(10, 16, 24))

    def test_duplicate_pair_rejected(self):
        source, saved = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate query-budget"):
            fiqa.recompute_oracles(pd.concat([source, source.iloc[[0]]]), saved, ef_values=(10, 16, 24))

    def test_changed_saved_oracle_rejected(self):
        source, saved = fixture()
        saved.loc[0, "ann_safe_ef"] = 24
        with self.assertRaisesRegex(ValueError, "Saved oracle mismatch"):
            fiqa.recompute_oracles(source, saved, ef_values=(10, 16, 24))

    def test_changed_unsafe_flag_rejected(self):
        source, saved = fixture()
        source.loc[0, "IR_unsafe"] = 1
        with self.assertRaisesRegex(ValueError, "IR unsafe flags"):
            fiqa.recompute_oracles(source, saved, ef_values=(10, 16, 24))

    def test_changed_query_ids_rejected(self):
        source, saved = fixture()
        saved.loc[0, "query_id"] = "01"
        with self.assertRaisesRegex(ValueError, "query IDs differ"):
            fiqa.recompute_oracles(source, saved, ef_values=(10, 16, 24))

    def test_paired_bootstrap_deterministic_and_constant_shift(self):
        gaps = np.array([-8, 0, 6, 20.])
        first = np.array(fiqa.paired_query_ci(gaps, resamples=2000))
        second = np.array(fiqa.paired_query_ci(gaps, resamples=2000))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(fiqa.paired_query_ci(gaps+7, resamples=2000), first+7)

    def test_saved_fiqa_independent_reproduction(self):
        report = fiqa.verify(fiqa.ROOT/"beir_fiqa/fiqa_query_budget_results.csv",
                             fiqa.ROOT/"beir_fiqa/fiqa_oracle_budgets.csv")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(report["supplied_claim_checks"].values()))
        self.assertEqual(report["results"]["saved_oracle_rows_compared"], 648)
        self.assertEqual(report["results"]["excluded_exact_positive_query_ids"], ["4946", "5862"])


if __name__ == "__main__":
    unittest.main()
