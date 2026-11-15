# Statistical analysis scope

Saved-data comparisons P1?P4 and the other previously completed rows remain unchanged. The 42 completed rows were preserved field-for-field. The six blocked P5/P6 rows were replaced by real density results; no historical bootstrap or algorithm comparison was rerun.

Twelve FULL-versus-thinned comparisons are now complete: ANN-minus-IR gap (P5), exact-zero fraction (P6), IR-safe budget (P7), and IR-allows-lower rate (P8), each for 50%, 25%, and one-positive conditions. Each effect uses 50 paired seed observations conditional on the same 43 queries and recovered rankings. All queries remain jointly feasible. The 64,930 query-budget records are not treated as independent inferential observations.

P5/P6 use the existing evaluator's 20,000 seed resamples and exploratory paired sign-flip tests. P7 effects are exactly the negative P5 effects under invariant ANN budgets; confidence endpoints are negated and reversed, and two-sided significance is reused. P7 is algebraically redundant, not independent corroboration. P8 adds three new paired seed tests. Sign-flip inference assumes sign exchangeability.

Confidence intervals are unadjusted percentile intervals. Holm-adjusted p-values cover the 12 density follow-up comparisons, including dependent/redundant endpoints; they do not claim familywise control over historical tests. Seed uncertainty describes the specified thinning mechanism, not the population of queries or deployment risk.

All actual effects, intervals, raw and adjusted p-values are in [statistical_tests.csv](statistical_tests.csv) and [the density-only table](density_statistical_tests_completed.csv). [Validation](density_statistics_validation.json) records preservation and completion checks. The pre-extension saved table remains byte-identical. Overall requested statistical analysis: COMPLETE.
