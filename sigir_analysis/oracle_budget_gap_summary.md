# Why oracle savings exist

| cohort | exact_group | estimator | query_count | ANN_not_safe_in_grid | IR_not_safe_in_grid | mean_ANN_safe_ef | mean_IR_safe_ef | mean_ANN_minus_IR_gap | gap_ci95_low | gap_ci95_high | IR_allows_lower_rate | relative_oracle_budget_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSMARCO_ALL | all | saved_grid_censored_fallback | 1000 | 6 | 3 | 21.784 | 12.626 | 9.158 | 7.182 | 11.194 | 0.271 | 42.04 |
| MSMARCO_ALL | all | both_criteria_feasible | 991 | 6 | 3 | 20.0444 | 11.9051 | 8.13925 | 6.57719 | 9.79213 | 0.267407 | 40.6061 |
| MSMARCO_ALL | exact_zero | saved_grid_censored_fallback | 181 | 2 | 0 | 23.1381 | 10 | 13.1381 | 8.26519 | 18.7956 | 0.325967 | 56.7813 |
| MSMARCO_ALL | exact_zero | both_criteria_feasible | 179 | 2 | 0 | 20.5363 | 10 | 10.5363 | 6.94972 | 14.7709 | 0.318436 | 51.3058 |
| MSMARCO_ALL | exact_positive | saved_grid_censored_fallback | 819 | 4 | 3 | 21.4847 | 13.2063 | 8.27839 | 6.22222 | 10.5275 | 0.258852 | 38.5315 |
| MSMARCO_ALL | exact_positive | both_criteria_feasible | 812 | 4 | 3 | 19.936 | 12.3251 | 7.61084 | 5.9335 | 9.3867 | 0.256158 | 38.1764 |
| MSMARCO_TEST | all | saved_grid_censored_fallback | 200 | 2 | 0 | 22.5 | 11.51 | 10.99 | 6.67 | 16.23 | 0.29 | 48.8444 |
| MSMARCO_TEST | all | both_criteria_feasible | 198 | 2 | 0 | 20.1414 | 11.5253 | 8.61616 | 5.41414 | 12.4242 | 0.282828 | 42.7783 |
| MSMARCO_TEST | exact_zero | saved_grid_censored_fallback | 41 | 1 | 0 | 22.9756 | 10 | 12.9756 | 2.2439 | 28.2451 | 0.243902 | 56.4756 |
| MSMARCO_TEST | exact_zero | both_criteria_feasible | 40 | 1 | 0 | 17.15 | 10 | 7.15 | 1.4 | 15.1 | 0.225 | 41.691 |
| MSMARCO_TEST | exact_positive | saved_grid_censored_fallback | 159 | 1 | 0 | 22.3774 | 11.8994 | 10.478 | 5.98711 | 15.8113 | 0.301887 | 46.8241 |
| MSMARCO_TEST | exact_positive | both_criteria_feasible | 158 | 1 | 0 | 20.8987 | 11.9114 | 8.98734 | 5.32911 | 13.2915 | 0.297468 | 43.0042 |
| TREC_DL_2019 | all | saved_grid_censored_fallback | 43 | 0 | 0 | 12.2791 | 12.4186 | -0.139535 | -1.16279 | 0.883721 | 0.0465116 | -1.13636 |
| TREC_DL_2019 | all | both_criteria_feasible | 43 | 0 | 0 | 12.2791 | 12.4186 | -0.139535 | -1.16279 | 0.930233 | 0.0465116 | -1.13636 |
| TREC_DL_2019 | exact_zero | saved_grid_censored_fallback | 0 | 0 | 0 | NA | NA | NA | NA | NA | NA | NA |
| TREC_DL_2019 | exact_zero | both_criteria_feasible | 0 | 0 | 0 | NA | NA | NA | NA | NA | NA | NA |
| TREC_DL_2019 | exact_positive | saved_grid_censored_fallback | 43 | 0 | 0 | 12.2791 | 12.4186 | -0.139535 | -1.16279 | 0.883721 | 0.0465116 | -1.13636 |
| TREC_DL_2019 | exact_positive | both_criteria_feasible | 43 | 0 | 0 | 12.2791 | 12.4186 | -0.139535 | -1.16279 | 0.930233 | 0.0465116 | -1.13636 |

ANN safety reconstructs at least 90% of exact vector neighbors. IR safety preserves judged, rank-discounted gain within absolute nDCG loss 0.025. Missing an irrelevant or unjudged exact neighbor may reduce ANN recall without reducing the measured IR score; reordering or recovering a few judged positives can preserve utility. This mismatch can persist when exact nDCG is positive. When the exact reference is zero, nonnegative one-sided loss makes every ranking IR-safe by construction. These are mechanisms of the metric, not proof of actual relevance preservation.

The saved first-crossing oracle substitutes ef=256 if no safe candidate exists. We report those explicitly flagged, grid-censored legacy summaries for traceability, plus summaries restricted to pairs where both criteria are feasible. A fallback of 256 is not a demonstrated safe minimum. First-safe means any tested budget meeting the criterion, not safety at every larger budget; oracle_nonmonotonicity.csv records reversals. Budget reduction is 100*(1-mean IR ef/mean ANN ef); ef reduction is not distance-computation reduction.

The all-MS-MARCO analysis uses 1,000 queries, with a separate 200-query held-out sensitivity. TREC uses all 43 queries and has no exact-zero stratum; its empty row is NA, not a zero effect. Paired query bootstrap intervals use 20,000 resamples. Paired randomization p-values are exploratory; for TREC inspect the number of nonzero pairs and CI before interpreting them. TREC-vs-MS-MARCO differences also confound query distribution and corpus coverage, so they cannot identify a causal effect of qrel density. The same-ranking thinning experiment is needed.

Nonmonotonic first-safe reversals:

| source | IR_safe_then_unsafe | ANN_safe_then_unsafe |
| --- | --- | --- |
| msmarco-dev-small | 0 | 0 |
| trec-dl-2019 | 0 | 0 |
