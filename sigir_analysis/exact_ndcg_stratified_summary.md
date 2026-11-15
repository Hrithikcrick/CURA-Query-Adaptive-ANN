# Exact nDCG zero versus positive

| exact_group | method | query_count | group_percentage | mean_efSearch | IR_violation_rate | ANN_Recall_at_10 | mean_nDCG_at_10 | mean_distance_computations | mean_latency_ms | p95_latency_ms | CURA_work_saving_vs_fixed_pct | saving_ci95_low_pct | saving_ci95_high_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_positive | Fixed-IR | 159 | 79.5 | 96 | 0.00628931 | 0.984906 | 0.786491 | 3550.35 | 0.85532 | 1.12832 | 23.0874 | 13.9148 | 31.5833 |
| exact_positive | CURA-Full | 159 | 79.5 | 50.6667 | 0.0125786 | 0.957233 | 0.784104 | 2730.67 | 0.697819 | 1.98477 | 23.0874 | 13.9148 | 31.5833 |
| exact_zero | Fixed-IR | 41 | 20.5 | 96 | 0 | 0.980488 | 0 | 2971.9 | 0.745733 | 1.08602 | 8.56067 | -15.9716 | 30.8904 |
| exact_zero | CURA-Full | 41 | 20.5 | 57.0244 | 0 | 0.968293 | 0 | 2717.49 | 0.663279 | 1.97744 | 8.56067 | -15.9716 | 30.8904 |

Savings are not concentrated in exact-zero queries: exact-positive queries contribute 92.59% of the total fixed-minus-CURA distance reduction. This addresses the narrow zero-reference explanation, not the broader sparse-judgment confound. Exact-zero queries are automatically safe under the nonnegative one-sided loss max(0, exact nDCG minus approximate nDCG), regardless of ANN recall.

CIs use 20,000 paired within-group query resamples, percentile 95% intervals. Saving is 100 times (1 minus ratio of mean work), not the mean of per-query percentages. Both methods use saved effective distance computations, including the CURA probe and independent selected search when ef>10. Latencies are saved per-query search means; p95 is across those query means, not request-level tail latency. Model inference, feature processing and end-to-end overhead are absent.

Paired work, nDCG and violation tests are in statistical_tests.csv; sparse violation discordances cannot establish equivalent risk. All tests are exploratory and unadjusted. Query grouping uses reference scores only for retrospective analysis. Intervals condition on the saved policy; they exclude training/calibration uncertainty and timing-run uncertainty. The paired sign-flip test assumes sign exchangeability; the exact sign test separately assesses direction among non-ties.
