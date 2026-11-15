# Fixed-budget risk/work frontier

**Post-hoc finding:** fixed ef=48 is the cheapest setting by measured mean work with observed held-out risk <= CURA-Full: 1.00% versus 1.00%. It uses 1956.35 distances/query versus 2727.97, 28.29% less work (paired query bootstrap 95% CI [20.07, 35.39]%).

This is a retrospective comparison on test outcomes, not model selection or an independently validated fixed policy. Observed-risk matching is not a population risk guarantee or an equivalence test. The original calibration-selected Fixed-IR remains ef=96.

| method | query_count | IR_violations | IR_violation_rate | wilson95_low | wilson95_upper | ANN_Recall_at_10 | mean_nDCG_at_10 | mean_distance_computations | mean_latency_ms | p95_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed ef=10 | 200 | 13 | 0.065 | 0.038376 | 0.10802 | 0.856 | 0.5858 | 707.685 | 0.256018 | 0.386299 |
| Fixed ef=16 | 200 | 7 | 0.035 | 0.0170553 | 0.0704715 | 0.909 | 0.608904 | 912.27 | 0.294119 | 0.423734 |
| Fixed ef=24 | 200 | 3 | 0.015 | 0.00511415 | 0.0431665 | 0.944 | 0.620581 | 1182.79 | 0.262981 | 0.356027 |
| Fixed ef=32 | 200 | 3 | 0.015 | 0.00511415 | 0.0431665 | 0.956 | 0.620325 | 1434.17 | 0.331733 | 0.433522 |
| Fixed ef=48 | 200 | 2 | 0.01 | 0.0027466 | 0.0357225 | 0.9715 | 0.623479 | 1956.35 | 0.517112 | 0.689866 |
| Fixed ef=64 | 200 | 1 | 0.005 | 0.000883146 | 0.0277744 | 0.9775 | 0.62526 | 2448.64 | 0.64171 | 0.828086 |
| Fixed ef=96 | 200 | 1 | 0.005 | 0.000883146 | 0.0277744 | 0.984 | 0.62526 | 3431.77 | 0.832854 | 1.12801 |
| Fixed ef=128 | 200 | 0 | 0 | 0 | 0.018846 | 0.9885 | 0.62697 | 4383.53 | 1.01614 | 1.3103 |
| Fixed ef=192 | 200 | 0 | 0 | 0 | 0.018846 | 0.993 | 0.62697 | 6217.09 | 1.46033 | 1.85536 |
| Fixed ef=256 | 200 | 0 | 0 | 0 | 0.018846 | 0.9935 | 0.62697 | 8005.64 | 1.9286 | 2.50285 |
| CURA-Full | 200 | 2 | 0.01 | 0.0027466 | 0.0357225 | 0.9595 | 0.623363 | 2727.97 | 0.690738 | 1.98503 |

## Why ef=96 was selected

The protocol selected the smallest fixed ef with a two-sided 95% Wilson interval upper endpoint <= 0.05 on the 200 calibration queries. At ef=64 there were 4/200 violations (2% observed), but the Wilson upper endpoint was 0.0502878655, just above the SLA. At ef=96 there was 1/200 (0.5%), upper endpoint 0.0277743999. ef=48 had 5/200, upper endpoint 0.0571791442. Therefore ef=96 was the first eligible candidate. These values were recomputed and checked against fixed_budget_calibration.csv, rather than using the test frontier to recalibrate. The confidence interval is a marginal interval, not a simultaneous or distribution-free guarantee after sweeping settings.

Latency p95 is across saved per-query means; CURA sums saved probe and selected-search means and excludes controller overhead. Work includes the probe. Risk intervals use z=1.96 to reproduce the existing protocol.
