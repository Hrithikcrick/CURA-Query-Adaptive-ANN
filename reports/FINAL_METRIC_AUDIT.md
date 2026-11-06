# CURA final metric audit

**Checked:** 6 November 2026  
**Source of truth:** `msmarco_large/msmarco_1000000_q1000/final_ablation_package/final_ablation_table.csv`

This note records the headline values directly from the committed result table. No experiment was rerun to create this audit.

## Held-out MS MARCO (200 queries)

| Method | IR violation | ANN violation | Mean ANN Recall@10 | Mean nDCG@10 | Mean work | p95 work | Mean latency (ms) | p95 latency (ms) | Work saving vs fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed-IR | 0.5% | 3.5% | 0.9840 | 0.625260 | 3431.770 | 4485.10 | 0.832854 | 1.128008 | 0.0% |
| ANN-Risk | 0.5% | 1.5% | 0.9865 | 0.625260 | 3930.615 | 10614.10 | 0.957955 | 2.475230 | -14.54% |
| CURA-Query | 0.0% | 2.0% | 0.9915 | 0.626970 | 7130.760 | 10348.75 | 1.715593 | 2.455460 | -107.79% |
| **CURA-Full** | **1.0%** | 10.0% | 0.9595 | 0.623363 | **2727.965** | 8532.65 | **0.690738** | 1.985027 | **20.51%** |
| Patience-Inspired | 0.5% | 5.0% | 0.9785 | 0.625260 | 8413.945 | 26404.20 | 2.169291 | 6.349966 | -145.18% |
| ANN-Oracle | 0.5% | 1.0% | 0.9590 | 0.626896 | 1208.020 | 3183.35 | 0.350603 | 0.808713 | 64.80% |
| IR-Oracle | 0.0% | 29.0% | 0.8895 | 0.630348 | 779.835 | 1275.90 | 0.269716 | 0.419549 | 77.28% |

### Verified headline claim

CURA-Full reduces mean realized HNSW distance computations from **3431.770** to **2727.965** per query, a **20.5085%** reduction relative to Fixed-IR.

The paired work reduction reported in the committed pairwise test table is:

- mean difference: **703.805** computations/query;
- bootstrap 95% CI: **[401.434, 984.115]**;
- paired randomization: **p ≈ 5.0×10^-5**.

### Tail-latency caveat

CURA-Full improves **mean latency** from **0.832854 ms** to **0.690738 ms** (about **17.06% lower**), but its **p95 latency worsens** from **1.128008 ms** to **1.985027 ms**. The current implementation therefore supports an average-efficiency improvement, not a p95 tail-latency improvement.

## TREC-DL-2019 OOD query set (43 queries)

| Method | IR violation | Mean ANN Recall@10 | Mean nDCG@10 | Mean work | Mean latency (ms) | p95 latency (ms) | Work saving vs fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-IR | 0.0% | 1.0000 | 0.630980 | 2721.884 | 0.682038 | 0.953098 | 0.0% |
| CURA-Full | 2.33% | 0.9930 | 0.627128 | 2384.581 | 0.581502 | 1.682058 | 12.39% |
| ANN-Risk | 4.65% | 0.9907 | 0.627630 | 1933.558 | 0.457745 | 2.124156 | 28.96% |
| CURA-Query | 0.0% | 1.0000 | 0.630980 | 5674.349 | 1.376989 | 2.171932 | -108.47% |
| Patience-Inspired | 0.0% | 1.0000 | 0.630980 | 4753.651 | 1.287862 | 2.831810 | -74.65% |

The OOD set is small and uses the same MS MARCO passage collection, so it should be treated as a query/judgment-regime shift rather than independent-domain transfer.

## Failure-analysis status

A separate MS MARCO 50-case analysis targets actual IR-unsafe query-budget pairs. Automatic structural validation passed for all 50 cases; 8 initially uncertain cases were resolved in a second model-assisted pass. This is **model-assisted diagnostic analysis, not independent human review**.

The separate FiQA human-review worksheet also remains distinct from this MS MARCO analysis.

## Claim boundaries

- ANN-Risk is not an official DARTH reproduction.
- Patience-Inspired is not an official Patience reproduction.
- The primary corpus is a controlled ~1M-passage subset, not full 8.8M MS MARCO.
- The primary system uses HNSW and one MiniLM encoder.
- The nominal 5% target is not a universal distribution-free guarantee.
- Model-assisted failure annotations must not be described as independent human judgments.
