# Same-ranking density study complete

Original exact/HNSW rankings were recovered directly from an external archive. All 473 FULL metric comparisons passed before scoring all 50 prepared thinning seeds. No retrieval, encoding, index construction or training was run.

Four conditions, 151 condition-seed summaries and 64,930 query-budget records are populated. All 43 queries remain jointly feasible. ANN recall, ANN-safe budgets and recorded search work/timing are invariant. Four density figures are present.

**Outcome A:** measured ANN-minus-IR budget gap increases from -0.140 (FULL) to 0.308 (50%), 0.848 (25%) and 1.956 (one positive). One-positive exact-zero fraction is 82.88%, making the reference-loss criterion vacuous there. This is measurement sensitivity under controlled thinning, not genuine relevance preservation or adaptive superiority.

[Summary](trec_judgment_density_summary.csv), [statistics](density_statistical_tests_completed.csv), [validation](validation.json), [recovery and reproducibility](RESTORED_RANKING_EVALUATION.md).
