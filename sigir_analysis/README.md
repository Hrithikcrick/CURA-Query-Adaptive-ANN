# Completed SIGIR evidence

Current evidence: original rankings **RECOVERED**, FULL reproduction **PASS**, 50-seed same-ranking study **COMPLETE**, statistics **COMPLETE**, density **Outcome A**. [Completion status](SIGIR_COMPLETION_STATUS.txt) and [validation](validation.json) are the current records. Earlier missing-input audits are historical snapshots superseded by external archive recovery.

CURA saves 20.51% recorded MS MARCO work versus calibration-selected ef96; exact-positive queries contribute 92.59% of that saving. However, post-hoc ef48 matches its observed 2/200 violations using 28.29% less work. The implementation does not dominate the observed fixed frontier. [Research report](../reports/SIGIR_RESEARCH_ANALYSIS.md) preserves both findings.

Thinning increases **measured** ANN-minus-IR safe-budget headroom: -0.140 (FULL), 0.308 (50%), 0.848 (25%), 1.956 (one positive). One-positive exact nDCG is zero in 82.88% of query-seed cases. This is judged-utility sensitivity, not improved retrieval or true relevance preservation.

## Evidence index

| Evidence | Location |
| --- | --- |
| Preserved original analyses | [input protocol](analysis_protocol.json), [six CSV hashes](preserved_analysis_hashes.json), [strata](exact_ndcg_stratified_results.csv), [fixed frontier](fixed_risk_work_frontier.csv), [oracle analysis](oracle_budget_gap_analysis.csv) |
| Original exact / HNSW ranked IDs | [exact](trec_exact_rankings.csv), [HNSW](trec_hnsw_rankings.csv), [provenance](trec_ranking_provenance.json), [final candidate audit](ranking_recovery_final_audit.json), [final retrieval-state audit](trec_retrieval_state_final_audit.json) |
| FULL: 43 exact + 430 HNSW comparisons | [comparisons](trec_retrieval_reproduction.csv), [gate](recovered_ranking_validation.json), [status](trec_retrieval_reproduction_status.md) |
| 43 queries × 10 budgets × 151 condition-seed rows | [per-query](trec_judgment_density_per_query.csv), [per-seed](trec_judgment_density_per_seed.csv), [summary](trec_judgment_density_summary.csv) |
| Fixed work/risk and frozen CURA policy | [fixed](trec_fixed_risk_work_by_density.csv), [frontier alias](trec_judgment_density_frontier.csv), [CURA](trec_cura_by_density.csv) |
| Raw evaluation bundle | [manifest](restored_ranking_evaluation/run_manifest.json), [validation](restored_ranking_evaluation/validation.json), [promotion hashes](density_promotion_validation.json) |
| Four density endpoints × three comparisons | [all statistics](statistical_tests.csv), [density only](density_statistical_tests_completed.csv), [preservation](density_statistics_validation.json), [interpretation](statistical_analysis_summary.md) |
| Scientific interpretation | [conclusion](scientific_conclusion.json), [report](../reports/SIGIR_RESEARCH_ANALYSIS.md), [short-paper source](../paper/sigir_short.tex) |
| Human-review provenance | [50/50 genuine FiQA reviews, none pending](../reports/50_CASE_FAILURE_REVIEW_STATUS.md), [human taxonomy](../reports/HUMAN_50_FAILURE_TAXONOMY.csv), [qualitative analysis](../reports/HUMAN_50_FAILURE_ANALYSIS.md) |

Four figures: [gap](judgment_density_vs_budget_gap.png), [zero reference](judgment_density_vs_zero_ndcg.png), [IR budget](judgment_density_vs_ir_safe_budget.png), [risk/work](judgment_density_vs_risk_work.png).

## Validate the completed state

From the repository root:

```powershell
python sigir_analysis/validate_analysis.py
python sigir_analysis/completion_status.py
```

These commands check saved-input/result hashes, FULL comparisons, 151 qrel hashes, metric/work invariants, statistical preservation, review provenance, figures and syntax. They perform no retrieval or scientific resampling. The ordered map is excluded from Git; when locally present its hash is checked. Published ranked IDs and recorded gate evidence remain in Git. Raw qrels must be locally available at the path in qrels_recovery.json; the checksum-verified recovery script restores them if missing.

Explicit [.gitattributes](../.gitattributes) preserve the raw line endings of checksum-backed research and review files. The [portable staged-checkout check](checkout_validation.json) verifies the files with automatic platform conversion disabled. [Byte policies](checkout_byte_integrity.json) record the exact choices; the checkout check changes no scientific data or human decisions.

Do not rerun completed original analyses, thinning preparation, density scoring or statistics. Historical missing-input generators are guarded against overwriting completed evidence. Historical uncertainty uses paired queries; density inference uses 50 paired seed effects conditional on 43 fixed queries. CIs are unadjusted; Holm p-values cover 12 density follow-ups. P7 is an algebraic reversal of P5.

[Fresh reproduction instructions](RESTORED_RANKING_EVALUATION.md) document the external archive and isolated reproduction. Large vectors, indexes, map files, datasets and backup/submission ZIPs are excluded; provenance records hashes. No independently logged historical map checksum or immutable encoder revision exists. Archive lineage and FULL agreement establish consistency, not independent historical identity proof.

The draft remains uncompiled and unchecked against official ACM formatting. ANN-Risk is not official DARTH; Patience-Inspired is not official Patience. Full 8.8M MS MARCO, another encoder/ANN family and population-risk certification remain publication extensions.
