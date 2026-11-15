# SIGIR research analysis

Start with [the main research report](../reports/SIGIR_RESEARCH_ANALYSIS.md) and [paper direction](../paper/SIGIR_REVISED_DIRECTION.md).

**Supported negative result:** post-hoc fixed ef48 uses 28.29% less work than CURA at the same observed MS MARCO violation count. **Density study blocked:** qrels are recovered and 50 paired judgment seeds are prepared, but rankings and original recovery caches remain missing. Zero ranking seeds were evaluated.

| Analysis | Artifacts |
| --- | --- |
| Exact-nDCG strata | [results](exact_ndcg_stratified_results.csv), [query records](exact_ndcg_stratified_per_query.csv), [summary](exact_ndcg_stratified_summary.md), [plot](exact_ndcg_stratified_work.png) |
| Fixed frontier | [test](fixed_risk_work_frontier.csv), [plot](fixed_risk_work_frontier.png), [calibration](calibration_fixed_frontier.csv), [interpretation](fixed_frontier_summary.md) |
| Oracle budget gap | [results](oracle_budget_gap_analysis.csv), [query records](oracle_budget_gap_per_query.csv), [summary](oracle_budget_gap_summary.md), [plot](oracle_budget_gap_analysis.png) |
| Statistics and provenance | [paired tests](statistical_tests.csv), [primary comparisons](primary_statistical_results.csv), [scope](statistical_analysis_summary.md), [validation](validation.json), [source protocol](analysis_protocol.json), [preserved numerical hashes](preserved_analysis_hashes.json) |
| Original qrels | [recovery](qrels_recovery.json), [query audit](trec_qrels_query_audit.csv), [original file](../local_data/trec2019/2019qrels-pass.txt) |
| Exhaustive local audit | [file inventory](local_artifact_inventory.csv), [content/schema audit](local_retrieval_audit.json), [archive members](archive_member_inventory.csv), [Git candidate review](git_candidate_review.json) |
| Judgment density: blocked | [summary](trec_judgment_density_summary.csv), [prepared seed metadata](trec_judgment_density_per_seed.csv), [retained judgments](trec_thinned_positive_judgments.jsonl.gz), [empty frontier schema](trec_judgment_density_frontier.csv), [protocol/status](trec_judgment_density_status.md), [missing-input paths](ranking_recovery_requirements.csv) |
| Historical TREC FULL | [fixed frontier](fixed_risk_work_frontier_trec_saved_full.csv), [plot](fixed_risk_work_frontier_trec_saved_full.png) |
| Recovery and completion gates | [ranking validation](trec_ranking_recovery_validation.json), [retrieval state](trec_retrieval_state.json), [completion report](COMPLETION_STATUS.txt), [short-paper gate](short_paper_evidence_gate.json) |

All density seed rows are explicitly evaluation_completed=False. Blank retrieval metrics mean unavailable; judgment counts and hashes are real preparation results. The four density plots are absent. Historical FULL metrics are not a new original-ranking validation or a completed density experiment. Full-study validation reports FAIL for unmet ranking/seed requirements; input and saved-data integrity checks pass separately. The source for the short paper remains unchanged until all required evidence is available.

Reproduce with NumPy, pandas, SciPy and Matplotlib (versions in analysis_protocol.json):

```powershell
python sigir_analysis/recover_trec_qrels.py
python sigir_analysis/audit_local_retrieval.py
python sigir_analysis/verify_trec_ranking_recovery.py
python sigir_analysis/analyze_saved_artifacts.py
python sigir_analysis/build_research_report.py
python sigir_analysis/validate_analysis.py
```

Recovery reuses verified local qrels. These commands never invoke retrieval, training, encoding or index construction. Source inputs and human reviews remain unchanged. The report builder updates the main research report, revised-direction Markdown and this index. The separately maintained standalone LaTeX source remains uncompiled.
