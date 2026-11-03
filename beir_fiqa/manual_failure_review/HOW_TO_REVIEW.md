# FiQA — manual review of 50 real low-budget HNSW failures

These are **true nDCG-loss violations at ef=10** in the saved FiQA sweep, NOT deployed CURA-FiQA failures.
The review form has **no human judgments filled in**. Inspect each query, the judged-relevant
passages, and the exact/HNSW ranked passages. Record your own evidence and explanation.

## Suggested categories (choose based on evidence, do not auto-fill)
- Relevant passage missed by ANN search
- Wrong ranking among retrieved passages
- Ambiguity, rare entity or abbreviation
- Several near-equivalent candidate passages
- Qrel/annotation or evaluation limitation
- Index/search approximation failure
- Other (describe specifically)

For each row, fill `human_failure_category`, `human_analysis_of_what_went_wrong`,
`human_evidence_doc_ids_or_ranks`, `human_suggested_remedy`, `reviewer`, and
`human_review_complete_YES_NO=YES` only after reviewing it.

If `reconstruction_matches_saved_metrics` is NO, investigate FAISS index reproducibility
before using the reconstructed top-5 examples as evidence. Original metric rows remain unchanged.

Do not reuse this analysis for model selection on the existing FiQA held-out partition.
