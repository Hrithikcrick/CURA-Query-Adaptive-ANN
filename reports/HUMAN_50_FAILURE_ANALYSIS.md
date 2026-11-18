# Human-verified FiQA failure analysis

50 actual fixed-HNSW low-budget IR failures were manually analyzed.
These are fixed efSearch=10 outcomes, not 50 deployed CURA-policy failures. Evidence and initial suggestions were model prepared; the researcher explicitly confirmed or changed the displayed category and evidence note. Preserved earlier human decisions and subsequent researcher-chat confirmations are recorded in the decision log.

| Category | Count | Share |
|---|---:|---:|
| RELEVANT_DOCUMENT_DROPPED | 36 | 72.00% |
| LOW_ANN_RECALL_AND_IR_FAILURE | 11 | 22.00% |
| HIGH_ANN_RECALL_BUT_IR_FAILURE | 2 | 4.00% |
| RANKING_BOUNDARY_INSTABILITY | 1 | 2.00% |
| AMBIGUOUS_QUERY | 0 | 0.00% |
| RARE_OR_SPECIALIZED_ENTITY | 0 | 0.00% |
| LONG_OR_COMPLEX_QUERY | 0 | 0.00% |
| DISTRIBUTION_OR_DOMAIN_EFFECT | 0 | 0.00% |
| OTHER | 0 | 0.00% |

## Representative evidence

- RELEVANT_DOCUMENT_DROPPED: query 10845 (Rationale behind using 12, 26 and 9 to calculate MACD). Researcher note: The exact search retrieved the relevant MACD explanation at rank 1, but HNSW with efSearch=10 missed it from the top 10. This caused nDCG@10 to fall from 1.0 to 0.0. The small search budget may have limited retrieval of the relevant neighbor.
- LOW_ANN_RECALL_AND_IR_FAILURE: query 3830 (US citizen transferring money to Indian fiance to buy property). Researcher note: ANN Recall@10 is 0.8, while nDCG@10 falls from 0.3562 to zero. The displayed top-five excerpts do not locate the lost relevant result or establish the underlying search-path cause.
- HIGH_ANN_RECALL_BUT_IR_FAILURE: query 2400 (Will I be paid dividends if I own shares?). Researcher note: ANN Recall@10 of 0.9 coexists with nDCG loss of 0.4585. Relevant document 1198 ranks first under exact search but is absent from the approximate top five. The underlying search-path cause is unverified.
- RANKING_BOUNDARY_INSTABILITY: query 11088 (Am I required to have a lawyer create / oversee creation of my will?). Researcher note: Relevant document 437100 ranks second under exact search but is absent from the approximate top five. Approximate nDCG@10 is zero with ANN Recall@10 of 0.2. No boundary gap is recorded, so a cutoff-instability mechanism is not established by the available evidence.

2/50 selected failures have ANN Recall@10 >=0.90; 48/50 have lower recall. These deliberately selected cases show coexistence patterns, not population prevalence or policy failure rates.
Coding is qualitative and may admit competing labels. Truncated passage evidence and sparse relevance judgments limit causal interpretation; no inter-annotator agreement or causal effect is inferred.
