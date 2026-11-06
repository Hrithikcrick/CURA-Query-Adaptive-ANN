# CURA: Cross-domain evidence and limitations (3 November 2026)

**Project:** Beyond Recall: Calibrated Utility-Risk Allocation (CURA) for Query-Specific Approximate Nearest Neighbor Search  
**Course:** COL764 — Information Retrieval and Web Search, IIT Delhi

> This is an evidence-based summary of saved CSV outputs. It is **not** a fresh training/indexing rerun, a proof of risk guarantees, or a claim of official reproduction of recent adaptive-HNSW systems.

## Main research question

Can a calibrated, query-specific HNSW search budget reduce total search computations relative to a fixed policy while keeping downstream ranking-utility degradation within a chosen risk constraint?

### MS MARCO: positive principal experiment

- Dataset: **1,000,629 passages**, 200 held-out MS MARCO test queries.
- Comparison: Fixed-IR HNSW (`ef=96`) vs CURA-Full.
- Total realized search work (distance computations/query, including adaptive-probe overhead): **3,431.770 → 2,727.965** = **20.5085% reduction**.
- Mean search latency: **0.8328545 → 0.6907383 ms**, a **17.06%** mean reduction under recorded conditions.
- p95 latency moves in the opposite direction: **1.128008 → 1.985027 ms** (Fixed-IR → CURA-Full). Thus the saved run supports an average-latency improvement but **not** a tail-latency improvement.
- Mean nDCG@10: **0.6252604 → 0.6233629**.
- Downstream IR-violation rate: **0.5% fixed**, **1.0% CURA**, both observed rates below the nominal 5% target.
- Paired mean search-work difference (fixed minus CURA): **703.805**, bootstrap 95% CI **[401.434, 984.115]**, paired work randomization `p≈5.0e-5`.
- Work improvement is supported by the paired test. This does **not** establish a statistically significant improvement in ranking quality or a formal distribution-free guarantee for every deployment setting.
- TREC-DL 2019 OOD evaluation uses the MS MARCO passage collection but a shifted query set (**43 queries**): CURA-Full recorded **12.39%** total-work savings relative to Fixed-IR, with observed downstream violations of **2.33%** versus **0%** for Fixed-IR. Do not confuse this with independent-domain transfer.

**Primary saved artifacts**
- [Final ablation table](../msmarco_large/msmarco_1000000_q1000/final_ablation_package/final_ablation_table.csv)
- [Paired significance tests](../msmarco_large/msmarco_1000000_q1000/final_ablation_package/pairwise_cura_tests.csv)
- [CURA controller code](../07_cura_controller.py)
- [Saved-results audit](../audit_cura_results.py)

### BEIR FiQA: negative second-domain experiment

- Separate financial-domain retrieval dataset: **57,638 documents**, **648** evaluation queries used for an HNSW sweep.
- FiQA within-domain controller experiment: **388 training queries**, **129 calibration**, **131 held-out**. This is **not** a zero-shot transfer of the MS MARCO CURA model.
- Model used a reduced feature set (query length and surface features plus inexpensive `ef=10` probe work/latency), not CURA-Full's rich partial-score features.
- Fixed-HNSW selected `ef=32` based on calibration; mean held-out work **1,097.084**.
- CURA-FiQA mean held-out work **1,922.870**; **75.27% more work** (negative saving).
- Fixed mean latency **0.21490 ms**, CURA **0.38564 ms**.
- Fixed mean nDCG@10 **0.22014**, CURA **0.21787**.
- **Both** policies had **4/131 = 3.05%** held-out IR violations, though they did not fail the exact same queries.
- **95% Wilson upper bound ~7.59%** for both held-out violation rates, so observed 3.05% is **not** a 95%-confidence guarantee of staying below 5%.
- For CURA-FiQA, the mean chosen search cost excluding extra probe was **1,471.55** distance computations/query, and the probe added **451.32**, yielding **1,922.87**. **51/131** queries received `ef>32`; **112/131** required a separate post-probe search.
- Both the fixed `ef=32` and CURA threshold satisfied the **calibration-set** Wilson criterion; their held-out Wilson upper bounds did not.
- Calibration and policy selection reuse the calibration query split: report this limitation and evaluate on fresh dev/test partitions for stronger guarantees.

**FiQA failure overlap:** 125 both safe, 2 both fail, 2 failed only under fixed HNSW, and 2 failed only under CURA-FiQA.

**FiQA saved artifacts**
- [Budget sweep](../beir_fiqa/fiqa_budget_summary.csv)
- [Held-out comparison](../beir_fiqa/fiqa_cura_controller/heldout_comparison.csv)
- [Controller protocol and findings](../beir_fiqa/fiqa_cura_controller/protocol_and_findings.json)
- [Failure audit](../beir_fiqa/fiqa_cura_controller/fiqa_failure_diagnostics_summary.json)
- [FiQA budget sweep implementation](../12_beir_fiqa_pilot.py)
- [FiQA within-domain controller](../13_fiqa_cura_controller.py)

## Reproducibility and interpretability boundaries

1. **No test-set optimization:** FiQA held-out predictions/results must remain frozen. The failure analysis of those cases is diagnostic; do not tune and then re-report performance on the same held-out cases.
2. **Training versus transfer:** FiQA used a newly trained and calibrated reduced-feature risk controller. Its failure does not, by itself, measure zero-shot MS MARCO→FiQA transfer of CURA-Full.
3. **Comparative baselines:** ANN-Risk is a local proxy, **not** official DARTH; Patience-Inspired is **not** an official reproduction of published Patience. The existing evidence should not claim superiority over untested official implementations.
4. **Latency and work:** Counts are from saved experiment outputs. CPU/software settings, warmup, caching, and paired end-to-end latency should be inspected and rerun for a systems-level reproducibility claim.
5. **MS MARCO model-assisted failure analysis:** A separate 50-case pass analyzes **actual IR-unsafe query-budget cases** from the MS MARCO HNSW sweep. Automatic structural validation passed for all 50 rows; **8 initially uncertain cases were resolved in a second model-assisted pass** using predefined decision rules. This is not independent human annotation and must not be described as "50 manually reviewed failures." See [MODEL_ASSISTED_50_FAILURE_STATUS.md](MODEL_ASSISTED_50_FAILURE_STATUS.md).
6. **FiQA manual failures:** The [50-case FiQA worksheet](../beir_fiqa/manual_failure_review/fiqa_50_actual_ef10_failures_TO_REVIEW.csv) consists of **true fixed-HNSW `ef=10` nDCG-loss violations**, not 50 deployed CURA failures. Reconstructed rankings matched the saved scores for all 50. [Assistant-prepared drafts](../beir_fiqa/manual_failure_review/ASSISTANT_DRAFT_ALL_50.md) are not verified human annotation; independent human review remains separate work.
7. **Dataset/artifact packaging:** Raw MS MARCO data, embeddings, FAISS indexes and other large artifacts were excluded from GitHub. A full-from-scratch reproduction, pinned versions/hashes, and independent data-validation check remain to be verified.

## Interpretation suitable for a COL764 final paper

**Supported:** CURA-Full shows a measurable, statistically supported reduction in total HNSW computation on the reported MS MARCO held-out set. A reduced-feature, separately fitted FiQA controller **fails** to improve efficiency on its held-out financial queries. This suggests that feature informativeness, calibration conservatism, workload characteristics, and probe-restart overhead are important design considerations.

**Not established:** A 5% distribution-free operational guarantee, general gains across unrelated domains, zero-shot transfer from MS MARCO to FiQA, or superiority over official recent adaptive search baselines.

## Highest-priority remaining work

1. Commit the exact locally generated MS MARCO 50-case taxonomy CSVs and, if the course requires independent manual review, complete and audit 50 human failure explanations with evidence.
2. Verify or faithfully reproduce a stronger published adaptive-search baseline; document any unavailable comparison.
3. Package exact commands and hashes to regenerate result CSVs; check dependencies and one clean environment.
4. Produce the short SIGIR-style final paper, limitations, transparent AI-use statement, and demo.
5. For publication-grade follow-up (not a precondition for the course): test continuation-based HNSW that reuses the probe frontier, richer features, and fresh held-out cross-domain evaluation with no test tuning.

**Result provenance:** all numeric claims in this note come from committed experiment CSV/JSON files linked above; no new model training was performed to write this document.
