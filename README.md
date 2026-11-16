# CURA — Calibrated Utility-Risk Allocation for Query-Specific ANN Search

**Beyond Recall: Calibrated Utility-Risk Allocation for Query-Specific Approximate Nearest Neighbor Search**  
**COL764: Information Retrieval and Web Search · IIT Delhi**

CURA now asks: **When do downstream-utility-aware ANN search budgets provide genuine efficiency gains, and how are those gains affected by relevance-judgment completeness and risk calibration?** The repository contains Python implementations, saved evaluation tables, statistical tests, ablations, and diagnostic reviews.

> **SIGIR analysis finding (16 November 2026): fixed ef=48 matches CURA's observed 1% MS MARCO test violation rate and uses 28.29% less work.** The original 20.51% CURA saving is relative to calibration-selected ef=96; it does not establish an adaptive advantage on the full observed frontier. The fixed comparison is post hoc. Exact-positive queries contribute 92.59% of that original saving. Original TREC exact/HNSW ranked IDs were recovered from an external archive and passed all 473 FULL-qrel comparisons. The same-ranking study scored all 50 prepared thinning seeds: measured ANN-minus-IR budget headroom increases from -0.140 (FULL) to 1.956 (one positive), **Outcome A: apparent evaluation headroom**. The one-positive exact-zero fraction is 82.88%; this is not actual relevance preservation. No retrieval, embedding generation, index construction or controller retraining was run.

The active research direction is **When Do Utility-Aware ANN Search Budgets Really Save Work? The Role of Relevance-Judgment Completeness and Risk Calibration.** Read the [research report](reports/SIGIR_RESEARCH_ANALYSIS.md), [analysis artifacts](sigir_analysis/README.md), [revised direction](paper/SIGIR_REVISED_DIRECTION.md), and [new short-paper source](paper/sigir_short.tex). The course evidence below records historical calibrated-baseline comparisons and retains its original review status.

<!-- BEGIN CURA FINAL EVIDENCE STATUS -->
**Saved-review evidence:** all 50 actual MS MARCO/TREC budget-sweep failure labels are model assisted, including 8 automatic second-review resolutions. FiQA human review is separate: **36/50 completed, 14 pending**. Case 37 awaits evidence-note confirmation. See the [audited review status](reports/50_CASE_FAILURE_REVIEW_STATUS.md); model suggestions do not count as completed human reviews.

Mean search latency estimates decrease by 17.06%, while p95 increases from 1.128008 to 1.985027 ms (75.98% higher). Adaptive effort improves the recorded average and worsens the tail in this setting. Latency is reconstructed from saved per-query mean search timings, including probe and selected-search costs; the p95 is across those estimates. It excludes model-inference/feature-processing overhead and is not an end-to-end request-latency measurement.
<!-- END CURA FINAL EVIDENCE STATUS -->

<!-- BEGIN CURA COURSE COMPLETION UPDATE -->
## Current COL764 evidence and remaining human work

**Original course question:** can query-specific ANN work be allocated using downstream ranking-risk rather than exact-neighbor recovery alone? The active SIGIR question additionally tests judgment completeness and the full fixed-budget frontier, as described above.

The historical MS MARCO comparison is 20.51% less realized work and 17.06% lower mean search latency than calibration-selected Fixed-IR (ef=96). **A post-hoc fixed ef=48 beats CURA at the same observed risk; CURA's p95 is also 75.98% worse than ef=96.** The separately fitted FiQA controller's negative result and original ablations remain part of the evidence.

### Added score-gap budget comparison

The cached `Score-Gap-Budget` baseline stops after the efSearch=10 probe when its approximate top1-top10 score spread is at least 0.266384125, otherwise reruns efSearch=96. Its threshold/fallback use training and calibration only. Test/OOD remain frozen; all probe and rerun work is charged. This is a **cached two-stage budget simulation**, not an implementation of actual traversal early exit. No exact-reference gap is a deployable feature. Calibration selection reuses the pointwise Wilson criterion and does not establish a selection-adjusted risk guarantee.

| Cohort | Method | IR violations | Mean work | p95 work | Mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|
| MSMARCO_TEST | Score-Gap-Budget | 0.50% | 3266.290 | 5318.200 | 0.807623 | 1.317973 |
| MSMARCO_TEST | Fixed-IR | 0.50% | 3431.770 | 4485.100 | 0.832854 | 1.128008 |
| MSMARCO_TEST | CURA-Full | 1.00% | 2727.965 | 8532.650 | 0.690738 | 1.985027 |
| TREC_DL_2019_OOD | Score-Gap-Budget | 2.33% | 3133.442 | 4713.300 | 0.763193 | 1.102646 |
| TREC_DL_2019_OOD | Fixed-IR | 0.00% | 2721.884 | 4138.100 | 0.682038 | 0.953098 |
| TREC_DL_2019_OOD | CURA-Full | 2.33% | 2384.581 | 7034.100 | 0.581502 | 1.682058 |

CURA-Full uses 16.48% less test work than this policy: paired difference 538.325, bootstrap 95% CI [210.230, 852.786], paired-randomization p=0.001100. The score-gap policy has lower observed test violation rate and lower p95 latency than CURA. Work savings therefore do not establish uniform superiority. [Full metrics, MRR, protocol and limitations](msmarco_large/msmarco_1000000_q1000/score_threshold_baseline/SCORE_THRESHOLD_BASELINE_REPORT.md) are saved separately; the primary ablation table is unchanged. Compatible policy Recall@100 is unavailable and remains blank.

**FiQA manual review: 36/50 human-verified decisions; 14 remain.** Only explicit researcher decisions backed by the saved decision log count. The separate 50-case MS MARCO/TREC taxonomy remains entirely model-assisted; it is not substituted for the manual requirement.

ANN-Risk is a local ANN-target proxy, not official DARTH. Patience-Inspired is a local stability heuristic, not official Patience. LADR is not claimed. One HNSW family, one primary MiniLM encoder, a controlled million-scale corpus, calibration reuse, and the 43-query OOD sample remain limitations. AI assisted code/debugging, documentation, and preliminary qualitative suggestions; it did not independently supply human decisions.

Course evidence: [requirements](FINAL_PROJECT_REQUIREMENTS.md), [reproducibility](REPRODUCIBILITY.md), [demo and viva](DEMO_AND_VIVA.md), [course audit](reports/FINAL_COURSE_AUDIT.md), [metric audit](reports/FINAL_METRIC_AUDIT.md), and [human checkpoint](beir_fiqa/manual_failure_review/fiqa_50_actual_failures_HUMAN_VERIFIED.csv).

Lightweight commands:

```powershell
python 18_score_threshold_baseline.py
python 18_final_course_audit.py
python 25_update_course_documents.py --apply
```

The course remains incomplete while genuine human-review decisions are missing. These commands do not run encoding, index construction, or retrieval. See REPRODUCIBILITY.md for expensive regeneration commands and cached artifact boundaries.
<!-- END CURA COURSE COMPLETION UPDATE -->

## Principal results

These are the original calibration-selected-baseline results. The [complete fixed frontier](sigir_analysis/fixed_risk_work_frontier.csv) and [research report](reports/SIGIR_RESEARCH_ANALYSIS.md) govern the revised research claims.

### MS MARCO held-out test (200 queries, 1,000,629 passages)

| Metric | Fixed HNSW (Fixed-IR) | CURA-Full |
|---|---:|---:|
| Mean realized search computations / query ↓ | 3,431.77 | **2,727.97** |
| Mean latency (ms) ↓ | 0.8329 | **0.6907** |
| p95 latency (ms) ↓ | **1.1280** | 1.9850 |
| Mean nDCG@10 ↑ | **0.6253** | 0.6234 |
| Observed IR violation rate ↓ | **0.5%** | 1.0% |

**20.51% less realized search work** and **17.06% lower mean latency**. Realized work includes CURA's extra probe overhead. The paired search-work difference is **703.805** computations/query (bootstrap 95% CI **[401.43, 984.11]**, paired randomization `p≈5.0e-5`). **Tail-latency caveat:** CURA-Full has worse recorded p95 latency than Fixed-IR (**1.9850 ms vs 1.1280 ms**), so the current implementation improves average cost/latency but not the p95 tail.

The reported 1.0% risk rate is an observed outcome below a nominal 5% target, **not a universal guarantee**.

### BEIR FiQA-2018 (separate financial domain)

| Metric | Tuned fixed HNSW | Simplified CURA-FiQA |
|---|---:|---:|
| Held-out queries | 131 | 131 |
| Mean realized work / query ↓ | **1,097.08** | 1,922.87 |
| Mean latency (ms) ↓ | **0.2149** | 0.3856 |
| Mean nDCG@10 ↑ | **0.2201** | 0.2179 |
| Observed IR violation rate ↓ | 3.05% | 3.05% |

**Negative result:** simplified CURA-FiQA used **75.27% more** search work than the calibrated fixed-budget policy. It was **trained on FiQA queries** with fewer features than CURA-Full; this does **not** test zero-shot MS MARCO→FiQA model transfer. The held-out 95% Wilson upper bound for the violation rate is ~7.59%, above the nominal 5% threshold.

Detailed provenance, limitations, and comparisons are in **[CURA results and limitations](reports/CURA_RESULTS_AND_LIMITATIONS.md)**.

## Overleaf paper source

- [Active SIGIR-short research draft](paper/sigir_short.tex) and [revised direction](paper/SIGIR_REVISED_DIRECTION.md), including the negative fixed-frontier result and completed same-ranking judgment experiment.

- [Two-column LaTeX manuscript](paper/main.tex) — revised with FiQA results and honest manual-review status.
- [Overleaf import/compile instructions](paper/README.md) — upload as main.tex, use pdfLaTeX.

## Final-paper preparation

- [Evidence-grounded SIGIR-style paper draft](reports/CURA_SIGIR_STYLE_FINAL_DRAFT.md) — includes both MS MARCO positive findings and FiQA negative results; manual taxonomy and fresh-run verification pending.
- [Final saved-artifact audit script](17_final_submission_audit.py) — read-only checks of headline result tables, hashes, and local human-review progress. Execute with `python 17_final_submission_audit.py` and inspect `reports/final_submission_audit.json`.
- [Detailed results and limitations](reports/CURA_RESULTS_AND_LIMITATIONS.md) — source-backed research claim boundaries.
- [Final metric audit](reports/FINAL_METRIC_AUDIT.md) — checked headline MS MARCO/TREC numbers, including the p95 latency trade-off.

## Repository map

| File / folder | Purpose |
|---|---|
| [`01_baseline_hnsw.py`](01_baseline_hnsw.py) | SciFact pilot: exact vs HNSW retrieval |
| [`02_ir_utility_labels.py`](02_ir_utility_labels.py) | Downstream IR-quality labels |
| [`04_msmarco_large_pilot.py`](04_msmarco_large_pilot.py) | Large MS MARCO HNSW sweep |
| [`05_research_grade_validation.py`](05_research_grade_validation.py) | Query-level validation and statistics |
| [`06_domain_specific_validation.py`](06_domain_specific_validation.py) | Domain-specific analysis |
| [`07_cura_controller.py`](07_cura_controller.py) | CURA risk model, calibration and budget policy |
| [`08_patience_style_baseline.py`](08_patience_style_baseline.py) | Patience-inspired proxy baseline |
| [`09_final_ablation_package.py`](09_final_ablation_package.py) | Final method and ablation tables |
| [`10_manual_taxonomy_summary.py`](10_manual_taxonomy_summary.py) | Taxonomy reporting code |
| [`11_manual_review_interactive.py`](11_manual_review_interactive.py) | Earlier MS MARCO manual-review tool |
| [`12_beir_fiqa_pilot.py`](12_beir_fiqa_pilot.py) | FiQA exact/HNSW budget sweep |
| [`13_fiqa_cura_controller.py`](13_fiqa_cura_controller.py) | FiQA within-domain risk controller |
| [`14_fiqa_failure_audit.py`](14_fiqa_failure_audit.py) | Analyze FiQA work overhead and failure overlap |
| [`15_fiqa_50_failure_review.py`](15_fiqa_50_failure_review.py) | Generate evidence for 50 low-budget failures |
| [`16_interactive_fiqa_review.py`](16_interactive_fiqa_review.py) | Human-in-the-loop case reviews; no automatic labels |
| [`audit_cura_results.py`](audit_cura_results.py) | Verify saved MS MARCO numerical consistency |
| [`msmarco_large/`](msmarco_large/) | Committed MS MARCO results, figures and reports |
| [`beir_fiqa/`](beir_fiqa/) | FiQA per-query data, calibration and diagnostics |
| [`reports/`](reports/) | Research findings and limitations |

## Setup and verification

Tested run environment has not been independently reproduced from a clean checkout; steps below are starting points, not a certified one-command reproduction.

```powershell
git clone https://github.com/Hrithikcrick/CURA-Query-Adaptive-ANN.git
cd CURA-Query-Adaptive-ANN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Checks saved MS MARCO result tables, NOT a fresh experiment rerun.
python audit_cura_results.py
```

**FiQA sequence** (downloads BEIR FiQA, generates embeddings and builds a new FAISS index; requires network access, memory, and compute):

```powershell
python 12_beir_fiqa_pilot.py
python 13_fiqa_cura_controller.py
python 14_fiqa_failure_audit.py
# Only run once to generate the worksheet, or it will overwrite existing annotations:
python 15_fiqa_50_failure_review.py
python 16_interactive_fiqa_review.py --progress
python 16_interactive_fiqa_review.py --limit 5
```

**Do not rerun** `15_fiqa_50_failure_review.py` after adding human annotations; it overwrites the review worksheet. The first MS MARCO experiments require reconstructing downloaded raw data, embeddings and FAISS indexes that are intentionally not committed. Follow each script's configuration and verify its saved input prerequisites.

## Failure-analysis status

### MS MARCO actual IR-unsafe query-budget cases

A separate 50-case analysis now targets **actual IR-unsafe query-budget pairs** from the MS MARCO HNSW budget sweep (nDCG@10 loss > 0.025). The structured annotation pass completed automatic validation over all 50 rows. **Eight initially uncertain cases were resolved in a second model-assisted pass using predefined numerical/semantic decision rules.** This analysis is therefore **model-assisted, not independent human annotation**; no claim of 50 manually reviewed failures is made. The cases are query-budget failures from the sweep, not 50 failures of the deployed CURA operating policy.

A status note is available in [reports/MODEL_ASSISTED_50_FAILURE_STATUS.md](reports/MODEL_ASSISTED_50_FAILURE_STATUS.md). Exact generated taxonomy CSVs should be committed from the local finalization workspace before using category-count claims in a submission.

### FiQA human-review worksheet

- [50 FiQA fixed-HNSW ef=10 failure cases](beir_fiqa/manual_failure_review/fiqa_50_actual_ef10_failures_TO_REVIEW.csv) were identified from genuine recorded nDCG losses; reconstructed scores match saved data.
- [Assistant-prepared drafts](beir_fiqa/manual_failure_review/ASSISTANT_DRAFT_ALL_50.md) provide reference evidence, **not completed manual judgments**.
- [Human-review instructions](beir_fiqa/manual_failure_review/HOW_TO_REVIEW.md) and [interactive tool](16_interactive_fiqa_review.py) support actual annotation.
- The audited FiQA checkpoint records **36/50 complete human reviews**. Case 37 (query 515) has category 2 recorded and awaits evidence-note confirmation; cases 38?50 await category decisions.
- These human decisions remain separate from the 50 model-assisted MS MARCO/TREC budget-sweep annotations and their eight automatic second-pass resolutions.
- These FiQA cases are **fixed low-budget HNSW failures**, not 50 violations by a deployed CURA policy.

## Interpretation and current limitations

- **ANN-Risk** is a local proxy, **not an official DARTH reproduction**.
- **Patience-Inspired** is not an official implementation of published Patience.
- FiQA controller features, corpus and calibration regime differ from MS MARCO CURA-Full. Cross-domain *transfer* has not yet been established.
- Some calibrations use the same calibration split for probability fitting and policy choice. Fresh split validation is required for stronger operational risk guarantees.
- Full model-training, index-building, raw datasets, cached candidates and certain environment artifacts are **not** packaged in this GitHub repository. Source code and many result tables are available, but a clean independent regeneration remains to be verified.
- Raw data, huge embeddings, `*.npy`/`*.npz` arrays and `*.faiss` indexes are intentionally excluded.

## Final COL764 deliverables — ongoing

- [x] Research hypothesis and CURA implementation
- [x] Saved MS MARCO, TREC-DL and FiQA experiments
- [x] Query-level ablations and MS MARCO significance testing
- [x] Results and limitations summary
- [x] 50 **model-assisted** MS MARCO actual IR-unsafe case analyses with structural validation and final taxonomy, including eight automatic second-pass resolutions
- [ ] 50 **independently human-verified** failure analyses and aggregate taxonomy
- [ ] Independent clean-environment reproduction and experiment manifests
- [ ] Strong published-method baseline verification where practical
- [ ] Final SIGIR-style paper, transparent AI-use statement, presentation/demo

**AI-use transparency:** Assistant tools have helped draft explanatory documentation, code and preliminary failure-review notes. Automated notes must not be represented as independently completed human annotations. The final paper should explicitly describe where and how such assistance was used.
