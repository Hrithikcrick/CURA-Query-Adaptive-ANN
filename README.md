# CURA — Calibrated Utility-Risk Allocation for Query-Specific ANN Search

**Beyond Recall: Calibrated Utility-Risk Allocation for Query-Specific Approximate Nearest Neighbor Search**  
**COL764: Information Retrieval and Web Search · IIT Delhi**

CURA studies whether HNSW's `efSearch` can be chosen **per query** according to predicted *downstream ranking-quality risk*, rather than requiring a single global ANN-recall target. The repository contains Python implementations, saved evaluation tables, statistical tests, ablations, and diagnostic reviews.

> **Research status:** Main MS MARCO evaluation shows an efficiency improvement. A simpler controller fitted within BEIR FiQA has a negative efficiency result. Both are reported transparently. This is a research artifact, not a production-ready calibrated service.

## Principal results

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
- The committed FiQA worksheet is still not evidence of 50 independent human reviews. Do not claim otherwise until the reviewed records themselves are committed and audited.
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
- [x] 50 **model-assisted** MS MARCO actual IR-unsafe case analyses with structural validation
- [ ] 50 **independently human-verified** failure analyses and aggregate taxonomy
- [ ] Independent clean-environment reproduction and experiment manifests
- [ ] Strong published-method baseline verification where practical
- [ ] Final SIGIR-style paper, transparent AI-use statement, presentation/demo

**AI-use transparency:** Assistant tools have helped draft explanatory documentation, code and preliminary failure-review notes. Automated notes must not be represented as independently completed human annotations. The final paper should explicitly describe where and how such assistance was used.
