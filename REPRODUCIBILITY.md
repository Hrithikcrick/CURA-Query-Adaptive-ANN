# CURA reproducibility audit

Audit date: 6 November 2026. Scope: existing source, configuration and saved results; no new encoding, index construction, retrieval sweep, training, package installation, or statistical resampling was performed.

**Human-review completion addendum, 18 November 2026:** the live FiQA checkpoint now records 50/50 explicit researcher category/note decisions, with zero pending. Cases 1-36 and their provenance were preserved; cases 37-50 were completed through researcher responses to displayed evidence. See the [review audit](reports/HUMAN_50_REVIEW_VALIDATION.json), [taxonomy](reports/HUMAN_50_FAILURE_TAXONOMY.csv), [analysis](reports/HUMAN_50_FAILURE_ANALYSIS.md) and [current status](reports/50_CASE_FAILURE_REVIEW_STATUS.md). Scientific SIGIR results and statistics were not recomputed. The command/environment findings below remain the historical 6 November audit.

**Status: PASS for the lightweight saved-artifact checks; PARTIAL for independent experimental reproduction.** The currently installed environment differs from the recorded experiment environment. A passing dependency check does not establish that a clean machine can regenerate the experiments or their timing results.

## Checks actually executed

Commands ran from the project root with the then-current `python` executable. Complete stdout, stderr, commands, exit codes, environment metadata, and before/after source hashes remain in the local historical record `reports/lightweight_check_logs/run_20261106T102303Z/command_results.json`. Those course-audit logs are outside this review-only Git update.

| Command | Result | Interpretation |
|---|---|---|
| `python audit_cura_results.py` | Exit 0 | Saved MS MARCO headline values and probe-accounting consistency passed. Its historical “Reviewed cases: 50” output describes an older diagnostic worksheet, including only three IR violations; it does not certify 50 human-reviewed failures. |
| `python 17_final_submission_audit.py` | Exit 0 | Historical 6 November snapshot: saved MS MARCO/FiQA checks passed and 50/50 reconstructions matched; then only 1/50 reviews had all required human fields, so that generated snapshot retained `submission_ready=false`. Current human completion is 50/50 in the separate addendum audit. |
| `python -m pip check` | Exit 0: `No broken requirements found.` | Installed distribution dependencies are consistent according to pip. This does not compare the environment to the historical requirements snapshot. |
| Runtime imports of `faiss`, `numpy`, `pandas`, `sklearn` | Exit 0 | Imported versions match the current distribution metadata below. No model was instantiated or trained. |

Thirty-nine watched source/result files were SHA-256 checked before and after the first three commands; all bytes were unchanged. Script 17 intentionally refreshed only its generated [audit JSON](reports/final_submission_audit.json); the previous report was preserved in `reports/course_completion_snapshot/reports__final_submission_audit.json`. No raw experimental CSV was overwritten. The logs do not claim independent verification of every stored number.

## Current environment versus recorded environment

The historical distribution inventory and build-tool discovery remain in the local record `reports/lightweight_check_logs/run_20261106T102303Z/environment.json`. Package versions in that inventory come from installed distribution metadata. The four runtime imports were also tested separately.

| Item | Current audit environment | Recorded experiment / requirements snapshot |
|---|---|---|
| Python | 3.12.10, MSC v.1943, AMD64 | 3.13.7, MSC v.1944 in `cura_protocol.json` and research protocol |
| Operating system | Windows-11-10.0.26300-SP0 | CURA protocol: Windows-11-10.0.26300-SP0; earlier validation protocol: Windows-11-10.0.26200-SP0 |
| Processor | AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD | Same processor description in research validation protocol |
| FAISS / `faiss-cpu` | 1.15.1, runtime import passed | 1.15.1 in both protocol and `requirements.txt` |
| NumPy | 2.5.3, runtime import passed | 2.3.3 in research protocol and requirements |
| pandas | 3.1.0, runtime import passed | 2.3.2 in requirements |
| scikit-learn | 1.9.1, runtime import passed | 1.7.2 in requirements |
| SciPy | 1.18.1, distribution metadata | 1.16.2 in requirements |
| sentence-transformers | 6.2.0, distribution metadata | 6.0.1 in requirements |
| transformers | 5.22.1, distribution metadata | 5.17.0 in requirements |
| PyTorch | 2.14.0, distribution metadata | 2.8.0 in requirements |
| BEIR | 2.2.0 | 2.2.0 in requirements |
| ir_datasets | 0.6.3 | 0.6.3 in requirements |

The recorded audit interpreter was Windows CPython 3.12.10. Reproduction commands use `python`; machine-specific installation paths are not prerequisites.

The file [requirements.txt](requirements.txt) is preserved as the recorded environment snapshot. It was not silently rewritten to match the installed environment, and installation from it was not tested in a fresh virtual environment. CMake, Java, javac, pdfLaTeX, latexmk and Tectonic were not found on PATH; MinGW `make` was available. Missing tools were not installed.

## Encoder and retrieval configuration

Both [MS MARCO preparation](04_msmarco_large_pilot.py) and [FiQA preparation](12_beir_fiqa_pilot.py) name `sentence-transformers/msmarco-MiniLM-L6-cos-v5`. The source normalizes embeddings and uses inner-product similarity. The encoder name is known; an immutable Hugging Face revision/commit and an independent hash of the exact encoder weights were not found in the recorded protocols. Consequently the model name alone is not an exact model-version lock. No encoder download was attempted during this audit.

| Configuration | Recorded value | Source |
|---|---|---|
| Random seed | 42 | CURA, research-validation and FiQA protocols; source defaults |
| HNSW `M` | 32 | Research/FiQA protocols and controller source |
| `efConstruction` | 200 | Research/FiQA protocols |
| Top-10 candidate `efSearch` grid | 10, 16, 24, 32, 48, 64, 96, 128, 192, 256 | CURA and research protocols |
| Cheap probe budget | `efSearch=10` | CURA protocol |
| Top-100 validation grid | 100, 150, 200, 300, 400, 600, 800 | Research protocol; separate from the top-10 controller grid |
| Unsafe IR event | nDCG@10 loss relative to the exhaustive dense reference greater than 0.025 | CURA/FiQA protocols |
| ANN-recall target | 0.90 | CURA/FiQA protocols |
| Nominal downstream-risk target | 0.05 | CURA and FiQA controller protocols |
| MS MARCO query split | 600 train / 200 calibration / 200 held-out test | `query_split.json` and CURA protocol |
| TREC-DL-2019 shifted set | 43 queries, evaluated separately | CURA/research protocols |
| Selected MS MARCO corpus | 1,000,629 passages; 1,000 dev queries plus 43 TREC queries | Existing large-pilot summary |
| Fixed-IR operating budget | `efSearch=96` | CURA protocol |
| CURA-Full selected risk threshold | 0.009000000000000001 | CURA protocol |
| Query-time FAISS threads | 1 | Research protocol; FiQA source also sets one thread |
| Research timing repetitions | 5 | Research protocol |
| FiQA corpus / judged-query set | 57,638 documents / 648 queries | FiQA protocol |
| FiQA query split | 388 train / 129 calibration / 131 held-out | FiQA controller protocol |
| FiQA fixed operating budget | `efSearch=32` | FiQA controller protocol |

The FiQA controller is separately fitted with reduced features. Its split repurposes the dataset's official test queries for development; it is not untouched official-test evaluation or zero-shot MS MARCO-to-FiQA transfer. Probability calibration and policy selection reuse the calibration split in the current design. Observed rates and Wilson bounds do not establish a universal 5% risk guarantee.

## Additional cached score-gap baseline

The course-finalization pass also produced a local score-gap budget baseline from existing score/work/timing CSVs, recorded under `msmarco_large/msmarco_1000000_q1000/score_threshold_baseline/` in `SCORE_THRESHOLD_BASELINE_REPORT.md` and `protocol.json`. These historical course artifacts are outside this review-only Git update. That protocol records training-only threshold candidates, train/calibration selection, a frozen policy before held-out/OOD evaluation, seed 42, 5,000 paired-bootstrap replicates and 20,000 randomization permutations (randomization seed 1042). This cached calculation did not reconstruct or search an index. The policy stops at the ef=10 probe when `probe_gap_1_10 >= 0.2663841247558594`; otherwise it uses ef=96 and charges both searches. Held-out mean work is 3266.29, compared with CURA-Full 2727.965. It is a two-stage budget simulation, not candidate-level traversal early exit. Compatible Recall@100 remains unavailable and is blank. The recorded probe/sweep equivalence assumption and pointwise calibration-selection limitation remain part of the protocol.

## Statistical and timing provenance

The final [paired comparison table](msmarco_large/msmarco_1000000_q1000/final_ablation_package/pairwise_cura_tests.csv) reports Fixed-IR minus CURA-Full work difference 703.805, bootstrap 95% interval [401.43425, 984.1146249999999], and paired randomization p=4.999750012499375e-05 over 200 matched held-out queries. These values were read and checked, not newly resampled.

The [CURA protocol](msmarco_large/msmarco_1000000_q1000/cura_controller/cura_protocol.json) records 5,000 bootstrap samples and 10,000 permutations for that earlier controller stage. [Final ablation source](09_final_ablation_package.py) defaults to 10,000 bootstrap samples, 20,000 permutations and seed 42; its saved randomization p-value is consistent with the corresponding add-one Monte Carlo resolution. A full raw command/transcript proving the final-stage invocation is not available, so source defaults must not be represented as a separately recovered execution log. Different historical statistical tables must retain their own provenance.

Use the corrected final ablation table for realized adaptive costs. Earlier `cura_controller/final_method_comparison.csv` has lower adaptive-cost values; the local historical metric audit (`FINAL_METRIC_AUDIT.txt`) discloses those discrepancies rather than silently substituting them. The final accounting includes the extra probe when a separate larger-budget search is used. The local Patience-Inspired ladder counts all visited searches.

Saved latency is search timing reconstructed from per-query measurements, including the accounted probe/search operations. p95 is across those query estimates; it is not a measured end-to-end request-latency percentile. Feature processing, predictor inference and serving overhead are not separately timed. CURA's mean latency improves while its p95 worsens; reproducing the saved CSV aggregates does not reproduce hardware timing.

## Lightweight verification commands

From the workspace root:

```powershell
python audit_cura_results.py
python 17_final_submission_audit.py
python -m pip check
# Write a new audit copy instead of replacing the root audit outputs:
python 19_final_metric_audit.py --output-dir reports/lightweight_demo_metric_audit
```

The first and third commands are read-only checks; script 17 refreshes its audit JSON, and script 19 writes only the chosen derived audit outputs. Back up a previous generated audit if it must be retained. None performs ANN search, encoding or model training.

For the previously built submission snapshot, `python CURA_FINAL_SUBMISSION/code/replay_saved.py all` verifies its manifest and runs only the permitted saved-artifact operations in temporary staging. It does not alter workspace results. Its original low-confidence display is the initial model snapshot; the final CSV and automatic-resolution log show the completed second model-assisted pass. All 50 MS MARCO final annotations remain model-assisted and are not substituted for human reviews. The separate live FiQA checkpoint now contains 50/50 genuine human category/note decisions with matching log evidence; the old submission snapshot was not rebuilt by this review continuation.

## Expensive regeneration commands — documented, NOT run

The following commands may download data, generate embeddings, build/search indexes, retrain models, or overwrite historical outputs. They are **EXPENSIVE** and are not demo or saved-result-verification commands. Use an isolated fresh workspace and inspect each script's prerequisites/output paths first.

```powershell
# EXPENSIVE: MS MARCO corpus, encoder, exhaustive reference and HNSW preparation.
python 04_msmarco_large_pilot.py --n_docs 1000000 --n_dev_queries 1000 --seed 42
# EXPENSIVE: searches the saved index; requires absent MS MARCO cache artifacts.
python 05_research_grade_validation.py --n_docs 1000000 --n_dev_queries 1000 --seed 42
# Requires query embeddings/index/probe inputs; trains controllers and writes outputs.
python 07_cura_controller.py --n_docs 1000000 --n_dev_queries 1000 --seed 42
# Requires candidate-identity searches or their cache; not official Patience.
python 08_patience_style_baseline.py --n_docs 1000000 --n_dev_queries 1000
# Resamples saved selected policies and regenerates final tables; not run in this audit.
python 09_final_ablation_package.py --n_docs 1000000 --n_dev_queries 1000 --bootstrap 10000 --permutations 20000 --seed 42
# EXPENSIVE: FiQA download/encoding when needed, and fresh exact/HNSW index builds.
python 12_beir_fiqa_pilot.py
# FiQA controller fitting; also requires raw query text in data/beir.
python 13_fiqa_cura_controller.py
```

Do not rerun `15_fiqa_50_failure_review.py` on an annotated worksheet: it regenerates that worksheet. Use the human-review workflow without replacing its source evidence.

## Artifact inventory and limits

Important source/result paths:

- [Final ablation results](msmarco_large/msmarco_1000000_q1000/final_ablation_package/final_ablation_table.csv), [paired tests](msmarco_large/msmarco_1000000_q1000/final_ablation_package/pairwise_cura_tests.csv), and `cura_controller/selected_*.csv` back the policy comparisons.
- [Calibration metrics](msmarco_large/msmarco_1000000_q1000/cura_controller/calibration_metrics.csv), [query split](msmarco_large/msmarco_1000000_q1000/cura_controller/query_split.json), [CURA protocol](msmarco_large/msmarco_1000000_q1000/cura_controller/cura_protocol.json), and [research protocol](msmarco_large/msmarco_1000000_q1000/research_grade_validation/experiment_protocol.json) record configuration and split evidence.
- [Top-10 sweep](msmarco_large/msmarco_1000000_q1000/research_grade_validation/top10_query_budget_results.csv) and [query metadata](msmarco_large/msmarco_1000000_q1000/results/queries.csv) establish provenance for the selected 50-case MS MARCO analysis.
- [FiQA protocol](beir_fiqa/fiqa_protocol.json), [held-out comparison](beir_fiqa/fiqa_cura_controller/heldout_comparison.csv), and [controller protocol](beir_fiqa/fiqa_cura_controller/protocol_and_findings.json) record the independent financial-domain evaluation.
- [FiQA evidence worksheet](beir_fiqa/manual_failure_review/fiqa_50_actual_ef10_failures_TO_REVIEW.csv) contains actual low-budget fixed-HNSW failure evidence; genuine human completion is checked independently of assistant drafts.
- [Current human-review checkpoint](beir_fiqa/manual_failure_review/fiqa_50_actual_failures_HUMAN_VERIFIED.csv) and [decision log](beir_fiqa/manual_failure_review/fiqa_50_human_review_decisions.json) contain all 50 complete human category/note decisions. Run `python sigir_analysis/audit_human_review_completion.py` for the lightweight provenance/evidence audit; it performs no retrieval or statistical resampling.

The lightweight package excludes raw corpora/downloads, `.npy`/`.npz` embeddings, `.faiss` indexes, model weights, temporary directories and caches. Current local `data/beir` contains regenerable FiQA raw files and an embedding NPZ, but those are not part of the lightweight artifact. The required MS MARCO query embeddings, exact-neighbor arrays and HNSW index are not supplied by the lightweight package. Saved result CSVs cannot reconstruct a traversal trace or a live index. Encoder revision pinning, a clean locked environment, full experiment transcripts, independent fresh regeneration and timing verification remain reproducibility extensions.
