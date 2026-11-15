# SIGIR research analysis

**Research question:** When do downstream-utility-aware ANN search budgets provide genuine efficiency gains, and how are those gains affected by relevance-judgment completeness and risk calibration?

**Supported finding.** CURA beats the original calibration-selected ef96 baseline on recorded MS MARCO work, but post-hoc fixed ef=48 matches CURA's observed test violation count and uses **28.29% less work**. This implementation has not demonstrated an advantage on the observed fixed risk/work frontier. The post-hoc comparison is a diagnostic, not a prospectively selected policy or population-risk guarantee.

**Recovery state:** original TREC-DL-2019 passage qrels are recovered and validated for all 43 saved query IDs. Fifty deterministic qrel seeds for each stochastic condition are prepared. **Same-ranking density evaluation remains BLOCKED:** original exact/HNSW ranked IDs and the original vectors/index/document map are absent. Zero ranking seeds were evaluated. No ranking-dependent density estimate or plot was invented.

## Exhaustive local recovery audit

The audit enumerated **3468 files** recursively in the entire workspace, including hidden and ignored files. [local_artifact_inventory.csv](../sigir_analysis/local_artifact_inventory.csv) lists every file at the scan snapshot and classifies known runtime dependencies and Git internals separately. It inspected **789 file/member contents or schemas**, archive content, NumPy headers and all **656 local Git blobs**, including all **278 unreachable blobs**. Git parsing produced 853 content/member inspections and included refs, reflogs, nested archives and backup/code clues. No inspection errors or unresolved ranking candidates were recorded. The [before-recovery snapshot](../sigir_analysis/local_retrieval_audit_before_recovery.json) and [current audit](../sigir_analysis/local_retrieval_audit.json) preserve the evidence.

CSV/TSV contents, JSON/JSONL structures, notebook outputs, script and backup path clues, text qrel/run formats, binary cache filenames and ZIP/gzip/tar members were checked. Git JSON structures and code/backup clues were inspected alongside tabular candidates. No original retrieval pickle, parquet, .trec/.run, NumPy ranking array or FAISS/HNSW index was found. Pickles would be inspected without execution. Git candidates contain policy parameters/configuration IDs, completion flags, research metadata or manual FiQA fields, as recorded in the [candidate review](../sigir_analysis/git_candidate_review.json). Query-ID lists and scalar metrics cannot reconstruct document rankings.

The only original vector cache is `data/beir/fiqa_embeddings.npz`: 57,638 x 384 document vectors and 648 x 384 query vectors, with FiQA IDs. It belongs to a different corpus and cannot substitute for the million-scale MS MARCO cache. Phase-04/05 source sweeps retain `first_indices` in memory and export scalar metrics; corresponding saved ranking IDs were not recovered.

| Required artifact | Recovery result |
| --- | --- |
| Original TREC-DL-2019 passage qrels | **RECOVERED**, 9,260 judgments, 43 queries |
| Original 43 query identities | **AVAILABLE**, exact set match with qrels |
| Exact top10 original document IDs for 43 queries | **MISSING** |
| HNSW top10 original document IDs for 43 queries x 10 ef settings | **MISSING** |
| Original FAISS row-to-document mapping | **MISSING** |
| Original query vectors and corpus vectors | **MISSING** |
| Original built million-scale HNSW index | **MISSING** |

See [ranking_recovery_requirements.csv](../sigir_analysis/ranking_recovery_requirements.csv) for exact paths. If rankings are restored, no retrieval is needed. If only the original query vectors, index and document map are restored, rerun HNSW retrieval only for these 43 queries at ef 10,16,24,32,48,64,96,128,192,256. Recover exact rankings from original exact arrays or exhaustive search over the original cached corpus vectors, preserving original normalization, similarity, k, mapping and index settings. Do not regenerate vectors, train the controller or rebuild an index.

The mandatory [ranking recovery validation](../sigir_analysis/trec_ranking_recovery_validation.json) records missing exact/HNSW rankings, no retrieval rerun, no embedding/index/model rebuild and metric reproduction **NOT RUN** (`null`, not a passing value). Candidate cells and coverage were checked in 135 Git candidate records and the working-tree/archive schemas. The requested `queries_found=0` and `ef_values_found=[]` count verified document rankings; 43 saved query identities and ten scalar-metric ef settings remain available. Original parameters are recoverable: `sentence-transformers/msmarco-MiniLM-L6-cos-v5`, normalized float32 vectors, inner-product similarity, HNSW M=32, efConstruction=200, top10 search, one search thread, original FAISS 1.15.1. The [detailed retrieval-state record](../sigir_analysis/trec_retrieval_state.json) and [requested state audit](../sigir_analysis/trec_retrieval_state_audit.json) distinguish known configured settings from missing original vectors/index/map. Actual vector normalization, index metric and corpus identity/order cannot be verified without those artifacts. No normalized ranking CSV was created without valid data.

The [retrieval reproduction table](../sigir_analysis/trec_retrieval_reproduction.csv) records 61 archived comparison targets: exact mean nDCG and six metrics at each fixed HNSW ef. Recreated values and differences are blank, every comparison is incomplete, and no reproduction pass is claimed. The [reproduction status](../sigir_analysis/trec_retrieval_reproduction_status.md) explains the missing-state gate, prospective metric checks and why saved work/timing are not new rerun measurements. No alternative corpus, embedding generation, index rebuild or controller training was used.

## Implemented evaluation path for restored rankings

The [restored-ranking evaluator](../sigir_analysis/run_trec_judgment_density.py) now provides a separate gated path from valid original ranked IDs to all four density conditions, seed summaries/CIs, fixed frontiers, frozen CURA diagnostics and requested plots. Its [input gate](../sigir_analysis/ranking_input_gate.py) verifies original query/document identity evidence, source hashes and every FULL per-query nDCG/loss/ANN Recall before any thinning result is written. It stops on missing inputs or inconsistent reproduction. Frozen CURA source hashes and probe/search accounting are checked. [Recovery instructions](../sigir_analysis/RESTORED_RANKING_EVALUATION.md) specify schemas, provenance limitations and commands.

[Implementation validation](../sigir_analysis/evaluation_pipeline_validation.json) uses isolated synthetic fixtures, including all 50 seeds in a temporary synthetic workspace. These checks do not count as real TREC observations. Actual-input preflight remains BLOCKED and writes no experiment results. The current study status, unavailable density plots and scientific conclusions are unchanged; a tested implementation cannot replace missing original rankings.

## Original qrels recovery and identity checks

The original [NIST passage qrels](https://trec.nist.gov/data/deep/2019qrels-pass.txt), linked by the [TREC track source](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html), were recovered from the [ir-datasets checksum mirror](https://mirror.ir-datasets.com/2f4be390198da108f6845c822e5ada14) after direct NIST certificate verification failed. TLS verification was retained. The file is at `local_data/trec2019/2019qrels-pass.txt`.

- Bytes: **187,092**; judgments: **9,260**; distinct raw query IDs: **43**.
- MD5: `2f4be390198da108f6845c822e5ada14`; SHA256: `8a1f10d550732e4cd91d7fc49846a3784de4040972f583e69285a88f3c5fee92`.
- The expected URL, MD5 and byte size independently match the installed ir-datasets 0.6.3 download registry's `trec-dl-2019/qrels` entry.
- All raw IDs exactly match the saved `trec19::...` mapping.
- Every query's grade>0 judgment count matches saved `positive_qrel_count`.
- Grade counts: 0 = 5,158; 1 = 1,601; 2 = 1,804; 3 = 697.

See [qrels_recovery.json](../sigir_analysis/qrels_recovery.json) and the [per-query audit](../sigir_analysis/trec_qrels_query_audit.csv). These checks validate identity and judgment-count agreement; they do **not** verify FULL nDCG without ranked IDs. FULL means all available original pooled judgments, not complete relevance ground truth.

## Same-ranking TREC judgment-density study

The [summary](../sigir_analysis/trec_judgment_density_summary.csv) has four explicitly blocked rows. The [per-seed table](../sigir_analysis/trec_judgment_density_per_seed.csv) has one FULL row plus 150 stochastic rows, all `evaluation_completed=False`. Retained-judgment counts and qrel hashes are real preparation results. All ranking-dependent fields remain blank. The [frontier schema](../sigir_analysis/trec_judgment_density_frontier.csv) has zero observations.

| Condition | Qrel sets prepared | Ranking evaluations | Retained grade>0 judgments | Realized positive fraction |
| --- | ---: | ---: | ---: | ---: |
| FULL | 1 | 0 | 4102 | 1.000000 |
| 50_PERCENT_POSITIVES | 50 | 0 | 2063 | 0.502925 |
| 25_PERCENT_POSITIVES | 50 | 0 | 1040 | 0.253535 |
| ONE_POSITIVE_PER_QUERY | 50 | 0 | 43 | 0.010483 |

For seeds 0..49, each query gets one permutation of lexicographically sorted grade>0 original document IDs. PCG64 uses the first eight bytes of SHA256 of `f'{seed}\0{raw_qid}'`, interpreted big-endian. Retain nested prefixes of `max(1,ceil(0.5*n))`, `max(1,ceil(0.25*n))`, or one positive. Preserve retained grades and all nonpositive judgments; removed positives become unjudged. Grade 1 remains positive because the original graded nDCG code gives it positive gain, rather than substituting a binary-relevance convention.

The [compressed retained judgments](../sigir_analysis/trec_thinned_positive_judgments.jsonl.gz) contain 6,450 real query/seed/condition records. Nonpositive judgments reference the unchanged original qrels. The [protocol/status](../sigir_analysis/trec_judgment_density_status.md) records rounding, deterministic generation and invariants.

Once rankings are available, freeze query identities, exact and all ten HNSW rankings, measured work, thresholds and CURA choices. Validate FULL per-query nDCG and ANN Recall against archived metrics before thinning. For every condition compute exact/approximate nDCG, losses, exact-zero fraction, smallest IR-safe tested ef and feasibility, ANN-safe ef, ANN-minus-IR gap, lower-budget rate, all ten fixed risk/work points and frozen CURA risk/work. Flag no-safe-in-grid outcomes explicitly.

Match the original protocol: unjudged=0; gain `2**grade-1`; discount `log2(rank+2)`; IDCG from all retained query judgments, not only retrieved documents; IDCG<=0 gives nDCG0. Loss=`max(0,exact-approximate nDCG)`, violation iff loss>0.025. ANN Recall and ANN-safe ef must be identical per query/budget across conditions; work and policy choices are fixed. This invariance has not been newly verified from rankings because they are missing.

Report 95% CIs over seed-average effects conditional on these 43 queries, with paired query/seed FULL-minus-thinned changes. A crossed query-and-seed bootstrap can assess query-sampling sensitivity; never treat 43 x 50 cells or repeated FULL values as independent queries. Thinning may increase or decrease nDCG and violations because both DCG and IDCG change. Positive-only thinning models missing positive judgments, not every incompleteness mechanism.

**Unavailable outputs:** `judgment_density_vs_budget_gap.png`, `judgment_density_vs_zero_ndcg.png`, `judgment_density_vs_ir_safe_budget.png` and `judgment_density_vs_risk_work.png` are intentionally absent. No ranking reevaluation observations exist from which to plot them. The canonical [fixed-by-density table](../sigir_analysis/trec_fixed_risk_work_by_density.csv), [per-query table](../sigir_analysis/trec_judgment_density_per_query.csv) and [unchanged-policy CURA table](../sigir_analysis/trec_cura_by_density.csv) have headers and zero observations. The CURA evaluation label will be `unchanged CURA policy under alternate relevance-judgment evaluation`. No density effect, seed CI or paired significance result is claimed.

## Preserved exact-nDCG stratified analysis

The 200 held-out queries contain **41 exact-zero (20.5%)** and **159 exact-positive (79.5%)**. Versus calibration-selected ef96, positive-stratum CURA work saving is **23.09%** (paired bootstrap 95% CI [13.91, 31.58]%). Positive queries contribute 92.59% of total work reduction, a point estimate. Zero-stratum saving is 8.56% (CI [-15.97, 30.89]%).

Savings are **not concentrated in exact-zero queries**. This rejects the narrow zero-reference explanation, not sparse-judgment effects within positive queries. Exact-zero safety follows from one-sided nonnegative reference loss; it is not evidence of actual relevance preservation. In the positive group Fixed-IR has 1/159 violations and CURA 2/159; sparse discordances cannot establish equal risk. Complete quality, work, timing and CIs are in [exact_ndcg_stratified_results.csv](../sigir_analysis/exact_ndcg_stratified_results.csv), [summary](../sigir_analysis/exact_ndcg_stratified_summary.md) and [plot](../sigir_analysis/exact_ndcg_stratified_work.png).

## Preserved complete fixed risk/work frontier

| Method | Role | Violations | Mean distances/query | Mean nDCG@10 |
| --- | --- | ---: | ---: | ---: |
| Fixed ef=96 | Protocol-selected calibration baseline | 1/200 | 3431.770 | 0.625260 |
| CURA-Full | Frozen adaptive case study | 2/200 | 2727.965 | 0.623363 |
| Fixed ef=48 | Post-hoc matched observed-risk diagnostic | 2/200 | 1956.345 | 0.623479 |
| Fixed ef=64 | Post-hoc fixed frontier point | 1/200 | 2448.645 | 0.625260 |


Fixed ef=48 and CURA each have 2/200 held-out violations. Fixed costs **1956.345** distances/query versus CURA **2727.965**, **28.29% less** (paired percentile bootstrap CI [20.07, 35.39]%). They share 1 failed query, with 1 fixed-only and 1 CURA-only failure. Equal counts do not establish matching failures or equivalent quality.

Original ef96 was selected independently by calibration: ef64 had 4/200 violations and a two-sided 95% Wilson upper endpoint of 0.0502879, narrowly above 0.05; ef96 had 1/200 and upper 0.0277744. ef48 did not pass that original calibration rule. The post-hoc comparator diagnoses headroom and calibration sensitivity; it does not replace the original baseline or establish a new risk guarantee. Its bootstrap CI conditions on its selection and excludes frontier-selection uncertainty.

All ten fixed points, nDCG, ANN Recall, Wilson intervals, measured work and saved timing summaries remain in [fixed_risk_work_frontier.csv](../sigir_analysis/fixed_risk_work_frontier.csv), [plot](../sigir_analysis/fixed_risk_work_frontier.png) and the separate [calibration frontier](../sigir_analysis/calibration_fixed_frontier.csv). CURA work charges probe plus fresh selected search when needed. Saved latency combines measured search components, omits controller/feature overhead, and is not end-to-end latency.

For historical TREC FULL outcomes, fixed ef=32 matches CURA's 1/43 violations with **53.48% less work**. This is also post hoc, with only 43 queries, and provides no SLA guarantee. The [TREC archived FULL frontier](../sigir_analysis/fixed_risk_work_frontier_trec_saved_full.csv) is not a new ranking reevaluation.

## Oracle opportunity, statistics and revised interpretation

For 812 jointly feasible exact-positive MS MARCO queries, mean ANN-safe ef is 19.936, IR-safe ef 12.325, gap **7.611** (95% CI [5.933, 9.387]); IR permits lower ef on 25.62%. This is divergence between judged-IR and ANN criteria on the tested grid, not evidence that most queries benefit or that a controller efficiently exploits it. Saved-grid fallback and jointly feasible sensitivity results are both in [oracle_budget_gap_analysis.csv](../sigir_analysis/oracle_budget_gap_analysis.csv). Maximum-grid fallback does not demonstrate safety; ef reduction is not distance reduction.

All 43 TREC queries are exact-positive. The ANN-minus-IR mean ef gap is -0.140, CI [-1.163, 0.884]. Only 2/43 allow lower IR ef; 38 agree and 3 require higher IR ef. ANN is safe at ef10 on 36/43, IR on 35/43: measured headroom is restricted on the tested grid. Five nonzero gaps provide little inferential power. Cross-dataset judgment and budget patterns do not identify density causality; the blocked same-ranking experiment remains necessary.

[statistical_tests.csv](../sigir_analysis/statistical_tests.csv) preserves 20,000-resample paired-query bootstrap results, exploratory paired mean sign-flip tests, direction sign tests and exact McNemar comparisons. Sign-flip inference assumes sign exchangeability; p-values are unadjusted. Query bootstrap excludes policy-fitting, comparator-selection and timing-run uncertainty. Degenerate exact-zero intervals do not establish equivalence. No density-study statistical result is present.

### Requested primary statistical comparisons

| Test | Cohort | Estimator | Paired mean difference | 95% CI | Units |
| --- | --- | --- | ---: | --- | --- |
| P1 | MSMARCO_ALL all | saved_grid_censored_fallback | 9.158000 | [7.182000, 11.194000] | efSearch |
| P1 | MSMARCO_ALL all | both_criteria_feasible | 8.139253 | [6.577195, 9.792129] | efSearch |
| P2 | MSMARCO_ALL exact_positive | saved_grid_censored_fallback | 8.278388 | [6.222222, 10.527473] | efSearch |
| P2 | MSMARCO_ALL exact_positive | both_criteria_feasible | 7.610837 | [5.933498, 9.386700] | efSearch |
| P3 | MSMARCO_TEST | paired mean difference | 703.805000 | [408.594375, 985.821250] | distance computations/query |
| P4 | MSMARCO_TEST | paired mean difference | 771.620000 | [488.510000, 1078.553500] | distance computations/query |


P1 covers all MS MARCO oracle pairs; P2 covers exact-positive pairs. Each retains both grid-capped and jointly feasible estimators. P3 is original fixed ef96 minus CURA work on the same 200 test queries; P4 is CURA minus post-hoc fixed ef48. Positive values have the direction stated by each comparison. The updated P3 interval uses the recorded 20,000-resample bootstrap seed; historical 5,000-resample intervals remain historical.

The [primary results](../sigir_analysis/primary_statistical_results.csv) include effect sizes, CIs, units, seeds and exploratory p-values. Exact paired violation tests show only three discordances versus ef96 and two versus ef48, providing little evidence about risk equivalence. P5/P6 (FULL versus each thinned condition: gap and zero-reference fraction) are explicit blocked rows with blank effects/CIs/p-values. [Statistical scope](../sigir_analysis/statistical_analysis_summary.md) and [status](../sigir_analysis/statistical_analysis_status.json) distinguish completed saved-data tests from the incomplete full statistical analysis. The original statistical CSV is preserved as [statistical_tests_saved.csv](../sigir_analysis/statistical_tests_saved.csv).

**Scientific outcome: UNRESOLVED.** Outcome A (gap grows), B (gap unchanged) and C (mixed/nonmonotone) all require scored same-ranking observations. None can be selected from prepared qrels or cross-dataset scalar comparisons.

The defensible story is: **ANN and judged-IR safe budgets differ even for exact-positive queries; baseline calibration and adaptive overhead matter enough that cheap fixed search may be preferable to current CURA.** Genuine adaptive gains require an independently validated full-cost improvement over a fixed risk/work frontier. Judgment completeness remains an unresolved modifier. CURA does not universally beat fixed HNSW.

## Reproduction and validation

All research artifacts are under `sigir_analysis/`; this is the main report. The [revised paper direction](../paper/SIGIR_REVISED_DIRECTION.md) distinguishes supported and pending claims. The [short-paper source](../paper/sigir_short.tex) remains uncompiled and unchecked against the official ACM format. Course evidence and human-review records were retained.

```powershell
python sigir_analysis/recover_trec_qrels.py
python sigir_analysis/audit_local_retrieval.py
python sigir_analysis/verify_trec_ranking_recovery.py
python sigir_analysis/record_trec_reproduction.py
python sigir_analysis/analyze_saved_artifacts.py
python sigir_analysis/build_research_report.py
python sigir_analysis/validate_analysis.py
python sigir_analysis/completion_status.py
```

Recovery reuses the locally verified qrels; network is needed only if absent. The other commands do not run retrieval, generate embeddings, construct an index or train a controller. [analysis_protocol.json](../sigir_analysis/analysis_protocol.json) records source hashes and versions. [preserved_analysis_hashes.json](../sigir_analysis/preserved_analysis_hashes.json) verifies unchanged numerical artifacts after renaming; [validation.json](../sigir_analysis/validation.json) separates passing input/saved-data checks from **FAIL for the requested full-study requirements**. Missing rankings prevent same-ranking/ANN-invariance checks, FULL metric reproduction, duplicate-ranking checks and 50 completed seed evaluations. The validator returns exit code 1 for these unmet requirements; it does not mislabel available-input checks as completion of the experiment. Query-ID and normalized exact-text leakage checks pass across the original 600/200/200/43 split.

The short paper remains a supported-evidence draft with the density experiment explicitly pending. After saved-data validation, its sole source edit corrected the reproducibility directory to `sigir_analysis`; reverse-substitution reproduces the prior complete source hash. [The maintenance/evidence record](../sigir_analysis/short_paper_evidence_gate.json) documents this correction and prohibits new density claims before the experiment validates. No manuscript result was added or changed. The [requested completion report](../sigir_analysis/SIGIR_COMPLETION_STATUS.txt) records the remaining evidence gap. No commit or push was performed.

SIGIR RESEARCH COMPLETION STATUS

TREC qrels:
RECOVERED

Exact rankings:
MISSING

HNSW rankings:
MISSING

Retrieval-only rerun:
NO

Embeddings regenerated:
NO

Index rebuilt:
NO

Controller retrained:
NO

Exact-nDCG stratified analysis:
COMPLETE

Fixed risk/work frontier:
COMPLETE

Same-ranking judgment-density study:
BLOCKED

50-seed judgment thinning:
INCOMPLETE

Statistical analysis:
INCOMPLETE

Validation:
FAIL

Risk-matched fixed beats current CURA:
YES

Judgment sparsity materially changes ANN-vs-IR budget gap:
UNRESOLVED

Strongest positive result:
On 200 held-out MS MARCO queries, CURA uses 20.51% less recorded distance work than the original calibration-selected fixed ef96 baseline. Exact-positive-nDCG queries contribute 92.59% of that saving, so the reduction is not primarily an exact-zero artifact.

Strongest negative result:
Post-hoc fixed ef48 matches CURA's observed 2/200 violation rate with 28.29% less work. Fixed ef64 also has lower observed risk and work than CURA. The current implementation does not dominate the complete observed fixed frontier; retrospective risk matching is a diagnostic, not a prospectively calibrated replacement or proof of equal population risk.

Current novelty:
ANN-neighbor fidelity and judged downstream IR safety can require different budgets even when exact nDCG is positive. Exact-zero stratification separates a trivial reference-loss case from this remaining metric-dependent opportunity. The complete fixed frontier shows that conservative calibration and adaptive probe overhead can account for a saving against a selected baseline without establishing adaptive superiority. CURA is a case study of these limits; the causal effect of relevance-judgment completeness remains unresolved until the same rankings are evaluated under full and thinned qrels.

Remaining blocker:
Original exact top-k document IDs and HNSW document IDs for all 43 queries at all ten ef values are missing. The original subset document map, query/corpus vectors and built HNSW index needed for retrieval-only recovery are also missing; zero ranking seeds have been evaluated.

SIGIR-short readiness:
NEEDS MORE EVIDENCE
