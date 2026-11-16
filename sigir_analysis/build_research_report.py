"""Build the neutral SIGIR report from preserved analyses and recovery evidence."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
QUESTION='When do downstream-utility-aware ANN search budgets provide genuine efficiency gains, and how are those gains affected by relevance-judgment completeness and risk calibration?'


def main():
    if json.loads((OUT/'ranking_recovery_status.json').read_text()).get('experiment_completed'):
        raise SystemExit('Completed evidence is already incorporated in reports/SIGIR_RESEARCH_ANALYSIS.md. Legacy missing-input report generation is disabled; run validate_analysis.py and completion_status.py.')
    data=json.loads((OUT/'analysis_results.json').read_text())
    state=json.loads((OUT/'ranking_recovery_status.json').read_text())
    recovery=json.loads((OUT/'qrels_recovery.json').read_text())
    audit=json.loads((OUT/'local_retrieval_audit.json').read_text())
    ranking_recovery=json.loads((OUT/'trec_ranking_recovery_validation.json').read_text())
    split=pd.read_csv(OUT/'exact_ndcg_stratified_results.csv')
    oracle=pd.read_csv(OUT/'oracle_budget_gap_analysis.csv')
    density=pd.read_csv(OUT/'trec_judgment_density_summary.csv')
    pos=split[(split.exact_group=='exact_positive')&(split.method=='CURA-Full')].iloc[0]
    zero=split[(split.exact_group=='exact_zero')&(split.method=='CURA-Full')].iloc[0]
    op=oracle[(oracle.cohort=='MSMARCO_ALL')&(oracle.exact_group=='exact_positive')&(oracle.estimator=='both_criteria_feasible')].iloc[0]
    tr=oracle[(oracle.cohort=='TREC_DL_2019')&(oracle.exact_group=='exact_positive')&(oracle.estimator=='saved_grid_censored_fallback')].iloc[0]
    fixed=data['fixed_frontier'];trec=data['trec_saved_full_frontier']
    primary=pd.read_csv(OUT/'primary_statistical_results.csv')
    primary_rows=primary[primary.primary_test_id.isin(['P1','P2','P3','P4'])]
    primary_table='| Test | Cohort | Estimator | Paired mean difference | 95% CI | Units |\n| --- | --- | --- | ---: | --- | --- |\n'
    for r in primary_rows.itertuples():
        primary_table+=f'| {r.primary_test_id} | {r.cohort} | {r.estimator} | {r.mean_difference:.6f} | [{r.ci95_low:.6f}, {r.ci95_high:.6f}] | {r.units} |\n'
    front=pd.read_csv(OUT/'fixed_risk_work_frontier.csv')
    selected_table='| Method | Role | Violations | Mean distances/query | Mean nDCG@10 |\n| --- | --- | ---: | ---: | ---: |\n'
    roles={'Fixed ef=96':'Protocol-selected calibration baseline','CURA-Full':'Frozen adaptive case study','Fixed ef=48':'Post-hoc matched observed-risk diagnostic','Fixed ef=64':'Post-hoc fixed frontier point'}
    for name,role in roles.items():
        r=front[front.method==name].iloc[0]
        selected_table+=f'| {name} | {role} | {int(r.IR_violations)}/200 | {r.mean_distance_computations:.3f} | {r.mean_nDCG_at_10:.6f} |\n'
    report=f'''# SIGIR research analysis

**Research question:** {QUESTION}

**Supported finding.** CURA beats the original calibration-selected ef96 baseline on recorded MS MARCO work, but post-hoc fixed ef={fixed['cheapest_fixed_ef']} matches CURA's observed test violation count and uses **{fixed['fixed_saving_vs_cura']['saving_pct']:.2f}% less work**. This implementation has not demonstrated an advantage on the observed fixed risk/work frontier. The post-hoc comparison is a diagnostic, not a prospectively selected policy or population-risk guarantee.

**Recovery state:** original TREC-DL-2019 passage qrels are recovered and validated for all 43 saved query IDs. Fifty deterministic qrel seeds for each stochastic condition are prepared. **Same-ranking density evaluation remains BLOCKED:** original exact/HNSW ranked IDs and the original vectors/index/document map are absent. Zero ranking seeds were evaluated. No ranking-dependent density estimate or plot was invented.

## Exhaustive local recovery audit

The audit enumerated **{audit['file_count']} files** recursively in the entire workspace, including hidden and ignored files. [local_artifact_inventory.csv](../sigir_analysis/local_artifact_inventory.csv) lists every file at the scan snapshot and classifies known runtime dependencies and Git internals separately. It inspected **{len(audit['content_inspections'])} file/member contents or schemas**, archive content, NumPy headers and all **{audit['git_blob_count']} local Git blobs**, including all **{audit['git_scope']['unreachable_blobs_inspected']} unreachable blobs**. Git parsing produced {len(audit['git_content_inspections'])} content/member inspections and included refs, reflogs, nested archives and backup/code clues. No inspection errors or unresolved ranking candidates were recorded. The [before-recovery snapshot](../sigir_analysis/local_retrieval_audit_before_recovery.json) and [current audit](../sigir_analysis/local_retrieval_audit.json) preserve the evidence.

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

The mandatory [ranking recovery validation](../sigir_analysis/trec_ranking_recovery_validation.json) records missing exact/HNSW rankings, no retrieval rerun, no embedding/index/model rebuild and metric reproduction **NOT RUN** (`null`, not a passing value). Candidate cells and coverage were checked in {ranking_recovery['git_candidate_blobs_reviewed']} Git candidate records and the working-tree/archive schemas. The requested `queries_found=0` and `ef_values_found=[]` count verified document rankings; 43 saved query identities and ten scalar-metric ef settings remain available. Original parameters are recoverable: `sentence-transformers/msmarco-MiniLM-L6-cos-v5`, normalized float32 vectors, inner-product similarity, HNSW M=32, efConstruction=200, top10 search, one search thread, original FAISS 1.15.1. The [detailed retrieval-state record](../sigir_analysis/trec_retrieval_state.json) and [requested state audit](../sigir_analysis/trec_retrieval_state_audit.json) distinguish known configured settings from missing original vectors/index/map. Actual vector normalization, index metric and corpus identity/order cannot be verified without those artifacts. No normalized ranking CSV was created without valid data.

The [retrieval reproduction table](../sigir_analysis/trec_retrieval_reproduction.csv) records 61 archived comparison targets: exact mean nDCG and six metrics at each fixed HNSW ef. Recreated values and differences are blank, every comparison is incomplete, and no reproduction pass is claimed. The [reproduction status](../sigir_analysis/trec_retrieval_reproduction_status.md) explains the missing-state gate, prospective metric checks and why saved work/timing are not new rerun measurements. No alternative corpus, embedding generation, index rebuild or controller training was used.

## Implemented evaluation path for restored rankings

The [restored-ranking evaluator](../sigir_analysis/run_trec_judgment_density.py) now provides a separate gated path from valid original ranked IDs to all four density conditions, seed summaries/CIs, fixed frontiers, frozen CURA diagnostics and requested plots. Its [input gate](../sigir_analysis/ranking_input_gate.py) verifies original query/document identity evidence, source hashes and every FULL per-query nDCG/loss/ANN Recall before any thinning result is written. It stops on missing inputs or inconsistent reproduction. Frozen CURA source hashes and probe/search accounting are checked. [Recovery instructions](../sigir_analysis/RESTORED_RANKING_EVALUATION.md) specify schemas, provenance limitations and commands.

[Implementation validation](../sigir_analysis/evaluation_pipeline_validation.json) uses isolated synthetic fixtures, including all 50 seeds in a temporary synthetic workspace. These checks do not count as real TREC observations. Actual-input preflight remains BLOCKED and writes no experiment results. The current study status, unavailable density plots and scientific conclusions are unchanged; a tested implementation cannot replace missing original rankings.

## Original qrels recovery and identity checks

The original [NIST passage qrels](https://trec.nist.gov/data/deep/2019qrels-pass.txt), linked by the [TREC track source](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html), were recovered from the [ir-datasets checksum mirror]({recovery['download_url']}) after direct NIST certificate verification failed. TLS verification was retained. The file is at `local_data/trec2019/2019qrels-pass.txt`.

- Bytes: **{recovery['bytes']:,}**; judgments: **{recovery['judgment_rows']:,}**; distinct raw query IDs: **43**.
- MD5: `{recovery['original_md5']}`; SHA256: `{recovery['sha256']}`.
- The expected URL, MD5 and byte size independently match the installed ir-datasets 0.6.3 download registry's `trec-dl-2019/qrels` entry.
- All raw IDs exactly match the saved `trec19::...` mapping.
- Every query's grade>0 judgment count matches saved `positive_qrel_count`.
- Grade counts: 0 = 5,158; 1 = 1,601; 2 = 1,804; 3 = 697.

See [qrels_recovery.json](../sigir_analysis/qrels_recovery.json) and the [per-query audit](../sigir_analysis/trec_qrels_query_audit.csv). These checks validate identity and judgment-count agreement; they do **not** verify FULL nDCG without ranked IDs. FULL means all available original pooled judgments, not complete relevance ground truth.

## Same-ranking TREC judgment-density study

The [summary](../sigir_analysis/trec_judgment_density_summary.csv) has four explicitly blocked rows. The [per-seed table](../sigir_analysis/trec_judgment_density_per_seed.csv) has one FULL row plus 150 stochastic rows, all `evaluation_completed=False`. Retained-judgment counts and qrel hashes are real preparation results. All ranking-dependent fields remain blank. The [frontier schema](../sigir_analysis/trec_judgment_density_frontier.csv) has zero observations.

| Condition | Qrel sets prepared | Ranking evaluations | Retained grade>0 judgments | Realized positive fraction |
| --- | ---: | ---: | ---: | ---: |
'''
    for r in density.itertuples():
        report+=f'| {r.condition} | {r.qrel_seeds_prepared} | 0 | {r.retained_positive_judgments} | {r.retained_positive_fraction:.6f} |\n'
    report+='''
For seeds 0..49, each query gets one permutation of lexicographically sorted grade>0 original document IDs. PCG64 uses the first eight bytes of SHA256 of `f'{seed}\\0{raw_qid}'`, interpreted big-endian. Retain nested prefixes of `max(1,ceil(0.5*n))`, `max(1,ceil(0.25*n))`, or one positive. Preserve retained grades and all nonpositive judgments; removed positives become unjudged. Grade 1 remains positive because the original graded nDCG code gives it positive gain, rather than substituting a binary-relevance convention.

The [compressed retained judgments](../sigir_analysis/trec_thinned_positive_judgments.jsonl.gz) contain 6,450 real query/seed/condition records. Nonpositive judgments reference the unchanged original qrels. The [protocol/status](../sigir_analysis/trec_judgment_density_status.md) records rounding, deterministic generation and invariants.

Once rankings are available, freeze query identities, exact and all ten HNSW rankings, measured work, thresholds and CURA choices. Validate FULL per-query nDCG and ANN Recall against archived metrics before thinning. For every condition compute exact/approximate nDCG, losses, exact-zero fraction, smallest IR-safe tested ef and feasibility, ANN-safe ef, ANN-minus-IR gap, lower-budget rate, all ten fixed risk/work points and frozen CURA risk/work. Flag no-safe-in-grid outcomes explicitly.

Match the original protocol: unjudged=0; gain `2**grade-1`; discount `log2(rank+2)`; IDCG from all retained query judgments, not only retrieved documents; IDCG<=0 gives nDCG0. Loss=`max(0,exact-approximate nDCG)`, violation iff loss>0.025. ANN Recall and ANN-safe ef must be identical per query/budget across conditions; work and policy choices are fixed. This invariance has not been newly verified from rankings because they are missing.

Report 95% CIs over seed-average effects conditional on these 43 queries, with paired query/seed FULL-minus-thinned changes. A crossed query-and-seed bootstrap can assess query-sampling sensitivity; never treat 43 x 50 cells or repeated FULL values as independent queries. Thinning may increase or decrease nDCG and violations because both DCG and IDCG change. Positive-only thinning models missing positive judgments, not every incompleteness mechanism.

**Unavailable outputs:** `judgment_density_vs_budget_gap.png`, `judgment_density_vs_zero_ndcg.png`, `judgment_density_vs_ir_safe_budget.png` and `judgment_density_vs_risk_work.png` are intentionally absent. No ranking reevaluation observations exist from which to plot them. The canonical [fixed-by-density table](../sigir_analysis/trec_fixed_risk_work_by_density.csv), [per-query table](../sigir_analysis/trec_judgment_density_per_query.csv) and [unchanged-policy CURA table](../sigir_analysis/trec_cura_by_density.csv) have headers and zero observations. The CURA evaluation label will be `unchanged CURA policy under alternate relevance-judgment evaluation`. No density effect, seed CI or paired significance result is claimed.

## Preserved exact-nDCG stratified analysis

'''
    report+=f'''The 200 held-out queries contain **{int(zero.query_count)} exact-zero ({zero.group_percentage:.1f}%)** and **{int(pos.query_count)} exact-positive ({pos.group_percentage:.1f}%)**. Versus calibration-selected ef96, positive-stratum CURA work saving is **{pos.CURA_work_saving_vs_fixed_pct:.2f}%** (paired bootstrap 95% CI [{pos.saving_ci95_low_pct:.2f}, {pos.saving_ci95_high_pct:.2f}]%). Positive queries contribute {pos.contribution_to_total_work_reduction_pct:.2f}% of total work reduction, a point estimate. Zero-stratum saving is {zero.CURA_work_saving_vs_fixed_pct:.2f}% (CI [{zero.saving_ci95_low_pct:.2f}, {zero.saving_ci95_high_pct:.2f}]%).

Savings are **not concentrated in exact-zero queries**. This rejects the narrow zero-reference explanation, not sparse-judgment effects within positive queries. Exact-zero safety follows from one-sided nonnegative reference loss; it is not evidence of actual relevance preservation. In the positive group Fixed-IR has 1/159 violations and CURA 2/159; sparse discordances cannot establish equal risk. Complete quality, work, timing and CIs are in [exact_ndcg_stratified_results.csv](../sigir_analysis/exact_ndcg_stratified_results.csv), [summary](../sigir_analysis/exact_ndcg_stratified_summary.md) and [plot](../sigir_analysis/exact_ndcg_stratified_work.png).

## Preserved complete fixed risk/work frontier

{selected_table}

Fixed ef={fixed['cheapest_fixed_ef']} and CURA each have 2/200 held-out violations. Fixed costs **{fixed['cheapest_fixed_mean_work']:.3f}** distances/query versus CURA **{fixed['cura_mean_work']:.3f}**, **{fixed['fixed_saving_vs_cura']['saving_pct']:.2f}% less** (paired percentile bootstrap CI [{fixed['fixed_saving_vs_cura']['saving_pct_ci'][0]:.2f}, {fixed['fixed_saving_vs_cura']['saving_pct_ci'][1]:.2f}]%). They share {fixed['shared_failure_queries']} failed query, with {fixed['fixed_only_failure_queries']} fixed-only and {fixed['cura_only_failure_queries']} CURA-only failure. Equal counts do not establish matching failures or equivalent quality.

Original ef96 was selected independently by calibration: ef64 had 4/200 violations and a two-sided 95% Wilson upper endpoint of 0.0502879, narrowly above 0.05; ef96 had 1/200 and upper 0.0277744. ef48 did not pass that original calibration rule. The post-hoc comparator diagnoses headroom and calibration sensitivity; it does not replace the original baseline or establish a new risk guarantee. Its bootstrap CI conditions on its selection and excludes frontier-selection uncertainty.

All ten fixed points, nDCG, ANN Recall, Wilson intervals, measured work and saved timing summaries remain in [fixed_risk_work_frontier.csv](../sigir_analysis/fixed_risk_work_frontier.csv), [plot](../sigir_analysis/fixed_risk_work_frontier.png) and the separate [calibration frontier](../sigir_analysis/calibration_fixed_frontier.csv). CURA work charges probe plus fresh selected search when needed. Saved latency combines measured search components, omits controller/feature overhead, and is not end-to-end latency.

For historical TREC FULL outcomes, fixed ef={trec['cheapest_fixed_ef']} matches CURA's 1/43 violations with **{trec['fixed_saving_vs_cura']['saving_pct']:.2f}% less work**. This is also post hoc, with only 43 queries, and provides no SLA guarantee. The [TREC archived FULL frontier](../sigir_analysis/fixed_risk_work_frontier_trec_saved_full.csv) is not a new ranking reevaluation.

## Oracle opportunity, statistics and revised interpretation

For {int(op.query_count)} jointly feasible exact-positive MS MARCO queries, mean ANN-safe ef is {op.mean_ANN_safe_ef:.3f}, IR-safe ef {op.mean_IR_safe_ef:.3f}, gap **{op.mean_ANN_minus_IR_gap:.3f}** (95% CI [{op.gap_ci95_low:.3f}, {op.gap_ci95_high:.3f}]); IR permits lower ef on {op.IR_allows_lower_rate:.2%}. This is divergence between judged-IR and ANN criteria on the tested grid, not evidence that most queries benefit or that a controller efficiently exploits it. Saved-grid fallback and jointly feasible sensitivity results are both in [oracle_budget_gap_analysis.csv](../sigir_analysis/oracle_budget_gap_analysis.csv). Maximum-grid fallback does not demonstrate safety; ef reduction is not distance reduction.

All 43 TREC queries are exact-positive. The ANN-minus-IR mean ef gap is {tr.mean_ANN_minus_IR_gap:.3f}, CI [{tr.gap_ci95_low:.3f}, {tr.gap_ci95_high:.3f}]. Only 2/43 allow lower IR ef; 38 agree and 3 require higher IR ef. ANN is safe at ef10 on 36/43, IR on 35/43: measured headroom is restricted on the tested grid. Five nonzero gaps provide little inferential power. Cross-dataset judgment and budget patterns do not identify density causality; the blocked same-ranking experiment remains necessary.

[statistical_tests.csv](../sigir_analysis/statistical_tests.csv) preserves 20,000-resample paired-query bootstrap results, exploratory paired mean sign-flip tests, direction sign tests and exact McNemar comparisons. Sign-flip inference assumes sign exchangeability; p-values are unadjusted. Query bootstrap excludes policy-fitting, comparator-selection and timing-run uncertainty. Degenerate exact-zero intervals do not establish equivalence. No density-study statistical result is present.

### Requested primary statistical comparisons

{primary_table}

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

## SIGIR ANALYSIS STATUS

Exact-nDCG stratified analysis: **COMPLETE**.

Fixed risk/work frontier: **COMPLETE**.

Same-ranking TREC judgment-density study: **BLOCKED**.

Reason if blocked: Exact and per-budget HNSW document rankings are absent. Original query/corpus vectors, document map and built index required for retrieval-only recovery are also absent. Qrels are recovered and 50 deterministic judgment seeds per stochastic condition are prepared; zero ranking seeds were evaluated.

Revised SIGIR story: **When utility-aware ANN budgets save work after accounting for judgment completeness and risk calibration. Preserve CURA's saving versus calibrated ef96, retain the cheaper post-hoc fixed result prominently, and leave density effects unclaimed until original rankings can be evaluated.**
'''
    from completion_status import build as build_completion
    completion=build_completion()
    report=report[:report.index('## SIGIR ANALYSIS STATUS')]+completion
    (ROOT/'reports/SIGIR_RESEARCH_ANALYSIS.md').write_text(report,encoding='utf-8')
    direction=f'''# Revised SIGIR research direction

**Research question:** {QUESTION}

**Supported finding:** CURA beats calibration-selected ef96 in recorded MS MARCO work; post-hoc fixed ef48 matches its observed 2/200 violations using {fixed['fixed_saving_vs_cura']['saving_pct']:.2f}% less work. This implementation has not demonstrated an advantage over the observed fixed risk/work frontier. ef48 remains a diagnostic, not a prospectively calibrated replacement.

1. **Metric-dependent oracle opportunity:** jointly feasible exact-positive MS MARCO queries retain an ANN-minus-IR gap of {op.mean_ANN_minus_IR_gap:.3f} ef (95% CI [{op.gap_ci95_low:.3f}, {op.gap_ci95_high:.3f}]). Preserve infeasible-grid flags and fallback sensitivity.
2. **Exact-zero explanation:** the 159 positive-reference test queries contribute {pos.contribution_to_total_work_reduction_pct:.2f}% of the original saving versus ef96. This rejects a purely zero-reference explanation, not all sparse-judgment effects.
3. **Judgment completeness remains pending:** original qrels are recovered, with 9,260 judgments and exact matches to saved query IDs and positive counts. FULL plus 50 paired seeds for each stochastic condition are prepared. Exact/HNSW rankings and recovery caches remain absent; **zero ranking seeds evaluated**. No density effect, CI or plot is claimed.
4. **Fixed frontier and overhead:** retain all ten fixed settings and distinguish calibration selection from retrospective matching. On TREC, fixed ef32 also matches CURA's observed 1/43 violations with {trec['fixed_saving_vs_cura']['saving_pct']:.2f}% less work. Charge probe work. This is two-stage budget allocation, not integrated traversal exit.
5. **When adaptivity is justified:** demonstrate an independently validated full-cost improvement over a fixed risk/work frontier. That condition has not been shown here. The contribution is an analysis of boundaries, including negative results and a clearly blocked completeness experiment.

Positive-only thinning tests missing positive judgments under the original graded nDCG protocol; FULL is pooled available judgments. Recompute IDCG, preserve unjudged=0, freeze rankings and CURA choices, and require ANN Recall invariance. Seed uncertainty is conditional on the 43 queries; paired query/seed inference must avoid pseudo-replication.

The [restored-ranking evaluation path](../sigir_analysis/RESTORED_RANKING_EVALUATION.md) is now implemented with a strict provenance/FULL-reproduction gate, frozen policy checks and seed-level inference. Its isolated synthetic tests validate implementation behavior only. Real-input preflight remains blocked; no synthetic fixture is counted as a TREC result or completed thinning seed.

The overall statistical analysis remains INCOMPLETE: saved-data primary comparisons P1-P4 are complete, while P5/P6 (FULL versus thinned gap and exact-zero fraction) have no estimable effects or CIs. A, B and C cannot be selected; judgment sparsity's effect is UNRESOLVED. Full-study validation is FAIL because missing ranking-based requirements cannot be checked, even though available-input and saved-analysis checks pass. The short paper contains only supported saved-data claims and a pending-density section. Its sole maintenance edit corrected the reproducibility directory after saved-data validation; the scientific claims are unchanged. SIGIR-short readiness is NEEDS MORE EVIDENCE. No commit or push was performed.

Read the [main research report](../reports/SIGIR_RESEARCH_ANALYSIS.md), [analysis index](../sigir_analysis/README.md), [qrels recovery](../sigir_analysis/qrels_recovery.json), [exhaustive audit](../sigir_analysis/local_retrieval_audit.json), [density protocol/status](../sigir_analysis/trec_judgment_density_status.md) and [short-paper source](sigir_short.tex). The historical course manuscript discloses the new findings. No retrieval, embedding generation, index construction or retraining was performed for this continuation.
'''
    (ROOT/'paper/SIGIR_REVISED_DIRECTION.md').write_text(direction,encoding='utf-8')
    index='''# SIGIR research analysis

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
| Restored-ranking evaluator | [instructions](RESTORED_RANKING_EVALUATION.md), [runner](run_trec_judgment_density.py), [strict input gate](ranking_input_gate.py), [provenance template](trec_ranking_provenance.template.json), [synthetic checks](evaluation_pipeline_validation.json), [test log](evaluation_pipeline_test_log.txt) |
| Recovery and completion gates | [ranking validation](trec_ranking_recovery_validation.json), [retrieval-state audit](trec_retrieval_state_audit.json), [detailed settings](trec_retrieval_state.json), [reproduction references](trec_retrieval_reproduction.csv), [reproduction status](trec_retrieval_reproduction_status.md), [completion report](SIGIR_COMPLETION_STATUS.txt), [short-paper record](short_paper_evidence_gate.json) |

All density seed rows are explicitly evaluation_completed=False. Blank retrieval metrics mean unavailable; judgment counts and hashes are real preparation results. The four density plots are absent. Historical FULL metrics are not a new original-ranking validation or a completed density experiment. Full-study validation reports FAIL for unmet ranking/seed requirements; input and saved-data integrity checks pass separately. The supported-only short-paper draft retains its pending-density disclosure; its only maintenance change is the neutral reproducibility path. New density claims remain gated on validated ranking evidence.

Reproduce with NumPy, pandas, SciPy and Matplotlib (versions in analysis_protocol.json):

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

Recovery reuses verified local qrels. These commands never invoke retrieval, training, encoding or index construction. Source inputs and human reviews remain unchanged. The report builder updates the main research report, revised-direction Markdown and this index. The separately maintained standalone LaTeX source remains uncompiled.
'''
    (OUT/'README.md').write_text(index,encoding='utf-8')
    print('Built main SIGIR research report, revised paper direction and artifact index.')


if __name__=='__main__':main()
