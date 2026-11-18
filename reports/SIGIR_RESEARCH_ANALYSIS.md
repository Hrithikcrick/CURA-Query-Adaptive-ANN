# SIGIR research analysis

**Research question:** When do downstream-utility-aware ANN search budgets provide genuine efficiency gains, and how are those gains affected by relevance-judgment completeness and risk calibration?

**Main finding.** Judgment thinning increases measured ANN-versus-IR budget headroom on the same 43 TREC-DL-2019 queries and recovered rankings: the mean ANN-minus-IR safe-ef gap changes from **-0.140 under FULL to 1.956 with one retained positive per query**. This is **Outcome A**, conditional on this query set and the specified thinning mechanism. The one-positive condition also produces **82.88% exact-zero nDCG references**, for which reference loss is automatically zero. This is evidence of annotation sensitivity, not improved retrieval or proof of actual relevance preservation.

**The negative adaptive result remains.** CURA uses 20.51% less recorded MS MARCO distance work than the original calibration-selected fixed ef96 baseline. However, post-hoc fixed ef48 matches CURA's observed 2/200 test violations with **28.29% less work**. This implementation has not demonstrated an advantage over the observed fixed risk/work frontier. Retrospective matching is a diagnostic, not a prospectively calibrated policy or population-risk guarantee.

Original rankings were **recovered**, not regenerated. FULL-qrel reproduction, the 50-seed same-ranking density evaluation, and its statistical comparisons are complete. Previously completed scientific analyses were preserved rather than rerun. Separate FiQA human annotation is **50/50 complete**, with every recorded decision supported by the human-review log; model suggestions alone do not count as human decisions.

## Evidence and recovery

The earlier [workspace/Git audit](../sigir_analysis/local_retrieval_audit.json) is a historical search snapshot. It did not locate the original ranking caches in that scope. A subsequent external-archive search found Downloads/IR_PROJECT.zip, with the original run under IR_PROJECT/PILOT/. The [recovery delta](../sigir_analysis/recovery_delta_20261116.json), [ranking provenance](../sigir_analysis/trec_ranking_provenance.json) and [recovery script](../sigir_analysis/recover_trec_cached_rankings.py) record the additional evidence. Former missing-input conclusions are superseded by this verified recovery.

Seven archived source, configuration, query and sweep files match the saved run byte-for-byte. The original 1,000,629-row document map translates original exact top100 row indices and cached HNSW top10 indices. The original baseline source specifies cache axes as sorted archived ef settings and unchanged query-file row order. TREC queries were selected by their recorded IDs. No document order was inferred from scalar metrics.

| Artifact | Current evidence |
| --- | --- |
| Original TREC-DL-2019 qrels | 9,260 judgments; exactly 43 saved raw query IDs |
| Exact document rankings | Recovered original top100 for 43 queries; scoring uses top10 |
| HNSW document rankings | Recovered original top10 for 43 queries at all ten ef settings |
| Row-to-document mapping | Recovered original ordered map; 1,000,629 unique document IDs |
| Original query/corpus vectors | Located in the archive; used only to reconstruct score evidence |
| Original HNSW index | Located in the archive; not loaded, rebuilt or searched |

The normalized [exact rankings](../sigir_analysis/trec_exact_rankings.csv) and [HNSW rankings](../sigir_analysis/trec_hnsw_rankings.csv) each contain 4,300 rows. Exact scores are original cached scores. HNSW ranked IDs are original cached search outputs; their scores were not cached and were reconstructed by inner products of original cached vectors **without changing ranks**. No retrieval, new encoding, index construction or controller training was performed.

**Provenance limitation.** No independent historical checksum of the corpus map was recorded. Archive lineage, seven byte-identical run files, the recovered map's checksum/count, judged-document membership and FULL metric reproduction establish consistency; they are not independent proof of historical corpus identity. The historical encoder revision was not pinned. These limitations remain explicit in the provenance and gate records.

## FULL-qrel reproduction gate

The unchanged [input gate](../sigir_analysis/ranking_input_gate.py) passed before thinning was scored. The [recorded validation](../sigir_analysis/recovered_ranking_validation.json) and [473-row reproduction table](../sigir_analysis/trec_retrieval_reproduction.csv) contain:

- 43 exact-query comparisons and 430 HNSW query-budget comparisons;
- matching exact/approximate nDCG and loss within absolute tolerance 1e-6;
- matching ANN Recall@10 within absolute tolerance 1e-9;
- identical per-query/per-budget violation indicators for loss > 0.025;
- verified query-ID bijection, document membership, unique documents and contiguous ranks.

The tested efSearch grid is 10,16,24,32,48,64,96,128,192,256. ANN safety is exact-neighbor Recall@10 >= 0.90. IR safety is max(0, exact nDCG@10 - approximate nDCG@10) <= 0.025. Original work and timing observations remain archived measurements; they were not newly measured during recovery.

The original qrels have SHA256 8a1f10d550732e4cd91d7fc49846a3784de4040972f583e69285a88f3c5fee92, 4,102 grade>0 judgments and 5,158 grade-zero judgments. Their identity and per-query positive counts are recorded in [qrels recovery](../sigir_analysis/qrels_recovery.json) and the [query audit](../sigir_analysis/trec_qrels_query_audit.csv). FULL means all available original pooled judgments, not complete relevance ground truth.

## Completed same-ranking judgment-density experiment

FULL was evaluated once. Each stochastic condition was evaluated for all **50 paired seeds, 0..49**, reusing the previously prepared qrel sets. Each query receives one deterministic permutation of its sorted positive document IDs per seed. Nested prefixes retain max(1,ceil(0.5*n)), max(1,ceil(0.25*n)), or one positive. Retained grades and all original nonpositive judgments are preserved; removed positives become unjudged. Grade 1 remains positive under the original graded-gain protocol.

nDCG uses exponential graded gain, logarithmic discount and IDCG over all retained query judgments, with unjudged relevance zero. Both exact and approximate nDCG are recomputed under each qrel condition. An exact-zero reference automatically yields zero one-sided loss, independently of approximate quality.

The experiment freezes query identities, exact and HNSW rankings, ANN Recall, search work, timing, thresholds and CURA choices. [Promotion validation](../sigir_analysis/density_promotion_validation.json) confirms all 151 scored qrel hashes match preparation and that the relevant invariants hold. **All 43 queries are jointly feasible under both safety criteria in every condition and seed**; no fallback or changing feasible cohort explains the density differences here.

| Condition | Evaluated seeds | Retained positives | Mean ANN-IR ef gap | Mean IR-safe ef | Exact-zero fraction | IR allows lower ef |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL | 1 | 4,102 | -0.140 | 12.419 | 0.00% | 4.65% |
| 50% positives | 50 | 2,063 | 0.308 | 11.971 | 1.67% | 6.33% |
| 25% positives | 50 | 1,040 | 0.848 | 11.431 | 9.49% | 8.42% |
| One positive/query | 50 | 43 | 1.956 | 10.323 | 82.88% | 13.40% |

ANN-safe mean ef remains **12.279** throughout. Values for stochastic conditions average over the 50 seeds. This sequence of condition means supports Outcome A; it does not assert that every query or seed changes monotonically. ef differences are not measured work or latency savings.

The [four-row summary](../sigir_analysis/trec_judgment_density_summary.csv), [151 seed summaries](../sigir_analysis/trec_judgment_density_per_seed.csv), [64,930 query-budget records](../sigir_analysis/trec_judgment_density_per_query.csv), [1,510 fixed risk/work points](../sigir_analysis/trec_fixed_risk_work_by_density.csv) and [151 frozen-CURA evaluations](../sigir_analysis/trec_cura_by_density.csv) are populated. CURA is evaluated as an **unchanged policy under alternate relevance-judgment evaluation**, not refit to each density condition.

The four requested figures are available:

- [Judgment density versus ANN-IR budget gap](../sigir_analysis/judgment_density_vs_budget_gap.png)
- [Judgment density versus exact-zero nDCG](../sigir_analysis/judgment_density_vs_zero_ndcg.png)
- [Judgment density versus IR-safe budget](../sigir_analysis/judgment_density_vs_ir_safe_budget.png)
- [Judgment density versus fixed/frozen-policy risk and work](../sigir_analysis/judgment_density_vs_risk_work.png)

## Density inference and interpretation

The statistical unit for new density comparisons is the **paired thinning seed**, conditional on these same 43 queries and rankings. The 64,930 query-budget rows are not independent observations. Percentile intervals use 20,000 seed resamples. Exploratory paired sign-flip tests use 20,000 draws and assume sign exchangeability. No query-population, policy-fitting or deployment-risk uncertainty is covered by these seed intervals, and no crossed query/seed bootstrap is claimed.

| Thinned minus FULL | ANN-IR gap effect (ef) | 95% CI |
| --- | ---: | --- |
| 50% positives | 0.447 | [0.300, 0.593] |
| 25% positives | 0.988 | [0.820, 1.153] |
| One positive/query | 2.096 | [2.001, 2.180] |

Exact-zero fraction increases by 1.67, 9.49 and 82.88 percentage points. The IR-allows-lower rate increases by 1.67, 3.77 and 8.74 percentage points. IR-safe ef effects are exactly the negatives of gap effects because ANN budgets and the feasible cohort are invariant. These IR-budget comparisons (P7) are algebraically redundant with P5, not independent corroboration.

[Density statistics](../sigir_analysis/density_statistical_tests_completed.csv) contain 12 completed comparisons: gap (P5), exact-zero fraction (P6), IR-safe budget (P7), and IR-allows-lower rate (P8), each for three thinning conditions. Every raw Monte Carlo p-value is approximately 0.000050; each Holm-adjusted value is approximately 0.000600. Holm correction covers these 12 dependent density follow-ups only; reported percentile CIs remain unadjusted. Historical tests retain their original scope. The [statistical summary](../sigir_analysis/statistical_analysis_summary.md) and [validation record](../sigir_analysis/density_statistics_validation.json) document the distinction and preservation of the prior 42 completed rows.

**Scientific outcome: A.** Under controlled positive-only thinning, less complete judged evidence increases apparent utility-aware budget headroom. The extreme one-positive condition mostly changes the reference metric into its automatic-zero-loss regime. Even exact-positive conditions can change because both DCG and IDCG change. This intervention identifies sensitivity of the measured criterion within this experiment; it does not recover true unjudged relevance, explain every incompleteness mechanism, establish universal behavior or demonstrate a better retrieval policy.

## Preserved exact-nDCG stratified analysis

The original controlled corpus has 1,000,629 passages and a query-disjoint 600/200/200 training/calibration/test split. The 200 held-out queries contain **41 exact-zero (20.5%)** and **159 exact-positive (79.5%)** references. Against calibration-selected ef96, positive-stratum CURA work saving is **23.09%** (paired-query bootstrap 95% CI [13.91, 31.58]%). Positive queries contribute **92.59%** of total work reduction, a point estimate. Zero-stratum saving is 8.56% (CI [-15.97, 30.89]%).

Thus savings against the original baseline are not concentrated in exact-zero queries. This rejects the narrow zero-reference explanation, not sparse-judgment effects within positive queries. In the positive group, CURA uses 2730.667 distances/query versus 3550.352, a paired reduction of 819.686 (CI [494.440, 1119.157]). Fixed-IR has 1/159 violations and CURA 2/159; mean nDCG is 0.786491 versus 0.784104. Three discordant violation pairs cannot establish equal risk.

These results were not rerun. Complete metrics and preserved numerical evidence remain in [stratified results](../sigir_analysis/exact_ndcg_stratified_results.csv), the [stratified summary](../sigir_analysis/exact_ndcg_stratified_summary.md), [plot](../sigir_analysis/exact_ndcg_stratified_work.png) and [preservation hashes](../sigir_analysis/preserved_analysis_hashes.json).

## Preserved complete fixed risk/work frontier

| Method | Role | Violations | Mean distances/query | Mean nDCG@10 |
| --- | --- | ---: | ---: | ---: |
| Fixed ef96 | Original calibration-selected baseline | 1/200 | 3431.770 | 0.625260 |
| CURA-Full | Frozen adaptive case study | 2/200 | 2727.965 | 0.623363 |
| Fixed ef48 | Post-hoc observed-risk comparator | 2/200 | 1956.345 | 0.623479 |
| Fixed ef64 | Post-hoc frontier point | 1/200 | 2448.645 | 0.625260 |

Fixed ef48 costs **28.29% less work** than CURA (paired-query percentile CI [20.07, 35.39]%), a paired difference of 771.620 distances/query (CI [488.510, 1078.553]). The policies share one failed query, with one fixed-only and one CURA-only failure. Equal counts do not establish equivalent risk or matching failures. Fixed ef64 has lower observed risk and work than CURA.

The original calibration rule selected the first candidate with a two-sided 95% Wilson upper endpoint <= 5%. ef64 had 4/200 violations and upper endpoint 5.0288%, narrowly failing; ef96 had 1/200 and upper endpoint 2.7774%. ef48 did not pass the original rule. Pointwise bounds do not establish selection-adjusted coverage. The post-hoc ef48 interval conditions on its selection and excludes frontier-selection uncertainty.

CURA uses a probe followed by a fresh selected search when needed; it is two-stage allocation, not integrated traversal exit. Its selected-search mean work is 2054.555 plus 673.410 extra charged probe work. Mean ef does not measure this overhead. Saved timing estimates combine search components, exclude feature/controller overhead and are not end-to-end service latency. The reported p95 is across per-query mean search estimates; the original reported p95 worsening of 75.98% remains a limitation.

All ten fixed points remain in the [frontier](../sigir_analysis/fixed_risk_work_frontier.csv), [plot](../sigir_analysis/fixed_risk_work_frontier.png) and separate [calibration frontier](../sigir_analysis/calibration_fixed_frontier.csv). For TREC under FULL, post-hoc fixed ef32 matches CURA's 1/43 violations using 1109.326 rather than 2384.581 distances/query: **53.48% less work** (CI [40.13, 63.06]%). The [archived TREC frontier](../sigir_analysis/fixed_risk_work_frontier_trec_saved_full.csv) remains preserved and same-ranking FULL reproduction confirms its quality inputs.

## Preserved oracle opportunity and historical statistics

Across 819 exact-positive MS MARCO queries, mean first ANN-safe and IR-safe ef are 21.485 and 13.206, with gap 8.278 (95% CI [6.222, 10.527]). The saved estimator substitutes ef256 when a safe candidate is absent; four ANN and three IR thresholds are absent. Restricting to **812 jointly feasible pairs** gives mean gap **7.611** (CI [5.933, 9.387]) and IR-allows-lower rate 25.62%. First-safe means first safe point on the tested grid; a maximum-grid fallback does not certify safety and ef reduction does not establish work reduction. Both estimators remain in [oracle analysis](../sigir_analysis/oracle_budget_gap_analysis.csv).

Under FULL, all 43 TREC queries are exact-positive. Mean ANN-safe and IR-safe ef are 12.279 and 12.419, gap -0.140 (historical paired-query CI [-1.163, 0.884]). Only 2/43 allow lower IR ef; 38 agree and three require higher IR ef. ANN is safe at ef10 for 36/43 and IR for 35/43. The tested-grid floor and five nonzero gaps limit observed headroom and inference. TREC has 95.395 positive judgments/query on average versus 1.071 for MS MARCO. Cross-dataset differences alone remain confounded by query distribution and corpus coverage; the new controlled thinning study addresses metric sensitivity within TREC rather than eliminating those confounders.

| Historical test | Cohort/estimator | Mean difference | 95% CI | Units |
| --- | --- | ---: | --- | --- |
| P1 | MS MARCO all; saved grid-capped | 9.158000 | [7.182000, 11.194000] | ef |
| P1 | MS MARCO all; jointly feasible | 8.139253 | [6.577195, 9.792129] | ef |
| P2 | Exact-positive; saved grid-capped | 8.278388 | [6.222222, 10.527473] | ef |
| P2 | Exact-positive; jointly feasible | 7.610837 | [5.933498, 9.386700] | ef |
| P3 | Test fixed ef96 minus CURA | 703.805000 | [408.594375, 985.821250] | distances/query |
| P4 | Test CURA minus post-hoc fixed ef48 | 771.620000 | [488.510000, 1078.553500] | distances/query |

These historical comparisons, including P1-P4, were not recomputed. Their 20,000-resample paired-query intervals condition on saved policies and selected comparators. Exploratory mean sign-flip, direction sign and exact McNemar tests retain their original unadjusted interpretation. Small violation discordances provide little evidence of risk equivalence. Historical intervals do not cover fitting, policy selection or timing-run uncertainty. See [primary results](../sigir_analysis/primary_statistical_results.csv), the now-complete [54-row statistical table](../sigir_analysis/statistical_tests.csv), and the byte-preserved [pre-extension table](../sigir_analysis/statistical_tests_saved.csv).

## Human review and scope limits

The [50-case review status](50_CASE_FAILURE_REVIEW_STATUS.md) records **50 completed genuine FiQA human reviews and no pending cases**. Every verified case has matching category and evidence-note decisions in the checkpoint and human decision log. The [final human taxonomy](HUMAN_50_FAILURE_TAXONOMY.csv) and [qualitative analysis](HUMAN_50_FAILURE_ANALYSIS.md) describe these fixed HNSW ef=10 failures. Model assistance supplied preliminary suggestions; only explicit researcher decisions counted toward completion, and previously completed decisions were preserved.

The separate 50-case MS MARCO/TREC analysis is model-assisted, including eight second-pass automatic resolutions, with zero human-verified cases. It must not be conflated with the FiQA checkpoint. The FiQA human-review requirement was completed through recorded researcher confirmations, independently of completing the scientific density study.

The main experiment uses a controlled approximately one-million-passage MS MARCO subset, one primary MiniLM encoder and HNSW only. Full 8.8M MS MARCO, a second ANN family, and official DARTH/Patience reproductions were not run. ANN-Risk is not official DARTH; Patience-Inspired is not official Patience. The separate reduced-feature FiQA experiment is not an independent confirmation of adaptive advantage: its recorded work is 75.27% higher than its fixed comparator. TREC inference rests on 43 queries. Search estimates omit controller overhead, and sparse judged utility is not actual relevance preservation.

## Reproducibility and next manuscript step

The [analysis index](../sigir_analysis/README.md), [restored-ranking instructions](../sigir_analysis/RESTORED_RANKING_EVALUATION.md), [density status](../sigir_analysis/trec_judgment_density_status.md), [run manifest](../sigir_analysis/restored_ranking_evaluation/run_manifest.json) and [completed experiment validation](../sigir_analysis/restored_ranking_evaluation/validation.json) describe inputs and outputs. Recovery, density promotion and statistics have separate passing evidence records linked above. The [canonical validation](../sigir_analysis/validation.json) and [completion report](../sigir_analysis/SIGIR_COMPLETION_STATUS.txt) record the integrated final checks.

For the present completed workspace, rerun only the lightweight audits:

    python sigir_analysis/validate_analysis.py
    python sigir_analysis/completion_status.py

Do not rerun completed saved analyses, corpus encoding, retrieval or index construction. A fresh external reproduction can follow the documented recovery and gated density commands using the original archive; those are separate from auditing these completed outputs.

The defensible paper story is now: **judged ANN/IR oracle headroom depends on annotation completeness, while fixed-frontier calibration and adaptive overhead determine whether an implementation realizes a genuine efficiency gain.** The current CURA case study does not demonstrate that gain. [Revised direction](../paper/SIGIR_REVISED_DIRECTION.md) and [short-paper source](../paper/sigir_short.tex) carry this supported interpretation. Scientific evidence supports drafting; final TeX compilation and official-format checking remain distinct work. Genuine FiQA human review is complete.
