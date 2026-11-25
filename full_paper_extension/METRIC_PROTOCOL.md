# Preregistered metric and false-safety protocol

Protocol date: 2026-11-18. This document fixes the new estimands before the extension computes false-safety or alternative-metric outcomes. The historical TREC 2019 standard-nDCG density results were already known; this is a prospective analysis specification for the new estimands, not a claim that the entire research program was registered before those historical results.

## Scope and reference condition

FULL means all supplied pooled relevance judgments for that collection, not complete relevance ground truth. A false-safe observation below is a disagreement with FULL evaluation, not proof of an actual user-relevance failure. Query IDs, corpus document IDs, ranking order, ANN Recall@10, measured work, and frozen policy choices remain unchanged within a collection across judgment conditions. New collection/corpus combinations require their own identity, coverage, and FULL-reproduction gates.

The tested grid is `10,16,24,32,48,64,96,128,192,256`. The ANN-safe criterion is exact-neighbor Recall@10 >= 0.90. The primary utility metric is the historical exponential-gain nDCG@10; loss tolerance is 0.025, with equality considered safe. These choices reproduce the existing protocol rather than being selected from extension outcomes.

Incomplete judgment evaluation and judgment-removal experiments have substantial prior literature. Buckley and Voorhees varied available judgments while retaining the document collection and comparing retrieval runs. Their contribution includes bpref; neither qrel thinning nor sensitivity of evaluation to incomplete qrels should be claimed as new here. The candidate extension is the effect on ANN budget safety decisions, subject to the broader novelty audit. [Buckley and Voorhees, 2004](https://www.nist.gov/publications/retrieval-evaluation-incomplete-information), [author PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469).

## Judgment intervention: removal, not relabeling

Replicate `sigir_analysis/run_trec_judgment_density.py::thinned_qrels` exactly. For each raw query ID and seed 0 through 49:

1. Sort the query's original positive document IDs (`grade > 0`) lexicographically.
2. Form `SHA256(f'{seed}\0{raw_qid}'.encode('utf-8'))`; initialize NumPy PCG64 with the first eight digest bytes interpreted as an unsigned big-endian integer.
3. Permute the positive IDs once. Retain nested prefixes of sizes `max(1, ceil(0.50*R))`, `max(1, ceil(0.25*R))`, and `1`.
4. Retain each selected document's original grade and every original nonpositive judgment. **Delete** the other positive qrel records. A deleted positive becomes unjudged; do not insert a grade-zero judgment for it.

Standard nDCG maps either an unjudged or judged-zero result to zero gain. Condensed metrics and bpref distinguish them, so converting removed positives to judged zero would implement a different intervention. FULL is evaluated once. Each seed is paired across density conditions. Positive-only removal with nonrelevant judgments retained is a specific class-dependent missingness mechanism; it does not model all forms of real pooling bias or unbiased thinning of all assessments.

Sakai's condensed-list analysis removes unjudged documents before evaluating the ranked list. Its robustness findings concern specified assessment-removal regimes; they do not guarantee robustness to biased pools. [Sakai and Kando, 2008](https://doi.org/10.1007/s10791-008-9059-7), [Sakai, 2009](https://doi.org/10.2197/ipsjjip.17.156).

## Ranking-depth implementability audit

A read-only inspection of the existing frozen TREC 2019 CSVs found:

| Artifact | Queries / query-budget lists | Saved depth per list |
|---|---:|---:|
| `sigir_analysis/trec_exact_rankings.csv` | 43 | 100 |
| `sigir_analysis/trec_hnsw_rankings.csv` | 430 | 10 |

This agrees with `sigir_analysis/trec_ranking_provenance.json`. No extension outcome analysis was needed for this audit. The historical `density_metrics.py` deliberately consumes the first ten results and cannot recover missing ANN ranks.

Standard nDCG@10 is exactly identifiable from these inputs. Condensed **first-ten-judged-results** nDCG@10 of a deeper ANN ranking is generally not identifiable: an unjudged top-ten result may be removed and a result at original rank 11 or below may move into the evaluated list. Do not fabricate such ranks, pad them with assumed nonrelevance, or condense the exact top100 while condensing only the ANN top10.

The feasible alternative is to declare that both systems return exactly ten results, truncate the exact ranking to ten too, and apply metrics to that returned list. Report these metrics with explicit names:

- `standard_ndcg_at10`;
- `condensed_returned10_ndcg_at10`;
- `bpref_returned10`.

The latter two are valid evaluation of the **submitted depth-ten output**, with no assumed additional returned documents. They are a restricted metric-robustness analysis, not a reproduction of deeper-run condensed nDCG or unrestricted bpref. Every table/figure must disclose this scope. A full deeper-run Phase 4 comparison remains blocked unless verified deeper ANN rankings become available. If TREC 2020 produces deeper rankings, retain them and separately label the deeper-run analysis; a cross-collection comparison must use the same depth contract.

## Metric definitions

Let `J_c(q)` be the retained judged documents for query `q` under condition `c`; its grade map includes judged zeros. Let `pi(q,e)` and `pi*(q)` be the saved ANN and exact output lists. All metric comparisons below use the same declared return depth for both lists.

### Standard nDCG@10

For retained grade `g`, gain is `2^g - 1`; absent judgments have gain zero. DCG sums gain divided by `log2(i+1)` for ranks `i=1,...,10`. IDCG sorts **all retained positive grades for the query**, takes the largest ten, and applies the same discount. It is not restricted to retrieved results. Recompute IDCG for every condition. If IDCG is zero, the historical convention assigns nDCG zero and records an explicit zero-reference flag. `density_metrics.py::ndcg_at_10` is the historical specification; a library with linear gain must not silently replace it.

### Condensed nDCG on the returned-ten output

First retain from the submitted original top10 only documents in `J_c(q)`, preserving order, including judged-zero results. Reindex those retained results starting at one; compute exponential-gain nDCG@10 using their new ranks and the same condition-specific global IDCG above. If fewer than ten judged results remain, evaluate the returned finite list; there is no invented deeper continuation. Report both returned length and judged-retained length. No document missing from the original top10 is promoted into this restricted output.

Condensation is established prior work, rather than a new metric proposed by CURA. [Sakai, *Alternatives to Bpref*, 2007](https://doi.org/10.1145/1277741.1277756).

### Bpref on the returned-ten output

Binarize grades as positive iff `grade >= 2`; judged grades zero and one are binary nonrelevant. This prespecified bpref mapping differs from the historical thinning-positive definition (`grade > 0`): thinning can retain a grade-one document and leave no binary positive judgments. Such queries must be explicitly flagged inapplicable for bpref, not marked safe. Let `R` and `N` count binary-positive and judged-binary-nonrelevant documents in the entire condition-specific qrels for the query. Let `n_r` count judged-binary-nonrelevant returned documents preceding returned binary-positive document `r`. Unjudged returned documents contribute neither to `n_r` nor to the positive sum. For `R > 0` and `N > 0`:

`bpref = (1/R) * sum_{positive r in returned list} [1 - min(n_r,R)/min(R,N)]`.

Unreturned positive documents make no positive contribution, but remain in `R`. This matches the current NIST implementation, including normalization by `min(R,N)`, rather than an older buggy variant. [NIST `trec_eval/m_bpref.c`](https://github.com/usnistgov/trec_eval/blob/main/m_bpref.c), [NIST measure definitions](https://trec.nist.gov/pubs/trec20/appendices/measures.pdf).

If `R=0` or `N=0`, report an applicability flag and `NA` for the primary bpref comparison, not a manufactured safe decision. NIST software has conventions for some degenerate cases, but no positive-versus-negative preference comparison is possible without both classes. Report all exclusions and retain the queries in other valid metrics. Pin any evaluator commit used for verification. Synthetic fixtures should check removal versus zeroing, rank condensation, missing positives, `R<N`, `R>N`, and degenerate cases before empirical calculation.

## Primary false-safety estimand

For metric `m` and condition `c`, define

`L_m,c(q,e) = max(0, M_m,c(pi*(q)) - M_m,c(pi(q,e)))`

and `safe_m,c(q,e) = [L_m,c(q,e) <= 0.025]`, provided that metric evaluation is applicable. The primary analysis uses standard nDCG@10 and all query-budget pairs in the complete ten-budget grid, without excluding exact-zero queries. For thinning seed `s` and density `c`, define

`F_c,s(q,e) = [safe_thin(q,e)=True AND safe_FULL(q,e)=False]`.

The **primary FSR** is

`FSR_all(c) = mean_s [(sum_q sum_e F_c,s(q,e)) / (Q*10)]`.

Thus each query has equal total weight and each tested budget has equal weight within a query. The budget grid is fixed, not a random population of budgets; changing its density changes this estimand.

Before the first extension outcome execution, additionally fix the structural decomposition `FSR_all = FSR_thin_exact_zero_component + FSR_thin_exact_positive_component` for applicable nonnegative metrics. Thin exact score zero implies zero clipped loss for every budget. Consequently the first component is the proportion of all planned pairs with FULL-unsafe labels and thin exact score zero, a lower bound on total FSR. This is an algebraic diagnostic, not an independent empirical contribution. The decomposition retains all budgets and is not an exclusion of zero-reference queries.

Also report **conditional FSR** as the ratio of aggregated counts across paired seeds:

`FSR_given_thin_safe(c) = (sum_s,q,e F_c,s(q,e)) / (sum_s,q,e [safe_thin(q,e)])`.

This is the probability of FULL-unsafe evaluation among decisions labeled safe under thinning. It is not the mean of per-seed ratios; separately report seedwise ratios as descriptive distributions. A zero denominator gives `NA`. For FULL both FSRs are zero by definition where applicable. For any metric with invalid evaluations, publish valid-pair count, total planned-pair count, missing count, and the reason; do not silently reduce the denominator.

Report the reverse disagreement (`FULL safe`, `thin unsafe`), all four confusion-matrix counts, raw agreement, and balanced accuracy when both FULL classes are present. Equal observed violation counts do not imply equal unsafe query identities or risk equivalence.

## Budget optimism and headroom inflation

For each query, define `B_m,c(q)` as the **smallest actually safe tested efSearch**, scanning the grid without a monotonicity assumption. An unsafe high-budget point does not invalidate a lower safe point. If no grid point is safe, store `safe_found=False` and a missing minimum; do not replace it with 256 and call it safe.

`budget_optimism_c,s(q) = B_FULL(q) - B_c,s(q)`

is defined on the pairwise feasible subset where both minima exist. Report its mean, median, distribution, fraction positive/zero/negative, and the subset denominator, alongside all four FULL/thin feasibility patterns. Do not assume thinning only makes budgets smaller.

Headroom inflation is

`[A(q)-B_c,s(q)] - [A(q)-B_FULL(q)] = B_FULL(q)-B_c,s(q)`

when the ANN minimum `A(q)` and both IR minima are available. The equality is algebraic: this is not a second independent empirical discovery. Budget optimism remains meaningful when the ANN criterion is infeasible; call it headroom inflation only on the jointly feasible domain. A largest-grid substitution, if shown solely to reproduce a historical estimator, must be separately labeled `legacy_grid_capped`, with feasibility flags preserved. No-safe status should be called grid-infeasible, not an identified true budget above 256: safety need not be monotone.

Correlation of thin and FULL minima uses Spearman correlation with midranks on pairwise feasible queries, with the number and proportion of retained pairs disclosed. Constant vectors or insufficient pairs yield `NA`. Such subset correlations cannot establish preservation of all-query safety.

## Alternative-metric comparison and mitigation criterion

For each alternative metric, compute its own FULL and thin scores and repeat the definitions above with the explicitly labeled depth contract. The requested shared numeric loss tolerance of 0.025 is a descriptive sensitivity comparison, **not equivalent utility tolerance across metrics**. Bpref is binary and has different normalization; condensed nDCG changes the evaluated ranks. A smaller FSR may simply reflect fewer FULL-unsafe pairs or a metric's inability to detect top-ten utility loss.

Therefore report per metric: FULL-unsafe prevalence, FULL/thin exact-zero rate, feasibility rates, both FSRs, reverse disagreements, score-loss distributions, and agreement with **standard FULL nDCG** safe/unsafe labels as a common-reference diagnostic. For the latter, identify the thin metric and FULL reference metric separately. Report false-safe fraction against that common reference and the sensitivity to its unsafe pairs. Do not call reduced within-metric FSR alone successful mitigation. No cross-metric threshold may be tuned on the extension's test outcomes to manufacture a mitigation result. A prospectively calibrated metric-specific threshold would require a separate calibration split and claim.

## Prespecified stratifications

All strata are frozen before thinning; queries never move strata when their thinned scores change.

- **Exact FULL nDCG strata:** compute the collection-specific 0.25, 0.50 and 0.75 quantiles of exact FULL standard nDCG using linear interpolation. Assign `<=Q25`, `(Q25,Q50]`, `(Q50,Q75]`, and `>Q75`. Keep tied scores together, disclose cutpoints and actual group sizes, and do not force equal-size quartiles by splitting tied zero scores. Empty groups remain explicit. Additionally report exact-FULL-zero versus exact-FULL-positive groups.
- **Query difficulty proxy:** use the qrel-independent ANN-safe first grid budget: `ef10`, `ef16-32`, `ef48-96`, `ef128-256`, and `ANN-grid-infeasible`. These are declared budget bins, not a validated semantic definition of query difficulty. They are invariant to qrel thinning.
- **By-budget:** retain all ten efSearch values and label their fixed denominator. For frozen policies, evaluate only their original choice and charge original effective work; an oracle that uses thinned qrels is explicitly retrospective and may change choices.

## Uncertainty, tests, and collections

Primary finite-collection estimates average over the 50 deterministic paired thinning seeds conditional on the fixed query set, corpus, rankings, and mechanism. Use a paired seed bootstrap with 20,000 resamples for 95% percentile intervals, recomputing conditional-rate ratios from counts. The same sampled seed indices apply to every paired density condition. FULL is one reference, not 50 independent FULL observations. Monte Carlo intervals describe thinning variability, not unseen-query generalization.

Report a separately labeled crossed query-and-seed bootstrap sensitivity: independently resample queries and seed indices with replacement within each collection and preserve each selected query's complete budget vector and paired condition/reference outcomes. Do not treat query-by-seed cells, nor ten budgets from one query, as independent observations. Prespecify RNG seed 20261118 for new resampling and record actual software versions and command lines.

Paired sign-flip tests apply only to suitable seed-level paired differences, with a stated symmetry/exchangeability assumption; do not sign-flip the nonnegative FSR against its structural FULL zero as if that automatically supplied a valid null test. Primary reporting emphasizes effect sizes and intervals. For budget optimism versus FULL zero, use signed seed-level average paired budget differences and Holm-correct three density comparisons within each metric's budget-optimism family. For each alternative-minus-standard metric FSR endpoint, Holm-correct the three density comparisons as a separately labeled exploratory family. Undefined tests remain visible and do not count as performed tests. Strata are descriptive; all pointwise intervals remain labeled unadjusted.

Show TREC 2019 and TREC 2020 separately before a cross-collection summary. An equal-weight mean of two collection-specific effects is a descriptive fixed-two-collection summary, not a meta-analysis of a population of collections. Resample within each collection independently, keep collections distinct, and display the signed effect difference as heterogeneity. With two collections do not claim a reliable random-effects heterogeneity variance or hide opposing results in a pooled mean. A blocked collection stays blocked; no zero, copied value, or hypothetical effect substitutes for a measurement.

## Interpretation boundary

This analysis isolates judgment intervention under frozen retrieval. It does not show how a controller trained anew with sparse labels would behave unless that separate training experiment is actually executed. A lower measured required ef is not itself measured work saved. Sparse safety is not true relevance preservation. Restricted-output metric results cannot establish that an incomplete-judgment metric fixes the general problem, and failure to reproduce a collection prevents a claim of independent replication.
