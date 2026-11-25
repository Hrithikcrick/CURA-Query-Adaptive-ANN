# Verification of saved CURA research extension

Verification date: **26 November 2026**. Historical repository base: `c41c089554a2596667c2626f7fa35118a586f10c`.

## Provenance and preservation

The completed research was already present locally. No file named `CURA_Full_Paper_Research_Extension.zip` or `CURA_GitHub_Push_Kit.zip`, nor an original `VERIFIED_SAVED_DATA_RESULTS.json`, was found in the workspace, Downloads, Desktop, Documents or Codex attachments search. Some unrelated cache directories denied traversal; no relevant named source was observed there. Existing metric code, tests, protocols and portable inputs were reused with their bytes preserved. Focused verification code and the machine-readable summary were created on this verification date and are explicitly labeled as such.

The historical research remains distinct from this addendum. No embeddings were generated, no index rebuilt, no controller retrained and no retrieval was rerun. Existing working-tree changes, including manuscript edits from earlier work, are outside the commit. A separate branch and isolated worktree are used because the main working tree contains unrelated unresolved changes.

## TREC DL 2019: false safety relative to pooled FULL qrels

The independent verifier reads the original document rankings and recorded retained judgments, reconstructs global-IDCG exponential-gain nDCG@10, and compares every saved exact/approximate score and ANN Recall value. **All 64,930 rows reproduce with maximum absolute score and recall error zero.** Original raw query IDs, all ten budgets and all 50 deterministic paired seeds are retained. The saved judgment selections also match the original PCG64 removal protocol.

There are **43 queries x 10 budgets = 430 FULL pairs**, with **14 FULL-unsafe pairs** at loss tolerance 0.025. Each thinning condition repeats those 14 reference-unsafe pairs over 50 seeds, yielding the denominator **700**.

| Positive judgments retained | False-safe instances / FULL-unsafe instances | Conditional false-safe percentage | False-safe percentage among all 21,500 pair instances |
|---|---:|---:|---:|
| 50% | 192 / 700 | 27.428571% | 0.893023% |
| 25% | 351 / 700 | 50.142857% | 1.632558% |
| One per query | 608 / 700 | 86.857143% | 2.827907% |

At 50% retention, **183 of 192 false-safe instances (95.3125%) have positive exact-reference nDCG under thinning**. False safety therefore is not confined to exact-zero collapse at this density. The counts repeat dependent query-budget pairs across seeds; they must not be interpreted as independent Bernoulli trials.

FULL is the available original pooled qrels, not exhaustive relevance truth. A false-safe instance means safe under thinned judgments but unsafe under this pooled reference. It is not proof of a user's actual relevance loss.

### Tolerance sensitivity

Safety is `max(0, exact nDCG - approximate nDCG) <= tolerance`. No monotonic safety assumption is made across efSearch.

| Loss tolerance | FULL-unsafe pairs | 50% false-safe | 25% false-safe | One-positive false-safe |
|---|---:|---:|---:|---:|
| 0.01 | 14 | 190/700 = 27.142857% | 350/700 = 50.000000% | 608/700 = 86.857143% |
| 0.025 | 14 | 192/700 = 27.428571% | 351/700 = 50.142857% | 608/700 = 86.857143% |
| 0.05 | 10 | 107/500 = 21.400000% | 216/500 = 43.200000% | 411/500 = 82.200000% |

Rankings, per-pair ANN Recall, fixed work, fixed timing observations and frozen CURA selections/work are invariant across judgment conditions. These checks use the original source work records as well as the density tables.

## FiQA: oracle-budget mismatch across domains

Both minima were independently scanned from `beir_fiqa/fiqa_query_budget_results.csv` and compared query by query with `beir_fiqa/fiqa_oracle_budgets.csv`. ANN safety requires Recall@10 >= 0.90. IR safety uses the original signed nDCG loss <= 0.025; clipping negative improvements would not change safety at this positive threshold. Missing feasible budgets remain missing.

| Quantity | Independently reproduced result |
|---|---:|
| Total queries / source query-budget rows | 648 / 6,480 |
| Exact-positive queries | 305 |
| Jointly feasible exact-positive queries | 303 |
| Saved oracle rows compared / mismatches | 648 / 0 |
| Mean ANN-safe efSearch | 24.085809 |
| Mean IR-safe efSearch | 15.524752 |
| Mean ANN-minus-IR gap | 8.561056 |
| IR-safe lower / equal / higher | 120 / 166 / 17 |
| Paired-query percentile 95% CI | [5.498185, 11.623927] |

Exact-positive query IDs **4946 and 5862** have no IR-safe tested budget and are excluded from the jointly feasible means. They are not assigned the maximum budget.

The interval uses a new declared retrospective verification configuration: NumPy PCG64 seed 42, 10,000 paired-query bootstrap resamples, canonical string-query-ID order. It reproduces the supplied rounded interval, but the original CI-generating script was not recovered. A separate seed 20261126 with 50,000 resamples yields **[5.531353, 11.584158]**. These intervals condition on saved retrieval and the exact-positive feasible cohort.

FiQA replicates **oracle ANN-safe versus IR-safe budget mismatch** in another domain. Judgments are unchanged: this does not replicate the TREC thinning/false-safety experiment and does not demonstrate cross-domain transfer of the CURA controller.

## Tests, figures and broader saved-data check

- **11/11 original metric tests passed**, including global IDCG, judgment missingness, bpref applicability, finite-grid infeasibility and clustered resampling.
- **15/15 new verification tests passed**, including safety denominator semantics, threshold equality, nonmonotone oracle minima, query/budget corruption detection, signed improvements and paired bootstrap behavior.
- Independent TREC scoring: **64,930/64,930 rows**, zero score/recall error.
- Independent FiQA minima: **648/648 queries**, zero mismatches.
- The existing original `analyze_false_safety.py` and `build_cross_collection.py` were also run in an isolated copy, with 50 seeds and 20,000 bootstrap resamples. **All cells of 16 saved 2019/2020 statistical tables matched** (numeric tolerance absolute 1e-12, relative 1e-10). The local historical outputs were not overwritten.

The focused figures under `figures/verified/` plot the conditional FULL-unsafe denominator explicitly, show all three loss tolerances, and display FiQA's feasible-cohort distributions and 120/166/17 split. The manifest records their plotted values and hashes. Visual inspection and clean-checkout checks are recorded in `PUSH_VERIFICATION.json`.

## Scientific boundaries

The original supported efficiency result is retained: CURA can beat the calibration-selected fixed baseline, while a post-hoc risk-matched fixed setting can be cheaper than the current CURA implementation. The new findings do not establish universal adaptive superiority, prospective equal-risk guarantees, exhaustive relevance truth, robustness to all pooling mechanisms or controller transfer.

The local workspace also contains completed TREC 2020, corpus-control and official-source DARTH work. Those files remain intact outside this focused push. The frozen protocol/audit documents retain their broader historical context; they are not evidence that every broader experiment was independently regenerated in this session. CURA-OnePass remains unmeasured. No claim is made that the existing 43-query intervention represents unseen query populations or that these saved-data checks reproduce expensive retrieval on a clean machine.
