# Extension analysis plan

Frozen before new retrieval and new false-safety/alternative-metric outcome computation on 18 November 2026. The historical 2019 mean-gap result was known. This is an internal timestamped analysis specification, not an externally registered study.

The literature audit in `NOVELTY_AUDIT.md` is the Phase 0 gate. Detailed metric definitions, finite-list scope, missingness, estimands, feasibility rules, and resampling are specified in `METRIC_PROTOCOL.md`. Changes required to correct implementation errors must be disclosed in execution validation; do not select definitions from favorable outcomes.

## Replication and corpus design

TREC DL 2020 has 200 released queries, of which 54 have passage judgments. Keep all 54 judged query IDs and all 11,386 qrels. The existing 1,000,629-document subset covers only 11.16% of unique judged documents. Therefore append the 9,971 missing judged passages to a separate copy, yielding a 1,010,600-document controlled corpus. Preserve the original ordered document map and index as inputs. Do not modify the 2019 experiment.

Adding judged documents is judgment-informed corpus selection. It permits measuring the intervention with all judged passages available, but the resulting experiment is not a full-corpus TREC result or a same-corpus replication. Report corpus differences in cross-collection interpretation. Reuse original embeddings only after a pinned local encoder passes a prespecified compatibility check on 32 historical queries and 32 passages: maximum absolute coordinate difference <=1e-4 and cosine similarity >=0.99999 for every example. Otherwise do not mix vectors.

Use exhaustive inner products for the new exact reference; original HNSW M=32, efConstruction=200, a copied index augmented with the missing passages, and the existing ten-budget grid. Use one FAISS thread. Timing records five repeated search calls; first-pass distances are the work measurement. Encode in a separate process from index loading, and use bounded-memory exact search. Record encoder revision, model-weight hash, vector/map/index hashes, FAISS version, commands, and timings.

## Primary claims and inference

1. Replication: collection-specific change in mean ANN-minus-IR first-safe budget gap under each of three thinning conditions relative to FULL. All minima must be genuinely feasible; publish infeasibility counts. A positive paired-seed 95% interval is evidence of a positive conditional thinning effect, not unseen-query generalization.
2. False safety: the two FSR definitions in `METRIC_PROTOCOL.md`. Primary FSR averages the complete fixed query-by-budget grid within seed. FULL is a reference condition rather than relevance truth. Nonnegative FSR versus its structural zero is reported with effect sizes and intervals, without an unjustified sign-flip null test.
3. Budget optimism: FULL first-safe IR budget minus thinned first-safe budget on paired feasible queries, including negative changes. Headroom inflation is algebraically the same change when the ANN minimum exists, not independent corroboration.
4. Metric robustness: returned-ten standard nDCG, condensed nDCG, and bpref. A reduction in a metric's own FSR does not establish mitigation. Report common FULL-standard-nDCG unsafe labels, failure sensitivity, applicability, and threshold/scale differences. General deeper-run condensed evaluation remains unidentified from saved ten-result ANN outputs.
5. Policy independence: evaluate frozen available selections against each qrel condition. Work and selections remain fixed. Separately label qrel-dependent IR-safe oracles, whose selections change retrospectively and are not deployable policies.

Use 50 nested deterministic paired seeds, seed bootstrap 20,000 replicates, RNG seed 20261118, pointwise 95% percentile intervals. Appropriate signed paired contrasts may use an explicitly assumption-dependent sign-flip diagnostic with Holm adjustment within declared three-condition families. Do not treat query-by-seed cells or repeated budgets as independent observations. Crossed query/seed bootstrap is a separately labeled sensitivity analysis.

Show each collection separately. The cross-collection summary is an equal-weight average of the two fixed collection estimates, with seed resampling independently within each collection. Show the signed 2020-minus-2019 effect difference and its conditional interval. Do not claim a random-effects meta-analysis or a population-of-collections result from two collections.

## Decision rule

For descriptive decision-gate language, call false safety material when the one-positive primary FSR is at least one percentage point of all query-budget pairs. This criterion is fixed before those outcomes. No alternative-metric mitigation claim follows from a lower raw FSR alone: its common-reference false safety must decrease without losing sensitivity to FULL-standard-nDCG unsafe pairs, and applicability losses must be reported. This is a descriptive criterion, not a calibrated operating guarantee.

Outcome A requires positive independent-collection replication, material false safety, evidence across multiple frozen policies, and defensible metric-level mitigation. Outcome B retains replication and material false safety without a supported mitigation. Outcome C records material collection heterogeneity. Outcome D applies if the proposed effect fails to replicate or lacks consistent false safety. If a critical prerequisite is blocked, report an inconclusive gate rather than assigning an unsupported empirical outcome.

Only after Phases 1-6 are executed or their concrete blockers recorded should one-pass traversal work begin. A design is not a benchmark. No result may be called one-pass unless the same traversal state actually continues, with controller overhead measured. Second-encoder work is conditional on compute and must not replace the higher-priority replication.
