# Same-ranking TREC judgment-density study: BLOCKED

The original qrels are recovered and validated. The remaining blocker is the missing exact/HNSW document rankings and original retrieval caches. No ranking-dependent density result or plot has been fabricated.

## Qrels recovered

The original [NIST passage judgments](https://trec.nist.gov/data/deep/2019qrels-pass.txt) were recovered via the [ir-datasets checksum mirror](https://mirror.ir-datasets.com/2f4be390198da108f6845c822e5ada14) after the NIST endpoint failed certificate validation. TLS verification was retained. The 187,092-byte file has MD5 `2f4be390198da108f6845c822e5ada14`, 9,260 judgments and exactly the original 43 query IDs. Every query's grade>0 count matches the saved exact-reference metadata. See qrels_recovery.json and trec_qrels_query_audit.csv. This does not yet validate FULL ranking metrics because ranking identities are unavailable.

## Exhaustive local audit

local_retrieval_audit.json and local_artifact_inventory.csv document recursive enumeration of the entire workspace, including hidden/ignored files. CSV/TSV contents, JSON/JSONL structures, text qrel/run formats, NumPy headers, archive members and all local Git blobs (including unreachable objects) were inspected. Archive content was inspected, not just filenames. No search or model generation was run. Pickles, if present, are inspected as opcodes without executing them.

The only workspace vector cache is data/beir/fiqa_embeddings.npz: 57,638 document vectors and 648 query vectors, dimension 384, with FiQA IDs. This is a different corpus and is not usable as the original million-scale MS MARCO cache. Git candidates are parameter/configuration IDs, completion flags or manual FiQA evidence, not ranked-document arrays. Source sweeps retain first_indices only in memory and export scalar metrics.

## Prepared conditions and exact deterministic protocol

FULL uses the complete supplied original qrels once. For seeds 0..49 and each original raw query ID, sort its grade>0 original document IDs lexicographically. Seed PCG64 with int.from_bytes(SHA256(f'{seed}\0{raw_qid}'.encode('utf-8')).digest()[:8], 'big'), and permute that sorted list. Retain nested prefixes of max(1,ceil(0.5*n)), max(1,ceil(0.25*n)), and 1 positive for 50_PERCENT_POSITIVES, 25_PERCENT_POSITIVES and ONE_POSITIVE_PER_QUERY. Retain original grades and every nonpositive judgment; drop the other positive judgments. Removed positives become unjudged, with relevance 0 under the original protocol. FULL means all available pooled judgments, not complete relevance ground truth. Grade 1 counts as positive because the original graded nDCG code gives it gain 1; no binary-relevance threshold is substituted.

trec_thinned_positive_judgments.jsonl.gz stores the actual retained original positive judgments for every query/condition/seed; nonpositive judgments are referenced from the unchanged original qrels. The per-seed CSV has 150 prepared stochastic seed rows plus one FULL row, all explicitly evaluation_completed=False. Its qrel counts/hashes are real preparation results. All retrieval-dependent fields are blank. No ranking seeds have been evaluated, so no seed CIs or paired density tests exist.

## Evaluation gate once original rankings are restored

Freeze the query set, exact and all ten HNSW top10 original-document rankings, work observations, thresholds and existing CURA selections. Require 43 exact rankings and 430 query/budget HNSW rankings, no duplicate documents within top10, valid original IDs and immutable hashes. Validate FULL nDCG and ANN Recall against archived per-query metrics before thinning. Unjudged=0; gain=2**grade-1; discount=log2(rank+2); IDCG is recomputed from all retained positive judgments for the query, not only retrieved documents. IDCG<=0 gives nDCG0. Loss=max(0,exact-approximate nDCG), violation iff loss>0.025.

Recompute zero-reference fraction, exact/approximate scores, losses, smallest actually safe tested ef and feasibility flags, ANN-minus-IR gap, lower-budget rate, all ten fixed points and frozen CURA risk/work. ANN Recall, ANN-safe ef and work must remain invariant across all conditions and seeds. Do not assume IR safety is monotone in ef or judgment density. Flag no-safe-in-grid outcomes; do not silently treat ef256 as safe.

Use seed bootstrap intervals for seed-average effects conditional on these 43 queries; pair each thinned query/seed with its FULL query. Add a crossed query-and-seed bootstrap for population sensitivity while preserving all paired conditions. FULL is one observation, not 50 independent replications. Pointwise Wilson intervals and post-hoc fixed matching do not confer new risk guarantees.

## Recovery boundary

Restore original exact/HNSW ranked document IDs to evaluate without retrieval. If HNSW rankings were not persisted, retrieval alone requires the original cached query vectors, built index and original row/document map. Exact rankings additionally require the original exact arrays or original corpus vectors for exhaustive search without constructing an index. These caches are missing. No embeddings, index construction or controller training were attempted. See ranking_recovery_requirements.csv for exact paths.

The four requested judgment-density plots (budget gap, exact-zero fraction, IR-safe budget, and risk/work) are intentionally absent because there are no ranking reevaluation observations. The canonical trec_fixed_risk_work_by_density.csv, per-query and unchanged-policy CURA tables are schemas with zero observations. Historical FULL results remain in fixed_risk_work_frontier_trec_saved_full.csv and oracle_budget_gap_analysis.csv and must not be confused with a completed density experiment.
