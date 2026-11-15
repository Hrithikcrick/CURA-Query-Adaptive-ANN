# TREC retrieval reproduction: NOT RUN

Original qrels and the 43 query identities are recovered. Exact and all ten
HNSW ranked-document outputs remain missing. The original controlled corpus
mapping/order, document vectors, query vectors and built HNSW index are also
missing, so a faithful retrieval-only rerun is impossible from available inputs.

[trec_retrieval_reproduction.csv](trec_retrieval_reproduction.csv) contains 61
archived comparison targets: exact mean nDCG and six metrics for each of ten
HNSW budgets. Reproduced values and differences are blank; every comparison is
explicitly incomplete. These are saved references, not reconstructed results or
a FULL-qrel ranking validation. Numerical comparison tolerances are prospective
(nDCG/loss 1e-6, ANN Recall 1e-9, identical violation rate); no tolerance was
applied to missing observations. Work and timing tolerances are unset because
their reproducibility conditions cannot be checked.

No retrieval, embedding generation, index rebuild or controller retraining was
performed. No alternative corpus was used. Configured encoder, normalization,
similarity and HNSW parameters can be read from source; that does not verify
the contents of an unavailable index or vector cache.

Restore original exact/HNSW document rankings for all 43 queries (and HNSW at
efSearch 10,16,24,32,48,64,96,128,192,256), or restore original cached vectors,
document map and built index. Verify corpus membership/order and query mapping,
then compare FULL per-query nDCG, ANN Recall and fixed violation rates before
thinning. Stop if recreated metrics disagree materially with the saved runs.
Latency is comparable only if hardware and timing conditions are established.

See [ranking validation](trec_ranking_recovery_validation.json),
[retrieval-state audit](trec_retrieval_state_audit.json) and
[detailed original settings](trec_retrieval_state.json).
