# TREC FULL-qrel reproduction: PASS

Original cached rankings were recovered from `~/Downloads/IR_PROJECT.zip`, under `IR_PROJECT/PILOT`. Seven archived source/config/query/sweep files match the current project byte-for-byte. The original ordered map contains 1,000,629 unique document IDs.

The strict input gate independently reproduced all 43 exact nDCG@10 results and all 430 HNSW query-budget results over efSearch 10,16,24,32,48,64,96,128,192,256. nDCG and loss agree within 1e-6, ANN Recall@10 within 1e-9, and every violation indicator agrees exactly at loss > 0.025.

[Per-query comparisons](trec_retrieval_reproduction.csv), [input validation](recovered_ranking_validation.json), and [archive/member provenance](trec_ranking_provenance.json) contain the actual observations. Exact rank order comes from the cached exact top100 arrays; HNSW rank order comes from the original stability top10 cache. Approximate score evidence was reconstructed from original cached query/document vectors without reordering any result.

No retrieval, encoder execution, index construction or controller training was performed. Original recorded work and timing remain fixed. There is no independently logged historical map checksum; co-archived sources, the supplied ordered-map hash and FULL scalar consistency provide the documented identity evidence. The encoder revision remains historically unpinned, but no encoder was invoked for this recovery.

The density experiment is now eligible to run on these fixed rankings. A successful reproduction alone is not a completed density result.
