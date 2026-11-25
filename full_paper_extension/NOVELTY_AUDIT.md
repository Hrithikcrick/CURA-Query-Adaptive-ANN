# Literature and novelty audit

Search cutoff: **18 November 2026**. This is a focused audit, not a systematic-review completeness claim. It precedes new extension experiments. Historical CURA results are not altered by this document.

## Verdict and defensible research question

Query-adaptive ANN effort, learned early termination, HNSW traversal stopping, and the difference between exact-neighbor recall and downstream usefulness all have substantial prior art. Relevance-judgment deletion with fixed system rankings is also established IR methodology. None of these ideas should be described as a first contribution of CURA.

The defensible extension question is narrower: **How does incomplete relevance assessment change measured safety labels and selected ANN budgets, holding rankings, work, and exact-neighbor recall fixed?** A useful empirical contribution would measure disagreement with the FULL-qrel reference, budget optimism, and metric/policy dependence, then test independent TREC collections. Whether these experiments support a strong claim remains an empirical question. FULL means the available pooled judgments, not complete relevance truth.

No located source established the specific combination of ANN query-budget false-safety rates, paired relevance thinning with invariant ANN computations, cross-collection replication, and incomplete-judgment metric comparisons. This is a bounded search finding, **not proof of priority**. Do not claim “first.”

## Search record and interpretation

Primary sources were searched and inspected through arXiv versioned papers, author/institutional PDFs, publisher proceedings, and author-linked repositories. Query families included `DARTH github`, `Patience in Proximity`, `Ada-ef`, `learned adaptive early termination`, `early exit dense retrieval`, `task-aware HNSW`, `downstream vector search recall`, `Semantic Recall`, `ANN incomplete relevance judgments`, `HNSW relevance judgment sparsity`, and `ANN judgment thinning`. The incomplete-judgment literature was independently checked against NIST, J-STAGE, and official `trec_eval` material.

Version dates below matter more than search-engine “published ago” labels. A current repository README may describe later work; its current state is not automatically evidence that a method or implementation existed before this cutoff. Sources discovered with a 2027 publication label and no verified pre-cutoff version were not used to establish prior art.

Matrix notation: **Yes** is directly supported; **No** is reserved for an explicit objective or design mismatch; **N/A** means the column does not apply; **UNKNOWN** means the inspected evidence does not establish the property. “Downstream task objective” distinguishes an optimization/stopping target from merely reporting an IR metric. “Explicit risk control” distinguishes a formal statistical guarantee from a predicted threshold or empirical validation. “Full fixed frontier” requires a substantial fixed-configuration sweep, rather than one convenient fixed comparator.

## Comparison matrix

| Work | Query adaptive | In-traversal | ANN Recall objective | Downstream task objective | Explicit risk control | Judgment sparsity studied | Full fixed frontier studied | Same-ranking qrel intervention |
|---|---|---|---|---|---|---|---|---|
| Li et al., LAET (2020) [1] | Yes | Yes, intermediate search features | Yes | No, exact-neighbor target | UNKNOWN; learned termination | UNKNOWN | UNKNOWN | UNKNOWN |
| Busolin et al., early exit in dense retrieval (2024) [2] | Yes | Yes, incremental IVF cluster visits | Yes, neighbor recovery | IR effectiveness evaluated; explicit utility target UNKNOWN | UNKNOWN; patience/cascade | UNKNOWN | UNKNOWN | UNKNOWN |
| Teofili and Lin, Patience in Proximity (2025) [3] | Yes | Yes, HNSW | Neighbor identity stability proxy | nDCG evaluated; no qrel-based stopping objective | No formal risk bound in described rule | UNKNOWN | UNKNOWN | UNKNOWN |
| Chatzakis et al., DARTH (2025/2026) [4] | Yes | Yes, HNSW/IVF | Yes, declarative recall | No, recall prediction target | Empirical target attainment; not a per-query guarantee | UNKNOWN | UNKNOWN | UNKNOWN |
| Zhang and Miller, Ada-ef (2025/2026) [5] | Yes | Yes, adaptive HNSW ef | Yes, declarative recall | No, recall estimation target | Approximate target via distributional model | UNKNOWN | Yes, recall/latency sweeps; downstream-risk frontier UNKNOWN | UNKNOWN |
| Horchidan et al., ConANN (2025) [6] | Yes | Yes, adaptive IVF probing | Yes, false-negative rate | RAG evaluated; calibrated loss is ANN FNR | Yes, conformal risk control under assumptions | UNKNOWN | UNKNOWN | UNKNOWN |
| Chen et al., Iceberg (2025/2026) [7] | Method-selection study; query stopping N/A | Benchmark, N/A | Compared with task metrics | Yes, task-centric evaluation | N/A | UNKNOWN | Parameter comparisons; full downstream-risk frontier UNKNOWN | UNKNOWN |
| Kuffo et al., Semantic Recall (2026) [8] | Metric/tuning study; N/A | N/A | Replaces exact-overlap target | Yes, semantic/tolerant recall | N/A | UNKNOWN; relevance availability discussed | Cost-quality tuning compared; full risk frontier UNKNOWN | UNKNOWN |
| Dimitropoulos and Mamoulis, Recall What Matters (2026) [9] | Metric/benchmark study; N/A | N/A | Critiqued; inverse ratio evaluated | Yes, classification/RAG evaluation | N/A | UNKNOWN | Yes, cost-accuracy curves; downstream violation frontier UNKNOWN | UNKNOWN |
| FARS, label-histogram HNSW stopping [10] | Yes, claimed | Yes, claimed checkpointed traversal | No, label distribution stability | Yes, Label Recall target | No formal bound shown | UNKNOWN | UNKNOWN | UNKNOWN |
| FARS, BH-Exit [11] | Yes, claimed | Yes, claimed checkpointed traversal | ID-overlap comparator | Yes, nDCG-constrained tuning | Empirical cross-validation threshold | UNKNOWN | UNKNOWN | UNKNOWN |
| Li et al., CTR (2026, v3) [12] | Yes | Recovery reuses graph/distances | Yes, exact-neighbor recall | No, downstream safety explicitly distinct | Yes, population certification | UNKNOWN | UNKNOWN | UNKNOWN |
| Loh and Lee, PQR-Search (2026, v2) [13] | Yes | Yes, pruning-based IVF search | Yes, per-query recall | Aggregate task reported; recall guarantee target | Yes, randomization-based per-query claim under stated assumptions | UNKNOWN | UNKNOWN | UNKNOWN |
| Buckley and Voorhees, bpref (2004) [14] | N/A | N/A | No, IR evaluation | Yes, relevance evaluation | N/A | Yes | N/A | Yes, judgments reduced with retrieval runs fixed |
| Sakai and Kando, condensed evaluation (2008) [15] | N/A | N/A | No, IR evaluation | Yes, relevance evaluation | N/A | Yes | N/A | Yes, incomplete-assessment experiments |
| Sakai, biased assessments (2009) [16] | N/A | N/A | No, IR evaluation | Yes, relevance evaluation | N/A | Yes, pool bias | N/A | Yes, pools reduced for fixed evaluation runs |
| Existing CURA, frozen historical study | Yes | No, probe plus possible fresh search | Comparator/risk proxy | Yes, nDCG-loss threshold | Calibrated predicted risk plus observed validation; no universal guarantee | Yes, TREC 2019 | Yes, current fixed HNSW grid | Yes |
| Proposed extension, before outcomes | Policy-dependent | Only if a genuine one-pass implementation is built | Held fixed as a control | FULL-reference safety disagreement | Estimation protocol; no new guarantee yet | Planned independent replication | Required comparison | Required invariant-ranking design |

## Adaptive search and overlap with CURA

**Learned adaptivity is established.** LAET uses intermediate search evidence and learned models to reduce query-specific search while retaining neighbor-search accuracy [1]. Dense-retrieval early exit already includes patience and cascade strategies over IVF clusters [2]. Patience in Proximity transfers candidate-set stability to HNSW and reports nDCG on BEIR [3]. Consequently, neither adaptive effort nor reporting downstream nDCG distinguishes CURA by itself.

**Declarative recall is an existing objective.** DARTH predicts achieved recall repeatedly during traversal and adjusts prediction intervals [4]. Ada-ef estimates an appropriate exploration factor using query–data similarity distributions rather than a learned utility model [5]. ConANN provides a stronger statistical comparator: its calibrated loss controls ANN false negatives, not query-specific nDCG loss [6]. A CURA probability estimate is not interchangeable with such a guarantee. Equal observed failure counts do not establish risk equivalence.

**Recent risk work strengthens the comparison requirements.** CTR combines acceptance, recovery with reused distances, and population certification [12]. PQR-Search states a per-query probability guarantee over index randomization, while also reporting a downstream aggregate [13]. These claims have different probability spaces and assumptions from CURA; a risk comparison must state the estimand rather than treating “calibration” as a shared guarantee.

## Task-aware ANN and recent utility studies

Iceberg explicitly evaluates vector search using application outcomes across text, image, recognition, and recommendation tasks [7]. Semantic Recall removes semantically irrelevant exact neighbors from its relevance-aware target and proposes Tolerant Recall as a proxy [8]. Recall What Matters studies inverse distance ratio alongside classification and RAG quality and shows why exact-overlap recall can exaggerate approximation costs [9]. These are direct prior art against a broad claim that CURA discovers downstream utility differs from ANN recall.

The located FARS manuscript proposes HNSW stopping from stability of retrieved label histograms [10]. A related manuscript, BH-Exit, uses clustered candidate histograms and tunes against an nDCG degradation constraint [11]. Both documents disclose automated generation. The label-stability PDF was searchable but direct fetch timed out; BH-Exit was directly inspected. An authoritative publication date and successful code reproduction were not established, and the path `live_live_20260213` is not sufficient proof of publication date. Treat these as **provisional public technical disclosures**, not validated peer-reviewed results. They still prevent an unqualified claim that task-aware HNSW stopping has never been proposed.

**Specific audit answers:**

- **Task-aware HNSW early stopping:** yes, publicly described in the provisional label-histogram work and BH-Exit; Patience already evaluates HNSW stopping with IR metrics. Their claims and provenance must be distinguished.
- **Downstream utility instead of exact-neighbor recall:** yes, substantial verified overlap in Iceberg, Semantic Recall, and Recall What Matters.
- **Judgment sparsity changing adaptive ANN safety/budget decisions:** no directly matching study was located in the inspected ANN sources. This remains UNKNOWN as a field-wide priority claim; broader incomplete-qrel distortions and fixed-run thinning are established.

## Incomplete judgments: established results and methodological implications

Buckley and Voorhees show that incomplete judgments can distort standard retrieval evaluation and introduce bpref [14]. Their reduced-judgment experiments already hold retrieval output fixed. Sakai and Kando study condensed evaluation, in which unjudged results are removed before evaluating the resulting ranked list [15]. Sakai demonstrates that robustness under random deletion need not survive system-biased or top-of-pool missingness [16]. Therefore, random positive thinning tests one intervention, not every realistic pooling process.

The extension should attribute bpref and condensed-list nDCG to that literature. It should not promise that either metric mitigates false safety. Standard nDCG, condensed nDCG, and bpref have different scales and reference values; each safety loss must compare approximate and exact rankings using the **same metric and same qrels condition**. The primary nDCG threshold is not automatically a meaningful bpref tolerance. If 0.025 is reused, label it an explicit sensitivity convention and add a justified threshold sensitivity analysis.

Official `trec_eval` documents the bpref computation [17]. It ignores unjudged documents and depends on judged relevant and nonrelevant items. A zero-relevant/zero-nonrelevant case requires explicit handling, rather than silently returning a convenient safe score. Condensed-list nDCG@10 over only a saved top-10 prefix is **prefix-truncated condensed evaluation**, because removal may expose unseen ranks below 10. Genuine condensed nDCG@10 requires enough saved ranking depth to obtain ten judged items or a declared exhaustion condition. Re-running search with larger `k` can alter ANN traversal, so such runs cannot silently replace the fixed historical rankings.

Positive-only thinning with all original nonpositives retained is asymmetric missingness. Report this mechanism explicitly and preserve original grades. FULL-reference disagreement identifies **evaluation-induced false safety relative to available pooled judgments**, not harm relative to complete relevance truth. Budget optimism and headroom inflation are algebraically identical when ANN-safe budget is invariant; do not count them as independent confirmations.

## Official implementation feasibility, before reproduction

| Method | Author-linked source | Reproduction feasibility and required evidence |
|---|---|---|
| DARTH | [MChatzakis/DARTH](https://github.com/MChatzakis/DARTH), author link in [4] | Native FAISS/LightGBM integration, CMake, vector/ground-truth formats and trained predictor required. Current README mixes DARTH and DARTH+ and advertises a 2027 preprint; pin an original/pre-cutoff commit and preserve method identity. Exact compatible commit: **not established by this audit**. |
| Patience in Proximity | [Apache Lucene PR 14094](https://github.com/apache/lucene/pull/14094), explicitly cited in [3] | Author-linked Java/Lucene change; paper uses Lucene 9.11.1 and Anserini. Porting requires matching saturation rule and traversal semantics; describe a port as a faithful implementation, not execution of the official Java artifact. Exact reproducible commit/configuration remains to be established. |
| Ada-ef | [chaozhang-cs/hnsw-ada-ef](https://github.com/chaozhang-cs/hnsw-ada-ef), linked from [5] | hnswlib/C++17 with Eigen, Boost, HDF5 and CMake; experiments include DARTH comparison drivers. This is a different graph implementation from the saved FAISS index. Same embeddings are possible only if available; same index topology is not automatic. Pin commit and record any corpus/index difference. |
| LAET | [efficient/faiss-learned-termination](https://github.com/efficient/faiss-learned-termination), author-linked in [1] | Historical native FAISS fork; useful reference for faithful learned stopping, subject to build/dependency audit. |
| IVF patience | [francescobusolin/faiss_pEE](https://github.com/francescobusolin/faiss_pEE), linked in [2] | Official dense-retrieval early-exit code; it is not HNSW Patience in Proximity. |

No native build or experiment was run for this literature audit. Code availability is not a successful reproduction. Local ANN-Risk and Patience-inspired policies remain local proxies.

## Novelty statement that survives this audit

> We investigate evaluation-induced false safety in utility-aware ANN allocation: with retrieval rankings, search work, and ANN recall held fixed, we quantify how removing relevance judgments changes safety labels and apparent safe budgets relative to the available FULL-qrel reference. We test replication across collections, dependence on search policy and fixed-budget comparators, and the behavior of incomplete-judgment evaluation metrics.

This statement specifies the intended design, not a result. The final manuscript must replace intentions with measured outcomes, retain heterogeneous or negative findings, and restrict any cross-collection claim if the corpus or ranking provenance prevents a valid TREC 2020 replication. A one-pass controller, if implemented, is an additional algorithmic result only if measured against the full fixed frontier including controller and feature overhead.

## Primary sources

1. Conglong Li, Minjia Zhang, David G. Andersen, Yuxiong He. **Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination.** SIGMOD 2020. [Author/institutional record](https://pdl.cmu.edu/PDL-FTP/BigLearning/mod0246-liA_abs.shtml); [DOI](https://doi.org/10.1145/3318464.3380600).
2. Francesco Busolin, Claudio Lucchese, Franco Maria Nardini, Salvatore Orlando, Raffaele Perego, Salvatore Trani. **Early Exit Strategies for Approximate k-NN Search in Dense Retrieval.** CIKM 2024; arXiv v1 9 August 2024. [Versioned paper](https://arxiv.org/html/2408.04981v1); [DOI](https://doi.org/10.1145/3627673.3679903).
3. Tommaso Teofili, Jimmy Lin. **Patience in Proximity: A Simple Early Termination Strategy for HNSW Graph Traversal in Approximate k-Nearest Neighbor Search.** ECIR 2025. [Author PDF](https://cs.uwaterloo.ca/~jimmylin/publications/Teofili_Lin_ECIR2025.pdf).
4. Manos Chatzakis, Yannis Papakonstantinou, Themis Palpanas. **DARTH: Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search.** arXiv v1 25 May 2025; inspected v2 18 August 2025; PACMMOD 3(4), September 2025, SIGMOD 2026. [Versioned paper](https://arxiv.org/html/2505.19001v2); [DOI](https://doi.org/10.1145/3749160).
5. Chao Zhang, Renée J. Miller. **Distribution-Aware Exploration for Adaptive HNSW Search.** arXiv v1 7 December 2025; PACMMOD 4(1), February 2026. [Versioned paper](https://arxiv.org/html/2512.06636v1); [DOI](https://doi.org/10.1145/3786639).
6. Sonia Horchidan, Fabian Zeiher, Henrik Boström, Paris Carbone. **ConANN: Conformal Approximate Nearest Neighbor Search.** PVLDB 19(1):29–42, 2025 (paper reference date). [Proceedings PDF](https://www.vldb.org/pvldb/vol19/p29-horchidan.pdf); [DOI](https://doi.org/10.14778/3772181.3772184).
7. Tingyang Chen and colleagues. **Reveal Hidden Pitfalls and Navigate Next Generation of Vector Similarity Search from Task-Centric Views.** arXiv v1 15 December 2025; inspected v2 31 December 2025. [Versioned Iceberg paper](https://arxiv.org/html/2512.12980v2).
8. Leonardo Kuffo, Ioanna Tsakalidou, Roberta De Viti, Albert Angel, Jiří Iša, Rastislav Lenhardt. **Semantic Recall for Vector Search.** arXiv v1 22 April 2026; SIGIR 2026. [Versioned paper](https://arxiv.org/html/2604.20417v1); [DOI](https://doi.org/10.1145/3805712.3809894).
9. Dimitris Dimitropoulos, Nikos Mamoulis. **ANN Search: Recall What Matters.** arXiv v1 3 June 2026. [Versioned paper](https://arxiv.org/html/2606.04522v1).
10. FARS / Analemma. **Task-Aware Early Termination for HNSW via Label-Histogram Stabilization.** Publication date/peer-review status UNKNOWN; public search disclosure inspected 18 November 2026. [Primary PDF](https://lemma-public-asset.analemma.ai/online/fars/live/live_live_20260213/idea_29a2ce03-c573-41ec-a413-a7c46e3b5aa6/main.pdf); [claimed code](https://gitlab.com/fars-a/vss-label-stability-early-exit). Direct PDF fetch failed; indexed primary-text excerpts supported the limited method description above.
11. FARS / Analemma. **BH-Exit: Label-Free Early Termination for HNSW Search via Bucket-Histogram Stability.** Date/peer-review status UNKNOWN; PDF inspected 18 November 2026. [Primary PDF](https://lemma-public-asset.analemma.ai/online/fars/live/live_live_20260213/idea_e67e388a-1487-487b-a328-30a5451364ed/main.pdf); [claimed code](https://gitlab.com/fars-a/vss-clusterid-hist-stability-early-exit).
12. Minghao Li, Raghav Mittal, Sanjivni Rana, Suraj Shetiya, Gautam Das, Nick Koudas. **Graph-Based Nearest-Neighbor Search with Recall Guarantees Using Graph Spanners.** arXiv v1 2 July 2026; inspected v3 11 November 2026. [Versioned CTR paper](https://arxiv.org/html/2607.02338v3).
13. Woong-Kee Loh, Suan Lee. **Per-Query Recall Guarantees for Approximate Nearest Neighbor Search in Dynamic Vector Databases.** arXiv v1 5 November 2026; inspected v2 11 November 2026; preprint. [Versioned PQR-Search record](https://arxiv.org/abs/2611.07579v2).
14. Chris Buckley, Ellen M. Voorhees. **Retrieval Evaluation with Incomplete Information.** SIGIR 2004. [NIST record](https://www.nist.gov/publications/retrieval-evaluation-incomplete-information); [Author PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469); [DOI](https://doi.org/10.1145/1008992.1009000).
15. Tetsuya Sakai, Noriko Kando. **On Information Retrieval Metrics Designed for Evaluation with Incomplete Relevance Assessments.** Information Retrieval, 2008. [Publisher/DOI](https://doi.org/10.1007/s10791-008-9059-7). See also Sakai, **Alternatives to Bpref**, SIGIR 2007, [DOI](https://doi.org/10.1145/1277741.1277756).
16. Tetsuya Sakai. **On the Robustness of Information Retrieval Metrics to Biased Relevance Assessments.** Journal of Information Processing 17:156–166, 2009. [Publisher primary article](https://www.jstage.jst.go.jp/article/ipsjjip/17/0/17_0_156/_article); [DOI](https://doi.org/10.2197/ipsjjip.17.156).
17. NIST. **trec_eval bpref implementation.** [Official source](https://github.com/usnistgov/trec_eval/blob/main/m_bpref.c); [TREC measure definitions](https://trec.nist.gov/pubs/trec20/appendices/measures.pdf). A runtime implementation should pin the evaluator version, not rely on mutable `main`.

## Dated follow-up: 19 November 2026

The local work session crossed midnight. A focused follow-up search for task-aware HNSW and ANN relevance-judgment publications dated 18–19 November located no additional verified matching primary study. This is not a completeness guarantee. The original 18 November audit and prospective metric protocol remain the analysis-design record; this dated literature update does not retrospectively change the registered estimands.

Official GitHub history was subsequently queried for implementation provenance. The original-paper DARTH release selected for the build attempt is `0d9bafcf31d1d79668bc71139fe93fa5e70b5185` (UTC 19 August 2025), before its November 2026 release/refactor; Ada-ef's latest located pre-cutoff commit is `ed463f9993868f7ecc7c103920644e7f94abb377` (19 June 2026). The author-linked Lucene Patience change merged as `525bf34bfdfc16cc220d326d2cf30541f1afef29`. These supersede only the earlier table's unresolved commit fields; they do not establish successful reproduction. Build evidence and limitations are recorded separately in `published_baseline_reproduction.md`.
