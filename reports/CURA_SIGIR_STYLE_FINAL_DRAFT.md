# Beyond Recall: CURA — Calibrated Utility-Risk Allocation for Query-Specific Approximate Nearest Neighbor Search

**Working COL764 final-paper draft — evidence-grounded, not final submitted manuscript**  
**Author:** Hrithik Sharma, Indian Institute of Technology Delhi  
**Version:** 2026-11-03  
**Implementation and results:** [CURA GitHub repository](https://github.com/Hrithikcrick/CURA-Query-Adaptive-ANN)

> **Submission audit:** This draft integrates later FiQA results and the 6 November 2026 MS MARCO failure-analysis status. The earlier manuscript's claim of 50 *manually inspected* cases is **not substantiated by the public human-review worksheet**. A separate 50-case MS MARCO analysis is now structurally validated and model-assisted, with eight initially uncertain cases resolved in a second model-assisted pass; it is **not** independent human annotation.

## Abstract

Approximate nearest-neighbor (ANN) indexes such as hierarchical navigable small world (HNSW) graphs usually expose a global search-effort parameter. Larger values can improve nearest-neighbor recall at increased computational cost, although exact-neighbor reconstruction is not identical to preserving downstream information-retrieval (IR) quality. We study **CURA**, a query-specific HNSW budget controller that predicts the probability of losing more than a specified amount of nDCG@10 relative to exact dense retrieval. The controller combines query features with a cheap partial HNSW probe, calibrates its risk predictions, and selects a budget using an empirically constrained policy. On a controlled collection of **1,000,629 MS MARCO passages** and **200 held-out queries**, CURA-Full reduces realized distance computations by **20.51%** versus calibrated fixed-budget HNSW (**3,431.77 → 2,727.97 per query**) with **1.0%** observed downstream-loss violations against a nominal **5%** risk target. The paired work reduction is **703.81 computations/query** (95% bootstrap CI **[401.43, 984.11]**, paired randomization **p≈5×10⁻⁵**). However, a reduced-feature controller fitted separately on BEIR FiQA uses **75.27% more** computation than a calibrated fixed baseline, exposing limitations in feature quality and repeated-search overhead. Together, the results motivate evaluating adaptive ANN policies on downstream utility, realized computation and genuine distribution shifts—not ANN recall or selected budget alone.

**Keywords:** approximate nearest neighbors; HNSW; dense retrieval; query-adaptive search; downstream utility; risk calibration; nDCG; reproducibility.

## 1. Introduction

Dense retrieval maps queries and passages into a shared vector space. Approximate nearest-neighbor (ANN) indexes reduce the cost of searching a large corpus, but the index must trade off search effort against the accuracy of the returned neighbor set. In HNSW, \`efSearch\` controls how extensively the graph is explored. A fixed \`efSearch\` treats simple and difficult queries equally.

Conventional ANN tuning often targets near-perfect recovery of exact vector neighbors. Yet for information retrieval, the application-level outcome is a relevance-ranked list. Missing some exact neighbors need not reduce nDCG if the replacements are similarly relevant; conversely, recovering many exact neighbors can still miss the one passage whose rank matters most. Therefore, minimizing neighbor-recall violations can allocate more search work than necessary for the downstream task.

**Hypothesis H1.** A query-adaptive HNSW controller using cheap query and partial-search features can reduce realized distance computations compared with a calibrated fixed search budget, while keeping the *observed rate* of nDCG degradation above tolerance at an acceptable level on held-out data.

**Backoff hypothesis.** Even where a practical controller fails to save work, the minimum ANN-safe and downstream-IR-safe budgets differ across queries, and analyzing those differences can reveal useful system-design limitations.

The intended contribution is a **supervision and retrieval-policy study**, not a new vector index or a claim to be the first adaptive ANN controller. Closely related early-exit and recall-/rank-aware research must be compared explicitly.

## 2. Background and Related Work

HNSW is a navigable graph index for approximate similarity search. FAISS supports exact and approximate dense-vector retrieval. ANN-Benchmarks has emphasized reporting an efficiency–recall frontier, rather than only one operating point. Application-oriented systems such as LADR, score-/termination-based dense retrieval, Patience, Ada-ef, DARTH and rank-aware early-termination approaches are related lines of work.

**Positioning:** CURA supervises a **downstream nDCG-loss event** rather than directly demanding a target ANN recall. In addition to query-only features, it uses measurements available after a cheap HNSW probe. The resulting controller predicts per-query/per-budget risk, calibrates its scores and uses a calibration policy to select an operating point.

**Important baseline limitation:** The current experiments compare a local ANN-risk proxy and a Patience-inspired repeated-search heuristic. These are **not** faithful official reproductions of DARTH or Patience. Therefore the experiments do not establish superiority over those published implementations. Verify the final bibliography against the papers cited in the earlier \`main_corrected.tex\` source.

## 3. Problem Formulation

Let \`q\` be a query and \`e\` a candidate HNSW \`efSearch\` value. Define the downstream ranking effectiveness

\[
U(q,e)=\mathrm{nDCG@10}\text{ under budget }e,\qquad
U^\star(q)=\mathrm{nDCG@10}\text{ using exact vector search}.
\]

The loss against exact dense retrieval is

\[
\Delta U(q,e)=U^\star(q)-U(q,e).
\]

A query–budget pair is considered **IR-unsafe** if

\[
y_{\mathrm{IR}}(q,e) = \mathbf{1}\{\Delta U(q,e)>0.025\}.
\]

The ANN-centric comparator labels a pair unsafe if its \`ANN Recall@10\` is below **0.90**. CURA seeks to estimate

\[
r(q,e) =
P\!\left(\Delta U(q,e)>0.025 \mid x_q,S_q,e\right),
\]

where \`x_q\` contains deployable query features and \`S_q\` contains the partial-search signals from a cheap probe. Given a calibrated decision threshold \`τ\`, it chooses the smallest candidate \`e\` with estimated risk not exceeding \`τ\`, with a high-budget fallback if none qualifies.

**Clarification:** The quoted 5% target concerns the rate of *large downstream nDCG losses*, not an allowed nDCG value of 5% and not a formal guaranteed per-query failure probability under all distribution shifts.

## 4. Method: CURA

### 4.1 Cheap features available at inference

**Query features:** character and token counts, mean token length, digit/question-mark indicators, candidate budget, and projection of a query embedding using PCA fitted only to training queries.

**Partial HNSW-state features:** a probe search at \`efSearch=10\` produces top score statistics, score gaps, graph hops and HNSW distance-computation counts. The probe does not access exact neighbors, relevance judgments or downstream test scores.

### 4.2 Risk prediction

The implementation trains a **histogram gradient-boosted classifier** on query–budget pairs, supervised by the IR-unsafe binary labels. The full controller uses the query-plus-probe features; a query-only ablation excludes the partial-search signals. An ANN-risk comparator uses ANN-recall-centric supervision rather than downstream nDCG loss.

### 4.3 Calibration and operating-point selection

Raw risk scores are calibrated using **isotonic regression** on a calibration query split. Candidate policies are evaluated on calibration queries using an empirical downstream violation rate and a conservative Wilson upper-confidence bound. The selected operating point trades off observed search work and the nominal risk limit.

**Calibration limitation:** In parts of this pipeline, probability fitting and threshold/policy selection reuse the same calibration split. This can create optimistic selection effects. A nested calibration/policy-selection procedure or fresh validation split would be needed for a stronger risk-guarantee claim.

### 4.4 Realized cost accounting

The main cost is **actual FAISS/HNSW distance computations**, not the chosen \`efSearch\` alone. CURA-Full's cost includes the extra probe plus the selected search when a separate run is required. For comparison, a repeated-search stability baseline includes the work of every search it runs. Latency is also recorded, though hardware and runtime conditions must be independently reproduced before interpreting small absolute differences as universal.

## 5. Experimental Setup

### 5.1 Controlled MS MARCO study

- Passage corpus: **1,000,629** (controlled subset; **not** the full 8.8M-passage MS MARCO collection).
- Approximately **1,000 MS MARCO development queries** are partitioned by query into **60% training / 20% calibration / 20% held-out testing**; the reported held-out comparison uses **200 queries**.
- All **43 TREC-DL-2019** queries are reserved for an OOD query-set evaluation. This is a shift in query/relevance-judgment regime on the **same passage domain**, not independent-domain transfer.
- HNSW parameters include \`M=32\`, \`efConstruction=200\`, probe \`ef=10\`, and seed **42**. The IR loss tolerance is **0.025**, ANN recall target **0.90**, and nominal risk target **0.05**.
- Methods: Fixed-IR HNSW; local ANN-Risk; CURA-Query; CURA-Full; Patience-inspired stability; diagnostic ANN and IR oracles. Oracles access the outcomes of the candidate budget grid and are **not deployable**.
- Metrics: nDCG@10, ANN Recall@10, mean/p95 work and latency, downstream violation rate, calibration measures and query-level statistical tests.

### 5.2 Financial-domain BEIR FiQA study

- **57,638 financial-domain documents** and **648 FiQA benchmark test queries** were included in the HNSW budget sweep.
- For the subsequent *exploratory within-FiQA* controller, those 648 published test queries were partitioned into **388 development/training**, **129 calibration**, and **131 locally held-out** queries.
- This re-use of official test queries for training means these numbers **must not** be presented as untouched official BEIR test-set results.
- The FiQA model is a newly fitted **reduced-feature CURA-style controller**, using query features and \`ef=10\` probe work/latency. It is **not** zero-shot transfer of the MS MARCO-trained CURA-Full weights, and it uses different feature information.
- Calibration selected fixed-HNSW \`ef=32\` as the comparison operating point. Both methods met the chosen calibration-set Wilson requirement; the held-out confidence bounds are reported separately.

## 6. Results

### 6.1 MS MARCO: main held-out comparison

| Method | Mean selected ef | IR violation rate ↓ | Mean nDCG@10 ↑ | Total realized work/query ↓ | Mean latency (ms) ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed-IR | 96.00 | 0.5% | 0.625260 | 3,431.770 | 0.8329 |
| ANN-Risk (local proxy) | 84.46 | 0.5% | 0.625260 | 3,930.615 | 0.9580 |
| CURA-Query | 225.92 | 0.0% | 0.626970 | 7,130.760 | 1.7156 |
| **CURA-Full** | **51.97** | **1.0%** | **0.623363** | **2,727.965** | **0.6907** |
| Patience-Inspired | 62.32 | 0.5% | 0.625260 | 8,413.945 | 2.1693 |
| ANN Oracle† | 22.50 | 0.5% | 0.626896 | 1,208.020 | 0.3506 |
| IR Oracle† | 11.51 | 0.0% | 0.630348 | 779.835 | 0.2697 |

† Oracle numbers indicate potential headroom only; do not compare them as deployment-ready methods.

**Core finding:** CURA-Full reduces realized work by **20.5085%**, corresponding to **703.805 fewer distance computations/query** relative to Fixed-IR. The paired bootstrap 95% CI for the **mean work difference** is **[401.434, 984.115]** and the paired work randomization test returns **p≈5.0×10⁻⁵**. This is evidence for an efficiency improvement on this particular held-out setting, **not a significance claim about the nDCG difference**. The recorded mean latency also improves (0.8329→0.6907 ms), but p95 latency worsens (1.1280→1.9850 ms), so the current implementation does not improve the tail.

The observed downstream violation rate is **1.0%** (versus **0.5%** for fixed); the mean nDCG@10 also falls slightly from **0.625260** to **0.623363**. Both quality and efficiency must be reported.

### 6.2 Two substantive ablations

1. **Target matters:** the local ANN-Risk comparator uses **3,930.615** total computations/query, versus **2,727.965** for CURA-Full, a **30.60% work reduction** relative to that comparator. This supports the importance of using the downstream IR event as the supervised target in this setup.
2. **Partial search matters:** CURA-Query uses **7,130.760**, versus **2,727.965** for CURA-Full, a **61.74% reduction** relative to the query-only variant. The result suggests early search-state evidence is important to avoid excessive conservatism.

A third internal comparator, Patience-Inspired, needs **8,413.945** computations/query and is **not** the official published implementation.

### 6.3 TREC-DL-2019: query-distribution shift

On **43 held-out queries** over the same MS MARCO passage domain, CURA-Full records **2,384.581** computations/query versus **2,721.884** for Fixed-IR, an observed **12.39%** work reduction. Its observed IR violation rate is **2.33%** compared with **0%** for Fixed-IR. This is a small-sample query-shift result, not a universal guarantee nor an independent-domain replication.

### 6.4 FiQA: contrasting negative result

| Metric | Calibrated Fixed-HNSW | Reduced-feature CURA-FiQA |
|---|---:|---:|
| Query count | 131 | 131 |
| Mean chosen ef | 32.00 | 47.74 |
| Mean realized work ↓ | **1,097.084** | 1,922.870 |
| Mean latency (ms) ↓ | **0.214905** | 0.385635 |
| p95 latency (ms) ↓ | **0.28940** | 0.86645 |
| Mean nDCG@10 ↑ | **0.220140** | 0.217867 |
| IR violation rate ↓ | 3.05% | 3.05% |
| Held-out Wilson 95% upper risk bound ↓ | 7.59% | 7.59% |

**Negative result:** CURA-FiQA uses **75.27% more realized work**, not less, and has slightly lower mean nDCG@10 at the same observed violation rate. The held-out **7.59% Wilson upper bound** exceeds the nominal 5% target, so neither method establishes a 95%-confidence 5% risk bound in this split.

The failure audit shows **51/131** queries receiving \`ef>32\`, **112/131** requiring a second search, and **451.32** average extra probe computations/query. Of the 131 held-out queries, **125** were safe for both methods, **2** failed for both, **2** failed only for fixed, and **2** failed only for CURA-FiQA.

This does **not** show that *MS MARCO-trained CURA-Full* fails in zero-shot finance-domain transfer. It shows that the **separately trained, reduced-feature FiQA variant** does not achieve positive savings under this setup.

## 7. Error Analysis

The earlier MS MARCO manuscript describes **50 high-priority diagnostic cases** with a nine-category summary; only **three** of those selected cases are recorded as actual downstream violations. The existing public records are insufficient to label that selection a verified independent manual analysis of 50 genuine failures.

A separate MS MARCO analysis now selects **50 actual IR-unsafe query-budget cases** from the HNSW budget sweep. The structured annotation file passes automatic validation over all 50 rows. **Eight initially uncertain cases were resolved in a second model-assisted pass** using predefined numerical and semantic decision rules. This is useful diagnostic evidence, but it remains **model-assisted rather than independently human-verified**. The cases are query-budget failures from the sweep, not 50 failures of the final deployed CURA policy.

In the FiQA budget sweep, **65 unique queries** have a recorded IR-unsafe outcome at fixed \`efSearch=10\`. A separate worksheet selected **50 actual low-budget HNSW failures** and reconstructed the saved metrics successfully for all 50. Of the selected outcomes, **44** have approximate nDCG@10 equal to zero, and **6** have degraded but nonzero approximate nDCG@10.

**FiQA annotation status:** assistant-prepared evidence descriptions are available, but independent human review remains incomplete. Until committed and audited human judgments are present, do **not** report a completed 50-failure human taxonomy.

**Boundary:** these 50 failures belong to **fixed HNSW at ef=10**, not to the deployed CURA-FiQA controller. They can help explain low-budget ANN behavior, but are not a sample of 50 CURA policy failures.

## 8. Discussion

The MS MARCO experiment supports the hypothesis that quality-constrained search can tolerate some deviations from exact nearest-neighbor identity and can reduce realized work. On that dataset, CURA-Full incurs **10%** ANN-target violations while the downstream IR-violation rate remains **1%**, illustrating the distinction between ANN and IR objectives.

The FiQA experiment highlights three potential limits: (i) reduced probe features may not estimate risk sufficiently well, (ii) conservative policies may select unnecessarily high budgets, and (iii) repeated-search probe overhead can overwhelm theoretical savings. These are plausible interpretations from the observed statistics; **causality requires controlled ablations** on development data.

A stronger implementation could reuse the HNSW search frontier/visited state after the initial probe rather than rerunning graph traversal. This is **future work**, not an implemented method or measured improvement.

## 9. Limitations and Threats to Validity

1. **Controlled corpus size:** MS MARCO uses ~1M passages, not the full 8.8M benchmark, so the scores are not official full-corpus effectiveness scores.
2. **Metric target:** nDCG loss is computed against exact dense retrieval under available relevance judgments. Sparse or incomplete qrels affect the observed utility event.
3. **Query sampling:** MS MARCO uses 200 held-out queries; TREC-DL uses 43. FiQA uses 131 locally held-out queries after repurposing official test queries for development.
4. **No proven risk guarantee:** A low empirical rate is not a distribution-free guarantee. The calibration/policy split reuse and distribution shift limit any SLA-like claim.
5. **No official adaptive-baseline reproduction:** ANN-Risk and Patience-inspired comparisons do not substitute for Ada-ef/DARTH or published Patience code on matched conditions.
6. **Reproducibility gap:** source and many result artifacts are public, but raw corpora, embedding caches, FAISS indexes, index hashes, and some run conditions are not included. Full independent reruns remain to be verified.
7. **System cost:** extra probe searches, model inference, CPU setup and wall-clock variance should be accounted for; distance computations are a strong counter but not the only deployment cost.
8. **Annotation boundary:** the new 50-case MS MARCO taxonomy is model-assisted (including second-pass resolution of eight uncertain cases), while the FiQA independent human review remains incomplete. Neither should be overstated as 50 independently reviewed failures.
9. **Domain transfer not tested:** the FiQA variant is trained anew; direct zero-shot transfer of the exact CURA-Full model was not evaluated.

## 10. Conclusion

CURA demonstrates that a downstream-risk objective can reduce HNSW search work under the reported controlled MS MARCO setting: **20.51% fewer realized computations/query**, supported by paired uncertainty testing, with **1% observed nDCG-loss violations** against a nominal 5% target. Yet the simplified finance-domain variant is **75.27% more expensive** than a calibrated fixed baseline. The combined evidence argues for quality-aware but *systems-conscious* adaptive ANN policies, careful treatment of calibration and reproducibility, and transparent reporting of negative results.

## 11. Generative-AI Assistance and Research Integrity

Generative-AI assistants were used to help structure code, prepare audit utilities, draft documentation, summarize saved experiment outputs, and generate **preliminary—not human-verified—failure-analysis notes**. All reported numeric results are drawn from committed CSV/JSON research outputs rather than invented by the assistant. The final submission must specify which experiments the student personally executed and checked, ensure the original code and outputs can be examined, and distinguish assistant-prepared materials from manually verified error analyses. The final paper and all claims remain the author's responsibility.

## References — verification needed before final submission

Use the bibliography already in the earlier \`main_corrected.tex\` manuscript as the starting point. Relevant works listed there include Malkov and Yashunin (HNSW), Johnson et al. (FAISS), Järvelin and Kekäläinen (nDCG), Kulkarni et al. (LADR), Busolin et al. (early exit), Teofili and Lin (Patience), Zhang and Miller (distribution-aware adaptive HNSW), Chatzakis et al. (DARTH), and Lu et al. (rank-aware vector-search termination). **Check the complete metadata, published years, author spellings, and corresponding claims before submitting the bibliography.**

---

### Before converting this draft into the final SIGIR-style PDF

- [ ] Insert correctly cited literature references and verify each novelty comparison.
- [ ] Reproduce the saved-result audit; document exact environment and hardware.
- [ ] Add the original figures from saved experiment artifacts with source captions.
- [x] Complete a structurally validated **model-assisted** 50-case MS MARCO IR-unsafe analysis.
- [ ] Commit the exact generated taxonomy CSVs and complete independent human review if required by the course.
- [ ] Confirm that a clean-environment implementation re-creates at least the key saved output tables.
- [ ] Explicitly label official FiQA query-split reuse and the absence of zero-shot transfer.
- [ ] Provide an accurate, detailed AI-usage statement consistent with course policy.
- [ ] Format to the professor's preferred SIGIR/ECIR-style short-paper length and rehearse the accompanying demo/viva.
