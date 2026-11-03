# Assistant-prepared evidence drafts — FiQA Cases 2–5

These are **assistant-generated suggestions**, NOT completed human reviews. The user must compare each displayed source passage and ranking evidence and confirm, correct, or reject the draft. Do not copy the drafts into `human_*` worksheet fields automatically, mark `human_review_complete_YES_NO=YES` automatically, or describe them as independently completed manual analysis.

All four were selected because the **fixed HNSW efSearch=10** output missed judged-relevant material. They are **not deployed CURA failures**. The stated *mechanism* (why an approximate index chose its candidates) remains a hypothesis unless tested.

## Case 2 — Query 1451
**Query:** How do you find an ethical, honest independent insurance broker in Canada?

**Suggested category:** `1` — Judged-relevant document missed by approximate search.

**What went wrong — draft:** Exact retrieval placed a judged-relevant answer about verifying a professional's honesty through small transactions, personal recommendations and public reviews at rank 1. HNSW at efSearch=10 instead ranked Canada-specific credit-score and taxation passages. The judged-relevant passage was absent from HNSW's top 10, reducing nDCG@10 from 1.0 to 0.0. The prominent Canada/finance context may have attracted loosely related passages, but that possible explanation is not experimentally established.

**Evidence — draft:** Exact rank 1: document `538005`, relevance 1. HNSW rank 1: document `533621`, relevance 0; HNSW rank 2: `440774`, relevance 0. Saved HNSW nDCG@10=0.0.

**Possible investigation/remedy — draft:** Check whether increasing efSearch recovers document `538005`. Separately inspect whether the embedding model confuses country-specific finance information with the query's actual intent about evaluating a broker's reliability.

## Case 3 — Query 4785
**Query:** What is the difference between a structured collar and a normal collar in finance?

**Suggested category:** `1` — Judged-relevant document missed by approximate search. A secondary observation is specialized financial terminology, but don't treat that as a proven cause.

**What went wrong — draft:** The exact top-ranked passage describes a collar as a protective strategy involving a put and a call, directly addressing the question. HNSW's top result instead discusses structured-product debt tranches and the remaining results include general finance topics. No judged-relevant passage was retrieved in the approximate top 10, dropping nDCG@10 from 1.0 to 0.0.

**Evidence — draft:** Exact rank 1: document `37517`, relevance 1. HNSW rank 1: `229582`, relevance 0; HNSW rank 2: `352154`, relevance 0. Saved HNSW nDCG@10=0.0.

**Possible investigation/remedy — draft:** Evaluate larger efSearch values and examine whether a finance-specialized retriever or reranker better distinguishes *collar options strategies* from unrelated uses of *structured products*.

## Case 4 — Query 5592
**Query:** Are “hard money loans” meant only for real estate?

**Suggested category:** `1` — Judged-relevant document missed by approximate search.

**What went wrong — draft:** Exact search retrieved a passage explaining hard-money lending as asset-backed financing commonly secured by real estate. HNSW's top ranks included a general discussion of loans and real savings plus unrelated real-estate/international debt content. The judged-relevant passage was absent from its top 10, giving nDCG@10=0 rather than the exact ranking's 1.0.

**Evidence — draft:** Exact rank 1: document `355241`, relevance 1. HNSW rank 1: `416617`, relevance 0; HNSW rank 2: `245532`, relevance 0.

**Possible investigation/remedy — draft:** Test whether greater search effort retrieves `355241`. Explore term sensitivity around the phrase *hard money loan* versus the broad concepts *loan* and *real estate*.

## Case 5 — Query 721
**Query:** What are “upstream investments” and “downstream investments” in this context?

**Suggested category:** `1` — Judged-relevant document missed by approximate search.

**What went wrong — draft:** Exact search found an answer defining upstream and downstream investments in terms of suppliers, customers and vertical integration. HNSW instead retrieved passages discussing ordinary financial investments, property markets and fixed-income products. The domain-specific explanation did not appear in its top 10, reducing nDCG@10 from 1.0 to 0.0.

**Evidence — draft:** Exact rank 1: document `496225`, relevance 1. HNSW rank 1: `589088`, relevance 0; HNSW rank 2: `576688`, relevance 0.

**Possible investigation/remedy — draft:** Test higher efSearch and inspect whether representations of *upstream/downstream investment* are dominated by the generic word *investment*, obscuring the supply-chain meaning. This is a hypothesis, not a proven index failure cause.

---

## Responsible review workflow

1. Keep your existing local annotations safe; do not rerun the Step 15 worksheet generator.
2. Open the evidence in `16_interactive_fiqa_review.py`.
3. Compare the actual relevant passage and exact/HNSW rankings against these draft statements.
4. Rewrite or accept a suggested category, explanation and remedy **only if supported by your own review**.
5. Mark a case completed only after you have verified and saved your judgment.

The saved nDCG and relevance labels demonstrate the outcome, not necessarily the internal reason HNSW traversed a particular graph path.
