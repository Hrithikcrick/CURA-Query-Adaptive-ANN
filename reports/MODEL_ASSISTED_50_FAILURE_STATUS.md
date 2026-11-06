# CURA model-assisted 50-case failure-analysis status

**Date:** 6 November 2026  
**Scope:** MS MARCO HNSW query-budget sweep  
**Status:** Model-assisted diagnostic analysis; not independent human annotation

## What was analyzed

A separate failure-analysis pass selected **50 actual IR-unsafe query-budget cases** from the MS MARCO HNSW budget sweep. A case is IR-unsafe when its recorded downstream loss satisfies:

`nDCG@10 loss > 0.025`.

These are **query-budget failures observed in the sweep**. They are **not** 50 failures of the final deployed CURA operating policy.

## Annotation process

- The 50-row analysis completed the project's automatic structural validation.
- All rows retained their original query IDs and recorded numeric experiment values.
- Each row received exactly one taxonomy label and an evidence-based note.
- **8 cases were initially flagged as uncertain.**
- Those 8 were resolved in a second **model-assisted** pass using predefined numerical and semantic priority rules.
- Because no independent reviewer signed off all 50 cases, this artifact must **not** be described as "50 manually reviewed failures."

Recommended methodology wording:

> We analyzed 50 actual IR-unsafe query-budget cases using a structured model-assisted taxonomy. Cases initially flagged as uncertain were resolved through a second evidence-based model-assisted review using predefined numerical and semantic decision rules.

## Relationship to the FiQA review worksheet

This MS MARCO analysis is separate from the existing BEIR FiQA human-review worksheet. The FiQA worksheet contains true fixed-HNSW low-budget failures selected for human inspection, but assistant-prepared drafts are not equivalent to independent human judgments.

## Headline results that remain unchanged

The failure-analysis pass does **not** modify the core saved experiment metrics:

- Fixed-IR realized work: **3431.770** distance computations/query.
- CURA-Full realized work: **2727.965** distance computations/query.
- Realized-work saving vs Fixed-IR: **20.5085%**.
- CURA-Full observed downstream violation rate: **1.0%** under the nominal **5%** target.
- Paired work difference: **703.805** computations/query.
- Bootstrap 95% CI: **[401.434, 984.115]**.
- Paired randomization: **p ≈ 5.0×10^-5**.
- Mean latency: **0.8328545 ms → 0.6907383 ms** (Fixed-IR → CURA-Full).
- p95 latency: **1.128008 ms → 1.985027 ms**. This is an important tail-latency trade-off: the current CURA implementation improves average latency but worsens p95 latency.

## Claim boundaries

Do not claim any of the following from this artifact:

- 50 independent human reviews;
- 50 deployed-CURA failures;
- official DARTH reproduction;
- official Patience reproduction;
- full 8.8M-passage MS MARCO evaluation;
- universal cross-domain efficiency gains;
- a distribution-free 5% risk guarantee.

Exact generated taxonomy CSVs from the local finalization workspace should be committed before citing category-count percentages in the paper.
