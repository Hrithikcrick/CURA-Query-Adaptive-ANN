# False safety relative to a pooled-judgment reference

FULL denotes the available pooled qrels, not exhaustive relevance truth. The statements below concern disagreement with that reference. They do not prove a user's actual relevance loss. These are elementary identities and sufficient conditions, rather than claims of mathematical priority.

Let `U(q,e)` indicate that a query-budget pair is unsafe under FULL, and let `S_c,s(q,e)` indicate safety under thinned condition `c` and seed `s`. Loss is the positive part of exact minus approximate metric value; safety uses the fixed tolerance 0.025. The primary probability weights queries uniformly and budgets uniformly over the specified ten-element grid, then averages over thinning seeds. That grid is part of the estimand.

## Two false-safe rates and an opportunity ceiling

`FSR_all = P(U=1 and S=1)`

`FSR_given_thin_safe = P(U=1 and S=1) / P(S=1)`

For `P(U=1)>0`, the first rate decomposes as

`FSR_all = P(U=1) * P(S=1 | U=1)`.

Consequently `FSR_all <= P(U=1)`. Collections with almost no FULL-unsafe budget pairs have a small absolute false-safety ceiling even if thinning hides most of their existing unsafe labels. Both the reference-unsafe count and its retention/masking fraction are therefore necessary interpretation aids. They supplement, rather than replace, the preregistered all-pair primary endpoint. With a zero denominator, conditional rates are undefined; publish the count instead of inventing immunity.

In the primary runs, the FULL-unsafe opportunities are 14/430 for 2019 and 1/540 for 2020. At one positive per query, the fraction of those unsafe labels masked is approximately 86.86% and 88.00%, respectively. The latter estimate concerns a single unsafe query-budget pair across thinning seeds. It is not evidence of broad query-population generalization.

## Zero-reference collapse

For an applicable nonnegative metric, if the exact score under thinning is zero, then

`max(0, exact_thin - approximate_thin) = 0`

for every candidate budget. Every candidate is therefore labeled safe under that condition, and the measured first-safe budget equals the grid minimum, 10. This conclusion follows from the loss definition, not improved retrieval.

Define `Z(q,s) = [exact_thin(q,s)=0]`. Then

`FSR_all >= P(U=1 and Z=1)`.

The right side is the zero-reference component of false safety. The measured decomposition additionally reports false-safe pairs whose exact thinned score remains positive. Bpref cases without binary-positive or binary-nonrelevant judgments are inapplicable in this analysis, not instances to which this argument assigns safety.

## Budget optimism and headroom

Let `B_FULL` and `B_THIN` be the first genuinely safe tested IR budgets and `A` the ANN-safe budget. Where both IR minima exist,

`budget_optimism = B_FULL - B_THIN`.

Where the ANN minimum also exists,

`headroom_inflation = (A-B_THIN) - (A-B_FULL) = budget_optimism`.

Thus headroom inflation is not independent corroboration of budget optimism. The identity relies on unchanged ANN rankings/recall and the same feasible query. Missing minima remain missing. Signed optimism can be negative, and safety need not be monotone in efSearch. A raw difference of means over changing feasible cohorts does not estimate the paired-query change; the corpus-control analysis reports both explicitly and interprets the paired version.

## What the intervention identifies

With rankings, work and frozen policy decisions held fixed, changing only the qrels identifies the effect of the specified judgment-removal intervention on measured labels. It does not identify the effect of every real pooling process, actual relevance preservation, or behavior of a controller retrained on sparse labels. Corpus augmentation is a separate intervention that changes both retrieval candidates and graph state; its bundled diagnostic cannot isolate those mechanisms individually.
