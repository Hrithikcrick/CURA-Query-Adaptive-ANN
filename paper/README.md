# Overleaf instructions — CURA research paper

**Active research draft:** [sigir_short.tex](sigir_short.tex), **When Do Utility-Aware ANN Search Budgets Really Save Work? The Role of Relevance-Judgment Completeness and Risk Calibration.** See [the revised direction](SIGIR_REVISED_DIRECTION.md) and [research report](../reports/SIGIR_RESEARCH_ANALYSIS.md). Fixed ef=48 uses 28.29% less work than CURA at the same observed MS MARCO test risk; this comparison is post hoc. Original TREC passage qrels are recovered and 50 deterministic seeds per stochastic judgment condition are prepared. Same-ranking evaluation is blocked by missing exact/HNSW document rankings and original recovery caches; zero ranking seeds were evaluated. No completed density result or general adaptive advantage is claimed. The source has not been compiled or checked against the official ACM template.

The existing `main.tex` remains the historical course report, retaining diagrams, tables, references and review evidence, with prominent notices of the new fixed-frontier finding. Instructions below describe that older course artifact. For the new standalone research draft, upload `sigir_short.tex` as the main source instead; it has no external figure dependencies.

<!-- BEGIN CURA FINAL EVIDENCE STATUS -->
The manuscript now includes the final actual budget-sweep taxonomy: all 50 labels are model assisted, including 8 automatic second-review resolutions. No new human verification is claimed. This is separate from FiQA's 1/50 recorded human reviews. The mean/p95 tradeoff and timing scope are disclosed. Bibliography verification, final rendering/format review, and clean-environment reproduction remain pending. See [the final evidence README](../FINAL_README.md).
<!-- END CURA FINAL EVIDENCE STATUS -->

## Open in Overleaf

1. Open [Overleaf](https://www.overleaf.com) and choose **New Project → Blank Project**. Name it `CURA COL764 Final Paper`.
2. Download [main.tex](main.tex) from this GitHub folder using **Raw**, or copy the complete source into Overleaf's `main.tex`. This manuscript uses a **single LaTeX source file**, with an inline `thebibliography`, so no separate `.bib` or figures are required.
3. In **Menu**, select **pdfLaTeX** as the compiler. Click **Recompile**. If references show `??`, recompile once more.
4. Review the rendered PDF for figure placement, float overflow, bibliography, and the professor's required page/format limits before exporting.

## What was updated

- Added a FiQA independent-document-domain section with 131-query held-out results for a **separately fitted reduced-feature controller**.
- Included the observed negative result (**75.27% more realized search work**) alongside the positive MS MARCO finding (**20.51% less work**).
- Replaced claims of completed human taxonomy with the **50-case fixed-HNSW review preparation**, **1/50** manually verified at the latest November 3 audit. This statistic is a snapshot and must be updated when reviews finish.
- Clarified that the 5% number is a nominal risk target, not a universal risk guarantee.
- Updated the AI-use statement and reproducibility limitations.

## Paper status

**Saved-evidence manuscript updated:** the actual 50-failure analysis is complete as model-assisted annotation, including eight automatic second-review resolutions. The separate FiQA worksheet has 1/50 recorded human reviews (49 remaining); those are not pending decisions for the completed automatic analysis. Bibliography verification, full clean-environment reproduction, and final rendered-format review remain unverified.

The current style is two-column **article** with SIGIR-like formatting, **not** an official ACM SIGIR `acmart` proceedings template. If an official template is required, transfer the content to that template before submission.

## Sync your VS Code checkout

In PowerShell at your project directory:

```powershell
git pull origin main
code paper/main.tex
```

To add revisions you make locally:

```powershell
git add paper/main.tex
git commit -m "Revise CURA final LaTeX paper"
git push origin main
```

Do not overwrite the human-review worksheet by rerunning `15_fiqa_50_failure_review.py`.
