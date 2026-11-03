# Overleaf instructions — CURA research paper

This directory contains a revised two-column LaTeX draft derived from the earlier `main_corrected.tex` manuscript, retaining diagrams, tables, references, and the underlying paper structure.

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

**Draft, not final:** the final version still needs 49 remaining human reviews, bibliography verification, full clean-environment reproducibility testing, and a final pass for the professor's page/format requirements.

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
