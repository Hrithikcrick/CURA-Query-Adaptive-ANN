# Verified false-safety audit and FiQA oracle replication

Verified on **26 November 2026**, against the existing saved research at repository commit `c41c089554a2596667c2626f7fa35118a586f10c`. This focused publication preserves the original implementation, datasets, rankings, model configurations, submitted manuscript and historical results.

The original local extension was found under `full_paper_extension/`. The two named distribution ZIPs and an original push-kit summary were not found. Existing metric code, tests, frozen protocols and small portable inputs are included unchanged. The focused verification utilities, summary, figures, report and manuscript addendum are new retrospective verification artifacts, not a recovered original ZIP or a new retrieval experiment.

## Findings

TREC DL 2019 has 43 queries, ten tested budgets, 430 FULL query-budget pairs and 14 FULL-unsafe pairs at nDCG-loss tolerance 0.025. Across 50 paired thinning seeds, **27.43%, 50.14% and 86.86%** of FULL-unsafe pair instances become labeled safe at 50%, 25% and one positive per query. The denominator is `14 x 50 = 700`, not all `430 x 50` pair instances. At 50% retention, **183/192 = 95.3125%** of false-safe instances still have positive exact-reference nDCG under thinning. FULL denotes available pooled judgments, not exhaustive relevance truth.

FiQA independently reproduces a different finding: ANN-safe and IR-safe oracle budgets differ across domains. All 648 saved oracle rows match independently recomputed minima. Of 305 exact-positive queries, 303 are jointly feasible; their mean ANN-minus-IR gap is **8.561 ef**, with **120 lower / 166 equal / 17 higher** IR-safe budgets. This is neither a FiQA judgment-density experiment nor evidence of controller transfer.

See [the verified summary](VERIFIED_SAVED_DATA_RESULTS.json), [full findings and limitations](RESEARCH_EXTENSION_REPORT.md), and [the separate manuscript addendum](MANUSCRIPT_ADDENDUM.tex). The commit changes only this extension directory.

## Reproduce without retrieval

Use the existing Python environment, or Python 3.12 with NumPy, pandas and SciPy. Matplotlib is required only for figures. Measured versions are recorded in the summary; these are observations, not a portable dependency lock. Run from the repository root:

```powershell
python -B -m unittest discover -s full_paper_extension -p test_false_safety.py -v
python -B -m unittest discover -s full_paper_extension/tests -v
python -B full_paper_extension/false_safety_audit.py
python -B full_paper_extension/fiqa_oracle_robustness.py
python -B full_paper_extension/verify_saved_results.py
python -B full_paper_extension/plot_verified_results.py --output-dir full_paper_extension/cache/figure_check
```

The first two commands run 11 original metric tests and 15 new verification tests. The verifiers print JSON by default, compare every saved TREC score and every FiQA minimum, and leave historical files unchanged. `verify_saved_results.py` compares fresh results with the published JSON. Explicit `--output` on this wrapper creates a new summary and refuses overwrite. The figure command requires a new output directory, reads the verified summary and FiQA observations, and records plotted values and output hashes.

The original broader analysis is retained as `analyze_false_safety.py`. Its `--collections 2019 --seeds 50 --bootstrap-resamples 20000` mode can generate broader 2019 analyses from saved rankings and portable inputs, but **writes analysis outputs**. Use a disposable checkout for that command. The focused verifiers above are the publication checks.

Small qrels and frozen Score-Gap policy inputs under `data/` are deliberately versioned. No embeddings, FAISS indexes, environments, trained models, build dependencies or raw million-passage corpus are included. An extension-local `.gitattributes` preserves published input bytes for checksum verification.

Historical FiQA CSVs use the parent repository's existing checkout rules. The summary records their measured raw hashes; verification additionally enforces canonical LF text hashes so a legitimate LF/CRLF checkout difference does not become a scientific mismatch. The input files are never normalized or rewritten by the verifier.

The complete publication file list and hashes are in [PUSH_FILE_MANIFEST.json](PUSH_FILE_MANIFEST.json). Verification and preservation checks are in [PUSH_VERIFICATION.json](PUSH_VERIFICATION.json).

## Scope of the publication

The frozen protocol/audit documents describe the broader research and retain their original dated context. Completed local TREC 2020, corpus-control and official-source DARTH files are preserved locally but excluded from this focused push. They are **outside this publication scope**, not newly declared pending experiments. An isolated check on 26 November additionally reproduced 16 existing combined 2019/2020 statistical tables; it does not turn this focused report into an end-to-end retrieval reproduction. CURA-OnePass remains an unmeasured design.

The FiQA bootstrap configuration is explicitly retrospective: 10,000 paired-query resamples, PCG64 seed 42, sorted string query IDs. It reproduces the supplied rounded interval `[5.498, 11.624]`; the original CI-generating script was not recovered. A separately declared 50,000-resample sensitivity, seed 20261126, gives `[5.531, 11.584]`. Neither interval accounts for new retrieval, corpus selection or relevance-pool uncertainty.
