# Original-ranking recovery and completed evaluation

**RECOVERED / FULL PASS / 50-seed COMPLETE.** Earlier missing-input audits are historical. Do not regenerate results that already exist and validate.

The external `~/Downloads/IR_PROJECT.zip` archive (3,780,386,484 bytes, SHA-256 `0d5d0e1e712bd555bd9171d6f4a84884eb9a62b2d230706caae5b180265ec21a`) contains original `IR_PROJECT/PILOT/` caches. [Provenance](trec_ranking_provenance.json) records member hashes and seven byte-identical source/config/query/run checks.

Exact top100 indices/scores cover 1,043 query rows. HNSW cached indices have axes 10 ef values × 1,043 queries × 10 ranks. Phase08 defines sorted ef values and unchanged query-table order. The ordered map has 1,000,629 unique IDs; TREC queries are selected by identity. Original vector inner products supply ANN score evidence without changing cached order. No model, index construction, search or training was invoked; the archived HNSW index was not loaded. No independent historical map checksum or immutable encoder revision exists: lineage and FULL reproduction support consistency rather than independent identity proof.

## Gate and completed evidence

[ranking_input_gate.py](ranking_input_gate.py) verifies query coverage, all ten ef values, contiguous ranks, unique IDs, map membership, finite scores and corpus count/order checksum. [FULL comparisons](trec_retrieval_reproduction.csv) contain 43 exact and 430 HNSW comparisons, all passing nDCG/loss tolerance 1e-6, recall tolerance 1e-9 and identical violation indicators at loss > .025. [The gate](recovered_ranking_validation.json) passed before density scoring.

The [completed bundle](restored_ranking_evaluation/validation.json) scores FULL once and all three thinned conditions over 50 prepared seeds, preserving all 151 qrel hashes. It contains 64,930 query-budget observations. ANN recall, ANN-safe budgets, fixed work/timing and frozen CURA costs are invariant; all 43 queries are jointly feasible throughout. [Promotion validation](density_promotion_validation.json) records canonical-output hashes.

## Current-state verification

```powershell
python sigir_analysis/validate_analysis.py
python sigir_analysis/completion_status.py
```

These commands audit existing evidence only. They do not rerun completed analyses or resampling.

## Fresh reproduction only

Use a separate fresh checkout and the checksum-matching original archive to recreate evidence. Restore original qrels with recover_trec_qrels.py only if missing. Prepared thinned qrels are supplied; preserve their hashes.

```powershell
python sigir_analysis/recover_trec_cached_rankings.py --archive <path-to-IR_PROJECT.zip>
python sigir_analysis/run_trec_judgment_density.py --seeds 50
```

Recovery refuses to overwrite normalized rankings; the runner refuses to overwrite its bundle. Do not remove validated outputs to run these commands. Published ranked IDs enable density scoring without a new million-document search; strict map-membership revalidation requires the original map. Large archive/vector/index/map files are not published.

[finalize_recovered_density.py](finalize_recovered_density.py) documents already-executed promotion/statistics. Its statistics command refuses a second run after P7 exists. It preserves 42 historical rows, reuses six P5/P6 results, derives P7 algebraically and adds three P8 tests. [Metadata reconciliation](finalize_completion_metadata.py) checks historical cells against Git commit 7968d6c without resampling.

The generic [delta audit](audit_recovery_delta.py) and [external metadata inventory](recovery_delta_external_inventory.json) document recovery history, not scientific observations. The saved inventory lists external scopes/inaccessible entries; it makes no whole-disk completeness claim. Dedicated NPY/NPZ provenance verification accepted the original rankings. Nineteen unrelated Temp parquet streams failed inspection through the legacy non-seekable parser; this does not affect the original archive verification.
