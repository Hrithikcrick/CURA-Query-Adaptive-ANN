# Evaluation after original rankings are restored

The evaluator is implemented. **The real study is still BLOCKED.** Original
exact/HNSW rankings and the original corpus document map remain unavailable.
Synthetic tests exercise implementation behavior; they are not TREC results,
retrieval reruns, or completed judgment-density seeds.

## Read-only preflight

```powershell
python sigir_analysis/run_trec_judgment_density.py --preflight
```

With the current missing inputs this reports BLOCKED, returns exit code 2, and
writes no experiment outputs. It does not encode, retrieve, rebuild an index,
load a model, or retrain a controller.

## Required restored inputs

Use `sigir_analysis/trec_exact_rankings.csv` with exactly these columns:

```text
query_key,query_id,rank,doc_id,score
```

Use `sigir_analysis/trec_hnsw_rankings.csv` with exactly these columns:

```text
query_key,query_id,efSearch,rank,doc_id,score
```

Query IDs and original passage IDs remain strings. Ranks are contiguous and
1-based, documents cannot repeat within a list, and scores must be finite.
Exact lists must have one uniform depth between 10 and 100; only top10 is
evaluated. HNSW requires top10 for each of the original 43 queries at every ef:
10,16,24,32,48,64,96,128,192,256. Query-key/raw-ID pairs must match the saved
query table exactly. Scores are retained as input evidence; ranking positions
determine evaluation.

Supply `sigir_analysis/trec_ranking_provenance.json` using the
[template](trec_ranking_provenance.template.json). Fill it only with actual
recovered provenance and SHA256 hashes, including the hash of the original
ordered `docids.txt` map. The template's null hashes and false identity flag
cannot pass the gate. The restored map must contain 1,000,629 unique raw IDs,
all original judged passages, and every ranked passage. Do not substitute a
different corpus or synthesize a map from retrieved IDs.

No independently recorded historical corpus-map checksum is available.
Restorer provenance, supplied map hash/count, membership checks and reproduced
metrics establish consistency, not independent cryptographic proof of the
historical corpus. That limitation is retained in the gate's validation record.
A trusted raw-ID export with different provenance needs explicit technical
review rather than bypassing this strict map-based gate.

## FULL reproduction gate

[ranking_input_gate.py](ranking_input_gate.py) checks input hashes against the
preserved analysis protocol, all ranking coverage and duplicates, original
qrels checksum, and positive counts. It independently recomputes 43 exact
scores and all 430 HNSW query/budget metrics. Every nDCG/loss must match the
saved value within 1e-6, ANN Recall within 1e-9, and every violation indicator
must agree exactly. Any mismatch stops the run before density outputs exist.
Saved work and timing are reused; no new work counters or latency measurements
are claimed.

Frozen CURA selections must match the preserved source checksum. Their selected
search measurements must match the archived sweep, and effective work/timing
must reproduce the original probe-plus-search formulas. No choices are refit.

## Run after all gates pass

```powershell
python sigir_analysis/run_trec_judgment_density.py
```

The default output directory is
`sigir_analysis/restored_ranking_evaluation/`. An existing evidence directory
is never overwritten; `--output-dir` can select another new directory inside
`sigir_analysis/`. The current canonical blocked tables and report remain
intact. A complete evidence bundle is staged and renamed only after every
condition has been scored and invariants checked.

The bundle contains:

- Per-query, per-seed and four-condition summary CSVs.
- All ten fixed risk/work points per condition/seed, including Wilson bounds.
- Unchanged CURA evaluation with frozen costs and post-hoc fixed matching.
- Per-query FULL reproduction comparisons, P5/P6 statistics and source hashes.
- All four requested judgment-density figures and a validation/run manifest.

FULL is scored once; stochastic conditions use at least 50 paired deterministic
seeds. The thinning function matches all 6,450 previously prepared records:
the same SHA256-derived PCG64 permutations, nested retained-positive prefixes,
retained grades, preserved nonpositive judgments and unjudged=0 rule. IDCG is
recomputed from all retained query judgments.

ANN Recall, ANN-safe ef, recorded fixed work/timing and frozen CURA costs must
remain invariant. First-safe search scans the tested grid without assuming
monotonic safety. Unavailable minima stay missing, with feasibility counts;
explicitly labelled legacy grid-capped fields provide a separate sensitivity.
Primary aggregate relative reduction is a ratio of mean budgets on the same
joint-feasible cohort, rather than a mean of per-query percentages.

P5 compares each thinned seed's gap with FULL on that identical feasible query
cohort. P6 compares exact-zero fractions. Bootstrap intervals and exploratory
sign-flip tests use seed-level effects conditional on these fixed queries;
query-by-seed cells are not independent observations. FULL's displayed seed
intervals are degenerate, not query-population intervals. Nonsignificance does
not establish equivalence or Outcome B. Censored/undefined effects remain
unavailable, and scoring completion is recorded separately from statistical
completion. Outcome A/B/C and manuscript claims require the actual evidence.

Validate and interpret the real evidence bundle before revising the main
research report, statistical table and manuscript. The existing missing-input
recovery scripts intentionally remain a blocked-fallback workflow; they must
not be rerun over restored ranking results to erase those observations.

## Implementation checks

```powershell
python sigir_analysis/validate_evaluation_pipeline.py
```

This runs isolated synthetic tests for metric definitions, censoring,
nonmonotone first-safe behavior, source/provenance rejection, FULL mismatch,
seed uncertainty, frozen costs, and a complete 50-seed orchestration fixture.
The orchestration fixture uses an explicit mocked gate in a temporary synthetic
workspace; independent tests check the real gate. Temporary fixtures are
removed. [Implementation validation](evaluation_pipeline_validation.json) and
[test log](evaluation_pipeline_test_log.txt) are separate from scientific
[validation](validation.json) and [completion](SIGIR_COMPLETION_STATUS.txt).
