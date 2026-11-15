# Statistical analysis scope

P1 (all 1,000 MS MARCO oracle pairs), P2 (exact-positive oracle pairs), P3 (CURA versus original calibrated ef96 on 200 test queries), and P4 (CURA versus post-hoc matched ef48) are complete from saved paired data. Both grid-capped and jointly feasible oracle summaries are retained. Source hashes, bootstrap seeds, effect directions, units, effect sizes and CIs are stored in primary_statistical_results.csv and statistical_analysis_status.json.

P5 and P6 (FULL versus each thinned TREC condition: oracle gap and exact-zero fraction) are blocked, with blank effect/CI/p-value fields. Original qrels and prepared seeds cannot replace missing rankings. No outcome A/B/C or seed-level significance claim can be made.

All p-values are exploratory and unadjusted. Paired sign-flip tests assume sign exchangeability; they do not reflect randomized assignment of algorithms. Bootstrap CIs preserve query pairing and condition on the frozen policy; post-hoc comparator intervals also condition on ef48's test-set selection. Exact McNemar comparisons have very few discordant failures and cannot demonstrate equivalent risk. Oracle ef256 fallbacks are grid-capped descriptive values, not verified safe thresholds; the joint-feasibility sensitivity targets a smaller, selected population. Seed uncertainty and query uncertainty are distinct and have not been conflated.

The pre-extension statistical table is preserved byte-for-byte as statistical_tests_saved.csv. statistical_tests.csv combines those rows with the explicitly identified primary comparisons and blocked tests. The analysis as a whole is INCOMPLETE until the density comparisons can be performed.
