# Frozen test summary

These files are aggregate research outputs from the completed 2022–2025 frozen test. They were copied from the local `outputs/frozen_test_v1/` run without refitting or changing predictions.

| File | Contents |
|---|---|
| `summary.csv` | Six portfolios under the primary 10 bps / `lag_2` scenario |
| `annual_metrics.csv` | Annual portfolio statistics; filter model, scenario and cost before comparing |
| `annual_rank_ic.csv` | Annual predictive rank correlations |
| `paired_confidence_intervals.csv` | Paired block-bootstrap differences and sensitivity to block size |
| `test_overview.png` | Test performance overview |

Returns and rates in CSV files are decimals, so `0.10` means 10%. `cost_bps` is in basis points. The predeclared primary model is `cross_sectional_rank__lightgbm_10`; Ridge results are comparisons, not a retrospective change to that selection.

The comparison includes explicit settlement accounting, a proxy-sale policy for three bank delistings, and exclusion of three nontradable CVRs from investable NAV. These aggregate results do not contain security-level observations, predictions or holdings. Raw data and fitted models are excluded from the repository.
