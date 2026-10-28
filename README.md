# How closely do India, Korea and Hong Kong equities move together?

Daily DCC-GARCH correlations between the Nifty 50, KOSPI and Hang Seng Index, September 2007 to October 2026 (4,305 common trading days).

The project asks three questions about Asian financial integration:

1. Do these markets move together more when markets are volatile, which is exactly when diversification matters most?
2. Have the links grown or weakened over two decades?
3. Did the extreme swings in Korean chip stocks in 2026 spread to India and Hong Kong?

## Main results

| | India and Korea | India and Hong Kong | Korea and Hong Kong |
|---|---|---|---|
| Average DCC correlation, 2007-2026 | 0.41 | 0.45 | 0.58 |
| Global financial crisis (Sep 2008 - Jun 2009) | 0.46 | 0.60 | 0.65 |
| Covid shock (Feb - Jun 2020) | 0.53 | 0.58 | 0.72 |
| 2021-2025 | 0.39 | 0.33 | 0.51 |
| 2026 to date | 0.39 | 0.43 | 0.43 |
| Rise in the most volatile fifth of months | +0.08 (t = 4.6) | +0.11 (t = 4.9) | +0.07 (t = 2.7) |
| Linear trend per decade | -0.03 (t = -1.6) | -0.13 (t = -5.2) | -0.09 (t = -4.0) |

t-statistics use Newey-West (HAC) standard errors on monthly averages.

- **Correlations rise in stress.** In the 46 most volatile months out of 228, average correlations were 0.07 to 0.11 higher than in other months. Using the top 10% of months instead, the two India pairs rise by 0.09 and 0.14 (both significant); the Korea and Hong Kong difference is smaller and no longer significant.
- **Links with Hong Kong have weakened.** The India and Hong Kong correlation averaged 0.59 in 2008 and 0.24 in 2024. The decline is significant in both the DCC model and plain 250-day rolling correlations (-0.18 per decade). The India and Korea link shows no clear trend.
- **Korea's 2026 turbulence mostly stayed in Korea.** KOSPI volatility reached an annualised 67% in 2026 (20% in 2021-2025), including a record one-day rise of 17.9% on 31 July. The sensitivity (beta) of the Nifty to the KOSPI fell from 0.31 in 2021-2025 to 0.10 in 2026, and of the Hang Seng from 0.63 to 0.10. On the nine days when the KOSPI moved 7% or more and every market had traded a normal single session, the KOSPI moved 9.9% on average, the Nifty 0.65% and the Hang Seng 0.72%.
- **Time zones matter.** Seoul closes first (06:30 UTC), then Hong Kong (08:00) and Mumbai (10:00). The India and Korea correlation is 0.43 on same-day daily returns but 0.54 on weekly returns, and today's Nifty return is correlated 0.09 with tomorrow's KOSPI return (0.03 the other way). Daily correlations across Asian time zones therefore understate the link.

Charts are in `outputs/figures`, tables in `outputs/tables`, headline numbers in `outputs/results.json`.

## Data

| Index | Main source | Backfill | Check on overlapping dates |
|---|---|---|---|
| Nifty 50 | Yahoo Finance ^NSEI (mirror: nathamlan2020-create/nsedata) | none needed | matches a second Yahoo download to 0.0001% on 3,345 dates |
| KOSPI | Naver Finance (mirror: kuthy9/korea-market-data), from Jan 2009 | Yahoo ^KS11 before 2009 | 3,062 common dates; median difference 0.0000%; one date above 0.5% (6 May 2021, 0.58%); 7 dates missing from Yahoo |
| Hang Seng | HKEX daily quotations and Eastmoney (sycheng0924/hsi-data), from Jan 2016 | Yahoo ^HSI before 2016 | 1,331 common dates, no difference above 0.0001% |

Further checks (`outputs/data_quality_report.json`):
- the published Hang Seng previous close matches the prior day's close on every date;
- no duplicate dates or non-positive prices;
- every daily move above 7% is listed; the largest (the KOSPI's +17.9% on 31 July 2026) was verified against press reports of the record close of 6,595.45.

Returns are local-currency daily log returns, computed on dates when all three markets traded, so each return covers the same calendar interval. Where a holiday closes one market, the return on the next common day spans more than one session in the others. The table of 2026 big days records how many sessions each return covers.

Index levels are price indices, so dividends are excluded. The source files are public mirrors of vendor data. Before relying on the numbers, re-run the analysis on Bloomberg or exchange series.

## Method

1. **Margins.** Each return series gets a GJR-GARCH(1,1) model with Student-t errors (`arch` package). This removes each market's own volatility clustering and the larger response to falls than to rises.
2. **Correlation.** The standardised residuals feed a DCC(1,1) model (Engle, 2002) with correlation targeting, estimated by quasi-maximum likelihood in two steps. Estimates: a = 0.021, b = 0.969. Correlation shocks therefore fade slowly, with a half-life of about 67 trading days.
3. **Tests.** Monthly averages of the DCC correlations are regressed on (i) a dummy for the most volatile 20% of months and (ii) a linear time trend. Both use Newey-West standard errors, because the series are persistent.
4. **Robustness.** The tests are repeated with a 10% stress cut-off, on plain 250-day rolling correlations, and on weekly returns built from each market's own Friday close.

The DCC estimator is written from scratch in `src/asiaint/dcc.py` and tested on simulated data with known parameters.

## Run it

```
pip install -r requirements.txt
python run_all.py            # downloads the source files on first run (about 9 MB)
python run_all.py --refresh  # re-download to extend the sample
python -m pytest             # 11 tests: DCC recursion, likelihood, parameter recovery, splicing, alignment
```

## Layout

```
src/asiaint/data.py      download, cross-source checks, splicing, alignment, returns
src/asiaint/dcc.py       GJR-GARCH margins, DCC recursion, likelihood, estimation, simulation
src/asiaint/analysis.py  rolling correlations, period averages, stress and trend tests, time-zone checks
run_all.py               runs everything, writes outputs/
tests/                   unit tests
```

## Limitations

- Correlation is not causation: a shared rise can come from a common global shock rather than spillover from one market to another.
- The stress dummy is defined from the same volatility estimates that standardise the returns. DCC addresses the mechanical upward bias in raw correlations during volatile periods (Forbes and Rigobon, 2002), but only as well as the GARCH margins fit.
- The two-step estimator's standard errors for a and b are not reported, because the first step is ignored in the usual Hessian; a bootstrap would be the next step.
- Index composition differs a lot: the 2026 KOSPI moves were dominated by two chip stocks, so a sector-level comparison would sharpen the 2026 result.
