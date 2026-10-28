"""Correlation analysis built on the DCC output."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

PERIODS = {
    "Global financial crisis (Sep 2008 - Jun 2009)": ("2008-09-01", "2009-06-30"),
    "2010-2019": ("2010-01-01", "2019-12-31"),
    "Covid shock (Feb - Jun 2020)": ("2020-02-15", "2020-06-30"),
    "2021-2025": ("2021-01-01", "2025-12-31"),
    "2026 to date (Korea chip-stock swings)": ("2026-01-01", "2026-12-31"),
}


def pairs(names):
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]


def rolling_corr(returns: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    out = {f"{a}-{b}": returns[a].rolling(window).corr(returns[b]) for a, b in pairs(list(returns.columns))}
    return pd.DataFrame(out).dropna()


def period_table(dcc_corr: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Average DCC correlation and plain sample correlation of returns in each period."""
    rows = []
    for label, (s, e) in PERIODS.items():
        c = dcc_corr.loc[s:e]
        r = returns.loc[s:e]
        if len(c) == 0:
            continue
        row = {"period": label, "days": len(c)}
        for a, b in pairs(list(returns.columns)):
            row[f"{a}-{b} DCC"] = c[f"{a}-{b}"].mean()
            row[f"{a}-{b} sample"] = r[a].corr(r[b])
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")


def monthly_panel(dcc_corr: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    """Monthly averages of each pair's DCC correlation and of average conditional volatility."""
    m = dcc_corr.resample("ME").mean()
    m["avg_vol"] = sigma.mean(axis=1).resample("ME").mean()
    m["n_days"] = dcc_corr.iloc[:, 0].resample("ME").count()
    return m[m["n_days"] >= 10].drop(columns="n_days")


def stress_test(monthly: pd.DataFrame, pair_cols: list[str], quantile: float = 0.8, hac_lags: int = 6) -> pd.DataFrame:
    """Do correlations rise in high-volatility months?

    Regress each pair's monthly average DCC correlation on a dummy for months in the top 20% of
    average conditional volatility, with Newey-West (HAC) standard errors because the correlation
    series is persistent.
    """
    stress = (monthly["avg_vol"] >= monthly["avg_vol"].quantile(quantile)).astype(float)
    X = sm.add_constant(stress.rename("stress"))
    rows = []
    for col in pair_cols:
        res = sm.OLS(monthly[col], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        rows.append({
            "pair": col,
            "normal_months": res.params["const"],
            "stress_months": res.params["const"] + res.params["stress"],
            "difference": res.params["stress"],
            "t_stat_HAC": res.tvalues["stress"],
            "p_value": res.pvalues["stress"],
            "n_stress_months": int(stress.sum()),
            "n_months": int(len(stress)),
        })
    return pd.DataFrame(rows).set_index("pair")


def trend_test(monthly: pd.DataFrame, pair_cols: list[str], hac_lags: int = 12) -> pd.DataFrame:
    """Linear time trend in monthly average DCC correlation (per decade), HAC standard errors."""
    years = (monthly.index - monthly.index[0]).days / 365.25
    X = sm.add_constant(pd.Series(years / 10.0, index=monthly.index, name="decades"))
    rows = []
    for col in pair_cols:
        res = sm.OLS(monthly[col], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        rows.append({"pair": col, "change_per_decade": res.params["decades"],
                     "t_stat_HAC": res.tvalues["decades"], "p_value": res.pvalues["decades"]})
    return pd.DataFrame(rows).set_index("pair")


def timing_check(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Markets close at different times (Seoul 06:30 UTC, Hong Kong 08:00, Mumbai 10:00), so a
    same-day daily correlation can understate the link. Compare same-day daily correlation,
    next-day correlation in both directions, and weekly correlation."""
    rows = []
    for a, b in pairs(list(daily.columns)):
        rows.append({
            "pair": f"{a}-{b}",
            "daily_same_day": daily[a].corr(daily[b]),
            f"{a}(t) with {b}(t+1)": daily[a].corr(daily[b].shift(-1)),
            f"{b}(t) with {a}(t+1)": daily[b].corr(daily[a].shift(-1)),
            "weekly": weekly[a].corr(weekly[b]),
        })
    return pd.DataFrame(rows).set_index("pair")


def next_day_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each market today with each other market tomorrow (rows: today, cols: tomorrow)."""
    names = list(daily.columns)
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            m.loc[a, b] = daily[a].corr(daily[b].shift(-1))
    return m


def stress_days_summary(dcc_corr: pd.DataFrame, sigma: pd.DataFrame, top: float = 0.9) -> pd.DataFrame:
    """Average DCC correlation on the 10% of days with the highest average conditional volatility."""
    avg = sigma.mean(axis=1)
    hi = avg >= avg.quantile(top)
    return pd.DataFrame({"high_vol_days": dcc_corr[hi].mean(), "other_days": dcc_corr[~hi].mean()})
