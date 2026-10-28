"""Run the full analysis: python run_all.py  (add --refresh to re-download the source files)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from asiaint import analysis as an  # noqa: E402
from asiaint import data, dcc  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
COLORS = {"Nifty-KOSPI": "#1f4e79", "Nifty-HSI": "#c55a11", "KOSPI-HSI": "#548235"}
LABELS = {"Nifty-KOSPI": "India and Korea", "Nifty-HSI": "India and Hong Kong", "KOSPI-HSI": "Korea and Hong Kong"}
SHADE = [("2008-09-01", "2009-06-30", "GFC"), ("2020-02-15", "2020-06-30", "Covid"), ("2026-01-01", "2026-10-16", "2026")]
SOURCE = ("Source: Yahoo Finance (Nifty 50; KOSPI before 2009; HSI before 2016), Naver Finance (KOSPI), "
          "HKEX and Eastmoney (HSI); author's calculations.")


def style(ax):
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def save(fig, name, title, subtitle, note=SOURCE):
    fig.text(0.01, 0.985, title, fontsize=11.5, fontweight="bold", va="top", ha="left")
    fig.text(0.01, 0.935, subtitle, fontsize=9, va="top", ha="left", color="#404040")
    fig.text(0.01, 0.01, note, fontsize=6.8, color="#595959", ha="left", va="bottom", wrap=True)
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


def fig_dcc(c: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True)
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.07, right=0.98, hspace=0.18)
    for ax, col in zip(axes, c.columns):
        for s, e, lab in SHADE:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#ececec", zorder=0)
            if col == c.columns[0]:
                ax.text(pd.Timestamp(s), 0.83, lab, fontsize=7.5, color="#7f7f7f")
        ax.plot(c.index, c[col], color=COLORS[col], lw=0.8)
        ax.axhline(c[col].mean(), color=COLORS[col], lw=0.8, ls="--")
        ax.text(0.005, 0.06, f"{LABELS[col]} (dashed line: average {c[col].mean():.2f})", transform=ax.transAxes,
                fontsize=8.5, color=COLORS[col], fontweight="bold")
        ax.set_ylim(-0.1, 0.92)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        style(ax)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "exhibit1_dcc_correlations.png", "Exhibit 1: Daily conditional correlations, Sep 2007 to Oct 2026",
         "DCC-GARCH(1,1) with GJR-GARCH(1,1) Student-t margins; local-currency daily log returns on common trading days")


def fig_stress(st: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.07, right=0.98)
    x = np.arange(len(st))
    ax.bar(x - 0.18, st["normal_months"], 0.36, color="#a6a6a6", label="Other months")
    ax.bar(x + 0.18, st["stress_months"], 0.36, color="#1f4e79", label="Most volatile fifth of months")
    for i, (_, row) in enumerate(st.iterrows()):
        ax.text(i + 0.18, row["stress_months"] + 0.012, f"+{row['difference']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [LABELS[p] for p in st.index], fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    save(fig, "exhibit2_stress_months.png", "Exhibit 2: Correlations are higher when markets are volatile",
         "Average monthly DCC correlation; stress months are the top 20% by average conditional volatility (46 of 228)")


def fig_korea_2026(big: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.0))
    fig.subplots_adjust(top=0.83, bottom=0.2, left=0.07, right=0.98)
    x = np.arange(len(big))
    ax.bar(x - 0.27, big["KOSPI"], 0.27, color="#548235", label="KOSPI")
    ax.bar(x, big["Nifty"], 0.27, color="#1f4e79", label="Nifty 50")
    ax.bar(x + 0.27, big["HSI"], 0.27, color="#c55a11", label="Hang Seng")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, [d.strftime("%d %b") for d in big.index], fontsize=8)
    ax.set_ylabel("Log return, %", fontsize=8.5)
    ax.set_ylim(-23, 19)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    style(ax)
    save(fig, "exhibit3_korea_2026_big_days.png", "Exhibit 3: Korea's biggest days in 2026 mostly stayed in Korea",
         "Return on each common trading day of 2026 when the KOSPI moved by 7% or more (can span a holiday in one market)")


def fig_rolling(roll: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.07, right=0.98)
    for col in roll.columns:
        ax.plot(roll.index, roll[col], color=COLORS[col], lw=0.9, label=LABELS[col])
    ax.set_ylim(-0.1, 0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    style(ax)
    save(fig, "exhibit4_rolling_250d.png", "Exhibit 4: 250-day rolling correlations (robustness check)",
         "Plain sample correlation of daily log returns over the previous 250 common trading days")


def main(refresh: bool = False):
    for d in (FIG, TAB, ROOT / "data" / "processed"):
        d.mkdir(parents=True, exist_ok=True)

    prices, raw, report = data.build_prices(ROOT / "data" / "raw", refresh=refresh)
    prices.to_csv(ROOT / "data" / "processed" / "prices_aligned.csv", float_format="%.4f")
    r = data.daily_log_returns(prices)
    w = data.weekly_log_returns(raw)
    w = w.loc[r.index.min():r.index.max()]

    uni, fit, z = dcc.fit_dcc_garch(r)
    names = list(r.columns)
    c = dcc.pairwise_frame(fit.R, r.index, names)
    sigma = pd.concat([u.sigma for u in uni], axis=1)
    sigma.columns = names
    pair_cols = list(c.columns)

    roll = an.rolling_corr(r, 250)
    periods = an.period_table(c, r)
    monthly = an.monthly_panel(c, sigma)
    stress = an.stress_test(monthly, pair_cols)
    stress10 = an.stress_test(monthly, pair_cols, quantile=0.9)
    trend = an.trend_test(monthly, pair_cols)
    roll_m = roll.resample("ME").mean().dropna()
    trend_roll = an.trend_test(roll_m, pair_cols)
    timing = an.timing_check(r, w)
    nextday = an.next_day_table(r)
    hi_days = an.stress_days_summary(c, sigma)

    y26 = r.loc["2026"]
    big = y26[y26["KOSPI"].abs() >= 7].copy()
    # How many of each market's own sessions does each common-day return cover? (>1 means a holiday elsewhere)
    prev_day = pd.Series(prices.index[:-1], index=prices.index[1:])
    for m, srs in raw.items():
        big[f"{m}_sessions"] = [int(((srs.index > prev_day[d]) & (srs.index <= d)).sum()) for d in big.index]
    betas = {}
    for label, (s, e) in {"2021-2025": ("2021", "2025"), "2026": ("2026", "2026")}.items():
        d = r.loc[s:e]
        betas[label] = {m: float(np.cov(d[m], d["KOSPI"])[0, 1] / d["KOSPI"].var()) for m in ("Nifty", "HSI")}
        betas[label]["KOSPI_ann_vol_pct"] = float(d["KOSPI"].std() * np.sqrt(250))
        betas[label]["Nifty_ann_vol_pct"] = float(d["Nifty"].std() * np.sqrt(250))
        betas[label]["HSI_ann_vol_pct"] = float(d["HSI"].std() * np.sqrt(250))
    weekly_corr = {"2021-2025": w.loc["2021":"2025"].corr().round(3).to_dict(),
                   "2026": w.loc["2026"].corr().round(3).to_dict()}

    for name, df in {"period_averages": periods, "stress_test_top20pct_months": stress,
                     "stress_test_top10pct_months": stress10, "trend_test_dcc": trend,
                     "trend_test_rolling250": trend_roll, "timing_check": timing,
                     "next_day_correlations": nextday, "high_vol_days": hi_days,
                     "kospi_big_days_2026": big, "dcc_correlations_daily": c,
                     "rolling_250d_correlations": roll}.items():
        df.to_csv(TAB / f"{name}.csv", float_format="%.4f")

    garch = {u.name: {**{k: round(v, 4) for k, v in u.params.items()}, "persistence": round(u.persistence, 4)} for u in uni}
    results = {
        "sample": report["aligned"],
        "garch": garch,
        "dcc": {"a": round(fit.a, 4), "b": round(fit.b, 4), "a_plus_b": round(fit.a + fit.b, 4),
                "converged": fit.converged, "loglik": round(fit.loglik, 2)},
        "full_sample_mean_dcc": c.mean().round(3).to_dict(),
        "betas_on_kospi": betas,
        "weekly_corr": weekly_corr,
        "kospi_big_days_2026_mean_abs_move": big[["Nifty", "KOSPI", "HSI"]].abs().mean().round(2).to_dict(),
        "kospi_big_days_2026_single_session_mean_abs_move": big[(big.filter(like="_sessions") == 1).all(axis=1)][
            ["Nifty", "KOSPI", "HSI"]].abs().mean().round(2).to_dict(),
        "kospi_big_days_2026_count": int(len(big)),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "data_quality_report.json").write_text(json.dumps(report, indent=2))

    fig_dcc(c)
    fig_stress(stress)
    fig_korea_2026(big[["Nifty", "KOSPI", "HSI"]])
    fig_rolling(roll)

    pd.set_option("display.width", 200)
    print(json.dumps(results, indent=2))
    print("\nPeriod averages\n", periods.round(3).T)
    print("\nStress test (top 20% months)\n", stress.round(3))
    print("\nStress test (top 10% months)\n", stress10.round(3))
    print("\nTrend in DCC correlation\n", trend.round(3))
    print("\nTrend in rolling 250-day correlation\n", trend_roll.round(3))
    print("\nTiming check\n", timing.round(3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="re-download the source files")
    main(p.parse_args().refresh)
