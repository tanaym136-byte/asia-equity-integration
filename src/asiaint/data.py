"""Download, check, splice and align daily closes for Nifty 50, KOSPI and the Hang Seng Index.

Sources (all public GitHub mirrors, downloaded as-is and cached under data/raw):
  - Nifty 50: Yahoo Finance ^NSEI, mirrored in nathamlan2020-create/nsedata
  - KOSPI: Naver Finance closes, mirrored in kuthy9/korea-market-data
  - Hang Seng Index: HKEX daily quotations / Eastmoney closes, published in sycheng0924/hsi-data
  - Backfill for KOSPI (before 2009) and HSI (before 2016): Yahoo Finance closes in the
    Kaggle "Stock Exchange Data" file mirrored in kishandutta/stockmarketlrmodel

Each backfill is used only for dates before the primary source starts, and only after the two
sources have been compared on their overlapping dates (see `overlap_check`).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_URLS = {
    "nifty": "https://raw.githubusercontent.com/nathamlan2020-create/nsedata/main/daily/indices/IDX_NSEI.csv",
    "kospi": "https://raw.githubusercontent.com/kuthy9/korea-market-data/main/kospi.csv",
    "hsi": "https://raw.githubusercontent.com/sycheng0924/hsi-data/main/docs/hsi.json",
    "backfill": "https://raw.githubusercontent.com/kishandutta/stockmarketlrmodel/main/model/Market.csv",
}
FILENAMES = {"nifty": "IDX_NSEI.csv", "kospi": "kospi.csv", "hsi": "hsi.json", "backfill": "Market.csv"}
NAMES = ["Nifty", "KOSPI", "HSI"]


def download(raw_dir: Path, refresh: bool = False) -> dict[str, Path]:
    """Fetch each source file into raw_dir unless it is already cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, url in RAW_URLS.items():
        path = raw_dir / FILENAMES[key]
        if refresh or not path.exists():
            urllib.request.urlretrieve(url, path)
        paths[key] = path
    return paths


def _clean(s: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = name
    return s


def load_nifty(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return _clean(df["close"], "Nifty")


def load_kospi(path: Path) -> pd.Series:
    df = pd.read_csv(path, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return _clean(df.set_index("date")["close"], "KOSPI")


def load_hsi(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Returns the close series and the full table (with prevClose for a consistency check)."""
    d = json.loads(Path(path).read_text())
    df = pd.DataFrame(d["rows"], columns=d["columns"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return _clean(df["close"], "HSI"), df


def load_backfill(path: Path, symbol: str, name: str) -> pd.Series:
    df = pd.read_csv(path, usecols=["Index", "Date", "Close"])
    df = df[df["Index"] == symbol].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return _clean(df.set_index("Date")["Close"], name)


@dataclass
class OverlapStats:
    name: str
    start: str
    end: str
    n_common: int
    median_abs_diff_pct: float
    max_abs_diff_pct: float
    max_diff_date: str
    n_diff_over_half_pct: int
    only_in_primary: int
    only_in_backfill: int


def overlap_check(primary: pd.Series, backfill: pd.Series, name: str) -> OverlapStats:
    """Compare two sources for the same index on the dates where both have a close."""
    j = pd.concat([primary, backfill], axis=1, join="inner").dropna()
    j.columns = ["primary", "backfill"]
    diff = (j["primary"] / j["backfill"] - 1.0).abs() * 100
    lo, hi = j.index.min(), j.index.max()
    p_win, b_win = primary[lo:hi].index, backfill[lo:hi].index
    return OverlapStats(
        name=name,
        start=str(lo.date()),
        end=str(hi.date()),
        n_common=len(j),
        median_abs_diff_pct=float(diff.median()),
        max_abs_diff_pct=float(diff.max()),
        max_diff_date=str(diff.idxmax().date()),
        n_diff_over_half_pct=int((diff > 0.5).sum()),
        only_in_primary=len(p_win.difference(b_win)),
        only_in_backfill=len(b_win.difference(p_win)),
    )


def splice(primary: pd.Series, backfill: pd.Series) -> pd.Series:
    """Use the backfill only for dates before the primary source begins."""
    early = backfill[backfill.index < primary.index.min()]
    out = pd.concat([early, primary]).sort_index()
    out.name = primary.name
    return out


def series_checks(s: pd.Series, big_move: float = 0.07) -> dict:
    """Basic checks on one price series: gaps, duplicates, bad values and very large daily moves."""
    r = np.log(s).diff().dropna()
    gaps = s.index.to_series().diff().dt.days.dropna()
    big = r[r.abs() > big_move]
    return {
        "series": s.name,
        "start": str(s.index.min().date()),
        "end": str(s.index.max().date()),
        "n_obs": int(len(s)),
        "duplicates": int(s.index.duplicated().sum()),
        "non_positive": int((s <= 0).sum()),
        "longest_gap_days": int(gaps.max()),
        "moves_over_7pct": [(str(d.date()), round(float(v) * 100, 1)) for d, v in big.items()],
    }


def hsi_prevclose_check(hsi_table: pd.DataFrame, tol_pct: float = 0.01) -> int:
    """Number of days where the published previous close disagrees with the prior row's close."""
    mismatch = (hsi_table["prevClose"] / hsi_table["close"].shift(1) - 1).abs() * 100
    return int((mismatch > tol_pct).sum())


def build_prices(raw_dir: Path, refresh: bool = False) -> tuple[pd.DataFrame, dict[str, pd.Series], dict]:
    """Return aligned closes (dates on which all three markets traded), each market's own full
    series (for weekly returns) and a data-quality report."""
    p = download(raw_dir, refresh=refresh)
    nifty = load_nifty(p["nifty"])
    kospi = load_kospi(p["kospi"])
    hsi, hsi_table = load_hsi(p["hsi"])
    kospi_bf = load_backfill(p["backfill"], "KS11", "KOSPI")
    hsi_bf = load_backfill(p["backfill"], "HSI", "HSI")
    nifty_bf = load_backfill(p["backfill"], "NSEI", "Nifty")

    report = {
        "overlap": [
            overlap_check(kospi, kospi_bf, "KOSPI: Naver vs Yahoo").__dict__,
            overlap_check(hsi, hsi_bf, "HSI: HKEX/Eastmoney vs Yahoo").__dict__,
            overlap_check(nifty, nifty_bf, "Nifty: two Yahoo downloads").__dict__,
        ],
        "hsi_prevclose_mismatches": hsi_prevclose_check(hsi_table),
    }
    kospi_full = splice(kospi, kospi_bf)
    hsi_full = splice(hsi, hsi_bf)
    report["series"] = [series_checks(s) for s in (nifty, kospi_full, hsi_full)]

    prices = pd.concat([nifty, kospi_full, hsi_full], axis=1, join="inner").dropna()
    prices.columns = NAMES
    report["aligned"] = {
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "n_common_days": int(len(prices)),
        "dropped_days": {s.name: int(len(s[prices.index.min():prices.index.max()]) - len(prices))
                         for s in (nifty, kospi_full, hsi_full)},
    }
    raw = {"Nifty": nifty, "KOSPI": kospi_full, "HSI": hsi_full}
    return prices, raw, report


def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns in percent, computed on common trading days so every return covers the same interval."""
    return (100 * np.log(prices).diff()).dropna()


def weekly_log_returns(raw_closes: dict[str, pd.Series], anchor: str = "W-FRI") -> pd.DataFrame:
    """Week-on-week log returns from each market's own last close of the week, then aligned."""
    wk = {k: s.resample(anchor).last() for k, s in raw_closes.items()}
    df = pd.concat(wk, axis=1).dropna()
    return (100 * np.log(df).diff()).dropna()
