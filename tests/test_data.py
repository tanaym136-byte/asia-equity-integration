import json

import numpy as np
import pandas as pd
import pytest

from asiaint import data


def s(values, start="2020-01-01", name="X", freq="B"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), name=name, dtype=float)


def test_splice_uses_backfill_only_before_primary_starts():
    backfill = s(np.arange(1, 11), "2020-01-01")
    primary = s(np.arange(100, 106), "2020-01-08")
    out = data.splice(primary, backfill)
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert (out.loc[primary.index] == primary).all()
    assert (out.loc[: "2020-01-07"] == backfill.loc[: "2020-01-07"]).all()


def test_overlap_check_flags_a_bad_print():
    a = s(np.linspace(100, 110, 50))
    b = a.copy()
    b.iloc[20] *= 1.01
    st = data.overlap_check(a, b, "test")
    assert st.n_diff_over_half_pct == 1
    assert st.max_diff_date == str(a.index[20].date())


def test_returns_on_common_days_span_the_same_interval():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    a = pd.Series([100, 101, 102, 103], index=idx, name="A")
    b = pd.Series([50, 55, 60], index=idx[[0, 1, 3]], name="B")  # B shut on 3 Jan
    prices = pd.concat([a, b], axis=1, join="inner")
    r = data.daily_log_returns(prices)
    assert list(r.index) == [idx[1], idx[3]]
    assert r.loc[idx[3], "A"] == pytest.approx(100 * np.log(103 / 101))


def test_hsi_prevclose_check(tmp_path):
    rows = [["2024-01-02", 100.0, 99.0, 1, 0], ["2024-01-03", 101.0, 100.0, 1, 0], ["2024-01-04", 102.0, 100.5, 1, 0]]
    p = tmp_path / "hsi.json"
    p.write_text(json.dumps({"columns": ["date", "close", "prevClose", "turnover", "halfDay"], "rows": rows}))
    close, table = data.load_hsi(p)
    assert len(close) == 3
    assert data.hsi_prevclose_check(table) == 1


def test_series_checks_reports_big_moves_and_duplicates():
    x = s([100, 101, 120, 121, 121], name="T")
    out = data.series_checks(x)
    assert out["duplicates"] == 0 and out["non_positive"] == 0
    assert len(out["moves_over_7pct"]) == 1


def test_weekly_returns_use_each_markets_own_week_end():
    a = s(np.linspace(100, 120, 20), "2024-01-01", "A")
    b = s(np.linspace(50, 60, 20), "2024-01-01", "B").drop(pd.Timestamp("2024-01-05"))  # B shut on a Friday
    w = data.weekly_log_returns({"A": a, "B": b})
    assert w.index.dayofweek.unique().tolist() == [4]
    assert not w.isna().any().any()
