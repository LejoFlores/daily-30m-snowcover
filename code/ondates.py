"""Snow-on / snow-off metrics from a daily snow/no-snow series for one water year.

Definitions (per water year, Oct 1 - Sep 30, day-of-WY 1 = Oct 1):
- Gaps (missing days) are filled when both neighbours agree; otherwise left missing.
- No-snow interruptions of <= GAP days inside snow cover are bridged.
- Main season = the longest continuous snow-covered run after bridging.
  snow_on  = first day of that run; snow_off = first snow-free day after it
  (the melt-out date).
- first_snow / last_snow = first and last snow day anywhere in the WY.
- scd = number of snow days (observed, unbridged); n_valid = days with a decision.
- Runs shorter than MINRUN days are not a "season": snow_on/off left NaN
  (ephemeral-snow years at low elevation), scd still reported.
"""
import numpy as np
import pandas as pd

GAP = 3
MINRUN = 14


def wy_of(dates):
    d = pd.DatetimeIndex(dates)
    return d.year + (d.month >= 10)


def dowy(dates):
    d = pd.DatetimeIndex(dates)
    start = pd.to_datetime([f"{y - 1}-10-01" for y in wy_of(d)])
    return (d - start).days + 1


def _runs(b):
    """start, end (exclusive) of True runs in boolean array"""
    x = np.concatenate([[0], b.astype(np.int8), [0]])
    d = np.diff(x)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def season(s, gap=GAP, minrun=MINRUN):
    """s: float array (1 snow, 0 no snow, nan missing) for consecutive days of one WY.
    Returns dict of day indices (0-based) and counts."""
    s = np.asarray(s, float).copy()
    n = len(s)
    out = dict(snow_on=np.nan, snow_off=np.nan, first_snow=np.nan, last_snow=np.nan,
               scd=np.nan, n_valid=int(np.isfinite(s).sum()), season_len=np.nan, censored=0)
    if out["n_valid"] < 0.5 * n:
        return out
    # fill missing where neighbours (last/next valid) agree
    v = pd.Series(s)
    f, b = v.ffill(), v.bfill()
    s = np.where(np.isnan(s) & (f == b), f, s)
    snow = s == 1
    out["scd"] = int(snow.sum())
    if not snow.any():
        out["scd"] = 0
        return out
    idx = np.flatnonzero(snow)
    out["first_snow"], out["last_snow"] = idx[0], idx[-1]
    # bridge short no-snow (or missing) interruptions
    ns = ~snow
    st, en = _runs(ns)
    br = snow.copy()
    for a, e in zip(st, en):
        if a > 0 and e < n and (e - a) <= gap:
            br[a:e] = True
    st, en = _runs(br)
    L = en - st
    k = int(np.argmax(L))
    if L[k] < minrun:
        return out
    out["snow_on"], out["snow_off"], out["season_len"] = st[k], en[k], L[k]
    if en[k] >= n or st[k] == 0:
        out["censored"] = 1
    return out


def by_wy(dates, status, **kw):
    """dates: DatetimeIndex (daily, may have holes); status: 1/0/nan. Returns DataFrame per WY
    with day-of-WY (1 = Oct 1) and calendar dates."""
    ser = pd.Series(np.asarray(status, float), index=pd.DatetimeIndex(dates)).sort_index()
    ser = ser[~ser.index.duplicated()]
    rows = []
    for wy in np.unique(wy_of(ser.index)):
        full = pd.date_range(f"{wy - 1}-10-01", f"{wy}-09-30")
        x = ser.reindex(full).values
        r = season(x, **kw)
        r["wy"] = wy
        for k in ["snow_on", "snow_off", "first_snow", "last_snow"]:
            i = r[k]
            r[k + "_date"] = full[int(i)] if np.isfinite(i) and int(i) < len(full) else pd.NaT
            r[k] = i + 1 if np.isfinite(i) else np.nan  # day of WY
        rows.append(r)
    return pd.DataFrame(rows)


def series_onoff(t, s, wy):
    """t: dates of clear obs (sorted), s: SCA (0-1). Main season = longest span of obs with s >= 0.5,
    bridging one isolated low obs when its neighbours are <= 16 d apart. Crossing dates by
    linear interpolation in SCA between the bracketing obs."""
    out = dict(wy=wy, snow_on=pd.NaT, snow_off=pd.NaT, on_bracket_days=np.nan, off_bracket_days=np.nan, n_obs=len(t))
    if len(t) < 6:
        return out
    b = s >= 0.5
    for i in range(1, len(b) - 1):
        if not b[i] and b[i - 1] and b[i + 1] and (t[i + 1] - t[i - 1]).days <= 16:
            b[i] = True
    x = np.concatenate([[0], b.astype(int), [0]])
    st, en = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
    if len(st) == 0:
        return out
    span = [(t[e - 1] - t[a]).days for a, e in zip(st, en)]
    k = int(np.argmax(span))
    if span[k] < 14:
        return out
    a, e = st[k], en[k]

    def cross(i0, i1):
        s0, s1 = s[i0], s[i1]
        w = 0.5 if s1 == s0 else (0.5 - s0) / (s1 - s0)
        return t[i0] + pd.Timedelta(days=float(np.clip(w, 0, 1)) * (t[i1] - t[i0]).days)

    if a > 0:
        out["snow_on"], out["on_bracket_days"] = cross(a - 1, a), (t[a] - t[a - 1]).days
    if e < len(t):
        out["snow_off"], out["off_bracket_days"] = cross(e - 1, e), (t[e] - t[e - 1]).days
    return out
