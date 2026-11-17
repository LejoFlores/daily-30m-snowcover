"""Ground-based snow-on/off per water year:
- snow depth (DCEW weather stations TL, LW, BRW): snow if daily mean depth > 5 cm
- Bogus Basin SNOTEL 978: snow if SWE > 0.1 in (2.5 mm) (or depth > 5 cm when SWE missing)
- DCEW aspect soil stations: snow if the 2 cm (5 cm at TL/BRW/LW) soil-temperature
  diurnal range < 1 C and daily max < 2 C (Lundquist & Lott 2008 style)
usage: python3 ground_truth.py DRYCREEK_DIR SNOTEL_JSON OUT_CSV OUT_DAILY_PARQUET
"""
import sys, json, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ondates import by_wy

dc, snotel_json, out_csv, out_daily = sys.argv[1:5]
sites = pd.read_csv(f"{dc}/meta/sites.csv").set_index("site_id")
daily = []

w = pd.read_parquet(f"{dc}/weather/weather_daily.parquet")
sd = w[w.variable == "snow_depth"].copy()
print("snow depth units:", sd.units.unique())
at = w[w.variable == "air_temp"]
for sid, g in sd.groupby("site_id"):
    g = g[g.n_valid_hours >= 12].set_index(pd.to_datetime(g[g.n_valid_hours >= 12].date))
    # the source blanks snow-free periods, so a missing depth on a day the station
    # logged air temperature is read as "no snow" -- except Dec-Mar, where it stays missing
    days = pd.DatetimeIndex(pd.to_datetime(at[at.site_id == sid].date)).union(g.index)
    v = g.value.reindex(days)
    st = pd.Series(np.where(v.notna(), (v > 5).astype(float), np.nan), index=days)
    st[v.isna() & ~days.month.isin([12, 1, 2, 3])] = 0.0
    daily.append(pd.DataFrame(dict(source="snow_depth", site_id=sid, date=days, snow=st.values)))

# SNOTEL
j = json.load(open(snotel_json))
el = {d["stationElement"]["elementCode"]: d["values"] for d in j[0]["data"]}
swe = pd.Series({pd.Timestamp(x["date"]): x.get("value") for x in el.get("WTEQ", [])}, dtype=float)
snwd = pd.Series({pd.Timestamp(x["date"]): x.get("value") for x in el.get("SNWD", [])}, dtype=float)
idx = swe.index.union(snwd.index)
swe, snwd = swe.reindex(idx), snwd.reindex(idx)
st = np.where(swe.notna(), (swe > 0.1).astype(float), np.where(snwd.notna(), (snwd > 2).astype(float), np.nan))
# summer SWE sensor noise: treat Jul-Sep as snow-free unless depth also > 2 in
daily.append(pd.DataFrame(dict(source="snotel", site_id="SNOTEL978", date=idx, snow=st)))
sites.loc["SNOTEL978", ["site_name", "lat", "lon", "elevation_m"]] = ["Bogus Basin SNOTEL", 43.76377, -116.09685, 1942]

# soil temperature
sh = pd.read_parquet(f"{dc}/soil_v2/soil_hourly.parquet",
                     columns=None, filters=[("variable", "==", "soil_temp")])
print("soil hourly cols:", sh.columns.tolist())
tcol = [c for c in sh.columns if c in ("datetime", "time", "timestamp")][0]
sh = sh[sh.depth_cm <= 5]
sh["date"] = pd.to_datetime(sh[tcol]).dt.floor("D")
for sid, g in sh.groupby("site_id"):
    # shallowest depth, one sensor (most data) per site
    dmin = g.depth_cm.min()
    g = g[g.depth_cm == dmin]
    best = g.sensor_id.value_counts().index[0]
    g = g[g.sensor_id == best]
    if "qc_flag" in g:
        g = g[g.qc_flag.astype(str).isin(["ok", "", "nan", "None"])]
    a = g.groupby("date").value.agg(["min", "max", "count"])
    a = a[a["count"] >= 20]
    st = ((a["max"] - a["min"] < 1.0) & (a["max"] < 2.0)).astype(float)
    daily.append(pd.DataFrame(dict(source=f"soil_temp_{dmin:g}cm", site_id=sid, date=a.index, snow=st.values)))

D = pd.concat(daily, ignore_index=True)
D.to_parquet(out_daily)
res = []
for (src, sid), g in D.groupby(["source", "site_id"]):
    r = by_wy(g.date, g.snow)
    r.insert(0, "site_id", sid); r.insert(0, "source", src)
    res.append(r)
R = pd.concat(res, ignore_index=True)
R = R.merge(sites[["site_name", "lat", "lon", "elevation_m"]], left_on="site_id", right_index=True, how="left")
R = R[(R.wy >= 1999) & (R.n_valid >= 250)]
R.to_csv(out_csv, index=False)
print(R.groupby(["source", "site_id"]).agg(n=("wy", "size"), on=("snow_on", "median"), off=("snow_off", "median"),
                                             scd=("scd", "median"), elev=("elevation_m", "first")).round(0))
