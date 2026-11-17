"""Reynolds Creek ground validation of the 30 m downscaled snow seasons against ARS daily snow depth.
Ground snow = depth > 5 cm (same rule as Dry Creek). Seasons from ondates.season (gaps <= 3 d bridged,
main season >= 14 d). Satellite dates read at the 30 m cell containing each site (and the 3x3 median).
usage: python3 validate_rcew_ground.py DEPTH_CSV META_CSV SEASONS_NC ONDATES_DIR OUT_DIR"""
import os, sys
import numpy as np
import pandas as pd
import netCDF4
from pyproj import Transformer

depth_csv, meta_csv, nc_path, ondir, out = sys.argv[1:6]
sys.path.insert(0, ondir)
from ondates import season, wy_of

THR = 5.0  # cm
D = pd.read_csv(depth_csv, parse_dates=["date"], date_format="%m/%d/%Y").set_index("date").asfreq("D")
M = pd.read_csv(meta_csv)
for c in ["latitude", "longitude"]:
    M[c] = M[c].astype(str).str.replace("°", "", regex=False).astype(float)
M = M.set_index("siteID")

N = netCDF4.Dataset(nc_path)
x, y = N["x"][:], N["y"][:]
wys = N["water_year"][:].astype(int)
on_nc, off_nc, scd_nc = N["snow_on"], N["snow_off"], N["snow_cover_days"]
forest, elev30 = N["forest"][:], N["elevation"][:]
tr = Transformer.from_crs(4326, 32611, always_xy=True)

rows = []
for sid in D.columns:
    if sid not in M.index:
        print("no metadata for", sid); continue
    m = M.loc[sid]
    ex, ny_ = tr.transform(m.longitude, m.latitude)
    ix, iy = int(np.argmin(abs(x - ex))), int(np.argmin(abs(y - ny_)))
    sat_on = np.ma.filled(on_nc[:, iy, ix].astype(float), np.nan)
    sat_off = np.ma.filled(off_nc[:, iy, ix].astype(float), np.nan)
    sat_scd = np.ma.filled(scd_nc[:, iy, ix].astype(float), np.nan)
    blk_on = np.ma.filled(on_nc[:, iy - 1:iy + 2, ix - 1:ix + 2].astype(float), np.nan).reshape(len(wys), -1)
    blk_off = np.ma.filled(off_nc[:, iy - 1:iy + 2, ix - 1:ix + 2].astype(float), np.nan).reshape(len(wys), -1)
    s = D[sid]
    st = np.where(s.isna(), np.nan, (s > THR).astype(float))
    wy = wy_of(D.index)
    for w in np.unique(wy):
        k = wy == w
        g = season(st[k])
        # day-of-WY 1 = Oct 1; the record starts mid-year in the first WY, so offset by the first date
        first = D.index[k][0]
        off0 = (first - pd.Timestamp(w - 1, 10, 1)).days
        gon = g["snow_on"] + off0 + 1 if np.isfinite(g["snow_on"]) else np.nan
        goff = g["snow_off"] + off0 + 1 if np.isfinite(g["snow_off"]) else np.nan
        # winter coverage: share of valid days Nov-May
        win = k & D.index.month.isin([11, 12, 1, 2, 3, 4, 5])
        cov = float(np.isfinite(st[win]).mean()) if win.any() else 0.0
        j = np.flatnonzero(wys == w)
        r = dict(site=sid, grouping=m.grouping, elev_site=m.elevation, elev_cell=float(elev30[iy, ix]),
                 forest=int(forest[iy, ix]), wy=int(w), cov_nov_may=round(cov, 2), censored=g["censored"],
                 ground_on=gon, ground_off=goff, ground_scd=g["scd"], max_depth=float(s[k].max()))
        if len(j):
            j = j[0]
            r.update(sat_on=sat_on[j], sat_off=sat_off[j], sat_scd=sat_scd[j],
                     sat_on_3x3=np.nanmedian(blk_on[j]) if np.isfinite(blk_on[j]).any() else np.nan,
                     sat_off_3x3=np.nanmedian(blk_off[j]) if np.isfinite(blk_off[j]).any() else np.nan)
        rows.append(r)
P = pd.DataFrame(rows)
os.makedirs(out, exist_ok=True)
P.to_csv(f"{out}/rcew_ground_vs_30m_by_site_wy.csv", index=False)


def stats(df, a, b):
    e = (df[b] - df[a]).dropna()
    if len(e) == 0:
        return dict(n=0)
    r = np.corrcoef(df.loc[e.index, a], df.loc[e.index, b])[0, 1] if len(e) > 2 else np.nan
    return dict(n=len(e), bias=round(float(e.median()), 1), mean_err=round(float(e.mean()), 1),
                mae=round(float(e.abs().mean()), 1), r=round(float(r), 2))


ok = P[(P.cov_nov_may >= 0.8) & (P.censored == 0)]
summ = []
for lab, sub in [("all", ok), ("open", ok[ok.forest == 0]), ("forest", ok[ok.forest == 1]),
                 ("< 1,600 m", ok[ok.elev_site < 1600]), ("1,600-1,900 m", ok[(ok.elev_site >= 1600) & (ok.elev_site < 1900)]),
                 (">= 1,900 m", ok[ok.elev_site >= 1900])]:
    for what, a, b in [("melt-out", "ground_off", "sat_off"), ("melt-out 3x3", "ground_off", "sat_off_3x3"),
                       ("snow-on", "ground_on", "sat_on"), ("snow days", "ground_scd", "sat_scd")]:
        summ.append(dict(subset=lab, metric=what, **stats(sub, a, b)))
S = pd.DataFrame(summ)
S.to_csv(f"{out}/rcew_ground_validation_summary.csv", index=False)
print(S.to_string())
# season agreement: does the satellite find a season when the ground does?
both = ok.dropna(subset=["sat_scd"])
print("ground season yes:", both.ground_off.notna().sum(), " sat season yes:", both.sat_off.notna().sum(),
      " both:", (both.ground_off.notna() & both.sat_off.notna()).sum(), " n site-years:", len(both))
bysite = pd.DataFrame({sid: stats(g, "ground_off", "sat_off") for sid, g in ok.groupby("site")}).T
bysite = bysite.join(M[["elevation", "grouping"]]).join(P.groupby("site")[["forest"]].first())
bysite.to_csv(f"{out}/rcew_ground_validation_by_site.csv")
print(bysite.sort_values("elevation").to_string())
