"""Satellite snow-on/off at ground sites vs ground-based dates.

usage: python3 validate.py DATA_DIR GRID_NPZ GROUND_CSV HIRES_DIR OUTDIR
"""
import os, sys, glob
import numpy as np
import pandas as pd
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ondates import by_wy, series_onoff, wy_of

ddir, grid_npz, ground_csv, hdir, outdir = sys.argv[1:6]
G = np.load(grid_npz)
gr = pd.read_csv(ground_csv)
sites = gr.groupby("site_id")[["lat", "lon", "elevation_m"]].first().reset_index()

R = 6371007.181; X0S, Y0S = -20015109.354, 10007554.677; TILE = 1111950.5197665
ulx, uly = X0S + 9 * TILE, Y0S - 4 * TILE
rows = []
for prod, npix, r0, c0 in [("MOD10A1F", 2400, 1478, 1400), ("MYD10A1F", 2400, 1478, 1400), ("VNP10A1F", 3000, 1847, 1750)]:
    files = sorted(glob.glob(f"{ddir}/{prod}/A*.npz"))
    dates = pd.to_datetime([os.path.basename(f)[1:8] for f in files], format="%Y%j")
    ps = TILE / npix
    pr = {}
    for s in sites.itertuples():
        sx = R * np.radians(s.lon) * np.cos(np.radians(s.lat)); sy = R * np.radians(s.lat)
        pr[s.site_id] = (int((uly - sy) // ps) - r0, int((sx - ulx) // ps) - c0)
    cube = {sid: [] for sid in pr}; cpc = {sid: [] for sid in pr}
    for f in files:
        d = np.load(f); a = d["CGF_NDSI_Snow_Cover"]; c = d["Cloud_Persistence"]
        for sid, (r, cc) in pr.items():
            cube[sid].append(a[r, cc]); cpc[sid].append(c[r, cc])
    for sid in pr:
        v = np.array(cube[sid], float); p = np.array(cpc[sid])
        ok = (v <= 100) & (p <= 10)
        for T in (10, 20, 30, 40):
            st = np.where(ok, (v >= T).astype(float), np.nan)
            r = by_wy(dates, st)
            r["site_id"], r["product"], r["ndsi_threshold"] = sid, prod, T
            rows.append(r)
    print(prod, "done", flush=True)
SAT = pd.concat(rows, ignore_index=True)

# hi-res: 3x3 neighbourhood SCA at each site
M = pd.read_csv(f"{hdir}/hires_scenes.csv", parse_dates=["date"])
S = np.load(f"{hdir}/hires_snowmaps.npy", mmap_mode="r")
tr = Transformer.from_crs(4326, 32611, always_xy=True)
hrows = []
wys = wy_of(M.date)
for s in sites.itertuples():
    x, y = tr.transform(s.lon, s.lat)
    c, r = int((x - 564480) // 30), int((4849500 - y) // 30)
    sub = np.asarray(S[:, r - 1:r + 2, c - 1:c + 2]).reshape(len(M), -1)
    ncl = (sub != 255).sum(1); sca = np.where(ncl >= 5, (sub == 1).sum(1) / np.maximum(ncl, 1), np.nan)
    for wy in range(1999, 2027):
        sel = (wys == wy) & np.isfinite(sca)
        o = series_onoff(list(pd.to_datetime(M.date[sel].values)), sca[sel], wy)
        o["site_id"] = s.site_id
        hrows.append(o)
H = pd.DataFrame(hrows)
start = pd.to_datetime((H.wy - 1).astype(str) + "-10-01")
for c in ["snow_on", "snow_off"]:
    H[c] = (pd.to_datetime(H[c]) - start).dt.days + 1
H["product"], H["ndsi_threshold"] = "Landsat+S2_30m", np.nan
SAT = pd.concat([SAT, H], ignore_index=True)
SAT.to_csv(f"{outdir}/sites_satellite_onoff_by_wy.csv", index=False)

# reference: snow depth / SNOTEL where present, else soil temperature
pref = {"snow_depth": 0, "snotel": 0, "soil_temp_2cm": 1, "soil_temp_5cm": 2}
gr["rank"] = gr.source.map(pref)
ref = gr.sort_values("rank").groupby(["site_id", "wy"]).first().reset_index()
ref = ref[ref.censored == 0]
J = SAT.merge(ref[["site_id", "wy", "source", "snow_on", "snow_off", "scd", "elevation_m"]],
              on=["site_id", "wy"], suffixes=("_sat", "_gnd"))
J["d_on"] = J.snow_on_sat - J.snow_on_gnd
J["d_off"] = J.snow_off_sat - J.snow_off_gnd
J.to_csv(f"{outdir}/validation_pairs.csv", index=False)


def summ(g):
    return pd.Series(dict(n_off=g.d_off.notna().sum(), bias_off=g.d_off.median(), mae_off=g.d_off.abs().median(),
                          r_off=g[["snow_off_sat", "snow_off_gnd"]].corr().iloc[0, 1],
                          n_on=g.d_on.notna().sum(), bias_on=g.d_on.median(), mae_on=g.d_on.abs().median()))


S1 = J.groupby(["product", "ndsi_threshold"], dropna=False).apply(summ).round(2)
S2 = J[J.ndsi_threshold.isin([20]) | J.ndsi_threshold.isna()].groupby(["product", "site_id"]).apply(summ).round(1)
S1.to_csv(f"{outdir}/validation_summary_by_product.csv")
S2.to_csv(f"{outdir}/validation_summary_by_site.csv")
print(S1.to_string()); print(S2.to_string())
