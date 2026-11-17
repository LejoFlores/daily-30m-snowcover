"""Landsat (C1 TOA) + Sentinel-2 (L2A BOA) 30 m snow maps -> snow-cover time series
per elevation/aspect band and snow-on/off dates per band and per pixel.

usage: python3 process_hires.py LANDSAT_DIR S2_DIR GRID_NPZ OUTDIR

Snow test (per clear pixel): NDSI = (green - swir1)/(green + swir1)
  open pixels:   NDSI >= 0.40 and nir >= 0.11 and green >= 0.10   (SNOMAP)
  forest pixels: NDSI >= 0.20 and nir >= 0.11 and green >= 0.08   (canopy-relaxed)
  forest = summer (Jul-Aug, 2013-2021) median NDVI >= 0.45
Clear: Landsat BQA not fill, no cloud bit / high cloud or shadow confidence (L8 also
cirrus); S2 SCL not 0,1,3,8,9,10. Pixels flagged cloud with NDSI >= 0.6 and
swir1 < 0.15 are taken as snow (cloud masks often call bright snow cloud; real
clouds are bright in SWIR).
"""
import os, sys, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ondates import wy_of, series_onoff

ls_dir, s2_dir, grid_npz, outdir = sys.argv[1:5]
os.makedirs(outdir, exist_ok=True)
G = np.load(grid_npz)
region, elev, aspect, slope = G["region"], G["elev"], G["aspect"], G["slope"]
NY, NX = region.shape


def bits(a, start, n=1):
    return (a >> start) & ((1 << n) - 1)


def load(f):
    d = np.load(f)
    g = {k: d[k].astype(np.float32) / 10000 for k in ("green", "red", "nir", "swir1")}
    bad = np.zeros((NY, NX), bool)
    for k in g:
        bad |= d[k] == -32768
        g[k][d[k] == -32768] = np.nan
    if "qa_pixel" in d:  # Landsat Collection 2 L2 (surface reflectance)
        q = d["qa_pixel"].astype(np.int32)
        sensor = str(d["sensor"])
        cloud = (bits(q, 3) == 1) | (bits(q, 4) == 1) | (bits(q, 8, 2) == 3)
        if sensor in ("LC08", "LC09"):
            cloud |= bits(q, 14, 2) == 3
        bad |= bits(q, 0) == 1
        src = sensor
    elif "bqa" in d:
        q = d["bqa"].astype(np.int32)
        sensor = str(d["sensor"])
        cloud = (bits(q, 4) == 1) | (bits(q, 5, 2) == 3) | (bits(q, 7, 2) == 3)
        if sensor == "LC08":
            cloud |= bits(q, 11, 2) == 3
        bad |= (q == 0) | (bits(q, 0) == 1) | (bits(q, 1) == 1)
        src = sensor
    else:
        s = d["scl"]
        cloud = np.isin(s, [3, 8, 9, 10])
        bad |= np.isin(s, [0, 1])
        src = "S2"
    return str(d["date"]), src, g, cloud, bad


ls_files = sorted(glob.glob(f"{ls_dir}/*.npz"))
s2_files = sorted(glob.glob(f"{s2_dir}/*.npz"))
print(len(ls_files), "landsat", len(s2_files), "s2", flush=True)

# pass 1: summer NDVI for forest flag
ndvi_stack = []
for f in ls_files + s2_files:
    base = os.path.basename(f)
    if base.startswith(("LC08", "LC09", "S2")):
        date, src, g, cloud, bad = load(f)
        dt = pd.Timestamp(date)
        if dt.month in (7, 8) and 2013 <= dt.year <= 2025:
            nd = (g["nir"] - g["red"]) / (g["nir"] + g["red"])
            nd[cloud | bad] = np.nan
            ndvi_stack.append(nd)
ndvi_summer = np.nanmedian(np.stack(ndvi_stack), 0)
forest = ndvi_summer >= 0.45
print("summer scenes", len(ndvi_stack), "forest share DC", forest[region == 1].mean().round(3),
      "Bogus", forest[region == 2].mean().round(3), flush=True)

# pass 2: snow maps (uint8: 0 no snow, 1 snow, 255 no data/cloud)
meta, maps = [], []
for f in ls_files + s2_files:
    date, src, g, cloud, bad = load(f)
    ndsi = (g["green"] - g["swir1"]) / (g["green"] + g["swir1"])
    nir_ok = g["nir"] >= 0.11
    snow = np.where(forest, (ndsi >= 0.20) & nir_ok & (g["green"] >= 0.08),
                    (ndsi >= 0.40) & nir_ok & (g["green"] >= 0.10))
    rescue = cloud & (ndsi >= 0.6) & (g["swir1"] < 0.15) & nir_ok
    clear = (~cloud | rescue) & ~bad & np.isfinite(ndsi)
    m = np.full((NY, NX), 255, np.uint8)
    m[clear] = snow[clear].astype(np.uint8)
    dom = region > 0
    cf = clear[dom].mean()
    if cf < 0.05:
        continue
    maps.append(m)
    meta.append(dict(date=pd.Timestamp(date), source=src, file=os.path.basename(f), clear_frac=cf,
                     snow_frac_clear=float(m[dom & clear].mean()) if cf > 0 else np.nan))
M = pd.DataFrame(meta)
S = np.stack(maps)
# same-day duplicates (two Landsat paths never share a day; S2A/B/L sometimes): keep max clear
order = np.argsort(M.date.values, kind="stable")
M, S = M.iloc[order].reset_index(drop=True), S[order]
M.to_csv(f"{outdir}/hires_scenes.csv", index=False)
np.save(f"{outdir}/hires_snowmaps.npy", S)
np.save(f"{outdir}/hires_forest.npy", forest)
print("usable scenes", len(M), M.groupby("source").size().to_dict(), flush=True)

# band definitions on the 30 m grid
asp = np.where(slope < 5, "flat", np.select([(aspect >= 315) | (aspect < 45), aspect < 135, aspect < 225],
                                             ["N", "E", "S"], "W"))
eb = (np.floor(elev / 200) * 200)
reg = np.where(region == 1, "dry_creek", np.where(region == 2, "bogus", ""))
cells = pd.DataFrame(dict(region=reg.ravel(), elev_band=eb.ravel(), aspect_class=asp.ravel(), idx=np.arange(reg.size)))
cells = cells[cells.region != ""]


wys = wy_of(M.date)
rows_sca, rows_on = [], []
for keys in (["region", "elev_band", "aspect_class"], ["region", "elev_band"], ["region"]):
    for k, g in cells.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        d = dict(zip(keys, k)); d.setdefault("elev_band", "all"); d.setdefault("aspect_class", "all")
        if len(g) < 30:  # < 2.7 ha
            continue
        r_, c_ = np.unravel_index(g.idx.values, (NY, NX))
        sub = S[:, r_, c_]
        nclear = (sub != 255).sum(1)
        nsnow = (sub == 1).sum(1)
        cf = nclear / len(g)
        sca = np.where(cf >= 0.6, nsnow / np.maximum(nclear, 1), np.nan)
        df = pd.DataFrame(dict(date=M.date, source=M.source, sca=sca, clear_frac=cf, **d, n_cells=len(g)))
        rows_sca.append(df)
        for wy in range(1985, 2027):
            sel = (wys == wy) & np.isfinite(sca)
            t = pd.to_datetime(M.date[sel].values)
            r = series_onoff(list(t), sca[sel], wy)
            r.update(d); r["n_cells"] = len(g)
            rows_on.append(r)

SCA = pd.concat(rows_sca, ignore_index=True)
SCA["elev_band"] = SCA.elev_band.astype(str)
SCA.to_parquet(f"{outdir}/hires_band_sca.parquet")
ON = pd.DataFrame(rows_on)
for c in ["snow_on", "snow_off"]:
    ON[c] = pd.to_datetime(ON[c])
    start = pd.to_datetime((ON.wy - 1).astype(str) + "-10-01")
    ON[c.replace("snow_", "") + "_dowy"] = (ON[c] - start).dt.days + 1
ON.to_csv(f"{outdir}/hires_band_onoff.csv", index=False)
print(ON[(ON.aspect_class == "all") & (ON.wy >= 2000)].groupby(["region", "elev_band"])[["on_dowy", "off_dowy", "off_bracket_days", "n_obs"]].median().round(0))

# per-pixel melt-out (snow-off) date per WY: last transition from a run of >= 2 clear
# snow observations to a clear no-snow observation between Jan 1 and Aug 31; date is the
# midpoint of the bracketing observations. Pixels never seen snow-covered twice -> NaN.
import rasterio
wy_list = list(range(1985, 2027))
OFF = np.full((len(wy_list), NY, NX), np.nan, np.float32)
BR = np.full((len(wy_list), NY, NX), np.nan, np.float32)
for iw, wy in enumerate(wy_list):
    sel = np.flatnonzero((wys == wy) & (M.date.dt.month.isin([1, 2, 3, 4, 5, 6, 7, 8])).values)
    if len(sel) < 6:
        continue
    start = pd.Timestamp(f"{wy - 1}-10-01")
    prev_t = np.full((NY, NX), np.nan); prev_s = np.zeros((NY, NX), bool); run = np.zeros((NY, NX), int)
    off = np.full((NY, NX), np.nan); br = np.full((NY, NX), np.nan)
    for i in sel:
        m = S[i]; t = (M.date.iat[i] - start).days + 1
        clear = m != 255; sn = m == 1
        trans = clear & ~sn & prev_s & (run >= 2)
        off[trans] = (prev_t[trans] + t) / 2; br[trans] = t - prev_t[trans]
        run = np.where(clear, np.where(sn, np.where(prev_s, run + 1, 1), 0), run)
        prev_s = np.where(clear, sn, prev_s); prev_t = np.where(clear, t, prev_t)
    OFF[iw], BR[iw] = off, br
dom = region > 0
OFF[:, ~dom] = np.nan; BR[:, ~dom] = np.nan
prof = dict(driver="GTiff", height=NY, width=NX, count=len(wy_list), dtype="float32", crs="EPSG:32611",
            transform=rasterio.transform.from_origin(564480, 4849500, 30, 30), nodata=np.nan, compress="deflate")
with rasterio.open(f"{outdir}/hires_pixel_snowoff_dowy_wy1985_2026.tif", "w", **prof) as r:
    r.write(OFF)
    r.descriptions = tuple(f"WY{w}" for w in wy_list)
with rasterio.open(f"{outdir}/hires_pixel_snowoff_bracket_days_wy1985_2026.tif", "w", **prof) as r:
    r.write(BR)
    r.descriptions = tuple(f"WY{w}" for w in wy_list)
iw0 = wy_list.index(2000); iw1 = wy_list.index(2025)
good = np.where(BR[iw0:iw1 + 1] <= 24, OFF[iw0:iw1 + 1], np.nan)
clim = np.nanmedian(good, 0); nyr = np.isfinite(good).sum(0).astype(np.float32)
p1 = dict(prof, count=2)
with rasterio.open(f"{outdir}/hires_pixel_snowoff_median_wy2000_2025.tif", "w", **p1) as r:
    r.write(np.stack([clim, nyr]).astype(np.float32))
    r.descriptions = ("median_snowoff_dowy_bracket_le_24d", "n_years")
print("pixel snow-off maps written", flush=True)
