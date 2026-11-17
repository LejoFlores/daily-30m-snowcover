"""Daily 30 m snow cover 2000-2026 from downscaled MOD10A1F, and per-cell snow-on/off per WY.

usage: python3 ds_run.py TERRAIN_PKL MOD_DIR KIND W OUTDIR
Writes (OUTDIR):
  ds30_<kind>_snow_on_dowy.tif, ds30_<kind>_snow_off_dowy.tif, ds30_<kind>_scd.tif  (bands = WY2000..2026)
  ds30_<kind>_daily_<WY>.npz  packed daily snow (domain cells), for reuse
  ds30_<kind>_band_onoff.csv  per region x elevation band x aspect from the daily band SCA (>= 50%)
Same season rules as ondates.py (gaps <= 3 d bridged, main season = longest run >= 14 d).
MODIS days with Cloud_Persistence > 10 at a pixel are missing there; WY2000 snow-on is
unobserved (MODIS starts 24 Feb 2000) and set missing.
"""
import os, sys, glob, pickle
import numpy as np
import pandas as pd
import numba as nb
import rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downscale import Terrain, ndsi_to_fsca
from ondates import by_wy

T = pickle.load(open(sys.argv[1], "rb"))
mdir, kind, w, outdir = sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
os.makedirs(outdir, exist_ok=True)
if kind.startswith("emp"):
    PP = np.load(os.environ["DS_PERSIST"])
    T.set_persistence(PP["P_all"], "all"); T.emp_fold = "all"
dom = (T.region > 0) & T.valid
iy, ix = np.nonzero(dom)
NY, NX = dom.shape


@nb.njit(parallel=True, cache=True)
def seasons(st, gap, minrun):
    """st: (ndays, ncell) int8 1/0/-1(missing). returns on, off (0-based day, -1 none), scd"""
    nd, nc = st.shape
    on = np.full(nc, -1); off = np.full(nc, -1); scd = np.full(nc, -1)
    for j in nb.prange(nc):
        s = st[:, j].copy()
        nval = 0
        for i in range(nd):
            if s[i] >= 0:
                nval += 1
        if nval < nd // 2:
            continue
        # fill missing where last and next valid agree
        last = -1
        for i in range(nd):
            if s[i] >= 0:
                last = s[i]
            elif last >= 0:
                k = i
                while k < nd and s[k] < 0:
                    k += 1
                if k < nd and s[k] == last:
                    for q in range(i, k):
                        s[q] = last
        c = 0
        for i in range(nd):
            if s[i] == 1:
                c += 1
        scd[j] = c
        # bridge short non-snow gaps between snow days
        b = np.zeros(nd, np.int8)
        for i in range(nd):
            b[i] = 1 if s[i] == 1 else 0
        i = 0
        while i < nd:
            if b[i] == 0:
                k = i
                while k < nd and b[k] == 0:
                    k += 1
                if i > 0 and k < nd and (k - i) <= gap:
                    for q in range(i, k):
                        b[q] = 1
                i = k
            else:
                i += 1
        best = 0; bs = -1; i = 0
        while i < nd:
            if b[i] == 1:
                k = i
                while k < nd and b[k] == 1:
                    k += 1
                if k - i > best:
                    best = k - i; bs = i
                i = k
            else:
                i += 1
        if best >= minrun:
            on[j] = bs; off[j] = bs + best
    return on, off, scd


files = sorted(glob.glob(f"{mdir}/A*.npz"))
dates = pd.to_datetime([os.path.basename(f)[1:8] for f in files], format="%Y%j")
wy = dates.year + (dates.month >= 10)
pr_cache = {}
wys = list(range(2000, 2027))
ON = np.full((len(wys), NY, NX), np.nan, np.float32); OFF = ON.copy(); SCD = ON.copy()
band_rows = []
asp = np.where(T.slope < 5, "flat", np.select([(T.aspect >= 315) | (T.aspect < 45), T.aspect < 135, T.aspect < 225],
                                               ["N", "E", "S"], "W"))
eb = np.floor(T.z / 200) * 200
from domain import X0 as _X0, Y1 as _Y1, region_name
reg = np.vectorize(region_name)(T.region)
cells = pd.DataFrame(dict(region=reg[iy, ix], elev_band=eb[iy, ix], aspect_class=asp[iy, ix]))
for k_, wyv in enumerate(wys):
    full = pd.date_range(f"{wyv - 1}-10-01", f"{wyv}-09-30")
    st = np.full((len(full), len(iy)), -1, np.int8)
    idx = np.flatnonzero(wy == wyv)
    for i in idx:
        d = np.load(files[i])
        nd = d["CGF_NDSI_Snow_Cover"].astype(float); cp = d["Cloud_Persistence"]
        f = np.where((nd <= 100) & (cp <= 10), ndsi_to_fsca(nd), np.nan)
        doy = dates[i].dayofyear
        key = 0 if kind.startswith("emp") else int(np.argmin(np.abs(T.doys - doy)))
        if key not in pr_cache:
            pr_cache[key] = T.prank(doy, w, kind)
        sn, ok = T.snow(pr_cache[key], T.fsca_field(f, kind))
        row = (dates[i] - full[0]).days
        st[row] = np.where(ok[iy, ix], sn[iy, ix].astype(np.int8), -1)
    if len(idx) == 0:
        continue
    np.savez_compressed(f"{outdir}/ds30_{kind}_daily_WY{wyv}.npz", snow=np.packbits(st == 1, axis=1),
                        valid=np.packbits(st >= 0, axis=1), dates=full.values.astype("datetime64[D]"),
                        iy=iy, ix=ix, ncell=len(iy))
    on, off, scd = seasons(st, 3, 14)
    if wyv == 2000:
        on[:] = -1
    ON[k_, iy, ix] = np.where(on >= 0, on + 1, np.nan)
    OFF[k_, iy, ix] = np.where((off >= 0) & (off < len(full)), off + 1, np.nan)
    SCD[k_, iy, ix] = np.where(scd >= 0, scd, np.nan)
    # band SCA -> on/off
    for keys in (["region", "elev_band", "aspect_class"], ["region", "elev_band"], ["region"]):
        for kk, g in cells.groupby(keys):
            kk = kk if isinstance(kk, tuple) else (kk,)
            sub = st[:, g.index.values]
            nv = (sub >= 0).sum(1); ns = (sub == 1).sum(1)
            frac = np.where(nv >= 0.5 * len(g), ns / np.maximum(nv, 1), np.nan)
            r = by_wy(full, np.where(np.isfinite(frac), (frac >= 0.5).astype(float), np.nan)).iloc[0].to_dict()
            dd = dict(zip(keys, kk)); dd.setdefault("elev_band", "all"); dd.setdefault("aspect_class", "all")
            r.update(dd); r["n_cells"] = len(g); r["mean_sca_dec_jun"] = float(np.nanmean(frac[61:273]))
            if wyv == 2000:
                r["snow_on"] = np.nan; r["snow_on_date"] = pd.NaT
            band_rows.append(r)
    print(wyv, len(idx), "days", flush=True)

prof = dict(driver="GTiff", height=NY, width=NX, count=len(wys), dtype="float32", crs="EPSG:32611",
            transform=rasterio.transform.from_origin(_X0, _Y1, 30, 30), nodata=np.nan, compress="deflate")
for name, arr in [("snow_on_dowy", ON), ("snow_off_dowy", OFF), ("scd", SCD)]:
    with rasterio.open(f"{outdir}/ds30_{kind}_{name}_wy2000_2026.tif", "w", **prof) as r:
        r.write(arr); r.descriptions = tuple(f"WY{x}" for x in wys)
B = pd.DataFrame(band_rows)
B.to_csv(f"{outdir}/ds30_{kind}_band_onoff.csv", index=False)
print(B[B.aspect_class == "all"].groupby(["region", "elev_band"])[["snow_on", "snow_off", "scd"]].median())
