"""Daily MODIS/VIIRS cloud-gap-filled snow cover -> snow-on/off per pixel and per band.

usage: python3 process_daily.py PRODUCT DATA_DIR GRID_NPZ OUTDIR [NDSI_THRESHOLDS]
PRODUCT in MOD10A1F, MYD10A1F, VNP10A1F.
Status per pixel-day: 1 if CGF NDSI snow cover >= T, 0 if 0 <= NDSI < T, NaN otherwise
(night, fill, missing). The CGF field carries forward the last clear-sky value
under cloud; Cloud_Persistence (days) is kept and pixel-days with persistence
> MAXPERSIST are set missing so a long cloudy spell cannot carry a stale state.
"""
import os, sys, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ondates import by_wy

prod, ddir, grid_npz, outdir = sys.argv[1:5]
TH = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [10, 20, 30, 40]
MAXPERSIST = 10
os.makedirs(outdir, exist_ok=True)
G = np.load(grid_npz)
tag = "viirs" if prod.startswith("V") else "modis"
pix_area = (370.65 if tag == "viirs" else 463.31) ** 2

files = sorted(glob.glob(f"{ddir}/A*.npz"))
dates = pd.to_datetime([os.path.basename(f)[1:8] for f in files], format="%Y%j")
cgf = np.stack([np.load(f)["CGF_NDSI_Snow_Cover"] for f in files]).astype(np.int16)
cp = np.stack([np.load(f)["Cloud_Persistence"] for f in files]).astype(np.int16)
print(prod, len(files), dates.min().date(), dates.max().date(), cgf.shape, flush=True)

# pixels: majority of area in Dry Creek (1) or Bogus box (2)
ndc, nbg = G[f"{tag}_n_dc"] * 900, G[f"{tag}_n_bogus"] * 900
region = np.where(ndc > 0.5 * pix_area, 1, np.where(nbg > 0.5 * pix_area, 2, 0))
rows, cols = np.nonzero(region)
pix = pd.DataFrame(dict(row=rows, col=cols, region=np.where(region[rows, cols] == 1, "dry_creek", "bogus"),
                        lat=G[f"{tag}_lat"][rows, cols], lon=G[f"{tag}_lon"][rows, cols],
                        elev=G[f"{tag}_elev"][rows, cols], northness=G[f"{tag}_northness"][rows, cols],
                        slope=G[f"{tag}_slope"][rows, cols]))
pix["pixel_id"] = [f"{tag}_r{r:02d}c{c:03d}" for r, c in zip(rows, cols)]
pix["elev_band"] = (np.floor(pix.elev / 200) * 200).astype(int)
pix["aspect_class"] = np.where(pix.northness >= 0, "N", "S")
pix.to_csv(f"{outdir}/{prod}_pixels.csv", index=False)

v = cgf[:, rows, cols].astype(float)
p = cp[:, rows, cols]
valid = (v >= 0) & (v <= 100) & (p <= MAXPERSIST)
res_pix, res_band, daily_band = [], [], []
for T in TH:
    st = np.where(valid, (v >= T).astype(float), np.nan)
    for j in range(len(pix)):
        r = by_wy(dates, st[:, j])
        r.insert(0, "pixel_id", pix.pixel_id.iat[j]); r.insert(0, "ndsi_threshold", T)
        res_pix.append(r)
    # band = region x elevation band (all aspects) and region x band x aspect
    for keys in (["region", "elev_band"], ["region", "elev_band", "aspect_class"], ["region"]):
        for k, g in pix.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            sub = st[:, g.index.values]
            nval = np.isfinite(sub).sum(1)
            frac = np.where(nval >= 0.5 * len(g), np.nansum(sub, 1) / np.maximum(nval, 1), np.nan)
            stat = np.where(np.isfinite(frac), (frac >= 0.5).astype(float), np.nan)
            r = by_wy(dates, stat)
            d = dict(zip(keys, k)); d.setdefault("elev_band", "all"); d.setdefault("aspect_class", "all")
            for kk, vv in d.items():
                r[kk] = vv
            r["ndsi_threshold"], r["n_pixels"] = T, len(g)
            res_band.append(r)
            if T == 10:
                daily_band.append(pd.DataFrame(dict(date=dates, sca=frac, n_valid=nval, **d)))

P = pd.concat(res_pix, ignore_index=True)
P.to_parquet(f"{outdir}/{prod}_pixel_onoff.parquet")
B = pd.concat(res_band, ignore_index=True)
B["product"] = prod
B.to_csv(f"{outdir}/{prod}_band_onoff.csv", index=False)
pd.concat(daily_band, ignore_index=True).astype({"elev_band": str}).to_parquet(f"{outdir}/{prod}_band_daily_sca.parquet")
print(B[(B.ndsi_threshold == 10) & (B.aspect_class == "all")].groupby(["region", "elev_band"])
      [["snow_on", "snow_off", "scd"]].median().round(0), flush=True)
