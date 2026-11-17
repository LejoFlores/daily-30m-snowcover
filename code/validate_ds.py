"""Downscaled 30 m daily snow at ground sites (3x3 cells, snow if >= 5 of the valid cells) vs ground dates.
usage: python3 validate_ds.py DS_DIR KIND GRID_NPZ GROUND_CSV SAT_SITES_CSV OUTDIR
Appends product 'MODIS_ds30_<kind>' to the site table and re-summarises."""
import os, sys, glob
import numpy as np
import pandas as pd
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ondates import by_wy

dsdir, kind, grid_npz, ground_csv, sat_csv, outdir = sys.argv[1:7]
G = np.load(grid_npz)
NY, NX = G["region"].shape
gr = pd.read_csv(ground_csv)
sites = gr.groupby("site_id")[["lat", "lon"]].first().reset_index()
tr = Transformer.from_crs(4326, 32611, always_xy=True)
rows = []
lut = None
for f in sorted(glob.glob(f"{dsdir}/ds30_{kind}_daily_WY*.npz")):
    d = np.load(f)
    nc = int(d["ncell"])
    sn = np.unpackbits(d["snow"], axis=1)[:, :nc].astype(bool)
    va = np.unpackbits(d["valid"], axis=1)[:, :nc].astype(bool)
    if lut is None:
        lut = np.full((NY, NX), -1); lut[d["iy"], d["ix"]] = np.arange(nc)
    dates = pd.to_datetime(d["dates"])
    for s in sites.itertuples():
        x, y = tr.transform(s.lon, s.lat)
        c, r = int((x - 564480) // 30), int((4849500 - y) // 30)
        ids = lut[r - 1:r + 2, c - 1:c + 2].ravel(); ids = ids[ids >= 0]
        if len(ids) < 5:
            continue
        nv = va[:, ids].sum(1); ns = (sn[:, ids] & va[:, ids]).sum(1)
        st = np.where(nv >= 5, (ns >= 0.5 * nv).astype(float), np.nan)
        r_ = by_wy(dates, st); r_["site_id"] = s.site_id
        rows.append(r_)
D = pd.concat(rows, ignore_index=True)
D["product"], D["ndsi_threshold"] = f"MODIS_ds30_{kind}", np.nan
D.loc[D.wy == 2000, "snow_on"] = np.nan
SAT = pd.concat([pd.read_csv(sat_csv), D], ignore_index=True)
SAT.to_csv(f"{outdir}/sites_satellite_onoff_by_wy.csv", index=False)

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


# common site-years only, so products are compared on the same sample
key = J[J["product"].str.startswith("MODIS_ds30") & J.d_off.notna()][["site_id", "wy"]].drop_duplicates()
Jc = J.merge(key, on=["site_id", "wy"])
S1 = J.groupby(["product", "ndsi_threshold"], dropna=False).apply(summ).round(2)
S1c = Jc[Jc.ndsi_threshold.isin([10]) | Jc.ndsi_threshold.isna()].groupby("product").apply(summ).round(2)
S2 = J[J.ndsi_threshold.isin([10]) | J.ndsi_threshold.isna()].groupby(["product", "site_id"]).apply(summ).round(1)
S1.to_csv(f"{outdir}/validation_summary_by_product.csv")
S1c.to_csv(f"{outdir}/validation_summary_common_sample.csv")
S2.to_csv(f"{outdir}/validation_summary_by_site.csv")
print(S1.to_string()); print("\ncommon sample (NDSI>=10 where applicable)\n", S1c.to_string())
print(S2.loc[[f"MODIS_ds30_{kind}", "MOD10A1F", "Landsat+S2_30m"]].to_string())
