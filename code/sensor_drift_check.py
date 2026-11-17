"""Do Terra, Aqua and VIIRS agree on trends in 'any snow' and 'full snow' days over upper Dry Creek?
Native pixels whose centre lies in Dry Creek above 1,700 m (and 1,400-1,700 m), Dec-Apr, NDSI >= 10 (any) / fSCA >= 0.9 (full).
usage: python3 sensor_drift_check.py DATA_DIR GRID_NPZ OUT_CSV"""
import sys, glob, datetime as dt
import numpy as np, pandas as pd
from scipy.stats import theilslopes, kendalltau
ddir, grid_npz, out = sys.argv[1:4]
G = np.load(grid_npz)
reg, elev = G["region"], G["elev"]
rows = []
for prod, rk, ck in [("MOD10A1F", "modis_row", "modis_col"), ("MYD10A1F", "modis_row", "modis_col"), ("VNP10A1F", "viirs_row", "viirs_col")]:
    pr, pc = G[rk], G[ck]
    m = (reg == 1) & np.isfinite(elev)
    df = pd.DataFrame(dict(r=pr[m], c=pc[m], z=elev[m])).groupby(["r", "c"]).agg(z=("z", "mean"), n=("z", "size")).reset_index()
    df = df[df.n >= 100]  # pixel mostly inside Dry Creek
    groups = {"upper >=1700": df[df.z >= 1700], "mid 1400-1700": df[(df.z >= 1400) & (df.z < 1700)]}
    for f in sorted(glob.glob(f"{ddir}/{prod}/A*.npz")):
        y, j = int(f[-11:-7]), int(f[-7:-4]); D = dt.date(y, 1, 1) + dt.timedelta(j - 1)
        if D.month not in (12, 1, 2, 3, 4):
            continue
        wy = D.year + (D.month >= 10)
        z = np.load(f); n = z["CGF_NDSI_Snow_Cover"].astype(float); cp = z["Cloud_Persistence"]
        for g, p in groups.items():
            v = n[p.r.values, p.c.values]; ok = (v <= 100) & (cp[p.r.values, p.c.values] <= 10)
            fs = np.clip(-0.01 + 1.45 * v / 100, 0, 1)
            rows.append((prod, g, wy, np.mean(ok & (v >= 10)), np.mean(ok & (fs >= 0.9)), ok.mean()))
D = pd.DataFrame(rows, columns=["product", "group", "wy", "any", "full", "valid"])
A = D.groupby(["product", "group", "wy"]).sum().reset_index()
A.to_csv(out, index=False)
res = []
for (p, g), x in A.groupby(["product", "group"]):
    for y0, y1 in [(2003, 2025), (2013, 2025)]:
        s = x[x.wy.between(y0, y1)]
        if s.wy.nunique() < y1 - y0 - 1:
            continue
        for c in ["any", "full"]:
            res.append(dict(product=p, group=g, period=f"{y0}-{y1}", metric=c, mean_days=s[c].mean().round(1),
                            sen_per_decade=round(theilslopes(s[c], s.wy)[0] * 10, 1), p=round(kendalltau(s.wy, s[c])[1], 2)))
R = pd.DataFrame(res)
print(R.pivot_table(index=["group", "period", "metric"], columns="product", values="sen_per_decade").to_string())
print(R.pivot_table(index=["group", "period", "metric"], columns="product", values="p").to_string())
R.to_csv(out.replace(".csv", "_trends.csv"), index=False)
