"""Empirical snow persistence (share of clear Feb-Jun 30 m observations that are snow) from the
Landsat/S2 maps, by year fold (odd/even WY, for out-of-sample scoring) and all years; plus a
'hybrid' version where forest cells (snow under canopy is invisible) take the value predicted
from elevation and spring insolation, fitted on open cells.
usage: python3 build_persistence.py HIRES_DIR TERRAIN_PKL OUT_NPZ OUT_HYBRID_NPZ"""
import os, sys, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downscale import Terrain
hdir, tpkl, out1, out2 = sys.argv[1:5]
M = pd.read_csv(f"{hdir}/hires_scenes.csv", parse_dates=["date"]); S = np.load(f"{hdir}/hires_snowmaps.npy", mmap_mode="r")
forest = np.load(f"{hdir}/hires_forest.npy")
T = pickle.load(open(tpkl, "rb"))
wy = M.date.dt.year + (M.date.dt.month >= 10)
sel = M.date.dt.month.isin([2, 3, 4, 5, 6]).values
out, hyb = {}, {}
k = (T.doys >= 32) & (T.doys <= 152); ins = T.fsl[k].mean(0); z = (T.z - 1600) / 400.0
X = np.stack([np.ones_like(z), z, z ** 2, ins, ins * z, ins ** 2], -1)
for name, fold in [("odd", wy % 2 == 1), ("even", wy % 2 == 0), ("all", wy > 0)]:
    n = np.zeros(S.shape[1:]); s = np.zeros(S.shape[1:])
    for i in np.flatnonzero(sel & fold.values):
        m = np.asarray(S[i]); n += m != 255; s += m == 1
    P = np.where(n >= 10, s / np.maximum(n, 1), np.nan)
    out["P_" + name], out["n_" + name] = P, n
    m = T.valid & ~forest & np.isfinite(P) & (T.region > 0)
    b, *_ = np.linalg.lstsq(X[m], P[m], rcond=None); ph = np.clip(X @ b, 0, 1)
    hyb["P_" + name] = np.where(forest | ~np.isfinite(P), ph, P).astype(np.float32)
    print(name, int((sel & fold.values).sum()), "scenes; open-cell fit r=%.2f" % np.corrcoef(ph[m], P[m])[0, 1])
np.savez_compressed(out1, **out); np.savez_compressed(out2, **hyb)
