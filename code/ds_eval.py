"""Calibrate w and score block vs smooth downscaling against Landsat/S2 30 m snow maps.

usage: python3 ds_eval.py TERRAIN_PKL HIRES_DIR MOD_DIR OUTDIR
Scenes used: Dec-Jul, >= 80% of the domain clear, domain snow fraction 0.05-0.95.
ideal : fSCA per MODIS cell taken from the hi-res map itself (cells >= 80% clear)  -> isolates the
        terrain-score skill (Walters et al. 'ideal simulation')
modis : same-day MOD10A1F CGF NDSI where Cloud_Persistence == 0 (a clear-sky MODIS view that day)
Metrics on domain cells (Dry Creek + Bogus) that are clear in the hi-res map and lie in MODIS
pixels whose hi-res fSCA is 0.1-0.9: F-measure, plus F from random placement with the same
per-MODIS-cell fSCA (mean of 20 draws).
"""
import os, sys, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downscale import Terrain, ndsi_to_fsca

T = pickle.load(open(sys.argv[1], "rb"))
hdir, mdir, outdir = sys.argv[2:5]
os.makedirs(outdir, exist_ok=True)
M = pd.read_csv(f"{hdir}/hires_scenes.csv", parse_dates=["date"])
S = np.load(f"{hdir}/hires_snowmaps.npy", mmap_mode="r")
dom = (T.region > 0) & T.valid
cid = T.cell_id
ucid, inv = np.unique(cid, return_inverse=True)
inv = inv.reshape(cid.shape)
rng = np.random.default_rng(0)


def fmeas(pred, obs, mask):
    p, o = pred[mask], obs[mask]
    tp = np.sum(p & o); fp = np.sum(p & ~o); fn = np.sum(~p & o)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def cell_fsca(obs, clear):
    n = np.bincount(inv[clear], minlength=len(ucid)).astype(float)
    s = np.bincount(inv[clear & obs], minlength=len(ucid)).astype(float)
    tot = np.bincount(inv[T.valid], minlength=len(ucid)).astype(float)
    f = np.where((n >= 0.8 * tot) & (tot > 0), s / np.maximum(n, 1), np.nan)
    from domain import D as _DOM
    NR, NC = _DOM["MODIS_SHAPE"]
    grid = np.full((NR, NC), np.nan)
    rr, cc = ucid // 1000, ucid % 1000
    ok = (rr >= 0) & (rr < NR) & (cc >= 0) & (cc < NC)
    grid[rr[ok], cc[ok]] = f[ok]
    return grid, f[inv]


def random_F(fs_cell, obs, mask, n=20):
    Fs = []
    for _ in range(n):
        u = rng.random(cid.shape)
        # random rank within MODIS cell: uniform draw vs fSCA
        Fs.append(fmeas(u <= fs_cell, obs, mask))
    return float(np.mean(Fs))


sel = M[(M.date.dt.month.isin([12, 1, 2, 3, 4, 5, 6, 7])) & (M.clear_frac >= 0.8)].copy()
ws = [float(x) for x in os.environ.get("DS_WS", "0,0.25,0.5,0.7,0.8,0.9,0.95,1").split(",")]
forest = np.load(f"{hdir}/hires_forest.npy")
KINDS = os.environ.get("DS_KINDS", "block,smooth").split(",")
if any(k.startswith("emp") for k in KINDS):
    PP = np.load(os.environ["DS_PERSIST"])
    for f_ in ("odd", "even"):
        T.set_persistence(PP["P_" + f_], f_)
rows = []
cache = {}
for i, s in sel.iterrows():
    m = np.asarray(S[i]); clear = (m != 255) & T.valid; obs = m == 1
    dsf = obs[dom & clear].mean()
    if not (0.05 <= dsf <= 0.95):
        continue
    doy = s.date.dayofyear
    fgrid, fcell30 = cell_fsca(obs, clear)
    partial = dom & clear & (fcell30 >= 0.1) & (fcell30 <= 0.9)
    if partial.sum() < 500:
        continue
    # MODIS same-day
    mf = f"{mdir}/A{s.date.year}{doy:03d}.npz"
    mod = None
    if os.path.exists(mf):
        d = np.load(mf)
        nd = d["CGF_NDSI_Snow_Cover"].astype(float); cp = d["Cloud_Persistence"]
        mod = np.where((nd <= 100) & (cp == 0), ndsi_to_fsca(nd), np.nan)
    base = dict(month=s.date.month, date=s.date, source=s.source, domain_sca=dsf, n_partial=int(partial.sum()))
    rF = random_F(fcell30, obs, partial)
    wyv = s.date.year + (s.date.month >= 10)
    T.emp_fold = "even" if wyv % 2 == 1 else "odd"  # persistence from the other half of the years
    for kind in KINDS:
        wlist = [0.0] if kind.startswith("emp") else ws
        fs_ideal = T.fsca_field(fgrid, kind)
        fs_mod = T.fsca_field(mod, kind) if mod is not None else None
        for w in wlist:
            key = (kind, w, T.emp_fold if kind.startswith("emp") else int(np.argmin(np.abs(T.doys - doy))))
            if key not in cache:
                cache[key] = T.prank(doy, w, kind)
            pr = cache[key]
            pred, _ = T.snow(pr, fs_ideal)
            r = dict(base, kind=kind, w=w, input="ideal", F=fmeas(pred, obs, partial), F_random=rF,
                     F_edge=fmeas(pred, obs, partial & (T.edge_dist_m < 60)),
                     F_interior=fmeas(pred, obs, partial & (T.edge_dist_m >= 120)),
                     F_open=fmeas(pred, obs, partial & ~forest), F_forest=fmeas(pred, obs, partial & forest),
                     F_random_open=random_F(fcell30, obs, partial & ~forest, 5))
            rows.append(r)
            if fs_mod is not None:
                pm, okm = T.snow(pr, fs_mod)
                mk = partial & okm
                if mk.sum() > 300:
                    rows.append(dict(base, kind=kind, w=w, input="modis", F=fmeas(pm, obs, mk), F_random=rF,
                                     F_edge=fmeas(pm, obs, mk & (T.edge_dist_m < 60)),
                                     F_interior=fmeas(pm, obs, mk & (T.edge_dist_m >= 120)),
                                     F_open=fmeas(pm, obs, mk & ~forest), F_forest=fmeas(pm, obs, mk & forest),
                                     acc=float((pm == obs)[mk].mean()),
                                     sca_bias=float(pm[mk].mean() - obs[mk].mean())))
    print(s.date.date(), s.source, round(dsf, 2), flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{outdir}/downscale_scores_by_scene.csv", index=False)
summ = R.groupby(["input", "kind", "w"])[["F", "F_random", "F_edge", "F_interior", "F_open", "F_forest", "F_random_open", "acc", "sca_bias"]].mean().round(3)
summ["n_scenes"] = R.groupby(["input", "kind", "w"]).size()
summ.to_csv(f"{outdir}/downscale_scores_summary.csv")
print(summ.to_string())
