"""N- vs S-facing snow-cover ranges by 200 m band for Dry Creek and Reynolds Creek (downscaled MODIS, WY2001-2025).
N = NW/N/NE (aspect >= 292.5 or < 67.5), S = SE/S/SW (112.5-247.5), slope >= 5; bands need >= 30 cells and >= 8 seasons.
Bar = median snow-on to median snow-off; whiskers = +/- 1 SD of each.
usage: python3 fig_ns_two_basins.py OUT_PNG OUT_CSV  (paths below are scratchpad-relative)"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ondates import by_wy
out_png, out_csv = sys.argv[1:3]
CASES = [("Dry Creek", "out/ds30h_c2", "data/grid.npz"), ("Reynolds Creek", "out/rcew/ds30h", "data/rcew/grid.npz")]
rows = []
for name, dsdir, gnpz in CASES:
    G = np.load(gnpz); asp, slope, elev, region = G["aspect"], G["slope"], G["elev"], G["region"]
    acl = np.where(slope < 5, "", np.where((asp >= 292.5) | (asp < 67.5), "N", np.where((asp >= 112.5) & (asp < 247.5), "S", "")))
    band = np.floor(elev / 200) * 200
    for f in sorted(glob.glob(f"{dsdir}/ds30_emp_smooth_daily_WY*.npz")):
        wy = int(f[-8:-4])
        if not 2001 <= wy <= 2025:
            continue
        d = np.load(f); nc = int(d["ncell"]); iy, ix = d["iy"], d["ix"]
        sn = np.unpackbits(d["snow"], axis=1)[:, :nc].astype(bool); va = np.unpackbits(d["valid"], axis=1)[:, :nc].astype(bool)
        dates = pd.to_datetime(d["dates"]); reg, ac, bd = region[iy, ix], acl[iy, ix], band[iy, ix]
        for b in np.unique(bd[reg == 1]):
            for a in ("N", "S"):
                idx = np.flatnonzero((reg == 1) & (bd == b) & (ac == a))
                if len(idx) < 30:
                    continue
                nv = va[:, idx].sum(1); ns = (sn[:, idx] & va[:, idx]).sum(1)
                frac = np.where(nv >= 0.5 * len(idx), ns / np.maximum(nv, 1), np.nan)
                o = by_wy(dates, np.where(np.isfinite(frac), (frac >= 0.5).astype(float), np.nan)).iloc[0]
                rows.append(dict(basin=name, band=int(b), aspect=a, wy=wy, n_cells=len(idx), snow_on=o.snow_on, snow_off=o.snow_off, scd=o.scd))
D = pd.DataFrame(rows)
S = D.groupby(["basin", "band", "aspect"]).agg(n_cells=("n_cells", "first"), n_years=("snow_off", lambda x: int(x.notna().sum())),
    on_med=("snow_on", "median"), on_sd=("snow_on", "std"), off_med=("snow_off", "median"), off_sd=("snow_off", "std"),
    scd_med=("scd", "median")).reset_index()
HAS = set(zip(S.basin, S.band, S.aspect))
S = S[S.n_years >= 8]; S.round(1).to_csv(out_csv, index=False)
TXT, TXT2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"N": "#2a78d6", "S": "#e8a33d"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG})
nb = [S[S.basin == n].band.nunique() for n, *_ in CASES]
fig, axs = plt.subplots(len(CASES), 1, figsize=(9, 1.2 + 0.6 * sum(nb)), sharex=True, gridspec_kw=dict(height_ratios=nb, hspace=0.18))
H = 0.36
for ax, (name, *_) in zip(axs, CASES):
    s = S[S.basin == name]; bands = sorted(s.band.unique())
    for i, b in enumerate(bands):
        for a, dy in (("N", H / 2 + 0.02), ("S", -H / 2 - 0.02)):
            x = s[(s.band == b) & (s.aspect == a)]
            y = i + dy
            if x.empty:
                ax.barh(y, 289, left=1, height=H, color="#d9d8d4", edgecolor="none")
                msg = (f"{a}-facing: lasting snow season in fewer than 8 of 25 years" if (name, b, a) in HAS
                       else f"too few {a}-facing cells in this band")
                ax.text(145, y, msg, ha="center", va="center", fontsize=7.5, color=TXT2)
                continue
            x = x.iloc[0]
            ax.barh(y, x.off_med - x.on_med, left=x.on_med, height=H, color=C[a])
            ax.errorbar([x.on_med, x.off_med], [y, y], xerr=[x.on_sd, x.off_sd], fmt="none", ecolor=TXT, elinewidth=1, capsize=3)
            ax.text(x.off_med + x.off_sd + 3, y, f"{x.scd_med:.0f} d", va="center", fontsize=7, color=TXT2)
    ax.set_yticks(range(len(bands))); ax.set_yticklabels([f"{b}-{b + 200} m" for b in bands]); ax.set_ylim(-0.6, len(bands) - 0.4)
    ax.set_title(name, loc="left", fontsize=10, fontweight="bold"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].legend(handles=[Patch(color=C["N"], label="N-facing (NW, N, NE)"), Patch(color=C["S"], label="S-facing (SE, S, SW)")],
              frameon=False, fontsize=8, loc="lower right")
ticks = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274]
axs[-1].set_xticks(ticks); axs[-1].set_xticklabels(["Oct 1", "Nov 1", "Dec 1", "Jan 1", "Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1", "Jul 1"])
axs[-1].set_xlim(1, 290); axs[-1].set_xlabel("date in water year")
fig.text(0.01, 0.0, "Bars span median snow-on to median snow-off, WY2001-2025 (daily MODIS downscaled to 30 m, Landsat C2 + Sentinel-2 ranking); "
         "whiskers = ±1 SD; label = median snow days.", fontsize=6.5, color=TXT2)
fig.savefig(out_png, dpi=170, bbox_inches="tight")
print(S.round(0).to_string())
