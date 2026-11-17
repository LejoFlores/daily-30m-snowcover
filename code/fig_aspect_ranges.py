"""Snow-cover range by elevation band, N- vs S-facing, Dry Creek and Bogus Basin.
N = NW, N, NE (aspect 292.5-67.5 deg); S = SE, S, SW (112.5-247.5); slope >= 5 deg.
Per band x aspect x WY (2001-2025), from the daily 30 m downscaled MODIS maps: band is snow when
>= 50% of its cells are; snow_on/off from ondates.season. Bar = median snow-on to median snow-off;
whiskers = +/- 1 SD of snow-on (left) and of snow-off (right) across years.
usage: python3 fig_aspect_ranges.py DS_DIR GRID_NPZ OUT_PNG OUT_CSV"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ondates import by_wy

dsdir, grid_npz, out_png, out_csv = sys.argv[1:5]
G = np.load(grid_npz)
asp, slope, elev, region = G["aspect"], G["slope"], G["elev"], G["region"]
aclass = np.where(slope < 5, "", np.where((asp >= 292.5) | (asp < 67.5), "N",
                  np.where((asp >= 112.5) & (asp < 247.5), "S", "")))
band = np.floor(elev / 200) * 200
rows = []
for f in sorted(glob.glob(f"{dsdir}/ds30_emp_smooth_daily_WY*.npz")):
    wy = int(f[-8:-4])
    if not 2001 <= wy <= 2025:
        continue
    d = np.load(f); nc = int(d["ncell"]); iy, ix = d["iy"], d["ix"]
    sn = np.unpackbits(d["snow"], axis=1)[:, :nc].astype(bool)
    va = np.unpackbits(d["valid"], axis=1)[:, :nc].astype(bool)
    dates = pd.to_datetime(d["dates"])
    reg, ac, bd = region[iy, ix], aclass[iy, ix], band[iy, ix]
    for r_, rname in [(1, "Dry Creek"), (2, "Bogus Basin")]:
        for b in np.unique(bd[reg == r_]):
            for a in ("N", "S"):
                idx = np.flatnonzero((reg == r_) & (bd == b) & (ac == a))
                if len(idx) < 30:
                    continue
                nv = va[:, idx].sum(1); ns = (sn[:, idx] & va[:, idx]).sum(1)
                frac = np.where(nv >= 0.5 * len(idx), ns / np.maximum(nv, 1), np.nan)
                o = by_wy(dates, np.where(np.isfinite(frac), (frac >= 0.5).astype(float), np.nan)).iloc[0]
                rows.append(dict(region=rname, band=int(b), aspect=a, wy=wy, n_cells=len(idx),
                                 snow_on=o.snow_on, snow_off=o.snow_off, scd=o.scd))
D = pd.DataFrame(rows)
S = D.groupby(["region", "band", "aspect"]).agg(
    n_cells=("n_cells", "first"), n_years=("snow_off", lambda x: int(x.notna().sum())),
    on_med=("snow_on", "median"), on_sd=("snow_on", "std"),
    off_med=("snow_off", "median"), off_sd=("snow_off", "std"), scd_med=("scd", "median")).reset_index()
S = S[S.n_years >= 8]
S.round(1).to_csv(out_csv, index=False)

TXT, TXT2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"Dry Creek": "#2a78d6", "Bogus Basin": "#eb6834"}
LIGHT = {"Dry Creek": "#b9d4f4", "Bogus Basin": "#f6c3ad"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG})
regs = ["Bogus Basin", "Dry Creek"]
nb = [S[S.region == r].band.nunique() for r in regs]
fig, axs = plt.subplots(2, 1, figsize=(9, 1.1 + 0.62 * sum(nb)), sharex=True, gridspec_kw=dict(height_ratios=nb, hspace=0.12))
H = 0.34
for ax, r in zip(axs, regs):
    s = S[S.region == r]
    bands = sorted(s.band.unique())
    for i, b in enumerate(bands):
        for a, dy in (("N", H / 2 + 0.02), ("S", -H / 2 - 0.02)):
            x = s[(s.band == b) & (s.aspect == a)]
            if x.empty:
                continue
            x = x.iloc[0]; y = i + dy
            fc = COL[r] if a == "N" else LIGHT[r]
            ax.barh(y, x.off_med - x.on_med, left=x.on_med, height=H, color=fc, edgecolor=COL[r], lw=1)
            ax.errorbar([x.on_med, x.off_med], [y, y], xerr=[x.on_sd, x.off_sd], fmt="none", ecolor=TXT, elinewidth=1, capsize=3)
            ax.text(x.off_med + x.off_sd + 3, y, f"{a} {x.scd_med:.0f} d", va="center", fontsize=7, color=TXT2)
    ax.set_yticks(range(len(bands))); ax.set_yticklabels([f"{b}-{b + 200} m" for b in bands])
    ax.set_ylim(-0.6, len(bands) - 0.4)
    ax.set_title(r if r == "Dry Creek" else "Bogus Basin (box north of Dry Creek, incl. ski area and SNOTEL)", loc="left", fontsize=10, color=COL[r], fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ticks = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274]
axs[-1].set_xticks(ticks)
axs[-1].set_xticklabels(["Oct 1", "Nov 1", "Dec 1", "Jan 1", "Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1", "Jul 1"])
axs[-1].set_xlim(20, 275)
axs[-1].set_xlabel("date in water year")
leg = [Patch(fc=COL["Dry Creek"], ec=COL["Dry Creek"], label="N-facing (NW, N, NE)"),
       Patch(fc=LIGHT["Dry Creek"], ec=COL["Dry Creek"], label="S-facing (SE, S, SW)"),
       Line2D([], [], color=TXT, lw=1, marker="|", ms=6, label="±1 SD of snow-on (left) / snow-off (right)")]
axs[0].legend(handles=leg, frameon=False, fontsize=7.5, loc="lower right", bbox_to_anchor=(1, 1.08), ncol=3)
fig.text(0.01, 0.003, "Bars span median snow-on to median snow-off, WY2001-2025, daily MODIS downscaled to 30 m; label = median snow days. "
         "Season = longest run >= 14 d with >= 50% of the band's cells snow-covered.", fontsize=6.5, color=TXT2)
fig.savefig(out_png, dpi=170, bbox_inches="tight")
print(S.round(0).to_string())
