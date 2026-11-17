"""Bivariate map: snow probability (mean share of days snow-covered) x year-to-year SD, near Jan 1, Feb 1, Mar 1, Apr 1.
Reads the grids written by fig_snow_prob_maps.py. Classes: probability <0.25 / 0.25-0.75 / >0.75; SD <0.2 / 0.2-0.35 / >0.35.
usage: SNOW_DOMAIN=dcew|rcew python3 fig_snow_prob_bivariate.py PROB_NPZ GRID_NPZ OUT_PNG"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain import X0, Y1, NAME

prob_npz, grid_npz, out_png = sys.argv[1:4]
P = np.load(prob_npz)
G = np.load(grid_npz); region, elev = G["region"], G["elev"]
NY, NX = region.shape
RES = 30
inside = region == 1
TARGETS = ["Jan 1", "Feb 1", "Mar 1", "Apr 1"]
PB, SB = [0.25, 0.75], [0.2, 0.35]
GRAY = "#bdbdbd"
# rows = SD class (low, mid, high), cols = probability class (bare, mixed, snow)
COLORS = [["#6b4a2b", GRAY, "#ffffff"],
          ["#9c7d86", GRAY, "#d6cdea"],
          ["#5e3c6e", "#7f5fb0", "#b3a6db"]]
cmap = ListedColormap([c for row in COLORS for c in row])

BASIN = {"dcew": "Dry Creek Experimental Watershed", "rcew": "Reynolds Creek Experimental Watershed"}[NAME]
LEVELS = [1400, 1600, 1800, 2000] if NAME == "dcew" else [1200, 1400, 1600, 1800, 2000, 2200]
ext = [X0, X0 + NX * RES, Y1 - NY * RES, Y1]
dy, dx = np.gradient(elev, RES)
sl = np.arctan(np.hypot(dx, dy)); asp = np.arctan2(-dx, dy)
az, alt = np.deg2rad(315), np.deg2rad(45)
hs = np.sin(alt) * np.cos(sl) + np.cos(alt) * np.sin(sl) * np.cos(az - asp)
rr, cc = np.nonzero(inside)

TXT2, BG = "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "figure.facecolor": BG, "axes.facecolor": BG})
if NAME == "dcew":
    fig = plt.figure(figsize=(14, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.62])
    axs = [fig.add_subplot(gs[0, j]) for j in range(4)]; lax = fig.add_subplot(gs[0, 4])
else:
    fig = plt.figure(figsize=(12, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.9])
    axs = [fig.add_subplot(gs[0, j]) for j in range(4)]; lax = fig.add_subplot(gs[0, 4])

shares = {}
for ax, name in zip(axs, TARGETS):
    k = name.replace(" ", "")
    p, s = P[f"mean_{k}"], P[f"sd_{k}"]
    ok = inside & np.isfinite(p) & np.isfinite(s)
    cls = np.where(ok, np.digitize(s, SB) * 3 + np.digitize(p, PB), -1).astype(float)
    cls[~ok] = np.nan
    shares[name] = [np.mean(cls[ok] == i) for i in range(9)]
    ax.imshow(np.where(inside, hs, np.nan), cmap="gray", extent=ext, vmin=-0.2, vmax=1.1, interpolation="nearest")
    ax.imshow(cls, cmap=cmap, vmin=-0.5, vmax=8.5, alpha=0.9, extent=ext, interpolation="nearest")
    ax.contour(np.where(inside, elev, np.nan), levels=LEVELS, colors="#33333380", linewidths=0.4,
               extent=[ext[0], ext[1], ext[3], ext[2]])
    ax.contour(inside, levels=[0.5], colors="k", linewidths=0.9, extent=[ext[0], ext[1], ext[3], ext[2]])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_xlim(X0 + (cc.min() - 5) * RES, X0 + (cc.max() + 5) * RES)
    ax.set_ylim(Y1 - (rr.max() + 5) * RES, Y1 - (rr.min() - 5) * RES)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(name, fontsize=12, fontweight="bold")

# 3x3 legend
lax.set_xlim(0, 3); lax.set_ylim(0, 3); lax.set_aspect("equal")
for r in range(3):
    for c in range(3):
        lax.add_patch(plt.Rectangle((c, r), 1, 1, fc=COLORS[r][c], ec="#888888", lw=0.6))
lax.set_xticks([0.5, 1.5, 2.5], ["< 0.25\nmostly\nbare", "0.25-0.75\nmixed", "> 0.75\nmostly\nsnow"], fontsize=8)
lax.set_yticks([0.5, 1.5, 2.5], ["< 0.2\nsteady", "0.2-0.35", "> 0.35\nswings\nyear to year"], fontsize=8)
lax.set_xlabel("Snow probability", fontsize=9)
lax.set_ylabel("Year-to-year SD", fontsize=9)
lax.tick_params(length=0)
for sp in lax.spines.values():
    sp.set_visible(False)
lax.text(0.5, 0.5, "reliably\nbare", ha="center", va="center", fontsize=7, color="white")
lax.text(2.5, 0.5, "reliably\nsnow", ha="center", va="center", fontsize=7, color="#1f2a33")
lax.text(1.5, 1.0, "not\nobserved", ha="center", va="center", fontsize=6.5, color="#444444")

fig.suptitle(f"{BASIN}: where snow cover is reliable, WY2001-2025 (daily MODIS downscaled to 30 m, ±3-day window)",
             fontsize=11, x=0.01, ha="left")
fig.text(0.01, -0.02, "Gray legend cells: a mixed probability cannot have a low SD, because each year a cell is "
         "nearly all snow or all bare in a week. Contours every 200 m. Under forest the downscaling relies on a terrain model.",
         fontsize=7, color=TXT2)
fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.15)
lab = ["bare/steady", "-", "snow/steady", "bare/mid", "-", "snow/mid", "bare/high", "mixed/high", "snow/high"]
for name, v in shares.items():
    print(name, {l: round(100 * x) for l, x in zip(lab, v) if l != "-"})
