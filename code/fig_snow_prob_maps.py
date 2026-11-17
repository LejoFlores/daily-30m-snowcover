"""Expected snow cover and its year-to-year variability on Jan 1, Feb 1, Mar 1, Apr 1 (WY2001-2025).
Per 30 m cell and year: share of valid days snow-covered in a 7-day window centred on the date
(daily MODIS downscaled to 30 m). Map 1 = mean across years (snow probability); map 2 = SD across years.
usage: python3 fig_snow_prob_maps.py DS_DIR GRID_NPZ OUT_PNG OUT_NPZ"""
import glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

dsdir, grid_npz, out_png, out_npz = sys.argv[1:5]
G = np.load(grid_npz); region, elev = G["region"], G["elev"]
TARGETS = [("Jan 1", 1, 1), ("Feb 1", 2, 1), ("Mar 1", 3, 1), ("Apr 1", 4, 1)]
NY, NX = region.shape
stack = {t[0]: [] for t in TARGETS}
for f in sorted(glob.glob(f"{dsdir}/ds30_emp_smooth_daily_WY*.npz")):
    wy = int(f[-8:-4])
    if not 2001 <= wy <= 2025:
        continue
    d = np.load(f); nc = int(d["ncell"]); iy, ix = d["iy"], d["ix"]
    sn = np.unpackbits(d["snow"], axis=1)[:, :nc].astype(bool)
    va = np.unpackbits(d["valid"], axis=1)[:, :nc].astype(bool)
    dates = pd.to_datetime(d["dates"])
    for name, m, dd in TARGETS:
        t = pd.Timestamp(wy - (m >= 10), m, dd)
        k = np.flatnonzero(abs((dates - t).days) <= 3)
        nv = va[k].sum(0); ns = (sn[k] & va[k]).sum(0)
        g = np.full((NY, NX), np.nan, np.float32)
        g[iy, ix] = np.where(nv >= 3, ns / np.maximum(nv, 1), np.nan)
        stack[name].append(g)
mean = {k: np.nanmean(np.array(v), 0) for k, v in stack.items()}
sd = {k: np.nanstd(np.array(v), 0, ddof=1) for k, v in stack.items()}
np.savez_compressed(out_npz, **{f"mean_{k.replace(' ', '')}": v for k, v in mean.items()},
                    **{f"sd_{k.replace(' ', '')}": v for k, v in sd.items()}, years=np.arange(2001, 2026))

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from domain import X0, Y1, NAME
RES = 30
BASIN = {"dcew": "Dry Creek Experimental Watershed", "rcew": "Reynolds Creek Experimental Watershed"}[NAME]
LEVELS = [1400, 1600, 1800, 2000] if NAME == "dcew" else [1200, 1400, 1600, 1800, 2000, 2200]
ext = [X0, X0 + NX * RES, Y1 - NY * RES, Y1]
# hillshade
dy, dx = np.gradient(elev, RES)
sl = np.arctan(np.hypot(dx, dy)); asp = np.arctan2(-dx, dy)
az, alt = np.deg2rad(315), np.deg2rad(45)
hs = np.sin(alt) * np.cos(sl) + np.cos(alt) * np.sin(sl) * np.cos(az - asp)
inside = region == 1
from matplotlib.colors import LinearSegmentedColormap
SNOW = LinearSegmentedColormap.from_list("snow", ["#5c4326", "#9c7a4e", "#cdbb98", "#e6e9ea", "#ffffff"])
VAR = plt.get_cmap("viridis")  # low (dark purple) to high (yellow) variability; no white, to avoid reading as snow
rr, cc = np.nonzero(inside)
xs, ys = Transformer.from_crs(4326, 32611, always_xy=True).transform(-116.09685, 43.76377)
TXT2, BG = "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "figure.facecolor": BG, "axes.facecolor": BG})
fig, axs = plt.subplots(2, 4, figsize=(13, 6.6) if NAME == "dcew" else (11, 9.5), constrained_layout=True)
for j, (name, _, _) in enumerate(TARGETS):
    for i, (fld, cmap, vmax, lab) in enumerate([(mean, SNOW, 1, "snow probability (mean share of days snow-covered)"),
                                                 (sd, VAR, 0.5, "year-to-year SD")]):
        ax = axs[i, j]
        ax.imshow(np.where(inside, hs, np.nan), cmap="gray", extent=ext, vmin=-0.2, vmax=1.1, interpolation="nearest")
        im = ax.imshow(np.where(inside, fld[name], np.nan), cmap=cmap, vmin=0, vmax=vmax, alpha=0.88, extent=ext, interpolation="nearest")
        ax.contour(np.where(inside, elev, np.nan), levels=LEVELS, colors="#33333380", linewidths=0.4,
                   extent=[ext[0], ext[1], ext[3], ext[2]])
        ax.contour(region == 1, levels=[0.5], colors="k", linewidths=0.9, extent=[ext[0], ext[1], ext[3], ext[2]])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_xlim(X0 + (cc.min() - 5) * RES, X0 + (cc.max() + 5) * RES); ax.set_ylim(Y1 - (rr.max() + 5) * RES, Y1 - (rr.min() - 5) * RES)
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_title(name, fontsize=12, fontweight="bold")
for i, (fld, cmap, vmax, lab) in enumerate([(mean, SNOW, 1, "Snow probability\n(mean share of days snow-covered)"),
                                             (sd, VAR, 0.5, "Year-to-year SD\nof that share")]):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axs[i, :], shrink=0.85, pad=0.01, label=lab)
fig.suptitle(f"{BASIN}: expected snow cover and its variability, WY2001-2025 (daily MODIS downscaled to 30 m, ±3-day window)",
             fontsize=11, x=0.01, ha="left")
fig.text(0.01, 0.0, f"{BASIN}; contours every 200 m. "
         "Under forest the downscaling relies on a terrain model (README).", fontsize=7, color=TXT2)
fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.15)
for k in mean:
    print(k, "DC mean", np.nanmean(mean[k][region == 1]).round(2), "Bogus", np.nanmean(mean[k][region == 2]).round(2),
          "DC sd", np.nanmean(sd[k][region == 1]).round(2))
