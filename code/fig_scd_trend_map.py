"""Per-cell Sen slope of snow-covered days (SCD) per water year, WY2001-2025, Dry Creek.
SCD from the daily MODIS-downscaled 30 m maps; missing or negative SCD -> NaN; needs >= 20 years.
Mann-Kendall p (normal approx., no tie correction) for hatching p < 0.1.
usage: python3 fig_scd_trend_map.py SCD_TIF GRID_NPZ OUT_PNG OUT_TIF"""
import sys
import numpy as np
import rasterio
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

scd_tif, grid_npz, out_png, out_tif = sys.argv[1:5]
with rasterio.open(scd_tif) as r:
    desc = list(r.descriptions); A = r.read().astype(float); prof = r.profile
G = np.load(grid_npz); region, elev, asp, slope = G["region"], G["elev"], G["aspect"], G["slope"]
yrs = np.array([int(d[2:]) for d in desc]); k = (yrs >= 2001) & (yrs <= 2025)
yrs, A = yrs[k], A[k]
A[A < 0] = np.nan
inside = region == 1
Y = A[:, inside]                                   # (nyr, ncell)
n = len(yrs); i, j = np.triu_indices(n, 1)
d = Y[j] - Y[i]; dt = (yrs[j] - yrs[i])[:, None]
sen = np.nanmedian(d / dt, 0) * 10                 # days per decade
S = np.nansum(np.sign(d), 0); nv = np.isfinite(Y).sum(0)
var = nv * (nv - 1) * (2 * nv + 5) / 18
z = np.where(S > 0, S - 1, np.where(S < 0, S + 1, 0)) / np.sqrt(var)
p = 2 * norm.sf(abs(z))
sen[nv < 20] = np.nan; p[nv < 20] = np.nan
SEN = np.full(region.shape, np.nan, np.float32); P = SEN.copy()
SEN[inside] = sen; P[inside] = p
prof.update(count=2, dtype="float32", nodata=np.nan)
with rasterio.open(out_tif, "w", **prof) as w:
    w.write(SEN, 1); w.write(P, 2); w.set_band_description(1, "sen_slope_scd_days_per_decade_wy2001_2025"); w.set_band_description(2, "mann_kendall_p")

NY, NX = region.shape; X0, Y1, RES = 564480, 4849500, 30
ext = [X0, X0 + NX * RES, Y1 - NY * RES, Y1]; cext = [ext[0], ext[1], ext[3], ext[2]]
dy, dx = np.gradient(elev, RES); sl = np.arctan(np.hypot(dx, dy)); a_ = np.arctan2(-dx, dy)
hs = np.sin(np.pi / 4) * np.cos(sl) + np.cos(np.pi / 4) * np.sin(sl) * np.cos(np.deg2rad(315) - a_)
rr, cc = np.nonzero(inside)
TXT2, BG = "#52514e", "#fcfcfb"
CM = LinearSegmentedColormap.from_list("trend", ["#5c4326", "#a8865a", "#e2d6bf", "#f4f4f2", "#bcd3e6", "#5f8fbf", "#1f4e85"])
plt.rcParams.update({"font.size": 9, "figure.facecolor": BG, "axes.facecolor": BG})
fig = plt.figure(figsize=(12, 5.6), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1])
ax = fig.add_subplot(gs[0])
vm = 12
ax.imshow(np.where(inside, hs, np.nan), cmap="gray", extent=ext, vmin=-0.2, vmax=1.1, interpolation="nearest")
im = ax.imshow(SEN, cmap=CM, norm=TwoSlopeNorm(0, -vm, vm), alpha=0.85, extent=ext, interpolation="nearest")
ax.contourf(np.where(inside, (P < 0.1).astype(float), np.nan), levels=[0.5, 1.5], colors="none", hatches=["...."], extent=cext)
ax.contour(np.where(inside, elev, np.nan), levels=[1400, 1600, 1800, 2000], colors="#33333380", linewidths=0.4, extent=cext)
ax.contour(inside, levels=[0.5], colors="k", linewidths=0.9, extent=cext)
ax.set_xlim(X0 + (cc.min() - 5) * RES, X0 + (cc.max() + 5) * RES); ax.set_ylim(Y1 - (rr.max() + 5) * RES, Y1 - (rr.min() - 5) * RES)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01, extend="both", label="Sen slope of snow-covered days (days per decade)")
ax.set_title("Trend in snow-covered days, WY2001-2025", loc="left", fontsize=11, fontweight="bold")
# by elevation and aspect
b = fig.add_subplot(gs[1])
N = ((asp >= 292.5) | (asp < 67.5)) & (slope >= 5); Sf = (asp >= 112.5) & (asp < 247.5) & (slope >= 5)
bins = np.arange(1000, 2201, 100)
for m, col, lab in [(N & inside, "#2a78d6", "N-facing (NW, N, NE)"), (Sf & inside, "#e8a33d", "S-facing (SE, S, SW)"), (inside, "#52514e", "all cells")]:
    med, q1, q3, xc = [], [], [], []
    for lo in bins[:-1]:
        v = SEN[m & (elev >= lo) & (elev < lo + 100)]; v = v[np.isfinite(v)]
        if len(v) >= 30:
            med.append(np.median(v)); q1.append(np.percentile(v, 25)); q3.append(np.percentile(v, 75)); xc.append(lo + 50)
    b.plot(med, xc, color=col, lw=2 if lab != "all cells" else 1.2, ls="-" if lab != "all cells" else "--", label=lab)
    if lab != "all cells":
        b.fill_betweenx(xc, q1, q3, color=col, alpha=0.15, lw=0)
b.axvline(0, color="#999", lw=0.8)
b.set_xlabel("Sen slope (days per decade)"); b.set_ylabel("elevation (m)")
b.set_title("By elevation (median, IQR shaded)", loc="left", fontsize=10)
b.legend(frameon=False, fontsize=8); b.spines[["top", "right"]].set_visible(False)
fig.text(0.01, -0.02, f"Daily MODIS downscaled to 30 m; dots = Mann-Kendall p < 0.1 ({np.nanmean(P[inside] < 0.1) * 100:.0f}% of cells; ~10% expected by chance). "
         "Within each 500 m MODIS pixel the 30 m pattern comes from a fixed persistence ranking, so fine-scale contrasts are partly structural.",
         fontsize=7, color=TXT2)
fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.15)
v = SEN[inside]; print("median", np.nanmedian(v), "pct neg", np.nanmean(v < 0), "p<0.1", np.nanmean(P[inside] < 0.1), "p<.1 neg", np.nanmean((P[inside] < 0.1) & (v < 0)), "range", np.nanpercentile(v, [5, 95]))
