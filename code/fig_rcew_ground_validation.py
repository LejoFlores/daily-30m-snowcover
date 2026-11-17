"""Reynolds Creek: 30 m downscaled melt-out vs ARS snow-depth melt-out (depth > 5 cm), per site-year.
usage: python3 fig_rcew_ground_validation.py PAIRS_CSV OUT_PNG"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pairs, out_png = sys.argv[1:3]
P = pd.read_csv(pairs)
ok = P[(P.cov_nov_may >= 0.8) & (P.censored == 0)].dropna(subset=["ground_off", "sat_off"]).copy()
ok["single"] = ((ok.ground_scd - (ok.ground_off - ok.ground_on)) <= 14) & ((ok.sat_scd - (ok.sat_off - ok.sat_on)) <= 14)
BG, TXT2 = "#fcfcfb", "#52514e"
plt.rcParams.update({"font.size": 9, "figure.facecolor": BG, "axes.facecolor": BG})
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
bands = [(0, 1600, "< 1,600 m", "#c98b3a"), (1600, 1900, "1,600-1,900 m", "#2a9d8f"), (1900, 3000, "≥ 1,900 m", "#5e3c99")]
lo, hi = 30, 260
doy_ticks = [32, 62, 93, 121, 152, 182, 213, 244]
doy_lab = ["Nov 1", "Dec 1", "Jan 1", "Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1"]
for ax, single, title in [(axs[0], True, "One clear snow season (ground and satellite)"),
                          (axs[1], False, "Season split by a mid-winter melt (either record)")]:
    d = ok[ok.single == single]
    for b0, b1, lab, col in bands:
        s = d[(d.elev_site >= b0) & (d.elev_site < b1)]
        ax.scatter(s.ground_off, s.sat_off, s=16, color=col, alpha=0.7, lw=0, label=f"{lab} (n={len(s)})")
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.8)
    e = d.sat_off - d.ground_off
    ax.text(0.03, 0.97, f"n = {len(d)}\nmedian bias {e.median():+.0f} d\nMAE {e.abs().mean():.0f} d\n"
                        f"r = {np.corrcoef(d.ground_off, d.sat_off)[0, 1]:.2f}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks(doy_ticks, doy_lab, rotation=45); ax.set_yticks(doy_ticks, doy_lab)
    ax.set_xlabel("Ground melt-out (ARS snow depth > 5 cm)"); ax.set_title(title, fontsize=10)
    ax.grid(color="#e6e4df", lw=0.5)
axs[0].set_ylabel("30 m product melt-out (site cell)")
axs[1].legend(loc="lower right", fontsize=8, frameon=False)
fig.suptitle("Reynolds Creek: downscaled MODIS melt-out vs ARS snow depth, 39 sites, WY2000-2026", x=0.01, ha="left", fontsize=11)
fig.text(0.01, -0.03, "Site-years with ≥ 80% of Nov-May depth records and a ≥ 14-day season in both. "
         "'Split' = more than 14 snow days fall outside the main season, so the two records can pick different runs.",
         fontsize=7, color=TXT2)
fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.15)
