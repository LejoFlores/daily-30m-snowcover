"""Bogus Basin SNOTEL 978 snow season (ground, SWE > 2.5 mm) above Dry Creek N- vs S-facing band ranges.
Bars = median snow-on to median snow-off, WY2001-2025; whiskers = +/- 1 SD of each.
usage: python3 fig_snotel_dc_ranges.py ASPECT_RANGES_CSV GROUND_ONOFF_CSV OUT_PNG"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

S = pd.read_csv(sys.argv[1]); G = pd.read_csv(sys.argv[2]); out = sys.argv[3]
S = S[S.region == "Dry Creek"]
g = G[(G.site_id == "SNOTEL978") & G.wy.between(2001, 2025)]
sn = dict(on_med=g.snow_on.median(), on_sd=g.snow_on.std(), off_med=g.snow_off.median(),
          off_sd=g.snow_off.std(), scd_med=g.scd.median())
TXT, TXT2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"SNOTEL": "#8c510a", "N": "#2a78d6", "S": "#e8a33d"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG})
bands = sorted(S.band.unique())
fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 1.6 + 0.62 * len(bands)), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, len(bands)], hspace=0.15))

def bar(ax, y, x, col, h, lab):
    ax.barh(y, x["off_med"] - x["on_med"], left=x["on_med"], height=h, color=col, edgecolor=col)
    ax.errorbar([x["on_med"], x["off_med"]], [y, y], xerr=[x["on_sd"], x["off_sd"]], fmt="none",
                ecolor=TXT, elinewidth=1, capsize=3)
    ax.text(x["off_med"] + x["off_sd"] + 3, y, lab, va="center", fontsize=7, color=TXT2)

bar(a0, 0, sn, C["SNOTEL"], 0.5, f"{sn['scd_med']:.0f} d")
a0.set_yticks([0]); a0.set_yticklabels(["1,942 m"]); a0.set_ylim(-0.6, 0.6)
a0.set_title("Bogus Basin SNOTEL 978 (ground, SWE > 2.5 mm)", loc="left", fontsize=10, color=C["SNOTEL"], fontweight="bold")
H = 0.36
for i, b in enumerate(bands):
    for a, dy in (("N", H / 2 + 0.02), ("S", -H / 2 - 0.02)):
        x = S[(S.band == b) & (S.aspect == a)]
        if not x.empty:
            bar(a1, i + dy, x.iloc[0], C[a], H, f"{x.iloc[0].scd_med:.0f} d")
        else:
            a1.barh(i + dy, 289, left=1, height=H, color="#d9d8d4", edgecolor="none")
            a1.text(145, i + dy, f"too few {a}-facing cells in this band (ridgetop)", ha="center", va="center", fontsize=7.5, color=TXT2)
a1.set_yticks(range(len(bands))); a1.set_yticklabels([f"{b}-{b + 200} m" for b in bands]); a1.set_ylim(-0.6, len(bands) - 0.4)
a1.set_title("Dry Creek (satellite, MODIS downscaled to 30 m)", loc="left", fontsize=10, color=TXT, fontweight="bold")
a1.legend(handles=[Patch(color=C["N"], label="N-facing (NW, N, NE)"), Patch(color=C["S"], label="S-facing (SE, S, SW)")],
          frameon=False, fontsize=8, loc="lower right")
for ax in (a0, a1):
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ticks = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274]
a1.set_xticks(ticks); a1.set_xticklabels(["Oct 1", "Nov 1", "Dec 1", "Jan 1", "Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1", "Jul 1"])
a1.set_xlim(1, 290); a1.set_xlabel("date in water year")
fig.text(0.01, 0.003, "Bars span median snow-on to median snow-off, WY2001-2025; whiskers = ±1 SD of snow-on (left) and snow-off (right); "
         "label = median snow days.", fontsize=6.5, color=TXT2)
fig.savefig(out, dpi=170, bbox_inches="tight")
print(sn)
