"""Assemble final tables and figures for /mnt/project-files/snowcover/.
usage: python3 assemble.py WORK_OUT_DIR FINAL_DIR GRID_NPZ WATERSHED_GPKG"""
import os, sys, glob, shutil
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import stats

W, F, grid_npz, ws_path = sys.argv[1:5]
for d in ("tables", "rasters", "figures", "validation"):
    os.makedirs(f"{F}/{d}", exist_ok=True)

# ---------- unified band table ----------
def norm(df, product, on="snow_on", off="snow_off"):
    out = pd.DataFrame(dict(product=product, region=df.region, elev_band=df.elev_band.astype(str).str.replace(".0", "", regex=False),
                            aspect_class=df.aspect_class, wy=df.wy.astype(int),
                            snow_on_dowy=df[on], snow_off_dowy=df[off]))
    for c in ["scd", "on_bracket_days", "off_bracket_days", "n_obs", "ndsi_threshold", "n_pixels", "n_cells", "censored"]:
        out[c] = df[c] if c in df else np.nan
    start = pd.to_datetime((out.wy - 1).astype(str) + "-10-01")
    for c in ["on", "off"]:
        v = out[f"snow_{c}_dowy"]
        out[f"snow_{c}_date"] = (start + pd.to_timedelta(v - 1, unit="D")).dt.date.where(v.notna())
    return out

parts = []
for p in ["MOD10A1F", "MYD10A1F", "VNP10A1F"]:
    b = pd.read_csv(f"{W}/daily/{p}_band_onoff.csv")
    # first water year of each record has no observed snow-on
    first = {"MOD10A1F": 2000, "MYD10A1F": 2002, "VNP10A1F": 2012}[p]
    b.loc[b.wy == first, "snow_on"] = np.nan
    parts.append(norm(b, p))
h = pd.read_csv(f"{W}/hires/hires_band_onoff.csv")
parts.append(norm(h, "Landsat_S2_30m", "on_dowy", "off_dowy"))
ds = pd.read_csv(f"{W}/ds30h/ds30_emp_smooth_band_onoff.csv")
parts.append(norm(ds, "MODIS_downscaled_30m"))
B = pd.concat(parts, ignore_index=True)
B = B[(B.wy >= 1985) & (B.wy <= 2026)]
B.loc[B["product"].isin(["MOD10A1F", "MYD10A1F", "VNP10A1F"]) & (B.wy < 2000), :] = np.nan
B = B.dropna(subset=["product"])
B = B[~((B["product"] == "MODIS_downscaled_30m") & (B.n_cells < 30))]  # bands under 2.7 ha are noise
B.to_csv(f"{F}/tables/band_snow_onoff_all_products.csv", index=False)

# tRIBS-facing table: downscaled daily (recommended) + Landsat/S2 check + native MODIS
key = ["region", "elev_band", "aspect_class", "wy"]
dsx = B[B["product"] == "MODIS_downscaled_30m"][key + ["snow_on_dowy", "snow_off_dowy", "snow_on_date", "snow_off_date", "scd", "n_cells"]]
lsx = B[B["product"] == "Landsat_S2_30m"][key + ["snow_on_dowy", "snow_off_dowy", "on_bracket_days", "off_bracket_days", "n_obs"]].add_prefix("l30_").rename(columns={f"l30_{k}": k for k in key})
mdx = B[(B["product"] == "MOD10A1F") & (B.ndsi_threshold == 10) & (B.aspect_class == "all")][["region", "elev_band", "wy", "snow_on_dowy", "snow_off_dowy"]].add_prefix("mod500_").rename(columns={"mod500_region": "region", "mod500_elev_band": "elev_band", "mod500_wy": "wy"})
TR = dsx.merge(lsx, on=key, how="outer").merge(mdx, on=["region", "elev_band", "wy"], how="left")
TR = TR[(TR.wy >= 2000) & (TR.wy <= 2026)].sort_values(key)
TR.to_csv(f"{F}/tables/tribs_snow_timing_by_band.csv", index=False)

# climatology WY2001-2025
C = B[(B.wy >= 2001) & (B.wy <= 2025) & (B.ndsi_threshold.isna() | (B.ndsi_threshold == 10))]
def q(x, p): return np.nanpercentile(x, p) if np.isfinite(x).any() else np.nan
clim = C.groupby(["product", "region", "elev_band", "aspect_class"]).agg(
    n_years_off=("snow_off_dowy", lambda x: int(np.isfinite(x).sum())),
    snow_on_median=("snow_on_dowy", "median"), snow_off_median=("snow_off_dowy", "median"),
    snow_off_p25=("snow_off_dowy", lambda x: q(x, 25)), snow_off_p75=("snow_off_dowy", lambda x: q(x, 75)),
    scd_median=("scd", "median")).round(1).reset_index()
# Sen slope of snow-off (days per decade) for the downscaled product
tr = []
for k, g in C[C["product"] == "MODIS_downscaled_30m"].groupby(["region", "elev_band", "aspect_class"]):
    g = g.dropna(subset=["snow_off_dowy"])
    if len(g) >= 15:
        sl = stats.theilslopes(g.snow_off_dowy, g.wy)[0] * 10
        tau, p = stats.kendalltau(g.wy, g.snow_off_dowy)
        tr.append(dict(region=k[0], elev_band=k[1], aspect_class=k[2], product="MODIS_downscaled_30m",
                       snow_off_trend_days_per_decade=round(sl, 1), kendall_p=round(p, 3)))
clim = clim.merge(pd.DataFrame(tr), on=["product", "region", "elev_band", "aspect_class"], how="left")
clim.to_csv(f"{F}/tables/band_climatology_wy2001_2025.csv", index=False)

# ---------- rasters ----------
for f in glob.glob(f"{W}/ds30h/ds30_emp_smooth_*_wy2000_2026.tif"):
    shutil.copy(f, f"{F}/rasters/" + os.path.basename(f).replace("ds30_emp_smooth", "modis_downscaled_30m"))
for f in ["hires_pixel_snowoff_median_wy2000_2025.tif", "hires_pixel_snowoff_dowy_wy1985_2026.tif", "hires_pixel_snowoff_bracket_days_wy1985_2026.tif"]:
    shutil.copy(f"{W}/hires/{f}", f"{F}/rasters/" + f.replace("hires_", "landsat_s2_"))
os.makedirs(f"{F}/rasters/daily_30m", exist_ok=True)
for f in glob.glob(f"{W}/ds30h/ds30_emp_smooth_daily_WY*.npz"):
    shutil.copy(f, f"{F}/rasters/daily_30m/" + os.path.basename(f).replace("ds30_emp_smooth_daily", "snow30m_daily"))
G = np.load(grid_npz)
P = np.load(f"{W}/../data/snow_persistence_hybrid.npz")["P_all"] if os.path.exists(f"{W}/../data/snow_persistence_hybrid.npz") else None
prof = dict(driver="GTiff", height=G["region"].shape[0], width=G["region"].shape[1], dtype="float32", crs="EPSG:32611",
            transform=rasterio.transform.from_origin(564480, 4849500, 30, 30), nodata=np.nan, compress="deflate")
layers = [("elevation_m", G["elev"]), ("slope_deg", G["slope"]), ("aspect_deg", G["aspect"]), ("region_1dc_2bogus", G["region"].astype(float))]
if P is not None:
    layers.append(("snow_persistence_feb_jun_hybrid", P))
with rasterio.open(f"{F}/rasters/grid_30m_terrain_region_persistence.tif", "w", count=len(layers), **prof) as r:
    for i, (n, a) in enumerate(layers, 1):
        r.write(a.astype(np.float32), i); r.set_band_description(i, n)

# ---------- figures ----------
S1, S2c, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": TXT2, "axes.labelcolor": TXT, "xtick.color": TXT2, "ytick.color": TXT2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
def dowy_ticks(ax, axis="y"):
    ticks = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274]
    labs = ["Oct 1", "Nov 1", "Dec 1", "Jan 1", "Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1", "Jul 1"]
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ticks); (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)(labs)

# Fig 1: median snow-on/off vs elevation by product
prods = [("MOD10A1F", "MODIS Terra 500 m (2000-)", S1), ("VNP10A1F", "VIIRS 375 m (2012-)", S2c),
         ("Landsat_S2_30m", "Landsat + Sentinel-2 30 m", S3), ("MODIS_downscaled_30m", "MODIS downscaled 30 m", S4)]
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
for ax, reg, title in zip(axs, ["dry_creek", "bogus"], ["Dry Creek", "Bogus Basin (outside Dry Creek)"]):
    for p, lab, col in prods:
        c = clim[(clim["product"] == p) & (clim.region == reg) & (clim.aspect_class == "all") & (clim.elev_band != "all")].copy()
        c["e"] = c.elev_band.astype(float) + 100
        c = c[c.n_years_off >= 5].sort_values("e")
        ax.plot(c.snow_off_median, c.e, "-o", color=col, lw=2, ms=5, label=lab)
        ax.plot(c.snow_on_median, c.e, "--o", color=col, lw=1.2, ms=4, mfc="white")
    ax.set_title(title, loc="left", color=TXT, fontsize=10)
    dowy_ticks(ax, "x"); ax.set_xlim(40, 250); ax.set_xlabel("median date, WY2001-2025 (dashed = snow-on, solid = snow-off)")
axs[0].set_ylabel("elevation band centre (m)")
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{F}/figures/fig1_onoff_vs_elevation.png", dpi=160); plt.close(fig)

# Fig 2: per-WY snow-off and snow-on by Dry Creek band (downscaled), sequential ramp by elevation
bl = ["1000", "1200", "1400", "1600", "1800", "2000"]
ramp = ["#b9d4f4", "#8fb9ee", "#5f9be3", "#2a78d6", "#1d5aa7", "#123c72"]
fig, axs = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
d = B[(B["product"] == "MODIS_downscaled_30m") & (B.region == "dry_creek") & (B.aspect_class == "all")]
for ax, col, lab in zip(axs, ["snow_off_dowy", "snow_on_dowy"], ["snow-off (melt-out)", "snow-on"]):
    for b, c in zip(bl, ramp):
        g = d[d.elev_band == b].sort_values("wy")
        ax.plot(g.wy, g[col], "-o", color=c, lw=1.6, ms=3.5, label=f"{b}-{int(b) + 200} m")
    dowy_ticks(ax); ax.set_title(f"Dry Creek {lab} by elevation band (MODIS downscaled to 30 m; band date = first/last day >= 50% snow)", loc="left", fontsize=9, color=TXT)
    ax.axvspan(2025.5, 2026.5, color=GRID, zorder=0)
axs[0].legend(frameon=False, ncol=6, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.02))
axs[1].set_xlabel("water year (shaded = WY2026)")
fig.tight_layout(); fig.savefig(f"{F}/figures/fig2_dry_creek_bands_by_wy.png", dpi=160); plt.close(fig)

# Fig 3: maps of median melt-out
with rasterio.open(f"{F}/rasters/modis_downscaled_30m_snow_off_dowy_wy2000_2026.tif") as r:
    A = r.read()[1:26]  # WY2001-2025
with rasterio.open(f"{F}/rasters/landsat_s2_pixel_snowoff_median_wy2000_2025.tif") as r:
    L = r.read(1); Ln = r.read(2)
med = np.nanmedian(A, 0)
L = np.where(Ln >= 8, L, np.nan)
ws = gpd.read_file(ws_path).to_crs(32611)
ext = [564480, 577020, 4836000, 4849500]
fig, axs = plt.subplots(1, 2, figsize=(9, 5))
for ax, a, t in zip(axs, [med, L], ["MODIS downscaled 30 m: median melt-out WY2001-2025", "Landsat + Sentinel-2 30 m: median melt-out (>= 8 yrs)"]):
    im = ax.imshow(a, extent=ext, cmap="Blues", vmin=100, vmax=240, interpolation="nearest")
    ws.boundary.plot(ax=ax, color=TXT, lw=0.8)
    from pyproj import Transformer
    sx, sy = Transformer.from_crs(4326, 32611, always_xy=True).transform(-116.09685, 43.76377)
    ax.plot(sx, sy, "^", color="#eb6834", ms=7, mec="white")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(t, fontsize=9, loc="left", color=TXT); ax.set_xticks([]); ax.spines[:].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, pad=0.04)
axs[0].text(ext[0] + 300, ext[2] + 400, "triangle = Bogus Basin SNOTEL; outline = Dry Creek", fontsize=7, color=TXT2)
cb.set_ticks([93, 124, 152, 183, 213, 244]); cb.set_ticklabels(["Jan 1", "Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1"])
fig.savefig(f"{F}/figures/fig3_meltout_maps.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# Fig 4: validation scatter
J = pd.read_csv(f"{W}/val/validation_pairs.csv")
fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True)
for ax, (p, lab, col) in zip(axs, [("MOD10A1F", "MODIS Terra 500 m (NDSI >= 10)", S1), ("MODIS_ds30_emp_smooth", "MODIS downscaled 30 m", S4), ("Landsat+S2_30m", "Landsat + Sentinel-2 30 m", S3)]):
    j = J[(J["product"] == p) & (J.ndsi_threshold.isna() | (J.ndsi_threshold == 10))].dropna(subset=["snow_off_sat", "snow_off_gnd"])
    for src, mk in [("snotel", "s"), ("snow_depth", "o"), ("soil_temp_2cm", "^"), ("soil_temp_5cm", "v")]:
        jj = j[j.source == src]
        ax.scatter(jj.snow_off_gnd, jj.snow_off_sat, s=16, color=col, marker=mk, alpha=0.8, edgecolor="#fcfcfb", lw=0.5, label=src.replace("_", " "))
    ax.plot([90, 260], [90, 260], color=TXT2, lw=1)
    d_ = j.snow_off_sat - j.snow_off_gnd
    ax.set_title(f"{lab}\nn={len(j)}, median bias {d_.median():+.0f} d, MAE {d_.abs().median():.0f} d", fontsize=8.5, loc="left", color=TXT)
    dowy_ticks(ax); dowy_ticks(ax, "x"); ax.set_xlim(90, 260); ax.set_ylim(90, 260)
    ax.set_xlabel("ground melt-out")
axs[0].set_ylabel("satellite melt-out"); axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/figures/fig4_validation_snowoff.png", dpi=160); plt.close(fig)

# Fig 5: north vs south melt-out by band
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for p, lab, col, off in [("MODIS_downscaled_30m", "MODIS downscaled 30 m", S4, -15), ("Landsat_S2_30m", "Landsat + Sentinel-2 30 m", S3, 15)]:
    c = clim[(clim["product"] == p) & (clim.elev_band != "all")]
    n = c[c.aspect_class == "N"].set_index(["region", "elev_band"]).snow_off_median
    s = c[c.aspect_class == "S"].set_index(["region", "elev_band"]).snow_off_median
    dd = (n - s).dropna().rename("d").reset_index()
    dd = dd[dd.region == "dry_creek"]
    x = dd.elev_band.astype(float) + 100 + off
    ax.bar(x, dd["d"], width=28, color=col, label=lab)
ax.axhline(0, color=TXT2, lw=0.8)
ax.set_xlabel("Dry Creek elevation band centre (m)"); ax.set_ylabel("north minus south melt-out (days)")
ax.legend(frameon=False, fontsize=8); ax.set_title("Aspect effect on median melt-out, WY2001-2025", loc="left", fontsize=9, color=TXT)
fig.tight_layout(); fig.savefig(f"{F}/figures/fig5_aspect_effect.png", dpi=160); plt.close(fig)

# validation tables
for f in ["validation_summary_by_product.csv", "validation_summary_common_sample.csv", "validation_summary_by_site.csv", "validation_pairs.csv", "sites_satellite_onoff_by_wy.csv"]:
    shutil.copy(f"{W}/val/{f}", f"{F}/validation/{f}")
shutil.copy(f"{W}/ground_snow_onoff_by_wy.csv", f"{F}/validation/ground_snow_onoff_by_wy.csv")
for f in ["downscale_scores_summary.csv"]:
    a = pd.read_csv(f"{W}/downscale_v2/{f}"); b = pd.read_csv(f"{W}/downscale_emp/{f}"); c = pd.read_csv(f"{W}/downscale_hyb/{f}")
    c["kind"] = c["kind"].str.replace("emp_", "hybrid_")
    pd.concat([a, b, c]).drop_duplicates().to_csv(f"{F}/validation/downscaling_scores_vs_landsat_s2.csv", index=False)
print(clim[(clim.aspect_class == "all") & (clim["product"].isin(["MODIS_downscaled_30m", "Landsat_S2_30m"]))].to_string())
