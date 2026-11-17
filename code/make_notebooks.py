"""Write the three workflow notebooks (pre-processing, downscaling, verification) into OUT_DIR.
usage: python3 make_notebooks.py OUT_DIR"""
import sys, os
import nbformat as nbf

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = dict(display_name="Python 3", language="python", name="python3")
    for kind, src in cells:
        src = src.strip("\n")
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return n


SETUP = '''
import os, sys, io, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4

ROOT = Path("..")                      # essd_draft/
CODE = ROOT / "code"
DATA = Path("data"); DATA.mkdir(exist_ok=True)
os.environ["SNOW_DOMAIN"] = "rcew"     # the pipeline code reads the domain from this variable
sys.path.insert(0, str(CODE))

SEASONS_NC = ROOT / "rcew_snow_seasons_30m_v1.nc"
DAILY_STEM = "rcew_snowcover_daily_30m_v1"   # one file, or parts along time (see daily_io.py)
MODIS_TAR = ROOT / "inputs" / "rcew_mod10a1f_window_2000_2026.tar"
EXAMPLE_DATE = pd.Timestamp("2019-03-18")   # clear Landsat 8 overpass with a partly snow-covered watershed

plt.rcParams.update({"font.size": 9, "figure.dpi": 110, "axes.facecolor": "#d4d4d4"})   # grey = outside or no data
from matplotlib.colors import LinearSegmentedColormap
SNOW = LinearSegmentedColormap.from_list("snow", ["#5c4326", "#9c7a4e", "#cdbb98", "#e6e9ea", "#ffffff"])   # bare ground to snow
'''

STATICS = '''
S = netCDF4.Dataset(SEASONS_NC)
grab = lambda k: np.ma.filled(S[k][:].astype(float), np.nan)
x, y = S["x"][:], S["y"][:]
region = np.ma.filled(S["region"][:], 0).astype(int)
inside = region > 0
elev, slope, aspect = grab("elevation"), grab("slope"), grab("aspect")
forest = np.ma.filled(S["forest"][:], 0).astype(bool)
persistence = grab("snow_persistence")
extent = [x.min() - 15, x.max() + 15, y.min() - 15, y.max() + 15]
NY, NX = region.shape
print(f"grid {NY} x {NX} cells at 30 m, {inside.sum() * 900 / 1e6:.0f} km2 inside the watershed, "
      f"{np.nanmin(elev):.0f}-{np.nanmax(elev):.0f} m, forest share {forest[inside].mean():.0%}")
'''

MODIS_DAY = '''
def modis_day(date):
    """One day's MOD10A1F window (89 x 179 pixels at 463 m) from the archive of pre-cut windows."""
    name = f"mod/A{date.year}{date.dayofyear:03d}.npz"
    with tarfile.open(MODIS_TAR) as t:
        return dict(np.load(io.BytesIO(t.extractfile(name).read())))
'''

# ---------------------------------------------------------------- notebook 1
nb1 = nb([
("md", '''
# 1. Pre-processing: from satellite products to downscaling inputs

**Daily 30 m snow cover for Reynolds Creek and Dry Creek, Idaho.** This notebook is the first of three that walk through the workflow on one example day at Reynolds Creek:

1. **Pre-processing** (this notebook): the 30 m grid, the daily MODIS snow fraction, a Landsat snow map, and the snow-persistence ranking.
2. **Downscaling**: placing the MODIS snow fraction on the 30 m grid.
3. **Verification**: comparing the product with ground snow depth and with held-out Landsat maps.

The notebooks run from the `notebooks/` folder of the data release and read the released files next to it. The production scripts are in `../code/`; the notebooks call the same functions where they can and repeat short pieces of logic inline where that is clearer.

*Disclosure: the processing code and these notebooks were written with the assistance of Claude (Anthropic), working under the direction of the authors.*
'''),
("code", SETUP),
("md", '''
## 1.1 The 30 m grid and its static layers

Everything is computed on a 30 m UTM 11N grid (EPSG:32611). Elevation, slope and aspect come from a 10 m DEM averaged to 30 m. The forest flag marks cells whose median summer NDVI (Landsat, July-August) is at least 0.45. Production script: `build_grid.py`.
'''),
("code", STATICS),
("code", '''
fig, axs = plt.subplots(1, 3, figsize=(10, 5.5), constrained_layout=True)
for ax, fld, cmap, lab in [(axs[0], elev, "terrain", "elevation (m)"), (axs[1], slope, "magma", "slope (degrees)"),
                           (axs[2], np.where(inside, forest, np.nan), "Greens", "forest flag")]:
    im = ax.imshow(fld, extent=extent, cmap=cmap, interpolation="nearest")
    ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.6)
plt.show()
'''),
("md", '''
## 1.2 Daily MODIS snow fraction

The daily input is **MOD10A1F v61**, the cloud-gap-filled MODIS/Terra NDSI snow cover at 463 m (tile h09v04). `fetch_tiles.py` downloads each daily tile from NSIDC with an Earthdata token, cuts the window that covers the watershed and discards the tile:

```
SNOW_DOMAIN=rcew TILE_WINDOW=1591,1680,1075,1254 python3 ../code/fetch_tiles.py MOD10A1F 61 2000-02-24 2026-09-30 mod/
```

The release includes those windows (`inputs/rcew_mod10a1f_window_2000_2026.tar`, 9,716 days), so the download is not repeated here.

NDSI (0-100) is converted to fractional snow-covered area with the relation of Salomonson & Appel (2004): `fSCA = clip(-0.01 + 1.45 * NDSI / 100, 0, 1)`. Values above 100 are fill codes. `Cloud_Persistence` counts the days since the pixel was last seen clear; pixels older than 10 days are treated as missing.
'''),
("code", MODIS_DAY),
("code", '''
from downscale import ndsi_to_fsca

m = modis_day(EXAMPLE_DATE)
ndsi = m["CGF_NDSI_Snow_Cover"].astype(float)
cloud_age = m["Cloud_Persistence"].astype(float)
fsca_modis = np.where((ndsi <= 100) & (cloud_age <= 10), ndsi_to_fsca(ndsi), np.nan)
seen_today = m["MOD10A1_NDSI_Snow_Cover"] <= 100
print(f"{EXAMPLE_DATE.date()}: window {ndsi.shape}, {seen_today.mean():.0%} of pixels seen clear today, "
      f"mean fSCA {np.nanmean(fsca_modis):.2f}")

fig, axs = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
for ax, fld, cmap, lab, vmax in [(axs[0], np.where(ndsi <= 100, ndsi, np.nan), SNOW, "gap-filled NDSI", 100),
                                 (axs[1], fsca_modis, SNOW, "fSCA", 1), (axs[2], cloud_age, "Oranges", "days since clear view", 10)]:
    im = ax.imshow(fld, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()
'''),
("md", '''
## 1.3 A 30 m snow map from Landsat

Landsat and Sentinel-2 give the fine-scale snow pattern on the days they see the ground. The production run read every Landsat 4-9 Collection 2 Level-2 scene (Tier 1) and every Sentinel-2 L2A scene over the watershed (3,696 usable scenes at Reynolds Creek; `fetch_landsat_c2.py`, `fetch_s2.py`).

Here we read one Landsat 8 scene from Microsoft Planetary Computer, the same source, and apply the snow test of `process_hires.py`. The scene is cached in `data/` after the first run, so the notebook also runs offline.
'''),
("code", '''
import json, urllib.request
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds

SCENE = "LC08_L2SP_042030_20190318_02_T1"
cache = DATA / f"{SCENE}.npz"
X0, Y1 = extent[0], extent[3]; X1, Y0 = extent[1], extent[2]
DST = rasterio.transform.from_origin(X0, Y1, 30, 30)


def warp(href, token, resampling, dtype, nodata):
    with rasterio.open(f"/vsicurl/{href}?{token}") as r:
        w = from_bounds(X0 - 200, Y0 - 200, X1 + 200, Y1 + 200, r.transform).round_offsets().round_lengths()
        a = r.read(1, window=w, boundless=True, fill_value=nodata)
        o = np.full((NY, NX), nodata, dtype)
        reproject(a, o, src_transform=r.window_transform(w), src_crs=r.crs, dst_transform=DST, dst_crs="EPSG:32611",
                  resampling=resampling, src_nodata=nodata, dst_nodata=nodata)
    return o


if not cache.exists():
    api = "https://planetarycomputer.microsoft.com/api"
    item = json.load(urllib.request.urlopen(f"{api}/stac/v1/collections/landsat-c2-l2/items/{SCENE}", timeout=60))
    token = json.load(urllib.request.urlopen(f"{api}/sas/v1/token/landsat-c2-l2", timeout=60))["token"]
    bands = {}
    for name, asset in [("green", "green"), ("red", "red"), ("nir", "nir08"), ("swir1", "swir16")]:
        dn = warp(item["assets"][asset]["href"], token, Resampling.average, np.float32, 0)
        bands[name] = np.where(dn > 0, dn * 0.0000275 - 0.2, np.nan).astype(np.float32)   # C2 L2 scale and offset
    bands["qa_pixel"] = warp(item["assets"]["qa_pixel"]["href"], token, Resampling.nearest, np.uint16, 1)
    np.savez_compressed(cache, **bands)
L = dict(np.load(cache))
print({k: v.shape for k, v in L.items()})
'''),
("md", '''
**Snow test** (per clear cell), with `NDSI = (green - swir1) / (green + swir1)`:

- open cells: NDSI >= 0.40, NIR >= 0.11 and green >= 0.10 (the SNOMAP test);
- forest cells: NDSI >= 0.20, NIR >= 0.11 and green >= 0.08, because canopy lowers the NDSI of a snow-covered stand.

**Clear** means not fill, and not flagged cloud, cloud shadow or cirrus in `QA_PIXEL`. Cloud masks often call bright snow cloud, so a cell flagged cloud with NDSI >= 0.6 and SWIR < 0.15 is kept and counted as snow (real cloud is bright in the SWIR).
'''),
("code", '''
bits = lambda a, start, n=1: (a.astype(np.int32) >> start) & ((1 << n) - 1)

q = L["qa_pixel"]
cloud = (bits(q, 3) == 1) | (bits(q, 4) == 1) | (bits(q, 8, 2) == 3) | (bits(q, 14, 2) == 3)
fill = (bits(q, 0) == 1) | ~np.isfinite(L["green"])
ndsi30 = (L["green"] - L["swir1"]) / (L["green"] + L["swir1"])
nir_ok = L["nir"] >= 0.11
snow = np.where(forest, (ndsi30 >= 0.20) & nir_ok & (L["green"] >= 0.08),
                (ndsi30 >= 0.40) & nir_ok & (L["green"] >= 0.10))
rescue = cloud & (ndsi30 >= 0.6) & (L["swir1"] < 0.15) & nir_ok
clear = (~cloud | rescue) & ~fill & np.isfinite(ndsi30)
landsat_snow = np.full((NY, NX), 255, np.uint8)          # 0 no snow, 1 snow, 255 not clear
landsat_snow[clear] = snow[clear]
print(f"clear {clear[inside].mean():.0%} of the watershed, snow on {snow[inside & clear].mean():.0%} of clear cells")

rgb = np.clip(np.dstack([L["swir1"], L["nir"], L["green"]]) / 0.6, 0, 1)    # snow shows cyan
show = np.where(inside & clear, snow.astype(float), np.nan)
fig, axs = plt.subplots(1, 2, figsize=(7.5, 6), constrained_layout=True)
axs[0].imshow(rgb, extent=extent); axs[0].set_title("Landsat 8, SWIR-NIR-green")
axs[1].imshow(show, extent=extent, cmap=SNOW, vmin=0, vmax=1, interpolation="nearest"); axs[1].set_title("30 m snow map")
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
plt.show()
np.savez_compressed(DATA / "example_landsat_snow_20190318.npz", snow=landsat_snow)
'''),
("md", '''
## 1.4 Snow persistence: the ranking used for downscaling

Stacking all such maps gives, for every 30 m cell, the **share of clear February-June views in which the cell is snow covered** (`build_persistence.py`). Cells with high persistence hold snow longest, so this is the order in which the watershed fills with snow and empties again.

Under forest, optical sensors miss snow beneath the canopy and the observed persistence is biased low. There the value is replaced by a prediction from elevation and spring insolation, fitted on the open cells (the "terrain model"; r = 0.92-0.93 on open cells at Reynolds Creek). The released `snow_persistence` layer already includes that replacement, and the `forest` flag marks where it applies.

This builds on Walters et al. (2014, *Remote Sensing of Environment* 152:413), who ranked cells by a terrain score of insolation and relative elevation. Notebook 2 compares the two rankings.
'''),
("code", '''
fig, axs = plt.subplots(1, 2, figsize=(9.5, 5), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.5]))
im = axs[0].imshow(persistence, extent=extent, cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest")
axs[0].set_title("snow persistence"); axs[0].set_xticks([]); axs[0].set_yticks([]); fig.colorbar(im, ax=axs[0], shrink=0.6)
north = (aspect >= 315) | (aspect < 45); south = (aspect >= 135) & (aspect < 225)
steep = inside & (slope >= 10)
for sel, col, lab in [(steep & north, "#2a78d6", "north-facing"), (steep & south, "#e07a1f", "south-facing")]:
    band = np.floor(elev[sel] / 100) * 100
    med = pd.Series(persistence[sel]).groupby(band).median()
    axs[1].plot(med.index + 50, med.values, "o-", color=col, label=lab)
axs[1].set_xlabel("elevation (m)"); axs[1].set_ylabel("median snow persistence"); axs[1].legend(frameon=False)
axs[1].set_title("persistence by elevation and aspect (slopes of 10 degrees or more)")
plt.show()
'''),
("md", '''
**Outputs of this step:** the daily MODIS fSCA windows, the static grid layers and the snow-persistence ranking. Notebook 2 combines them.
'''),
])

# ---------------------------------------------------------------- notebook 2
nb2 = nb([
("md", '''
# 2. Downscaling: placing the MODIS snow fraction at 30 m

For each day, MODIS says **how much** of each 463 m pixel is snow covered. The snow-persistence ranking from notebook 1 says **where** within the pixel that snow most likely is. The downscaling fills each neighbourhood with snow in rank order until the MODIS fraction is used up.

This notebook does that for the example day (18 March 2019), compares the result with the Landsat snow map of the same day, and then turns one cell's daily series into snow-on and snow-off dates. Production scripts: `downscale.py` (the method), `ds_run.py` (all days) and `build_essd_nc.py` (the released NetCDF files).

*Disclosure: the processing code and these notebooks were written with the assistance of Claude (Anthropic), working under the direction of the authors.*
'''),
("code", SETUP),
("code", STATICS),
("code", MODIS_DAY),
("md", '''
## 2.1 Set up the terrain object

`Terrain` holds, for every 30 m cell, its position inside the MODIS grid and the terrain quantities used for ranking. It is built here from the static layers of the released seasons file (the same route as `rebuild_inputs_from_nc.py`). The first call compiles a few functions and takes about a minute.
'''),
("code", '''
from downscale import Terrain, ndsi_to_fsca

G = dict(region=region, elev=elev, slope=slope, aspect=aspect, lon=grab("lon"), lat=grab("lat"))
T = Terrain(G)
T.set_persistence(persistence, "all"); T.emp_fold = "all"
print("cells ranked:", T.valid.sum())
'''),
("md", '''
## 2.2 One day, step by step

1. **MODIS fSCA** at 463 m (notebook 1).
2. **Interpolate** fSCA bilinearly between MODIS pixel centres to every 30 m cell. Ranking inside whole MODIS pixels would leave visible seams along pixel edges; interpolating removes them.
3. **Rank** each cell by snow persistence within a 15 x 15 cell (450 m) window centred on it, as a percentile from 0 (holds snow longest) to 1.
4. A cell is **snow covered** when its percentile is at or below its interpolated fSCA.
'''),
("code", '''
m = modis_day(EXAMPLE_DATE)
ndsi = m["CGF_NDSI_Snow_Cover"].astype(float)
fsca_modis = np.where((ndsi <= 100) & (m["Cloud_Persistence"] <= 10), ndsi_to_fsca(ndsi), np.nan)

fsca30 = T.fsca_field(fsca_modis, "emp_smooth")          # step 2
rank = T.prank(EXAMPLE_DATE.dayofyear, 0, "emp_smooth")   # step 3 (the same for every day)
snow30, ok = T.snow(rank, fsca30)                         # step 4
ok &= inside
print(f"snow on {snow30[ok].mean():.0%} of cells; mean interpolated fSCA {np.nanmean(fsca30[ok]):.0%}")

landsat = np.load(DATA / "example_landsat_snow_20190318.npz")["snow"]
fsca_block = T.fsca_field(fsca_modis, "block")
panels = [(np.where(inside, fsca_block, np.nan), "1. MODIS fSCA (463 m pixels)"),
          (np.where(inside, fsca30, np.nan), "2. interpolated fSCA"),
          (np.where(inside, 1 - rank, np.nan), "3. persistence rank in 450 m window (white = holds snow longest)"),
          (np.where(ok, snow30.astype(float), np.nan), "4. downscaled snow, 30 m"),
          (np.where(inside & (landsat < 255), landsat.astype(float), np.nan), "Landsat 8 snow, same day")]
fig, axs = plt.subplots(1, 5, figsize=(15, 5.6), constrained_layout=True)
for ax, (fld, title) in zip(axs, panels):
    ax.imshow(fld, extent=extent, cmap=SNOW, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
plt.show()
'''),
("md", '''
## 2.3 Does the ranking matter?

Score the downscaled map against the Landsat map with the F-measure (2TP / (2TP + FP + FN)) on cells that Landsat saw clearly. The score is restricted to cells whose interpolated fSCA lies between 5% and 95%, because where MODIS says "all snow" or "no snow" every ranking gives the same answer.

Three rankings are compared with the same fSCA:

- **snow persistence** (the product);
- a **terrain score** in the style of Walters et al. (2014): 0.5 x insolation + 0.5 x relative elevation;
- a **random** order, which shows what MODIS alone gives.

One caveat for this single-day example: the persistence layer was built from all scenes, including this one. The published skill uses persistence from the other half of the years (odd versus even water years); that table is in notebook 3.
'''),
("code", '''
def f_measure(pred, ref, sel):
    tp = (pred & ref & sel).sum(); fp = (pred & ~ref & sel).sum(); fn = (~pred & ref & sel).sum()
    return 2 * tp / (2 * tp + fp + fn)

from downscale import window_prank, WIN
ref = landsat == 1
partial = ok & (landsat < 255) & (fsca30 > 0.05) & (fsca30 < 0.95)
rng = np.random.default_rng(0)
ranks = {"snow persistence": rank,
         "terrain score": T.prank(EXAMPLE_DATE.dayofyear, 0.5, "smooth"),
         "random": window_prank(np.where(T.valid, rng.random(rank.shape), 0.0), T.valid, WIN)}
rows = []
for name, r in ranks.items():
    pred, _ = T.snow(r, fsca30)
    rows.append(dict(ranking=name, F_all=f_measure(pred, ref, partial), F_open=f_measure(pred, ref, partial & ~forest),
                     F_forest=f_measure(pred, ref, partial & forest)))
print(f"{partial.sum():,} cells in partly snow-covered neighbourhoods ({partial.sum() / ok.sum():.0%} of the watershed)")
pd.DataFrame(rows).set_index("ranking").round(3)
'''),
("md", '''
## 2.4 Why interpolate? Pixel seams

Filling each MODIS pixel separately ("block", as in the original method) leaves straight edges where neighbouring pixels have different snow fractions. A zoom on the snow line shows the difference.
'''),
("code", '''
snow_block, _ = T.snow(T.prank(EXAMPLE_DATE.dayofyear, 0, "emp_block"), fsca_block)
iy0, ix0 = [int(v) for v in np.argwhere(inside & (fsca30 > 0.3) & (fsca30 < 0.7))[len(np.argwhere(inside & (fsca30 > 0.3) & (fsca30 < 0.7))) // 2]]
win = (slice(max(iy0 - 90, 0), iy0 + 90), slice(max(ix0 - 90, 0), ix0 + 90))
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9), constrained_layout=True)
for ax, fld, title in [(axs[0], snow_block, "ranked inside each MODIS pixel"), (axs[1], snow30, "interpolated fSCA + 450 m window"),
                       (axs[2], np.where(landsat == 255, np.nan, landsat), "Landsat 8")]:
    ax.imshow(np.where(inside[win], np.asarray(fld, float)[win], np.nan), cmap=SNOW, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
plt.show()
'''),
("md", '''
## 2.5 From daily maps to snow-on and snow-off dates

`ds_run.py` repeats step 2.2 for all 9,716 days and `build_essd_nc.py` writes the daily file with its quality flags. Per cell and water year, the season rules (`ondates.py`) are:

- a missing day is filled when the days before and after agree;
- snow-free gaps of up to 3 days inside snow cover are bridged;
- the **main season** is the longest continuous run of at least 14 days; `snow_on` is its first day and `snow_off` (melt-out) the first snow-free day after it.

Below, one cell's daily series for water year 2019, read from the released daily file.
'''),
("code", '''
from ondates import season
from daily_io import Daily

D = Daily(ROOT, DAILY_STEM)
iy, ix = [int(v) for v in np.argwhere(inside & (np.abs(elev - 1900) < 5) & ~forest)[0]]
days = pd.date_range("2018-10-01", "2019-09-30")
sc = D.cell("snow_cover", iy, ix, days[0], days[-1]).reindex(days).values
fs = D.cell("input_fsca", iy, ix, days[0], days[-1]).reindex(days).values
s = season(sc)
t = days
fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True); ax.set_facecolor("white")
ax.plot(t, fs, color="#9aa7b1", lw=0.8, label="MODIS fSCA interpolated to the cell (%)")
ax.fill_between(t, 0, np.where(sc == 1, 100, 0), step="mid", color="#2a78d6", alpha=0.25, label="downscaled snow cover")
for key, col in [("snow_on", "#1a7f4e"), ("snow_off", "#c0392b")]:
    ax.axvline(t[int(s[key])], color=col, lw=1.5, label=f"{key}: {t[int(s[key])].date()}")
ax.set_ylabel("%"); ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right"); ax.set_ylim(0, 125)
ax.set_title(f"cell at {elev[iy, ix]:.0f} m, water year 2019: {int(s['scd'])} snow days")
plt.show()

stored = int(S["snow_off"][list(S["water_year"][:]).index(2019), iy, ix])
print("melt-out from this notebook: day", int(s["snow_off"]) + 1, "of the water year; stored in the seasons file: day", stored)
'''),
("md", '''
**Outputs of this step:** `rcew_snowcover_daily_30m_v1.nc` (daily snow cover, input fSCA, cloud age and quality flags) and `rcew_snow_seasons_30m_v1.nc` (snow-on, snow-off and snow-cover days per water year). Notebook 3 checks them against independent data.
'''),
])

# ---------------------------------------------------------------- notebook 3
nb3 = nb([
("md", '''
# 3. Verification: ground snow depth and held-out satellite maps

Two independent checks of the Reynolds Creek product:

1. **Ground snow depth** at the USDA-ARS Northwest Watershed Research Center network (daily, 40 sites, water years 2000-2026): melt-out dates and day-by-day snow presence.
2. **Held-out Landsat and Sentinel-2 maps**: how well the downscaling reproduces the 30 m pattern when the persistence ranking is built from other years.

Production scripts: `validate_rcew_ground.py`, `fig_rcew_ground_validation.py`, `ds_eval.py`.

The snow depth record was provided by Andrew Hedrick (USDA-ARS) and is read here from `../../ground/rcew_ars/`. It is not part of the snow-cover data release unless ARS agrees to include it.

*Disclosure: the processing code and these notebooks were written with the assistance of Claude (Anthropic), working under the direction of the authors.*
'''),
("code", SETUP),
("code", STATICS),
("md", '''
## 3.1 The ground network

Snow is counted as present on the ground when depth exceeds 5 cm. Rows flagged `qc_suspect_zero` (a one-day drop from 20 cm or more straight to zero, which looks like a processing break) are left out.
'''),
("code", '''
from pyproj import Transformer

GROUND = ROOT / ".." / "ground" / "rcew_ars"
depth = pd.read_csv(GROUND / "rcew_ars_snowdepth_daily_tidy.csv", parse_dates=["date"])
depth = depth[depth.qc_suspect_zero == 0]
sites = pd.read_csv(GROUND / "rcew_ars_snowdepth_sites.csv").set_index("site")
sites["x"], sites["y"] = Transformer.from_crs(4326, 32611, always_xy=True).transform(sites.longitude.values, sites.latitude.values)
sites["ix"] = [int(np.argmin(abs(x - v))) for v in sites.x]; sites["iy"] = [int(np.argmin(abs(y - v))) for v in sites.y]
sites["forest_cell"] = forest[sites.iy, sites.ix]
print(f"{depth.site.nunique()} sites with data, {len(depth):,} site-days, {depth.date.min().date()} to {depth.date.max().date()}")

fig, ax = plt.subplots(figsize=(4.2, 6), constrained_layout=True)
im = ax.imshow(elev, extent=extent, cmap="terrain", interpolation="nearest")
ax.scatter(sites.x, sites.y, s=14, c=np.where(sites.forest_cell, "#1a7f4e", "k"), edgecolor="w", lw=0.5)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("snow depth sites (green: forest cell)"); fig.colorbar(im, ax=ax, shrink=0.6, label="elevation (m)")
plt.show()
'''),
("md", '''
## 3.2 One site, one winter

Depth at a high-elevation site against the product's daily snow cover at the 30 m cell that contains it, with the main season found by the same rules in both records.
'''),
("code", '''
from ondates import season

from daily_io import Daily

D = Daily(ROOT, DAILY_STEM)


def site_series(site, wy):
    """Daily ground snow (depth > 5 cm) and product snow cover at the site's cell for one water year."""
    days = pd.date_range(f"{wy - 1}-10-01", f"{wy}-09-30")
    g = depth[depth.site == site].set_index("date").depth_cm.reindex(days)
    sc = D.cell("snow_cover", sites.iy[site], sites.ix[site], days[0], days[-1]).reindex(days)
    return days, g, sc


SITE, WY = "RC.163", 2019
days, g, sc = site_series(SITE, WY)
sg = season(np.where(g.isna(), np.nan, (g > 5).astype(float))); ss = season(sc.values)
fig, ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True); ax.set_facecolor("white")
ax.plot(days, g, color="#5a4632", label="snow depth (cm)")
ax.fill_between(days, 0, np.where(sc == 1, g.max() * 1.05, 0), step="mid", color="#2a78d6", alpha=0.2, label="product: snow covered")
ax.fill_between(days, 0, np.where(sc.isna(), g.max() * 1.05, 0), step="mid", color="#999999", alpha=0.35, label="product: no data (cloud for more than 10 days)")
ax.axvline(days[int(sg["snow_off"])], color="#5a4632", ls="--", label=f"ground melt-out {days[int(sg['snow_off'])].date()}")
ax.axvline(days[int(ss["snow_off"])], color="#2a78d6", ls="--", label=f"product melt-out {days[int(ss['snow_off'])].date()}")
ax.set_title(f"{SITE} ({sites.elevation_m[SITE]} m), water year {WY}"); ax.legend(frameon=False, fontsize=8, ncol=2)
plt.show()
'''),
("md", '''
## 3.3 Melt-out dates at all sites and years

`validate_rcew_ground.py` repeats that comparison for every site and water year using the seasons file. A site-year is used when at least 80% of its November-May days have a depth record and both records have a main season that does not run into the start or end of the water year.

In shallow, intermittent snowpacks a mid-winter melt can split the winter into two runs of similar length, and the ground and satellite records may then pick different runs as the "main season". The comparison is therefore shown separately for site-years with **one clear season** in both records (no more than 14 snow days outside the main season) and for **split** seasons.
'''),
("code", '''
import subprocess

VAL = DATA / "validation"
uploads = ROOT / ".." / ".." / "uploads" / "hearth"
raw_depth, raw_meta = uploads / "8e1479ae-0a70-4296-930e-d1502ee197f7", uploads / "2eb797c9-804b-4c1c-93dd-f125d85f9f06"
if raw_depth.exists():       # rerun the production script on the files as received
    subprocess.run([sys.executable, str(CODE / "validate_rcew_ground.py"), str(raw_depth), str(raw_meta), str(SEASONS_NC),
                    str(CODE), str(VAL)], check=True, capture_output=True)
    pairs = pd.read_csv(VAL / "rcew_ground_vs_30m_by_site_wy.csv")
else:                        # otherwise read the released table
    pairs = pd.read_csv(ROOT / "validation" / "rcew_ground_vs_30m_by_site_wy.csv")

ok = pairs[(pairs.cov_nov_may >= 0.8) & (pairs.censored == 0)].dropna(subset=["ground_off", "sat_off"]).copy()
ok["single"] = ((ok.ground_scd - (ok.ground_off - ok.ground_on)) <= 14) & ((ok.sat_scd - (ok.sat_off - ok.sat_on)) <= 14)
ok["err"] = ok.sat_off - ok.ground_off


def summary(d):
    return pd.Series({"site-years": len(d), "bias, median (d)": d.err.median(), "MAE (d)": d.err.abs().mean(),
                      "r": np.corrcoef(d.ground_off, d.sat_off)[0, 1]})


table = pd.DataFrame({"all": summary(ok), "one clear season": summary(ok[ok.single]),
                      "one clear season, open cells": summary(ok[ok.single & (ok.forest == 0)]),
                      "one clear season, forest cells": summary(ok[ok.single & (ok.forest == 1)]),
                      "split season": summary(ok[~ok.single])}).T
table.round(2)
'''),
("code", '''
ticks = [32, 62, 93, 121, 152, 182, 213, 244]; labels = ["Nov 1", "Dec 1", "Jan 1", "Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1"]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
for ax, single, title in [(axs[0], True, "one clear snow season"), (axs[1], False, "season split by a mid-winter melt")]:
    d = ok[ok.single == single]
    sc_ = ax.scatter(d.ground_off, d.sat_off, c=d.elev_site, cmap="viridis", s=12, alpha=0.8, lw=0, vmin=1200, vmax=2200)
    ax.plot([30, 260], [30, 260], color="#888888", lw=0.8); ax.set_facecolor("white")
    ax.set_xlim(30, 260); ax.set_ylim(30, 260); ax.set_aspect("equal"); ax.set_title(f"{title} (n = {len(d)})")
    ax.set_xticks(ticks, labels, rotation=45); ax.set_yticks(ticks, labels); ax.set_xlabel("ground melt-out (depth > 5 cm)")
axs[0].set_ylabel("product melt-out at the site cell"); fig.colorbar(sc_, ax=axs, shrink=0.7, label="site elevation (m)")
plt.show()
'''),
("md", '''
## 3.4 Day-by-day agreement

A second check does not depend on the season definition: on each November-May day, does the product show snow where the ground has more than 5 cm? The full table for all sites is in `../validation/rcew_ground_daily_agreement.csv`. Reading 27 years of daily values at every site takes several minutes, so the cell below computes the score live for four sites across the elevation range and then loads the released table.
'''),
("code", '''
def daily_pairs(site):
    """All Nov-May days with both a depth record and a product value at one site."""
    g = depth[depth.site == site].set_index("date").depth_cm
    sc = D.cell("snow_cover", sites.iy[site], sites.ix[site], g.index.min(), g.index.max())
    j = pd.concat([g.rename("depth"), sc.rename("sat")], axis=1, join="inner").dropna()
    j = j[j.index.month.isin([11, 12, 1, 2, 3, 4, 5])]
    return pd.DataFrame({"ground": j.depth > 5, "sat": j.sat == 1})


def scores(j):
    tp = (j.ground & j.sat).sum(); fp = (~j.ground & j.sat).sum(); fn = (j.ground & ~j.sat).sum(); tn = (~j.ground & ~j.sat).sum()
    return {"days": len(j), "accuracy": (tp + tn) / len(j), "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "F": 2 * tp / (2 * tp + fp + fn), "ground snow share": j.ground.mean(), "product snow share": j.sat.mean()}


example_sites = ["RC.095b", "RC.snotel", "RC.167b", "RC.163"]
live = pd.DataFrame({f"{s} ({sites.elevation_m[s]} m)": scores(daily_pairs(s)) for s in example_sites}).T
live.round(2)
'''),
("code", '''
pd.read_csv(ROOT / "validation" / "rcew_ground_daily_agreement.csv").set_index("subset")
'''),
("md", '''
The product rarely misses snow that the sensors record (recall), but it shows snow on more days than the 5 cm depth rule does (precision). Agreement is highest on days when MODIS saw the pixel and when its snow fraction was near 0 or 1, and lowest on gap-filled days and where the persistence ranking had to place a partial snow fraction. Those cases are identified by the `quality` flags in the daily file.

## 3.5 Held-out satellite maps (split-year cross-validation)

The 30 m pattern is checked against Landsat and Sentinel-2 snow maps with the F-measure of notebook 2, on cells in partly snow-covered MODIS pixels. To keep the test independent, the persistence ranking is built from odd water years when scoring scenes from even years, and the reverse (`ds_eval.py`). Rows compare the persistence ranking (`emp_smooth`) with the terrain score (`smooth`), fed either the true snow fraction from the 30 m map itself (`ideal`) or the MODIS fraction (`modis`); `F_random` is a random order.
'''),
("code", '''
cv = pd.read_csv(ROOT / "validation" / "rcew_downscale_crossval_split_years.csv")
cv["ranking"] = cv.kind.map({"emp_smooth": "snow persistence", "smooth": "terrain score"})
cv[["input", "ranking", "F", "F_open", "F_forest", "F_random", "n_scenes"]].set_index(["input", "ranking"])
'''),
("md", '''
## Summary

| check | result at Reynolds Creek |
|---|---|
| Melt-out against ground depth, all site-years | median bias near zero, mean absolute error about 18 days |
| Melt-out, one clear season in both records | mean absolute error about 11 days, r about 0.9 |
| Forest cells | melt-out about 9 days early |
| Daily snow presence, November-May | about 79% agreement; 85% on days MODIS saw the pixel |
| 30 m pattern against held-out Landsat and Sentinel-2 | F 0.73 with the persistence ranking, 0.69 with the terrain score, 0.65 random |

The tables printed above are the source for these numbers.
'''),
])

for name, n in [("01_preprocessing", nb1), ("02_downscaling", nb2), ("03_verification", nb3)]:
    nbf.write(n, os.path.join(out, name + ".ipynb"))
    print("wrote", name)
