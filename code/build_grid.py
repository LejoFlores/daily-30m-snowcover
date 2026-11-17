"""Analysis domain and terrain attributes.

30 m grid (same as fetch_landsat/fetch_s2): elevation, slope, aspect, northness,
region (1 = Dry Creek watershed, 2 = Bogus Basin box outside Dry Creek, 0 = other).
MODIS 500 m and VIIRS 375 m pixels in the fetch windows get the area-weighted
(30 m cell count) region shares and terrain means.
usage: python3 build_grid.py DEM_BOX WATERSHED_GPKG OUT_NPZ
"""
import sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from shapely.geometry import box
from pyproj import Transformer

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain import X0, Y1, X1, Y0, RES, NX, NY, NAME, D as DOM
T = rasterio.transform.from_origin(X0, Y1, RES, RES)
BOGUS_LL = (-116.13, 43.753, -116.07, 43.790)  # lon0, lat0, lon1, lat1

dem_path, ws_path, out = sys.argv[1:4]

with rasterio.open(dem_path) as r:
    z10 = r.read(1).astype(np.float64)
    z10[z10 == r.nodata] = np.nan if r.nodata is not None else z10[0, 0]
    z = np.full((NY, NX), np.nan)
    reproject(z10, z, src_transform=r.transform, src_crs=r.crs, dst_transform=T, dst_crs="EPSG:32611",
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)

gy, gx = np.gradient(z, RES)  # gy along rows (southward), gx eastward
dzdx, dzdy = gx, -gy          # dz/dnorth
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
aspect = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360) % 360  # downslope direction, 0 = north
northness = np.cos(np.radians(aspect)) * np.sin(np.radians(slope))

if ws_path.endswith(".tif"):  # watershed mask raster (1 = inside)
    with rasterio.open(ws_path) as r:
        mk = np.zeros((NY, NX), np.float32)
        reproject(r.read(1).astype(np.float32), mk, src_transform=r.transform, src_crs=r.crs, dst_transform=T,
                  dst_crs="EPSG:32611", resampling=Resampling.average, src_nodata=r.nodata, dst_nodata=0)
    dc = mk >= 0.5
else:
    ws = gpd.read_file(ws_path).to_crs(32611)
    dc = rasterize([(g, 1) for g in ws.geometry], out_shape=(NY, NX), transform=T, fill=0).astype(bool)
tr_ll = Transformer.from_crs(4326, 32611, always_xy=True)
bx = gpd.GeoSeries([box(*BOGUS_LL)], crs=4326).to_crs(32611)
bg = rasterize([(bx.iloc[0], 1)], out_shape=(NY, NX), transform=T, fill=0).astype(bool) & ~dc
if NAME != "dcew":
    bg[:] = False
region = np.where(dc, 1, np.where(bg, 2, 0)).astype(np.uint8)

cols, rows = np.meshgrid(np.arange(NX), np.arange(NY))
xc, yc = X0 + (cols + 0.5) * RES, Y1 - (rows + 0.5) * RES
lon, lat = Transformer.from_crs(32611, 4326, always_xy=True).transform(xc, yc)

# MODIS / VIIRS sinusoidal pixel index of each 30 m cell (windows from fetch_tiles.py)
R = 6371007.181
X0S, Y0S = -20015109.354, 10007554.677
TILE = 1111950.5197665
sx = R * np.radians(lon) * np.cos(np.radians(lat))
sy = R * np.radians(lat)
ulx, uly = X0S + 9 * TILE, Y0S - 4 * TILE
res = dict(region=region, elev=z, slope=slope, aspect=aspect, northness=northness, lon=lon, lat=lat)
for tag, npix, (r0, c0), shp in [("modis", 2400, DOM["MODIS_RC"], DOM["MODIS_SHAPE"]), ("viirs", 3000, DOM["VIIRS_RC"], DOM["VIIRS_SHAPE"])]:
    ps = TILE / npix
    pc = np.floor((sx - ulx) / ps).astype(int) - c0
    pr = np.floor((uly - sy) / ps).astype(int) - r0
    res[f"{tag}_row"], res[f"{tag}_col"] = pr, pc
    for name, mask in [("dc", region == 1), ("bogus", region == 2)]:
        cnt = np.zeros(shp)
        np.add.at(cnt, (pr[mask], pc[mask]), 1)
        res[f"{tag}_n_{name}"] = cnt
    ok = np.isfinite(z)
    tot = np.zeros(shp); np.add.at(tot, (pr[ok], pc[ok]), 1)
    for v in ["elev", "northness", "slope"]:
        s = np.zeros(shp); np.add.at(s, (pr[ok], pc[ok]), res[v][ok])
        res[f"{tag}_{v}"] = np.where(tot > 0, s / np.maximum(tot, 1), np.nan)
    res[f"{tag}_ncell"] = tot
    # pixel centre lon/lat
    cc, rr = np.meshgrid(np.arange(shp[1]) + c0 + 0.5, np.arange(shp[0]) + r0 + 0.5)
    py_ = uly - rr * ps; px_ = ulx + cc * ps
    plat = np.degrees(py_ / R); plon = np.degrees(px_ / (R * np.cos(np.radians(plat))))
    res[f"{tag}_lat"], res[f"{tag}_lon"] = plat, plon

np.savez_compressed(out, **res)
for tag, ps in [("modis", 463.3), ("viirs", 370.7)]:
    full = (res[f"{tag}_n_dc"] * 900 > 0.5 * ps * ps).sum()
    anyc = (res[f"{tag}_n_dc"] > 0).sum()
    fb = (res[f"{tag}_n_bogus"] * 900 > 0.5 * ps * ps).sum()
    print(tag, "pixels >50% in Dry Creek:", full, "touching:", anyc, " >50% Bogus box:", fb)
print("30 m cells Dry Creek", (region == 1).sum(), "Bogus", (region == 2).sum(),
      "elev DC", np.nanmin(z[region == 1]), np.nanmax(z[region == 1]),
      "Bogus", np.nanmin(z[region == 2]), np.nanmax(z[region == 2]))
