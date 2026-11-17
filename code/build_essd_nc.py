"""ESSD release: daily 30 m snow cover with per-cell inputs and quality flags, plus per-WY seasonal metrics.

Re-runs the production downscaling (emp_smooth, hybrid persistence, P_all) day by day, keeping:
  snow_cover    uint8 0/1, 255 = missing
  input_fsca    uint8 percent, MOD10A1F fSCA interpolated to the 30 m cell (what the ranking had to place)
  cloud_age     uint8 days, MOD10A1F Cloud_Persistence at the cell's MODIS pixel (0 = observed that day)
  quality       uint8 bit flags (see attrs)
Static layers: elevation, slope, aspect, region, forest (terrain-model ranking), snow_persistence (ranking input).
Seasonal file: snow_on / snow_off (day of water year), snow_cover_days per WY, from ds_run output rasters.

usage: python3 build_essd_nc.py TERRAIN_PKL PERSIST_NPZ FOREST_NPY MOD_DIR SEASON_DIR OUT_DAILY_NC OUT_SEASON_NC
"""
import os, sys, glob, pickle, datetime as dt
import numpy as np
import pandas as pd
import rasterio
import netCDF4 as nc
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downscale import Terrain, ndsi_to_fsca  # noqa: F401 (Terrain needed for unpickling)

tpkl, pnpz, fnpy, mdir, sdir, out_daily, out_season = sys.argv[1:8]
T = pickle.load(open(tpkl, "rb"))
P = np.load(pnpz)["P_all"]
T.set_persistence(P, "all"); T.emp_fold = "all"
forest = np.load(fnpy)
dom = (T.region > 0) & T.valid
NY, NX = dom.shape
from domain import X0, Y1, NAME, D as DOM
RES = 30
x = X0 + (np.arange(NX) + 0.5) * RES; y = Y1 - (np.arange(NY) + 0.5) * RES
xx, yy = np.meshgrid(x, y)
lon, lat = Transformer.from_crs(32611, 4326, always_xy=True).transform(xx, yy)
prank = T.prank(1, 0, "emp_smooth")
mr = np.clip(T.mrow, 0, DOM["MODIS_SHAPE"][0] - 1); mc = np.clip(T.mcol, 0, DOM["MODIS_SHAPE"][1] - 1)

QDOC = ("bit 0 (1): input gap-filled, cloud_age >= 1 d; bit 1 (2): cloud_age 4-10 d; "
        "bit 2 (4): partial-snow input (5% < input_fsca < 95%), so the persistence ranking decided this cell; "
        "bit 3 (8): forest cell, ranking from the terrain model rather than observed snow persistence; "
        "bit 4 (16): within 210 m of the domain edge, where the 15 x 15 cell ranking window is truncated")

files = sorted(glob.glob(f"{mdir}/A*.npz"))
dates = pd.to_datetime([os.path.basename(f)[1:8] for f in files], format="%Y%j")
keep = dates <= pd.Timestamp("2026-09-30")
files = [f for f, k in zip(files, keep) if k]; dates = dates[keep]
t0 = dt.datetime(1999, 10, 1)


def common(ds):
    ds.createDimension("y", NY); ds.createDimension("x", NX)
    vx = ds.createVariable("x", "f8", ("x",)); vx[:] = x
    vx.setncatts(dict(standard_name="projection_x_coordinate", units="m", long_name="UTM 11N easting of cell centre"))
    vy = ds.createVariable("y", "f8", ("y",)); vy[:] = y
    vy.setncatts(dict(standard_name="projection_y_coordinate", units="m", long_name="UTM 11N northing of cell centre"))
    crs = ds.createVariable("crs", "i4")
    crs.setncatts(dict(grid_mapping_name="transverse_mercator", longitude_of_central_meridian=-117.0,
                       latitude_of_projection_origin=0.0, scale_factor_at_central_meridian=0.9996,
                       false_easting=500000.0, false_northing=0.0, semi_major_axis=6378137.0,
                       inverse_flattening=298.257223563, epsg_code="EPSG:32611",
                       GeoTransform=f"{X0} {RES} 0 {Y1} 0 {-RES}"))
    for name, arr, att in [("lat", lat, dict(standard_name="latitude", units="degrees_north")),
                           ("lon", lon, dict(standard_name="longitude", units="degrees_east"))]:
        v = ds.createVariable(name, "f4", ("y", "x"), zlib=True); v[:] = arr; v.setncatts(att)
    stat = [("region", np.where(dom, T.region, 0).astype("u1"), "u1",
             dict(long_name="domain", flag_values=np.array([0] + sorted(DOM["REGIONS"]), "u1"),
                  flag_meanings=" ".join(["outside"] + [DOM["REGIONS"][k] for k in sorted(DOM["REGIONS"])]))),
            ("elevation", np.where(dom, T.z, np.nan).astype("f4"), "f4",
             dict(long_name="cell mean elevation (tRIBS 10 m DEM aggregated to 30 m)", units="m")),
            ("slope", np.where(dom, T.slope, np.nan).astype("f4"), "f4", dict(long_name="terrain slope", units="degree")),
            ("aspect", np.where(dom, T.aspect, np.nan).astype("f4"), "f4",
             dict(long_name="terrain aspect, clockwise from north", units="degree")),
            ("forest", np.where(dom, forest, 0).astype("u1"), "u1",
             dict(long_name="forest flag (summer Landsat NDVI >= 0.45); ranking from terrain model",
                  flag_values=np.array([0, 1], "u1"), flag_meanings="open forest")),
            ("snow_persistence", np.where(dom, P, np.nan).astype("f4"), "f4",
             dict(long_name="share of clear Feb-Jun Landsat/Sentinel-2 views that are snow (forest: terrain-model "
                            "prediction); the ranking used for downscaling", units="1"))]
    for name, arr, typ, att in stat:
        fv = np.float32(np.nan) if typ == "f4" else None
        v = ds.createVariable(name, typ, ("y", "x"), zlib=True, fill_value=fv)
        v[:] = arr; att["grid_mapping"] = "crs"; att["coordinates"] = "lat lon"; v.setncatts(att)


INFO = {
    "dcew": dict(name="Dry Creek Experimental Watershed and Bogus Basin",
                 area="the Dry Creek Experimental Watershed and an adjacent Bogus Basin box (Boise Front, Idaho, USA)", s2="11TNJ"),
    "rcew": dict(name="Reynolds Creek Experimental Watershed",
                 area="the Reynolds Creek Experimental Watershed (Owyhee Mountains, Idaho, USA)", s2="11TNH"),
}[NAME]


def gattrs(ds, title, extra):
    ds.setncatts(dict(
        title=title, Conventions="CF-1.8, ACDD-1.3", product_version="1.0-draft",
        summary=(f"Daily 30 m binary snow cover for {INFO['area']}, made by downscaling NASA MOD10A1F v61 cloud-gap-filled NDSI snow cover "
                 "(500 m) with an empirical snow-persistence ranking from Landsat and Sentinel-2."),
        source="MOD10A1F v61 (NSIDC); Landsat 4-9 Collection 2 Level-2 surface reflectance, Tier 1 (Microsoft Planetary Computer); Sentinel-2 L2A COGs (sentinel-cogs, tile " + INFO["s2"] + ")",
        geospatial_lat_min=float(lat[dom].min()), geospatial_lat_max=float(lat[dom].max()),
        geospatial_lon_min=float(lon[dom].min()), geospatial_lon_max=float(lon[dom].max()),
        crs="EPSG:32611", spatial_resolution="30 m", license="CC-BY-4.0 (proposed)",
        creator_name="Lejo Flores (Boise State University)",
        history=f"{dt.date.today()} build_essd_nc.py", **extra))


# ---------------- daily ----------------
def build_daily():
    D = nc.Dataset(out_daily, "w", format="NETCDF4")
    common(D)
    D.createDimension("time", None)
    vt = D.createVariable("time", "i4", ("time",))
    vt.setncatts(dict(standard_name="time", units="days since 1999-10-01 00:00:00", calendar="standard",
                      long_name="calendar day of the MODIS Terra overpass (~10:30 local solar time)"))
    ch = (16, NY, NX)
    def dv(name, att, fill=255):
        v = D.createVariable(name, "u1", ("time", "y", "x"), zlib=True, complevel=5, shuffle=True, chunksizes=ch, fill_value=np.uint8(fill))
        att.update(grid_mapping="crs", coordinates="lat lon"); v.setncatts(att); return v
    vs = dv("snow_cover", dict(long_name="binary snow cover", flag_values=np.array([0, 1], "u1"), flag_meanings="snow_free snow",
                               comment="255 = missing: no MODIS input, cloud_age > 10 d, or outside domain"))
    vf = dv("input_fsca", dict(long_name="MOD10A1F fractional snow cover interpolated to the cell (bilinear between pixel centres)",
                               units="percent", valid_range=np.array([0, 100], "u1")))
    vc = dv("cloud_age", dict(long_name="MOD10A1F Cloud_Persistence at the cell's MODIS pixel: consecutive days without a clear view (0 = observed that day in MOD10A1)",
                              units="day"))
    vq = dv("quality", dict(long_name="quality bit flags", flag_masks=np.array([1, 2, 4, 8, 16], "u1"),
                            flag_meanings="gap_filled cloud_age_4_to_10d partial_snow_input forest_terrain_ranking near_domain_edge",
                            comment=QDOC), fill=0)
    iy, ix = np.nonzero(dom)
    # edge flag: within 1 km of domain boundary (cells whose 15x15 ranking window is truncated)
    from scipy.ndimage import distance_transform_edt
    edge = (distance_transform_edt(dom) * RES) <= 210
    static_q = (forest.astype("u1") * 8) | (edge.astype("u1") * 16)
    buf = []
    def flush():
        if not buf:
            return
        n0 = len(vt); n = len(buf)
        vt[n0:n0 + n] = [b[0] for b in buf]
        vs[n0:n0 + n] = np.stack([b[1] for b in buf]); vf[n0:n0 + n] = np.stack([b[2] for b in buf])
        vc[n0:n0 + n] = np.stack([b[3] for b in buf]); vq[n0:n0 + n] = np.stack([b[4] for b in buf])
        buf.clear()
    stats = dict(n=0, valid=0, partial=0, gap=0)
    for f, d in zip(files, dates):
        z = np.load(f)
        nd = z["CGF_NDSI_Snow_Cover"].astype(float); cp = z["Cloud_Persistence"].astype(float)
        fm = np.where((nd <= 100) & (cp <= 10), ndsi_to_fsca(nd), np.nan)
        f30 = T.fsca_field(fm, "emp_smooth")
        sn, ok = T.snow(prank, f30)
        ok &= dom
        raw = z["MOD10A1_NDSI_Snow_Cover"]
        clear = (raw <= 100)[mr, mc]
        cage = np.where(nd[mr, mc] <= 100, np.where(clear, 0, np.maximum(cp[mr, mc], 1)), np.nan)
        s = np.full((NY, NX), 255, "u1"); s[ok] = sn[ok]
        fp = np.full((NY, NX), 255, "u1"); g = dom & np.isfinite(f30); fp[g] = np.round(np.clip(f30[g], 0, 1) * 100)
        ca = np.full((NY, NX), 255, "u1"); g = dom & np.isfinite(cage); ca[g] = np.clip(cage[g], 0, 254)
        q = np.where(dom, static_q, 0).astype("u1")
        q |= (dom & np.isfinite(cage) & (cage >= 1)).astype("u1")
        q |= ((dom & np.isfinite(cage) & (cage >= 4)).astype("u1") * 2)
        part = dom & np.isfinite(f30) & (f30 > 0.05) & (f30 < 0.95)
        q |= part.astype("u1") * 4
        buf.append(((d - t0).days, s, fp, ca, q))
        stats["n"] += 1; stats["valid"] += ok[dom].mean(); stats["partial"] += part[dom].mean()
        if len(buf) == 64:
            flush()
    flush()
    gattrs(D, f"Daily 30 m snow cover, {INFO['name']}, Idaho, WY2000-2026",
           dict(time_coverage_start=str(dates[0].date()), time_coverage_end=str(dates[-1].date())))
    D.close()
    print("days", stats["n"], "mean valid frac", round(stats["valid"] / stats["n"], 3), "mean partial frac", round(stats["partial"] / stats["n"], 3))



if not os.environ.get("SKIP_DAILY"):
    build_daily()

# ---------------- seasonal ----------------
S = nc.Dataset(out_season, "w", format="NETCDF4")
common(S)
S.createDimension("water_year", 27)
w = S.createVariable("water_year", "i2", ("water_year",)); w[:] = np.arange(2000, 2027)
w.setncatts(dict(long_name="water year (1 Oct of previous year to 30 Sep)"))
for key, fn, att in [("snow_on", "*snow_on_dowy*.tif",
                      dict(long_name="first day of the main snow season (longest run >= 14 d, gaps <= 3 d bridged)", units="day of water year (1 = 1 Oct)")),
                     ("snow_off", "*snow_off_dowy*.tif",
                      dict(long_name="first snow-free day after the main snow season (melt-out)", units="day of water year (1 = 1 Oct)")),
                     ("snow_cover_days", "*scd*.tif", dict(long_name="number of snow-covered days in the water year", units="day"))]:
    with rasterio.open(glob.glob(os.path.join(sdir, fn))[0]) as r:
        a = r.read().astype("f4")
    a[:, ~dom] = np.nan
    v = S.createVariable(key, "i2", ("water_year", "y", "x"), zlib=True, fill_value=np.int16(-1))
    v[:] = np.where(np.isfinite(a), a, -1).astype("i2")
    att.update(grid_mapping="crs", coordinates="lat lon",
               comment="-1 / missing: no main season that year (intermittent snow), or unobserved (WY2000 snow_on: MODIS starts 24 Feb 2000)")
    v.setncatts(att)
gattrs(S, f"Annual snow-on, snow-off and snow-cover days at 30 m, {INFO['name']}, WY2000-2026", {})
S.close()
print("done")
