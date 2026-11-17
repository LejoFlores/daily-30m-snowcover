"""Windowed reads of Landsat Collection 1 L1 scenes (GCP public bucket) over
Dry Creek / Bogus Basin. Saves TOA green, red, nir, swir1 (x10000, int16) and
the BQA band on a fixed 30 m UTM 11N grid, one npz per scene.

usage: python3 fetch_landsat.py index.csv OUTDIR [NWORKERS]
"""
import os, sys, re, urllib.request
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from concurrent.futures import ThreadPoolExecutor

if os.path.exists("/root/.ccr/ca-bundle.crt"):  # proxy CA bundle of the cloud container this was developed in
    os.environ.setdefault("GDAL_CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".TIF"

# common analysis grid, EPSG:32611, 30 m
X0, Y1, X1, Y0 = 564480, 4849500, 577020, 4836000
RES = 30
NX, NY = (X1 - X0) // RES, (Y1 - Y0) // RES
DST_T = from_origin(X0, Y1, RES, RES)
BANDS = {  # name -> band number per sensor
    "LC08": dict(green=3, red=4, nir=5, swir1=6),
    "LE07": dict(green=2, red=3, nir=4, swir1=5),
    "LT05": dict(green=2, red=3, nir=4, swir1=5),
    "LT04": dict(green=2, red=3, nir=4, swir1=5),
}

idx, outdir = sys.argv[1], sys.argv[2]
nw = int(sys.argv[3]) if len(sys.argv) > 3 else 8
os.makedirs(outdir, exist_ok=True)


def mtl(base):
    t = urllib.request.urlopen(base + "_MTL.txt").read().decode()
    kv = dict(re.findall(r"^\s*(\w+)\s*=\s*\"?([^\"\n]*)\"?\s*$", t, re.M))
    return kv


def warp(url, resamp):
    with rasterio.open("/vsicurl/" + url) as r:
        # read only the needed window, then reproject to the common grid
        from rasterio.windows import from_bounds
        w = from_bounds(X0 - 300, Y0 - 300, X1 + 300, Y1 + 300, r.transform).round_offsets().round_lengths()
        a = r.read(1, window=w, boundless=True, fill_value=0)
        st = r.window_transform(w)
        out = np.zeros((NY, NX), a.dtype)
        reproject(a, out, src_transform=st, src_crs=r.crs, dst_transform=DST_T, dst_crs="EPSG:32611",
                  resampling=resamp, src_nodata=0, dst_nodata=0)
        return out


def one(row):
    pid = row.PRODUCT_ID
    dst = os.path.join(outdir, pid + ".npz")
    if os.path.exists(dst):
        return
    base = row.BASE_URL.replace("gs://", "https://storage.googleapis.com/") + "/" + pid
    try:
        kv = mtl(base)
        sensor = pid[:4]
        sun = float(kv["SUN_ELEVATION"])
        out = {}
        for name, b in BANDS[sensor].items():
            dn = warp(f"{base}_B{b}.TIF", Resampling.nearest).astype(np.float32)
            m, a = float(kv[f"REFLECTANCE_MULT_BAND_{b}"]), float(kv[f"REFLECTANCE_ADD_BAND_{b}"])
            rho = (m * dn + a) / np.sin(np.radians(sun))
            rho[dn == 0] = np.nan
            out[name] = np.where(np.isfinite(rho), np.clip(rho, -1, 3) * 10000, -32768).astype(np.int16)
        out["bqa"] = warp(f"{base}_BQA.TIF", Resampling.nearest)
        np.savez_compressed(dst, **out, sun_elev=sun, sensor=sensor,
                            time=kv.get("SCENE_CENTER_TIME", ""), date=kv["DATE_ACQUIRED"])
    except Exception as e:
        print("FAIL", pid, repr(e)[:200], flush=True)


if __name__ == "__main__":
    cols = "SCENE_ID,PRODUCT_ID,SPACECRAFT_ID,SENSOR_ID,DATE_ACQUIRED,COLLECTION_NUMBER,COLLECTION_CATEGORY,SENSING_TIME,DATA_TYPE,WRS_PATH,WRS_ROW,CLOUD_COVER,NORTH_LAT,SOUTH_LAT,WEST_LON,EAST_LON,TOTAL_SIZE,BASE_URL".split(",")
    df = pd.read_csv(idx, names=cols)
    df = df[df.WRS_PATH.isin([41, 42]) & (df.COLLECTION_CATEGORY == "T1") &
            df.PRODUCT_ID.str[:4].isin(list(BANDS)) & (df.DATE_ACQUIRED >= "1984-10-01")]
    df = df.assign(key=df.PRODUCT_ID.str[:4] + df.PRODUCT_ID.str[10:25]).sort_values("PRODUCT_ID").drop_duplicates("key", keep="last").sort_values("DATE_ACQUIRED")
    print(len(df), "scenes", flush=True)
    with ThreadPoolExecutor(nw) as ex:
        for i, _ in enumerate(ex.map(one, df.itertuples())):
            if i % 100 == 0:
                print(i, flush=True)
    print("done", flush=True)
