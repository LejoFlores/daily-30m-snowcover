"""Windowed reads of Sentinel-2 L2A COGs (s3://sentinel-cogs, tile 11TNJ) on the
same 30 m UTM 11N grid as fetch_landsat.py. Saves BOA green, red, nir (B8A),
swir1 (B11) x10000 (int16) and SCL, one npz per scene.

usage: python3 fetch_s2.py OUTDIR START_YEAR END_YEAR [NWORKERS]
"""
import os, sys, re, json, urllib.request
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.exists("/root/.ccr/ca-bundle.crt"):  # proxy CA bundle of the cloud container this was developed in
    os.environ.setdefault("GDAL_CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["CPL_VSIL_CURL_USE_HEAD"] = "NO"
os.environ["GDAL_HTTP_MAX_RETRY"] = "4"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"

os.environ["GDAL_HTTP_MAX_RETRY"] = "8"
os.environ["GDAL_HTTP_TIMEOUT"] = "60"
os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = "20"

X0, Y1, X1, Y0 = map(float, os.environ.get("S2_GRID", "564480,4849500,577020,4836000").split(","))
RES = 30
NX, NY = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
DST_T = rasterio.transform.from_origin(X0, Y1, RES, RES)
BUCKET = "https://sentinel-cogs.s3.amazonaws.com/"
TILE = "sentinel-s2-l2a-cogs/11/T/%s/" % os.environ.get("S2_TILE", "NJ")
BANDS = dict(green="B03", red="B04", nir="B8A", swir1="B11")

outdir, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
nw = int(sys.argv[4]) if len(sys.argv) > 4 else 8
os.makedirs(outdir, exist_ok=True)


def ls(prefix):
    import time
    for attempt in range(8):
        try:
            x = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={prefix}&delimiter=/").read().decode()
            break
        except Exception:
            time.sleep(2 + 2 * attempt)
    return re.findall(r"<Prefix>([^<]*)</Prefix>", x)[1:]


def warp(url, resamp, dtype):
    with rasterio.open("/vsicurl/" + url) as r:
        w = from_bounds(X0 - 200, Y0 - 200, X1 + 200, Y1 + 200, r.transform).round_offsets().round_lengths()
        a = r.read(1, window=w, boundless=True, fill_value=0).astype(dtype)
        out = np.zeros((NY, NX), dtype)
        reproject(a, out, src_transform=r.window_transform(w), src_crs=r.crs, dst_transform=DST_T,
                  dst_crs="EPSG:32611", resampling=resamp, src_nodata=0, dst_nodata=0)
        return out


BUCKETS = ["https://s3.us-west-2.amazonaws.com/sentinel-cogs/", "https://sentinel-cogs.s3.us-west-2.amazonaws.com/",
           "https://sentinel-cogs.s3.amazonaws.com/"]


def one(prefix):
    import time
    global BUCKET
    for attempt in range(6):
        BUCKET = BUCKETS[attempt % 3]
        if _one(prefix) or attempt == 5:
            return
        time.sleep(2 + 3 * attempt)


def _one(prefix):
    name = prefix.rstrip("/").split("/")[-1]
    dst = os.path.join(outdir, name + ".npz")
    if os.path.exists(dst):
        return True
    try:
        meta = json.load(urllib.request.urlopen(BUCKET + prefix + name + ".json"))
        pr = meta["properties"]
        base = float(pr.get("s2:processing_baseline", "0") or 0)
        sub = 1000 if (base >= 4 and not pr.get("earthsearch:boa_offset_applied", False)) else 0
        out = {}
        for k, b in BANDS.items():
            a = warp(BUCKET + prefix + b + ".tif", Resampling.average, np.float32)
            v = np.where(a > 0, a - sub, np.nan)
            out[k] = np.where(np.isfinite(v), np.clip(v, -10000, 30000), -32768).astype(np.int16)
        out["scl"] = warp(BUCKET + prefix + "SCL.tif", Resampling.mode, np.uint8)
        np.savez_compressed(dst, **out, date=pr["datetime"][:10], datetime=pr["datetime"],
                            cloud=pr.get("eo:cloud_cover", np.nan), baseline=base, offset_sub=sub)
        return True
    except Exception as e:
        print("FAIL", name, repr(e)[:200], flush=True)
        return False


if __name__ == "__main__":
    scenes = []
    for y in range(y0, y1 + 1):
        for m in ls(f"{TILE}{y}/"):
            scenes += ls(m)
    if os.environ.get("S2_REVERSE"):
        scenes = scenes[::-1]
    print(len(scenes), "scenes", flush=True)
    with ThreadPoolExecutor(nw) as ex:
        for i, _ in enumerate(ex.map(one, scenes, chunksize=4)):
            if i % 100 == 0:
                print(i, flush=True)
    print("done", flush=True)
