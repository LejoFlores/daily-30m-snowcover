"""Windowed reads of Landsat Collection 2 Level-2 surface reflectance (Landsat 4-9) from Microsoft
Planetary Computer, on a 30 m UTM 11N grid. One npz per scene: green, red, nir, swir1 (SR x 10000, int16,
-32768 = fill) and qa_pixel (uint16, C2 QA_PIXEL bits).

usage: python3 fetch_landsat_c2.py OUTDIR X0 Y1 X1 Y0 [NWORKERS] [START END]
  (grid corners in EPSG:32611 metres; START/END as YYYY-MM-DD, default 1984-01-01 2026-09-30)
"""
import os, sys, json, time, urllib.request
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from concurrent.futures import ProcessPoolExecutor

if os.path.exists("/root/.ccr/ca-bundle.crt"):  # proxy CA bundle of the cloud container this was developed in
    os.environ.setdefault("GDAL_CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.TIF"
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"; os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
os.environ["GDAL_HTTP_TIMEOUT"] = "60"

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
outdir = sys.argv[1]
X0, Y1, X1, Y0 = map(float, sys.argv[2:6])
nw = int(sys.argv[6]) if len(sys.argv) > 6 else 8
start, end = (sys.argv[7], sys.argv[8]) if len(sys.argv) > 8 else ("1984-01-01", "2026-09-30")
RES = 30
NX, NY = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
DST_T = rasterio.transform.from_origin(X0, Y1, RES, RES)
BANDS = dict(green="green", red="red", nir="nir08", swir1="swir16")
os.makedirs(outdir, exist_ok=True)
_tok = {"t": None, "exp": 0}


def token():
    if _tok["t"] is None or time.time() > _tok["exp"]:
        for a in range(6):
            try:
                _tok["t"] = json.load(urllib.request.urlopen(TOKEN_URL, timeout=30))["token"]
                _tok["exp"] = time.time() + 1800
                break
            except Exception:
                time.sleep(3 + 3 * a)
    return _tok["t"]


def search():
    bbox = transform_bounds("EPSG:32611", "EPSG:4326", X0, Y0, X1, Y1)
    body = {"collections": ["landsat-c2-l2"], "bbox": list(bbox), "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": 500}
    items, url = [], STAC
    while url:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        for a in range(6):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=60)); break
            except Exception:
                time.sleep(3 + 3 * a)
        items += r["features"]
        nxt = [l for l in r.get("links", []) if l.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    return items


def warp(href, resamp, dtype, nodata):
    with rasterio.open("/vsicurl/" + href + "?" + token()) as r:
        w = from_bounds(X0 - 200, Y0 - 200, X1 + 200, Y1 + 200, r.transform).round_offsets().round_lengths()
        a = r.read(1, window=w, boundless=True, fill_value=nodata)
        out = np.full((NY, NX), nodata, dtype)
        reproject(a, out, src_transform=r.window_transform(w), src_crs=r.crs, dst_transform=DST_T, dst_crs="EPSG:32611",
                  resampling=resamp, src_nodata=nodata, dst_nodata=nodata)
        return out


def one(it):
    name = it["id"]; dst = os.path.join(outdir, name + ".npz")
    if os.path.exists(dst):
        return "skip"
    pr = it["properties"]
    for a in range(4):
        try:
            out = {}
            for k, b in BANDS.items():
                dn = warp(it["assets"][b]["href"], Resampling.average, np.float32, 0).astype(np.float32)
                sr = np.where(dn > 0, (dn * 0.0000275 - 0.2) * 10000, np.nan)
                out[k] = np.where(np.isfinite(sr), np.clip(sr, -10000, 30000), -32768).astype(np.int16)
            out["qa_pixel"] = warp(it["assets"]["qa_pixel"]["href"], Resampling.nearest, np.uint16, 1)
            plat = pr["platform"]; sensor = {"landsat-4": "LT04", "landsat-5": "LT05", "landsat-7": "LE07",
                                             "landsat-8": "LC08", "landsat-9": "LC09"}[plat]
            if (out["green"] == -32768).all():
                return "empty"
            np.savez_compressed(dst, **out, sensor=sensor, date=pr["datetime"][:10], time=pr["datetime"],
                                sun_elev=pr.get("view:sun_elevation", np.nan), cloud=pr.get("eo:cloud_cover", np.nan),
                                path=pr.get("landsat:wrs_path", ""), row=pr.get("landsat:wrs_row", ""),
                                tier=pr.get("landsat:collection_category", ""))
            return "ok"
        except Exception as e:
            err = repr(e)[:200]; time.sleep(3 + 4 * a)
    print("FAIL", name, err, flush=True)
    return "fail"


if __name__ == "__main__":
    items = search()
    # keep T1/T2 L2 SR scenes; drop duplicates of the same sensor+date (adjacent rows) keeping the first
    seen, keep = set(), []
    for it in sorted(items, key=lambda i: i["id"]):
        pr = it["properties"]
        k = (pr["platform"], pr["datetime"][:10])
        if k in seen or "green" not in it["assets"]:
            continue
        if os.environ.get("LS_T1_ONLY") and pr.get("landsat:collection_category") != "T1":
            continue
        seen.add(k); keep.append(it)
    print(len(items), "items", len(keep), "scenes", flush=True)
    res = {}
    with ProcessPoolExecutor(nw) as ex:
        for i, r in enumerate(ex.map(one, keep, chunksize=4)):
            res[r] = res.get(r, 0) + 1
            if i % 200 == 0:
                print(i, res, flush=True)
    print("done", res, flush=True)
