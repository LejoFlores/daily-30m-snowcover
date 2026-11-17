"""Fetch daily MODIS/VIIRS snow-cover tiles (h09v04) from NSIDC cloud, cut a
window around Dry Creek / Bogus Basin, save per-day npz, delete the tile.

usage: python3 fetch_tiles.py SHORT_NAME VERSION START END OUTDIR [NWORKERS]
"""
import json, os, sys, subprocess, tempfile, urllib.request, urllib.parse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

BBOX = tuple(map(float, os.environ.get("TILE_BBOX", "-116.26,43.62,-116.00,43.84").split(",")))  # lon0, lat0, lon1, lat1 (padded)
R = 6371007.181
X0, Y0 = -20015109.354, 10007554.677
TILE = 1111950.5197665  # tile width (m)

short, ver, start, end, outdir = sys.argv[1:6]
nw = int(sys.argv[6]) if len(sys.argv) > 6 else 12
os.makedirs(outdir, exist_ok=True)
TOKEN = os.environ["EARTHDATA_TOKEN"]
h, v = 9, 4


def window(npix):
    ps = TILE / npix
    if os.environ.get("TILE_WINDOW") and npix == 2400:  # r0,r1,c0,c1 of the 500 m grid (domain.py MODIS_RC/SHAPE)
        r0, r1, c0, c1 = map(int, os.environ["TILE_WINDOW"].split(","))
        return r0, r1, c0, c1, X0 + h * TILE, Y0 - v * TILE, ps
    lon0, lat0, lon1, lat1 = BBOX
    xs, ys = [], []
    for lon in (lon0, lon1):
        for lat in (lat0, lat1):
            xs.append(R * np.radians(lon) * np.cos(np.radians(lat)))
            ys.append(R * np.radians(lat))
    ulx, uly = X0 + h * TILE, Y0 - v * TILE
    c0 = int(np.floor((min(xs) - ulx) / ps)); c1 = int(np.ceil((max(xs) - ulx) / ps))
    r0 = int(np.floor((uly - max(ys)) / ps)); r1 = int(np.ceil((uly - min(ys)) / ps))
    return r0, r1, c0, c1, ulx, uly, ps


def granules():
    out, page = [], 1
    while True:
        q = urllib.parse.urlencode(dict(short_name=short, version=ver, page_size=2000, page_num=page,
                                        temporal=f"{start}T00:00:00Z,{end}T23:59:59Z",
                                        bounding_box=",".join(map(str, BBOX))))
        d = json.load(urllib.request.urlopen("https://cmr.earthdata.nasa.gov/search/granules.json?" + q))
        ents = d["feed"]["entry"]
        for e in ents:
            if f"h{h:02d}v{v:02d}" not in e["title"]:
                continue
            hrefs = [l["href"] for l in e["links"] if l["href"].startswith("https") and l["href"].endswith((".hdf", ".h5"))]
            if hrefs:
                out.append(hrefs[0])
        if len(ents) < 2000:
            return out
        page += 1


def read(path):
    if path.endswith(".hdf"):
        from pyhdf.SD import SD
        f = SD(path)
        return {k: f.select(k)[:] for k in f.datasets()}
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.shape[0] > 1000:
                out[name.split("/")[-1]] = obj[:]
        f.visititems(visit)
    return out


def one(url):
    name = os.path.basename(url)
    doy = name.split(".")[1]  # AYYYYDDD
    dst = os.path.join(outdir, doy + ".npz")
    if os.path.exists(dst):
        return
    with tempfile.TemporaryDirectory(dir=outdir) as td:
        p = os.path.join(td, name)
        for attempt in range(4):
            r = subprocess.run(["curl", "-sSfL", "--retry", "3", "-H", f"Authorization: Bearer {TOKEN}", "-o", p, url])
            if r.returncode == 0:
                break
        else:
            print("FAIL", url, flush=True); return
        d = read(p)
        npix = next(iter(d.values())).shape[0]
        r0, r1, c0, c1, *_ = window(npix)
        np.savez_compressed(dst, **{k: a[r0:r1, c0:c1] for k, a in d.items()})


if __name__ == "__main__":
    urls = granules()
    print(short, len(urls), "granules", flush=True)
    for npix in (2400, 3000):
        print(npix, window(npix), flush=True)
    with ThreadPoolExecutor(nw) as ex:
        for i, _ in enumerate(ex.map(one, urls)):
            if i % 250 == 0:
                print(i, flush=True)
    print("done", flush=True)
