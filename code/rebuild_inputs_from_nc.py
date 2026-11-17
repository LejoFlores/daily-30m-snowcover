"""Rebuild the downscaling inputs (terrain pickle, persistence, forest) from the static layers of a seasons NetCDF,
so the daily product can be regenerated without refetching Landsat/Sentinel-2.
Layers are masked outside the watershed, so cells within 210 m of the edge (quality bit 16) rank in a truncated window
and can differ slightly from the original run; interior cells reproduce it (99.9% identical season dates).
usage: SNOW_DOMAIN=rcew python3 rebuild_inputs_from_nc.py SEASONS_NC OUT_DIR"""
import os, sys, pickle
import numpy as np
import netCDF4 as nc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downscale import Terrain

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
d = nc.Dataset(src)
g = lambda k: np.ma.filled(d[k][:].astype(float), np.nan)
G = dict(region=np.ma.filled(d["region"][:], 0).astype(int), elev=g("elevation"), slope=g("slope"),
         aspect=g("aspect"), lon=g("lon"), lat=g("lat"))
pickle.dump(Terrain(G), open(f"{out}/terrain_from_nc.pkl", "wb"))
np.savez_compressed(f"{out}/persistence_from_nc.npz", P_all=g("snow_persistence"))
np.save(f"{out}/forest_from_nc.npy", np.ma.filled(d["forest"][:], 0).astype(bool))
