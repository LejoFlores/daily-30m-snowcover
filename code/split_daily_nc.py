"""Split a daily snow-cover NetCDF into N parts along time (same variables, attributes and compression),
for storage systems with a per-file size limit. Static layers are repeated in every part.
usage: python3 split_daily_nc.py DAILY_NC OUT_DIR [NPARTS]   -> <name>_partK_of_N.nc"""
import os, sys
import numpy as np
import netCDF4 as nc

src_path, out_dir = sys.argv[1:3]
nparts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
os.makedirs(out_dir, exist_ok=True)
src = nc.Dataset(src_path)
nt = len(src.dimensions["time"])
edges = np.linspace(0, nt, nparts + 1).astype(int)
stem = os.path.basename(src_path)[:-3]
for p in range(nparts):
    a, b = edges[p], edges[p + 1]
    dst = nc.Dataset(os.path.join(out_dir, f"{stem}_part{p + 1}_of_{nparts}.nc"), "w", format="NETCDF4")
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    dst.setncattr("part", f"{p + 1} of {nparts}: time steps {a} to {b - 1} of {os.path.basename(src_path)}")
    for name, dim in src.dimensions.items():
        dst.createDimension(name, None if name == "time" else len(dim))
    for name, v in src.variables.items():
        fill = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
        f = v.filters() or {}
        ch = v.chunking()
        o = dst.createVariable(name, v.dtype, v.dimensions, zlib=f.get("zlib", False), complevel=f.get("complevel", 4),
                               shuffle=f.get("shuffle", False), fill_value=fill,
                               chunksizes=ch if isinstance(ch, list) and "time" in v.dimensions else None)
        o.set_auto_maskandscale(False); v.set_auto_maskandscale(False)
        o.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
        if "time" not in v.dimensions:
            o[...] = v[...]
        elif v.ndim == 1:
            o[:] = v[a:b]
        else:
            for i in range(a, b, 64):
                j = min(i + 64, b)
                o[i - a:j - a] = v[i:j]
    dst.close()
    print("part", p + 1, a, b, flush=True)
