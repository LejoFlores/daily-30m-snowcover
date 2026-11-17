"""Read cell time series from the daily snow-cover NetCDF, whether it is stored as one file
(<stem>.nc) or as parts along time (<stem>_partK_of_N.nc, see split_daily_nc.py)."""
import glob, os
import numpy as np
import pandas as pd
import netCDF4

T0 = pd.Timestamp("1999-10-01")


class Daily:
    def __init__(self, folder, stem):
        one = os.path.join(folder, stem + ".nc")
        paths = [one] if os.path.exists(one) else sorted(glob.glob(os.path.join(folder, stem + "_part*_of_*.nc")))
        if not paths:
            raise FileNotFoundError(f"no {stem}.nc or {stem}_part*.nc in {folder}")
        self.files = [netCDF4.Dataset(p) for p in paths]
        self.times = [T0 + pd.to_timedelta(np.asarray(f["time"][:]), "D") for f in self.files]

    def cell(self, var, iy, ix, start, end):
        """Daily values of `var` at cell (iy, ix) between two dates, as a float Series (255 -> NaN)."""
        out = []
        for f, t in zip(self.files, self.times):
            k = np.flatnonzero((t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)))
            if len(k):
                v = np.ma.filled(f[var][k[0]:k[-1] + 1, iy, ix], 255).astype(float)
                out.append(pd.Series(np.where(v == 255, np.nan, v), index=t[k[0]:k[-1] + 1]))
        return pd.concat(out) if out else pd.Series(dtype=float)
