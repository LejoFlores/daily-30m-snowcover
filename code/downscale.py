"""Downscale daily MODIS snow fraction to 30 m with the Walters et al. (2014, RSE 152:413)
physiographic terrain score, in two flavours:

  block  : as in Walters et al. -- within each MODIS pixel, the fSCA share of 30 m cells with
           the lowest terrain score is snow covered (hard quota per MODIS cell).
  smooth : fSCA is bilinearly interpolated between MODIS pixel centres to every 30 m cell and
           the terrain score is ranked inside a 15 x 15 cell (450 m) window centred on the cell;
           a cell is snow covered when its percentile rank <= its interpolated fSCA. This removes
           the seams along MODIS pixel edges.

Terrain score Ts = w * f_norm + (1 - w) * z_norm
  f_norm = f_sl / (f_sl,max - f_sl,min), f_sl = potential direct insolation on the slope /
           on a horizontal surface for the day (no atmosphere, no horizon shading; Swift 1976 style),
           extremes over the domain and season
  z_norm = (z_max,local - z) / z_range,max (local = MODIS cell for block, window for smooth)
fSCA from MOD10A1F CGF NDSI (0-100): fSCA = clip(-0.01 + 1.45 * NDSI/100, 0, 1)
(Salomonson & Appel 2004, the relation MOD10A1 C5 used).
"""
import numpy as np
import numba as nb

WIN = 7  # half width (cells) of the smooth ranking window -> 15 x 15
from domain import LAT, D as _DOM
R = 6371007.181; X0S, Y0S = -20015109.354, 10007554.677; TILE = 1111950.5197665
ULX, ULY = X0S + 9 * TILE, Y0S - 4 * TILE
PS = TILE / 2400
R0, C0 = _DOM["MODIS_RC"]  # MODIS window origin (fetch_tiles.py)


def ndsi_to_fsca(ndsi):
    return np.clip(-0.01 + 1.45 * ndsi / 100.0, 0, 1)


def slope_factor(slope_deg, aspect_deg, doy, lat=LAT, step_min=10):
    """f_sl = sum over daylight of max(0, n.s) / sum of s_up"""
    phi = np.radians(lat)
    dec = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
    om = np.radians(np.arange(-180, 180, step_min / 4.0))
    up = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(om)
    keep = up > 0
    om, up = om[keep], up[keep]
    east = -np.cos(dec) * np.sin(om)
    north = np.cos(phi) * np.sin(dec) - np.sin(phi) * np.cos(dec) * np.cos(om)
    b, a = np.radians(slope_deg), np.radians(aspect_deg)
    nx, ny, nz = np.sin(b) * np.sin(a), np.sin(b) * np.cos(a), np.cos(b)
    tot = np.zeros(slope_deg.shape)
    for e, n_, u in zip(east, north, up):
        tot += np.maximum(0, nx * e + ny * n_ + nz * u)
    return tot / up.sum()


@nb.njit(parallel=True, cache=True)
def window_prank(ts, valid, h):
    """percentile rank of each cell's score inside a (2h+1)^2 window (valid cells only)"""
    ny, nx = ts.shape
    out = np.full((ny, nx), np.nan)
    for i in nb.prange(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            le = 0; n = 0
            for di in range(-h, h + 1):
                ii = i + di
                if ii < 0 or ii >= ny:
                    continue
                for dj in range(-h, h + 1):
                    jj = j + dj
                    if jj < 0 or jj >= nx or not valid[ii, jj]:
                        continue
                    n += 1
                    if ts[ii, jj] <= ts[i, j]:
                        le += 1
            out[i, j] = le / n
    return out


@nb.njit(cache=True)
def window_zmax(z, valid, h):
    ny, nx = z.shape
    zmax = np.full((ny, nx), np.nan); zmin = np.full((ny, nx), np.nan)
    for i in range(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            mx = -1e9; mn = 1e9
            for ii in range(max(0, i - h), min(ny, i + h + 1)):
                for jj in range(max(0, j - h), min(nx, j + h + 1)):
                    if valid[ii, jj]:
                        mx = max(mx, z[ii, jj]); mn = min(mn, z[ii, jj])
            zmax[i, j] = mx; zmin[i, j] = mn
    return zmax, zmin


def block_prank(ts, cell_id, valid):
    """percentile rank within each MODIS cell (ties -> max rank)"""
    import pandas as pd
    out = np.full(ts.shape, np.nan)
    s = pd.Series(ts[valid])
    r = s.groupby(cell_id[valid]).rank(method="max", pct=True)
    out[valid] = r.values
    return out


class Terrain:
    def __init__(self, G):
        self.region = G["region"]; self.z = G["elev"]
        self.valid = np.isfinite(self.z) & np.isfinite(G["slope"])
        self.slope = np.where(self.valid, G["slope"], 0); self.aspect = np.where(self.valid, G["aspect"], 0)
        lon, lat = G["lon"], G["lat"]
        sx = R * np.radians(lon) * np.cos(np.radians(lat)); sy = R * np.radians(lat)
        self.fc = (sx - ULX) / PS - C0  # fractional MODIS col (pixel edges at integers)
        self.fr = (ULY - sy) / PS - R0
        self.mrow = np.floor(self.fr).astype(int); self.mcol = np.floor(self.fc).astype(int)
        self.cell_id = self.mrow * 1000 + self.mcol
        # distance (in 30 m cells) to the nearest MODIS pixel edge
        dr = np.minimum(self.fr - self.mrow, 1 - (self.fr - self.mrow))
        dc = np.minimum(self.fc - self.mcol, 1 - (self.fc - self.mcol))
        self.edge_dist_m = np.minimum(dr, dc) * PS
        dom = self.region > 0
        # elevation terms
        import pandas as pd
        zs = pd.Series(self.z[self.valid]); cid = self.cell_id[self.valid]
        zmax_c = zs.groupby(cid).transform("max").values; zmin_c = zs.groupby(cid).transform("min").values
        self.zmax_block = np.full(self.z.shape, np.nan); self.zmax_block[self.valid] = zmax_c
        zr_block = np.full(self.z.shape, np.nan); zr_block[self.valid] = zmax_c - zmin_c
        self.zR_block = np.nanmax(zr_block[dom])
        zmx, zmn = window_zmax(np.where(self.valid, self.z, 0.0), self.valid, WIN)
        self.zmax_win = zmx; self.zR_win = np.nanmax((zmx - zmn)[dom])
        self.znorm = dict(block=(self.zmax_block - self.z) / self.zR_block,
                          smooth=(self.zmax_win - self.z) / self.zR_win)
        # slope factor for 5-day DOY bins
        self.doys = np.arange(3, 366, 5)
        fsl = np.stack([slope_factor(self.slope, self.aspect, d) for d in self.doys])
        season = np.isin(self.doys, np.r_[1:213, 274:366])  # Oct-Jul
        sel = fsl[season][:, dom]
        self.fsl = fsl; self.fsl_rng = sel.max() - sel.min()

    def fnorm(self, doy):
        k = int(np.argmin(np.abs(self.doys - doy)))
        return self.fsl[k] / self.fsl_rng

    def set_persistence(self, P, name):
        """empirical score: low snow persistence -> snow-free first (ties broken by elevation)"""
        if not hasattr(self, "emp"):
            self.emp = {}
        zn = self.znorm["smooth"]
        self.emp[name] = np.where(np.isfinite(P), -P, np.nanmax(-P)) + 1e-3 * np.nan_to_num(zn)

    def score(self, doy, w, kind):
        if kind.startswith("emp"):
            return self.emp[self.emp_fold]
        return w * self.fnorm(doy) + (1 - w) * self.znorm[kind]

    def prank(self, doy, w, kind):
        ts = self.score(doy, w, kind)
        kind = kind.replace("emp_", "")
        if kind == "block":
            return block_prank(ts, self.cell_id, self.valid)
        return window_prank(np.where(self.valid, ts, 0.0), self.valid, WIN)

    def fsca_field(self, fsca_modis, kind):
        """fsca_modis: (54, 120) MODIS-grid fSCA (nan allowed) -> 30 m field"""
        kind = kind.replace("emp_", "")
        if kind == "block":
            r = np.clip(self.mrow, 0, fsca_modis.shape[0] - 1); c = np.clip(self.mcol, 0, fsca_modis.shape[1] - 1)
            return fsca_modis[r, c]
        # bilinear between pixel centres (centres at +0.5); renormalise weights over finite values
        y = self.fr - 0.5; x = self.fc - 0.5
        r0 = np.floor(y).astype(int); c0 = np.floor(x).astype(int)
        ty, tx = y - r0, x - c0
        acc = np.zeros(y.shape); wsum = np.zeros(y.shape)
        for dr, dc, wt in [(0, 0, (1 - ty) * (1 - tx)), (0, 1, (1 - ty) * tx), (1, 0, ty * (1 - tx)), (1, 1, ty * tx)]:
            rr = np.clip(r0 + dr, 0, fsca_modis.shape[0] - 1); cc = np.clip(c0 + dc, 0, fsca_modis.shape[1] - 1)
            v = fsca_modis[rr, cc]; ok = np.isfinite(v)
            acc += np.where(ok, v * wt, 0); wsum += np.where(ok, wt, 0)
        return np.where(wsum > 0.5, acc / np.maximum(wsum, 1e-9), np.nan)

    def snow(self, prank, fsca30):
        out = np.where(np.isfinite(fsca30) & np.isfinite(prank), (prank <= fsca30 + 1e-9) & (fsca30 > 0), False)
        return out, np.isfinite(fsca30) & self.valid
