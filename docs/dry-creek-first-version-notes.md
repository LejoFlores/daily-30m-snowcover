# snowcover/ — satellite snow-on and snow-off dates, Dry Creek and Bogus Basin (2026-10-08)

Built in the "Bogus Basin snow on/off" thread at Lejo's request, and meant for checking the timing of tRIBS snow in Dry Creek.
The domain is the Dry Creek Experimental Watershed (27 km², 1,035–2,136 m) plus a Bogus Basin box (lon -116.13 to -116.07, lat 43.753 to 43.790, outside Dry Creek, 1,345–2,309 m), which holds the ski area and the Bogus Basin SNOTEL.
Dates are reported per water year, as day of water year (1 = Oct 1) and as calendar date.

## What to use

| need | file | notes |
|---|---|---|
| **Dates by elevation and aspect band for tRIBS** | `tables/tribs_snow_timing_by_band.csv` | One row per region × 200 m band × aspect (N/E/S/W/flat/all) × WY2000–2026. Main columns come from daily MODIS downscaled to 30 m. `l30_*` columns hold the 30 m Landsat + Sentinel-2 dates and their bracket width, and `mod500_*` holds native MODIS (all aspects). |
| **Dates per 30 m cell** | `rasters/modis_downscaled_30m_{snow_on,snow_off}_dowy_wy2000_2026.tif`, `..._scd_...` | 27 bands (WY2000–2026), EPSG:32611, 30 m, same grid as `grid_30m_terrain_region_persistence.tif`. |
| **Daily 30 m snow maps** | `rasters/daily_30m/snow30m_daily_WY*.npz` | For sampling at mesh nodes. `snow` and `valid` are bit-packed (`np.unpackbits(a, axis=1)[:, :ncell]`); `iy`, `ix` are row/col on the 30 m grid; `dates` holds the days. |
| All products, all bands | `tables/band_snow_onoff_all_products.csv` | MOD10A1F, MYD10A1F and VNP10A1F at NDSI thresholds of 10, 20, 30 and 40, plus Landsat_S2_30m and MODIS_downscaled_30m. |
| Band climatology and trend | `tables/band_climatology_wy2001_2025.csv` | Medians, IQR, and the Sen slope of snow-off in days per decade (downscaled product). |
| Independent 30 m melt-out | `rasters/landsat_s2_pixel_snowoff_*` | Per-cell melt-out from clear Landsat and S2 scenes, WY1985–2026, with bracket width (days between the last snow and first snow-free clear view). |

## Definitions
- A daily series is snow (1), no snow (0) or missing. Missing days are filled when both neighbours agree, and gaps of up to 3 days without snow are bridged. The **main season** is the longest continuous snow run of at least 14 days. **snow_on** is its first day, and **snow_off** (melt-out) is the first snow-free day after it. **scd** counts snow days. In years without a 14-day season (low elevations, WY2026), snow_on and snow_off are blank and scd still counts snow days. For a band, a day counts as snow when at least 50% of its valid cells are snow covered.
- Landsat and S2 (sparse clear views): the main season is the longest span of clear views with band SCA ≥ 50%. Crossing dates are interpolated between the bracketing views, and `*_bracket_days` gives the gap. Median off-bracket is about 2–3 weeks.
- MODIS and VIIRS use the cloud-gap-filled products (MOD10A1F/MYD10A1F v61, VNP10A1F v2). A pixel-day whose cloud persistence exceeds 10 days counts as missing. **NDSI ≥ 10 is the default threshold**, because it matched the ground best (see validation).
- First years have no snow-on: MODIS Terra starts 24 Feb 2000, Aqua in Jul 2002 and VIIRS in Jan 2012.

## Sources (all pulled 2026-10-08)
- MOD10A1F and MYD10A1F (500 m, daily, 2000–2026) and VNP10A1F (375 m, 2012–2026), tile h09v04, from NSIDC Earthdata cloud.
- Landsat 4/5/7/8 Collection 1 L1 TOA, paths 41 and 42 row 30, 1984–2021, from the Google Cloud public bucket. Collection 2 on AWS is requester-pays and was not used, and GCP has no Landsat 9.
- Sentinel-2 L2A COGs, tile 11TNJ, 2017–2026, from `s3://sentinel-cogs`; 1,876 of 1,908 scenes were read and 32 kept failing. Together that gives 2,961 usable 30 m scenes.
- Ground data: DCEW snow depth at TL, LW and BRW and shallow soil temperature (`/mnt/project-files/drycreek/`), plus Bogus Basin SNOTEL 978 (SWE and depth) via NRCS AWDB.
- Terrain comes from the tRIBS 10 m DEM (`tribs/basins/dry_creek/dem_box.tif`) aggregated to 30 m.

30 m snow test: NDSI ≥ 0.40 with NIR ≥ 0.11 and green ≥ 0.10 (SNOMAP). Where summer NDVI ≥ 0.45 (treated as forest), NDSI ≥ 0.20 instead. Clouds come from the Landsat BQA or S2 SCL; bright, low-SWIR pixels flagged as cloud are rescued as snow. The NDVI forest flag covers 55% of Dry Creek, against 41% forest in NLCD, because it also catches some tall shrub.

## Downscaling MODIS to 30 m (following Walters et al. 2014, RSE 152:413)
Each day, MOD10A1F NDSI is converted to fSCA (−0.01 + 1.45·NDSI; Salomonson & Appel 2004). Cells are then ranked and the fSCA share with the highest rank is set to snow.
- **Seam fix (Lejo's point about MODIS pixel edges):** fSCA is interpolated bilinearly between MODIS pixel centres, and each 30 m cell is ranked within a 450 m window centred on itself rather than within its MODIS pixel. This holds the snow fraction close to MODIS while removing the block edges. The cost was small (F 0.717 against 0.725 for the block version), and skill near the old pixel edges equals interior skill.
- **Ranking:** I tested the Walters terrain score (w·insolation + (1−w)·relative elevation) against an empirical **snow-persistence** rank, defined as the share of clear Feb–Jun Landsat/S2 views in which each cell is snow. Scored on 313–375 scenes, with persistence built from the other half of the years (odd/even WY):

| F-measure vs Landsat/S2 (cells in partly snowy MODIS pixels) | fed true fSCA | fed MODIS fSCA |
|---|---|---|
| random placement | 0.65 | 0.65 |
| terrain score, w = 0 / 0.5 / 0.9 (block) | 0.73 / 0.72 / 0.71 | 0.65 / 0.64 / 0.62 |
| persistence rank (block) | **0.84** | **0.73** |
| open cells only: terrain w = 0.5 / persistence hybrid | 0.72 / 0.80 (random 0.59) | 0.63 / 0.72 |

  Insolation helps only on open cells: there, w = 0.5 gives F 0.72 against 0.67 for elevation alone, which is consistent with Reggie's w ≈ 0.9 in open terrain.
- **Canopy artifact and fix:** under dense north-facing forest, optical sensors miss the snow, so raw persistence is lower on forested north slopes than on south slopes (1,600–2,000 m), and N-facing cells melted out first. The production ranking is therefore a **hybrid**: forest cells take the persistence predicted from elevation and spring insolation (fit on open cells, r = 0.96). With the hybrid, north slopes melt out 7–33 days after south slopes in every band (fig5), matching the open-terrain Landsat/S2 signal. The raw F score drops (0.64) only because the Landsat/S2 "truth" cannot see under the canopy. On open cells the hybrid scores F 0.72.
- Treat the downscaled dates under forest as terrain-model inference, not observation.

## Validation against the ground (`validation/`)
The ground references are snow depth > 5 cm (TL, LW, BRW), SNOTEL SWE > 2.5 mm, and soil temperature at 2–5 cm with a diurnal range < 1 °C and max < 2 °C (8 aspect stations, SCR, TL, BRW). Matched site-years are 68–110 per product (common sample):

| product | melt-out bias (median) | MAE | r | snow-on bias | MAE |
|---|---|---|---|---|---|
| Landsat + S2 30 m | −2 d | 7 d | 0.90 | −6 d | 12 d |
| MODIS Terra 500 m (NDSI ≥ 10) | −9 d | 12 d | 0.89 | 0 d | 10 d |
| MODIS Aqua | −5 d | 12 d | 0.88 | +2 d | 9 d |
| VIIRS 375 m | −6 d | 10 d | 0.92 | +2 d | 10 d |
| MODIS downscaled 30 m | −11 d | 13 d | 0.88 | +1 d | 10 d |

- Satellites put melt-out 1–2 weeks earlier than the point sensors. At the SNOTEL the gap is 10–12 days for MODIS but only 2 days for Landsat/S2. That fits Lejo's note that the SNOTEL sits in a deep-snow north-facing toe slope; the 500 m pixel melts out before that spot does. Higher NDSI thresholds widen the early bias (Terra −8 d at NDSI ≥ 10, −18 d at ≥ 40).
- Downscaling does not improve point timing relative to native MODIS. Its value is the spatial pattern (aspect and elevation) on a daily clock. For timing at a point, the Landsat/S2 dates are sharper when their bracket is short.
- Soil-temperature references are noisy at S-facing ephemeral-snow sites (LS, MLS, HS), where few site-years qualify.

## Main patterns (downscaled product, Dry Creek, median WY2001–2025)
| band (m) | snow-on | snow-off | snow days |
|---|---|---|---|
| 1000–1200 | Dec 23 | Feb 1 | 24 |
| 1200–1400 | Dec 18 | Feb 6 | 46 |
| 1400–1600 | Dec 18 | Feb 1 | 70 |
| 1600–1800 | Dec 11 | Mar 21 | 109 |
| 1800–2000 | Dec 2 | Apr 15 | 130 |
| 2000–2200 | Nov 29 | Apr 23 | 148 |

- Melt-out jumps about 6 weeks between the 1400–1600 and 1600–1800 m bands. Below about 1,600 m, snow is intermittent, and in many years there is no 14-day season.
- North slopes hold snow about 1–5 weeks longer than south slopes (fig5).
- No band shows a significant melt-out trend over WY2001–2025 (Kendall p > 0.3).
- **WY2026 stands out:** no continuous season below 1,800 m, and at 1,800–2,200 m only Feb 19 to Mar 15, with 42–58 snow days against medians of 130–148 (fig2).

## Figures
`fig1_onoff_vs_elevation.png` (all products), `fig2_dry_creek_bands_by_wy.png`, `fig3_meltout_maps.png`, `fig4_validation_snowoff.png`, `fig5_aspect_effect.png`, `fig6_ns_snow_ranges_by_band.png` (N = NW/N/NE, S = SE/S/SW snow-cover ranges with ±1 SD whiskers; numbers in `tables/ns_snow_ranges_by_band_wy2001_2025.csv`, code `fig_aspect_ranges.py`), `fig7_snotel_and_dry_creek_ns_ranges.png` (SNOTEL 978 ground record above Dry Creek N vs S, code `fig_snotel_dc_ranges.py`), `fig8_snow_probability_jan_apr.png` (per-cell snow probability and year-to-year SD near Jan 1, Feb 1, Mar 1, Apr 1; grids in `rasters/snow_probability_*.npz` on the 30 m grid, code `fig_snow_prob_maps.py`), `fig9_snow_days_trend_map.png` (per-cell Sen slope of snow-covered days WY2001-2025, raster `rasters/scd_sen_slope_wy2001_2025.tif` with MK p in band 2; code `fig_scd_trend_map.py`; the N-gain/S-loss split mirrors a native-MODIS rise in partial-snow days that Landsat alone does not confirm).

## Code (`code/`), run order
`fetch_tiles.py` (MODIS/VIIRS) → `fetch_landsat.py` → `fetch_s2.py` → `build_grid.py` → `ground_truth.py` → `process_daily.py` → `process_hires.py` → `build_persistence.py` → `ds_eval.py` → `ds_run.py` → `validate.py` → `validate_ds.py` → `assemble.py`. Shared rules live in `ondates.py`, and the downscaler in `downscale.py`.
Raw tiles and scene windows (about 1 GB) stayed in the thread's scratchpad, so rerun the fetch scripts to regenerate them. Sentinel-2 endpoints on S3 returned intermittent 404s; `fetch_s2.py` retries and rotates endpoints.

## Possible next steps (not done)
- A ConvLSTM, or a similar learned downscaler, trained on daily MODIS/VIIRS sequences against the ~3,000 Landsat/S2 maps, with held-out years and the persistence hybrid as the baseline to beat (Lejo's suggestion).
- Fuse Terra, Aqua and VIIRS before downscaling (VIIRS had the best point skill).
- Add Landsat Collection 2 / Landsat 9 for 2022 onward if requester-pays access is set up.
