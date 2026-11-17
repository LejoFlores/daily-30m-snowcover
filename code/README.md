# code/

All scripts take their paths on the command line and print their usage in the module docstring. The watershed is chosen with the environment variable `SNOW_DOMAIN` (`dcew` or `rcew`; see `domain.py`).

## Run order (current pipeline)

| step | script | what it does |
|---|---|---|
| 1 | `fetch_tiles.py` | daily MOD10A1F (or MYD10A1F / VNP10A1F) tiles from NSIDC, cut to the watershed window; needs `EARTHDATA_TOKEN` |
| 2 | `fetch_landsat_c2.py` | Landsat 4-9 Collection 2 Level-2 scenes from Microsoft Planetary Computer, on the 30 m grid |
| 3 | `fetch_s2.py` | Sentinel-2 L2A scenes from the `sentinel-cogs` bucket |
| 4 | `build_grid.py` | 30 m grid: elevation, slope, aspect, watershed mask |
| 5 | `process_hires.py` | 30 m snow maps from every Landsat / Sentinel-2 scene; forest flag from summer NDVI |
| 6 | `build_persistence.py` | snow persistence per cell, and the terrain model used under forest |
| 7 | `ds_eval.py` | split-year cross-validation of the downscaling against the 30 m maps |
| 8 | `ds_run.py` | daily downscaling for all days; snow-on, snow-off and snow-cover days per water year |
| 9 | `build_essd_nc.py` | CF NetCDF files with quality flags |
| 10 | `validate_rcew_ground.py`, `fig_rcew_ground_validation.py` | ground validation at Reynolds Creek |

Shared modules: `downscale.py` (the method), `ondates.py` (season rules), `domain.py` (watershed settings), `daily_io.py` (reads the daily file as one NetCDF or as parts).

Utilities: `rebuild_inputs_from_nc.py` (rebuild the downscaling inputs from a seasons file, without refetching Landsat), `split_daily_nc.py` (split a daily file along time), `sensor_drift_check.py` (Terra against Aqua and VIIRS), `make_notebooks.py` (writes the notebooks in `../notebooks/`).

Figures: `fig_snow_prob_maps.py`, `fig_snow_prob_bivariate.py`, `fig_ns_two_basins.py`, `fig_aspect_ranges.py`, `fig_snotel_dc_ranges.py`, `fig_scd_trend_map.py`.

## First Dry Creek version (October 2026)

These scripts belong to the first, Dry Creek-only run and have not been generalised to other watersheds. They are kept for the record: `fetch_landsat.py` (Landsat Collection 1, since retired), `process_daily.py`, `ground_truth.py`, `validate.py`, `validate_ds.py`, `assemble.py`. Some of them expect the Dry Creek station files and folder layout of the project they were written in.
