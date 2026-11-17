# daily-30m-snowcover

Daily 30 m snow cover and snow-on / snow-off dates for mountain research watersheds, made by downscaling the MODIS cloud-gap-filled snow product (MOD10A1F, 500 m) with a snow-persistence ranking built from Landsat and Sentinel-2.

Built so far for two watersheds in southwest Idaho, water years 2000-2026:

- **Dry Creek Experimental Watershed** (plus an adjacent Bogus Basin box), Boise Front
- **Reynolds Creek Experimental Watershed**, Owyhee Mountains

The method is not specific to these sites. A new watershed needs an entry in `code/domain.py`, a DEM and a watershed mask.

## How it works

1. **MODIS says how much.** Each day, MOD10A1F NDSI is converted to fractional snow-covered area (fSCA) per 463 m pixel.
2. **Landsat and Sentinel-2 say where.** Every clear 30 m scene is turned into a snow map. The share of clear February-June views in which a cell is snow covered is its *snow persistence*. Under forest, where optical sensors miss snow, persistence is predicted from elevation and insolation instead.
3. **Downscaling.** fSCA is interpolated between MODIS pixel centres, and each 30 m cell is ranked by persistence within a 450 m window. A cell is snow covered when its rank is within the local fSCA.
4. **Seasons.** Per cell and water year, the main snow season is the longest snow-covered run of at least 14 days (gaps of up to 3 days bridged); its first day is snow-on and the first snow-free day after it is snow-off.

This builds on the physiographic downscaling of Walters et al. (2014), *Remote Sensing of Environment* 152:413, which ranked cells by a terrain score of insolation and relative elevation.

## Start with the notebooks

`notebooks/` has three executed notebooks that walk through one example day at Reynolds Creek (18 March 2019):

| notebook | what it shows |
|---|---|
| `01_preprocessing.ipynb` | grid and static layers, daily MODIS snow fraction, a Landsat 8 snow map, the persistence ranking |
| `02_downscaling.ipynb` | MODIS fraction to 30 m snow cover step by step, scored against the same-day Landsat map; snow-on/off dates for one cell |
| `03_verification.ipynb` | comparison with ground snow depth and with held-out Landsat / Sentinel-2 maps |

They open with their outputs on GitHub. To rerun them you need the data files (below); see `notebooks/README.md`.

## Layout

| folder | contents |
|---|---|
| `code/` | the processing pipeline (see `code/README.md` for the run order) |
| `notebooks/` | the three workflow notebooks, their small cached inputs, `requirements.txt` |
| `validation/` | summary tables: split-year cross-validation, ground validation, sensor agreement |
| `docs/` | working notes from the data release and from the first Dry Creek version |

## Data

The data files are **not in this repository**. They will be archived separately under a DOI:

- daily snow cover with quality flags (`*_snowcover_daily_30m_v1.nc`) and seasonal dates (`*_snow_seasons_30m_v1.nc`), CF-1.8 NetCDF, EPSG:32611;
- the MODIS windows used as input.

To rerun the notebooks, put those files in the repository root (the daily file may be one NetCDF or parts along time; `code/daily_io.py` reads either).

The ground snow depth used for verification at Reynolds Creek belongs to the USDA-ARS Northwest Watershed Research Center and is not distributed here. Only summary statistics derived from it are included.

## Status

Draft, ahead of a data description paper. Results and file formats may still change.

## Use of generative AI

The processing code and the notebooks in this repository were written with the assistance of Claude (Anthropic), working under the direction of the authors.

## Contact

Lejo Flores, Boise State University.
