# essd_draft/ — draft release files for the ESSD data paper (Dry Creek and Reynolds Creek, v1.0-draft, 2026-10-23)

These are not final. Reynolds Creek ground validation was added 2026-11-17 from ARS snow depth (see the end of this file). The plan is in `../publication/essd-readiness-plan.md`.

| file | contents |
|---|---|
| `dcew_snowcover_daily_30m_v1.nc` | CF-1.8 NetCDF4, 9,716 days (24 Feb 2000 – 30 Sep 2026), 450 × 418 cells at 30 m, EPSG:32611. Variables: `snow_cover` (0/1, 255 missing); `input_fsca` (MODIS fSCA at the cell, %); `cloud_age` (days since the MODIS pixel was last seen clear); `quality` (bit flags: 1 gap-filled, 2 cloud_age ≥ 4 d, 4 partial-snow input, 8 forest/terrain-model ranking, 16 within 210 m of the domain edge); static `region`, `elevation`, `slope`, `aspect`, `forest`, `snow_persistence`. |
| `dcew_snow_seasons_30m_v1.nc` | Per water year 2000–2026: `snow_on`, `snow_off` (day of water year) and `snow_cover_days`. |
| `validation/` | Split-year cross-validation of the downscaling (F scores) and ground validation (melt-out and onset bias, MAE and r). |
| `code/` | `fetch_landsat_c2.py` (Landsat C2 L2 from Planetary Computer), `process_hires.py`, `build_persistence.py`, `build_essd_nc.py`. The rest of the pipeline is in `../code/`. |

Changes from the October 8 product: Landsat now comes from Collection 2 Level-2, Tier 1, with Landsat 9 added and coverage through 2026 (3,398 usable 30 m scenes, up from 2,961). The snow-persistence ranking and the forest mask were rebuilt from these scenes.

## Results on Collection 2

**Downscaling skill.** F scores on open cells, with persistence taken from the other half of the years:

| input | persistence ranking | terrain ranking (Walters-style) | random |
|---|---|---|---|
| MODIS | 0.74 | 0.65 | 0.60 |
| true fSCA | 0.78 | 0.72 | 0.60 |

**Ground validation (melt-out).** Landsat+S2 30 m: bias −3 d, MAE 5 d, r 0.93 (was −2 d, 7 d and 0.90 on Collection 1). The downscaled product is unchanged (−11 d, 13 d).

Band melt-out medians moved by at most 2 days relative to Collection 1.

## Reynolds Creek (added 2026-10-23)

Files: `rcew_snowcover_daily_30m_v1.nc` (612 MB) and `rcew_snow_seasons_30m_v1.nc`. They have the same layout and flags as Dry Creek. The grid is EPSG:32611, 30 m, 560 × 1030 cells (E 510000–526800, N 4766100–4797000). The watershed is region 1 (240 km², 1,104–2,241 m, from the tRIBS DEM and mask).

Inputs:
- 3,696 usable Landsat C2 T1 and Sentinel-2 scenes (tile 11TNH).
- MOD10A1F, 24 Feb 2000 – 30 Sep 2026.
- The forest flag covers 15% of the watershed (57% in Dry Creek).

Downscaling skill, F scores with split-year persistence:

| input | persistence ranking | terrain ranking | random |
|---|---|---|---|
| MODIS, all cells | 0.73 | 0.69 | 0.65 |
| MODIS, open cells | 0.73 | 0.66 | — |
| true fSCA, all cells | 0.78 | 0.72 | 0.64 |

Median downscaled melt-out by band:

| band | melt-out | snow days |
|---|---|---|
| < 1,600 m | about Jan 18–29 | 13–27 |
| 1,600–1,800 m | Mar 8 | 80 |
| 1,800–2,000 m | Apr 1 | 115 |
| 2,000–2,200 m | Apr 29 | 155 |
| 2,200 m+ | May 6 | 169 |

Quality flags (share of cell-days): gap-filled 45%, cloud age ≥ 4 d 15%, partial-snow input 13%, forest 15%, edge 8%.

The Reynolds band tables are in `validation/rcew_*`. The domain settings are in `code/domain.py` (`SNOW_DOMAIN=rcew`).

## Reynolds Creek ground validation (added 2026-11-17)

Ground data: ARS daily snow depth at 40 sites (WY2000–2026; RC.138l21 has only two seasons and gives no pairs), from Andrew Hedrick via Lejo. Same rule as Dry Creek: snow when depth > 5 cm, then the same season rules. Site-years need ≥ 80% of Nov–May depth records and an uncensored ≥ 14-day season. Satellite dates are read at the 30 m cell holding each site (3×3 median gives the same answer within 2 d).

| subset | site-years | melt-out bias (median) | MAE | r | snow-on MAE |
|---|---|---|---|---|---|
| all | 595 | −1 d | 18 d | 0.76 | 17 d |
| one clear season in both records | 296 | 0 d | 11 d | 0.89 | 9 d |
| one clear season, open cells | 201 | +3 d | 11 d | 0.92 | 9 d |
| one clear season, forest cells | 95 | −9 d | 13 d | 0.81 | 8 d |
| season split by a mid-winter melt | 299 | −1 d | 25 d | 0.59 | 25 d |

- Unlike Dry Creek (−11 d), there is no overall early bias. Forest cells melt out about 9–11 days early, as in Dry Creek.
- Most large errors come from intermittent snowpacks, where a mid-winter melt splits the season and the ground and satellite records pick different runs as the "main season". This is a metric problem as much as a product one; a daily snow/no-snow agreement score would avoid it but needs the Reynolds daily file.
- Sites with large, consistent offsets (possibly drift or scour sites unrepresentative of 30 m; inferred, not checked): RC.167 (−45 d, forest cell), RC.138d03, RC.138f06 and RC.147 (+22 to +27 d).
- Files: `validation/rcew_ground_vs_30m_by_site_wy.csv`, `rcew_ground_validation_summary.csv`, `rcew_ground_validation_by_site.csv`; figure `../figures/fig14_rcew_ground_validation_meltout.png`; code `code/validate_rcew_ground.py`, `code/fig_rcew_ground_validation.py`.

## Reynolds daily file rebuilt (2026-11-17)

`rcew_snowcover_daily_30m_v1.nc` (613 MB) was missing from this folder and was regenerated. **The shared folder dropped the single 613 MB file twice, so it is stored here as four parts along time, `rcew_snowcover_daily_30m_v1_partK_of_4.nc` (about 155 MB each; `code/split_daily_nc.py` makes them and `code/daily_io.py` reads either layout).** How it was regenerated: MOD10A1F refetched (9,716 days; window archive in `inputs/rcew_mod10a1f_window_2000_2026.tar`), and the terrain, persistence and forest inputs rebuilt from the static layers of `rcew_snow_seasons_30m_v1.nc` (`code/rebuild_inputs_from_nc.py`), then `ds_run.py` and `build_essd_nc.py` (domain-aware; `fetch_tiles.py` takes `TILE_WINDOW`).
- Check against the October seasons file: cells more than 210 m from the watershed edge are 99.9% identical in snow_on/snow_off. Edge cells (7.5% of cells, quality bit 16) differ by 3–4 days on average, because the static layers are masked outside the watershed and the ranking window is truncated there.
- `rcew_snow_seasons_30m_v1.nc` is now the rebuilt seasons file, which matches the daily file exactly (Lejo's choice, 2026-11-17; the October version was replaced). The ground-validation tables above were rerun on it.

**Daily agreement with ARS snow depth (> 5 cm), Nov–May, 213,186 site-days** (`validation/rcew_ground_daily_agreement.csv`):

| subset | accuracy | precision | recall | F |
|---|---|---|---|---|
| all | 0.79 | 0.61 | 0.85 | 0.71 |
| MODIS saw the pixel that day | 0.85 | 0.65 | 0.88 | 0.75 |
| gap-filled days | 0.76 | 0.60 | 0.84 | 0.70 |
| full or zero snow input | 0.89 | 0.70 | 0.84 | 0.76 |
| partial-snow input | 0.61 | 0.55 | 0.86 | 0.67 |

The product calls snow on 42% of site-days against 30% on the ground: it rarely misses snow the sensors see, but often shows snow where depth is under 5 cm (thin or patchy snow around the sensor counts as snow from above). Depth rows flagged `qc_suspect_zero` were left out.

## Workflow notebooks (added 2026-11-17)

`notebooks/` holds three executed Jupyter notebooks (pre-processing, downscaling, verification) for one example day at Reynolds Creek; see `notebooks/README.md`. They are generated by `code/make_notebooks.py`.
