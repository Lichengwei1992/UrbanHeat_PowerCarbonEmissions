@echo off
REM ============================================================
REM cleanup_corrupt_outputs.bat
REM Run ONCE before re-running the pipeline to clear all
REM outputs that were built with the buggy (pre-fix) scripts.
REM ============================================================

SET BASE=%~dp0

echo === Deleting corrupt ERA5 baseline cache ===
del /F /Q "%BASE%ERA5\baseline\threshold_table_doy.parquet"
del /F /Q "%BASE%ERA5\baseline\city_baseline_all_years.parquet"
del /F /Q "%BASE%ERA5\baseline\city_baseline_????.parquet"

echo === Deleting corrupt study-period extraction ===
del /F /Q "%BASE%outputs\city_climate_2019_2021.parquet"

echo === Deleting corrupt event flags ===
del /F /Q "%BASE%outputs\04_events\event_flags.parquet"
del /F /Q "%BASE%outputs\04_events\*.parquet"
del /F /Q "%BASE%outputs\04_events\*.csv"

echo === Deleting corrupt descriptive statistics ===
del /F /Q "%BASE%outputs\06_descriptive\*.*"

echo.
echo === DONE. Files to keep (do NOT delete): ===
echo   ERA5\temperature\era5_temp_daily_*.nc  (study-period temperature downloads)
echo   ERA5\utci\utci_daily_*.nc              (study-period UTCI downloads)
echo   ShapefileData\*                         (FUA shapefiles)
echo   koppen_geiger_tif\*                     (Koppen-Geiger raster)
echo.
echo === Re-run order: ===
echo   1. python 05c_download_era5_baseline.py    (re-downloads 1991-2020 baseline at 0.25 deg)
echo   2. python 05d_extract_era5_cities.py       (re-extracts 2019-2021 city pixels, fixed lons)
echo   3. python 04_define_events.py              (recomputes event flags with correct thresholds)
echo   4. python 06_descriptive_climate.py        (regenerates descriptive stats + QC check)
echo.
pause
