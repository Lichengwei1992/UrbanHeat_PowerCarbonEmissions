"""
07_download_chi_indices.py

Stage CHI heat-index data from Globus for city-level analysis.

Remote source:
  f55046b8-f1c3-424d-8016-082271acf156:/CHI_GLOBAL_DATASETS_1950-2024/

Directory pattern:
  <INDEX>/<YEAR>/CHI_<INDEX>_<YYYY-MM>.nc

Default behavior is intentionally conservative:
  1. Transfer one monthly NetCDF from Globus.
  2. Extract all CHI grid cells whose centers fall inside each FUA polygon.
  3. Save city-day parquet, and optionally city-hour parquet.
  4. Delete the raw monthly NetCDF unless --keep-raw is set.

This avoids trying to store the full CHI archive locally. One raw monthly file is
usually 6-7 GB; all 13 indices for 1991-2021 would be tens of TB.

Examples
--------
Dry-run a small test:
  python 07_download_chi_indices.py --indices Twb UTCI --years 2019 --months 7 ^
      --local-globus-root "/D/CHI_Data" --local-dir "D:\\CHI_Data" --dry-run

Run one-month extraction test:
  python 07_download_chi_indices.py --indices Twb --years 2019 --months 7 ^
      --local-globus-root "/D/CHI_Data" --local-dir "D:\\CHI_Data"

Process study period for all 13 indices, daily outputs only:
  python 07_download_chi_indices.py --years 2019-2021 ^
      --local-globus-root "/D/CHI_Data" --local-dir "D:\\CHI_Data"

Process baseline + study period:
  python 07_download_chi_indices.py --years 1991-2021 ^
      --local-globus-root "/D/CHI_Data" --local-dir "D:\\CHI_Data"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


BASE = Path(__file__).resolve().parent
DEFAULT_LOOKUP = BASE / "outputs" / "fua_city_lookup.csv"
DEFAULT_SHPFILE = BASE / "ShapefileData" / "FUA_cities_final.shp"

REMOTE_ENDPOINT = "f55046b8-f1c3-424d-8016-082271acf156"
REMOTE_BASE = "/CHI_GLOBAL_DATASETS_1950-2024"

INDEX_INFO = {
    "AT": "Apparent Temperature",
    "HI": "Heat Index",
    "Hu": "Humidex",
    "Lsi": "Indoor Lethal Heat Stress Index",
    "Lsin": "Natural Lethal Heat Stress Index",
    "NET": "Normal Effective Temperature",
    "Tg": "Globe Temperature",
    "Tmrt": "Mean Radiant Temperature",
    "Tnwb": "Natural Wet-Bulb Temperature",
    "Twb": "Indoor/Shaded Wet-Bulb Temperature",
    "Twbg": "Wet-Bulb Globe Temperature",
    "Twc": "Wind Chill Temperature",
    "UTCI": "Universal Thermal Climate Index",
}

ALL_INDICES = list(INDEX_INFO)


def parse_int_range(values: Iterable[str], lo: int, hi: int, label: str) -> list[int]:
    out: set[int] = set()
    for raw in values:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                a, b = token.split("-", 1)
                start, end = int(a), int(b)
                out.update(range(start, end + 1))
            else:
                out.add(int(token))
    vals = sorted(out)
    bad = [v for v in vals if v < lo or v > hi]
    if bad:
        raise ValueError(f"{label} outside allowed range {lo}-{hi}: {bad}")
    return vals


def find_globus_cli(user_path: str | None = None) -> str:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Globus CLI not found at --globus-cli={p}")

    exe = shutil.which("globus")
    if exe:
        return exe

    conda_guess = (
        Path.home()
        / ".conda"
        / "envs"
        / "PythonProject_LCZCARBON"
        / "Scripts"
        / "globus.exe"
    )
    if conda_guess.exists():
        return str(conda_guess)

    raise FileNotFoundError(
        "Globus CLI not found. Install globus-cli or pass --globus-cli."
    )


def run_cmd(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")
    return proc


def globus_local_endpoint(globus_cli: str) -> str:
    proc = run_cmd([globus_cli, "endpoint", "local-id"], timeout=60)
    endpoint = proc.stdout.strip().splitlines()[-1].strip()
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", endpoint):
        raise RuntimeError(f"Could not parse local endpoint id: {proc.stdout!r}")
    return endpoint


def norm_globus_dir(path: str) -> str:
    p = path.replace("\\", "/").strip()
    if not p.startswith("/"):
        raise ValueError(
            f"Globus paths must start with '/'. Got: {path!r}. "
            "Example: /D/CHI_Data or /~/CHI_Data"
        )
    return p.rstrip("/")


def infer_globus_path(local_dir: Path) -> str:
    """
    Convert a Windows local path into the usual Globus Connect Personal path.

    Example:
      D:\\CHI_Data -> /D/CHI_Data

    This is the convention used by Globus Connect Personal for local drive
    roots. Users can override with --local-globus-root when their endpoint uses
    a different mapping.
    """
    resolved = local_dir.resolve()
    drive = resolved.drive.rstrip(":")
    if drive:
        rest = resolved.relative_to(resolved.anchor).as_posix()
        return f"/{drive}/{rest}".rstrip("/")

    # Fallback for non-drive paths or unusual endpoint mappings.
    return "/" + resolved.as_posix().lstrip("/")


def remote_file(index: str, year: int, month: int) -> str:
    return f"{REMOTE_BASE}/{index}/{year}/CHI_{index}_{year}-{month:02d}.nc"


def local_raw_file(local_dir: Path, index: str, year: int, month: int) -> Path:
    return local_dir / "raw" / index / f"{year}" / f"CHI_{index}_{year}-{month:02d}.nc"


def processed_daily_file(local_dir: Path, spatial_method: str, index: str, year: int, month: int) -> Path:
    return local_dir / f"processed_daily_{spatial_method}" / index / f"{year}" / (
        f"chi_{index}_{year}-{month:02d}_city_day.parquet"
    )


def processed_hourly_file(local_dir: Path, spatial_method: str, index: str, year: int, month: int) -> Path:
    return local_dir / f"processed_hourly_{spatial_method}" / index / f"{year}" / (
        f"chi_{index}_{year}-{month:02d}_city_hour.parquet"
    )


def local_globus_file(root: str, index: str, year: int, month: int) -> str:
    root = norm_globus_dir(root)
    return f"{root}/raw/{index}/{year}/CHI_{index}_{year}-{month:02d}.nc"


def remote_file_size(globus_cli: str, endpoint: str, source_path: str) -> int:
    """Return the expected byte size for one remote Globus file."""
    source = Path(source_path.replace("\\", "/"))
    parent = source.parent.as_posix()
    name = source.name
    proc = subprocess.run(
        [globus_cli, "ls", "-F", "json", f"{endpoint}:{parent}/"],
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"Globus remote listing failed with exit code {proc.returncode}")
    payload = json.loads(proc.stdout)
    for item in payload.get("DATA", []):
        if item.get("name") == name:
            return int(item["size"])
    raise FileNotFoundError(f"Remote file not found in Globus listing: {source_path}")


def quarantine_file(path: Path, reason: str) -> Path:
    """Move an invalid raw file aside so Globus can redownload the target path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.{reason}.{stamp}")
    path.replace(target)
    return target


def submit_transfer(
    globus_cli: str,
    source_endpoint: str,
    dest_endpoint: str,
    source_path: str,
    dest_path: str,
    label: str,
    dry_run: bool,
) -> str | None:
    cmd = [
        globus_cli,
        "transfer",
        "-F",
        "json",
        "--sync-level",
        "size",
        "--label",
        label,
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend([f"{source_endpoint}:{source_path}", f"{dest_endpoint}:{dest_path}"])
    proc = run_cmd(cmd, timeout=120)
    if dry_run:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        match = re.search(r"[0-9a-fA-F-]{36}", proc.stdout)
        if not match:
            raise RuntimeError(f"Could not parse Globus task id: {proc.stdout!r}")
        return match.group(0)

    task_id = payload.get("task_id") or payload.get("task", {}).get("task_id")
    if not task_id:
        match = re.search(r"[0-9a-fA-F-]{36}", proc.stdout)
        if match:
            task_id = match.group(0)
    if not task_id:
        raise RuntimeError(f"No task_id in Globus transfer response: {payload}")
    return task_id


def wait_for_task(
    globus_cli: str,
    task_id: str,
    polling_interval: int,
    timeout_seconds: int,
) -> None:
    run_cmd(
        [
            globus_cli,
            "task",
            "wait",
            task_id,
            "--polling-interval",
            str(polling_interval),
            "--timeout",
            str(timeout_seconds),
        ],
        timeout=timeout_seconds + 120,
    )
    proc = run_cmd([globus_cli, "task", "show", "-F", "json", task_id], timeout=60)
    try:
        payload = json.loads(proc.stdout)
        status = payload.get("status")
        fatal_error = payload.get("fatal_error")
        if status != "SUCCEEDED":
            raise RuntimeError(f"Globus task {task_id} ended with status={status}: {fatal_error}")
    except json.JSONDecodeError:
        pass


def load_city_lookup(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"City lookup not found: {path}")
    coords = pd.read_csv(path, dtype={"city_id": str})
    required = {"city_id", "lat", "lon"}
    missing = required - set(coords.columns)
    if missing:
        raise ValueError(f"City lookup missing columns: {sorted(missing)}")
    coords["city_id"] = coords["city_id"].str.zfill(4)
    coords = coords.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return coords


def load_fua_polygons(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"FUA shapefile not found: {path}")
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit("Install geopandas to use --spatial-method polygon") from exc

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.reset_index(drop=True)
    gdf["city_id"] = gdf.index.astype(str).str.zfill(4)
    gdf = gdf[["city_id", "city", "country", "geometry"]].copy()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    return gdf


def first_present(names: tuple[str, ...], available: Iterable[str], kind: str) -> str:
    available_set = set(available)
    for name in names:
        if name in available_set:
            return name
    raise ValueError(f"Could not find {kind}; checked {names}, available={sorted(available_set)}")


def detect_variable(ds: xr.Dataset, index: str, time_name: str, lat_name: str, lon_name: str) -> str:
    candidates = [index, index.lower(), index.upper(), f"CHI_{index}"]
    for name in candidates:
        if name in ds.data_vars:
            return name

    scored = []
    required_dims = {time_name, lat_name, lon_name}
    for name, da in ds.data_vars.items():
        if required_dims.issubset(set(da.dims)) and np.issubdtype(da.dtype, np.number):
            scored.append(name)
    if len(scored) == 1:
        return scored[0]
    if not scored:
        raise ValueError(
            f"No numeric data variable with dims {sorted(required_dims)}. "
            f"Data vars: {list(ds.data_vars)}"
        )
    raise ValueError(
        f"Multiple possible data variables for {index}: {scored}. "
        "Pass support for explicit variable selection if needed."
    )


def should_convert_kelvin(values: np.ndarray | pd.Series, units: str) -> bool:
    """Return True when a CHI array is finite Kelvin data that should become Celsius."""
    if units not in {"k", "kelvin"}:
        return False
    arr = np.asarray(values, dtype="float64")
    finite = arr[np.isfinite(arr)]
    return finite.size > 0 and float(finite.mean()) > 100.0


def nearest_indices(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    values = np.asarray(values)
    return np.abs(grid[None, :] - values[:, None]).argmin(axis=1).astype(np.int32)


def normalize_lon_for_grid(lon: float, lons: np.ndarray) -> float:
    """Convert one longitude to the convention used by a CHI longitude grid."""
    if np.nanmin(lons) >= 0 and np.nanmax(lons) > 180:
        return float(lon % 360)
    if np.nanmin(lons) < 0:
        return float(((lon + 180) % 360) - 180)
    return float(lon)


def nearest_valid_grid_cell(
    lats: np.ndarray,
    lons: np.ndarray,
    valid_mask: np.ndarray,
    lat: float,
    lon: float,
    max_radius_cells: int = 50,
) -> tuple[int, int] | None:
    """
    Find the nearest finite CHI grid cell around a point.

    This is used only when a small coastal FUA has polygon-selected grid-cell
    centers, but all of those cells are masked as ocean/invalid in CHI.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lon = normalize_lon_for_grid(float(lon), lons)
    lat0 = int(nearest_indices(lats, np.array([lat]))[0])
    lon0 = int(nearest_indices(lons, np.array([lon]))[0])
    n_lat, n_lon = valid_mask.shape

    for radius in (2, 5, 10, 20, max_radius_cells):
        i0 = max(0, lat0 - radius)
        i1 = min(n_lat, lat0 + radius + 1)
        j0 = max(0, lon0 - radius)
        j1 = min(n_lon, lon0 + radius + 1)
        window = valid_mask[i0:i1, j0:j1]
        if not np.any(window):
            continue

        rel_i, rel_j = np.where(window)
        cand_i = rel_i + i0
        cand_j = rel_j + j0
        lat_diff = lats[cand_i] - lat
        lon_diff = ((lons[cand_j] - lon + 180) % 360) - 180
        best = int(np.argmin(lat_diff * lat_diff + lon_diff * lon_diff))
        return int(cand_i[best]), int(cand_j[best])

    return None


def grid_mapping_cache_path(cache_dir: Path, lats: np.ndarray, lons: np.ndarray) -> Path:
    lat0, lat1 = float(np.nanmin(lats)), float(np.nanmax(lats))
    lon0, lon1 = float(np.nanmin(lons)), float(np.nanmax(lons))
    name = (
        f"fua_grid_cells_{len(lats)}x{len(lons)}_"
        f"lat{lat0:.2f}_{lat1:.2f}_lon{lon0:.2f}_{lon1:.2f}.parquet"
    )
    return cache_dir / name.replace("-", "m")


def load_or_build_polygon_grid_map(
    lats: np.ndarray,
    lons: np.ndarray,
    polygons,
    cache_dir: Path,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = grid_mapping_cache_path(cache_dir, lats, lons)
    if cache_path.exists():
        mapping = pd.read_parquet(cache_path)
        print(f"  loaded polygon grid cache: {cache_path} ({len(mapping):,} city-grid rows)")
        return mapping

    from shapely import contains_xy

    print("  building polygon grid cache from FUA boundaries ...")
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if np.nanmin(lons) >= 0 and np.nanmax(lons) > 180:
        lons_for_geom = ((lons + 180) % 360) - 180
    else:
        lons_for_geom = lons

    rows = []
    n = len(polygons)
    for i, row in polygons.iterrows():
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        lat_idx = np.where((lats >= miny) & (lats <= maxy))[0]
        lon_idx = np.where((lons_for_geom >= minx) & (lons_for_geom <= maxx))[0]

        selected_lat_idx: np.ndarray
        selected_lon_idx: np.ndarray
        if len(lat_idx) and len(lon_idx):
            lon_mesh, lat_mesh = np.meshgrid(lons_for_geom[lon_idx], lats[lat_idx])
            inside = contains_xy(geom, lon_mesh.ravel(), lat_mesh.ravel())
            if inside.any():
                lat_mesh_idx, lon_mesh_idx = np.meshgrid(lat_idx, lon_idx, indexing="ij")
                selected_lat_idx = lat_mesh_idx.ravel()[inside].astype(np.int32)
                selected_lon_idx = lon_mesh_idx.ravel()[inside].astype(np.int32)
            else:
                selected_lat_idx = np.array([], dtype=np.int32)
                selected_lon_idx = np.array([], dtype=np.int32)
        else:
            selected_lat_idx = np.array([], dtype=np.int32)
            selected_lon_idx = np.array([], dtype=np.int32)

        # Small FUAs may not contain any 0.1-degree grid center. Keep them in
        # the panel with a nearest-cell fallback, but mark this explicitly.
        fallback = False
        if len(selected_lat_idx) == 0:
            pt = geom.representative_point()
            selected_lat_idx = nearest_indices(lats, np.array([pt.y]))
            selected_lon_idx = nearest_indices(lons_for_geom, np.array([pt.x]))
            fallback = True

        rows.append(
            pd.DataFrame(
                {
                    "city_id": row["city_id"],
                    "lat_idx": selected_lat_idx,
                    "lon_idx": selected_lon_idx,
                    "n_grid_cells": len(selected_lat_idx),
                    "fallback_nearest": fallback,
                }
            )
        )

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"    mapped {i + 1:,}/{n:,} FUA polygons")

    mapping = pd.concat(rows, ignore_index=True)
    mapping.to_parquet(cache_path, index=False)
    n_fallback = int(mapping.groupby("city_id")["fallback_nearest"].max().sum())
    print(
        f"  saved polygon grid cache: {cache_path} | "
        f"{len(mapping):,} city-grid rows | fallback cities: {n_fallback}"
    )
    return mapping


def extract_city_hourly_centroid(
    nc_path: Path,
    index: str,
    coords: pd.DataFrame,
) -> pd.DataFrame:
    with xr.open_dataset(nc_path, chunks={}) as ds:
        coord_names = set(ds.coords)
        time_name = first_present(("time", "valid_time"), coord_names, "time coordinate")
        lat_name = first_present(("lat", "latitude"), coord_names, "latitude coordinate")
        lon_name = first_present(("lon", "longitude"), coord_names, "longitude coordinate")
        var_name = detect_variable(ds, index, time_name, lat_name, lon_name)

        lats = ds[lat_name].values.astype(float)
        lons = ds[lon_name].values.astype(float)
        city_lats = coords["lat"].to_numpy(dtype=float)
        city_lons = coords["lon"].to_numpy(dtype=float)

        # Match the longitude convention of the CHI grid.
        if np.nanmin(lons) >= 0 and np.nanmax(lons) > 180:
            city_lons = city_lons % 360
        elif np.nanmin(lons) < 0:
            city_lons = ((city_lons + 180) % 360) - 180

        lat_idx = nearest_indices(lats, city_lats)
        lon_idx = nearest_indices(lons, city_lons)

        city_dim = xr.DataArray(
            np.arange(len(coords), dtype=np.int32),
            dims="city",
            coords={"city": coords["city_id"].to_numpy()},
        )

        field = ds[var_name].squeeze(drop=True)
        units = str(field.attrs.get("units", "")).strip().lower()
        selected = field.isel(
            {
                lat_name: xr.DataArray(lat_idx, dims="city", coords={"city": city_dim.city}),
                lon_name: xr.DataArray(lon_idx, dims="city", coords={"city": city_dim.city}),
            }
        )

        # Ensure the result is time x city before converting.
        selected = selected.transpose(time_name, "city")
        wide = selected.to_pandas()

    long = wide.stack().rename(index).reset_index()
    long = long.rename(columns={time_name: "datetime", "city": "city_id"})
    long["datetime"] = pd.to_datetime(long["datetime"])
    long["city_id"] = long["city_id"].astype(str).str.zfill(4)
    if should_convert_kelvin(long[index], units):
        long[index] = long[index] - 273.15
    long[index] = long[index].astype("float32")
    long = long[["city_id", "datetime", index]]
    return long


def extract_city_hourly_polygon(
    nc_path: Path,
    index: str,
    polygons,
    cache_dir: Path,
) -> pd.DataFrame:
    with xr.open_dataset(nc_path, chunks={}) as ds:
        coord_names = set(ds.coords)
        time_name = first_present(("time", "valid_time"), coord_names, "time coordinate")
        lat_name = first_present(("lat", "latitude"), coord_names, "latitude coordinate")
        lon_name = first_present(("lon", "longitude"), coord_names, "longitude coordinate")
        var_name = detect_variable(ds, index, time_name, lat_name, lon_name)

        lats = ds[lat_name].values.astype(float)
        lons = ds[lon_name].values.astype(float)
        mapping = load_or_build_polygon_grid_map(lats, lons, polygons, cache_dir)

        field = ds[var_name].squeeze(drop=True)
        units = str(field.attrs.get("units", "")).strip().lower()
        times = pd.to_datetime(ds[time_name].values)
        city_points = polygons.set_index("city_id").geometry.representative_point()

        lat_da = xr.DataArray(mapping["lat_idx"].to_numpy(dtype=np.int32), dims="cell")
        lon_da = xr.DataArray(mapping["lon_idx"].to_numpy(dtype=np.int32), dims="cell")
        selected = field.isel({lat_name: lat_da, lon_name: lon_da}).transpose(
            time_name, "cell"
        )
        cell_values = np.asarray(selected.values, dtype=np.float32)
        if should_convert_kelvin(cell_values, units):
            cell_values = cell_values - 273.15

        cell_city_ids = mapping["city_id"].astype(str).to_numpy()
        cell_wide = pd.DataFrame(cell_values, index=times, columns=cell_city_ids)
        city_wide = cell_wide.T.groupby(level=0, sort=False).mean().T

        recovered_nan_cities = 0
        unrecovered_nan_cities = 0
        all_nan_mask = ~np.isfinite(city_wide.to_numpy()).any(axis=0)
        all_nan_city_ids = city_wide.columns[all_nan_mask].tolist()
        if all_nan_city_ids:
            valid_mask = np.isfinite(field.isel({time_name: 0}).values)
            for city_id in all_nan_city_ids:
                pt = city_points.loc[city_id]
                fallback = nearest_valid_grid_cell(
                    lats=lats,
                    lons=lons,
                    valid_mask=valid_mask,
                    lat=float(pt.y),
                    lon=float(pt.x),
                )
                if fallback is None:
                    unrecovered_nan_cities += 1
                    continue

                fallback_lat_idx, fallback_lon_idx = fallback
                values = np.asarray(
                    field.isel({lat_name: fallback_lat_idx, lon_name: fallback_lon_idx}).values,
                    dtype=np.float32,
                )
                if should_convert_kelvin(values, units):
                    values = values - 273.15
                city_wide[city_id] = values.astype("float32")
                recovered_nan_cities += 1

        if recovered_nan_cities or unrecovered_nan_cities:
            print(
                f"  polygon extraction note: recovered {recovered_nan_cities:,} "
                f"coastal/all-NaN cities with nearest valid CHI grid cell; "
                f"unrecovered: {unrecovered_nan_cities:,}"
            )

    long = city_wide.reset_index(names="datetime").melt(
        id_vars="datetime",
        var_name="city_id",
        value_name=index,
    )
    long["city_id"] = long["city_id"].astype(str).str.zfill(4)
    long[index] = long[index].astype("float32")
    n_grid_cells = (
        mapping.groupby("city_id", sort=False)["n_grid_cells"]
        .first()
        .reset_index()
    )
    long = long.merge(n_grid_cells, on="city_id", how="left")
    return long[["city_id", "datetime", index, "n_grid_cells"]]


def daily_from_hourly(hourly: pd.DataFrame, index: str) -> pd.DataFrame:
    tmp = hourly.copy()
    tmp["date"] = tmp["datetime"].dt.floor("D")
    daily = (
        tmp.groupby(["city_id", "date"], sort=False)[index]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{index}_mean",
                "max": f"{index}_max",
                "min": f"{index}_min",
            }
        )
    )
    for col in [f"{index}_mean", f"{index}_max", f"{index}_min"]:
        daily[col] = daily[col].astype("float32")
    if "n_grid_cells" in tmp.columns:
        cells = tmp.groupby(["city_id", "date"], sort=False)["n_grid_cells"].max().reset_index()
        daily = daily.merge(cells, on=["city_id", "date"], how="left")
    return daily


def process_month(
    nc_path: Path,
    index: str,
    coords: pd.DataFrame | None,
    polygons,
    out_dir: Path,
    write_hourly: bool,
    spatial_method: str,
    cache_dir: Path,
) -> tuple[Path | None, Path]:
    match = re.search(r"(\d{4})-(\d{2})", nc_path.name)
    if not match:
        raise ValueError(f"Could not parse year-month from {nc_path.name}")
    year, month = int(match.group(1)), int(match.group(2))

    if spatial_method == "polygon":
        hourly = extract_city_hourly_polygon(nc_path, index, polygons, cache_dir)
    else:
        if coords is None:
            raise ValueError("Centroid extraction requires city lookup coordinates.")
        hourly = extract_city_hourly_centroid(nc_path, index, coords)
    daily = daily_from_hourly(hourly, index)

    hourly_path = None
    if write_hourly:
        hourly_path = processed_hourly_file(out_dir, spatial_method, index, year, month)
        hourly_path.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_parquet(hourly_path, index=False)

    daily_path = processed_daily_file(out_dir, spatial_method, index, year, month)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_path, index=False)
    return hourly_path, daily_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CHI monthly NetCDFs from Globus and extract FUA city series."
    )
    parser.add_argument("--indices", nargs="+", default=["Twb", "Twc", "Twbg"], help="Index names or 'all'.")
    parser.add_argument("--years", nargs="+", default=["2019-2021"], help="Years/ranges, e.g. 2019-2021.")
    parser.add_argument("--months", nargs="+", default=["1-12"], help="Months/ranges, e.g. 6-8.")
    parser.add_argument("--local-dir", default=r"D:\CHI_Data", help="Local filesystem folder for raw and processed data.")
    parser.add_argument(
        "--local-globus-root",
        default=None,
        help=(
            "Destination path as seen by Globus, e.g. /D/CHI_Data or /~/CHI_Data. "
            "If omitted, it is inferred from --local-dir on Windows."
        ),
    )
    parser.add_argument("--city-lookup", default=str(DEFAULT_LOOKUP), help="FUA city lookup CSV.")
    parser.add_argument("--shapefile", default=str(DEFAULT_SHPFILE), help="FUA polygon shapefile.")
    parser.add_argument(
        "--spatial-method",
        choices=["polygon", "centroid"],
        default="polygon",
        help="polygon averages all CHI grid cells inside each FUA; centroid uses nearest city centroid grid cell.",
    )
    parser.add_argument("--globus-cli", default=None, help="Path to globus.exe.")
    parser.add_argument("--source-endpoint", default=REMOTE_ENDPOINT)
    parser.add_argument("--local-endpoint", default=None, help="Local Globus endpoint id. Default: globus endpoint local-id.")
    parser.add_argument("--download-only", action="store_true", help="Only download NetCDFs; do not extract or delete.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw monthly NetCDF after successful extraction.")
    parser.add_argument("--write-hourly", action="store_true", default=True, help="Also save city-hour parquet (default: True). Pass --no-write-hourly to disable.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned transfers without submitting.")
    parser.add_argument("--max-files", type=int, default=None, help="Stop after N monthly files for testing.")
    parser.add_argument("--polling-interval", type=int, default=60, help="Seconds between Globus task polling.")
    parser.add_argument("--task-timeout-hours", type=float, default=24.0, help="Per-file Globus wait timeout.")
    args = parser.parse_args()

    if args.indices == ["all"]:
        indices = ALL_INDICES
    else:
        indices = []
        for idx in args.indices:
            if idx not in INDEX_INFO:
                raise ValueError(f"Unknown index {idx!r}. Allowed: {ALL_INDICES}")
            indices.append(idx)

    years = parse_int_range(args.years, 1950, 2024, "year")
    months = parse_int_range(args.months, 1, 12, "month")

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_globus_root = args.local_globus_root or infer_globus_path(local_dir)

    globus_cli = find_globus_cli(args.globus_cli)
    local_endpoint = args.local_endpoint or globus_local_endpoint(globus_cli)

    coords = None
    polygons = None
    if not args.download_only and not args.dry_run:
        if args.spatial_method == "polygon":
            polygons = load_fua_polygons(Path(args.shapefile))
        else:
            coords = load_city_lookup(Path(args.city_lookup))
    cache_dir = local_dir / "cache"

    plan = [(idx, y, m) for idx in indices for y in years for m in months]
    if args.max_files is not None:
        plan = plan[: args.max_files]

    print("=" * 72)
    print("CHI GLOBUS STAGED DOWNLOAD")
    print(f"Indices        : {indices}")
    print(f"Years          : {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"Months         : {months}")
    print(f"Files planned  : {len(plan)}")
    print(f"Local endpoint : {local_endpoint}")
    print(f"Local dir      : {local_dir}")
    print(f"Globus root    : {norm_globus_dir(local_globus_root)}")
    if args.download_only:
        mode = "download-only"
    elif args.write_hourly:
        mode = "extract daily + hourly"
    else:
        mode = "extract daily"
    print(f"Mode           : {mode}")
    print(f"Spatial method : {args.spatial_method}")
    print(f"Keep raw       : {args.keep_raw or args.download_only}")
    print("=" * 72)

    timeout_seconds = int(args.task_timeout_hours * 3600)
    done = 0
    for idx, year, month in plan:
        raw_path = local_raw_file(local_dir, idx, year, month)
        src = remote_file(idx, year, month)
        dst = local_globus_file(local_globus_root, idx, year, month)
        daily_out = processed_daily_file(local_dir, args.spatial_method, idx, year, month)
        hourly_out = processed_hourly_file(local_dir, args.spatial_method, idx, year, month)

        print(f"\n[{done + 1}/{len(plan)}] {idx} {year}-{month:02d}: {INDEX_INFO[idx]}")
        expected_size = remote_file_size(globus_cli, args.source_endpoint, src)
        if args.dry_run:
            local_state = "missing"
            if raw_path.exists():
                local_state = f"{raw_path.stat().st_size / 1e9:.2f} GB"
            print(f"  dry-run: remote size {expected_size / 1e9:.2f} GB; local raw: {local_state}")
            print(f"  dry-run: would transfer {src} -> {dst}")
            done += 1
            continue

        if (
            daily_out.exists()
            and not args.download_only
            and (not args.write_hourly or hourly_out.exists())
        ):
            print(f"  required processed output exists, skip: {daily_out}")
            if args.write_hourly:
                print(f"  required processed hourly exists, skip: {hourly_out}")
            done += 1
            continue

        if raw_path.exists():
            local_size = raw_path.stat().st_size
            if local_size == expected_size:
                print(
                    f"  raw exists locally and size matches: {raw_path} "
                    f"({local_size / 1e9:.2f} GB)"
                )
            else:
                bad_path = quarantine_file(raw_path, "partial")
                print(
                    f"  raw size mismatch: local {local_size / 1e9:.2f} GB "
                    f"!= remote {expected_size / 1e9:.2f} GB"
                )
                print(f"  moved incomplete file aside: {bad_path}")

        if not raw_path.exists():
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            task_id = submit_transfer(
                globus_cli=globus_cli,
                source_endpoint=args.source_endpoint,
                dest_endpoint=local_endpoint,
                source_path=src,
                dest_path=dst,
                label=f"CHI {idx} {year}-{month:02d}",
                dry_run=args.dry_run,
            )
            if args.dry_run:
                done += 1
                continue
            assert task_id is not None
            print(f"  Globus task: {task_id}")
            wait_for_task(globus_cli, task_id, args.polling_interval, timeout_seconds)

        if args.download_only:
            done += 1
            continue

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Transfer completed but local file was not found at {raw_path}. "
                "Check that --local-dir matches --local-globus-root."
            )

        t0 = time.time()
        try:
            hourly_out, daily_out = process_month(
                raw_path,
                idx,
                coords,
                polygons,
                local_dir,
                args.write_hourly,
                args.spatial_method,
                cache_dir,
            )
        except OSError as exc:
            if raw_path.exists():
                bad_path = quarantine_file(raw_path, "corrupt")
                print(f"  moved unreadable NetCDF aside: {bad_path}", file=sys.stderr)
            raise RuntimeError(f"NetCDF read/parse failed for {raw_path}") from exc
        elapsed = (time.time() - t0) / 60
        print(f"  saved daily: {daily_out}")
        if hourly_out:
            print(f"  saved hourly: {hourly_out}")
        print(f"  extraction elapsed: {elapsed:.1f} min")

        if not args.keep_raw:
            size_gb = raw_path.stat().st_size / 1e9
            raw_path.unlink()
            print(f"  deleted raw NetCDF ({size_gb:.2f} GB): {raw_path}")

        done += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
