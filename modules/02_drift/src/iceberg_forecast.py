"""
OceanTrace - Iceberg Forward Drift Forecasting Module
======================================================

observed iceberg polygon/location -> forward drift forecast + uncertainty ellipses + time series

This is a NEW, SEPARATE module for forward iceberg drift forecasting.
Existing oil spill backward-hindcast logic (source_reconstruction.py) remains completely untouched.

Physics & Drift Modeling:
  - Uses OpenDrift's OceanDrift model with iceberg-appropriate drift properties.
  - Icebergs have a significant above-water sail area and respond strongly to 10m wind forcing
    in addition to ocean surface currents (higher wind_drift_factor than surface oil slicks).
  - Runs FORWARD in time: o.run(steps=N, time_step=+dt, time_step_output=+dt).
  - Provides 68% and 95% confidence/uncertainty ellipses per time step along the forecasted trajectory.
  - Compatible with the project-wide reconstruction.json contract with mode="forward_forecast".
"""
import sys
from pathlib import Path

# Add environment and source_reconstruction search paths
_module_dir = Path(__file__).resolve().parent
_repo_root = _module_dir.parent.parent
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / '02_environment' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / '03_source_reconstruction' / 'src'))

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from opendrift.models.oceandrift import OceanDrift
from environment_readers import get_readers

logger = logging.getLogger(__name__)

OBSERVATION_TIME_INDEX = 0
FORECAST_TIME_INDEX = -1

METERS_PER_DEG_LAT = 111320.0


# ---------------------------------------------------------------------------
# Return Types & Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Uncertainty:
    """Spatial uncertainty of a particle cloud at a given timestep.
    semi_major_m, semi_minor_m: 1-sigma covariance ellipse semi-axes in meters.
    orientation_deg: bearing of major axis in degrees clockwise from true North.
    radius_68_m, radius_95_m: 68th and 95th percentile distance radii from centroid.
    method: 'covariance_ellipse' (normal) or 'isotropic_fallback' (degenerate cloud).
    """
    semi_major_m: float
    semi_minor_m: float
    orientation_deg: float
    radius_68_m: float
    radius_95_m: float
    method: str


@dataclass
class TimeStepForecast:
    """Forecast state at an individual simulation timestep."""
    time: str
    step_hours: float
    centroid: List[float]
    uncertainty: Uncertainty
    n_particles: int

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class IcebergForecast:
    """Complete forward forecast result for an iceberg."""

    observation_time: datetime
    forecast_time: datetime
    forecast_hours: float

    forecast_centroid: Tuple[float, float]  # (lon, lat) at final forecast time
    forecast_positions: Dict[str, np.ndarray]  # {'lon': np.ndarray, 'lat': np.ndarray}
    uncertainty: Uncertainty  # Uncertainty at final forecast time

    time_series: List[Dict[str, Any]]  # Step-by-step centroid and uncertainty

    trajectories: xr.Dataset  # Full xarray dataset containing all particle paths

    n_particles_seeded: int
    n_particles_valid: int

    env_mode: str
    env_kwargs: dict

    mode: str = "forward_forecast"
    hazard_type: str = "iceberg"
    coverage_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable summary conforming to reconstruction.json contract."""
        d = {
            "contract_version": "1.0",
            "mode": self.mode,
            "hazard_type": self.hazard_type,
            "observation_time": self.observation_time.isoformat(),
            "forecast_time": self.forecast_time.isoformat(),
            # Backward-compatible fields for generic reconstruction readers:
            "origin_time": self.observation_time.isoformat(),
            "search_window_hours": self.forecast_hours,
            "forecast_hours": self.forecast_hours,
            "forecast_centroid": [float(self.forecast_centroid[0]), float(self.forecast_centroid[1])],
            "origin_centroid": [float(self.forecast_centroid[0]), float(self.forecast_centroid[1])],
            "forecast_positions": {
                "lon": [float(x) for x in self.forecast_positions["lon"]],
                "lat": [float(y) for y in self.forecast_positions["lat"]],
            },
            "origin_positions": {
                "lon": [float(x) for x in self.forecast_positions["lon"]],
                "lat": [float(y) for y in self.forecast_positions["lat"]],
            },
            "uncertainty": asdict(self.uncertainty),
            "time_series": self.time_series,
            "n_particles_seeded": self.n_particles_seeded,
            "n_particles_valid": self.n_particles_valid,
            "env_mode": self.env_mode,
            "coverage_warnings": self.coverage_warnings,
        }
        return d

    def save_json(self, filepath: Union[str, Path]):
        """Save forecast summary to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Polygon Parsing & Geometry Helpers
# ---------------------------------------------------------------------------

def _extract_lon_lat(polygon: Union[dict, str]) -> Tuple[List[float], List[float]]:
    """Accepts a GeoJSON geometry dict, a Feature dict, or a geojson string.
    Returns (lons, lats) as plain lists without duplicate closing points.
    Also supports Point geometry for point-source icebergs.
    """
    if isinstance(polygon, str):
        polygon = json.loads(polygon)

    if polygon.get("type") == "Feature":
        geometry = polygon["geometry"]
    else:
        geometry = polygon

    geom_type = geometry.get("type")

    if geom_type == "Polygon":
        ring = geometry["coordinates"][0]
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        lons = [float(c[0]) for c in ring]
        lats = [float(c[1]) for c in ring]
        return lons, lats

    elif geom_type == "Point":
        coords = geometry["coordinates"]
        return [float(coords[0])], [float(coords[1])]

    elif geom_type == "MultiPolygon":
        # Use largest exterior ring
        largest_ring = max(geometry["coordinates"], key=lambda poly: len(poly[0]))[0]
        if len(largest_ring) > 1 and largest_ring[0] == largest_ring[-1]:
            largest_ring = largest_ring[:-1]
        lons = [float(c[0]) for c in largest_ring]
        lats = [float(c[1]) for c in largest_ring]
        return lons, lats

    else:
        raise ValueError(f"Unsupported geometry type '{geom_type}' for iceberg forecast")


# ---------------------------------------------------------------------------
# Uncertainty Calculations
# ---------------------------------------------------------------------------

def _to_local_meters(lons, lats, ref_lon, ref_lat) -> Tuple[np.ndarray, np.ndarray]:
    """Flat-Earth local ENU projection for local uncertainty analysis."""
    east_m = (np.asarray(lons) - ref_lon) * METERS_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    north_m = (np.asarray(lats) - ref_lat) * METERS_PER_DEG_LAT
    return east_m, north_m


def compute_uncertainty(lons, lats, centroid_lon, centroid_lat) -> Uncertainty:
    """Calculates covariance ellipse (1-sigma) and 68%/95% percentile radii in meters."""
    if len(lons) == 0:
        return Uncertainty(0.0, 0.0, 0.0, 0.0, 0.0, "empty_cloud")

    east_m, north_m = _to_local_meters(lons, lats, centroid_lon, centroid_lat)
    distances_m = np.sqrt(east_m ** 2 + north_m ** 2)
    radius_68 = float(np.percentile(distances_m, 68))
    radius_95 = float(np.percentile(distances_m, 95))

    if len(east_m) < 3 or (np.allclose(np.std(east_m), 0) and np.allclose(np.std(north_m), 0)):
        iso = float(np.std(distances_m)) if len(distances_m) > 0 else 0.0
        return Uncertainty(
            semi_major_m=iso,
            semi_minor_m=iso,
            orientation_deg=0.0,
            radius_68_m=radius_68,
            radius_95_m=radius_95,
            method="isotropic_fallback",
        )

    cov = np.cov(np.vstack([east_m, north_m]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    semi_minor_m, semi_major_m = np.sqrt(eigvals[0]), np.sqrt(eigvals[1])
    major_vec = eigvecs[:, 1]  # (east, north) components of major axis
    orientation_deg = (math.degrees(math.atan2(major_vec[0], major_vec[1]))) % 180

    return Uncertainty(
        semi_major_m=float(semi_major_m),
        semi_minor_m=float(semi_minor_m),
        orientation_deg=float(orientation_deg),
        radius_68_m=radius_68,
        radius_95_m=radius_95,
        method="covariance_ellipse",
    )


# ---------------------------------------------------------------------------
# Coverage Verification
# ---------------------------------------------------------------------------

def _check_coverage(readers, observation_time, forecast_time, lons, lats) -> List[str]:
    """Verifies environmental reader temporal and spatial coverage."""
    warnings = []
    if not readers:
        warnings.append("No environment readers attached -- run used fallback constant values.")
        return warnings

    for r in readers:
        name = getattr(r, "name", repr(r))
        for label, t in (("observation_time", observation_time), ("forecast_time", forecast_time)):
            t_cmp = t
            if r.start_time is not None and getattr(r.start_time, "tzinfo", None) is None and getattr(t_cmp, "tzinfo", None) is not None:
                t_cmp = t_cmp.astimezone(timezone.utc).replace(tzinfo=None)
            elif r.start_time is not None and getattr(r.start_time, "tzinfo", None) is not None and getattr(t_cmp, "tzinfo", None) is None:
                t_cmp = t_cmp.replace(tzinfo=timezone.utc)
            if not r.covers_time(t_cmp):
                warnings.append(
                    f"Reader '{name}' does not cover {label}={t} (range: {r.start_time} .. {r.end_time})."
                )

        covered_idx, _, _ = r.covers_positions(np.asarray(lons), np.asarray(lats))
        if len(covered_idx) < len(lons):
            n_outside = len(lons) - len(covered_idx)
            warnings.append(
                f"Reader '{name}' does not spatially cover {n_outside} of {len(lons)} initial positions."
            )
    return warnings


def pd_to_datetime(value) -> datetime:
    """Converts numpy.datetime64 / pandas Timestamp to plain datetime."""
    import pandas as pd
    return pd.Timestamp(value).to_pydatetime()


# ---------------------------------------------------------------------------
# Main Iceberg Forward Drift Function
# ---------------------------------------------------------------------------

def forecast_iceberg_drift(
    polygon: Union[dict, str],
    observation_time: datetime,
    forecast_hours: float = 48.0,
    number: int = 1000,
    env_mode: str = "netcdf",
    env_kwargs: Optional[dict] = None,
    time_step_seconds: int = 3600,
    wind_drift_factor: float = 0.035,
    current_drift_factor: float = 1.0,
    loglevel: int = 20,
) -> IcebergForecast:
    """Runs a forward drift forecast for an iceberg starting at observation_time.

    :param polygon: GeoJSON Polygon, MultiPolygon, or Point geometry/feature representing iceberg location.
    :param observation_time: Initial observed timestamp (temporal anchor).
    :param forecast_hours: Forward simulation horizon in hours (e.g. 24.0, 48.0, 72.0).
    :param number: Number of Monte Carlo particles seeded to evaluate spatial dispersion.
    :param env_mode: Environmental data provider ('netcdf', 'copernicus', 'synthetic', etc.).
    :param env_kwargs: Keyword arguments for environment_readers.get_readers().
    :param time_step_seconds: Forward simulation timestep in seconds (default 3600s = 1 hour).
    :param wind_drift_factor: Wind drift factor for iceberg sail area (default 0.035 = 3.5%).
    :param current_drift_factor: Ocean current velocity factor (default 1.0).
    :param loglevel: Logging level for OpenDrift.
    :return: IcebergForecast object containing forecast state, time series, and uncertainty ellipses.
    """
    env_kwargs = dict(env_kwargs or {})
    lons, lats = _extract_lon_lat(polygon)

    # Initialize OpenDrift OceanDrift model
    o = OceanDrift(loglevel=loglevel)

    # Iceberg-tailored drift configuration
    o.set_config("general:use_auto_landmask", False)
    o.set_config("environment:fallback:land_binary_mask", 0)
    o.set_config("environment:fallback:x_wind", 0.0)
    o.set_config("environment:fallback:y_wind", 0.0)
    o.set_config("environment:fallback:x_sea_water_velocity", 0.0)
    o.set_config("environment:fallback:y_sea_water_velocity", 0.0)

    # Set iceberg wind and current drift response
    try:
        o.set_config("drift:wind_drift_factor", wind_drift_factor)
        o.set_config("drift:current_drift_factor", current_drift_factor)
    except Exception as e:
        logger.debug(f"Configuring drift factors: {e}")

    # Attach environmental readers
    readers = get_readers(env_mode, **env_kwargs)
    if readers:
        o.add_reader(readers)

    # Ensure timezone-naive UTC for OpenDrift internal reader compatibility
    seed_time = observation_time
    if seed_time is not None and getattr(seed_time, "tzinfo", None) is not None:
        seed_time = seed_time.astimezone(timezone.utc).replace(tzinfo=None)

    # Seed particles within the iceberg polygon or at point location
    if len(lons) == 1 and len(lats) == 1:
        # Point location: seed with small radius around point
        o.seed_elements(lon=lons[0], lat=lats[0], radius=500, time=seed_time, number=number)
    else:
        o.seed_within_polygon(lons=lons, lats=lats, time=seed_time, number=number)

    n_seeded = o.num_elements_scheduled()

    # Calculate steps for positive forward run
    steps = max(1, round(forecast_hours * 3600 / time_step_seconds))
    if not math.isclose(steps * time_step_seconds, forecast_hours * 3600, abs_tol=1):
        logger.warning(
            f"forecast_hours={forecast_hours} is not an exact multiple of "
            f"time_step_seconds={time_step_seconds}s; rounded to {steps} steps "
            f"({steps * time_step_seconds / 3600:.2f}h)."
        )

    # RUN FORWARD IN TIME (positive time_step)
    o.run(steps=steps, time_step=time_step_seconds, time_step_output=time_step_seconds)

    # In forward run, time axis is ascending:
    # index 0 is observation_time, index -1 is final forecast_time
    num_times = len(o.result.time.values)
    time_series = []

    for t_idx in range(num_times):
        t_val = pd_to_datetime(o.result.time.values[t_idx])
        lons_step = o.result.lon.isel(time=t_idx).values.astype(float)
        lats_step = o.result.lat.isel(time=t_idx).values.astype(float)
        val_mask = ~np.isnan(lons_step) & ~np.isnan(lats_step)
        if np.any(val_mask):
            c_lon = float(np.mean(lons_step[val_mask]))
            c_lat = float(np.mean(lats_step[val_mask]))
            u_step = compute_uncertainty(lons_step[val_mask], lats_step[val_mask], c_lon, c_lat)
            n_part = int(val_mask.sum())
        else:
            c_lon, c_lat = float(lons[0]), float(lats[0])
            u_step = Uncertainty(0.0, 0.0, 0.0, 0.0, 0.0, "no_particles")
            n_part = 0

        time_series.append({
            "time": t_val.isoformat(),
            "step_hours": round(t_idx * (time_step_seconds / 3600.0), 2),
            "centroid": [c_lon, c_lat],
            "uncertainty": asdict(u_step),
            "n_particles": n_part,
        })

    # Final forecast positions
    forecast_lon = o.result.lon.isel(time=FORECAST_TIME_INDEX).values.astype(float)
    forecast_lat = o.result.lat.isel(time=FORECAST_TIME_INDEX).values.astype(float)
    valid_final = ~np.isnan(forecast_lon) & ~np.isnan(forecast_lat)
    n_valid = int(valid_final.sum())
    forecast_lon_valid = forecast_lon[valid_final]
    forecast_lat_valid = forecast_lat[valid_final]

    observation_time_out = pd_to_datetime(o.result.time.values[OBSERVATION_TIME_INDEX])
    forecast_time_out = pd_to_datetime(o.result.time.values[FORECAST_TIME_INDEX])

    if n_valid > 0:
        centroid_lon = float(np.mean(forecast_lon_valid))
        centroid_lat = float(np.mean(forecast_lat_valid))
        final_uncertainty = compute_uncertainty(forecast_lon_valid, forecast_lat_valid, centroid_lon, centroid_lat)
    else:
        centroid_lon = float(lons[0])
        centroid_lat = float(lats[0])
        final_uncertainty = Uncertainty(0.0, 0.0, 0.0, 0.0, 0.0, "no_particles")

    coverage_warnings = _check_coverage(readers, observation_time_out, forecast_time_out, lons, lats)
    for w in coverage_warnings:
        logger.warning(w)

    return IcebergForecast(
        observation_time=observation_time_out,
        forecast_time=forecast_time_out,
        forecast_hours=steps * time_step_seconds / 3600.0,
        forecast_centroid=(centroid_lon, centroid_lat),
        forecast_positions={"lon": forecast_lon_valid, "lat": forecast_lat_valid},
        uncertainty=final_uncertainty,
        time_series=time_series,
        trajectories=o.result,
        n_particles_seeded=n_seeded,
        n_particles_valid=n_valid,
        env_mode=env_mode,
        env_kwargs=env_kwargs,
        mode="forward_forecast",
        hazard_type="iceberg",
        coverage_warnings=coverage_warnings,
    )
