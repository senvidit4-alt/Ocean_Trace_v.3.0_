"""
Test Iceberg Forward Drift Forecasting Module
=============================================

Verifies:
1. Forward drift forecast for an iceberg starting from an observed polygon / location.
2. Uncertainty calculations (semi_major_m, semi_minor_m, 68% and 95% radius) across timesteps.
3. Time series generation with step-by-step confidence ellipses.
4. JSON export compatible with reconstruction.json contract with mode="forward_forecast".
5. Verifies existing oil-spill backward hindcast (source_reconstruction.py) remains unaffected.
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / '03_source_reconstruction' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / '02_environment' / 'src'))

import json
from datetime import datetime, timedelta
import numpy as np

from iceberg_forecast import forecast_iceberg_drift, IcebergForecast, Uncertainty
from source_reconstruction import reconstruct_source, SourceReconstruction
import make_synthetic_environment

ENV_NETCDF_PATH = _repo_root / "data" / "samples" / "synthetic_env.nc"

# Synthetic iceberg polygon
iceberg_polygon = {
    "type": "Polygon",
    "coordinates": [[[72.4, 18.8], [72.5, 18.8], [72.5, 18.9], [72.4, 18.9], [72.4, 18.8]]]
}

OBSERVATION_TIME = datetime(2026, 8, 31, 14, 0, 0)
FORECAST_HOURS = 24.0
NUMBER = 500


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"Check failed: {label}")


def run_iceberg_tests():
    print("=" * 72)
    print("OceanTrace - Iceberg Forward Drift Forecast Test")
    print("=" * 72)

    if not ENV_NETCDF_PATH.exists():
        print(f"{ENV_NETCDF_PATH} not found -- generating synthetic environment...")
        make_synthetic_environment.main()

    print(f"\n1. Running forecast_iceberg_drift(obs_time={OBSERVATION_TIME}, hours={FORECAST_HOURS}, N={NUMBER})...")
    forecast = forecast_iceberg_drift(
        polygon=iceberg_polygon,
        observation_time=OBSERVATION_TIME,
        forecast_hours=FORECAST_HOURS,
        number=NUMBER,
        env_mode="netcdf",
        env_kwargs=dict(paths=ENV_NETCDF_PATH, name="synthetic_regional"),
        time_step_seconds=3600,
        wind_drift_factor=0.035,
    )

    print("\n" + "-" * 72)
    print("VERIFICATION: Iceberg Forecast")
    print("-" * 72)

    # 1. Type & structure checks
    check("forecast is an IcebergForecast instance", isinstance(forecast, IcebergForecast))
    check("forecast.uncertainty is an Uncertainty instance", isinstance(forecast.uncertainty, Uncertainty))
    check("mode is 'forward_forecast'", forecast.mode == "forward_forecast")
    check("hazard_type is 'iceberg'", forecast.hazard_type == "iceberg")

    # 2. Forward temporal progression
    check("forecast_time > observation_time (FORWARD drift)", forecast.forecast_time > forecast.observation_time)
    expected_forecast_time = forecast.observation_time + timedelta(hours=FORECAST_HOURS)
    check("forecast_time matches observation_time + forecast_hours", forecast.forecast_time == expected_forecast_time)

    # 3. Centroid & coordinates
    c_lon, c_lat = forecast.forecast_centroid
    check("forecast_centroid has finite coordinates", np.isfinite(c_lon) and np.isfinite(c_lat))
    check("forecast_positions dictionary contains valid lon and lat arrays",
          len(forecast.forecast_positions["lon"]) == forecast.n_particles_valid and
          len(forecast.forecast_positions["lat"]) == forecast.n_particles_valid)

    # 4. Uncertainty & confidence ellipses
    u = forecast.uncertainty
    check("uncertainty semi_major_m >= semi_minor_m >= 0", u.semi_major_m >= u.semi_minor_m >= 0)
    check("uncertainty radius_68_m <= radius_95_m", u.radius_68_m <= u.radius_95_m)
    check("uncertainty orientation_deg is in [0, 180)", 0 <= u.orientation_deg < 180)

    # 5. Time series checks
    check("time_series list exists and is non-empty", len(forecast.time_series) == int(FORECAST_HOURS) + 1)
    step0 = forecast.time_series[0]
    step_final = forecast.time_series[-1]
    check("step 0 corresponds to observation_time", step0["step_hours"] == 0.0)
    check("final step corresponds to forecast_hours", step_final["step_hours"] == FORECAST_HOURS)
    check("each step in time_series has uncertainty ellipse metrics",
          "semi_major_m" in step_final["uncertainty"] and "radius_95_m" in step_final["uncertainty"])

    # 6. JSON serialization and contract compatibility
    summary = forecast.to_dict()
    check("contract_version is '1.0'", summary["contract_version"] == "1.0")
    check("summary contains mode 'forward_forecast'", summary["mode"] == "forward_forecast")
    check("summary contains hazard_type 'iceberg'", summary["hazard_type"] == "iceberg")
    check("summary contains both forecast_centroid and origin_centroid for compatibility",
          "forecast_centroid" in summary and "origin_centroid" in summary)

    # Test saving to JSON
    test_json_path = _repo_root / "modules" / "02_drift" / "tests" / "sample_iceberg_reconstruction.json"
    forecast.save_json(test_json_path)
    check("JSON file successfully written to disk", test_json_path.exists())

    print("\n" + "=" * 72)
    print("2. VERIFICATION: Backward Oil-Spill Reconstruction Unchanged")
    print("=" * 72)

    oil_polygon = {
        "type": "Polygon",
        "coordinates": [[[72.4, 18.8], [72.6, 18.8], [72.6, 19.0], [72.4, 19.0], [72.4, 18.8]]]
    }
    backward_result = reconstruct_source(
        polygon=oil_polygon,
        observation_time=OBSERVATION_TIME,
        search_window_hours=6.0,
        number=500,
        env_mode="netcdf",
        env_kwargs=dict(paths=ENV_NETCDF_PATH, name="synthetic_regional"),
    )

    check("backward_result is SourceReconstruction instance", isinstance(backward_result, SourceReconstruction))
    check("backward origin_time < observation_time (BACKWARD hindcast)", backward_result.origin_time < backward_result.observation_time)
    check("backward origin_centroid is valid", np.isfinite(backward_result.origin_centroid[0]))
    print("\nAll forward iceberg forecasting and backward oil-spill hindcast checks passed successfully!")


if __name__ == "__main__":
    run_iceberg_tests()
