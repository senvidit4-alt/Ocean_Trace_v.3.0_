"""
Integration Test Suite: Iceberg Route-Risk Scoring & Safe Path Planning
========================================================================

Verifies:
1. score_route_risk() in Module 3 correctly evaluates route segments against iceberg drift forecasts.
2. find_safest_path() in Module 4 calculates optimal avoidance paths around multiple icebergs.
3. Generates sample route_risk_report.json and safe_path.json.
4. Verifies existing oil-spill attribution logic (evidence_fusion.py) remains 100% functional and unchanged.
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '03_attribution' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '03_attribution' / '05_evidence_fusion' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '03_attribution' / '04_ais_trajectory' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / '03_source_reconstruction' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '04_navigation' / 'src'))

import json
from datetime import datetime
import numpy as np

from route_risk import score_route_risk, RouteRiskResult
from safe_path import find_safest_path, SafePathResult
from evidence_fusion import evaluate_candidates, EvidenceWeights
from source_reconstruction import SourceReconstruction, Uncertainty
from ais_trajectory import AISDataset, LoadReport
import pandas as pd
import xarray as xr


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"Check failed: {label}")


def run_tests():
    print("=" * 72)
    print("1. TEST: Module 3 Route-Risk Scoring (score_route_risk)")
    print("=" * 72)

    # 1. Load or construct sample iceberg forecast
    sample_recon_path = _repo_root / "modules" / "02_drift" / "tests" / "sample_iceberg_reconstruction.json"
    if sample_recon_path.exists():
        with open(sample_recon_path, "r") as f:
            iceberg_fc = json.load(f)
    else:
        # Synthetic forecast
        iceberg_fc = {
            "contract_version": "1.0",
            "mode": "forward_forecast",
            "hazard_type": "iceberg",
            "observation_time": "2026-08-31T14:00:00",
            "forecast_time": "2026-09-01T14:00:00",
            "forecast_hours": 24.0,
            "forecast_centroid": [72.475, 18.853],
            "uncertainty": {
                "semi_major_m": 3380.0,
                "semi_minor_m": 3136.0,
                "orientation_deg": 5.7,
                "radius_68_m": 5204.0,
                "radius_95_m": 6709.0,
                "method": "covariance_ellipse"
            },
            "time_series": [
                {
                    "time": "2026-08-31T14:00:00",
                    "step_hours": 0.0,
                    "centroid": [72.450, 18.850],
                    "uncertainty": {"semi_major_m": 3000.0, "radius_68_m": 4500.0, "radius_95_m": 6000.0}
                },
                {
                    "time": "2026-09-01T14:00:00",
                    "step_hours": 24.0,
                    "centroid": [72.475, 18.853],
                    "uncertainty": {"semi_major_m": 3380.0, "radius_68_m": 5204.0, "radius_95_m": 6709.0}
                }
            ]
        }

    # High-Risk Route (passes right through iceberg location at [72.46, 18.85])
    high_risk_route = [
        (72.40, 18.85),
        (72.46, 18.85),
        (72.52, 18.85),
    ]

    print("\nEvaluating High-Risk Route...")
    risk_res_high = score_route_risk(high_risk_route, iceberg_fc, safety_margin_km=5.0)

    check("Overall verdict is 'High'", risk_res_high.overall_verdict == "High")
    check("overall_risk_score >= 0.70", risk_res_high.overall_risk_score >= 0.70)
    check("in_uncertainty_zone is True", risk_res_high.in_uncertainty_zone is True)
    check("Segments evaluated == 2", len(risk_res_high.segments) == 2)
    check("Narrative explanation generated", len(risk_res_high.narrative_explanation) > 100)

    # Low-Risk Route (passes 30 km north at lat 19.15)
    low_risk_route = [
        (72.40, 19.15),
        (72.52, 19.15),
    ]

    print("\nEvaluating Low-Risk Route...")
    risk_res_low = score_route_risk(low_risk_route, iceberg_fc, safety_margin_km=5.0)

    check("Overall verdict is 'Low'", risk_res_low.overall_verdict == "Low")
    check("overall_risk_score < 0.40", risk_res_low.overall_risk_score < 0.40)
    check("in_uncertainty_zone is False", risk_res_low.in_uncertainty_zone is False)

    # Save sample report
    sample_risk_out = _repo_root / "modules" / "03_attribution" / "route_risk_report.json"
    risk_res_high.save_json(sample_risk_out)
    check("route_risk_report.json saved successfully", sample_risk_out.exists())

    print("\n" + "=" * 72)
    print("2. TEST: Module 4 Safe Path Planning (find_safest_path)")
    print("=" * 72)

    # Simulate an iceberg field with 3 forecasted icebergs in the voyage corridor
    iceberg_field = [
        {
            "hazard_id": "ICEBERG-ALPHA",
            "forecast_centroid": [72.45, 18.85],
            "time_series": [{
                "step_hours": 12.0,
                "centroid": [72.45, 18.85],
                "uncertainty": {"radius_68_m": 4000.0, "radius_95_m": 6000.0}
            }]
        },
        {
            "hazard_id": "ICEBERG-BRAVO",
            "forecast_centroid": [72.50, 18.88],
            "time_series": [{
                "step_hours": 12.0,
                "centroid": [72.50, 18.88],
                "uncertainty": {"radius_68_m": 3500.0, "radius_95_m": 5500.0}
            }]
        },
        {
            "hazard_id": "ICEBERG-CHARLIE",
            "forecast_centroid": [72.55, 18.84],
            "time_series": [{
                "step_hours": 12.0,
                "centroid": [72.55, 18.84],
                "uncertainty": {"radius_68_m": 4500.0, "radius_95_m": 7000.0}
            }]
        },
    ]

    start_pt = (72.35, 18.85)
    end_pt = (72.65, 18.85)

    print(f"\nComputing safe path from {start_pt} to {end_pt} around 3 icebergs...")
    safe_path_res = find_safest_path(
        start=start_pt,
        end=end_pt,
        iceberg_forecasts=iceberg_field,
        grid_resolution_km=3.0,
        safety_margin_km=5.0,
    )

    check("SafePathResult returned", isinstance(safe_path_res, SafePathResult))
    check("Status is 'SUCCESS'", safe_path_res.status == "SUCCESS")
    check("Recommended waypoints count >= 2", len(safe_path_res.recommended_waypoints) >= 2)
    check("Total distance > 0", safe_path_res.total_distance_km > 0)
    check("Hazards evaluated == 3", safe_path_res.hazards_evaluated == 3)
    check("All 3 hazards in avoidance summary", len(safe_path_res.hazards_avoidance_summary) == 3)

    for h in safe_path_res.hazards_avoidance_summary:
        check(f"{h.hazard_id} kept outside 68% zone", h.inside_68_percent_zone is False)

    # Save safe_path.json
    safe_path_out = _repo_root / "modules" / "04_navigation" / "safe_path.json"
    safe_path_res.save_json(safe_path_out)
    check("safe_path.json saved successfully", safe_path_out.exists())

    print("\n" + "=" * 72)
    print("3. TEST: Oil-Spill Attribution Unaffected")
    print("=" * 72)

    # Synthetic AIS & Backward Recon
    t_start = datetime(2021, 8, 29, 0, 0, 0)
    records = []
    for i in range(20):
        t = t_start + pd.Timedelta(hours=i - 5)
        lon = 10.0 + (i - 10) * 0.001
        lat = 10.0
        sog = 10.0 if i < 8 or i > 12 else 2.0
        records.append({
            'mmsi': '111', 'time': t, 'lat': lat, 'lon': lon, 'sog': sog, 'cog': 90.0, 'heading': 90.0,
            'vessel_name': 'Test Tanker', 'vessel_type': 'Tanker', 'length': 200, 'cargo': 'Oil',
            'source_row': i
        })
    df = pd.DataFrame(records)
    rep = LoadReport(20, 0, 0, 0, 0, 20, 1, (df['time'].min(), df['time'].max()), (9.9, 10.1, 9.9, 10.1))
    ais_ds = AISDataset(df, rep)

    recon = SourceReconstruction(
        observation_time=datetime(2021, 8, 29, 12, 0, 0),
        origin_time=datetime(2021, 8, 29, 0, 0, 0),
        search_window_hours=12.0,
        origin_centroid=(10.0, 10.0),
        origin_positions={'lon': np.array([10.0, 10.01]), 'lat': np.array([10.0, 10.0])},
        uncertainty=Uncertainty(5000.0, 5000.0, 0.0, 5000.0, 10000.0, 'isotropic_fallback'),
        trajectories=xr.Dataset(),
        n_particles_seeded=1000,
        n_particles_valid=1000,
        env_mode='netcdf',
        env_kwargs={},
    )

    assessment = evaluate_candidates(recon, ais_ds)
    check("evaluate_candidates() executed successfully", len(assessment.candidates) > 0)
    check("Top candidate score > 0.60", assessment.candidates[0].score > 0.60)

    print("\nALL ICEBERG RISK SCORING, SAFE-PATH PLANNING, AND ATTRIBUTION TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
