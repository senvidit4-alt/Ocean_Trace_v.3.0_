"""
OceanTrace - Iceberg Route-Risk Scoring Module
==============================================

Evaluates a planned vessel route against forward-forecasted iceberg trajectories and spatial uncertainty ellipses.

This is a NEW, SEPARATE module for iceberg collision risk evaluation.
Existing oil spill vessel attribution logic (evidence_fusion.py) remains completely untouched.

Methodology:
  - Consumes a ship's planned route (sequence of waypoints) and IcebergForecast output.
  - Analyzes each route segment against forecasted iceberg centroid positions and 68%/95% uncertainty ellipses.
  - Assigns per-segment risk scores, identifies critical near-miss zones, and delivers an overall route verdict.
  - Generates a human-readable explainability narrative mirroring the OceanTrace attribution report style.
"""
import sys
from pathlib import Path

# Add drift modules to sys.path
_module_dir = Path(__file__).resolve().parent
_repo_root = _module_dir.parent.parent
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / '03_source_reconstruction' / 'src'))

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

METERS_PER_DEG_LAT = 111320.0


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SegmentRisk:
    """Risk evaluation for an individual route segment between two waypoints."""
    segment_index: int
    start_point: List[float]  # [lon, lat]
    end_point: List[float]    # [lon, lat]
    segment_length_km: float
    min_distance_km: float
    closest_step_hours: float
    in_68_percent_ellipse: bool
    in_95_percent_ellipse: bool
    risk_score: float  # 0.0 (safest) to 1.0 (imminent danger)
    risk_category: str  # "Low", "Medium", "High"
    explanation: str


@dataclass
class RouteRiskResult:
    """Overall route risk evaluation report."""
    contract_version: str = "1.0"
    hazard_type: str = "iceberg"
    mode: str = "route_risk_assessment"
    overall_verdict: str = "Low"  # "Low", "Medium", "High"
    overall_risk_score: float = 0.0  # 0.0 to 1.0
    min_distance_to_hazard_km: float = 0.0
    closest_segment_index: int = 0
    in_uncertainty_zone: bool = False
    segments: List[SegmentRisk] = field(default_factory=list)
    hazard_metadata: Dict[str, Any] = field(default_factory=dict)
    narrative_explanation: str = ""

    def to_dict(self) -> dict:
        d = {
            "contract_version": self.contract_version,
            "mode": self.mode,
            "hazard_type": self.hazard_type,
            "overall_verdict": self.overall_verdict,
            "overall_risk_score": round(self.overall_risk_score, 3),
            "min_distance_to_hazard_km": round(self.min_distance_to_hazard_km, 3),
            "closest_segment_index": self.closest_segment_index,
            "in_uncertainty_zone": self.in_uncertainty_zone,
            "segments": [asdict(s) for s in self.segments],
            "hazard_metadata": self.hazard_metadata,
            "narrative_explanation": self.narrative_explanation,
        }
        return d

    def save_json(self, filepath: Union[str, Path]):
        """Save report to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Geodesic & Distance Math
# ---------------------------------------------------------------------------

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance between two coordinates in kilometers."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def point_to_segment_distance_km(
    p_lon: float, p_lat: float,
    a_lon: float, a_lat: float,
    b_lon: float, b_lat: float
) -> Tuple[float, float, float]:
    """Calculates minimum distance from point P to line segment AB, returning (dist_km, proj_lon, proj_lat)."""
    # Convert to local Cartesian coordinates (km) relative to A
    avg_lat = (a_lat + b_lat + p_lat) / 3.0
    cos_lat = math.cos(math.radians(avg_lat))
    deg_to_km = 111.32

    ax, ay = 0.0, 0.0
    bx = (b_lon - a_lon) * deg_to_km * cos_lat
    by = (b_lat - a_lat) * deg_to_km
    px = (p_lon - a_lon) * deg_to_km * cos_lat
    py = (p_lat - a_lat) * deg_to_km

    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-9:
        # Segment is essentially a single point
        d = math.sqrt(px * px + py * py)
        return d, a_lon, a_lat

    # Projection parameter t clamped to [0, 1]
    t = max(0.0, min(1.0, (px * dx + py * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy

    dist_km = math.hypot(px - proj_x, py - proj_y)
    proj_lon = a_lon + (proj_x / (deg_to_km * cos_lat))
    proj_lat = a_lat + (proj_y / deg_to_km)

    return dist_km, proj_lon, proj_lat


def point_in_ellipse(
    pt_lon: float, pt_lat: float,
    center_lon: float, center_lat: float,
    semi_major_m: float, semi_minor_m: float,
    orientation_deg: float
) -> bool:
    """Tests if (pt_lon, pt_lat) falls inside a specified spatial covariance ellipse."""
    if semi_major_m <= 0 or semi_minor_m <= 0:
        return False

    # Convert offset to local ENU meters
    ref_lat_rad = math.radians(center_lat)
    dx_m = (pt_lon - center_lon) * METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    dy_m = (pt_lat - center_lat) * METERS_PER_DEG_LAT

    # Rotate coordinates by orientation_deg (measured clockwise from True North = y-axis)
    theta_rad = math.radians(orientation_deg)
    # Along-major and along-minor components
    along_major = dx_m * math.sin(theta_rad) + dy_m * math.cos(theta_rad)
    along_minor = dx_m * math.cos(theta_rad) - dy_m * math.sin(theta_rad)

    # Standard ellipse equation (x/a)^2 + (y/b)^2 <= 1
    val = (along_major / semi_major_m) ** 2 + (along_minor / semi_minor_m) ** 2
    return val <= 1.0


# ---------------------------------------------------------------------------
# Route Risk Scoring Implementation
# ---------------------------------------------------------------------------

def score_route_risk(
    route_waypoints: List[Tuple[float, float]],
    iceberg_forecast: Any,
    time_padding_hours: float = 2.0,
    safety_margin_km: float = 5.0,
) -> RouteRiskResult:
    """Scores collision and proximity risk for a planned vessel route against a forward iceberg forecast.

    :param route_waypoints: List of (lon, lat) waypoints defining the planned voyage route.
    :param iceberg_forecast: IcebergForecast dataclass instance or reconstruction.json dict.
    :param time_padding_hours: Temporal buffer applied to forecast steps.
    :param safety_margin_km: Minimum desired navigational safety buffer distance in km.
    :return: RouteRiskResult containing segment breakdown, overall risk verdict, and explainability narrative.
    """
    if len(route_waypoints) < 2:
        raise ValueError("Route must contain at least 2 waypoints (start and destination)")

    # Extract forecast data
    if hasattr(iceberg_forecast, "to_dict"):
        f_data = iceberg_forecast.to_dict()
    elif isinstance(iceberg_forecast, dict):
        f_data = iceberg_forecast
    else:
        raise TypeError(f"Unsupported iceberg_forecast type: {type(iceberg_forecast)}")

    time_series = f_data.get("time_series", [])
    if not time_series:
        # Fallback to single forecast point if time_series is absent
        centroid = f_data.get("forecast_centroid") or f_data.get("origin_centroid", [0.0, 0.0])
        unc = f_data.get("uncertainty", {})
        time_series = [{
            "step_hours": float(f_data.get("forecast_hours", 24.0)),
            "centroid": centroid,
            "uncertainty": unc
        }]

    segments: List[SegmentRisk] = []
    global_min_dist = float("inf")
    closest_seg_idx = 0
    overall_score = 0.0
    any_in_zone = False

    for s_idx in range(len(route_waypoints) - 1):
        p1 = route_waypoints[s_idx]
        p2 = route_waypoints[s_idx + 1]
        seg_len = haversine_km(p1[0], p1[1], p2[0], p2[1])

        seg_min_dist = float("inf")
        seg_closest_step = 0.0
        in_68 = False
        in_95 = False

        # Check against all forecasted positions and uncertainty footprints along the trajectory
        for step in time_series:
            step_hrs = float(step.get("step_hours", 0.0))
            c_lon, c_lat = step.get("centroid", [0.0, 0.0])
            u_info = step.get("uncertainty", {})

            dist_km, proj_lon, proj_lat = point_to_segment_distance_km(
                c_lon, c_lat, p1[0], p1[1], p2[0], p2[1]
            )

            if dist_km < seg_min_dist:
                seg_min_dist = dist_km
                seg_closest_step = step_hrs

            # Check if closest projected point is inside 68% or 95% ellipse
            semi_maj = float(u_info.get("semi_major_m", 0.0))
            semi_min = float(u_info.get("semi_minor_m", 0.0))
            orient = float(u_info.get("orientation_deg", 0.0))
            r68 = float(u_info.get("radius_68_m", semi_maj))
            r95 = float(u_info.get("radius_95_m", semi_maj * 1.5))

            # 1. Check exact covariance ellipse (1-sigma ~ 68%)
            if point_in_ellipse(proj_lon, proj_lat, c_lon, c_lat, max(semi_maj, r68), max(semi_min, r68), orient):
                in_68 = True
            # 2. Check 95% expanded confidence ellipse/radius
            if point_in_ellipse(proj_lon, proj_lat, c_lon, c_lat, max(semi_maj * 2.0, r95), max(semi_min * 2.0, r95), orient):
                in_95 = True

            # Radius-based distance fallback check
            dist_m = dist_km * 1000.0
            if dist_m <= r68:
                in_68 = True
            if dist_m <= r95:
                in_95 = True

        # Calculate segment risk score (0.0 to 1.0)
        # Factor 1: Proximity penalty relative to safety margin
        if in_68:
            seg_risk_score = 0.85 + 0.15 * max(0.0, 1.0 - (seg_min_dist / max(safety_margin_km, 1.0)))
            seg_cat = "High"
            any_in_zone = True
            expl = (f"CRITICAL: Segment passes directly through the 68% iceberg uncertainty ellipse "
                    f"(closest approach {seg_min_dist:.2f} km at T+{seg_closest_step:.1f}h).")
        elif in_95:
            seg_risk_score = 0.50 + 0.30 * max(0.0, 1.0 - (seg_min_dist / (safety_margin_km * 2.0)))
            seg_cat = "Medium"
            any_in_zone = True
            expl = (f"CAUTION: Segment penetrates the 95% confidence dispersal buffer "
                    f"(closest approach {seg_min_dist:.2f} km at T+{seg_closest_step:.1f}h).")
        elif seg_min_dist < safety_margin_km:
            seg_risk_score = 0.30 + 0.20 * (1.0 - (seg_min_dist / safety_margin_km))
            seg_cat = "Medium"
            expl = (f"MODERATE: Segment is outside the uncertainty envelope but breaches the {safety_margin_km:.1f} km "
                    f"safety perimeter (min distance {seg_min_dist:.2f} km).")
        else:
            seg_risk_score = max(0.0, 0.25 * math.exp(-(seg_min_dist - safety_margin_km) / 10.0))
            seg_cat = "Low"
            expl = f"CLEAR: Segment maintains a safe clearance of {seg_min_dist:.2f} km from forecasted iceberg drift."

        if seg_min_dist < global_min_dist:
            global_min_dist = seg_min_dist
            closest_seg_idx = s_idx + 1

        overall_score = max(overall_score, seg_risk_score)

        segments.append(SegmentRisk(
            segment_index=s_idx + 1,
            start_point=[float(p1[0]), float(p1[1])],
            end_point=[float(p2[0]), float(p2[1])],
            segment_length_km=round(seg_len, 2),
            min_distance_km=round(seg_min_dist, 2),
            closest_step_hours=round(seg_closest_step, 1),
            in_68_percent_ellipse=in_68,
            in_95_percent_ellipse=in_95,
            risk_score=round(seg_risk_score, 3),
            risk_category=seg_cat,
            explanation=expl,
        ))

    # Determine overall verdict
    if overall_score >= 0.70:
        verdict = "High"
    elif overall_score >= 0.40:
        verdict = "Medium"
    else:
        verdict = "Low"

    # Construct explainability narrative
    narrative = _build_narrative(
        verdict=verdict,
        overall_score=overall_score,
        min_dist=global_min_dist,
        closest_seg=closest_seg_idx,
        segments=segments,
        hazard_data=f_data,
        safety_margin_km=safety_margin_km
    )

    return RouteRiskResult(
        contract_version="1.0",
        hazard_type="iceberg",
        mode="route_risk_assessment",
        overall_verdict=verdict,
        overall_risk_score=round(overall_score, 3),
        min_distance_to_hazard_km=round(global_min_dist, 2),
        closest_segment_index=closest_seg_idx,
        in_uncertainty_zone=any_in_zone,
        segments=segments,
        hazard_metadata={
            "observation_time": f_data.get("observation_time"),
            "forecast_time": f_data.get("forecast_time"),
            "forecast_hours": f_data.get("forecast_hours"),
            "safety_margin_km": safety_margin_km,
        },
        narrative_explanation=narrative,
    )


def _build_narrative(
    verdict: str,
    overall_score: float,
    min_dist: float,
    closest_seg: int,
    segments: List[SegmentRisk],
    hazard_data: dict,
    safety_margin_km: float
) -> str:
    """Generates an explainability report structured like the OceanTrace attribution report."""
    lines = [
        "================================================================================",
        "OCEANTRACE VOYAGE RISK ASSESSMENT REPORT (ICEBERG HAZARD BRANCH)",
        "================================================================================",
        f"Overall Route Verdict:       {verdict.upper()} RISK (Composite Score: {overall_score:.2f} / 1.00)",
        f"Minimum Clearance Distance:  {min_dist:.2f} km (at Route Segment {closest_seg})",
        f"Recommended Safety Margin:   {safety_margin_km:.1f} km",
        f"Target Hazard:               Iceberg (Forecast Horizon: {hazard_data.get('forecast_hours', 24.0)}h)",
        "--------------------------------------------------------------------------------",
        "EXECUTIVE SUMMARY & NAVIGATION ADVICE:",
    ]

    if verdict == "High":
        lines.append(
            f"The proposed voyage plan intersects the high-probability dispersion zone of the tracked iceberg. "
            f"Segment {closest_seg} breaches the 68% spatial covariance ellipse with a closest approach of {min_dist:.2f} km. "
            f"REROUTING IS STRONGLY RECOMMENDED to avoid potential hull collision or heavy drift pack ice."
        )
    elif verdict == "Medium":
        lines.append(
            f"The proposed route maintains separation from the primary iceberg path but enters the 95% uncertainty envelope "
            f"or breaches the {safety_margin_km:.1f} km navigational safety buffer. "
            f"Heightened radar watch, speed reduction, or a minor waypoint detour is advised."
        )
    else:
        lines.append(
            f"The planned route remains entirely clear of the forecasted iceberg trajectory and all associated "
            f"uncertainty envelopes with a minimum safety margin of {min_dist:.2f} km. The route is considered safe to proceed."
        )

    lines.append("\nSEGMENT-BY-SEGMENT BREAKDOWN:")
    for s in segments:
        lines.append(
            f"  • Segment {s.segment_index} ({s.segment_length_km:.1f} km): [{s.risk_category.upper()}] "
            f"Min dist: {s.min_distance_km:.2f} km (T+{s.closest_step_hours:.1f}h) | Score: {s.risk_score:.2f} | {s.explanation}"
        )

    lines.append("================================================================================")
    return "\n".join(lines)
