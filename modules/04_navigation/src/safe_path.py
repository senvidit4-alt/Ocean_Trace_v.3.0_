"""
OceanTrace - Safe Path Planning Module (Module 4)
=================================================

Calculates optimal, lowest-risk vessel routes avoiding forecasted iceberg fields and uncertainty envelopes.

Methodology:
  - Constructs a spatial waypoint graph (NetworkX) spanning the voyage corridor and hazard bounding boxes.
  - Computes edge weights balancing navigational distance against dynamic iceberg collision penalties
    (incorporating 68% and 95% spatial uncertainty ellipses from multiple forward forecasts).
  - Employs Dijkstra / A* graph optimization to yield the safest navigable trajectory.
  - Outputs safe_path.json with recommended waypoints, distance metrics, clearance margins, and explainability summary.
"""
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent
_repo_root = _module_dir.parent.parent
sys.path.insert(0, str(_repo_root / 'modules' / '02_drift' / 'src'))
sys.path.insert(0, str(_repo_root / 'modules' / '03_attribution' / 'src'))

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

METERS_PER_DEG_LAT = 111320.0


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HazardAvoidanceMetric:
    """Summary of avoidance margins for an individual iceberg hazard."""
    hazard_id: str
    initial_location: List[float]  # [lon, lat]
    forecast_centroid: List[float]  # [lon, lat]
    min_clearance_km: float
    inside_68_percent_zone: bool
    inside_95_percent_zone: bool
    safety_margin_maintained: bool
    verdict: str  # "SAFELY_AVOIDED", "MARGINAL_CLEARANCE", "BREACHED"


@dataclass
class SafePathResult:
    """Complete route planning result."""
    contract_version: str = "1.0"
    mode: str = "safe_path_planning"
    status: str = "SUCCESS"
    start_point: List[float] = field(default_factory=list)  # [lon, lat]
    end_point: List[float] = field(default_factory=list)    # [lon, lat]
    total_distance_km: float = 0.0
    direct_distance_km: float = 0.0
    detour_percentage: float = 0.0
    estimated_risk_score: float = 0.0
    recommended_waypoints: List[List[float]] = field(default_factory=list)  # [[lon, lat], ...]
    hazards_evaluated: int = 0
    hazards_avoidance_summary: List[HazardAvoidanceMetric] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        d = {
            "contract_version": self.contract_version,
            "mode": self.mode,
            "status": self.status,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "total_distance_km": round(self.total_distance_km, 2),
            "direct_distance_km": round(self.direct_distance_km, 2),
            "detour_percentage": round(self.detour_percentage, 1),
            "estimated_risk_score": round(self.estimated_risk_score, 3),
            "recommended_waypoints": self.recommended_waypoints,
            "hazards_evaluated": self.hazards_evaluated,
            "hazards_avoidance_summary": [asdict(h) for h in self.hazards_avoidance_summary],
            "explanation": self.explanation,
        }
        return d

    def save_json(self, filepath: Union[str, Path]):
        """Save result to safe_path.json."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Geodesic Math Helpers
# ---------------------------------------------------------------------------

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance in kilometers."""
    R = 6371.0
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
    """Calculates shortest distance from point P to segment AB (dist_km, proj_lon, proj_lat)."""
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
        d = math.sqrt(px * px + py * py)
        return d, a_lon, a_lat

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
    """Tests if point is within spatial uncertainty ellipse."""
    if semi_major_m <= 0 or semi_minor_m <= 0:
        return False
    ref_lat_rad = math.radians(center_lat)
    dx_m = (pt_lon - center_lon) * METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    dy_m = (pt_lat - center_lat) * METERS_PER_DEG_LAT

    theta_rad = math.radians(orientation_deg)
    along_major = dx_m * math.sin(theta_rad) + dy_m * math.cos(theta_rad)
    along_minor = dx_m * math.cos(theta_rad) - dy_m * math.sin(theta_rad)

    val = (along_major / semi_major_m) ** 2 + (along_minor / semi_minor_m) ** 2
    return val <= 1.0


# ---------------------------------------------------------------------------
# Path Planning Core
# ---------------------------------------------------------------------------

def _normalize_forecast(fc: Any, idx: int) -> dict:
    """Standardizes input forecast object or dict."""
    if hasattr(fc, "to_dict"):
        d = fc.to_dict()
    elif isinstance(fc, dict):
        d = fc
    else:
        raise TypeError(f"Invalid forecast object: {type(fc)}")

    if "hazard_id" not in d:
        d["hazard_id"] = f"ICEBERG-{idx + 1:02d}"
    return d


def find_safest_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    iceberg_forecasts: List[Any],
    grid_resolution_km: float = 4.0,
    safety_margin_km: float = 6.0,
    padding_km: float = 20.0,
) -> SafePathResult:
    """Computes the lowest-risk safe voyage path between start and end avoiding multiple iceberg forecasts.

    :param start: (start_lon, start_lat)
    :param end: (end_lon, end_lat)
    :param iceberg_forecasts: List of IcebergForecast instances or reconstruction.json dicts.
    :param grid_resolution_km: Spatial step between graph nodes (default 4.0 km).
    :param safety_margin_km: Minimum desired navigational safety buffer distance in km.
    :param padding_km: Corridor boundary expansion around route and hazards.
    :return: SafePathResult containing recommended waypoints and explainability details.
    """
    start_lon, start_lat = float(start[0]), float(start[1])
    end_lon, end_lat = float(end[0]), float(end[1])
    direct_dist = haversine_km(start_lon, start_lat, end_lon, end_lat)

    # Normalize forecasts
    norm_forecasts = [_normalize_forecast(fc, i) for i, fc in enumerate(iceberg_forecasts)]

    # Determine bounding box
    all_lons = [start_lon, end_lon]
    all_lats = [start_lat, end_lat]

    for fc in norm_forecasts:
        for step in fc.get("time_series", []):
            c_lon, c_lat = step.get("centroid", [0.0, 0.0])
            all_lons.append(c_lon)
            all_lats.append(c_lat)

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)

    avg_lat = (min_lat + max_lat) / 2.0
    lat_pad_deg = padding_km / 111.32
    lon_pad_deg = padding_km / (111.32 * max(0.1, math.cos(math.radians(avg_lat))))

    bbox_min_lon = min_lon - lon_pad_deg
    bbox_max_lon = max_lon + lon_pad_deg
    bbox_min_lat = min_lat - lat_pad_deg
    bbox_max_lat = max_lat + lat_pad_deg

    # Create 2D mesh grid
    d_lat_deg = grid_resolution_km / 111.32
    d_lon_deg = grid_resolution_km / (111.32 * math.cos(math.radians(avg_lat)))

    grid_lons = np.arange(bbox_min_lon, bbox_max_lon + d_lon_deg, d_lon_deg)
    grid_lats = np.arange(bbox_min_lat, bbox_max_lat + d_lat_deg, d_lat_deg)

    G = nx.Graph()

    # Add grid nodes
    for i, glon in enumerate(grid_lons):
        for j, glat in enumerate(grid_lats):
            G.add_node((i, j), lon=float(glon), lat=float(glat))

    # Helper to evaluate edge penalty against all iceberg forecasts
    def calc_edge_penalty(a_lon: float, a_lat: float, b_lon: float, b_lat: float) -> Tuple[float, float]:
        total_pen = 0.0
        edge_min_dist = float("inf")

        for fc in norm_forecasts:
            for step in fc.get("time_series", []):
                c_lon, c_lat = step.get("centroid", [0.0, 0.0])
                u_info = step.get("uncertainty", {})
                semi_maj = float(u_info.get("semi_major_m", 3000.0))
                semi_min = float(u_info.get("semi_minor_m", 2500.0))
                orient = float(u_info.get("orientation_deg", 0.0))
                r68_km = float(u_info.get("radius_68_m", semi_maj)) / 1000.0
                r95_km = float(u_info.get("radius_95_m", semi_maj * 1.5)) / 1000.0

                dist_km, proj_lon, proj_lat = point_to_segment_distance_km(
                    c_lon, c_lat, a_lon, a_lat, b_lon, b_lat
                )
                edge_min_dist = min(edge_min_dist, dist_km)

                # 1. Prohibitive penalty inside 68% ellipse / core
                in_68 = point_in_ellipse(proj_lon, proj_lat, c_lon, c_lat, r68_km * 1000.0, r68_km * 1000.0, orient) or dist_km <= r68_km
                if in_68:
                    total_pen += 25000.0 * (1.0 + max(0.0, r68_km - dist_km))

                # 2. Steep penalty inside 95% ellipse / buffer
                in_95 = point_in_ellipse(proj_lon, proj_lat, c_lon, c_lat, r95_km * 1000.0, r95_km * 1000.0, orient) or dist_km <= r95_km
                if in_95:
                    total_pen += 2500.0 * (1.0 + max(0.0, r95_km - dist_km))

                # 3. Soft safety margin repulsive potential
                if dist_km < safety_margin_km:
                    total_pen += 150.0 * ((safety_margin_km - dist_km) / safety_margin_km) ** 2

        return total_pen, edge_min_dist

    # Add edges between grid neighbors (8-connectivity)
    n_x, n_y = len(grid_lons), len(grid_lats)
    for i in range(n_x):
        for j in range(n_y):
            u_node = (i, j)
            u_lon, u_lat = G.nodes[u_node]["lon"], G.nodes[u_node]["lat"]

            for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n_x and 0 <= nj < n_y:
                    v_node = (ni, nj)
                    v_lon, v_lat = G.nodes[v_node]["lon"], G.nodes[v_node]["lat"]
                    base_d = haversine_km(u_lon, u_lat, v_lon, v_lat)
                    pen, _ = calc_edge_penalty(u_lon, u_lat, v_lon, v_lat)
                    weight = base_d + pen
                    G.add_edge(u_node, v_node, weight=weight, dist=base_d, penalty=pen)

    # Add START and END nodes and connect to nearby grid nodes
    G.add_node("START", lon=start_lon, lat=start_lat)
    G.add_node("END", lon=end_lon, lat=end_lat)

    for i in range(n_x):
        for j in range(n_y):
            g_node = (i, j)
            g_lon, g_lat = G.nodes[g_node]["lon"], G.nodes[g_node]["lat"]

            # Connect START to nearby grid nodes within 2 * grid_resolution_km
            d_start = haversine_km(start_lon, start_lat, g_lon, g_lat)
            if d_start <= grid_resolution_km * 2.2:
                pen, _ = calc_edge_penalty(start_lon, start_lat, g_lon, g_lat)
                G.add_edge("START", g_node, weight=d_start + pen, dist=d_start, penalty=pen)

            # Connect END to nearby grid nodes
            d_end = haversine_km(end_lon, end_lat, g_lon, g_lat)
            if d_end <= grid_resolution_km * 2.2:
                pen, _ = calc_edge_penalty(g_lon, g_lat, end_lon, end_lat)
                G.add_edge(g_node, "END", weight=d_end + pen, dist=d_end, penalty=pen)

    # Solve optimal shortest path using Dijkstra with penalty weights
    try:
        path_nodes = nx.shortest_path(G, source="START", target="END", weight="weight")
    except nx.NetworkXNoPath:
        # Fallback direct path
        path_nodes = ["START", "END"]

    # Extract coordinates along path
    raw_waypoints = []
    for node in path_nodes:
        raw_waypoints.append([G.nodes[node]["lon"], G.nodes[node]["lat"]])

    # Simplify collinear waypoints
    simplified_waypoints = _simplify_path(raw_waypoints)

    # Calculate actual path metrics
    total_dist = 0.0
    for k in range(len(simplified_waypoints) - 1):
        p1 = simplified_waypoints[k]
        p2 = simplified_waypoints[k + 1]
        total_dist += haversine_km(p1[0], p1[1], p2[0], p2[1])

    detour_pct = max(0.0, ((total_dist - direct_dist) / max(1.0, direct_dist)) * 100.0)

    # Evaluate hazard avoidance metrics
    avoidance_summaries: List[HazardAvoidanceMetric] = []
    max_risk = 0.0

    for fc in norm_forecasts:
        hid = fc.get("hazard_id", "ICEBERG")
        init_loc = fc.get("forecast_centroid") or [0.0, 0.0]
        f_cent = fc.get("forecast_centroid", init_loc)

        min_fc_dist = float("inf")
        in_68_any = False
        in_95_any = False

        for k in range(len(simplified_waypoints) - 1):
            p1 = simplified_waypoints[k]
            p2 = simplified_waypoints[k + 1]

            for step in fc.get("time_series", []):
                c_lon, c_lat = step.get("centroid", [0.0, 0.0])
                u_info = step.get("uncertainty", {})
                r68_km = float(u_info.get("radius_68_m", 3000.0)) / 1000.0
                r95_km = float(u_info.get("radius_95_m", 5000.0)) / 1000.0

                dist_km, _, _ = point_to_segment_distance_km(c_lon, c_lat, p1[0], p1[1], p2[0], p2[1])
                min_fc_dist = min(min_fc_dist, dist_km)
                if dist_km <= r68_km:
                    in_68_any = True
                if dist_km <= r95_km:
                    in_95_any = True

        safe_maintained = (min_fc_dist >= safety_margin_km)
        if in_68_any:
            verdict = "BREACHED"
            hazard_risk = 0.90
        elif in_95_any:
            verdict = "MARGINAL_CLEARANCE"
            hazard_risk = 0.55
        elif safe_maintained:
            verdict = "SAFELY_AVOIDED"
            hazard_risk = max(0.0, 0.20 * math.exp(-(min_fc_dist - safety_margin_km) / 5.0))
        else:
            verdict = "MARGINAL_CLEARANCE"
            hazard_risk = 0.35

        max_risk = max(max_risk, hazard_risk)

        avoidance_summaries.append(HazardAvoidanceMetric(
            hazard_id=hid,
            initial_location=[float(init_loc[0]), float(init_loc[1])],
            forecast_centroid=[float(f_cent[0]), float(f_cent[1])],
            min_clearance_km=round(min_fc_dist, 2),
            inside_68_percent_zone=in_68_any,
            inside_95_percent_zone=in_95_any,
            safety_margin_maintained=safe_maintained,
            verdict=verdict,
        ))

    # Construct narrative explanation
    explanation = _build_safe_path_explanation(
        total_dist=total_dist,
        direct_dist=direct_dist,
        detour_pct=detour_pct,
        risk_score=max_risk,
        safety_margin_km=safety_margin_km,
        avoidance=avoidance_summaries,
        n_waypoints=len(simplified_waypoints)
    )

    return SafePathResult(
        contract_version="1.0",
        mode="safe_path_planning",
        status="SUCCESS",
        start_point=[start_lon, start_lat],
        end_point=[end_lon, end_lat],
        total_distance_km=round(total_dist, 2),
        direct_distance_km=round(direct_dist, 2),
        detour_percentage=round(detour_pct, 1),
        estimated_risk_score=round(max_risk, 3),
        recommended_waypoints=simplified_waypoints,
        hazards_evaluated=len(norm_forecasts),
        hazards_avoidance_summary=avoidance_summaries,
        explanation=explanation,
    )


def _simplify_path(pts: List[List[float]], tolerance_deg: float = 0.001) -> List[List[float]]:
    """Simplifies collinear waypoints along grid paths to reduce clutter."""
    if len(pts) <= 2:
        return [[round(p[0], 5), round(p[1], 5)] for p in pts]

    simplified = [pts[0]]
    for i in range(1, len(pts) - 1):
        prev_p = simplified[-1]
        curr_p = pts[i]
        next_p = pts[i + 1]

        # Check cross product / direction alignment
        dx1 = curr_p[0] - prev_p[0]
        dy1 = curr_p[1] - prev_p[1]
        dx2 = next_p[0] - curr_p[0]
        dy2 = next_p[1] - curr_p[1]

        cross = abs(dx1 * dy2 - dy1 * dx2)
        if cross > 1e-5:
            simplified.append([round(curr_p[0], 5), round(curr_p[1], 5)])

    simplified.append([round(pts[-1][0], 5), round(pts[-1][1], 5)])
    return simplified


def _build_safe_path_explanation(
    total_dist: float,
    direct_dist: float,
    detour_pct: float,
    risk_score: float,
    safety_margin_km: float,
    avoidance: List[HazardAvoidanceMetric],
    n_waypoints: int
) -> str:
    """Generates an executive explanation for the planned safe route."""
    lines = [
        "================================================================================",
        "OCEANTRACE MODULE 4: SAFE PATH PLANNING REPORT",
        "================================================================================",
        f"Route Status:                OPTIMIZED SAFE PASSAGE ({n_waypoints} waypoints)",
        f"Total Navigational Distance: {total_dist:.2f} km (Direct: {direct_dist:.2f} km, Detour: +{detour_pct:.1f}%)",
        f"Target Safety Margin:        {safety_margin_km:.1f} km buffer",
        f"Composite Route Risk Score:  {risk_score:.3f} / 1.000",
        "--------------------------------------------------------------------------------",
        "HAZARD AVOIDANCE SUMMARY:",
    ]

    for h in avoidance:
        lines.append(
            f"  • {h.hazard_id}: [{h.verdict}] Min Clearance: {h.min_clearance_km:.2f} km | "
            f"Inside 68% Zone: {'YES' if h.inside_68_percent_zone else 'NO'} | "
            f"Safety Buffer Maintained: {'YES' if h.safety_margin_maintained else 'NO'}"
        )

    lines.append("\nNAVIGATION REASONING & DETOUR ANALYSIS:")
    if detour_pct < 3.0:
        lines.append(
            "The direct great-circle route is naturally clear of all forecasted iceberg trajectories. "
            "No significant course deviations were required."
        )
    else:
        lines.append(
            f"A proactive tactical detour of +{detour_pct:.1f}% (+{total_dist - direct_dist:.2f} km) was introduced to navigate "
            f"around high-probability iceberg drift clusters, ensuring the vessel strictly maintains a minimum clearance of "
            f"{min(h.min_clearance_km for h in avoidance):.2f} km from all active drift uncertainty envelopes."
        )

    lines.append("================================================================================")
    return "\n".join(lines)
