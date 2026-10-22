#!/usr/bin/env python3
"""
Acceptance Criterion 4 Programmatic Collision-Detection & Spatial Clearance Verifier.

Mathematically proves:
1. The minimum 3D Euclidean distance between every ray segment and every unrelated mirror housing center
   is strictly greater than 15.0 mm for both Broadband NIR (800-2700 nm) and Raman (1064 nm) layouts.
2. Checks for physical beam obstruction against all mirror substrate solid volumes and mount envelopes.

Usage:
    python spectrometer_v3_nir_raman/verify_collision_clearance.py
    python spectrometer_v3_nir_raman/verify_collision_clearance.py --nir <path> --raman <path> --threshold 15.0 --rings 5
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Ensure repository root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

V2_ROOT = os.path.join(PROJECT_ROOT, "v2")
if V2_ROOT not in sys.path:
    sys.path.insert(0, V2_ROOT)

from v2.engine.models import SpectrometerSystem
from v2.engine.tracer import OpticalSystem, check_segment_surface_interference
from v2.engine.surface import Surface, SurfaceType
from v2.engine.ray import Ray


MIN_CLEARANCE_THRESHOLD_MM = 15.0000  # Acceptance Criterion 4: strictly > 15.0 mm


def point_to_segment_distance_3d(
    p_a: np.ndarray,
    p_b: np.ndarray,
    point: np.ndarray
) -> Tuple[float, np.ndarray, float]:
    """
    Compute shortest 3D Euclidean distance from point C to line segment [A, B].
    
    Returns:
        (distance, closest_point_on_segment, projection_parameter_t_in_[0,1])
    """
    ab = p_b - p_a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        dist = float(np.linalg.norm(point - p_a))
        return dist, p_a.copy(), 0.0

    t = float(np.dot(point - p_a, ab) / ab_len_sq)
    t_clamped = max(0.0, min(1.0, t))
    closest_pt = p_a + t_clamped * ab
    dist = float(np.linalg.norm(point - closest_pt))
    return dist, closest_pt, t_clamped


def verify_layout_collision_clearance(
    filepath: str,
    layout_name: str,
    threshold_mm: float = MIN_CLEARANCE_THRESHOLD_MM,
    rings: int = 5,
    beam_radius: float = 5.0
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Trace full 3D beam and verify spatial clearance to all unrelated mirror housing centers.
    """
    messages = []
    if not os.path.exists(filepath):
        messages.append(f"[FAIL] Layout file does not exist: {filepath}")
        return False, {"error": f"File not found: {filepath}"}, messages

    try:
        system = SpectrometerSystem.load_from_json(filepath)
    except Exception as e:
        messages.append(f"[FAIL] Failed to load JSON prescription: {e}")
        return False, {"error": str(e)}, messages

    opt_sys = OpticalSystem(system.surface_stack)
    opt_sys._update_accumulated_positions()
    surfaces = opt_sys.surfaces

    # 1. Identify all reflective / mirror elements and their global housing centers
    mirror_surfaces: List[Tuple[int, Surface, np.ndarray]] = []
    for idx, s in enumerate(surfaces):
        glass_name = (s.glass or "AIR").upper().strip()
        stype_str = str(getattr(s.surface_type, "name", s.surface_type)).upper()
        is_mirror = (glass_name in ("MIRROR", "MIRR")) or (stype_str in ("MIRROR", "GRATING")) or ("GRATING" in (s.name or "").upper())
        if is_mirror:
            origin = s.global_origin.copy()
            mirror_surfaces.append((idx, s, origin))

    messages.append(f"Identified {len(mirror_surfaces)} reflective optical elements:")
    for idx, s, orig in mirror_surfaces:
        stype_display = getattr(s.surface_type, "name", str(s.surface_type))
        messages.append(
            f"  Surface {idx:2d} ('{s.name:25s}'): Type={stype_display:7s}, "
            f"Center=({orig[0]:7.2f}, {orig[1]:7.2f}, {orig[2]:7.2f}) mm, R={s.radius:6.1f} mm, SemiDiam={s.semi_diameter:4.1f} mm"
        )

    # 2. Build dense 3D ray fan across all specified wavelengths
    wavelengths = system.wavelengths if system.wavelengths else [1.100, 1.225, 1.350]
    all_rays: List[Ray] = []
    for wl in wavelengths:
        # Chief ray
        all_rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
        # Hexapolar / ring ray pattern
        for ring in range(1, rings + 1):
            r_dist = (ring / rings) * beam_radius
            num_pts = ring * 6
            for pt in range(num_pts):
                phi = 2.0 * np.pi * pt / num_pts
                px = r_dist * np.cos(phi)
                py = r_dist * np.sin(phi)
                all_rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))

    traced_rays = opt_sys.trace_rays(all_rays)

    # 3. Perform 3D ray segment to unrelated mirror housing center distance evaluation
    overall_min_clearance = float("inf")
    worst_case_event: Optional[Dict[str, Any]] = None
    mirror_clearances: Dict[int, float] = {m_idx: float("inf") for m_idx, _, _ in mirror_surfaces}
    clearance_records: List[Dict[str, Any]] = []
    solid_interference_count = 0

    for r_idx, r in enumerate(traced_rays):
        history = r.history
        if len(history) < 2:
            continue

        for seg_idx in range(len(history) - 1):
            h_start = history[seg_idx]
            h_end = history[seg_idx + 1]

            surf_start_idx = h_start["surface_index"]
            surf_end_idx = h_end["surface_index"]

            p_start = np.array([h_start["x_global"], h_start["y_global"], h_start["z_global"]])
            p_end = np.array([h_end["x_global"], h_end["y_global"], h_end["z_global"]])

            # Check every mirror element
            for m_idx, m_surf, m_center in mirror_surfaces:
                # Unrelated mirror: mirror index is neither the start nor the end of this ray segment
                if m_idx == surf_start_idx or m_idx == surf_end_idx:
                    continue

                dist, closest_pt, t_val = point_to_segment_distance_3d(p_start, p_end, m_center)

                if dist < mirror_clearances[m_idx]:
                    mirror_clearances[m_idx] = dist

                if dist < overall_min_clearance:
                    overall_min_clearance = dist
                    worst_case_event = {
                        "ray_index": r_idx,
                        "wavelength_um": r.wavelength,
                        "segment_start_surf": surf_start_idx,
                        "segment_start_name": surfaces[surf_start_idx].name,
                        "segment_end_surf": surf_end_idx,
                        "segment_end_name": surfaces[surf_end_idx].name,
                        "mirror_surf_idx": m_idx,
                        "mirror_name": m_surf.name,
                        "mirror_center": m_center.tolist(),
                        "closest_point_on_segment": closest_pt.tolist(),
                        "distance_mm": dist,
                        "t_param": t_val
                    }

                # Also test solid substrate intersection
                if check_segment_surface_interference(p_start, p_end, m_surf):
                    solid_interference_count += 1
                    messages.append(
                        f"[INTERFERENCE] Ray {r_idx} (wl={r.wavelength:.3f}um) segment "
                        f"{surfaces[surf_start_idx].name}->{surfaces[surf_end_idx].name} intersects solid substrate of {m_surf.name}!"
                    )

    # 4. Summary & Evaluation
    passed = True
    messages.append("\nMinimum Ray-to-Mirror Housing Center Clearances:")
    for m_idx, m_surf, m_center in mirror_surfaces:
        min_d = mirror_clearances[m_idx]
        status = "[PASS]" if min_d > threshold_mm else "[FAIL]"
        if min_d <= threshold_mm:
            passed = False
        messages.append(
            f"  {status} Mirror Surface {m_idx:2d} ('{m_surf.name:25s}'): Min Clearance = {min_d:6.2f} mm (Threshold: > {threshold_mm:.1f} mm)"
        )

    if solid_interference_count > 0:
        passed = False
        messages.append(f"\n[FAIL] Detected {solid_interference_count} solid volume ray-mirror substrate penetrations!")
    else:
        messages.append(f"\n[PASS] Zero physical ray penetrations through mirror substrates.")

    if overall_min_clearance <= threshold_mm:
        passed = False
        messages.append(
            f"\n[FAIL] Minimum spatial clearance {overall_min_clearance:.2f} mm fails strict > {threshold_mm:.1f} mm requirement!"
        )
        if worst_case_event:
            messages.append(
                f"  Worst-case collision proximity: Mirror '{worst_case_event['mirror_name']}' vs Ray Segment "
                f"'{worst_case_event['segment_start_name']}' -> '{worst_case_event['segment_end_name']}' "
                f"(Distance: {worst_case_event['distance_mm']:.3f} mm at wavelength {worst_case_event['wavelength_um']:.3f} um)."
            )
    else:
        messages.append(
            f"\n[PASS] Acceptance Criterion 4 MET: Minimum spatial clearance is {overall_min_clearance:.2f} mm (> {threshold_mm:.1f} mm threshold)."
        )
        if worst_case_event:
            messages.append(
                f"  Tightest clearance margin: {overall_min_clearance:.2f} mm (Mirror '{worst_case_event['mirror_name']}' to "
                f"segment '{worst_case_event['segment_start_name']}' -> '{worst_case_event['segment_end_name']}')."
            )

    summary: Dict[str, Any] = {
        "overall_min_clearance_mm": overall_min_clearance,
        "threshold_mm": threshold_mm,
        "mirror_clearances": mirror_clearances,
        "worst_case_event": worst_case_event,
        "solid_interference_count": solid_interference_count,
        "passed": passed
    }

    return passed, summary, messages


def main():
    parser = argparse.ArgumentParser(
        description="Spectrometer V3 AC4 Collision-Detection & Spatial Clearance Verifier"
    )
    default_nir = os.path.join(PROJECT_ROOT, "spectrometer_v3_nir_raman", "spectrometer_v3_nir_800_2700nm.json")
    default_raman = os.path.join(PROJECT_ROOT, "spectrometer_v3_nir_raman", "spectrometer_v3_raman_1064nm.json")

    parser.add_argument("--nir", default=default_nir, help="Path to NIR 800-2700nm layout JSON")
    parser.add_argument("--raman", default=default_raman, help="Path to Raman 1064nm layout JSON")
    parser.add_argument("--threshold", type=float, default=MIN_CLEARANCE_THRESHOLD_MM, help="Clearance threshold in mm (default: 15.0)")
    parser.add_argument("--rings", type=int, default=5, help="Number of hexapolar ray rings (default: 5)")
    parser.add_argument("--beam-radius", type=float, default=5.0, help="Input beam radius in mm (default: 5.0)")

    args = parser.parse_args()

    print("\n" + "#" * 80)
    print("# SPECTROMETER V3 ACCEPTANCE CRITERION 4 COLLISION & CLEARANCE VERIFICATION")
    print("#" * 80 + "\n")

    all_passed = True

    # 1. NIR Layout Collision Check
    print("=" * 80)
    print("VERIFYING SPATIAL CLEARANCE: Broadband NIR Layout (800 - 2700 nm)")
    print(f"File Path: {args.nir}")
    print("=" * 80)
    nir_passed, nir_summary, nir_msgs = verify_layout_collision_clearance(
        args.nir, "Broadband NIR (800 - 2700 nm)", threshold_mm=args.threshold, rings=args.rings, beam_radius=args.beam_radius
    )
    for m in nir_msgs:
        print(m)
    if not nir_passed:
        all_passed = False

    print("\n" + "=" * 80)
    print("VERIFYING SPATIAL CLEARANCE: Raman 1064 nm Layout (1064 - 1250 nm)")
    print(f"File Path: {args.raman}")
    print("=" * 80)
    raman_passed, raman_summary, raman_msgs = verify_layout_collision_clearance(
        args.raman, "Raman 1064 nm Layout (1064 - 1250 nm)", threshold_mm=args.threshold, rings=args.rings, beam_radius=args.beam_radius
    )
    for m in raman_msgs:
        print(m)
    if not raman_passed:
        all_passed = False

    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT - ACCEPTANCE CRITERION 4")
    print("=" * 80)
    print(f"NIR Layout Min Clearance:        {nir_summary.get('overall_min_clearance_mm', 0.0):.2f} mm {'[PASS]' if nir_passed else '[FAIL]'}")
    print(f"Raman Layout Min Clearance:      {raman_summary.get('overall_min_clearance_mm', 0.0):.2f} mm {'[PASS]' if raman_passed else '[FAIL]'}")
    print(f"Overall Acceptance Criterion 4:  {'[PASS] ALL CLEARANCES > 15.0 MM PROVEN' if all_passed else '[FAIL] CRITERIA NOT MET'}")
    print("=" * 80 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
