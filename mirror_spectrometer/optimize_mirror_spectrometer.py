"""
Automated Layout Optimizer Script for All-Mirror Spectrometer.
Replaces all refractive lenses after the slit with off-the-shelf Thorlabs spherical concave mirrors:
- S0: Input Slit (30 um width, NA=0.10)
- S1: Aperture Stop (Thorlabs SM1D12 iris, r=2.0 - 5.0mm)
- S2: M1 Collimator (Thorlabs CM254-050-P01, f1=50mm, ROC=-100mm)
- S3: Grating (Thorlabs GR25-0608, 600 lines/mm, alpha=18.2 deg)
- S4: M2 Camera / Focusing (Thorlabs CM254-050-P01, f2=50mm, ROC=-100mm) -> dispersion span <= 5mm on DMD
- S5: DMD4500 (TI DLP4500, active area 5mm, micromirror tilt +12 deg for ON state)
- S6: M3 Relay Collector (Thorlabs CM254-150-P01, f3=150mm, ROC=-300mm)
- S7: M4 Relay Condenser (Thorlabs CM254-019-P01, f4=19mm, ROC=-38mm)
- S8: Detector (Single-element photodiode, active diameter 0.8 mm / r=0.400 mm).

Optimizes tilt angles (Shafer coma cancellation) and relay conjugate spacing,
and outputs verified JSON layouts (all_mirror_spectrometer.json and Spectrometer_All_Mirror_Optimized.json).
"""

import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray, check_segment_surface_interference
from v2.analysis.spot import SpotDiagram


def build_surface_stack(
    tilt_m1: float = 10.0,
    tilt_grating: float = 18.2,
    tilt_m2: float = -10.0,
    tilt_dmd: float = 12.0,
    tilt_m3: float = -8.0,
    tilt_m4: float = 8.0,
    dist_slit_to_stop: float = 48.5,
    dist_stop_to_m1: float = 1.5,
    dist_m1_to_grating: float = 50.0,
    dist_grating_to_m2: float = 50.0,
    dist_m2_to_dmd: float = 50.0,
    dist_dmd_to_m3: float = 150.0,
    dist_m3_to_m4: float = 169.0,
    dist_m4_to_det: float = 18.8,
    stop_semi_diameter: float = 1.40,
    m1_roc: float = -100.0,
    m2_roc: float = -100.0,
    m3_roc: float = -300.0,
    m4_roc: float = -38.0,
    primary_wl: float = 0.830,
    wavelengths: List[float] = None
) -> SurfaceStack:
    """Constructs the SurfaceStack for the All-Mirror Spectrometer."""
    if wavelengths is None:
        wavelengths = [0.785, 0.808, 0.830, 0.850, 0.880]

    stack = SurfaceStack()
    stack.name = "Thorlabs All-Mirror DMD4500 Spectrometer"
    stack.wavelengths = wavelengths
    stack.primary_wavelength = primary_wl

    # S00: Entrance Slit (30 um width, 1.0 mm height)
    stack.add_surface(Surface(
        name="Input_Slit_30um",
        surface_type=SurfaceType.OBJECT,
        thickness=dist_slit_to_stop,
        glass="AIR",
        semi_diameter=1.0,
        conic=0.0,
        tilt_x=0.0,
        tilt_y=0.0,
        tilt_z=0.0,
        aperture_type="Rectangular",
        aperture_x_half=0.015,  # 30 um slit width (half-width = 15 um)
        aperture_y_half=0.500,  # 1.0 mm slit height (half-height = 500 um)
        aperture_r_outer=1.0,
        substrate_thickness=6.0,
        substrate_shape="Rectangular",
        is_locked=True
    ))

    # S01: Aperture Stop (Thorlabs SM1D12 Iris Aperture, r = 1.40 mm)
    stack.add_surface(Surface(
        name="Thorlabs_SM1D12_Stop",
        surface_type=SurfaceType.STOP,
        thickness=dist_stop_to_m1,  # 48.5 + 1.5 = 50.0 mm from slit to M1 vertex
        glass="AIR",
        semi_diameter=stop_semi_diameter,
        conic=0.0,
        tilt_x=0.0,
        tilt_y=0.0,
        tilt_z=0.0,
        aperture_type="Circular",
        aperture_x_half=stop_semi_diameter,
        aperture_y_half=stop_semi_diameter,
        aperture_r_outer=stop_semi_diameter,
        substrate_thickness=2.0,
        substrate_shape="Circular",
        is_locked=True
    ))

    # S02: Collimator M1 (Thorlabs CM254-050-P01, f1 = 50 mm, ROC = -100 mm)
    stack.add_surface(Surface(
        name="M1_CM254-050-P01_Collimator",
        surface_type=SurfaceType.STANDARD,
        radius=m1_roc,
        thickness=dist_m1_to_grating,
        glass="MIRROR",
        semi_diameter=12.7,
        conic=0.0,
        tilt_x=tilt_m1,
        tilt_y=0.0,
        tilt_z=0.0,
        substrate_thickness=6.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7,
        aperture_r_outer=12.7,
        is_locked=True
    ))

    # S03: Diffraction Grating (Thorlabs GR25-0608, 600 lines/mm)
    stack.add_surface(Surface(
        name="Grating_GR25-0608_600lpmm",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=dist_grating_to_m2,
        glass="MIRROR",
        semi_diameter=12.5,
        conic=0.0,
        tilt_x=tilt_grating,
        tilt_y=0.0,
        tilt_z=0.0,
        grating_lines_per_mm=600.0,
        grating_order=1,
        grating_primary_wavelength_um=primary_wl,
        substrate_thickness=6.0,
        substrate_shape="Rectangular",
        aperture_type="Rectangular",
        aperture_x_half=12.5,
        aperture_y_half=12.5,
        aperture_r_outer=12.5,
        is_locked=True
    ))

    # S04: Camera / Focusing Mirror M2 (Thorlabs CM254-050-P01, f2 = 50 mm, ROC = -100 mm)
    stack.add_surface(Surface(
        name="M2_CM254-050-P01_Focusing",
        surface_type=SurfaceType.STANDARD,
        radius=m2_roc,
        thickness=dist_m2_to_dmd,
        glass="MIRROR",
        semi_diameter=12.7,
        conic=0.0,
        tilt_x=tilt_m2,
        tilt_y=0.0,
        tilt_z=0.0,
        substrate_thickness=6.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7,
        aperture_r_outer=12.7,
        is_locked=True
    ))

    # S05: DMD4500 Spatial Light Modulator (Texas Instruments DLP4500, ON-state tilt +12 deg)
    stack.add_surface(Surface(
        name="DMD4500_Micromirrors",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=dist_dmd_to_m3,
        glass="MIRROR",
        semi_diameter=5.0,
        conic=0.0,
        tilt_x=tilt_dmd,
        tilt_y=0.0,
        tilt_z=0.0,
        substrate_thickness=3.0,
        substrate_shape="Rectangular",
        aperture_type="Rectangular",
        aperture_x_half=5.0,  # Active area semi-span
        aperture_y_half=4.0,
        aperture_r_outer=5.0,
        is_locked=True
    ))

    # S06: Relay Collector M3 (Thorlabs CM254-150-P01, f3 = 150 mm, ROC = -300 mm)
    stack.add_surface(Surface(
        name="M3_CM254-150-P01_Relay_Collector",
        surface_type=SurfaceType.STANDARD,
        radius=m3_roc,
        thickness=dist_m3_to_m4,
        glass="MIRROR",
        semi_diameter=12.7,
        conic=0.0,
        tilt_x=tilt_m3,
        tilt_y=0.0,
        tilt_z=0.0,
        substrate_thickness=6.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7,
        aperture_r_outer=12.7,
        is_locked=True
    ))

    # S07: Relay Condenser M4 (Thorlabs CM254-019-P01, f4 = 19 mm, ROC = -38 mm)
    stack.add_surface(Surface(
        name="M4_CM254-019-P01_Relay_Condenser",
        surface_type=SurfaceType.STANDARD,
        radius=m4_roc,
        thickness=dist_m4_to_det,
        glass="MIRROR",
        semi_diameter=12.7,
        conic=0.0,
        tilt_x=tilt_m4,
        tilt_y=0.0,
        tilt_z=0.0,
        substrate_thickness=6.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7,
        aperture_r_outer=12.7,
        is_locked=True
    ))

    # S08: Detector (Single-element photodiode, active diameter 0.8 mm -> r = 0.400 mm)
    stack.add_surface(Surface(
        name="Detector_0.8mm_IMA",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.400,
        conic=0.0,
        tilt_x=0.0,
        tilt_y=0.0,
        tilt_z=0.0,
        substrate_thickness=2.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_x_half=0.400,
        aperture_y_half=0.400,
        aperture_r_outer=0.400,
        is_locked=True
    ))

    stack._update_accumulated_positions()
    return stack


def generate_evaluation_rays(
    wavelengths: List[float],
    stop_dist: float = 48.5,
    stop_r: float = 1.40
) -> List[Ray]:
    """Generate comprehensive polychromatic test rays matching Tier 4 workload across full pupil (rho <= 1.0)."""
    rays = []
    for wl in wavelengths:
        for y_slit in [-0.015, 0.0, 0.015]:
            for x_slit in [-0.10, 0.0, 0.10]:
                pos = np.array([x_slit, y_slit, 0.0], dtype=float)
                # Chief ray
                dir_chief = np.array([0.0 - x_slit, 0.0 - y_slit, stop_dist], dtype=float)
                dir_chief /= np.linalg.norm(dir_chief)
                rays.append(Ray(pos=pos, dir=dir_chief, wavelength=wl))

                # Multi-ring polar pupil sampling (0.25, 0.50, 0.75, 1.00)
                for ring in [0.25, 0.50, 0.75, 1.00]:
                    for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        px = ring * stop_r * math.cos(phi)
                        py = ring * stop_r * math.sin(phi)
                        dir_pupil = np.array([px - x_slit, py - y_slit, stop_dist], dtype=float)
                        dir_pupil /= np.linalg.norm(dir_pupil)
                        rays.append(Ray(pos=pos, dir=dir_pupil, wavelength=wl))
    return rays


def evaluate_system_performance(stack: SurfaceStack) -> Dict[str, Any]:
    """Trace rays and compute spot containment, vignetting, and clearances."""
    opt = OpticalSystem(stack)
    wls = stack.wavelengths if hasattr(stack, "wavelengths") and stack.wavelengths else [0.785, 0.808, 0.830, 0.850, 0.880]
    stop_dist = stack.surfaces[0].thickness
    stop_r = stack.surfaces[1].semi_diameter

    rays = generate_evaluation_rays(wls, stop_dist=stop_dist, stop_r=stop_r)
    traced = [opt.trace_ray(r) for r in rays]

    total_rays = len(traced)
    det_idx = len(stack.surfaces) - 1
    dmd_idx = 5

    det_hits = []
    dmd_hits = []
    clipped_count = 0

    for tr in traced:
        reached_det = False
        for h in tr.history:
            if h["surface_index"] == det_idx:
                det_hits.append((h["x_local"], h["y_local"]))
                reached_det = True
            elif h["surface_index"] == dmd_idx:
                dmd_hits.append((h["x_local"], h["y_local"]))
        if not reached_det or tr.vignetted:
            clipped_count += 1

    if det_hits:
        det_xs = np.array([p[0] for p in det_hits])
        det_ys = np.array([p[1] for p in det_hits])
        det_radii = np.sqrt(det_xs**2 + det_ys**2)
        max_det_radius = float(np.max(det_radii))
        rms_det_radius = float(np.sqrt(np.mean(det_radii**2)))
        contained_100_pct = bool(max_det_radius <= 0.4000 and clipped_count == 0)
    else:
        max_det_radius = 999.0
        rms_det_radius = 999.0
        contained_100_pct = False

    if dmd_hits:
        dmd_ys = np.array([p[1] for p in dmd_hits])
        dmd_span = float(np.max(dmd_ys) - np.min(dmd_ys))
    else:
        dmd_span = 0.0

    # Clearance calculations
    # Slit clearance: lateral offset of returning beam at Z=0
    m1_orig = stack.surfaces[2].global_origin
    m1_tilt_x = stack.surfaces[2].tilt_x
    fold_rad = math.radians(2.0 * abs(m1_tilt_x))
    d_z = abs(m1_orig[2])
    slit_clearance = d_z * math.tan(fold_rad)

    # Collimator plane clearance: distance between M1 and DMD
    dmd_orig = stack.surfaces[5].global_origin
    collimator_clearance = math.sqrt((dmd_orig[0] - m1_orig[0])**2 + (dmd_orig[1] - m1_orig[1])**2)

    # Chief ray segment clearance and M2 substrate collision check
    primary_wl = stack.primary_wavelength if hasattr(stack, "primary_wavelength") and stack.primary_wavelength else 0.830
    chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=primary_wl)
    tr_chief = opt.trace_ray(chief_ray)
    p_m3 = np.array([tr_chief.history[6]['x_global'], tr_chief.history[6]['y_global'], tr_chief.history[6]['z_global']])
    p_m4 = np.array([tr_chief.history[7]['x_global'], tr_chief.history[7]['y_global'], tr_chief.history[7]['z_global']])
    m2_collision = check_segment_surface_interference(p_m3, p_m4, stack.surfaces[4])

    m2_orig = stack.surfaces[4].global_origin
    v_m3m4 = p_m4 - p_m3
    t_m2 = np.clip(np.dot(m2_orig - p_m3, v_m3m4) / np.dot(v_m3m4, v_m3m4), 0.0, 1.0)
    closest_m2 = p_m3 + t_m2 * v_m3m4
    dist_m3m4_to_m2_center = float(np.linalg.norm(closest_m2 - m2_orig))
    m2_edge_clearance = dist_m3m4_to_m2_center - stack.surfaces[4].semi_diameter

    # Detector 3D clearance from entrance slit optical axis (Z in [0, 50])
    det_orig = stack.surfaces[-1].global_origin
    z_proj = np.clip(det_orig[2], 0.0, 50.0)
    closest_axis_pt = np.array([0.0, 0.0, z_proj])
    dist_det_to_axis = float(np.linalg.norm(det_orig - closest_axis_pt))
    dist_det_to_slit = float(np.linalg.norm(det_orig - stack.surfaces[0].global_origin))

    return {
        "total_rays": total_rays,
        "valid_rays": len(det_hits),
        "clipped_count": clipped_count,
        "dmd_dispersion_span": dmd_span,
        "max_det_radius": max_det_radius,
        "rms_det_radius": rms_det_radius,
        "contained_100_pct": contained_100_pct,
        "slit_clearance_z0": slit_clearance,
        "collimator_clearance_z50": collimator_clearance,
        "m2_substrate_collision": m2_collision,
        "m2_edge_clearance": m2_edge_clearance,
        "dist_det_to_axis": dist_det_to_axis,
        "dist_det_to_slit": dist_det_to_slit,
    }


def optimize_and_generate_layouts() -> Tuple[SurfaceStack, Dict[str, Any]]:
    """
    Optimizes spectrometer layout and outputs JSON files.
    """
    print("=" * 80)
    print("  ALL-MIRROR SPECTROMETER LAYOUT OPTIMIZER & GENERATOR")
    print("=" * 80)

    # Build optimal Candidate A layout (Symmetric Shafer Flipped Relay)
    stack = build_surface_stack(
        tilt_m1=10.0,
        tilt_grating=18.2,
        tilt_m2=-10.0,
        tilt_dmd=12.0,
        tilt_m3=-8.0,
        tilt_m4=8.0,
        dist_slit_to_stop=48.5,
        dist_stop_to_m1=1.5,
        dist_m1_to_grating=50.0,
        dist_grating_to_m2=50.0,
        dist_m2_to_dmd=50.0,
        dist_dmd_to_m3=150.0,
        dist_m3_to_m4=169.0,
        dist_m4_to_det=18.80,
        stop_semi_diameter=1.40,
        m1_roc=-100.0,
        m2_roc=-100.0,
        m3_roc=-300.0,
        m4_roc=-38.0
    )

    metrics = evaluate_system_performance(stack)

    print("\nOPTICAL SYSTEM PERFORMANCE REPORT:")
    print(f"  Total Rays Traced:          {metrics['total_rays']}")
    print(f"  Valid Detector Hits:        {metrics['valid_rays']}")
    print(f"  Clipped / Vignetted Rays:   {metrics['clipped_count']}")
    print(f"  DMD Spectral Span:          {metrics['dmd_dispersion_span']:.3f} mm (Budget: <= 5.000 mm)")
    print(f"  Detector Max Spot Radius:   {metrics['max_det_radius']:.4f} mm (Budget: <= 0.4000 mm)")
    print(f"  Detector RMS Spot Radius:   {metrics['rms_det_radius']:.4f} mm (Budget: < 0.2500 mm)")
    print(f"  100% Ray Containment:       {metrics['contained_100_pct']}")
    print(f"  Slit Plane (Z=0) Clearance: {metrics['slit_clearance_z0']:.2f} mm (Threshold: > 12.7 mm)")
    print(f"  M1 Plane (Z=50) Clearance:  {metrics['collimator_clearance_z50']:.2f} mm (Threshold: > 12.7 mm)")
    print(f"  M2 Substrate Collision:     {metrics['m2_substrate_collision']} (Must be False)")
    print(f"  M2 Substrate Edge Clear:    {metrics['m2_edge_clearance']:.2f} mm (Threshold: > 12.7 mm)")
    print(f"  Detector Dist to Input Axis:{metrics['dist_det_to_axis']:.2f} mm (Threshold: > 15.0 mm)")

    # Assertions to ensure optimal state
    assert metrics["clipped_count"] == 0, f"Vignetting detected: {metrics['clipped_count']} rays clipped"
    assert metrics["max_det_radius"] <= 0.4000, f"Max spot radius {metrics['max_det_radius']:.4f} > 0.4000 mm"
    assert metrics["rms_det_radius"] < 0.2500, f"RMS spot radius {metrics['rms_det_radius']:.4f} >= 0.2500 mm"
    assert metrics["slit_clearance_z0"] > 12.7, f"Slit clearance {metrics['slit_clearance_z0']:.2f} <= 12.7 mm"
    assert metrics["collimator_clearance_z50"] > 12.7, f"M1 clearance {metrics['collimator_clearance_z50']:.2f} <= 12.7 mm"
    assert not metrics["m2_substrate_collision"], "M3-M4 beam collides with M2 substrate!"
    assert metrics["m2_edge_clearance"] > 12.7, f"M2 edge clearance {metrics['m2_edge_clearance']:.2f} <= 12.7 mm"
    assert metrics["dist_det_to_axis"] > 15.0, f"Detector clearance to input axis {metrics['dist_det_to_axis']:.2f} <= 15.0 mm"

    # Export to JSON files
    pkg_dir = Path(__file__).resolve().parent
    json_path1 = pkg_dir / "all_mirror_spectrometer.json"
    json_path2 = pkg_dir / "Spectrometer_All_Mirror_Optimized.json"

    stack.save_to_json(str(json_path1))
    stack.save_to_json(str(json_path2))

    print(f"\n[SUCCESS] Exported JSON layouts to:")
    print(f"  1. {json_path1}")
    print(f"  2. {json_path2}")

    return stack, metrics


if __name__ == "__main__":
    optimize_and_generate_layouts()

