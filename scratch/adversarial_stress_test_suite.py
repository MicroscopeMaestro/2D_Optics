#!/usr/bin/env python3
"""
adversarial_stress_test_suite.py: Comprehensive Empirical Adversarial Ray Tracing
& Tolerance Stress Test Suite for hybrid_raman_1064_full_range.json.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram
from verify_layout import RAMAN_WAVELENGTHS_UM

LAYOUT_FILE = "hybrid_raman_1064_full_range.json"

CONTINUOUS_TEST_WAVELENGTHS = [
    1.11130, # Band edge min (400 cm^-1)
    1.15000,
    1.20000,
    1.25000,
    1.30000,
    1.35000,
    1.38919, # Center / Primary design
    1.45000,
    1.50000,
    1.55000,
    1.60000,
    1.65000,
    1.66708  # Band edge max (3400 cm^-1)
]

def generate_random_4d_bundle(
    n_rays_per_wl: int = 500,
    wavelengths = CONTINUOUS_TEST_WAVELENGTHS,
    fiber_radius_mm: float = 0.0125,
    na: float = 0.12,
    seed: int = 42
) -> list:
    """Generate high-density uniform random 4D phase space rays across fiber disk and NA cone."""
    np.random.seed(seed)
    rays = []
    max_theta = np.arcsin(na)

    for wl in wavelengths:
        for _ in range(n_rays_per_wl):
            # Uniform disk spatial sampling
            r_sp = fiber_radius_mm * np.sqrt(np.random.uniform(0.0, 1.0))
            theta_sp = np.random.uniform(0.0, 2.0 * np.pi)
            sx = r_sp * np.cos(theta_sp)
            sy = r_sp * np.sin(theta_sp)

            # Uniform cone angular sampling
            # cos(theta) in [cos(max_theta), 1]
            cos_th = np.random.uniform(np.cos(max_theta), 1.0)
            sin_th = np.sqrt(max(0.0, 1.0 - cos_th**2))
            phi_ang = np.random.uniform(0.0, 2.0 * np.pi)
            vx = sin_th * np.cos(phi_ang)
            vy = sin_th * np.sin(phi_ang)
            vz = cos_th

            rays.append(Ray(
                pos=np.array([sx, sy, 0.0], dtype=float),
                dir=np.array([vx, vy, vz], dtype=float),
                wavelength=float(wl)
            ))
    return rays

def run_stress_test():
    print("=" * 80)
    print(" EMPIRICAL ADVERSARIAL STRESS TEST: hybrid_raman_1064_full_range.json ")
    print("=" * 80)

    if not os.path.exists(LAYOUT_FILE):
        print(f"[FATAL] {LAYOUT_FILE} does not exist!")
        return

    loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
    stack = loaded["stack"]
    opt_sys = OpticalSystem(stack)

    # 1. High-Density 4D Phase Space Test (13 wavelengths x 500 rays = 6500 rays)
    print("\n--- TEST 1: HIGH-DENSITY 4D PHASE SPACE (6,500 RAYS ACROSS 13 WAVELENGTHS) ---")
    dense_rays = generate_random_4d_bundle(n_rays_per_wl=500, wavelengths=CONTINUOUS_TEST_WAVELENGTHS)
    print(f"Generated {len(dense_rays)} rays across {len(CONTINUOUS_TEST_WAVELENGTHS)} wavelengths.")
    
    traced_dense = opt_sys.trace_rays(dense_rays)
    vignetted_dense = [r for r in traced_dense if r.vignetted]
    unvignetted_dense = [r for r in traced_dense if not r.vignetted]
    trans_pct = len(unvignetted_dense) / len(traced_dense) * 100.0
    vig_pct = len(vignetted_dense) / len(traced_dense) * 100.0

    print(f"Transmission: {trans_pct:.2f}% ({len(unvignetted_dense)}/{len(traced_dense)})")
    print(f"Vignetting:   {vig_pct:.2f}% ({len(vignetted_dense)}/{len(traced_dense)})")

    # Locate DMD and Detector indices
    dmd_idx = next((i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper()), -1)
    det_idx = len(stack.surfaces) - 1

    # 2. Per-Wavelength Detailed Metrics
    print("\n--- TEST 2: SPECTRAL BAND-EDGE & CONTINUOUS INTERMEDIATE WAVELENGTHS ---")
    per_wl_results = {}
    for wl in CONTINUOUS_TEST_WAVELENGTHS:
        wl_rays = [r for r in traced_dense if abs(r.wavelength - wl) < 1e-5]
        wl_vig = [r for r in wl_rays if r.vignetted]
        wl_det_hits_x = []
        wl_det_hits_y = []
        wl_dmd_hits_x = []
        wl_dmd_hits_y = []

        for r in wl_rays:
            for h in r.history:
                if h["surface_index"] == dmd_idx:
                    wl_dmd_hits_x.append(h["x"])
                    wl_dmd_hits_y.append(h["y"])
                if h["surface_index"] == det_idx:
                    wl_det_hits_x.append(h["x"])
                    wl_det_hits_y.append(h["y"])

        dmd_span_x = (max(wl_dmd_hits_x) - min(wl_dmd_hits_x)) if wl_dmd_hits_x else 0.0
        dmd_center_x = np.mean(wl_dmd_hits_x) if wl_dmd_hits_x else 0.0
        
        if wl_det_hits_x:
            det_r = np.sqrt(np.array(wl_det_hits_x)**2 + np.array(wl_det_hits_y)**2)
            max_r = float(np.max(det_r))
            rms_r = float(np.sqrt(np.mean(det_r**2)))
            cx = float(np.mean(wl_det_hits_x))
            cy = float(np.mean(wl_det_hits_y))
        else:
            max_r, rms_r, cx, cy = 999.0, 999.0, 999.0, 999.0

        per_wl_results[f"{wl*1000:.1f}nm"] = {
            "wavelength_nm": wl * 1000.0,
            "total_rays": len(wl_rays),
            "vignetted_rays": len(wl_vig),
            "trans_pct": (len(wl_rays) - len(wl_vig)) / len(wl_rays) * 100.0,
            "dmd_x_mean_mm": dmd_center_x,
            "dmd_x_span_mm": dmd_span_x,
            "det_max_radius_mm": max_r,
            "det_rms_radius_mm": rms_r,
            "det_centroid_x_mm": cx,
            "det_centroid_y_mm": cy
        }
        status_wl = "PASS" if max_r <= 0.500 and len(wl_vig) == 0 else "FAIL"
        print(f"  [{status_wl:4s}] {wl*1000:7.1f} nm: Trans={per_wl_results[f'{wl*1000:.1f}nm']['trans_pct']:5.1f}% | DMD_X={dmd_center_x:6.3f} mm | Det_Max_R={max_r:6.3f} mm (Max_Dia={max_r*2:6.3f} mm) | Det_Centroid=({cx:6.3f}, {cy:6.3f})")

    # Global DMD and Detector statistics across all rays
    all_dmd_x = [h["x"] for r in traced_dense for h in r.history if h["surface_index"] == dmd_idx]
    all_dmd_y = [h["y"] for r in traced_dense for h in r.history if h["surface_index"] == dmd_idx]
    all_det_x = [h["x"] for r in traced_dense for h in r.history if h["surface_index"] == det_idx]
    all_det_y = [h["y"] for r in traced_dense for h in r.history if h["surface_index"] == det_idx]

    total_dmd_span_x = max(all_dmd_x) - min(all_dmd_x) if all_dmd_x else 0.0
    total_dmd_max_y = max(map(abs, all_dmd_y)) if all_dmd_y else 0.0
    total_det_radii = np.sqrt(np.array(all_det_x)**2 + np.array(all_det_y)**2) if all_det_x else np.array([999.0])
    total_det_max_r = float(np.max(total_det_radii))
    total_det_max_dia = total_det_max_r * 2.0

    print(f"\n--- OVERALL NOMINAL METRICS ---")
    print(f"  TI DMD4500 Total Dispersion Width: {total_dmd_span_x:.4f} mm (Spec: <= 8.664 mm) -> {'PASS' if total_dmd_span_x <= 8.664 else 'FAIL'}")
    print(f"  TI DMD4500 Max Y Beam Height:     {total_dmd_max_y:.4f} mm (Spec: <= 2.4365 mm) -> {'PASS' if total_dmd_max_y <= 2.4365 else 'FAIL'}")
    print(f"  Detector Max Spot Radius:          {total_det_max_r:.4f} mm (Spec: <= 0.5000 mm) -> {'PASS' if total_det_max_r <= 0.500 else 'FAIL'}")
    print(f"  Detector Max Spot Diameter:        {total_det_max_dia:.4f} mm (Spec: <= 1.0000 mm) -> {'PASS' if total_det_max_dia <= 1.000 else 'FAIL'}")
    print(f"  Overall Ray Transmission:          {trans_pct:.2f}% (Spec: 100.0%) -> {'PASS' if trans_pct == 100.0 else 'FAIL'}")

    # 3. Mechanical Perturbation & Tolerance Analysis (+/- 0.2 mm air gaps, +/- 0.1 deg tilts)
    print("\n--- TEST 3: MECHANICAL PERTURBATION & TOLERANCE STRESS TESTING ---")
    perturbation_results = []

    # Test cases: (surface_index, parameter_name, delta_values)
    perturbation_cases = [
        (0, "thickness", [-0.5, -0.2, +0.2, +0.5], "Fiber Object Air Gap (Surf 0)"),
        (3, "thickness", [-0.5, -0.2, +0.2, +0.5], "Collimator-to-Grating Air Gap (Surf 3)"),
        (4, "tilt_y",    [-0.2, -0.1, +0.1, +0.2], "Grating Tilt Y (Surf 4)"),
        (7, "thickness", [-0.5, -0.2, +0.2, +0.5], "Focus-to-DMD Air Gap (Surf 7)"),
        (8, "tilt_y",    [-0.2, -0.1, +0.1, +0.2], "DMD Tilt Y (Surf 8)"),
        (8, "thickness", [-0.5, -0.2, +0.2, +0.5], "DMD-to-Relay Air Gap (Surf 8)"),
        (11, "thickness", [-0.5, -0.2, +0.2, +0.5], "Relay-to-Condenser Air Gap (Surf 11)"),
        (13, "thickness", [-0.5, -0.2, +0.2, +0.5], "Condenser-to-Detector Air Gap (Surf 13)"),
    ]

    test_bundle_tol = generate_random_4d_bundle(n_rays_per_wl=50, wavelengths=[1.11130, 1.38919, 1.66708], seed=123)

    for s_idx, param, deltas, desc in perturbation_cases:
        print(f"\nPerturbation: {desc} ({param})")
        for d in deltas:
            # Clone stack
            p_stack = SurfaceStack.load_from_json(LAYOUT_FILE)["stack"]
            surf = p_stack.surfaces[s_idx]
            orig_val = getattr(surf, param)
            setattr(surf, param, orig_val + d)
            p_stack._update_accumulated_positions()
            p_sys = OpticalSystem(p_stack)

            p_traced = p_sys.trace_rays(test_bundle_tol)
            p_unvig = [r for r in p_traced if not r.vignetted]
            p_trans = len(p_unvig) / len(p_traced) * 100.0

            p_det_x = [h["x"] for r in p_traced for h in r.history if h["surface_index"] == det_idx]
            p_det_y = [h["y"] for r in p_traced for h in r.history if h["surface_index"] == det_idx]
            if p_det_x:
                p_r = np.sqrt(np.array(p_det_x)**2 + np.array(p_det_y)**2)
                p_max_r = float(np.max(p_r))
            else:
                p_max_r = 999.0

            p_dmd_x = [h["x"] for r in p_traced for h in r.history if h["surface_index"] == dmd_idx]
            p_dmd_span = (max(p_dmd_x) - min(p_dmd_x)) if p_dmd_x else 0.0

            unit = "deg" if "tilt" in param else "mm"
            print(f"  Delta = {d:+5.2f} {unit}: Trans={p_trans:5.1f}% | Det_Max_R={p_max_r:6.3f} mm (Dia={p_max_r*2:6.3f} mm) | DMD_Span={p_dmd_span:6.3f} mm")
            perturbation_results.append({
                "surface_index": s_idx,
                "surface_name": surf.name,
                "parameter": param,
                "delta": d,
                "nominal_val": orig_val,
                "perturbed_val": orig_val + d,
                "transmission_pct": p_trans,
                "detector_max_radius_mm": p_max_r,
                "detector_max_diameter_mm": p_max_r * 2.0,
                "dmd_dispersion_span_mm": p_dmd_span
            })

    # Output Summary JSON
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "layout_file": LAYOUT_FILE,
        "total_test_rays": len(dense_rays),
        "nominal_metrics": {
            "transmission_pct": trans_pct,
            "vignetting_pct": vig_pct,
            "dmd_total_dispersion_width_mm": total_dmd_span_x,
            "dmd_max_y_mm": total_dmd_max_y,
            "detector_max_spot_radius_mm": total_det_max_r,
            "detector_max_spot_diameter_mm": total_det_max_dia
        },
        "per_wavelength_results": per_wl_results,
        "perturbation_results": perturbation_results
    }
    with open("scratch/adversarial_test_results.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("\n[SUCCESS] Wrote detailed adversarial test results to scratch/adversarial_test_results.json")

if __name__ == "__main__":
    run_stress_test()
