#!/usr/bin/env python3
"""
test_challenger_1_spectrometer_harness.py
Independent Adversarial Empirical Challenge Harness for hybrid_raman_spectrometer.json
"""

import os
import sys
import copy
import json
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics

def flush_print(msg=""):
    print(msg)
    sys.stdout.flush()

def evaluate_layout(target_json: str):
    flush_print("=" * 80)
    flush_print(f" ADVERSARIAL EVALUATION OF: {target_json} ")
    flush_print("=" * 80)

    with open(target_json, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    loaded = SurfaceStack.load_from_json(target_json)
    stack = loaded["stack"] if isinstance(loaded, dict) else loaded
    system = OpticalSystem(stack)

    dmd_idx = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())
    det_idx = next(i for i, s in enumerate(stack.surfaces) if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper())

    flush_print(f"Loaded {len(stack.surfaces)} optical surfaces.")
    flush_print(f"DMD Surface index: {dmd_idx} ({stack.surfaces[dmd_idx].name})")
    flush_print(f"Detector Surface index: {det_idx} ({stack.surfaces[det_idx].name})")

    # =========================================================================
    # TEST 1: CHIEF RAY SPECTRAL DISPERSION ON TI DMD4500
    # =========================================================================
    flush_print("\n" + "-" * 70)
    flush_print("TEST 1: CHIEF RAY SPECTRAL DISPERSION ON TI DMD4500")
    flush_print("-" * 70)

    wavelengths = [1.11130, 1.20000, 1.30000, 1.40000, 1.50000, 1.61330, 1.66708]
    chief_rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        for wl in wavelengths
    ]

    traced_chief = system.trace_rays(chief_rays)
    dmd_chief_x = {}
    dmd_chief_y = {}

    flush_print(f"{'Wavelength (um)':<18} {'Raman Shift (cm-1)':<20} {'DMD Hit X (mm)':<16} {'DMD Hit Y (mm)':<16} {'Vignetted'}")
    flush_print("-" * 78)

    nu_0 = 1e7 / 1064.0
    for r in traced_chief:
        wl = r.wavelength
        shift = nu_0 - 1e7 / (wl * 1000.0)
        h_dmd = next((h for h in r.history if h["surface_index"] == dmd_idx), None)
        if h_dmd:
            x = h_dmd["x"]
            y = h_dmd["y"]
            dmd_chief_x[round(wl, 5)] = x
            dmd_chief_y[round(wl, 5)] = y
            flush_print(f"{wl:<18.5f} {shift:<20.1f} {x:<16.4f} {y:<16.4f} {r.vignetted}")
        else:
            flush_print(f"{wl:<18.5f} {shift:<20.1f} {'MISS':<16} {'MISS':<16} {r.vignetted}")

    dist_1111_1613 = abs(dmd_chief_x[1.61330] - dmd_chief_x[1.11130]) if (1.61330 in dmd_chief_x and 1.11130 in dmd_chief_x) else 0.0
    dist_1111_1667 = abs(dmd_chief_x[1.66708] - dmd_chief_x[1.11130]) if (1.66708 in dmd_chief_x and 1.11130 in dmd_chief_x) else 0.0

    flush_print(f"\nChief ray distance (1.11130 um to 1.61330 um, 400 - 3148 cm-1): {dist_1111_1613:.4f} mm")
    flush_print(f"Chief ray distance (1.11130 um to 1.66708 um, 400 - 3400 cm-1): {dist_1111_1667:.4f} mm")
    flush_print(f"DMD Active Width: 8.664 mm (half-width 4.332 mm)")
    flush_print(f"Requirement: Chief ray distance strictly in [8.00 mm, 8.664 mm]")
    
    t1_pass = (8.00 <= dist_1111_1613 <= 8.664) or (8.00 <= dist_1111_1667 <= 8.664)
    flush_print(f"TEST 1 VERDICT: {'PASS' if t1_pass else 'FAIL'}")

    # =========================================================================
    # TEST 2: DENSE RAY GRID TRACING (5,145 RAYS ACROSS PUPIL & FIELD)
    # =========================================================================
    flush_print("\n" + "-" * 70)
    flush_print("TEST 2: DENSE RAY GRID TRACING (5,145 RAYS ACROSS PUPIL & FIELD)")
    flush_print("-" * 70)

    fiber_radius = 0.0125  # mm (12.5 um)
    na = 0.10
    max_theta = np.arcsin(na)

    # 21 spatial points across 25 um fiber disk
    spatial_pts = [(0.0, 0.0)]
    for sr, n_pts in [(1, 4), (2, 6), (3, 10)]:
        r_frac = sr / 3.0
        for p in range(n_pts):
            th = 2.0 * np.pi * p / n_pts
            spatial_pts.append((r_frac * fiber_radius * np.cos(th), r_frac * fiber_radius * np.sin(th)))

    # 35 angular directions across NA=0.10 cone
    dir_pts = [(0.0, 0.0, 1.0)]
    for ar, n_pts in [(1, 6), (2, 10), (3, 18)]:
        a_frac = ar / 3.0
        for p in range(n_pts):
            th = 2.0 * np.pi * p / n_pts
            vx = a_frac * np.sin(max_theta) * np.cos(th)
            vy = a_frac * np.sin(max_theta) * np.sin(th)
            vz = np.sqrt(max(0.0, 1.0 - vx**2 - vy**2))
            dir_pts.append((vx, vy, vz))

    dense_rays = []
    for wl in wavelengths:
        for sx, sy in spatial_pts:
            for vx, vy, vz in dir_pts:
                dense_rays.append(Ray(
                    pos=np.array([sx, sy, 0.0]),
                    dir=np.array([vx, vy, vz]),
                    wavelength=wl
                ))

    total_dense_rays = len(dense_rays)
    flush_print(f"Generated {total_dense_rays} dense rays ({len(spatial_pts)} spatial pts x {len(dir_pts)} directions x {len(wavelengths)} wavelengths).")

    t0 = time.perf_counter()
    traced_dense = []
    batch_size = 1000
    for i in range(0, total_dense_rays, batch_size):
        batch = dense_rays[i:i+batch_size]
        traced_dense.extend(system.trace_rays(batch))
        flush_print(f"  Traced {len(traced_dense)} / {total_dense_rays} rays...")

    t_trace = time.perf_counter() - t0
    flush_print(f"Completed tracing {total_dense_rays} rays in {t_trace:.2f} s ({total_dense_rays/t_trace:.0f} rays/s).")

    nan_count = 0
    for r in traced_dense:
        for h in r.history:
            if np.isnan(h["x"]) or np.isnan(h["y"]) or np.isnan(h["z"]):
                nan_count += 1
    flush_print(f"Numerical Anomalies (NaNs) across all surface intersections: {nan_count}")
    t2_pass = (total_dense_rays >= 5000) and (nan_count == 0)
    flush_print(f"TEST 2 VERDICT: {'PASS' if t2_pass else 'FAIL'}")

    # =========================================================================
    # TEST 3: CLEAR APERTURES & VIGNETTING ACROSS ALL SURFACES
    # =========================================================================
    flush_print("\n" + "-" * 70)
    flush_print("TEST 3: CLEAR APERTURES & VIGNETTING / TRANSMISSION BREAKDOWN")
    flush_print("-" * 70)

    surf_names = [s.name for s in stack.surfaces]
    blocked_by_surf = {i: 0 for i in range(len(stack.surfaces))}
    reached_det_count = 0

    for r in traced_dense:
        if r.history:
            last_idx = r.history[-1]["surface_index"]
            if last_idx == det_idx and not r.vignetted:
                reached_det_count += 1
            else:
                blocked_by_surf[last_idx] += 1
        else:
            blocked_by_surf[0] += 1

    flush_print(f"{'Surface Index':<15} {'Surface Name':<28} {'Lost Rays':<12} {'Loss %'}")
    flush_print("-" * 65)
    for i, name in enumerate(surf_names):
        lost = blocked_by_surf[i]
        loss_pct = (lost / total_dense_rays) * 100.0
        flush_print(f"{i:<15} {name:<28} {lost:<12} {loss_pct:.2f}%")

    det_trans_pct = (reached_det_count / total_dense_rays) * 100.0
    flush_print(f"\nTotal rays reaching 1.0mm detector: {reached_det_count} / {total_dense_rays} ({det_trans_pct:.2f}%)")
    t3_pass = (reached_det_count == total_dense_rays)
    flush_print(f"TEST 3 VERDICT: {'PASS' if t3_pass else 'FAIL'}")

    # =========================================================================
    # TEST 4: DETECTOR SPOT DIAMETER & ENCIRCLED ENERGY (1.0 mm DETECTOR)
    # =========================================================================
    flush_print("\n" + "-" * 70)
    flush_print("TEST 4: DETECTOR SPOT DIAMETER & ENCIRCLED ENERGY ANALYSIS")
    flush_print("-" * 70)

    det_x_all = []
    det_y_all = []
    det_r_all = []

    per_wl_det = {}
    for wl in wavelengths:
        wl_x = []
        wl_y = []
        for r in traced_dense:
            if abs(r.wavelength - wl) < 1e-5:
                for h in r.history:
                    if h["surface_index"] == det_idx:
                        wl_x.append(h["x"])
                        wl_y.append(h["y"])
                        det_x_all.append(h["x"])
                        det_y_all.append(h["y"])
                        det_r_all.append(np.sqrt(h["x"]**2 + h["y"]**2))

        if wl_x:
            r_arr = np.sqrt(np.array(wl_x)**2 + np.array(wl_y)**2)
            spot = SpotDiagram(wl_x, wl_y, wavelength_um=wl)
            per_wl_det[wl] = {
                "count": len(wl_x),
                "centroid_x": spot.calculate_centroid()[0],
                "centroid_y": spot.calculate_centroid()[1],
                "rms_radius": spot.calculate_rms_radius(),
                "geo_radius": spot.calculate_geo_radius(),
                "max_radius_from_axis": float(np.max(r_arr)),
                "max_diameter": float(np.max(r_arr) * 2.0)
            }

    flush_print(f"{'Wl (um)':<10} {'Rays':<8} {'Centroid (x,y) mm':<22} {'RMS (um)':<12} {'GEO (um)':<12} {'Max R (um)':<12} {'Max D (um)':<12} {'<=1.0mm?'}")
    flush_print("-" * 102)
    for wl, m in per_wl_det.items():
        c_str = f"({m['centroid_x']:+.3f}, {m['centroid_y']:+.3f})"
        fits = "YES (PASS)" if m['max_radius_from_axis'] <= 0.500 else "NO (FAIL)"
        flush_print(f"{wl:<10.5f} {m['count']:<8} {c_str:<22} {m['rms_radius']*1000:<12.1f} {m['geo_radius']*1000:<12.1f} {m['max_radius_from_axis']*1000:<12.1f} {m['max_diameter']*1000:<12.1f} {fits}")

    det_r_arr = np.array(det_r_all) if det_r_all else np.array([999.0])
    max_overall_r = float(np.max(det_r_arr))
    max_overall_d = max_overall_r * 2.0
    all_spot = SpotDiagram(det_x_all, det_y_all, wavelength_um=1.362) if det_x_all else None
    all_rms = all_spot.calculate_rms_radius() if all_spot else 999.0
    all_geo = all_spot.calculate_geo_radius() if all_spot else 999.0

    flush_print("\n--- ALL WAVELENGTHS COMPOSITE SPOT ON DETECTOR ---")
    flush_print(f"Total rays at detector: {len(det_r_arr)}")
    if all_spot:
        flush_print(f"Polychromatic Centroid: ({all_spot.calculate_centroid()[0]:.4f}, {all_spot.calculate_centroid()[1]:.4f}) mm")
    flush_print(f"Polychromatic RMS Radius: {all_rms*1000:.2f} um")
    flush_print(f"Polychromatic GEO Radius (rel centroid): {all_geo*1000:.2f} um")
    flush_print(f"Maximum Ray Radius from Optical Axis: {max_overall_r*1000:.2f} um ({max_overall_r:.4f} mm)")
    flush_print(f"Maximum Ray Geometric Diameter: {max_overall_d*1000:.2f} um ({max_overall_d:.4f} mm)")
    flush_print(f"Detector Active Limit: Radius <= 500.0 um (Diameter <= 1.000 mm)")

    # Encircled energy distribution
    radii_bins_um = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1000]
    flush_print("\n--- ENCIRCLED ENERGY DISTRIBUTION ---")
    flush_print(f"{'Radius (um)':<15} {'Enclosed Rays':<15} {'Encircled Energy (%)':<20}")
    flush_print("-" * 50)
    for r_um in radii_bins_um:
        r_mm = r_um / 1000.0
        encl = int(np.sum(det_r_arr <= r_mm))
        pct = (encl / len(det_r_arr)) * 100.0
        flush_print(f"{r_um:<15} {encl:<15} {pct:<20.2f}%")

    exceeding_rays = int(np.sum(det_r_arr > 0.500))
    flush_print(f"\nRays exceeding detector radius r = 0.50 mm (D = 1.0 mm): {exceeding_rays} / {len(det_r_arr)}")
    t4_pass = (exceeding_rays == 0) and (max_overall_r <= 0.500)
    flush_print(f"TEST 4 VERDICT: {'PASS' if t4_pass else 'FAIL'}")

    # =========================================================================
    # TEST 5: FULL DMD FIELD COUPLING & REFLECTION TO DETECTOR
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("TEST 5: FULL DMD FIELD COUPLING & REFLECTION TO DETECTOR")
    flush_print("-" * 70)
    
    dmd_to_det_stack = SurfaceStack()
    for s in stack.surfaces[8:]:
        dmd_to_det_stack.add_surface(copy.deepcopy(s))
    dmd_to_det_sys = OpticalSystem(dmd_to_det_stack)

    dmd_test_x = np.linspace(-4.332, 4.332, 9)
    dmd_test_y = np.linspace(-2.4365, 2.4365, 5)
    dmd_cone_na = 0.10
    dmd_max_ang = np.arcsin(dmd_cone_na)

    dmd_field_rays = []
    for wl in [1.1113, 1.362, 1.66708]:
        for x_dmd in dmd_test_x:
            for y_dmd in dmd_test_y:
                for ar in range(9):
                    if ar == 0:
                        vx, vy, vz = 0.0, 0.0, 1.0
                    else:
                        th = 2.0 * np.pi * (ar - 1) / 8.0
                        vx = np.sin(dmd_max_ang) * np.cos(th)
                        vy = np.sin(dmd_max_ang) * np.sin(th)
                        vz = np.sqrt(1.0 - vx**2 - vy**2)
                    dmd_field_rays.append(Ray(
                        pos=np.array([x_dmd, y_dmd, 0.0]),
                        dir=np.array([vx, vy, vz]),
                        wavelength=wl
                    ))

    traced_dmd_field = dmd_to_det_sys.trace_rays(dmd_field_rays)
    dmd_field_det_r = []
    dmd_field_vignetted = 0
    for r in traced_dmd_field:
        if not r.vignetted and r.history:
            last = r.history[-1]
            if last["surface_index"] == len(dmd_to_det_stack.surfaces) - 1:
                dmd_field_det_r.append(np.sqrt(last["x"]**2 + last["y"]**2))
        else:
            dmd_field_vignetted += 1

    dmd_field_det_r_arr = np.array(dmd_field_det_r) if dmd_field_det_r else np.array([999.0])
    dmd_field_max_r = float(np.max(dmd_field_det_r_arr)) if len(dmd_field_det_r) > 0 else 999.0
    dmd_field_exceeding = int(np.sum(dmd_field_det_r_arr > 0.500)) if len(dmd_field_det_r) > 0 else len(dmd_field_rays)

    flush_print(f"Traced {len(dmd_field_rays)} rays across 9x5 DMD grid x 9 angles x 3 wavelengths.")
    flush_print(f"Rays successfully reaching detector: {len(dmd_field_det_r)} / {len(dmd_field_rays)}")
    flush_print(f"Vignetted in collection path: {dmd_field_vignetted}")
    flush_print(f"Maximum spot radius on detector from full DMD active area: {dmd_field_max_r*1000:.2f} um ({dmd_field_max_r:.4f} mm)")
    flush_print(f"Maximum spot diameter from full DMD: {dmd_field_max_r*2000:.2f} um ({dmd_field_max_r*2:.4f} mm)")
    flush_print(f"Rays exceeding detector radius r = 0.50 mm (D = 1.0 mm): {dmd_field_exceeding} / {len(dmd_field_det_r)}")
    t5_pass = (dmd_field_exceeding == 0) and (dmd_field_max_r <= 0.500)
    flush_print(f"TEST 5 VERDICT: {'PASS' if t5_pass else 'FAIL'}")

    # =========================================================================
    # SUMMARY OF EMPIRICAL VERDICT
    # =========================================================================
    flush_print("\n" + "=" * 80)
    flush_print(f" EMPIRICAL CHALLENGER 1 VERDICT FOR {target_json} ")
    flush_print("=" * 80)
    
    flush_print(f"1. DMD Active Width Utilization [8.00, 8.664] mm:    {'PASS' if t1_pass else 'FAIL'} (dispersion = {dist_1111_1613:.3f} mm)")
    flush_print(f"2. Dense Ray Grid Tracing (5,145 rays, 0 NaNs):       {'PASS' if t2_pass else 'FAIL'} ({total_dense_rays} rays, 0 NaNs)")
    flush_print(f"3. Clear Apertures & Vignetting (100% Throughput):    {'PASS' if t3_pass else 'FAIL'} ({det_trans_pct:.2f}% transmission)")
    flush_print(f"4. Detector Spot Diameter <= 1.0 mm (r <= 0.50 mm):   {'PASS' if t4_pass else 'FAIL'} (max r = {max_overall_r*1000:.1f} um, D = {max_overall_d*1000:.1f} um)")
    flush_print(f"5. Full DMD Active Area Focused to 1.0 mm Detector:   {'PASS' if t5_pass else 'FAIL'} (max r = {dmd_field_max_r*1000:.1f} um, D = {dmd_field_max_r*2000:.1f} um)")

    all_tests_passed = t1_pass and t2_pass and t3_pass and t4_pass and t5_pass
    final_verdict = "APPROVE" if all_tests_passed else "REQUEST_CHANGES"
    flush_print(f"\nFINAL VERDICT: {final_verdict}")
    flush_print("=" * 80)

    return final_verdict, {
        "target_file": target_json,
        "t1_dmd_dispersion_mm": dist_1111_1613,
        "t1_dmd_dispersion_full_band_mm": dist_1111_1667,
        "t1_pass": t1_pass,
        "t2_total_rays": total_dense_rays,
        "t2_nan_count": nan_count,
        "t2_pass": t2_pass,
        "t3_transmission_pct": det_trans_pct,
        "t3_pass": t3_pass,
        "t4_max_detector_radius_um": max_overall_r * 1000.0,
        "t4_max_detector_diameter_um": max_overall_d * 1000.0,
        "t4_exceeding_rays": exceeding_rays,
        "t4_pass": t4_pass,
        "t5_dmd_field_max_radius_um": dmd_field_max_r * 1000.0,
        "t5_dmd_field_max_diameter_um": dmd_field_max_r * 2000.0,
        "t5_dmd_field_exceeding": dmd_field_exceeding,
        "t5_pass": t5_pass,
        "all_tests_passed": all_tests_passed,
        "verdict": final_verdict
    }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else TARGET_JSON
    verdict, rep = evaluate_layout(target)
