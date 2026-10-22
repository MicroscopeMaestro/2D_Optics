"""
Adversarial Stress Test Harness for All-Mirror DMD4500 Spectrometer.
Challenger 1 Empirical Verification Suite.
"""

import os
import sys
import copy
import math
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray, check_segment_surface_interference
from v2.analysis.spot import SpotDiagram


def load_layout(layout_path: Path = None) -> Tuple[SurfaceStack, OpticalSystem, Dict[str, Any]]:
    if layout_path is None:
        layout_path = REPO_ROOT / "mirror_spectrometer" / "all_mirror_spectrometer.json"
    with open(layout_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    loaded = SurfaceStack.load_from_json(str(layout_path))
    stack = loaded["stack"]
    stack._update_accumulated_positions()
    opt_sys = OpticalSystem(stack)
    return stack, opt_sys, raw_json


def run_suite_1_pupil_and_field_sampling(stack: SurfaceStack):
    print("\n" + "=" * 80)
    print("  SUITE 1: PUPIL FILL FRACTION & SPATIAL FIELD SPOT SIZE BEHAVIOR")
    print("=" * 80)

    opt_sys = OpticalSystem(stack)
    stop_dist = stack.surfaces[0].thickness
    stop_r = stack.surfaces[1].semi_diameter
    det_idx = len(stack.surfaces) - 1

    wavelengths = [0.785, 0.808, 0.830, 0.850, 0.880]
    pupil_fractions = [0.35, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]

    print(f"\n1. Maximum Detector Spot Radius (mm) vs Pupil Fill Fraction & Wavelength:")
    print(f"   (Slit Center Point x=0, y=0)")
    header = f"   {'Pupil Fraction':<16} " + " ".join([f"{int(wl*1000)}nm ({wl:.3f}):<14" for wl in wavelengths]) + " Max Across All"
    print(header)
    print("   " + "-" * (16 + 14 * len(wavelengths) + 16))

    for frac in pupil_fractions:
        row_str = f"   {frac*100:5.1f}% (r={frac*stop_r:.2f}mm) "
        max_across_all = 0.0
        for wl in wavelengths:
            wl_radii = []
            for phi in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                px = frac * stop_r * math.cos(phi)
                py = frac * stop_r * math.sin(phi)
                d = np.array([px, py, stop_dist], dtype=float)
                d /= np.linalg.norm(d)
                tr = opt_sys.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=wl))
                h = [x for x in tr.history if x["surface_index"] == det_idx]
                if h:
                    wl_radii.append(math.sqrt(h[0]["x_local"]**2 + h[0]["y_local"]**2))
            max_wl = max(wl_radii) if wl_radii else 999.0
            row_str += f"{max_wl:<14.4f} "
            if max_wl > max_across_all:
                max_across_all = max_wl
        row_str += f"{max_across_all:<14.4f}"
        print(row_str)

    print(f"\n2. Maximum Detector Spot Radius (mm) vs Slit Field Coordinates (at ring=0.70 & ring=0.90):")
    slit_points = [
        ("Center (0, 0)", 0.0, 0.0),
        ("Slit Width Edge (+0.015, 0)", 0.015, 0.0),
        ("Slit Width Edge (-0.015, 0)", -0.015, 0.0),
        ("Slit Height (+0.10, 0)", 0.10, 0.0),
        ("Slit Height (+0.25, 0)", 0.25, 0.0),
        ("Slit Height (+0.50, 0)", 0.50, 0.0),
        ("Slit Corner (+0.015, +0.10)", 0.015, 0.10),
        ("Slit Corner (+0.015, +0.25)", 0.015, 0.25),
        ("Slit Corner (+0.015, +0.50)", 0.015, 0.50),
    ]

    print(f"   {'Field Point':<30} {'Ring 0.70 Max R':<18} {'Ring 0.90 Max R':<18} {'Ring 1.00 Max R':<18}")
    print("   " + "-" * 84)

    for f_name, x0, y0 in slit_points:
        res = []
        for frac in [0.70, 0.90, 1.00]:
            radii = []
            for wl in wavelengths:
                d_c = np.array([0.0 - x0, 0.0 - y0, stop_dist], dtype=float)
                d_c /= np.linalg.norm(d_c)
                tr_c = opt_sys.trace_ray(Ray(pos=np.array([x0, y0, 0.0]), dir=d_c, wavelength=wl))
                h = [x for x in tr_c.history if x["surface_index"] == det_idx]
                if h:
                    radii.append(math.sqrt(h[0]["x_local"]**2 + h[0]["y_local"]**2))

                for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    px = frac * stop_r * math.cos(phi)
                    py = frac * stop_r * math.sin(phi)
                    d_p = np.array([px - x0, py - y0, stop_dist], dtype=float)
                    d_p /= np.linalg.norm(d_p)
                    tr_p = opt_sys.trace_ray(Ray(pos=np.array([x0, y0, 0.0]), dir=d_p, wavelength=wl))
                    h = [x for x in tr_p.history if x["surface_index"] == det_idx]
                    if h:
                        radii.append(math.sqrt(h[0]["x_local"]**2 + h[0]["y_local"]**2))
            res.append(max(radii) if radii else 999.0)
        print(f"   {f_name:<30} {res[0]:<18.4f} {res[1]:<18.4f} {res[2]:<18.4f}")


def run_suite_2_continuous_wavelength_sweep(stack: SurfaceStack):
    print("\n" + "=" * 80)
    print("  SUITE 2: CONTINUOUS WAVELENGTH SWEEP (785 nm to 880 nm in 1 nm steps)")
    print("=" * 80)

    opt_sys = OpticalSystem(stack)
    stop_dist = stack.surfaces[0].thickness
    stop_r = stack.surfaces[1].semi_diameter
    det_idx = len(stack.surfaces) - 1
    dmd_idx = 5

    wavelengths = np.linspace(0.785, 0.880, 96)  # 96 steps
    dmd_positions = []
    det_max_radii = []
    det_rms_radii = []

    for wl in wavelengths:
        wl_hits = []
        dmd_hits = []
        for x_slit in [-0.015, 0.0, 0.015]:
            for y_slit in [-0.015, 0.0, 0.015]:
                pos = np.array([x_slit, y_slit, 0.0], dtype=float)
                d_chief = np.array([0.0 - x_slit, 0.0 - y_slit, stop_dist], dtype=float)
                d_chief /= np.linalg.norm(d_chief)
                tr_c = opt_sys.trace_ray(Ray(pos=pos, dir=d_chief, wavelength=float(wl)))

                for h in tr_c.history:
                    if h["surface_index"] == dmd_idx:
                        dmd_hits.append(h["y_local"])
                    if h["surface_index"] == det_idx:
                        wl_hits.append((h["x_local"], h["y_local"]))

                for ring in [0.35, 0.70]:
                    for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        px = ring * stop_r * math.cos(phi)
                        py = ring * stop_r * math.sin(phi)
                        d_p = np.array([px - x_slit, py - y_slit, stop_dist], dtype=float)
                        d_p /= np.linalg.norm(d_p)
                        tr_p = opt_sys.trace_ray(Ray(pos=pos, dir=d_p, wavelength=float(wl)))
                        for h in tr_p.history:
                            if h["surface_index"] == det_idx:
                                wl_hits.append((h["x_local"], h["y_local"]))

        dmd_positions.append(np.mean(dmd_hits) if dmd_hits else 0.0)
        radii = [math.sqrt(x**2 + y**2) for x, y in wl_hits]
        det_max_radii.append(max(radii) if radii else 999.0)
        det_rms_radii.append(math.sqrt(np.mean(np.array(radii)**2)) if radii else 999.0)

    dmd_span = max(dmd_positions) - min(dmd_positions)
    max_all_wl_r = max(det_max_radii)

    print(f"  Total Wavelength Samples:   {len(wavelengths)} (785.00 nm to 880.00 nm)")
    print(f"  DMD Dispersion Span:        {dmd_span:.4f} mm (Budget <= 5.0000 mm)")
    print(f"  DMD Y Min / Max:            {min(dmd_positions):.4f} mm / {max(dmd_positions):.4f} mm")
    print(f"  Peak Spot Radius Across All:{max_all_wl_r:.4f} mm (Threshold <= 0.4000 mm)")
    print(f"  Status:                     {'PASS' if max_all_wl_r <= 0.4000 and dmd_span <= 5.0 else 'FAIL'}")


def run_suite_3_alignment_tolerance_stress(stack: SurfaceStack):
    print("\n" + "=" * 80)
    print("  SUITE 3: MIRROR TILT MISALIGNMENT (+/-0.1 deg) & SENSITIVITY")
    print("=" * 80)

    perturb_surfaces = [
        (2, "M1 Collimator", 10.0),
        (3, "Grating", 18.2),
        (4, "M2 Camera Mirror", -10.0),
        (5, "DMD4500", 12.0),
        (6, "M3 Relay Collector", 8.0),
        (7, "M4 Relay Condenser", -8.0),
    ]

    stop_dist = stack.surfaces[0].thickness
    stop_r = stack.surfaces[1].semi_diameter
    det_idx = len(stack.surfaces) - 1

    # Base rays for evaluation (pupil at ring 0.70)
    wls = [0.785, 0.808, 0.830, 0.850, 0.880]
    base_rays = []
    for wl in wls:
        for x_slit in [-0.015, 0.0, 0.015]:
            for y_slit in [-0.015, 0.0, 0.015]:
                pos = np.array([x_slit, y_slit, 0.0])
                d_chief = np.array([0.0 - x_slit, 0.0 - y_slit, stop_dist])
                d_chief /= np.linalg.norm(d_chief)
                base_rays.append(Ray(pos=pos, dir=d_chief, wavelength=wl))
                for ring in [0.70]:
                    for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        px = ring * stop_r * math.cos(phi)
                        py = ring * stop_r * math.sin(phi)
                        d_p = np.array([px - x_slit, py - y_slit, stop_dist])
                        d_p /= np.linalg.norm(d_p)
                        base_rays.append(Ray(pos=pos, dir=d_p, wavelength=wl))

    print(f"\n1. Single-Element Tilt Perturbations (+/-0.1 deg on tilt_x and tilt_y):")
    print(f"   {'Surface':<26} {'Perturbation':<18} {'Max R (mm)':<14} {'RMS R (mm)':<14} {'Containment':<12}")
    print("   " + "-" * 75)

    for s_idx, s_name, nominal_tilt in perturb_surfaces:
        for delta_tx in [-0.1, 0.1]:
            cloned_stack = copy.deepcopy(stack)
            cloned_stack.surfaces[s_idx].tilt_x = nominal_tilt + delta_tx
            cloned_stack._update_accumulated_positions()
            opt = OpticalSystem(cloned_stack)

            hits = []
            for r in base_rays:
                tr = opt.trace_ray(r)
                h = [x for x in tr.history if x["surface_index"] == det_idx]
                if h:
                    hits.append((h[0]["x_local"], h[0]["y_local"]))

            if hits:
                radii = [math.sqrt(x**2 + y**2) for x, y in hits]
                max_r = max(radii)
                rms_r = math.sqrt(np.mean(np.array(radii)**2))
                cont = sum(1 for r in radii if r <= 0.4000) / len(radii) * 100.0
            else:
                max_r = 999.0
                rms_r = 999.0
                cont = 0.0

            pert_str = f"tilt_x {delta_tx:+.1f} deg"
            print(f"   {s_name:<26} {pert_str:<18} {max_r:<14.4f} {rms_r:<14.4f} {cont:<11.1f}%")

        for delta_ty in [-0.1, 0.1]:
            cloned_stack = copy.deepcopy(stack)
            cloned_stack.surfaces[s_idx].tilt_y = delta_ty
            cloned_stack._update_accumulated_positions()
            opt = OpticalSystem(cloned_stack)

            hits = []
            for r in base_rays:
                tr = opt.trace_ray(r)
                h = [x for x in tr.history if x["surface_index"] == det_idx]
                if h:
                    hits.append((h[0]["x_local"], h[0]["y_local"]))

            if hits:
                radii = [math.sqrt(x**2 + y**2) for x, y in hits]
                max_r = max(radii)
                rms_r = math.sqrt(np.mean(np.array(radii)**2))
                cont = sum(1 for r in radii if r <= 0.4000) / len(radii) * 100.0
            else:
                max_r = 999.0
                rms_r = 999.0
                cont = 0.0

            pert_str = f"tilt_y {delta_ty:+.1f} deg"
            print(f"   {s_name:<26} {pert_str:<18} {max_r:<14.4f} {rms_r:<14.4f} {cont:<11.1f}%")


def run_suite_4_position_shift_stress(stack: SurfaceStack):
    print("\n" + "=" * 80)
    print("  SUITE 4: POSITION SHIFTS & AXIAL SPACING TOLERANCES (+/-0.5 mm)")
    print("=" * 80)

    stop_dist = stack.surfaces[0].thickness
    stop_r = stack.surfaces[1].semi_diameter
    det_idx = len(stack.surfaces) - 1

    wls = [0.785, 0.830, 0.880]
    base_rays = []
    for wl in wls:
        for x_slit in [-0.015, 0.0, 0.015]:
            for y_slit in [-0.015, 0.0, 0.015]:
                pos = np.array([x_slit, y_slit, 0.0])
                d_chief = np.array([0.0 - x_slit, 0.0 - y_slit, stop_dist])
                d_chief /= np.linalg.norm(d_chief)
                base_rays.append(Ray(pos=pos, dir=d_chief, wavelength=wl))
                for ring in [0.70]:
                    for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        px = ring * stop_r * math.cos(phi)
                        py = ring * stop_r * math.sin(phi)
                        d_p = np.array([px - x_slit, py - y_slit, stop_dist])
                        d_p /= np.linalg.norm(d_p)
                        base_rays.append(Ray(pos=pos, dir=d_p, wavelength=wl))

    thickness_tests = [
        (0, "Slit-to-Stop Distance", 48.5),
        (2, "M1-to-Grating Distance", 50.0),
        (3, "Grating-to-M2 Distance", 50.0),
        (4, "M2-to-DMD Distance", 50.0),
        (5, "DMD-to-M3 Distance", 150.0),
        (6, "M3-to-M4 Distance", 169.0),
        (7, "M4-to-Detector Distance", 19.0),
    ]

    print(f"1. Axial Distance Variations (+/-0.5 mm):")
    print(f"   {'Spacing':<28} {'Delta':<12} {'Max R (mm)':<14} {'RMS R (mm)':<14} {'Containment':<12}")
    print("   " + "-" * 75)

    for s_idx, s_name, nominal_th in thickness_tests:
        for delta_th in [-0.5, 0.5]:
            cloned_stack = copy.deepcopy(stack)
            cloned_stack.surfaces[s_idx].thickness = nominal_th + delta_th
            cloned_stack._update_accumulated_positions()
            opt = OpticalSystem(cloned_stack)

            hits = []
            for r in base_rays:
                tr = opt.trace_ray(r)
                h = [x for x in tr.history if x["surface_index"] == det_idx]
                if h:
                    hits.append((h[0]["x_local"], h[0]["y_local"]))

            if hits:
                radii = [math.sqrt(x**2 + y**2) for x, y in hits]
                max_r = max(radii)
                rms_r = math.sqrt(np.mean(np.array(radii)**2))
                cont = sum(1 for r in radii if r <= 0.4000) / len(radii) * 100.0
            else:
                max_r = 999.0
                rms_r = 999.0
                cont = 0.0

            delta_str = f"{delta_th:+.1f} mm"
            print(f"   {s_name:<28} {delta_str:<12} {max_r:<14.4f} {rms_r:<14.4f} {cont:<11.1f}%")


def run_suite_5_full_monte_carlo_compound_tolerance(stack: SurfaceStack, num_trials: int = 200):
    print("\n" + "=" * 80)
    print(f"  SUITE 5: COMPOUND MONTE CARLO TOLERANCE ANALYSIS ({num_trials} RANDOM BUILDS)")
    print("  (Simultaneous random tilts U[-0.05, +0.05] deg & axial shifts U[-0.2, +0.2] mm)")
    print("=" * 80)

    stop_dist = stack.surfaces[0].thickness
    stop_r = stack.surfaces[1].semi_diameter
    det_idx = len(stack.surfaces) - 1

    wls = [0.785, 0.830, 0.880]
    base_rays = []
    for wl in wls:
        for x_slit in [-0.015, 0.0, 0.015]:
            for y_slit in [-0.015, 0.0, 0.015]:
                pos = np.array([x_slit, y_slit, 0.0])
                d_chief = np.array([0.0 - x_slit, 0.0 - y_slit, stop_dist])
                d_chief /= np.linalg.norm(d_chief)
                base_rays.append(Ray(pos=pos, dir=d_chief, wavelength=wl))
                for ring in [0.70]:
                    for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        px = ring * stop_r * math.cos(phi)
                        py = ring * stop_r * math.sin(phi)
                        d_p = np.array([px - x_slit, py - y_slit, stop_dist])
                        d_p /= np.linalg.norm(d_p)
                        base_rays.append(Ray(pos=pos, dir=d_p, wavelength=wl))

    np.random.seed(12345)
    max_radii = []
    containment_rates = []

    for trial in range(num_trials):
        cloned_stack = copy.deepcopy(stack)
        for s_idx in [2, 3, 4, 5, 6, 7]:
            cloned_stack.surfaces[s_idx].tilt_x += float(np.random.uniform(-0.05, 0.05))
            cloned_stack.surfaces[s_idx].tilt_y += float(np.random.uniform(-0.05, 0.05))
            cloned_stack.surfaces[s_idx].thickness += float(np.random.uniform(-0.2, 0.2))

        cloned_stack._update_accumulated_positions()
        opt = OpticalSystem(cloned_stack)

        hits = []
        for r in base_rays:
            tr = opt.trace_ray(r)
            h = [x for x in tr.history if x["surface_index"] == det_idx]
            if h:
                hits.append((h[0]["x_local"], h[0]["y_local"]))

        if hits:
            radii = [math.sqrt(x**2 + y**2) for x, y in hits]
            max_radii.append(max(radii))
            containment_rates.append(sum(1 for r in radii if r <= 0.4000) / len(radii) * 100.0)
        else:
            max_radii.append(999.0)
            containment_rates.append(0.0)

    max_radii = np.array(max_radii)
    containment_rates = np.array(containment_rates)

    yield_100 = np.sum(containment_rates == 100.0) / num_trials * 100.0
    yield_95 = np.sum(containment_rates >= 95.0) / num_trials * 100.0
    median_max_r = float(np.median(max_radii))
    p95_max_r = float(np.percentile(max_radii, 95))

    print(f"  Total Monte Carlo Trials:       {num_trials}")
    print(f"  Median Max Spot Radius:         {median_max_r:.4f} mm")
    print(f"  95th Percentile Max Spot Radius:{p95_max_r:.4f} mm")
    print(f"  100% Ray Containment Yield:     {yield_100:.1f}%")
    print(f"  >=95% Ray Containment Yield:    {yield_95:.1f}%")


if __name__ == "__main__":
    stack, opt_sys, raw_json = load_layout()
    run_suite_1_pupil_and_field_sampling(stack)
    run_suite_2_continuous_wavelength_sweep(stack)
    run_suite_3_alignment_tolerance_stress(stack)
    run_suite_4_position_shift_stress(stack)
    run_suite_5_full_monte_carlo_compound_tolerance(stack, num_trials=200)
