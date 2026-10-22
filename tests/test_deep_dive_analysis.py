"""
tests/test_deep_dive_analysis.py
Detailed Empirical Ray-Tracing and Optical Physics Analysis for hybrid_raman_1064.json.
Author: Challenger 1 (critic, specialist)
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics
from optics_app.utils.materials import get_refractive_index

RAMAN_WAVELENGTHS = [
    1.11130,  # 400 cm^-1
    1.20000,  # 1065 cm^-1
    1.30000,  # 1706 cm^-1
    1.38919,  # 2200 cm^-1
    1.50000,  # 2732 cm^-1
    1.60000,  # 3148 cm^-1
    1.61330,  # 3200 cm^-1
    1.66708,  # 3400 cm^-1
]

INTERMEDIATE_WAVELENGTHS = [
    1.15000,
    1.25000,
    1.35000,
    1.45000,
    1.55000,
    1.64000
]

ALL_TEST_WAVELENGTHS = sorted(list(set(RAMAN_WAVELENGTHS + INTERMEDIATE_WAVELENGTHS)))

def run_deep_dive(json_path="hybrid_raman_1064.json"):
    print("=" * 80)
    print(f" DEEP-DIVE EMPIRICAL ANALYSIS: {json_path} ")
    print("=" * 80)

    loaded = SurfaceStack.load_from_json(json_path)
    stack = loaded["stack"] if isinstance(loaded, dict) else loaded
    system = OpticalSystem(stack)

    print(f"Total Surfaces: {len(stack.surfaces)}")
    for i, s in enumerate(stack.surfaces):
        st_val = s.surface_type.value if hasattr(s.surface_type, 'value') else str(s.surface_type)
        print(f"  [{i:02d}] {s.name:<28} Type: {st_val:<10} R={s.radius:<8.2f} T={s.thickness:<8.2f} Glass={s.glass:<10} SD={s.semi_diameter:<6.2f} TiltX={s.tilt_x:<6.2f} TiltY={s.tilt_y:<6.2f}")

    dmd_idx = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())
    det_idx = next(i for i, s in enumerate(stack.surfaces) if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper())

    # 1. Chief ray tracing across all wavelengths
    print("\n" + "=" * 80)
    print(" 1. CHIEF RAY TRACING & DMD ACTIVE AREA ANALYSIS ")
    print("=" * 80)
    print(f"{'Wavelength (um)':<18} {'DMD X (mm)':<14} {'DMD Y (mm)':<14} {'Det X (mm)':<14} {'Det Y (mm)':<14} {'Within DMD?':<12}")
    print("-" * 88)

    dmd_hits = {}
    det_hits = {}
    for wl in ALL_TEST_WAVELENGTHS:
        cr = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = system.trace_rays([cr])[0]
        h_dmd = next((h for h in tr.history if h["surface_index"] == dmd_idx), None)
        h_det = next((h for h in tr.history if h["surface_index"] == det_idx), None)

        dmd_x = h_dmd["x"] if h_dmd else None
        dmd_y = h_dmd["y"] if h_dmd else None
        det_x = h_det["x"] if h_det else None
        det_y = h_det["y"] if h_det else None

        dmd_hits[wl] = (dmd_x, dmd_y)
        det_hits[wl] = (det_x, det_y)

        within_dmd = (abs(dmd_x) <= 4.332 and abs(dmd_y) <= 2.4365) if dmd_x is not None else False
        dmd_x_str = f"{dmd_x:.4f}" if dmd_x is not None else "N/A"
        dmd_y_str = f"{dmd_y:.4f}" if dmd_y is not None else "N/A"
        det_x_str = f"{det_x:.4f}" if det_x is not None else "N/A"
        det_y_str = f"{det_y:.4f}" if det_y is not None else "N/A"

        print(f"{wl:<18.5f} {dmd_x_str:<14} {dmd_y_str:<14} {det_x_str:<14} {det_y_str:<14} {'YES' if within_dmd else 'NO (CLIPPED)'}")

    disp_1111_1613 = abs(dmd_hits[1.61330][0] - dmd_hits[1.11130][0])
    disp_1111_1667 = abs(dmd_hits[1.66708][0] - dmd_hits[1.11130][0])
    print(f"\nChief Ray Dispersion (1.11130 um -> 1.61330 um): {disp_1111_1613:.4f} mm")
    print(f"Chief Ray Dispersion (1.11130 um -> 1.66708 um): {disp_1111_1667:.4f} mm")
    print(f"TI DLP4500 Active Area: 8.664 mm width ([-4.332, +4.332] mm), 4.873 mm height ([-2.4365, +2.4365] mm)")
    print(f"DMD Dispersion Constraint [8.00 mm, 8.66 mm]: {'PASS' if 8.00 <= disp_1111_1613 <= 8.66 else 'FAIL'}")

    # 2. Dense Ray Grid Tracing (NA=0.12 cone, 25 um slit source)
    print("\n" + "=" * 80)
    print(" 2. DENSE RAY GRID TRACING & TRANSMISSION BREAKDOWN ")
    print("=" * 80)

    # 25 um slit (or 25 um diameter fiber) with NA=0.12 cone
    fiber_radius = 0.0125  # mm (12.5 um)
    na = 0.12
    max_theta = np.arcsin(na)

    spatial_pts = [(0.0, 0.0)]
    for sr, n_pts in [(1, 4), (2, 8)]:
        r_frac = sr / 2.0
        for p in range(n_pts):
            th = 2.0 * np.pi * p / n_pts
            spatial_pts.append((r_frac * fiber_radius * np.cos(th), r_frac * fiber_radius * np.sin(th)))

    dir_pts = [(0.0, 0.0, 1.0)]
    for ar, n_pts in [(1, 6), (2, 12), (3, 18)]:
        a_frac = ar / 3.0
        for p in range(n_pts):
            th = 2.0 * np.pi * p / n_pts
            vx = a_frac * np.sin(max_theta) * np.cos(th)
            vy = a_frac * np.sin(max_theta) * np.sin(th)
            vz = np.sqrt(max(0.0, 1.0 - vx**2 - vy**2))
            dir_pts.append((vx, vy, vz))

    all_rays = []
    for wl in ALL_TEST_WAVELENGTHS:
        for sx, sy in spatial_pts:
            for vx, vy, vz in dir_pts:
                all_rays.append(Ray(
                    pos=np.array([sx, sy, 0.0]),
                    dir=np.array([vx, vy, vz]),
                    wavelength=wl
                ))

    total_dense = len(all_rays)
    print(f"Generated {total_dense} dense rays ({len(spatial_pts)} spatial pts x {len(dir_pts)} angular directions x {len(ALL_TEST_WAVELENGTHS)} wavelengths).")

    traced_dense = system.trace_rays(all_rays)

    # Surface-by-surface loss
    surf_loss = {i: 0 for i in range(len(stack.surfaces))}
    rays_reaching_det_unvignetted = 0
    rays_reaching_det_all = 0

    per_wl_data = {wl: {"x": [], "y": [], "r": [], "vignetted": 0, "total": 0} for wl in ALL_TEST_WAVELENGTHS}

    for r in traced_dense:
        wl = round(float(r.wavelength), 5)
        per_wl_data[wl]["total"] += 1
        if r.history:
            last = r.history[-1]
            last_idx = last["surface_index"]
            if last_idx == det_idx:
                rays_reaching_det_all += 1
                det_r = np.sqrt(last["x"]**2 + last["y"]**2)
                per_wl_data[wl]["x"].append(last["x"])
                per_wl_data[wl]["y"].append(last["y"])
                per_wl_data[wl]["r"].append(det_r)
                if not r.vignetted:
                    rays_reaching_det_unvignetted += 1
                else:
                    surf_loss[det_idx] += 1
                    per_wl_data[wl]["vignetted"] += 1
            else:
                surf_loss[last_idx] += 1
                per_wl_data[wl]["vignetted"] += 1
        else:
            surf_loss[0] += 1
            per_wl_data[wl]["vignetted"] += 1

    print("\n--- SURFACE-BY-SURFACE CLIPPING BREAKDOWN ---")
    print(f"{'Idx':<5} {'Surface Name':<28} {'Semi-Diam (mm)':<16} {'Aperture Type':<15} {'Lost Rays':<10} {'Loss %'}")
    print("-" * 88)
    for i, s in enumerate(stack.surfaces):
        lost = surf_loss[i]
        pct = (lost / float(total_dense)) * 100.0
        ap_val = s.aperture_type.value if hasattr(s.aperture_type, 'value') else str(s.aperture_type)
        print(f"{i:<5} {s.name:<28} {s.semi_diameter:<16.2f} {ap_val:<15} {lost:<10} {pct:.2f}%")

    print(f"\nOverall Optical Transmission (unvignetted at 1.0mm detector): {rays_reaching_det_unvignetted} / {total_dense} ({(rays_reaching_det_unvignetted/total_dense)*100.0:.2f}%)")

    # 3. Detector Spot Size Breakdown Across All Wavelengths
    print("\n" + "=" * 95)
    print(" 3. DETECTOR SPOT SIZE BREAKDOWN ACROSS ALL DISCRETE & INTERMEDIATE WAVELENGTHS ")
    print("=" * 95)
    print(f"{'Wl (um)':<10} {'Rays':<8} {'Centroid (x,y) mm':<22} {'RMS Radius (um)':<16} {'GEO Radius (um)':<16} {'Max Radius (um)':<16} {'Max Diam (um)':<14} {'<=1.0mm?'}")
    print("-" * 115)

    all_det_radii = []
    for wl in ALL_TEST_WAVELENGTHS:
        d = per_wl_data[wl]
        xs = d["x"]
        ys = d["y"]
        rs = d["r"]
        all_det_radii.extend(rs)

        if xs:
            spot = SpotDiagram(xs, ys, wavelength_um=wl)
            cx, cy = spot.calculate_centroid()
            rms_r = spot.calculate_rms_radius() * 1000.0
            geo_r = spot.calculate_geo_radius() * 1000.0
            max_r = np.max(rs) * 1000.0
            max_d = max_r * 2.0
            fits = "PASS" if max_r <= 500.0 else "FAIL"
            print(f"{wl:<10.5f} {len(xs):<8} ({cx:+.3f}, {cy:+.3f}){'':<7} {rms_r:<16.1f} {geo_r:<16.1f} {max_r:<16.1f} {max_d:<14.1f} {fits}")
        else:
            print(f"{wl:<10.5f} {'0':<8} {'NO RAYS':<22} {'N/A':<16} {'N/A':<16} {'N/A':<16} {'N/A':<14} FAIL")

    all_det_r_arr = np.array(all_det_radii) if all_det_radii else np.array([999.0])
    overall_max_r_um = float(np.max(all_det_r_arr)) * 1000.0
    overall_max_d_um = overall_max_r_um * 2.0
    rays_exceeding = int(np.sum(all_det_r_arr > 0.500))

    print("\n--- COMPOSITE POLYCHROMATIC SPOT METRICS ---")
    print(f"Total rays reaching detector plane: {len(all_det_r_arr)}")
    print(f"Maximum Ray Radius from Optical Axis: {overall_max_r_um:.2f} um ({overall_max_r_um/1000.0:.4f} mm)")
    print(f"Maximum Ray Spot Diameter:            {overall_max_d_um:.2f} um ({overall_max_d_um/1000.0:.4f} mm)")
    print(f"Detector Active Area Limit:           Radius <= 500.0 um (Diameter <= 1000.0 um / 1.0 mm)")
    print(f"Rays Exceeding Detector Active Area:  {rays_exceeding} / {len(all_det_r_arr)} ({(rays_exceeding/len(all_det_r_arr))*100.0:.2f}%)")

    # 4. Component Authenticity Audit
    print("\n" + "=" * 80)
    print(" 4. THORLABS 1-INCH ACHROMATIC DOUBLETS PRESCRIPTION AUDIT ")
    print("=" * 80)
    catalog = {
        "AC254-050-C": {"r1": 34.0, "r2": -27.8, "r3": -140.0, "t1": 6.0, "t2": 3.0, "g1": "N-BK7", "g2": "SF5", "sd": 12.7},
        "AC254-040-C": {"r1": 24.3, "r2": -20.6, "r3": -91.9, "t1": 7.0, "t2": 2.5, "g1": "N-LAK22", "g2": "N-SF6HT", "sd": 12.7},
        "AC254-075-C": {"r1": 46.8, "r2": -37.4, "r3": -185.0, "t1": 5.0, "t2": 2.5, "g1": "N-BK7", "g2": "SF5", "sd": 12.7},
        "AC508-075-C": {"r1": 46.8, "r2": -37.4, "r3": -185.0, "t1": 11.5, "t2": 4.0, "g1": "N-BK7", "g2": "SF5", "sd": 25.4},
        "ACL12708U-C": {"r1": 7.6, "r2": 0.0, "t1": 6.5, "g1": "N-BK7", "sd": 6.35},
    }
    for name, spec in catalog.items():
        print(f"Catalog {name}: R1={spec.get('r1')}, R2={spec.get('r2')}, R3={spec.get('r3', 'N/A')}, T1={spec.get('t1')}, Glass={spec.get('g1')}/{spec.get('g2', '')}, SD={spec.get('sd')}")

    print("\nSurfaces in layout:")
    for s in stack.surfaces:
        if "Coll" in s.name or "Focus" in s.name or "Relay" in s.name or "Cond" in s.name:
            print(f"  {s.name:<28} R={s.radius:<8.2f} T={s.thickness:<8.2f} Glass={s.glass:<10} SD={s.semi_diameter:<6.2f}")

    print("\n" + "=" * 80)
    print(" 5. FINAL CHALLENGER 1 SUMMARY ")
    print("=" * 80)
    print(f"1. DMD Dispersion [8.00 - 8.66] mm:        {'PASS' if 8.00 <= disp_1111_1613 <= 8.66 else 'FAIL'} ({disp_1111_1613:.3f} mm)")
    print(f"2. 0.0% Vignetting / 100% Transmission:    {'PASS' if rays_reaching_det_unvignetted == total_dense else 'FAIL'} ({(rays_reaching_det_unvignetted/total_dense)*100.0:.2f}% reaching detector)")
    print(f"3. Spot Size <= 1.0 mm (r <= 0.50 mm):      {'PASS' if rays_exceeding == 0 and overall_max_r_um <= 500.0 else 'FAIL'} (Max r = {overall_max_r_um:.1f} um, Max D = {overall_max_d_um:.1f} um)")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "hybrid_raman_1064.json"
    run_deep_dive(target)
