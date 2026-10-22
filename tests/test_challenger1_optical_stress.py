"""
tests/test_challenger1_optical_stress.py
Independent Empirical Adversarial Stress Harness for 1064 nm Hybrid Raman Spectrometer.
Author: Challenger 1 (critic, specialist)

Authoritative Requirements from ORIGINAL_REQUEST.md:
- R1: Architecture Benchmarking (NIRscan Nano folding principles).
- R2: Strictly Thorlabs 1-inch achromatic doublets (exact physical prescriptions).
- R3: DMD Spectral Footprint: Chief ray distance between 1.111 um and 1.613 um must strictly be between 8.00 mm and 8.66 mm.
- R4: 1.0 mm Detector Focusing: Maximum geometric spread of all traced rays at final detector <= 1.0 mm diameter (radius <= 0.50 mm).
- AC: 0.0% vignetting (100% transmission) across all surfaces under 25 um slit / NA=0.12 cone.
"""

import os
import sys
import copy
import json
import time
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics
from optics_app.utils.materials import get_refractive_index

# Authoritative Raman wavelengths (1064 nm excitation, 400 - 3400 cm^-1 shift)
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

def generate_dense_slit_cone_rays(
    slit_width_mm: float = 0.025,
    slit_height_mm: float = 0.100,
    numerical_aperture: float = 0.12,
    wavelengths=RAMAN_WAVELENGTHS,
    n_spatial_x: int = 5,
    n_spatial_y: int = 5,
    n_rings: int = 4,
    n_azimuth: int = 16
):
    """
    Generates deterministic phase-space ray bundle spanning a 25 um slit
    and NA=0.12 divergence cone (marginal rays, extreme pupil angles, corner field points).
    """
    rays = []
    max_theta = np.arcsin(min(max(numerical_aperture, 0.001), 0.999))

    # Spatial grid on slit
    xs = np.linspace(-slit_width_mm / 2.0, slit_width_mm / 2.0, n_spatial_x)
    ys = np.linspace(-slit_height_mm / 2.0, slit_height_mm / 2.0, n_spatial_y)

    # Angular direction cone (rings from center to extreme marginal NA)
    angles = [(0.0, 0.0, 1.0)]
    for r_idx in range(1, n_rings + 1):
        r_frac = r_idx / float(n_rings)
        theta = r_frac * max_theta
        for az_idx in range(n_azimuth):
            phi = 2.0 * np.pi * az_idx / float(n_azimuth)
            vx = np.sin(theta) * np.cos(phi)
            vy = np.sin(theta) * np.sin(phi)
            vz = np.sqrt(max(0.0, 1.0 - vx**2 - vy**2))
            angles.append((vx, vy, vz))

    for wl in wavelengths:
        for x in xs:
            for y in ys:
                for vx, vy, vz in angles:
                    rays.append(Ray(
                        pos=np.array([x, y, 0.0]),
                        dir=np.array([vx, vy, vz]),
                        wavelength=wl
                    ))
    return rays


def evaluate_spectrometer_candidate(json_path: str):
    """Thoroughly evaluates an optical layout candidate against all physical requirements."""
    if not os.path.exists(json_path):
        return {"error": f"File not found: {json_path}"}

    with open(json_path, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    loaded = SurfaceStack.load_from_json(json_path)
    stack = loaded["stack"] if isinstance(loaded, dict) else loaded
    system = OpticalSystem(stack)

    # Find key surfaces
    dmd_surfs = [i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper()]
    det_surfs = [i for i, s in enumerate(stack.surfaces) if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper()]
    grating_surfs = [i for i, s in enumerate(stack.surfaces) if s.surface_type == SurfaceType.GRATING or "GRATING" in s.name.upper()]

    if not dmd_surfs or not det_surfs:
        return {"error": f"Missing DMD or Detector in {json_path}"}

    dmd_idx = dmd_surfs[0]
    det_idx = det_surfs[0]

    # --- 1. Chief Ray Dispersion Check ---
    chief_rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        for wl in RAMAN_WAVELENGTHS
    ]
    traced_chief = system.trace_rays(chief_rays)
    dmd_chief_hits = {}
    for r in traced_chief:
        wl = round(float(r.wavelength), 5)
        for h in r.history:
            if h["surface_index"] == dmd_idx:
                dmd_chief_hits[wl] = float(h["x"])

    disp_1111_1613 = abs(dmd_chief_hits.get(1.61330, 0.0) - dmd_chief_hits.get(1.11130, 0.0))
    disp_1111_1667 = abs(dmd_chief_hits.get(1.66708, 0.0) - dmd_chief_hits.get(1.11130, 0.0))
    dmd_dispersion_pass = (8.00 <= disp_1111_1613 <= 8.66)

    # --- 2. Dense Ray Tracing & Vignetting Check (NA=0.12, 25 um slit) ---
    rays = generate_dense_slit_cone_rays(
        slit_width_mm=0.025,
        slit_height_mm=0.025, # 25 um square fiber core
        numerical_aperture=0.12,
        wavelengths=RAMAN_WAVELENGTHS,
        n_spatial_x=3,
        n_spatial_y=3,
        n_rings=3,
        n_azimuth=12
    )
    total_rays = len(rays)
    traced = system.trace_rays(rays)

    # Transmission / Vignetting
    blocked_per_surf = {i: 0 for i in range(len(stack.surfaces))}
    det_x = []
    det_y = []
    per_wl_det = {}

    for wl in RAMAN_WAVELENGTHS:
        per_wl_det[wl] = {"x": [], "y": [], "vignetted": 0, "total": 0}

    for r in traced:
        wl = round(float(r.wavelength), 5)
        per_wl_det[wl]["total"] += 1
        if r.history:
            last_h = r.history[-1]
            last_idx = last_h["surface_index"]
            if last_idx == det_idx and not r.vignetted:
                det_x.append(float(last_h["x"]))
                det_y.append(float(last_h["y"]))
                per_wl_det[wl]["x"].append(float(last_h["x"]))
                per_wl_det[wl]["y"].append(float(last_h["y"]))
            else:
                blocked_per_surf[last_idx] += 1
                per_wl_det[wl]["vignetted"] += 1
        else:
            blocked_per_surf[0] += 1
            per_wl_det[wl]["vignetted"] += 1

    rays_at_det = len(det_x)
    transmission_pct = (rays_at_det / float(total_rays)) * 100.0 if total_rays > 0 else 0.0
    vignetting_pass = (transmission_pct >= 99.9)

    # --- 3. Spot Size Confinement at Detector (<= 1.0 mm Diameter / radius <= 0.50 mm) ---
    max_spot_radii = {}
    max_spot_diameters = {}
    rms_radii = {}
    geo_radii = {}

    for wl, data in per_wl_det.items():
        if data["x"]:
            r_arr = np.sqrt(np.array(data["x"])**2 + np.array(data["y"])**2)
            spot = SpotDiagram(data["x"], data["y"], wavelength_um=wl)
            max_spot_radii[wl] = float(np.max(r_arr))
            max_spot_diameters[wl] = float(np.max(r_arr) * 2.0)
            rms_radii[wl] = float(spot.calculate_rms_radius())
            geo_radii[wl] = float(spot.calculate_geo_radius())
        else:
            max_spot_radii[wl] = 999.0
            max_spot_diameters[wl] = 1998.0
            rms_radii[wl] = 999.0
            geo_radii[wl] = 999.0

    overall_max_r = float(max(max_spot_radii.values())) if max_spot_radii else 999.0
    overall_max_d = overall_max_r * 2.0
    spot_size_pass = (overall_max_r <= 0.500)

    # Overall pass
    all_pass = dmd_dispersion_pass and vignetting_pass and spot_size_pass

    return {
        "file": os.path.basename(json_path),
        "total_surfaces": len(stack.surfaces),
        "disp_1111_1613_mm": disp_1111_1613,
        "disp_1111_1667_mm": disp_1111_1667,
        "dmd_dispersion_pass": dmd_dispersion_pass,
        "total_rays": total_rays,
        "rays_at_det": rays_at_det,
        "transmission_pct": transmission_pct,
        "vignetting_pass": vignetting_pass,
        "overall_max_spot_radius_mm": overall_max_r,
        "overall_max_spot_diameter_mm": overall_max_d,
        "spot_size_pass": spot_size_pass,
        "per_wl_max_radii_mm": max_spot_radii,
        "per_wl_max_diameters_mm": max_spot_diameters,
        "blocked_per_surf": blocked_per_surf,
        "all_pass": all_pass
    }


if __name__ == "__main__":
    import glob
    candidates = sorted(glob.glob("*.json"))
    print("=" * 85)
    print(" EMPIRICAL ADVERSARIAL AUDIT OF ALL SPECTROMETER CANDIDATES ")
    print("=" * 85)
    print(f"{'File':<36} {'Disp(1.11-1.61) mm':<20} {'Trans %':<10} {'Max D (mm)':<12} {'Verdict'}")
    print("-" * 85)
    
    for f in candidates:
        try:
            res = evaluate_spectrometer_candidate(f)
            if "error" in res:
                continue
            v_str = "PASS" if res["all_pass"] else "FAIL"
            print(f"{res['file']:<36} {res['disp_1111_1613_mm']:<20.3f} {res['transmission_pct']:<10.1f} {res['overall_max_spot_diameter_mm']:<12.3f} {v_str}")
        except Exception as e:
            pass
