#!/usr/bin/env python3
"""
verify_polarization_physics.py: Standalone Agent-as-Judge verification script
for Requirement R5 (Fresnel equations, Brewster angle, TIR phase shift,
Jones calculus, Complex Apertures vignetting, and Gaussian Beam propagation).
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone
from typing import Tuple, Dict, Any
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.engine.polarization import (
    compute_fresnel_coefficients, compute_sp_basis, rotate_jones_vector
)
from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.utils.materials import get_complex_refractive_index
from optics_app.engine.gaussian import GaussianBeam, trace_system_gaussian_beam


def run_verification() -> Tuple[bool, Dict[str, Any]]:
    assertions = []
    metrics = {}

    def log_assertion(name: str, passed: bool, expected: Any, actual: Any, detail: str, tol: float = 1e-5):
        assertions.append({
            "name": name,
            "passed": bool(passed),
            "expected": str(expected),
            "actual": str(actual),
            "tolerance": tol,
            "detail": detail
        })
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_str} {name}: expected={expected}, actual={actual}")

    print("======================================================================")
    print("      Requirement R5 Physics & Polarization Engine Verification       ")
    print("======================================================================")

    # 1. Dielectric Fresnel Equations
    print("\n--- 1. Dielectric Fresnel Equations (Air -> Glass n=1.5) ---")
    f_diel = compute_fresnel_coefficients(1.0, 1.5, 1.0)
    r_s_abs = abs(f_diel.r_s)
    r_p_abs = abs(f_diel.r_p)
    R_norm = r_s_abs**2
    T_norm = (1.5 / 1.0) * abs(f_diel.t_s)**2

    log_assertion(
        "Dielectric Normal Reflectance R (n=1.5)",
        abs(R_norm - 0.04) < 1e-5,
        0.04, round(R_norm, 6),
        "Air to glass normal incidence reflectance R = 4.000%"
    )
    log_assertion(
        "Dielectric Normal Transmittance T (n=1.5)",
        abs(T_norm - 0.96) < 1e-5,
        0.96, round(T_norm, 6),
        "Air to glass normal incidence transmittance T = 96.000%"
    )
    log_assertion(
        "Dielectric Normal Energy Conservation R+T=1",
        abs((R_norm + T_norm) - 1.0) < 1e-5,
        1.0, round(R_norm + T_norm, 6),
        "Energy conservation R + T = 1.0"
    )
    metrics["dielectric_normal_reflectance"] = float(R_norm)
    metrics["dielectric_normal_transmittance"] = float(T_norm)

    # Oblique angle energy conservation
    f_obl = compute_fresnel_coefficients(1.0, 1.5, np.cos(np.radians(45)))
    sin_t2 = (1.0 / 1.5) * np.sin(np.radians(45))
    cos_t2 = np.sqrt(1.0 - sin_t2**2)
    R_s_obl = abs(f_obl.r_s)**2
    T_s_obl = (1.5 * cos_t2) / (1.0 * np.cos(np.radians(45))) * abs(f_obl.t_s)**2
    log_assertion(
        "Dielectric Oblique 45deg Energy Conservation (s-pol)",
        abs((R_s_obl + T_s_obl) - 1.0) < 1e-5,
        1.0, round(R_s_obl + T_s_obl, 6),
        "Energy conservation at 45 deg incidence for s-polarization"
    )

    # 2. Metal Fresnel Equations
    print("\n--- 2. Metal Complex Index Fresnel Reflection ---")
    au_n = get_complex_refractive_index("GOLD")
    f_au = compute_fresnel_coefficients(1.0, au_n, 1.0)
    R_au = abs(f_au.r_s)**2
    log_assertion(
        "Gold (Au) Normal Reflectance",
        abs(R_au - 0.930718) < 1e-3,
        "~0.930718", round(R_au, 6),
        "Gold normal incidence reflectance R = 93.072%"
    )
    metrics["gold_normal_reflectance"] = float(R_au)

    ag_n = get_complex_refractive_index("SILVER")
    f_ag = compute_fresnel_coefficients(1.0, ag_n, 1.0)
    R_ag = abs(f_ag.r_s)**2
    log_assertion(
        "Silver (Ag) Normal Reflectance",
        abs(R_ag - 0.989656) < 1e-3,
        "~0.989656", round(R_ag, 6),
        "Silver normal incidence reflectance R = 98.966%"
    )
    metrics["silver_normal_reflectance"] = float(R_ag)

    # 3. Brewster Angle
    print("\n--- 3. Brewster Angle Zero r_p Condition ---")
    theta_B = np.arctan(1.5 / 1.0)
    f_brewster = compute_fresnel_coefficients(1.0, 1.5, np.cos(theta_B))
    log_assertion(
        "Brewster Angle Zero r_p Magnitude",
        abs(f_brewster.r_p) < 1e-6,
        "<1e-6", f"{abs(f_brewster.r_p):.2e}",
        "p-polarization reflection vanishes at Brewster angle ~56.31 deg"
    )
    metrics["brewster_angle_deg"] = float(np.degrees(theta_B))
    metrics["brewster_rp_abs"] = float(abs(f_brewster.r_p))

    # 4. Total Internal Reflection & Fresnel Rhomb Phase Shift
    print("\n--- 4. Total Internal Reflection (TIR) & Phase Shift ---")
    theta_c = np.arcsin(1.0 / 1.5)
    f_tir = compute_fresnel_coefficients(1.5, 1.0, np.cos(np.radians(50)))
    log_assertion(
        "TIR Critical Angle Condition (n1=1.5 -> n2=1.0)",
        f_tir.tir and abs(abs(f_tir.r_s) - 1.0) < 1e-6,
        "tir=True, |r_s|=1.0", f"tir={f_tir.tir}, |r_s|={abs(f_tir.r_s):.6f}",
        "TIR yields unit reflectance above critical angle ~41.81 deg"
    )
    metrics["tir_critical_angle_deg"] = float(np.degrees(theta_c))

    f_rhomb = compute_fresnel_coefficients(1.5, 1.0, np.cos(np.radians(53.3)))
    delta_s = np.angle(f_rhomb.r_s)
    delta_p = np.angle(f_rhomb.r_p)
    raw_diff_deg = np.degrees(delta_p - delta_s)
    delta_deg = (raw_diff_deg + 180.0) % 360.0 - 180.0
    retardance = abs(delta_deg)
    # Two reflections yield 90 deg QWP retardance (~45 deg per reflection)
    log_assertion(
        "Fresnel Rhomb Phase Retardance per TIR Reflection",
        abs(retardance - 45.0) < 1.0,
        "45.0 deg", f"{retardance:.2f} deg",
        "Phase retardation per glass TIR reflection (Fresnel rhomb angle ~53.3 deg)"
    )
    metrics["fresnel_rhomb_single_reflection_phase_deg"] = float(retardance)

    # 5. Jones Calculus & Component Propagation
    print("\n--- 5. Jones Calculus (Malus Law & 3-Polarizer Paradox) ---")
    # Malus Law at 45 deg
    def make_lp(angle_deg):
        th = np.radians(angle_deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)

    lp45 = make_lp(45)
    J_h = np.array([1.0+0j, 0.0+0j])
    J_out_45 = lp45 @ J_h
    I_45 = float(np.sum(np.abs(J_out_45)**2))
    log_assertion(
        "Malus Law 45deg Transmitted Intensity",
        abs(I_45 - 0.5) < 1e-6,
        0.5, I_45,
        "Linear polarizer at 45 deg transmits 50% intensity of horizontal input"
    )
    metrics["malus_law_45deg_intensity"] = float(I_45)

    # 3-Polarizer Paradox
    lp0 = make_lp(0)
    lp90 = make_lp(90)
    J_3pol = lp90 @ (lp45 @ (lp0 @ J_h))
    I_3pol = float(np.sum(np.abs(J_3pol)**2))
    log_assertion(
        "3-Polarizer Paradox Transmitted Fraction",
        abs(I_3pol - 0.25) < 1e-6,
        0.25, I_3pol,
        "Adding 45 deg polarizer between crossed polarizers transmits 25% of horizontal input (12.5% of unpolarized)"
    )
    metrics["three_polarizer_transmitted_fraction"] = float(I_3pol * 0.5)

    # 6. Complex Apertures & Vignetting
    print("\n--- 6. Complex Apertures & Spatial Vignetting ---")
    surf_rect = Surface(aperture_type="Rectangular", aperture_x_half=10.0, aperture_y_half=20.0)
    pass_rect = surf_rect.check_aperture(5.0, 15.0)
    block_rect = surf_rect.check_aperture(15.0, 5.0)
    log_assertion(
        "Rectangular Aperture Pass/Block Check",
        pass_rect and not block_rect,
        "pass=(5,15), block=(15,5)", f"pass={pass_rect}, block={block_rect}",
        "Rectangular boundary clipping check"
    )

    surf_spider = Surface(
        aperture_type="Spider",
        aperture_r_outer=25.0,
        aperture_center_radius=5.0,
        aperture_spider_width=2.0,
        aperture_spider_count=4,
        aperture_spider_angle=0.0
    )
    block_hub = surf_spider.check_aperture(3.0, 0.0)
    block_leg = surf_spider.check_aperture(15.0, 0.5)
    pass_spider = surf_spider.check_aperture(15.0, 5.0)
    log_assertion(
        "Spider Aperture Central Hub and Leg Obscuration",
        (not block_hub) and (not block_leg) and pass_spider,
        "hub_blocked=True, leg_blocked=True, clear_passed=True",
        f"hub_passed={block_hub}, leg_passed={block_leg}, clear_passed={pass_spider}",
        "Spider secondary mirror support obscuration"
    )

    # 7. Gaussian Beam Propagation (ABCD Matrix)
    print("\n--- 7. Gaussian Beam Propagation (ABCD Matrix & q-Parameter) ---")
    w0 = 0.1  # 0.1 mm
    wl = 0.6328  # 632.8 nm in um
    beam = GaussianBeam(w0=w0, wavelength=wl, n=1.0, z=0.0)
    wl_mm = wl * 1e-3
    zR = (np.pi * (w0**2)) / wl_mm

    beam.propagate_distance(zR)
    w_zR = beam.spot_radius
    expected_w_zR = w0 * np.sqrt(2.0)
    log_assertion(
        "Gaussian Beam Free-Space Expansion at z = zR",
        abs(w_zR - expected_w_zR) < 1e-5,
        round(expected_w_zR, 6), round(w_zR, 6),
        "Spot radius w(zR) = sqrt(2) * w0"
    )
    metrics["gaussian_beam_w0_mm"] = float(w0)
    metrics["gaussian_beam_zR_mm"] = float(zR)
    metrics["gaussian_beam_w_at_zR_mm"] = float(w_zR)

    all_passed = all(a["passed"] for a in assertions)
    passed_cnt = sum(1 for a in assertions if a["passed"])
    total_cnt = len(assertions)

    result_doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_passed else "FAIL",
        "total_assertions": total_cnt,
        "passed_assertions": passed_cnt,
        "failed_assertions": total_cnt - passed_cnt,
        "metrics": metrics,
        "assertions": assertions
    }

    print("\n======================================================================")
    print(f" Verification Summary: {passed_cnt}/{total_cnt} assertions passed.")
    print(f" OVERALL STATUS: {'PASS' if all_passed else 'FAIL'}")
    print("======================================================================")

    return all_passed, result_doc


def main():
    parser = argparse.ArgumentParser(description="Agent-as-Judge Physics Verification for Requirement R5")
    parser.add_argument("--json-out", default="polarization_verification_results.json", help="Output JSON path")
    args = parser.parse_args()

    success, report = run_verification()

    out_path = os.path.abspath(args.json_out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Verification JSON exported to: {out_path}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
