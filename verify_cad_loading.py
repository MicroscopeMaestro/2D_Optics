#!/usr/bin/env python3
"""
verify_cad_loading.py: Standalone verification script for 3D STEP & SolidWorks CAD loading,
Z-axis section slicing, and optical sag parameterization.
"""

import os
import sys
import json
import argparse
from typing import Tuple, Dict, List, Optional, Any
import numpy as np

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.cad.importer import CADImporter, CADSurfaceImporter, CADFileFormat
from optics_app.cad.slicer import ZSlicer, CADAligner
from optics_app.cad.fitter import SagFitter, ProfilePoints
from optics_app.cad.profile import OpticalProfile, FitType


def generate_synthetic_cad_sphere(radius: float = 50.0, thickness: float = 10.0, semi_diameter: float = 20.0, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic 3D point cloud for a biconvex spherical lens."""
    r = np.linspace(0.0, semi_diameter, num_points)
    c = 1.0 / radius
    sag_front = c * r**2 / (1.0 + np.sqrt(np.maximum(0.0, 1.0 - c**2 * r**2)))
    sag_back = thickness - (c * r**2 / (1.0 + np.sqrt(np.maximum(0.0, 1.0 - c**2 * r**2))))

    # 3D points
    theta = np.linspace(0, 2 * np.pi, 30)
    R_grid, T_grid = np.meshgrid(r, theta)
    X = (R_grid * np.cos(T_grid)).flatten()
    Y = (R_grid * np.sin(T_grid)).flatten()

    Z_front = (c * (X**2 + Y**2) / (1.0 + np.sqrt(np.maximum(0.0, 1.0 - c**2 * (X**2 + Y**2))))).flatten()
    Z_back = (thickness - c * (X**2 + Y**2) / (1.0 + np.sqrt(np.maximum(0.0, 1.0 - c**2 * (X**2 + Y**2))))).flatten()

    pts_front = np.column_stack((X, Y, Z_front))
    pts_back = np.column_stack((X, Y, Z_back))
    pts_all = np.vstack((pts_front, pts_back))

    return pts_all, r, sag_front


def generate_synthetic_cad_asphere(radius: float = 50.0, conic: float = -1.0, a4: float = 1e-6, semi_diameter: float = 20.0, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic 2D contour for a conic/aspheric surface."""
    r = np.linspace(0.0, semi_diameter, num_points)
    c = 1.0 / radius
    z_conic = c * r**2 / (1.0 + np.sqrt(np.maximum(0.0, 1.0 - (1.0 + conic) * c**2 * r**2)))
    z_asph = z_conic + a4 * (r**4)
    return r, z_asph


def verify_cad_loading(
    file_path: Optional[str] = None,
    use_synthetic: bool = False,
    target_fit: str = "conic",
    json_out_path: str = "cad_verification_results.json"
) -> bool:
    """Runs full CAD loading, slicing, parameterization, and verification assertions."""
    print("=" * 70)
    print(" 2D OPTICS ENGINE — CAD IMPORT & PROFILE EXTRACTION VERIFICATION ")
    print("=" * 70)

    results = {
        "status": "PASS",
        "file_tested": file_path or "Synthetic CAD Sphere/Asphere",
        "assertions": [],
        "metrics": {}
    }

    all_passed = True

    def record_assertion(name: str, passed: bool, expected: Any, actual: Any, detail: str = ""):
        nonlocal all_passed
        p_bool = bool(passed)
        if not p_bool:
            all_passed = False
        status_str = "PASS" if p_bool else "FAIL"
        print(f" [{status_str}] {name}: Expected {expected}, Got {actual} {detail}")
        results["assertions"].append({
            "name": name,
            "passed": p_bool,
            "expected": str(expected),
            "actual": str(actual),
            "detail": detail
        })

    # Scenario 1: Load file or test sample files
    sample_dir = os.path.join(os.path.dirname(__file__), "tests", "sample_cad")
    if file_path and os.path.exists(file_path):
        test_files = [file_path]
    elif not use_synthetic and os.path.exists(sample_dir):
        test_files = [
            os.path.join(sample_dir, "spherical_lens.step"),
            os.path.join(sample_dir, "aspheric_lens.step"),
            os.path.join(sample_dir, "sample_solidworks.sldprt"),
        ]
    else:
        test_files = []

    if test_files:
        for tf in test_files:
            if not os.path.exists(tf):
                continue
            print(f"\n---> Testing CAD File: {os.path.basename(tf)}")
            try:
                profile = CADSurfaceImporter.extract_optical_profile(tf, axis='Z')
                record_assertion(f"CAD File Exists & Profile Extracted ({os.path.basename(tf)})", profile is not None, "OpticalProfile", type(profile).__name__)

                # Test normal vector magnitude
                nx, ny, nz = profile.normal_vector(3.0, 4.0, profile.sag(5.0))
                norm_mag = float(np.sqrt(nx**2 + ny**2 + nz**2))
                record_assertion(f"Unit Normal Vector Magnitude ({os.path.basename(tf)})", abs(norm_mag - 1.0) < 1e-4, 1.0, round(norm_mag, 6))

                # Test LDE dict conversion
                lde_dict = profile.to_lde_dict()
                record_assertion(f"LDE Dict Conversion ({os.path.basename(tf)})", "radius" in lde_dict and "semi_diameter" in lde_dict, True, True)

            except Exception as e:
                record_assertion(f"CAD Load Error ({os.path.basename(tf)})", False, "Success", str(e))

    # Scenario 2: Synthetic CAD Sphere Verification Math & Fitting Precision
    print("\n---> Verification math & fitting precision (Synthetic CAD Sphere & Asphere)")
    pts_mesh, r_sph, z_sph = generate_synthetic_cad_sphere(radius=50.0, thickness=10.0, semi_diameter=20.0)

    # Test ZSlicer
    slicer = ZSlicer()
    contour = slicer.slice_cad_mesh(pts_mesh)
    sliced_geo = slicer.decompose_contour(contour[:, 0], contour[:, 1])

    record_assertion("Center Thickness Extraction", abs(sliced_geo.center_thickness - 10.0) < 0.1, 10.0, round(sliced_geo.center_thickness, 4), "mm")
    record_assertion("Semi-Diameter Extraction", abs(sliced_geo.semi_diameter_front - 20.0) < 0.1, 20.0, round(sliced_geo.semi_diameter_front, 4), "mm")

    # Fit spherical
    front_fit = SagFitter.fit(sliced_geo.front_profile, surface_name="Synthetic Front", target_fit=FitType.SPHERICAL)
    record_assertion("Spherical Radius Recovery", abs(front_fit.radius_of_curvature - 50.0) < 0.5, 50.0, round(front_fit.radius_of_curvature, 4), "mm")
    record_assertion("Spherical Conic Recovery", abs(front_fit.conic_constant - 0.0) < 0.01, 0.0, round(front_fit.conic_constant, 4))
    record_assertion("Spherical Fit Residual RMS", front_fit.rms_residual_um < 0.1, "< 0.1 um", round(front_fit.rms_residual_um, 4), "um")

    # Fit synthetic asphere (conic = -1.0)
    r_asph, z_asph = generate_synthetic_cad_asphere(radius=50.0, conic=-1.0, a4=0.0, semi_diameter=20.0)
    prof_pts_asph = ProfilePoints(r=r_asph, z=z_asph, semi_diameter=20.0)
    asph_fit = SagFitter.fit(prof_pts_asph, surface_name="Synthetic Asphere", target_fit=FitType.CONIC)
    record_assertion("Paraboloid Conic Recovery (k=-1.0)", abs(asph_fit.conic_constant - (-1.0)) < 0.05, -1.0, round(asph_fit.conic_constant, 4))
    record_assertion("Aspheric Fit Residual RMS", asph_fit.rms_residual_um < 0.1, "< 0.1 um", round(asph_fit.rms_residual_um, 4), "um")

    # Test 3D Ray-Surface Newton-Raphson Intersection Solver
    t_hit, P_hit, n_hit, is_valid = front_fit.intersect_ray(ray_origin=(5.0, 0.0, -10.0), ray_direction=(0.0, 0.0, 1.0))
    z_expected = front_fit.sag(5.0)
    record_assertion("Ray-Surface Intersection Solver Validity", is_valid, True, is_valid)
    record_assertion("Ray-Surface Intersection Solver Z Accuracy", abs(P_hit[2] - z_expected) < 1e-6, round(z_expected, 6), round(P_hit[2], 6), "mm")

    results["status"] = "PASS" if all_passed else "FAIL"
    results["metrics"] = {
        "spherical_radius_mm": front_fit.radius_of_curvature,
        "spherical_rms_um": front_fit.rms_residual_um,
        "aspheric_conic": asph_fit.conic_constant,
        "aspheric_rms_um": asph_fit.rms_residual_um,
        "center_thickness_mm": sliced_geo.center_thickness,
    }

    # Save JSON results
    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnostic verification report written to: {os.path.abspath(json_out_path)}")

    print("=" * 70)
    if all_passed:
        print(" VERIFICATION RESULT: OVERALL PASS (100% assertions satisfied)")
    else:
        print(" VERIFICATION RESULT: FAIL (One or more assertions failed)")
    print("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify 3D CAD loading, Z-slicing, and sag profile extraction.")
    parser.add_argument("--file", type=str, default=None, help="Path to STEP/SolidWorks CAD file to test")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic CAD geometry verification")
    parser.add_argument("--fit-mode", type=str, default="conic", choices=["spherical", "conic", "aspheric", "spline"], help="Target fit mode")
    parser.add_argument("--json-out", type=str, default="cad_verification_results.json", help="Output path for JSON report")

    args = parser.parse_args()

    success = verify_cad_loading(
        file_path=args.file,
        use_synthetic=args.synthetic,
        target_fit=args.fit_mode,
        json_out_path=args.json_out
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
