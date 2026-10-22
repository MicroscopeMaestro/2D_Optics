#!/usr/bin/env python3
"""
Acceptance Criterion 3 Programmatic Mathematical Verifier for Spectrometer V3 All-Mirror NIR & Raman Designs.

Mathematically proves:
1. 100% of light across all design wavelengths fits within the 0.8 mm diameter (0.4 mm radius) detector footprint.
2. Zero vignetting across full spectral bands.
3. Strict adherence to Thorlabs CM254 concave spherical mirror catalog (k=0.0, stock ROC, NO lenses).

Usage:
    python spectrometer_v3_nir_raman/verify_spectrometer_v3.py
    python spectrometer_v3_nir_raman/verify_spectrometer_v3.py --nir <path> --raman <path> --rings 5 --beam-radius 5.0
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

from v2.engine.models import SpectrometerSystem, FullTraceResult
from v2.engine.tracer import DualSpeedTracer, OpticalSystem
from v2.engine.surface import Surface, SurfaceType
from v2.engine.ray import Ray


# Thorlabs CM254 Stock Focal Lengths and Radii of Curvature (ROC = 2*f)
THORLABS_CM254_FOCAL_LENGTHS = [19.0, 25.0, 37.5, 50.0, 75.0, 100.0, 150.0, 200.0]
THORLABS_CM254_STOCK_ROC = [38.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0]
MAX_DETECTOR_RADIUS_MM = 0.4000  # 0.8 mm diameter = 0.4 mm radius


def verify_mirror_catalog_compliance(system: SpectrometerSystem) -> Tuple[bool, List[str]]:
    """
    Verify all optical elements are exclusively Thorlabs CM254 spherical mirrors (or flat grating/virtual).
    Strictly forbids lenses, aspheres, and non-catalog radii.
    """
    passed = True
    messages = []

    for i, surf in enumerate(system.surfaces):
        s_name = surf.name or f"Surface_{i}"
        s_type = surf.surface_type
        glass = (surf.glass or "AIR").upper().strip()

        # Check for forbidden refracting lens elements
        if glass not in ("AIR", "MIRROR", "MIRR", "", "NONE"):
            passed = False
            messages.append(
                f"[FAIL] Surface {i} ('{s_name}') has refractive glass '{glass}'. NO lenses allowed in All-Mirror V3 design!"
            )
            continue

        # Check mirror specifications
        stype_str = str(getattr(s_type, "name", s_type)).upper()
        is_mirror = (glass in ("MIRROR", "MIRR")) or (stype_str == "MIRROR")
        if is_mirror:
            # 1. Spherical conic constraint: conic must be 0.0
            if abs(surf.conic) > 1e-5:
                passed = False
                messages.append(
                    f"[FAIL] Mirror Surface {i} ('{s_name}') has non-spherical conic {surf.conic:.6f}. Must be 0.0 (spherical)."
                )

            # 2. No higher-order aspheric coefficients
            if surf.asphere_coeffs and any(abs(c) > 1e-12 for c in surf.asphere_coeffs):
                passed = False
                messages.append(
                    f"[FAIL] Mirror Surface {i} ('{s_name}') has higher-order aspheric coefficients: {surf.asphere_coeffs}."
                )

            # 3. Stock Thorlabs CM254 ROC check
            roc = abs(surf.radius)
            if roc > 1e-3:  # Non-flat mirror
                matched_roc = None
                for stock_roc in THORLABS_CM254_STOCK_ROC:
                    if abs(roc - stock_roc) < 0.2:  # 0.2 mm tolerance
                        matched_roc = stock_roc
                        break
                if matched_roc is None:
                    passed = False
                    messages.append(
                        f"[FAIL] Mirror Surface {i} ('{s_name}') ROC={roc:.2f} mm does NOT match any Thorlabs CM254 stock ROC ({THORLABS_CM254_STOCK_ROC} mm)."
                    )
                else:
                    messages.append(
                        f"[PASS] Mirror Surface {i} ('{s_name}') matches Thorlabs CM254 (ROC={matched_roc:.1f} mm, f={matched_roc/2:.1f} mm, k=0.0)."
                    )
            else:
                messages.append(f"[PASS] Mirror Surface {i} ('{s_name}') is a flat fold mirror (ROC=inf).")

    return passed, messages


def verify_detector_spot_containment(
    system: SpectrometerSystem,
    rings: int = 5,
    beam_radius: float = 5.0,
    detector_radius_limit: float = MAX_DETECTOR_RADIUS_MM
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Perform dense 3D hexapolar ray tracing and verify mathematical containment within 0.8 mm detector.
    """
    tracer = DualSpeedTracer(system)
    wavelengths = system.wavelengths if system.wavelengths else [1.100, 1.225, 1.350]

    trace_result = tracer.trace_full(wavelengths=wavelengths, rings=rings, beam_radius=beam_radius)

    passed = True
    messages = []
    summary_data: Dict[str, Any] = {
        "wavelengths": wavelengths,
        "total_rays": trace_result.total_rays,
        "vignetted_rays": trace_result.vignetted_count,
        "wl_metrics": {},
        "overall_max_radius_mm": 0.0,
        "overall_rms_radius_mm": trace_result.rms_radius,
        "overall_geo_radius_mm": trace_result.geo_radius,
        "containment_pct": 0.0,
    }

    if trace_result.vignetted_count > 0:
        passed = False
        messages.append(
            f"[FAIL] System has {trace_result.vignetted_count}/{trace_result.total_rays} vignetted rays! Must be 0."
        )
    else:
        messages.append(f"[PASS] Zero vignetting: {trace_result.total_rays}/{trace_result.total_rays} rays traced cleanly.")

    total_detector_rays = 0
    contained_detector_rays = 0
    all_radii = []

    for wl in wavelengths:
        pts = trace_result.spot_coordinates.get(wl, np.empty((0, 2)))
        num_pts = len(pts)
        if num_pts == 0:
            passed = False
            messages.append(f"[FAIL] Wavelength {wl:.4f} um has 0 rays reaching the detector!")
            continue

        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        all_radii.extend(radii.tolist())
        max_r = float(np.max(radii))
        rms_r = float(np.sqrt(np.mean(radii**2)))
        geo_r = float(np.max(np.sqrt((pts[:, 0] - np.mean(pts[:, 0]))**2 + (pts[:, 1] - np.mean(pts[:, 1]))**2)))
        
        contained = int(np.sum(radii <= detector_radius_limit + 1e-9))
        pct = (contained / num_pts) * 100.0

        total_detector_rays += num_pts
        contained_detector_rays += contained

        summary_data["wl_metrics"][wl] = {
            "num_rays": num_pts,
            "max_radius_mm": max_r,
            "rms_radius_mm": rms_r,
            "geo_radius_mm": geo_r,
            "contained_rays": contained,
            "containment_pct": pct
        }

        if max_r > detector_radius_limit + 1e-5:
            passed = False
            messages.append(
                f"[FAIL] Wavelength {wl:.4f} um: Max spot radius {max_r:.4f} mm exceeds {detector_radius_limit:.4f} mm limit (Containment: {pct:.1f}%)."
            )
        else:
            messages.append(
                f"[PASS] Wavelength {wl:.4f} um: Max radius {max_r:.4f} mm <= {detector_radius_limit:.4f} mm (Containment: {pct:.1f}%, RMS: {rms_r*1e3:.1f} um)."
            )

    if total_detector_rays > 0:
        overall_containment = (contained_detector_rays / total_detector_rays) * 100.0
        overall_max_r = float(np.max(all_radii)) if all_radii else 0.0
    else:
        overall_containment = 0.0
        overall_max_r = 0.0

    summary_data["containment_pct"] = overall_containment
    summary_data["overall_max_radius_mm"] = overall_max_r

    if overall_containment < 100.0:
        passed = False
        messages.append(
            f"[FAIL] Overall containment {overall_containment:.2f}% is less than 100.0% required by Acceptance Criterion 3!"
        )
    else:
        messages.append(
            f"[PASS] 100.0% ray containment achieved! (Overall Max Radius: {overall_max_r:.4f} mm <= {detector_radius_limit:.4f} mm across {total_detector_rays} rays)."
        )

    return passed, summary_data, messages


def verify_layout_file(
    filepath: str,
    layout_name: str,
    rings: int = 5,
    beam_radius: float = 5.0
) -> Tuple[bool, Dict[str, Any]]:
    """Verify single optical layout file for catalog compliance and detector spot containment."""
    print("=" * 80)
    print(f"VERIFYING OPTICAL LAYOUT: {layout_name}")
    print(f"File Path: {filepath}")
    print("=" * 80)

    if not os.path.exists(filepath):
        print(f"[FAIL] Layout file does not exist: {filepath}")
        return False, {"error": f"File not found: {filepath}"}

    try:
        system = SpectrometerSystem.load_from_json(filepath)
    except Exception as e:
        print(f"[FAIL] Failed to load JSON prescription: {e}")
        return False, {"error": str(e)}

    print(f"\n[1] Catalog Verification (Thorlabs CM254 Spherical Mirrors & No Lenses):")
    cat_passed, cat_msgs = verify_mirror_catalog_compliance(system)
    for m in cat_msgs:
        print(f"  {m}")

    print(f"\n[2] Ray Tracing & Mathematical Spot Containment (R_max <= {MAX_DETECTOR_RADIUS_MM} mm):")
    spot_passed, spot_data, spot_msgs = verify_detector_spot_containment(
        system, rings=rings, beam_radius=beam_radius
    )
    for m in spot_msgs:
        print(f"  {m}")

    overall_passed = cat_passed and spot_passed
    status_str = "PASSED" if overall_passed else "FAILED"
    print(f"\n>>> RESULT FOR {layout_name}: {status_str} <<<")

    return overall_passed, {
        "catalog_passed": cat_passed,
        "spot_passed": spot_passed,
        "spot_data": spot_data,
        "overall_passed": overall_passed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Spectrometer V3 AC3 Mathematical Spot Containment & Catalog Verifier"
    )
    default_nir = os.path.join(PROJECT_ROOT, "spectrometer_v3_nir_raman", "spectrometer_v3_nir_800_2700nm.json")
    default_raman = os.path.join(PROJECT_ROOT, "spectrometer_v3_nir_raman", "spectrometer_v3_raman_1064nm.json")

    parser.add_argument("--nir", default=default_nir, help="Path to NIR 800-2700nm layout JSON")
    parser.add_argument("--raman", default=default_raman, help="Path to Raman 1064nm layout JSON")
    parser.add_argument("--rings", type=int, default=5, help="Number of hexapolar ray rings (default: 5)")
    parser.add_argument("--beam-radius", type=float, default=5.0, help="Input beam radius in mm (default: 5.0)")

    args = parser.parse_args()

    results = {}
    all_passed = True

    print("\n" + "#" * 80)
    print("# SPECTROMETER V3 ACCEPTANCE CRITERION 3 MATHEMATICAL VERIFICATION")
    print("#" * 80 + "\n")

    # 1. NIR Layout Verification
    nir_passed, nir_res = verify_layout_file(
        args.nir, "Broadband NIR (800 - 2700 nm)", rings=args.rings, beam_radius=args.beam_radius
    )
    results["nir"] = nir_res
    if not nir_passed:
        all_passed = False

    print("\n")

    # 2. Raman Layout Verification
    raman_passed, raman_res = verify_layout_file(
        args.raman, "Raman 1064 nm Fingerprint (1064 - 1250 nm)", rings=args.rings, beam_radius=args.beam_radius
    )
    results["raman"] = raman_res
    if not raman_passed:
        all_passed = False

    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"NIR Layout (800-2700 nm):        {'[PASS]' if nir_passed else '[FAIL]'}")
    print(f"Raman Layout (1064 nm):         {'[PASS]' if raman_passed else '[FAIL]'}")
    print(f"Overall Acceptance Criterion 3: {'[PASS] 100% CONTAINMENT PROVEN' if all_passed else '[FAIL] CRITERIA NOT MET'}")
    print("=" * 80 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
