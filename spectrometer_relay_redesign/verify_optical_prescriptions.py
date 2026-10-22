#!/usr/bin/env python3
"""
Mathematical Spot Footprint and Optical Prescription Verification Engine.

This script programmatically loads and traces dense 2D pupil grids (hexapolar/polar >= 300 rays)
through both spectrometer optical prescriptions:
  1. Primary Spectrometer Relay Redesign (1.100 - 1.350 um)
  2. Dedicated NIR Spectrometer Layout (0.900 - 1.700 um)

Acceptance Criteria & Mathematical Proofs:
  - 100% ray containment within 0.8 mm detector aperture (radius r_i <= 0.4000 mm).
  - 100% optical transmission efficiency (zero intermediate or aperture vignetting).
  - Comprehensive calculation of centroid, RMS spot radius, and 100% geometric spot radius.
  - Formatted diagnostic ASCII tables reporting all per-channel metrics.
"""

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2.engine.surface import Surface, SurfaceType
from v2.engine.tracer import OpticalSystem, Ray
from v2.analysis.spot import SpotDiagram, calculate_spot_metrics


# ==============================================================================
# DEFAULT CONSTANTS & PATHS
# ==============================================================================

DEFAULT_PRIMARY_PATH = SCRIPT_DIR / "spectrometer_primary_relay_redesign.json"
DEFAULT_NIR_PATH = SCRIPT_DIR / "spectrometer_nir_900_1700nm.json"

PRIMARY_WAVELENGTHS = [1.100, 1.225, 1.350]
NIR_WAVELENGTHS = [0.900, 1.100, 1.300, 1.500, 1.700]

NA_ENTRANCE = 0.12
DETECTOR_RADIUS_LIMIT_MM = 0.4000
MIN_RAYS_PER_WAVELENGTH = 300


# ==============================================================================
# PUPIL GRID GENERATION
# ==============================================================================

def generate_hexapolar_pupil_grid(
    na: float = NA_ENTRANCE,
    num_rings: int = 10,
    wavelength_um: float = 1.225,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> List[Ray]:
    """
    Generate dense hexapolar 2D pupil ray grid within numerical aperture cone sin(theta) <= NA.
    For num_rings=10, yields 1 + 3*10*11 = 331 rays (>= 300 requirement).
    """
    rays = [
        Ray(
            pos=np.array(origin, dtype=float),
            dir=np.array([0.0, 0.0, 1.0], dtype=float),
            wavelength=wavelength_um
        )
    ]
    max_theta = math.asin(min(1.0, max(0.0, na)))

    for ring_idx in range(1, num_rings + 1):
        frac = ring_idx / float(num_rings)
        theta = frac * max_theta
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        n_pts = 6 * ring_idx

        for p in range(n_pts):
            phi = 2.0 * math.pi * p / float(n_pts)
            dx = sin_t * math.cos(phi)
            dy = sin_t * math.sin(phi)
            dz = cos_t
            d_vec = np.array([dx, dy, dz], dtype=float)
            d_vec /= np.linalg.norm(d_vec)
            rays.append(Ray(pos=np.array(origin, dtype=float), dir=d_vec, wavelength=wavelength_um))

    return rays


def generate_polar_pupil_grid(
    na: float = NA_ENTRANCE,
    num_rings: int = 10,
    rays_per_ring: int = 6,
    wavelength_um: float = 1.225,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> List[Ray]:
    """
    Generate dense polar 2D pupil ray grid within NA cone.
    Yields 1 + 6 * (1 + 2 + ... + 10) = 331 rays for num_rings=10.
    """
    return generate_hexapolar_pupil_grid(na=na, num_rings=num_rings, wavelength_um=wavelength_um, origin=origin)


# ==============================================================================
# SINGLE PRESCRIPTION EVALUATION ENGINE
# ==============================================================================

def verify_optical_prescription(
    layout_path: Path,
    wavelengths: List[float],
    layout_name: str,
    na: float = NA_ENTRANCE,
    num_rings: int = 10,
    detector_radius_limit: float = DETECTOR_RADIUS_LIMIT_MM
) -> Dict[str, Any]:
    """
    Trace dense 2D pupil grid across all wavelengths and mathematically prove:
      1. R_max <= detector_radius_limit (0.4000 mm) for 100% of arrived rays.
      2. Transmission efficiency T == 100% (0 vignetted rays).
      3. Compute spot centroid (x_bar, y_bar), RMS radius R_RMS, and geometric radius R_GEO.
    """
    if not layout_path.is_file():
        return {
            "layout_name": layout_name,
            "layout_path": str(layout_path),
            "file_exists": False,
            "all_passed": False,
            "error": f"File not found: {layout_path}",
            "channels": {}
        }

    system = OpticalSystem.from_json(str(layout_path))
    channel_results: Dict[str, Dict[str, Any]] = {}
    layout_all_passed = True

    for wl in wavelengths:
        rays = generate_hexapolar_pupil_grid(na=na, num_rings=num_rings, wavelength_um=wl)
        n_launched = len(rays)
        traced_rays = system.trace_rays(rays)

        x_hits: List[float] = []
        y_hits: List[float] = []
        radii: List[float] = []
        n_vignetted = 0

        for r in traced_rays:
            if r.vignetted:
                n_vignetted += 1
                # If ray vignetted at the detector plane, still record its hit position if reached
                if len(r.history) > 0 and r.history[-1]["surface_index"] == len(system.surfaces) - 1:
                    xh = float(r.history[-1]["x"])
                    yh = float(r.history[-1]["y"])
                    x_hits.append(xh)
                    y_hits.append(yh)
                    radii.append(math.sqrt(xh**2 + yh**2))
            else:
                if len(r.history) > 0:
                    xh = float(r.history[-1]["x"])
                    yh = float(r.history[-1]["y"])
                    x_hits.append(xh)
                    y_hits.append(yh)
                    radii.append(math.sqrt(xh**2 + yh**2))

        n_passed = len([r for r in traced_rays if not r.vignetted])
        transmission = (n_passed / float(n_launched)) if n_launched > 0 else 0.0

        if radii:
            r_max = max(radii)
            spot = SpotDiagram(x_hits, y_hits, wavelength_um=wl)
            cx, cy = spot.calculate_centroid()
            rms_r = spot.calculate_rms_radius()
            geo_r = spot.calculate_geo_radius()
            rms_center = float(np.sqrt(np.mean(np.array(radii)**2)))
        else:
            r_max, cx, cy, rms_r, geo_r, rms_center = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Mathematical Proof Criteria:
        # 1. Zero vignetting across full pupil
        # 2. Maximum radial arrival <= 0.4000 mm
        pass_containment = (r_max <= detector_radius_limit) if radii else False
        pass_transmission = (n_vignetted == 0 and n_passed == n_launched)
        channel_passed = pass_containment and pass_transmission

        if not channel_passed:
            layout_all_passed = False

        channel_key = f"{wl:.3f}"
        channel_results[channel_key] = {
            "wavelength_um": wl,
            "n_launched": n_launched,
            "n_passed": n_passed,
            "n_vignetted": n_vignetted,
            "transmission_pct": transmission * 100.0,
            "centroid_x_mm": cx,
            "centroid_y_mm": cy,
            "rms_radius_mm": rms_r,
            "rms_radius_center_mm": rms_center,
            "geo_radius_mm": geo_r,
            "r_max_mm": r_max,
            "containment_limit_mm": detector_radius_limit,
            "pass_containment": pass_containment,
            "pass_transmission": pass_transmission,
            "passed": channel_passed,
            "x_hits": x_hits,
            "y_hits": y_hits,
            "radii": radii
        }

    return {
        "layout_name": layout_name,
        "layout_path": str(layout_path),
        "file_exists": True,
        "surface_count": len(system.surfaces),
        "all_passed": layout_all_passed,
        "channels": channel_results
    }


# ==============================================================================
# DIAGNOSTIC TABLE FORMATTING
# ==============================================================================

def print_diagnostic_table(result: Dict[str, Any]) -> None:
    """Format and print beautiful ASCII diagnostic table."""
    layout_name = result.get("layout_name", "Unknown Layout")
    layout_path = result.get("layout_path", "")
    all_passed = result.get("all_passed", False)
    channels = result.get("channels", {})

    sep_double = "=" * 105
    sep_single = "-" * 105

    print(sep_double)
    print(f" OPTICAL PRESCRIPTION VERIFICATION: {layout_name}")
    print(f" File: {layout_path}")
    print(sep_double)

    header = (
        f"| {'Wavelength':^11} | {'Launched':^8} | {'Passed':^6} | {'Vignetted':^9} | "
        f"{'Trans %':^7} | {'Centroid (X, Y) mm':^18} | {'RMS (mm)':^8} | {'GEO (mm)':^8} | {'R_max (mm)':^10} | {'Status':^6} |"
    )
    print(header)
    print(sep_single)

    for ch_key, ch in channels.items():
        wl_str = f"{ch['wavelength_um']:.3f} µm"
        n_l = ch["n_launched"]
        n_p = ch["n_passed"]
        n_v = ch["n_vignetted"]
        t_pct = f"{ch['transmission_pct']:.1f}%"
        cent_str = f"({ch['centroid_x_mm']:+.4f}, {ch['centroid_y_mm']:+.4f})"
        rms_str = f"{ch['rms_radius_mm']:.4f}"
        geo_str = f"{ch['geo_radius_mm']:.4f}"
        rmax_str = f"{ch['r_max_mm']:.4f}"
        status_str = "PASS" if ch["passed"] else "FAIL"

        row = (
            f"| {wl_str:^11} | {n_l:^8} | {n_p:^6} | {n_v:^9} | "
            f"{t_pct:^7} | {cent_str:^18} | {rms_str:^8} | {geo_str:^8} | {rmax_str:^10} | {status_str:^6} |"
        )
        print(row)

    print(sep_double)
    summary_status = "PASSED (100% Confinement & 0% Vignetting)" if all_passed else "FAILED"
    print(f" OVERALL LAYOUT VERDICT: {summary_status}")
    print(sep_double)
    print()


# ==============================================================================
# MAIN PROGRAMMATIC ENTRY POINT
# ==============================================================================

def verify_all_layouts(
    primary_path: Optional[Path] = None,
    nir_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Programmatic entry point evaluating both primary relay and NIR 900-1700nm optical layouts.
    
    Returns structured dictionary with overall status and per-layout diagnostic details.
    """
    p_path = Path(primary_path) if primary_path else DEFAULT_PRIMARY_PATH
    n_path = Path(nir_path) if nir_path else DEFAULT_NIR_PATH

    env_p = os.environ.get("SPECTROMETER_PRIMARY_RELAY_PATH")
    if env_p and Path(env_p).is_file():
        p_path = Path(env_p)

    env_n = os.environ.get("SPECTROMETER_NIR_LAYOUT_PATH")
    if env_n and Path(env_n).is_file():
        n_path = Path(env_n)

    # 1. Verify Primary Relay Redesign
    primary_result = verify_optical_prescription(
        layout_path=p_path,
        wavelengths=PRIMARY_WAVELENGTHS,
        layout_name="Primary Spectrometer Relay Redesign (1.100 - 1.350 µm)",
        na=NA_ENTRANCE,
        num_rings=10
    )

    # 2. Verify NIR 900-1700nm Layout
    nir_result = verify_optical_prescription(
        layout_path=n_path,
        wavelengths=NIR_WAVELENGTHS,
        layout_name="NIR 900-1700nm Dedicated Spectrometer Layout",
        na=NA_ENTRANCE,
        num_rings=10
    )

    overall_passed = primary_result.get("all_passed", False) and nir_result.get("all_passed", False)

    if verbose:
        print("\n" + "#" * 105)
        print(" PROGRAMMATIC OPTICAL PRESCRIPTION VERIFICATION REPORT")
        print("#" * 105 + "\n")
        print_diagnostic_table(primary_result)
        print_diagnostic_table(nir_result)

        print("#" * 105)
        if overall_passed:
            print(" FINAL VERDICT: SUCCESS — ALL ACCEPTANCE CRITERIA MATHEMATICALLY PROVED")
            print(" 1. 100% of light fits strictly within 0.8 mm detector aperture (radius <= 0.4000 mm).")
            print(" 2. Zero vignetting / 100.0% transmission across all design wavelengths.")
        else:
            print(" FINAL VERDICT: FAILURE — ONE OR MORE CHANNELS EXCEEDED DETECTOR APERTURE OR VIGNETTED")
        print("#" * 105 + "\n")

    return {
        "success": overall_passed,
        "all_passed": overall_passed,
        "primary_relay": primary_result,
        "nir_layout": nir_result
    }


# Convenience function aliases for compatibility with test runners
verify_all_prescriptions = verify_all_layouts
verify_prescriptions = verify_all_layouts


def verify_primary_relay(layout_path: Optional[Path] = None, verbose: bool = False) -> Dict[str, Any]:
    """Verify primary relay redesign layout individually."""
    p_path = Path(layout_path) if layout_path else DEFAULT_PRIMARY_PATH
    res = verify_optical_prescription(
        layout_path=p_path,
        wavelengths=PRIMARY_WAVELENGTHS,
        layout_name="Primary Spectrometer Relay Redesign",
        na=NA_ENTRANCE,
        num_rings=10
    )
    if verbose:
        print_diagnostic_table(res)
    return res


def verify_nir_layout(layout_path: Optional[Path] = None, verbose: bool = False) -> Dict[str, Any]:
    """Verify dedicated NIR layout individually."""
    n_path = Path(layout_path) if layout_path else DEFAULT_NIR_PATH
    res = verify_optical_prescription(
        layout_path=n_path,
        wavelengths=NIR_WAVELENGTHS,
        layout_name="NIR 900-1700nm Spectrometer Layout",
        na=NA_ENTRANCE,
        num_rings=10
    )
    if verbose:
        print_diagnostic_table(res)
    return res


# ==============================================================================
# CLI EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    results = verify_all_layouts(verbose=True)
    sys.exit(0 if results.get("all_passed", False) else 1)
