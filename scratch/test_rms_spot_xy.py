"""
Test & Verification Suite for Zemax RMS Spot X & Y Error Estimation & Optimization.
Tests:
1. Mathematical precision of RMS X, RMS Y, GEO X, GEO Y.
2. Identity check: r_RMS = sqrt(RMS_x^2 + RMS_y^2).
3. Astigmatic / Anamorphic test case with known analytical distribution.
4. Optical optimizer targeting RMS Spot X vs RMS Spot Y.
5. calculate_spot_metrics() dictionary interface.
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics, generate_spot_diagram_from_system
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer

def test_rms_spot_xy_math():
    print("=" * 80)
    print("1. TESTING MATHEMATICAL PRECISION OF RMS SPOT X & Y")
    print("=" * 80)

    # Create synthetic elliptical spot (sigma_x = 15.0 um, sigma_y = 45.0 um) with centroid offset (+2.0, -3.0)
    np.random.seed(42)
    n_pts = 100000
    sigma_x_mm = 0.015  # 15 um
    sigma_y_mm = 0.045  # 45 um
    offset_x = 2.0
    offset_y = -3.0

    x_data = np.random.normal(loc=offset_x, scale=sigma_x_mm, size=n_pts)
    y_data = np.random.normal(loc=offset_y, scale=sigma_y_mm, size=n_pts)

    spot = SpotDiagram(x_data, y_data, wavelength_um=0.785, f_number=2.8)
    cx, cy = spot.calculate_centroid()
    rms_r = spot.calculate_rms_radius()
    rms_x = spot.calculate_rms_x()
    rms_y = spot.calculate_rms_y()
    geo_r = spot.calculate_geo_radius()
    geo_x = spot.calculate_geo_x()
    geo_y = spot.calculate_geo_y()

    print(f"   Sample size:           {n_pts:,} rays")
    print(f"   Centroid (X, Y):       ({cx:.4f} mm, {cy:.4f} mm) [Expected: ({offset_x:.1f}, {offset_y:.1f})]")
    print(f"   RMS Spot X:            {rms_x * 1e3:.3f} um [Expected: {sigma_x_mm*1e3:.1f} um]")
    print(f"   RMS Spot Y:            {rms_y * 1e3:.3f} um [Expected: {sigma_y_mm*1e3:.1f} um]")
    print(f"   Total RMS Radius:      {rms_r * 1e3:.3f} um [Expected: {np.sqrt(sigma_x_mm**2 + sigma_y_mm**2)*1e3:.3f} um]")
    print(f"   GEO Spot X:            {geo_x * 1e3:.3f} um")
    print(f"   GEO Spot Y:            {geo_y * 1e3:.3f} um")
    print(f"   GEO Spot Radius:       {geo_r * 1e3:.3f} um")

    # Verify Pythagorean relation: r_RMS^2 = RMS_x^2 + RMS_y^2
    pythagorean_diff = abs(rms_r**2 - (rms_x**2 + rms_y**2))
    print(f"   Pythagorean Error:     {pythagorean_diff:.2e} mm^2")
    assert pythagorean_diff < 1e-12, f"Pythagorean error too high: {pythagorean_diff}"
    assert abs(cx - offset_x) < 1e-2, "Centroid X mismatch"
    assert abs(cy - offset_y) < 1e-2, "Centroid Y mismatch"
    assert abs(rms_x - sigma_x_mm) < 1e-3, "RMS X mismatch"
    assert abs(rms_y - sigma_y_mm) < 1e-3, "RMS Y mismatch"

    # Test calculate_spot_metrics
    metrics = calculate_spot_metrics(x_data, y_data, wavelength_um=0.785, f_number=2.8)
    assert "rms_x" in metrics and "rms_y" in metrics
    assert "geo_x" in metrics and "geo_y" in metrics
    print("   -> calculate_spot_metrics() returns all Zemax RMS and GEO keys: PASS")
    print("   [+] Mathematical Verification PASSED!")

def test_spectrometer_rms_spot_xy():
    print("\n" + "=" * 80)
    print("2. TESTING RMS SPOT X & Y ON PORTABLE SPECTROMETER")
    print("=" * 80)

    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raman_785nm_spectrometer_dmd4500_1mm.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portable_fiber_spectrometer_dmd4500_1mm.json")
    
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"] if isinstance(data, dict) else data
    stack.primary_wavelength = 0.8500
    stack._update_accumulated_positions()
    opt_sys = OpticalSystem(stack)

    # Trace multi-ray bundle at 850 nm
    sd = generate_spot_diagram_from_system(
        system=opt_sys,
        num_rays=200,
        wavelength_um=0.850,
        target_surface_index=-1
    )

    cx, cy = sd.calculate_centroid()
    rms_r_um = sd.calculate_rms_radius() * 1e3
    rms_x_um = sd.calculate_rms_x() * 1e3
    rms_y_um = sd.calculate_rms_y() * 1e3
    geo_r_um = sd.calculate_geo_radius() * 1e3
    geo_x_um = sd.calculate_geo_x() * 1e3
    geo_y_um = sd.calculate_geo_y() * 1e3
    airy_um = sd.calculate_airy_disk_radius()

    print(f"   Detector Plane Analysis (850 nm):")
    print(f"     Total Ray Hits:        {len(sd.x)}")
    print(f"     Centroid (X, Y):       ({cx:+.4f} mm, {cy:+.4f} mm)")
    print(f"     RMS Spot X (Spectral): {rms_x_um:6.2f} um")
    print(f"     RMS Spot Y (Spatial):  {rms_y_um:6.2f} um")
    print(f"     Total RMS Radius:      {rms_r_um:6.2f} um")
    print(f"     GEO Spot X:            {geo_x_um:6.2f} um")
    print(f"     GEO Spot Y:            {geo_y_um:6.2f} um")
    print(f"     GEO 100% Spot Radius:  {geo_r_um:6.2f} um")
    print(f"     Airy Disk Radius:      {airy_um:6.2f} um")
    print(f"     Diffraction Limited?   {sd.is_diffraction_limited()}")

    assert len(sd.x) > 0, "No rays traced to detector"
    assert rms_r_um > 0 and rms_x_um > 0 and rms_y_um > 0, "RMS metrics must be positive"
    assert math.isclose(rms_r_um**2, rms_x_um**2 + rms_y_um**2, rel_tol=1e-5), "RMS Pythagorean identity failed"
    print("   [+] Spectrometer Spot Diagnostics PASSED!")

def test_optimization_targeting_rms_x_and_y():
    print("\n" + "=" * 80)
    print("3. TESTING OPTIMIZATION TARGETING RMS SPOT X VS RMS SPOT Y")
    print("=" * 80)

    # Build an astigmatic test doublet
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=20.0, glass="AIR"))
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=25.0, thickness=4.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-35.0, thickness=30.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=5.0, thickness_max=60.0))
    stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=5.0))
    stack._update_accumulated_positions()
    opt_sys = OpticalSystem(stack)

    # 1. Optimize for RMS Spot X
    opt_x = OpticalOptimizer(
        system=opt_sys,
        wavelengths=0.5876,
        target="RMS Spot X (Spectral Focus)",
        max_iter=30
    )
    res_x = opt_x.optimize(variable_indices={"thickness": [2], "radius": []})
    print(f"   RMS Spot X Target Optimization:")
    print(f"     Initial Merit: {res_x.initial_rms:.3f} mm")
    print(f"     Final Merit:   {res_x.final_rms:.3f} mm")
    print(f"     Reduction:     {res_x.rms_reduction_pct:.1f}%")
    print(f"     Opt Thickness: {res_x.optimized_thicknesses[2]:.2f} mm")
    assert res_x.success, "RMS Spot X optimization failed"

    # 2. Optimize for RMS Spot Y
    opt_y = OpticalOptimizer(
        system=opt_sys,
        wavelengths=0.5876,
        target="RMS Spot Y (Spatial Focus)",
        max_iter=30
    )
    res_y = opt_y.optimize(variable_indices={"thickness": [2], "radius": []})
    print(f"\n   RMS Spot Y Target Optimization:")
    print(f"     Initial Merit: {res_y.initial_rms:.3f} mm")
    print(f"     Final Merit:   {res_y.final_rms:.3f} mm")
    print(f"     Reduction:     {res_y.rms_reduction_pct:.1f}%")
    print(f"     Opt Thickness: {res_y.optimized_thicknesses[2]:.2f} mm")
    assert res_y.success, "RMS Spot Y optimization failed"

    print("\n" + "=" * 80)
    print("ALL RMS SPOT X & Y TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_rms_spot_xy_math()
    test_spectrometer_rms_spot_xy()
    test_optimization_targeting_rms_x_and_y()
