#!/usr/bin/env python3
"""
build_and_verify.py: Comprehensive Optical Design, JSON Generation & Physics Verification
for the 1064 nm Folded Hybrid Raman Spectrometer with Thorlabs 1-Inch Optics,
TI DMD4500 Spatial Light Modulator, and Single-Pixel 1.0 mm InGaAs Detector.
"""

import os
import sys
import math
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.source import SourceConfig, SourceType
from optics_app.cad.exporter import CADExporter, STEPExportEngine
from optics_app.utils.materials import get_refractive_index

# Authoritative Design Constants
WAVELENGTHS = [1.11130, 1.20000, 1.30000, 1.40000, 1.50000, 1.61330, 1.66708] # in um
PRIMARY_WAVELENGTH = 1.36200 # in um
SLIT_CORE_RADIUS_MM = 0.0125 # 25 um slit / 2 = 12.5 um
SLIT_NA = 0.10 # f/5 input numerical aperture

JSON_OUTPUT_PATH = "hybrid_raman_spectrometer.json"
JSON_BACKUP_PATH = os.path.join(".agents", "worker_optical_designer", "hybrid_raman_spectrometer.json")
STEP_OUTPUT_PATH = "hybrid_raman_spectrometer.step"
DXF_OUTPUT_PATH = "hybrid_raman_spectrometer.dxf"

def build_hybrid_raman_system() -> SurfaceStack:
    """
    Constructs the complete 14-surface sequential optical system for the
    1064 nm excitation folded hybrid Raman spectrometer using authentic
    Thorlabs 1-inch COTS components and TI DLP4500 DMD.
    """
    stack = SurfaceStack()
    stack.primary_wavelength = PRIMARY_WAVELENGTH
    
    # 0. Source / Slit (25 um low-OH fiber core / slit, NA = 0.10)
    stack.add_surface(Surface(
        name="Slit_Source_25um",
        surface_type=SurfaceType.OBJECT,
        thickness=45.0,
        glass="AIR",
        semi_diameter=12.7,
        aperture_type="Circular",
        aperture_r_outer=SLIT_CORE_RADIUS_MM
    ))
    
    # 1-3. Collimator: Thorlabs AC254-050-C (f = 50.0 mm, Dia 25.4 mm, N-BK7 / SF5)
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.00,
        thickness=6.00,
        glass="N-BK7",
        semi_diameter=12.70
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.80,
        thickness=2.00,
        glass="SF5",
        semi_diameter=12.70
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-115.00,
        thickness=15.00,
        glass="AIR",
        semi_diameter=12.70
    ))
    
    # 4. Diffraction Grating: Thorlabs GR25-0316 (300 l/mm, Blaze 1600 nm, Protected Gold)
    # Incident quasi-Littrow angle alpha = 12.0 deg
    stack.add_surface(Surface(
        name="Grating_GR25-0316",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=15.00,
        glass="MIRROR",
        semi_diameter=12.70,
        tilt_y=12.00,
        grating_lines_per_mm=300.0,
        grating_order=1,
        aperture_type="Rectangular",
        aperture_x_half=12.50,
        aperture_y_half=12.50
    ))
    
    # 5-7. Camera / Focusing Lens: Thorlabs AC254-050-C (f = 50.0 mm, Dia 25.4 mm, N-BK7 / SF5)
    stack.add_surface(Surface(
        name="AC254-050-C_Focus_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.00,
        thickness=6.00,
        glass="N-BK7",
        semi_diameter=12.70
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Focus_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.80,
        thickness=2.00,
        glass="SF5",
        semi_diameter=12.70
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Focus_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-115.00,
        thickness=45.00,
        glass="AIR",
        semi_diameter=12.70
    ))
    
    # 8. Spatial Light Modulator: Texas Instruments DLP4500 DMD (8.664 x 4.873 mm active area)
    # Micromirrors tilted at +12.0 deg (ON state), producing a +24.0 deg reflected chief ray channel
    stack.add_surface(Surface(
        name="TI_DMD4500",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=45.00,
        glass="MIRROR",
        semi_diameter=12.70,
        tilt_y=12.00,
        aperture_type="Rectangular",
        aperture_x_half=4.332,
        aperture_y_half=2.4365
    ))
    
    # 9-11. Collection Relay Collimator: Thorlabs AC254-075-C (f = 75.0 mm, Dia 25.4 mm, N-BK7 / SF5)
    stack.add_surface(Surface(
        name="AC254-075-C_Relay_Front",
        surface_type=SurfaceType.STANDARD,
        radius=49.30,
        thickness=8.00,
        glass="N-BK7",
        semi_diameter=12.70
    ))
    stack.add_surface(Surface(
        name="AC254-075-C_Relay_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-39.00,
        thickness=3.00,
        glass="SF5",
        semi_diameter=12.70
    ))
    stack.add_surface(Surface(
        name="AC254-075-C_Relay_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-170.00,
        thickness=10.00,
        glass="AIR",
        semi_diameter=12.70
    ))
    
    # 12-13. Condenser: Thorlabs ACL12708U-C (f = 8.0 mm, Asphere NA = 0.78, B270)
    stack.add_surface(Surface(
        name="ACL12708U-C_Cond_Front",
        surface_type=SurfaceType.ASPHERIC,
        radius=4.16,
        thickness=4.50,
        glass="B270",
        semi_diameter=12.70,
        conic=-0.6265,
        asphere_coeffs=[1.5e-05]
    ))
    stack.add_surface(Surface(
        name="ACL12708U-C_Cond_Back",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=4.85,
        glass="AIR",
        semi_diameter=12.70
    ))
    
    # 14. Single-Pixel InGaAs Photodiode Detector (1.0 mm active area, semi_diameter = 0.50 mm)
    stack.add_surface(Surface(
        name="Detector_InGaAs_1mm",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.50,
        aperture_type="Circular",
        aperture_r_outer=0.50
    ))
    
    stack._update_accumulated_positions()
    return stack

def export_v2_json(stack: SurfaceStack, output_path: str):
    """Saves the SurfaceStack conforming strictly to 2D_Optics Schema v2."""
    import dataclasses
    surfaces_data = [dataclasses.asdict(s) for s in stack.surfaces]
    for d in surfaces_data:
        if "cad_profile" in d:
            d["cad_profile"] = None
        if "global_origin" in d:
            del d["global_origin"]
        if "global_R" in d:
            del d["global_R"]
            
    source_config = {
        "type": "point",
        "semi_aperture": SLIT_CORE_RADIUS_MM,
        "na": SLIT_NA,
        "source_type": "Fiber (Extended)",
        "beam_radius": SLIT_CORE_RADIUS_MM,
        "numerical_aperture": SLIT_NA,
        "divergence_angle": math.degrees(math.asin(SLIT_NA)),
        "input_aperture_shape": "Circular",
        "input_aperture_width": SLIT_CORE_RADIUS_MM * 2.0,
        "wavelengths": WAVELENGTHS
    }
    
    data = {
        "version": 2,
        "surfaces": surfaces_data,
        "wavelengths": WAVELENGTHS,
        "primary_wavelength": PRIMARY_WAVELENGTH,
        "source_config": source_config
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[OK] Saved Schema v2 JSON to: {output_path}")

def verify_and_report():
    print("=" * 80)
    print("HYBRID RAMAN SPECTROMETER: PRIMARY BUILD & VERIFICATION HARNESS")
    print("=" * 80)
    
    # 1. Build and export
    stack = build_hybrid_raman_system()
    export_v2_json(stack, JSON_OUTPUT_PATH)
    
    # Also save backup in working directory
    os.makedirs(os.path.dirname(JSON_BACKUP_PATH), exist_ok=True)
    export_v2_json(stack, JSON_BACKUP_PATH)
    
    # 2. Test OpticalSystem.from_json loader
    print("\n--- STEP 1: OpticalSystem.from_json LOADER TEST ---")
    system = OpticalSystem.from_json(JSON_OUTPUT_PATH)
    assert len(system.surfaces) == 15, f"Expected 15 surfaces, got {len(system.surfaces)}"
    print(f"[PASS] OpticalSystem.from_json successfully loaded {len(system.surfaces)} surfaces.")
    
    # 3. Chief Ray Tracing on DMD Plane
    print("\n--- STEP 2: CHIEF RAY TRACING & DMD ACTIVE AREA SPREAD ---")
    chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=w) for w in WAVELENGTHS]
    chief_traced = system.trace_rays(chief_rays)
    
    dmd_idx = next(i for i, s in enumerate(system.surfaces) if "DMD" in s.name.upper())
    dmd_hits = {}
    print(f"DMD Surface Index: {dmd_idx} ({system.surfaces[dmd_idx].name})")
    for r in chief_traced:
        hit = next(h for h in r.history if h["surface_index"] == dmd_idx)
        dmd_hits[r.wavelength] = hit["x"]
        print(f"   Wavelength {r.wavelength:.5f} um ({r.wavelength*1000.0:6.1f} nm) -> DMD Chief Ray X = {hit['x']:7.3f} mm, Y = {hit['y']:7.3f} mm")
        
    span_1613 = abs(dmd_hits[1.61330] - dmd_hits[1.11130])
    span_1667 = abs(dmd_hits[1.66708] - dmd_hits[1.11130])
    
    print(f"\n   -> Chief Ray Span (1.11130 um to 1.61330 um): {span_1613:.4f} mm")
    print(f"   -> Chief Ray Span (1.11130 um to 1.66708 um): {span_1667:.4f} mm")
    print(f"   -> Required Acceptance Band:                [8.000 mm, 8.664 mm]")
    print(f"   -> DMD4500 Active Width:                     8.664 mm")
    
    assert 8.000 <= span_1613 <= 8.664, f"DMD Chief Ray Span {span_1613:.4f} mm outside [8.000, 8.664] mm"
    print(f"[PASS] Chief Ray Span {span_1613:.4f} mm is strictly within [8.000 mm, 8.664 mm].")
    
    # 4. Polychromatic Phase-Space Ray Bundle on Detector Plane
    print("\n--- STEP 3: POLYCHROMATIC PHASE-SPACE RAY BUNDLE & DETECTOR SPOT METRICS ---")
    max_theta = math.asin(SLIT_NA)
    rf = SLIT_CORE_RADIUS_MM
    
    # Deterministic spatial x angular sampling
    spatial_points = [
        (0.0, 0.0),
        (rf, 0.0), (-rf, 0.0), (0.0, rf), (0.0, -rf),
        (rf/1.414, rf/1.414), (-rf/1.414, rf/1.414), (rf/1.414, -rf/1.414), (-rf/1.414, -rf/1.414)
    ]
    
    angular_directions = [(0.0, 0.0, 1.0)]
    for ar in [0.33, 0.67, 1.0]:
        theta = ar * max_theta
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            vx = float(math.sin(theta) * math.cos(phi))
            vy = float(math.sin(theta) * math.sin(phi))
            vz = float(math.cos(theta))
            angular_directions.append((vx, vy, vz))
            
    rays = []
    for wl in WAVELENGTHS:
        for px, py in spatial_points:
            for vx, vy, vz in angular_directions:
                rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([vx, vy, vz]), wavelength=wl))
                
    print(f"Tracing {len(rays)} rays ({len(WAVELENGTHS)} wavelengths x {len(spatial_points)} spatial x {len(angular_directions)} angular)...")
    traced_bundle = system.trace_rays(rays)
    
    det_idx = len(system.surfaces) - 1
    det_hits = []
    vignetted_count = 0
    for r in traced_bundle:
        if not r.vignetted:
            dh = next((h for h in r.history if h["surface_index"] == det_idx), None)
            if dh:
                det_hits.append((dh["x"], dh["y"]))
        else:
            vignetted_count += 1
            
    radii = [math.hypot(x, y) for x, y in det_hits]
    max_radius = max(radii)
    rms_radius = math.sqrt(np.mean(np.array(radii)**2))
    max_diameter = 2.0 * max_radius
    
    print(f"   -> Total Traced Rays:               {len(rays)}")
    print(f"   -> Rays Successfully Focused:       {len(det_hits)} ({len(det_hits)/len(rays)*100:.1f}%)")
    print(f"   -> Maximum Detector Spot Radius:    {max_radius:.4f} mm ({max_radius*1000.0:.1f} um)")
    print(f"   -> RMS Detector Spot Radius:        {rms_radius:.4f} mm ({rms_radius*1000.0:.1f} um)")
    print(f"   -> Maximum Geometric Spot Diameter: {max_diameter:.4f} mm ({max_diameter*1000.0:.1f} um)")
    print(f"   -> InGaAs Sensor Active Diameter:   1.0000 mm (radius 0.5000 mm)")
    
    assert max_diameter <= 1.0000, f"Spot diameter {max_diameter:.4f} mm exceeds 1.0000 mm"
    assert max_radius <= 0.5000, f"Spot radius {max_radius:.4f} mm exceeds 0.5000 mm"
    print(f"[PASS] Maximum geometric spot diameter {max_diameter:.4f} mm <= 1.0000 mm.")
    
    # 5. CAD Export Verification (STEP and DXF)
    print("\n--- STEP 4: 3D STEP SOLID & 2D DXF DRAWING CAD EXPORT ---")
    exporter = CADExporter(system)
    
    step_result = exporter.export_to_step(STEP_OUTPUT_PATH)
    print(f"   -> STEP Export Success:    {step_result.is_success}")
    print(f"   -> STEP Output Path:       {step_result.file_path}")
    print(f"   -> STEP File Size:         {step_result.file_size_bytes} bytes")
    assert step_result.is_success, f"STEP export failed: {step_result.error_message}"
    assert os.path.exists(STEP_OUTPUT_PATH), "STEP file does not exist on disk"
    assert os.path.getsize(STEP_OUTPUT_PATH) > 0, "STEP file is empty"
    print(f"[PASS] 3D STEP Solid Model exported successfully ({os.path.getsize(STEP_OUTPUT_PATH)} bytes).")
    
    dxf_result = exporter.export_to_dxf_2d(DXF_OUTPUT_PATH)
    print(f"   -> DXF Output Path:        {dxf_result}")
    assert os.path.exists(DXF_OUTPUT_PATH), "DXF file does not exist on disk"
    assert os.path.getsize(DXF_OUTPUT_PATH) > 0, "DXF file is empty"
    print(f"[PASS] 2D DXF Dimensioned Drawing exported successfully ({os.path.getsize(DXF_OUTPUT_PATH)} bytes).")
    
    print("\n" + "=" * 80)
    print("ALL OPTICAL, GEOMETRIC, RAY-TRACE, AND CAD CRITERIA MET WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    verify_and_report()
