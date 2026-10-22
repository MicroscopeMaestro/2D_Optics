#!/usr/bin/env python3
"""
scratch/test_reviewer2_verification.py
Independent verification script by Reviewer 2.
Tests CAD export (STEP, DXF), optical geometry, ray tracing, spot size on detector,
and DMD spectral footprint on hybrid_raman_spectrometer.json.
"""

import os
import sys
import math
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.cad.exporter import CADExporter, STEPExporter, STEPExportEngine
from optics_app.cad.dxf_exporter import DXF2DExporter

def verify_all():
    print("=== REVIEWER 2 INDEPENDENT VERIFICATION ===")
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hybrid_raman_spectrometer.json"))
    step_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hybrid_raman_spectrometer.step"))
    dxf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hybrid_raman_spectrometer.dxf"))

    print(f"Loading JSON from: {json_path}")
    assert os.path.exists(json_path), f"File {json_path} does not exist!"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("version") == 2, f"Expected version 2, got {data.get('version')}"
    surfaces = data.get("surfaces", [])
    print(f"Loaded {len(surfaces)} surfaces from JSON.")
    assert len(surfaces) >= 10, f"Expected at least 10 surfaces, got {len(surfaces)}"

    # Load into SurfaceStack
    loaded = SurfaceStack.load_from_json(json_path)
    stack = loaded["stack"] if isinstance(loaded, dict) else loaded
    assert isinstance(stack, SurfaceStack), "Failed to load SurfaceStack"

    print("\n--- 1. Testing CAD STEP Export ---")
    step_res = CADExporter.export_to_step(stack, step_path)
    print(f"STEP Export Result: success={step_res.is_success}, path={step_res.file_path}, components={step_res.num_components_exported}, size={step_res.file_size_bytes} bytes, engine={step_res.engine_used}")
    assert step_res.is_success, f"STEP export failed: {step_res.error_message}"
    assert os.path.exists(step_path), f"STEP file not found at {step_path}"
    assert os.path.getsize(step_path) > 0, "STEP file is empty!"

    with open(step_path, "r", encoding="utf-8", errors="ignore") as f:
        step_text = f.read()

    assert "ISO-10303-21;" in step_text, "STEP header missing ISO-10303-21"
    assert "HEADER;" in step_text, "STEP header missing HEADER"
    assert "DATA;" in step_text, "STEP missing DATA section"
    assert "END-ISO-10303-21;" in step_text, "STEP footer missing END-ISO-10303-21"
    print(f"STEP file verified: {len(step_text.splitlines())} lines, {len(step_text)} bytes.")

    print("\n--- 2. Testing CAD DXF Export ---")
    dxf_res = CADExporter.export_to_dxf_2d(stack, dxf_path, include_dimensions=True, include_rays=True, include_hatches=True, include_table=True, folded=True)
    print(f"DXF Export Result: {dxf_res}")
    assert os.path.exists(dxf_path), f"DXF file not found at {dxf_path}"
    dxf_size = os.path.getsize(dxf_path)
    assert dxf_size > 0, "DXF file is empty!"
    print(f"DXF file verified: {dxf_size} bytes.")

    with open(dxf_path, "r", encoding="utf-8", errors="ignore") as f:
        dxf_text = f.read()
    assert "HEADER" in dxf_text, "DXF missing HEADER section"
    assert "ENTITIES" in dxf_text, "DXF missing ENTITIES section"
    assert "EOF" in dxf_text, "DXF missing EOF tag"

    print("\n--- 3. Testing Optical Ray Tracing & Performance ---")
    system = OpticalSystem(stack)
    wavelengths = data.get("wavelengths", [1.1113, 1.2, 1.3, 1.4, 1.5, 1.6133, 1.66708])
    print(f"Testing wavelengths: {wavelengths} um")

    # Find DMD surface index and detector index
    dmd_idx = -1
    det_idx = len(stack.surfaces) - 1
    for idx, s in enumerate(stack.surfaces):
        if "DMD" in (s.name or "").upper():
            dmd_idx = idx
        if s.surface_type == SurfaceType.DETECTOR:
            det_idx = idx

    print(f"DMD Surface Index: {dmd_idx}, Detector Surface Index: {det_idx}")

    # Check chief ray position on DMD plane across wavelength band
    dmd_positions = []
    det_spot_radii = []

    for wl in wavelengths:
        # Trace chief ray (on-axis from center of slit)
        chief_ray = Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=wl)
        traced_chief = system.trace_ray(chief_ray)
        assert not traced_chief.vignetted, f"Chief ray at {wl} um got vignetted!"
        
        # Check history
        for hist in traced_chief.history:
            if hist["surface_index"] == dmd_idx:
                dmd_positions.append((wl, hist["x"], hist["y"], hist["x_global"], hist["y_global"], hist["z_global"]))

        # Trace ray bundle across NA = 0.10
        rays_bundle = []
        for theta in np.linspace(-math.asin(0.10), math.asin(0.10), 15):
            for phi in [0.0, math.pi/4, math.pi/2, 3*math.pi/4]:
                dx = math.sin(theta) * math.cos(phi)
                dy = math.sin(theta) * math.sin(phi)
                dz = math.cos(theta)
                rays_bundle.append(Ray(pos=(0.0, 0.0, 0.0), dir=(dx, dy, dz), wavelength=wl))
        
        det_pts = []
        vignetted_count = 0
        for r in rays_bundle:
            tr = system.trace_ray(r)
            if not tr.vignetted:
                for hist in tr.history:
                    if hist["surface_index"] == det_idx:
                        det_pts.append((hist["x"], hist["y"]))
            else:
                vignetted_count += 1
        
        print(f"Wavelength {wl:.4f} um: Traced {len(rays_bundle)} rays, Vignetted = {vignetted_count}, Hits at detector = {len(det_pts)}")
        assert vignetted_count == 0, f"Rays vignetted at wavelength {wl} um: {vignetted_count}/{len(rays_bundle)}"

        if det_pts:
            pts_arr = np.array(det_pts)
            # Local distance from origin (0,0) on detector surface
            r_from_center = np.sqrt(pts_arr[:,0]**2 + pts_arr[:,1]**2)
            max_r_center = np.max(r_from_center)
            rms_r = np.sqrt(np.mean(r_from_center**2))
            det_spot_radii.append((wl, max_r_center, rms_r))

    if dmd_positions:
        print("\nDMD Chief Ray Intersections (Local coordinates on DMD):")
        for pos in dmd_positions:
            print(f"  Lambda = {pos[0]:.4f} um: local_x = {pos[1]:.4f} mm, local_y = {pos[2]:.4f} mm")

        # Check dispersion spread between 1.1113 um and 1.6133 um
        wl_1111_x = next((pos[1] for pos in dmd_positions if abs(pos[0] - 1.1113) < 1e-4), None)
        wl_1613_x = next((pos[1] for pos in dmd_positions if abs(pos[0] - 1.6133) < 1e-4), None)
        wl_1667_x = next((pos[1] for pos in dmd_positions if abs(pos[0] - 1.66708) < 1e-4), None)
        
        if wl_1111_x is not None and wl_1613_x is not None:
            span_1613 = abs(wl_1613_x - wl_1111_x)
            print(f"\nDMD Chief Ray Dispersion Span (1.1113 to 1.6133 um): {span_1613:.3f} mm")
            assert 8.00 <= span_1613 <= 8.664, f"DMD span {span_1613:.3f} mm not in [8.00, 8.664] mm"
        if wl_1111_x is not None and wl_1667_x is not None:
            span_1667 = abs(wl_1667_x - wl_1111_x)
            print(f"DMD Chief Ray Dispersion Span (1.1113 to 1.6671 um): {span_1667:.3f} mm")

    print("\nDetector Spot Radii (Distance from detector center (0,0)):")
    for wl, max_r, rms_r in det_spot_radii:
        print(f"  Lambda = {wl:.4f} um: max_r = {max_r:.4f} mm (max diameter = {2*max_r:.4f} mm), rms_r = {rms_r:.4f} mm")
        assert max_r <= 0.50, f"Detector spot radius {max_r:.4f} mm exceeds 0.50 mm (diameter > 1.0 mm)!"

    print("\n--- ALL INDEPENDENT CHECKS PASSED PERFECTLY ---")

if __name__ == "__main__":
    verify_all()
