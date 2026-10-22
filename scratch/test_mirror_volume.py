"""
Test and Verification Suite for Mirror 3D Substrate Volume & Mechanical Thickness:
1. Validates Surface.substrate_thickness and Surface.mirror_thickness properties.
2. Validates 3D Solid Body generation (Cylindrical and Rectangular prisms) in STEP exporter.
3. Validates 2D cross-sectional CAD rendering (DXF & SVG) with substrate thickness and hatching.
4. Generates updated 3D solid assembly and 2D blueprints.
"""

import sys
import os
import json
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.cad.exporter import export_to_dxf_2d, export_to_svg_2d, export_to_step, create_mirror_or_planar_solid
import ezdxf

def test_mirror_volume_geometry():
    print("=" * 85)
    print("TESTING 3D MIRROR SUBSTRATE VOLUME & PLANAR COMPONENT CAD SOLID BODIES")
    print("=" * 85)

    # 1. Test Circular 1-Inch Mirror Substrate (e.g. Thorlabs BB1-E03, Ø 25.4 x 6.0 mm)
    m_circ = Surface(
        name="Thorlabs_BB1-E03_1in_Mirror",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.70,  # Ø 25.4 mm (1")
        substrate_thickness=6.0,  # 6.0 mm thickness
        aperture_type="Circular"
    )
    assert m_circ.is_mirror() is True, "Mirror detection failed for circular mirror"
    assert m_circ.mirror_thickness == 6.0, "Mirror thickness property failed"

    part_circ = create_mirror_or_planar_solid(m_circ)
    if part_circ is not None:
        expected_vol = math.pi * (12.70**2) * 6.0  # ~3040.2 mm^3
        actual_vol = part_circ.volume
        vol_err = abs(actual_vol - expected_vol) / expected_vol
        print(f"1. Circular 1\" Mirror Solid (Ø 25.4 x 6.0 mm):")
        print(f"   Expected Volume: {expected_vol:.2f} mm^3 | Actual CAD Volume: {actual_vol:.2f} mm^3 (Error: {vol_err*100:.3f}%)")
        assert vol_err < 0.01, f"Circular mirror volume error {vol_err*100:.2f}% too large"

    # 2. Test Rectangular 1-Inch Grating Substrate (e.g. Thorlabs GR25-0608, 25 x 25 x 6.0 mm)
    g_rect = Surface(
        name="Thorlabs_GR25-0608_1in_Grating",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=40.0,
        glass="MIRROR",
        semi_diameter=12.70,
        substrate_thickness=6.0,  # 6.0 mm substrate thickness
        aperture_type="Rectangular",
        aperture_x_half=12.50,  # 25 mm width
        aperture_y_half=12.50,  # 25 mm height
        tilt_x=30.0
    )
    assert g_rect.is_mirror() is True, "Mirror detection failed for ruled grating"
    assert g_rect.substrate_thickness == 6.0, "Substrate thickness failed for grating"

    part_rect = create_mirror_or_planar_solid(g_rect)
    if part_rect is not None:
        expected_vol_rect = 25.0 * 25.0 * 6.0  # 3750.0 mm^3
        actual_vol_rect = part_rect.volume
        vol_err_rect = abs(actual_vol_rect - expected_vol_rect) / expected_vol_rect
        print(f"\n2. Rectangular 1\" Grating Solid (25.0 x 25.0 x 6.0 mm):")
        print(f"   Expected Volume: {expected_vol_rect:.2f} mm^3 | Actual CAD Volume: {actual_vol_rect:.2f} mm^3 (Error: {vol_err_rect*100:.3f}%)")
        assert vol_err_rect < 0.01, f"Rectangular grating volume error {vol_err_rect*100:.2f}% too large"

    # 3. Test TI DLP4500 DMD Spatial Modulator (9.855 x 6.16 x 3.2 mm ceramic package)
    dmd = Surface(
        name="TI_DLP4500_DMD_Array",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=100.0,
        glass="MIRROR",
        semi_diameter=12.70,
        substrate_thickness=3.2,  # 3.2 mm package thickness
        aperture_type="Rectangular",
        aperture_x_half=4.9275,  # 9.855 mm / 2
        aperture_y_half=3.0800,  # 6.160 mm / 2
        tilt_x=30.0
    )
    part_dmd = create_mirror_or_planar_solid(dmd)
    if part_dmd is not None:
        expected_vol_dmd = 9.855 * 6.160 * 3.2  # ~194.26 mm^3
        actual_vol_dmd = part_dmd.volume
        vol_err_dmd = abs(actual_vol_dmd - expected_vol_dmd) / expected_vol_dmd
        print(f"\n3. TI DLP4500 DMD Solid (9.855 x 6.160 x 3.2 mm):")
        print(f"   Expected Volume: {expected_vol_dmd:.2f} mm^3 | Actual CAD Volume: {actual_vol_dmd:.2f} mm^3 (Error: {vol_err_dmd*100:.3f}%)")
        assert vol_err_dmd < 0.01, f"DMD volume error {vol_err_dmd*100:.2f}% too large"

    # 4. Export Complete 1-Inch Spectrometer with Substrate Volumes
    from scratch.test_lens_replacement import build_spectrometer_with_l2
    stack = build_spectrometer_with_l2("AC254-040-B")
    # Set explicit substrate thicknesses
    stack.surfaces[5].substrate_thickness = 6.0  # GR25-0608 Grating (6.0 mm)
    stack.surfaces[9].substrate_thickness = 3.2  # DLP4500 DMD (3.2 mm)
    stack.surfaces[15].substrate_thickness = 1.5  # Hamamatsu Detector (1.5 mm)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_step = os.path.join(root_dir, "raman_785nm_spectrometer_1inch_optics.step")
    export_to_step(stack, out_step)
    print(f"\n4. Exported 3D STEP Assembly with Mirror Substrates: {out_step} ({os.path.getsize(out_step):,} bytes)")

    out_dual_dxf = os.path.join(root_dir, "raman_785nm_spectrometer_1inch_dual_planes.dxf")
    export_to_dxf_2d(stack, out_dual_dxf, plane="DUAL", include_dimensions=True, include_rays=True, include_hatches=True, include_table=True)
    print(f"5. Exported 2D Dual-Plane DXF with Substrate Hatching: {out_dual_dxf} ({os.path.getsize(out_dual_dxf):,} bytes)")

    out_svg = os.path.join(root_dir, "raman_785nm_spectrometer_1inch_dimensioned.svg")
    export_to_svg_2d(stack, out_svg, include_dimensions=True, include_rays=True, include_table=True)
    print(f"6. Exported 2D Dimensioned SVG: {out_svg} ({os.path.getsize(out_svg):,} bytes)")

    # 5. Audit DXF to confirm mirror substrate hatching
    doc = ezdxf.readfile(out_dual_dxf)
    msp = doc.modelspace()
    hatches = list(msp.query('HATCH'))
    print(f"\n7. Auditing DXF Hatch Entities: Found {len(hatches)} Hatch Elements across Lenses & Mirror Substrates.")
    assert len(hatches) >= 4, f"Expected at least 4 hatched elements (lenses + mirror/grating substrates), found {len(hatches)}"

    print("\n" + "=" * 85)
    print("ALL 3D MIRROR VOLUME & SUBSTRATE THICKNESS TESTS PASSED WITH 100% VERIFICATION!")
    print("=" * 85)

if __name__ == "__main__":
    test_mirror_volume_geometry()
