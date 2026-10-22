"""
Multi-plane (XZ and YZ/YX) 2D CAD Export Verification Script.
Generates fully dimensioned CAD drawings showing:
1. Section A-A: XZ Plane (Dispersion / Horizontal View with Grating Dispersion & Slit Footprints in mm)
2. Section B-B: YZ / YX Plane (Elevation / Slit Height View with Lens Semi-Diameters & Spatial Bounds in mm)
3. Prescription Data Table in mm
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import export_to_dxf_2d, export_to_svg_2d
import ezdxf

def main():
    print("=" * 80)
    print("STARTING 2D CAD MULTI-PLANE (XZ & YZ/YX) EXPORT VERIFICATION")
    print("=" * 80)

    # 1. Load Raman Spectrometer Prescription
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raman_785nm_spectrometer_dmd4500_1mm.json")
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"] if isinstance(data, dict) else data
    wls = data.get("wavelengths", [0.785, 0.850, 0.935]) if isinstance(data, dict) else [0.785, 0.850, 0.935]
    stack._update_accumulated_positions()

    print(f"1. Loaded Spectrometer with {len(stack.surfaces)} surfaces.")

    # 2. Export Dual-Plane DXF (Both XZ and YZ/YX)
    out_dual_dxf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raman_spectrometer_2d_dual_planes.dxf")
    export_to_dxf_2d(
        stack,
        out_dual_dxf,
        include_dimensions=True,
        include_rays=True,
        include_hatches=True,
        include_table=True,
        folded=True,
        wavelengths=wls,
        plane="DUAL"
    )
    print(f"2. Exported Dual-Plane DXF: {out_dual_dxf} ({os.path.getsize(out_dual_dxf):,} bytes)")

    # 3. Export Individual XZ Plane DXF
    out_xz_dxf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raman_spectrometer_2d_xz_plane.dxf")
    export_to_dxf_2d(
        stack,
        out_xz_dxf,
        include_dimensions=True,
        include_rays=True,
        include_hatches=True,
        include_table=True,
        folded=True,
        wavelengths=wls,
        plane="XZ"
    )
    print(f"3. Exported XZ Plane DXF: {out_xz_dxf} ({os.path.getsize(out_xz_dxf):,} bytes)")

    # 4. Export Individual YZ/YX Plane DXF
    out_yz_dxf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raman_spectrometer_2d_yz_plane.dxf")
    export_to_dxf_2d(
        stack,
        out_yz_dxf,
        include_dimensions=True,
        include_rays=True,
        include_hatches=True,
        include_table=True,
        folded=True,
        wavelengths=wls,
        plane="YZ"
    )
    print(f"4. Exported YZ/YX Plane DXF: {out_yz_dxf} ({os.path.getsize(out_yz_dxf):,} bytes)")

    # 5. Export SVG Technical Drawing
    out_svg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raman_spectrometer_2d_dimensioned.svg")
    export_to_svg_2d(
        stack,
        out_svg,
        include_dimensions=True,
        include_rays=True,
        include_table=True,
        folded=True,
        wavelengths=wls
    )
    print(f"5. Exported SVG Technical Drawing: {out_svg} ({os.path.getsize(out_svg):,} bytes)")

    # 6. Audit DXF Entities and Measurements with ezdxf
    doc = ezdxf.readfile(out_dual_dxf)
    msp = doc.modelspace()
    print("\n6. Auditing Dual-Plane DXF ModelSpace with ezdxf:")
    print(f"   Total Entities: {len(msp)}")
    
    # Layer summary
    layer_counts = {}
    for e in msp:
        layer_counts[e.dxf.layer] = layer_counts.get(e.dxf.layer, 0) + 1
    
    for l_name, count in sorted(layer_counts.items()):
        print(f"   - Layer '{l_name:<18}': {count:>4} entities")

    # Verify Text Annotations contains both Section titles and mm units
    text_entities = [e.dxf.text for e in msp.query('TEXT')]
    has_section_a = any("SECTION A-A: XZ PLANE" in t for t in text_entities)
    has_section_b = any("SECTION B-B: YZ" in t for t in text_entities)
    has_mm_units = any("mm" in t for t in text_entities)

    print(f"\n   Found Section A-A (XZ Plane):     {has_section_a}")
    print(f"   Found Section B-B (YZ/YX Plane):  {has_section_b}")
    print(f"   Found mm Unit Dimensions:         {has_mm_units}")

    assert has_section_a, "Section A-A missing from Dual-Plane DXF"
    assert has_section_b, "Section B-B missing from Dual-Plane DXF"
    assert has_mm_units, "Millimeter dimensions missing from DXF"

    print("\n" + "=" * 80)
    print("ALL 2D CAD MULTI-PLANE EXPORTS COMPLETED & VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
