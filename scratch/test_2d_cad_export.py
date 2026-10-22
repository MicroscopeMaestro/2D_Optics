"""
Comprehensive Verification Test for 2D CAD Model Export with Dimensions.
Exports the Portable Fiber Spectrometer into DXF and SVG formats for FreeCAD drafting.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import ezdxf
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import export_to_dxf_2d, export_to_svg_2d

def test_2d_cad_export():
    print("=" * 80)
    print("STARTING 2D CAD MODEL EXPORT VERIFICATION (DXF & SVG WITH DIMENSIONS)")
    print("=" * 80)

    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portable_fiber_spectrometer_dmd4500_1mm.json")
    assert os.path.exists(json_path), f"Spectrometer JSON prescription not found: {json_path}"

    print(f"\n1. Loading Optical Prescription: {os.path.basename(json_path)}")
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"] if isinstance(data, dict) else data
    wls = data.get("wavelengths", [0.785, 0.850, 0.900]) if isinstance(data, dict) else [0.785, 0.850, 0.900]
    print(f"   Loaded {len(stack.surfaces)} optical surfaces across {wls} um")

    # Output paths
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dxf_folded_path = os.path.join(out_dir, "portable_fiber_spectrometer_2d_dimensioned.dxf")
    dxf_unfolded_path = os.path.join(out_dir, "portable_fiber_spectrometer_2d_unfolded.dxf")
    svg_path = os.path.join(out_dir, "portable_fiber_spectrometer_2d_dimensioned.svg")

    # 1. Test 2D Folded DXF Export
    print(f"\n2. Exporting 2D Folded DXF CAD Model...")
    res_dxf_f = export_to_dxf_2d(
        stack,
        dxf_folded_path,
        include_dimensions=True,
        include_rays=True,
        include_hatches=True,
        include_table=True,
        folded=True,
        wavelengths=wls
    )
    assert os.path.exists(dxf_folded_path) and os.path.getsize(dxf_folded_path) > 1000
    print(f"   -> Saved: {dxf_folded_path} ({os.path.getsize(dxf_folded_path):,} bytes)")

    # 2. Test 2D Unfolded DXF Export
    print(f"\n3. Exporting 2D Unfolded Bench DXF CAD Model...")
    res_dxf_u = export_to_dxf_2d(
        stack,
        dxf_unfolded_path,
        include_dimensions=True,
        include_rays=True,
        include_hatches=True,
        include_table=True,
        folded=False,
        wavelengths=wls
    )
    assert os.path.exists(dxf_unfolded_path) and os.path.getsize(dxf_unfolded_path) > 1000
    print(f"   -> Saved: {dxf_unfolded_path} ({os.path.getsize(dxf_unfolded_path):,} bytes)")

    # 3. Test 2D SVG Technical Drawing Export
    print(f"\n4. Exporting 2D SVG Technical Drawing...")
    res_svg = export_to_svg_2d(
        stack,
        svg_path,
        include_dimensions=True,
        include_rays=True,
        include_table=True,
        folded=True,
        wavelengths=wls
    )
    assert os.path.exists(svg_path) and os.path.getsize(svg_path) > 1000
    print(f"   -> Saved: {svg_path} ({os.path.getsize(svg_path):,} bytes)")

    # 4. Deep Inspection of Generated DXF Entities
    print(f"\n5. Inspecting DXF Structure with ezdxf (FreeCAD Compatibility Audit)...")
    doc = ezdxf.readfile(dxf_folded_path)
    msp = doc.modelspace()

    # Count layers and entities
    layers = [layer.dxf.name for layer in doc.layers]
    entities = list(msp)
    entity_types = {}
    for e in entities:
        t = e.dxftype()
        entity_types[t] = entity_types.get(t, 0) + 1

    print(f"   Total Layers Defined ({len(layers)}): {', '.join(layers)}")
    print(f"   Total ModelSpace Entities ({len(entities)}):")
    for t, count in entity_types.items():
        print(f"     - {t:<15}: {count:4d}")

    assert "OPTICS_LENSES" in layers, "OPTICS_LENSES layer missing"
    assert "DIMENSIONS_LINEAR" in layers, "DIMENSIONS_LINEAR layer missing"
    assert "OPTICAL_AXIS" in layers, "OPTICAL_AXIS layer missing"
    assert entity_types.get("LWPOLYLINE", 0) > 0, "No lens cross section polylines found"
    assert entity_types.get("LINE", 0) > 0, "No dimension / optical axis lines found"
    assert entity_types.get("TEXT", 0) > 0, "No dimension / table text found"

    print("\n" + "=" * 80)
    print("ALL 2D CAD EXPORT VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_2d_cad_export()
