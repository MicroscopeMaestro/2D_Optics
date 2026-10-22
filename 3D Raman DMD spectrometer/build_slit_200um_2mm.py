"""Generate and validate the 200 um optical slit cartridge with 2.0 mm thickness.

Supports:
1. Standard cartridge orientation (200 um opening along X, 8.0 mm length along Y, with single-sided knife-edge relief)
2. Height-opening cartridge orientation (200 um opening along Y / height, 8.0 mm length along X, with single-sided knife-edge relief)
3. Direct STL exports with fine mesh deflection (<= 0.01 mm)
4. Copies all files to C:\\Users\\Juan\\Desktop\\Spectrometer_Print_Files\\
"""

import os
import sys
import shutil
from pathlib import Path
import json

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD as App
import Part
import MeshPart

OUTPUT_DIR = Path(__file__).resolve().parent
DESKTOP_DIR = Path(r"C:\Users\Juan\Desktop\Spectrometer_Print_Files")

def build_slit_cartridge_2mm(slit_width_um=200, thickness=2.0, web_thickness=0.50, slit_along_height=False, straight_cut=False):
    """
    Builds a drop-in optical slit cartridge:
    - Width: 32.0 mm (X in [-14.0, 18.0])
    - Height: 22.0 mm (Y in [-11.0, 11.0])
    - Thickness: thickness mm (Z in [-thickness/2, thickness/2])
    - Slit: 200 um opening, 8.0 mm length
    - If slit_along_height is True: 200 um opening along Y (height), 8.0 mm along X
    - If slit_along_height is False: 200 um opening along X (width), 8.0 mm along Y
    - Single-sided knife-edge relief expands towards rear face (Z = +thickness/2)
    - Front face (Z = -thickness/2) is completely flat with web_thickness membrane
    - Lateral M4 guide channels (at X = 4.0 mm, depth 1.2 mm, width 3.0 mm)
    - Top M4 extraction pull hole (radius 1.65 mm, depth 6.0 mm at X = 18.0 mm)
    """
    slit_opening_mm = slit_width_um / 1000.0
    slit_length_mm = 8.00
    width = 32.0
    height = 22.0

    box = Part.makeBox(width, height, thickness, App.Vector(-14.0, -11.0, -thickness / 2.0))
    z_web_inner = -thickness / 2.0 + web_thickness
    z_back_face = thickness / 2.0

    if not straight_cut:
        if slit_along_height:
            # 200 um opening along Y, 8.0 mm length along X
            x_inner = slit_length_mm / 2.0 + 0.50  # 4.5 mm
            x_outer = x_inner + 3.0                # 7.5 mm
            y_inner = 0.50                         # 0.5 mm
            y_outer = 3.50                         # 3.5 mm (7.0 mm wide opening)
            slit_cut = Part.makeBox(slit_length_mm, slit_opening_mm, web_thickness + 0.4,
                                   App.Vector(-slit_length_mm / 2.0, -slit_opening_mm / 2.0, -thickness / 2.0 - 0.2))
        else:
            # 200 um opening along X, 8.0 mm length along Y
            x_inner = 0.50                         # 0.5 mm
            x_outer = 3.50                         # 3.5 mm (7.0 mm wide opening)
            y_inner = slit_length_mm / 2.0 + 0.50  # 4.5 mm
            y_outer = y_inner + 3.0                # 7.5 mm
            slit_cut = Part.makeBox(slit_opening_mm, slit_length_mm, web_thickness + 0.4,
                                   App.Vector(-slit_opening_mm / 2.0, -slit_length_mm / 2.0, -thickness / 2.0 - 0.2))

        w_in = Part.makePolygon([
            App.Vector(-x_inner, -y_inner, z_web_inner),
            App.Vector( x_inner, -y_inner, z_web_inner),
            App.Vector( x_inner,  y_inner, z_web_inner),
            App.Vector(-x_inner,  y_inner, z_web_inner),
            App.Vector(-x_inner, -y_inner, z_web_inner)
        ])
        w_out = Part.makePolygon([
            App.Vector(-x_outer, -y_outer, z_back_face),
            App.Vector( x_outer, -y_outer, z_back_face),
            App.Vector( x_outer,  y_outer, z_back_face),
            App.Vector(-x_outer,  y_outer, z_back_face),
            App.Vector(-x_outer, -y_outer, z_back_face)
        ])
        loft_relief = Part.makeLoft([w_in, w_out], True)
        cartridge = box.cut(loft_relief).cut(slit_cut)
    else:
        # Straight cut through 2mm thickness
        if slit_along_height:
            slit_cut = Part.makeBox(slit_length_mm, slit_opening_mm, thickness + 0.4,
                                   App.Vector(-slit_length_mm / 2.0, -slit_opening_mm / 2.0, -thickness / 2.0 - 0.2))
        else:
            slit_cut = Part.makeBox(slit_opening_mm, slit_length_mm, thickness + 0.4,
                                   App.Vector(-slit_opening_mm / 2.0, -slit_length_mm / 2.0, -thickness / 2.0 - 0.2))
        cartridge = box.cut(slit_cut)

    # Lateral guide channels for casing M4 locking set screws (aligned at X = 4.0 mm)
    channel_depth = 1.2
    channel_width = 3.0
    channel_x = 4.0
    right_channel = Part.makeBox(channel_width, channel_depth + 0.1, thickness + 0.4,
                                 App.Vector(channel_x - channel_width / 2.0, height / 2.0 - channel_depth, -thickness / 2.0 - 0.2))
    left_channel = Part.makeBox(channel_width, channel_depth + 0.1, thickness + 0.4,
                                App.Vector(channel_x - channel_width / 2.0, -height / 2.0 - 0.1, -thickness / 2.0 - 0.2))
    cartridge = cartridge.cut(right_channel).cut(left_channel)

    # Top M4 extraction pull-handle blind hole at X = 18.0 mm
    pull_hole = Part.makeCylinder(1.65, 6.0, App.Vector(18.0, 0, 0), App.Vector(-1, 0, 0))
    cartridge = cartridge.cut(pull_hole)

    return cartridge

def export_and_validate(shape, step_filename, stl_filename, slit_width_um=200, slit_along_height=False):
    assert shape.isValid() and shape.isClosed() and len(shape.Solids) == 1, "Must be single closed valid solid"
    bb = shape.BoundBox
    assert abs(bb.ZLength - 2.0) < 1e-6, f"ZLength must be exactly 2.0 mm, got {bb.ZLength}"
    assert abs(bb.XLength - 32.0) < 1e-6, f"XLength must be 32.0 mm, got {bb.XLength}"
    assert abs(bb.YLength - 22.0) < 1e-6, f"YLength must be 22.0 mm, got {bb.YLength}"

    # Verify optical opening via planar section at front membrane
    sec_plane = Part.makePlane(60, 60, App.Vector(-30, -30, -0.75))
    sec = shape.section(sec_plane)
    wires = [Part.Wire(e) for e in Part.sortEdges(sec.Edges)]
    opening_found = False
    expected_x = 8.0 if slit_along_height else (slit_width_um / 1000.0)
    expected_y = (slit_width_um / 1000.0) if slit_along_height else 8.0
    for w in wires:
        wbb = w.BoundBox
        if abs(wbb.XLength - expected_x) < 1e-4 and abs(wbb.YLength - expected_y) < 1e-4:
            opening_found = True
            break
    assert opening_found, f"Optical slit opening ({expected_x:.3f} x {expected_y:.3f} mm) not found in membrane section"

    # Export STEP
    step_path = OUTPUT_DIR / step_filename
    shape.exportStep(str(step_path))
    reread = Part.read(str(step_path))
    assert reread.isValid() and len(reread.Solids) == 1
    assert abs(reread.Volume - shape.Volume) < 1e-4

    # Export STL
    stl_path = OUTPUT_DIR / stl_filename
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.01, AngularDeflection=0.1, Relative=False)
    assert mesh.isSolid(), "Mesh must be solid and watertight"
    mesh.write(str(stl_path))

    # Copy to Desktop print folder
    if DESKTOP_DIR.exists():
        shutil.copy2(step_path, DESKTOP_DIR / step_filename)
        shutil.copy2(stl_path, DESKTOP_DIR / stl_filename)
        print(f"Copied to Desktop: {step_filename}, {stl_filename}")

    return {
        "step_file": step_filename,
        "stl_file": stl_filename,
        "volume_mm3": shape.Volume,
        "bbox": [bb.XLength, bb.YLength, bb.ZLength],
        "slit_aperture_mm": [expected_x, expected_y],
        "mesh_points": mesh.CountPoints,
        "mesh_facets": mesh.CountFacets
    }

def main():
    print("Generating 2.0 mm thick 200 um slit cartridges...")
    results = {}

    # 1. Primary: 200 um height opening (Y = 0.20 mm, X = 8.00 mm), 2 mm thickness with knife-edge relief
    s_height = build_slit_cartridge_2mm(200, thickness=2.0, web_thickness=0.50, slit_along_height=True)
    res_height = export_and_validate(s_height, "external_slit_cartridge_200um_height_2mm.step",
                                     "external_slit_cartridge_200um_height_2mm.stl",
                                     slit_width_um=200, slit_along_height=True)
    results["200um_height_2mm_beveled"] = res_height

    # 2. Standard rotated orientation (X = 0.20 mm, Y = 8.00 mm), 2 mm thickness with knife-edge relief
    s_std = build_slit_cartridge_2mm(200, thickness=2.0, web_thickness=0.50, slit_along_height=False)
    res_std = export_and_validate(s_std, "external_slit_cartridge_200um_2mm.step",
                                  "external_slit_cartridge_200um_2mm.stl",
                                  slit_width_um=200, slit_along_height=False)
    results["200um_standard_2mm_beveled"] = res_std

    # Also update the canonical external_slit_cartridge.step and external_slit_cartridge_200um.step
    s_std.exportStep(str(OUTPUT_DIR / "external_slit_cartridge_200um.step"))
    s_std.exportStep(str(OUTPUT_DIR / "external_slit_cartridge.step"))
    if DESKTOP_DIR.exists():
        shutil.copy2(OUTPUT_DIR / "external_slit_cartridge_200um.step", DESKTOP_DIR / "external_slit_cartridge_200um.step")
        shutil.copy2(OUTPUT_DIR / "external_slit_cartridge.step", DESKTOP_DIR / "external_slit_cartridge.step")

    # 3. Straight through-cut versions (without bevel, for comparison)
    s_height_straight = build_slit_cartridge_2mm(200, thickness=2.0, slit_along_height=True, straight_cut=True)
    res_height_straight = export_and_validate(s_height_straight, "external_slit_cartridge_200um_height_2mm_straight.step",
                                              "external_slit_cartridge_200um_height_2mm_straight.stl",
                                              slit_width_um=200, slit_along_height=True)
    results["200um_height_2mm_straight"] = res_height_straight

    s_std_straight = build_slit_cartridge_2mm(200, thickness=2.0, slit_along_height=False, straight_cut=True)
    res_std_straight = export_and_validate(s_std_straight, "external_slit_cartridge_200um_2mm_straight.step",
                                           "external_slit_cartridge_200um_2mm_straight.stl",
                                           slit_width_um=200, slit_along_height=False)
    results["200um_standard_2mm_straight"] = res_std_straight

    # Save summary report
    report_file = OUTPUT_DIR / "slit_200um_2mm_validation.json"
    report_file.write_text(json.dumps(results, indent=2))
    print(f"\nAll models successfully generated and validated!")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
