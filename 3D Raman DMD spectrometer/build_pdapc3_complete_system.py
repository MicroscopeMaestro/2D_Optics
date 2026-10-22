"""Build, validate, and export the complete 3D-printable system for Thorlabs PDAPC3:
1. Stage 1: Base Mount Plate (with casing M4 counterbores, captive nuts, & light-trap tongue)
2. Stage 2: XY Translation Slider (with +-2.75mm cross-slots & stress-relieved pinch clamp)
3. Stage 3: Z-Focus Carrier (with 8mm axial focus travel & form-fitting PDAPC3 pocket)
4. Light-Tight Hood: External stray-light enclosure with labyrinth seam
5. Light-Tight Cable Cap: Baffled light-trap cable clamp
6. Full System Assembly: Complete assembled CAD model

Optimized for PLA 3D printing on Prusa (stress-relief fillets, anti-binding tolerances, zero supports).
Copies all files to C:\\Users\\Juan\\Desktop\\Spectrometer_Print_Files\\
"""

import os
import sys
import math
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

def make_hex_pocket(r_circ, depth, center, normal=App.Vector(0,0,1)):
    pts = []
    for i in range(7):
        ang = i * math.pi / 3.0
        pts.append(App.Vector(center.x + r_circ * math.cos(ang), center.y + r_circ * math.sin(ang), center.z))
    poly = Part.makePolygon(pts)
    return Part.Face(poly).extrude(normal * depth)

def build_stage1_base():
    """Stage 1: Base Plate with casing M4 counterbores & light-trap perimeter rim."""
    width = 64.0
    height = 64.0
    thickness = 6.0

    base = Part.makeBox(width, height, thickness, App.Vector(-width/2, -height/2, 0))

    # Central optical aperture bore
    bore = Part.makeCylinder(10.0, thickness + 0.4, App.Vector(0, 0, -0.2), App.Vector(0, 0, 1))
    base = base.cut(bore)

    # 4 Casing mounting holes (M4 counterbored at +-15, +-15)
    for dx in [-15.0, 15.0]:
        for dy in [-15.0, 15.0]:
            th = Part.makeCylinder(2.25, thickness + 0.4, App.Vector(dx, dy, -0.2), App.Vector(0, 0, 1))
            cb = Part.makeCylinder(4.00, 3.6, App.Vector(dx, dy, 2.5), App.Vector(0, 0, 1))
            base = base.cut(th).cut(cb)

    # 4 XY stage clamping holes with captive M4 hex nuts on rear face (Z=0) at +-20.5, +-20.5
    for dx in [-20.5, 20.5]:
        for dy in [-20.5, 20.5]:
            th = Part.makeCylinder(2.25, thickness + 0.4, App.Vector(dx, dy, -0.2), App.Vector(0, 0, 1))
            nut = make_hex_pocket(4.16, 3.2, App.Vector(dx, dy, 0), App.Vector(0, 0, 1))
            base = base.cut(th).cut(nut)

    # Stepped tongue rim for light-trap box (Z from 3.5 to 6.0 mm, outer step cut)
    step_cut = Part.makeBox(width + 0.4, height + 0.4, 2.6, App.Vector(-width/2 - 0.2, -height/2 - 0.2, 3.5))
    inner_keep = Part.makeBox(58.0, 58.0, 3.0, App.Vector(-29.0, -29.0, 3.3))
    outer_rim_cut = step_cut.cut(inner_keep)
    base = base.cut(outer_rim_cut)

    # 2 M4 threaded holes on sides (X = +-26.0, Y = 0) for light-shield hood lock screws
    for dx in [-26.0, 26.0]:
        th = Part.makeCylinder(1.65, thickness + 0.4, App.Vector(dx, 0, -0.2), App.Vector(0, 0, 1))
        nut = make_hex_pocket(4.16, 3.0, App.Vector(dx, 0, 0), App.Vector(0, 0, 1))
        base = base.cut(th).cut(nut)

    return base

def build_stage2_xy_slider():
    """Stage 2: XY Slider with anti-binding bore & stress-relieved split clamp for PLA."""
    width = 52.0
    height = 52.0
    flange_th = 5.0
    sleeve_h = 16.0
    total_z = flange_th + sleeve_h  # 21.0 mm

    plate = Part.makeBox(width, height, flange_th, App.Vector(-width/2, -height/2, 0))
    sleeve = Part.makeCylinder(16.0, sleeve_h, App.Vector(0, 0, flange_th), App.Vector(0, 0, 1))
    stage2 = plate.fuse(sleeve)

    # Pinch lug on +X side
    lug = Part.makeBox(10.0, 16.0, 12.0, App.Vector(14.0, -8.0, 9.0))
    stage2 = stage2.fuse(lug)

    # Inner precision guide bore (26.0 mm diameter)
    bore = Part.makeCylinder(13.0, total_z + 1.0, App.Vector(0, 0, -0.5), App.Vector(0, 0, 1))
    stage2 = stage2.cut(bore)

    # Split slot along +X (1.5 mm wide, from Z=7 to top)
    slot = Part.makeBox(15.0, 1.5, 15.0, App.Vector(12.0, -0.75, 7.0))
    stage2 = stage2.cut(slot)

    # PLA Stress-relief round bore at root of slot (radius 1.75 mm along Y at X=13.5, Z=7.0)
    relief = Part.makeCylinder(1.75, 16.0, App.Vector(13.5, -8.0, 7.0), App.Vector(0, 1, 0))
    stage2 = stage2.cut(relief)

    # M4 clamping screw hole through pinch lug along Y (at X=19, Z=15)
    th = Part.makeCylinder(2.25, 20.0, App.Vector(19.0, -10.0, 15.0), App.Vector(0, 1, 0))
    stage2 = stage2.cut(th)

    # Counterbore on +Y side for screw head (Y from 3.0 to 9.0)
    cb = Part.makeCylinder(4.0, 6.0, App.Vector(19.0, 3.0, 15.0), App.Vector(0, 1, 0))
    stage2 = stage2.cut(cb)

    # Captive M4 hex nut on -Y side (Y from -8.0 to -4.8)
    pts = []
    r_circ = 4.16
    for i in range(7):
        ang = i * math.pi / 3.0
        pts.append(App.Vector(19.0 + r_circ * math.cos(ang), -8.0, 15.0 + r_circ * math.sin(ang)))
    poly = Part.makePolygon(pts)
    nut_pocket = Part.Face(poly).extrude(App.Vector(0, 3.2, 0))
    stage2 = stage2.cut(nut_pocket)

    # 4 XY translation slots (9.5x9.5 mm) at (+-20.5, +-20.5)
    for dx in [-20.5, 20.5]:
        for dy in [-20.5, 20.5]:
            slot_xy = Part.makeBox(9.5, 9.5, flange_th + 0.4, App.Vector(dx - 4.75, dy - 4.75, -0.2))
            stage2 = stage2.cut(slot_xy)

    return stage2

def build_stage3_z_carrier():
    """Stage 3: Z-Focus Carrier with anti-binding barrel & PDAPC3 pocket."""
    barrel_od = 25.4  # 0.30 mm radial clearance inside 26.0 mm sleeve (PLA anti-binding)
    barrel_len = 16.0
    flange_w = 48.0
    flange_th = 6.0
    total_len = barrel_len + flange_th  # 22.0 mm

    barrel = Part.makeCylinder(barrel_od / 2.0, barrel_len, App.Vector(0, 0, 0), App.Vector(0, 0, 1))
    flange = Part.makeBox(flange_w, flange_w, flange_th, App.Vector(-flange_w/2, -flange_w/2, barrel_len))
    stage3 = barrel.fuse(flange)

    # Central optical aperture tunnel (18.0 mm diameter)
    opt = Part.makeCylinder(9.0, total_len + 1.0, App.Vector(0, 0, -0.5), App.Vector(0, 0, 1))
    stage3 = stage3.cut(opt)

    # Form-fitting alignment pocket for Thorlabs PDAPC3 body (37.8 x 37.8 mm, 2.5 mm deep)
    pocket = Part.makeBox(37.8, 37.8, 2.6, App.Vector(-18.9, -18.9, total_len - 2.5))
    stage3 = stage3.cut(pocket)

    # 4 M4 mounting holes for PDAPC3 at (+-15.0, +-15.0) with captive M4 hex nuts on rear of flange
    for dx in [-15.0, 15.0]:
        for dy in [-15.0, 15.0]:
            th = Part.makeCylinder(2.25, flange_th + 1.0, App.Vector(dx, dy, barrel_len - 0.5), App.Vector(0, 0, 1))
            nut = make_hex_pocket(4.16, 3.0, App.Vector(dx, dy, barrel_len), App.Vector(0, 0, 1))
            stage3 = stage3.cut(th).cut(nut)

    return stage3

def build_light_shield_hood():
    """Light-Tight Enclosure Hood with labyrinth mating groove and cable slot."""
    outer_w = 64.0
    outer_h = 64.0
    length_z = 68.0

    outer = Part.makeBox(outer_w, outer_h, length_z, App.Vector(-outer_w/2, -outer_h/2, 0))
    inner = Part.makeBox(58.0, 58.0, length_z - 3.0, App.Vector(-29.0, -29.0, 0))
    hood = outer.cut(inner)

    # Mating groove at bottom (Z=0 to 2.5 mm) matching Stage 1 tongue
    groove = Part.makeBox(58.3, 58.3, 2.7, App.Vector(-29.15, -29.15, -0.1))
    groove_core = Part.makeBox(55.0, 55.0, 3.0, App.Vector(-27.5, -27.5, -0.2))
    groove_cut = groove.cut(groove_core)
    hood = hood.cut(groove_cut)

    # Cable pass-through slot on rear wall (Z=68) at +Y side (width 14mm in X, depth 16mm in Y)
    slot = Part.makeBox(14.0, 20.0, 6.0, App.Vector(-7.0, 16.0, length_z - 4.0))
    hood = hood.cut(slot)

    # 2 side holes for M4 thumbscrews at X = +-26.0, Y = 0
    th1 = Part.makeCylinder(2.25, 6.0, App.Vector(-26.0, 0, -0.1), App.Vector(0, 0, 1))
    th2 = Part.makeCylinder(2.25, 6.0, App.Vector( 26.0, 0, -0.1), App.Vector(0, 0, 1))
    hood = hood.cut(th1).cut(th2)

    # 2 M3 screw tap holes for cable cap at X=+-11.0, Y=23.0
    for dx in [-11.0, 11.0]:
        h = Part.makeCylinder(1.25, 8.0, App.Vector(dx, 23.0, length_z - 5.0), App.Vector(0, 0, 1))
        hood = hood.cut(h)

    return hood

def build_light_shield_cable_cap():
    """Baffled Light-Trap Cable Clamp Cap."""
    cap_w = 32.0
    cap_len = 16.0
    cap_th = 3.0

    cap_plate = Part.makeBox(cap_w, cap_len, cap_th, App.Vector(-cap_w/2, 15.0, 68.0))
    # Baffle tongue extending down into slot (Z from 65 to 68)
    tongue = Part.makeBox(12.0, 14.0, 3.5, App.Vector(-6.0, 15.0, 64.5))
    cap = cap_plate.fuse(tongue)

    # Labyrinth cable groove in tongue: width 8mm, depth 2.5mm
    cable_slot = Part.makeBox(8.0, 16.0, 2.5, App.Vector(-4.0, 14.5, 64.5))
    cap = cap.cut(cable_slot)

    # 2 M3 clearance holes for fastening screws
    for dx in [-11.0, 11.0]:
        th = Part.makeCylinder(1.7, 8.0, App.Vector(dx, 23.0, 64.0), App.Vector(0, 0, 1))
        cap = cap.cut(th)

    return cap

def export_model(shape, step_name, stl_name):
    assert shape.isValid() and shape.isClosed() and len(shape.Solids) == 1, f"{step_name} must be a single closed solid"
    step_path = OUTPUT_DIR / step_name
    stl_path = OUTPUT_DIR / stl_name

    shape.exportStep(str(step_path))
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.01, AngularDeflection=0.1, Relative=False)
    assert mesh.isSolid(), f"{stl_name} mesh must be solid/watertight"
    mesh.write(str(stl_path))

    if DESKTOP_DIR.exists():
        shutil.copy2(step_path, DESKTOP_DIR / step_name)
        shutil.copy2(stl_path, DESKTOP_DIR / stl_name)
        print(f"Exported & Copied to Desktop: {step_name}, {stl_name}")

    return {
        "step": step_name,
        "stl": stl_name,
        "volume_mm3": shape.Volume,
        "bbox": [shape.BoundBox.XLength, shape.BoundBox.YLength, shape.BoundBox.ZLength],
        "facets": mesh.CountFacets
    }

def main():
    print("Building Complete PDAPC3 XYZ Mount & Light-Tight Enclosure System...")
    results = {}

    s1 = build_stage1_base()
    results["Stage1_Base"] = export_model(s1, "PDAPC3_XYZ_Stage1_Base.step", "PDAPC3_XYZ_Stage1_Base.stl")

    s2 = build_stage2_xy_slider()
    results["Stage2_XY_Slider"] = export_model(s2, "PDAPC3_XYZ_Stage2_XY_Slider.step", "PDAPC3_XYZ_Stage2_XY_Slider.stl")

    s3 = build_stage3_z_carrier()
    results["Stage3_Z_Carrier"] = export_model(s3, "PDAPC3_XYZ_Stage3_Z_Carrier.step", "PDAPC3_XYZ_Stage3_Z_Carrier.stl")

    hood = build_light_shield_hood()
    results["Light_Shield_Hood"] = export_model(hood, "PDAPC3_Light_Shield_Hood.step", "PDAPC3_Light_Shield_Hood.stl")

    cap = build_light_shield_cable_cap()
    results["Light_Shield_Cable_Cap"] = export_model(cap, "PDAPC3_Light_Shield_Cable_Cap.step", "PDAPC3_Light_Shield_Cable_Cap.stl")

    # Full System Assembly (nominal focus)
    s2_placed = s2.copy()
    s2_placed.translate(App.Vector(0, 0, 6.0))
    s3_placed = s3.copy()
    s3_placed.translate(App.Vector(0, 0, 6.0 + 5.0 + 4.0))

    assembly = Part.Compound([s1, s2_placed, s3_placed, hood, cap])
    assembly_path = OUTPUT_DIR / "PDAPC3_Full_System_Assembly.step"
    assembly.exportStep(str(assembly_path))
    if DESKTOP_DIR.exists():
        shutil.copy2(assembly_path, DESKTOP_DIR / "PDAPC3_Full_System_Assembly.step")
        print("Exported & Copied to Desktop: PDAPC3_Full_System_Assembly.step")

    report_path = OUTPUT_DIR / "pdapc3_complete_system_validation.json"
    report_path.write_text(json.dumps(results, indent=2))
    print("\nAll components generated, validated, and exported successfully!")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
