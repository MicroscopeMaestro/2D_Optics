"""Build, validate, and export the Slim-Profile 3-Axis Mount & Light-Tight Enclosure for Thorlabs PDAPC3,
specifically tailored for low-clearance rubber feet (drops only 2.5 mm below the spectrometer bottom).

Spectrometer height: 35.0 mm (X in [-17.5, +17.5 mm])
Slim Mount profile: 40.0 mm along X (X in [-20.0, +20.0 mm]), 54.0 mm along Y
Overhang below casing: ONLY 2.5 mm (cleared by any standard 3.5 - 5.0 mm rubber foot!)

Exports:
1. PDAPC3_Slim_Stage1_Base (STEP + STL)
2. PDAPC3_Slim_Stage2_XY_Slider (STEP + STL)
3. PDAPC3_Slim_Stage3_Z_Carrier (STEP + STL)
4. PDAPC3_Slim_Light_Shield_Hood (STEP + STL)
5. PDAPC3_Slim_Light_Shield_Cable_Cap (STEP + STL)
6. Spectrometer_Leveling_Feet_5mm (STL, set of 4x 5mm feet)
7. PDAPC3_Slim_Full_Assembly (STEP)
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

def build_slim_stage1_base():
    """Slim Base Plate: 40 mm (along X) x 54 mm (along Y) x 6 mm."""
    width_x = 40.0
    height_y = 54.0
    thickness = 6.0

    base = Part.makeBox(width_x, height_y, thickness, App.Vector(-width_x/2, -height_y/2, 0))

    # Central optical aperture bore
    bore = Part.makeCylinder(10.0, thickness + 0.4, App.Vector(0, 0, -0.2), App.Vector(0, 0, 1))
    base = base.cut(bore)

    # 4 Casing mounting holes (M4 counterbored at +-15, +-15)
    for dx in [-15.0, 15.0]:
        for dy in [-15.0, 15.0]:
            th = Part.makeCylinder(2.25, thickness + 0.4, App.Vector(dx, dy, -0.2), App.Vector(0, 0, 1))
            cb = Part.makeCylinder(4.00, 3.6, App.Vector(dx, dy, 2.5), App.Vector(0, 0, 1))
            base = base.cut(th).cut(cb)

    # 4 XY stage clamping holes with captive M4 hex nuts at (+-10.0, +-21.5)
    for dx in [-10.0, 10.0]:
        for dy in [-21.5, 21.5]:
            th = Part.makeCylinder(2.25, thickness + 0.4, App.Vector(dx, dy, -0.2), App.Vector(0, 0, 1))
            nut = make_hex_pocket(4.16, 3.2, App.Vector(dx, dy, 0), App.Vector(0, 0, 1))
            base = base.cut(th).cut(nut)

    # Stepped perimeter tongue for light-trap box (Z from 3.5 to 6.0 mm)
    step_cut = Part.makeBox(width_x + 0.4, height_y + 0.4, 2.6, App.Vector(-width_x/2 - 0.2, -height_y/2 - 0.2, 3.5))
    inner_keep = Part.makeBox(36.0, 50.0, 3.0, App.Vector(-18.0, -25.0, 3.3))
    outer_rim_cut = step_cut.cut(inner_keep)
    base = base.cut(outer_rim_cut)

    # 2 M4 side lock holes for hood at X = 0, Y = +-24.5
    for dy in [-24.5, 24.5]:
        th = Part.makeCylinder(1.65, thickness + 0.4, App.Vector(0, dy, -0.2), App.Vector(0, 0, 1))
        nut = make_hex_pocket(4.16, 3.0, App.Vector(0, dy, 0), App.Vector(0, 0, 1))
        base = base.cut(th).cut(nut)

    return base

def build_slim_stage2_xy_slider():
    """Slim XY Slider: 36 mm x 50 mm plate with +-2.5 mm cross-slots & pinch sleeve."""
    width_x = 36.0
    height_y = 50.0
    flange_th = 5.0
    sleeve_h = 16.0
    total_z = flange_th + sleeve_h

    plate = Part.makeBox(width_x, height_y, flange_th, App.Vector(-width_x/2, -height_y/2, 0))
    sleeve = Part.makeCylinder(16.0, sleeve_h, App.Vector(0, 0, flange_th), App.Vector(0, 0, 1))
    stage2 = plate.fuse(sleeve)

    # Pinch lug on +Y side (along Y to stay slim in X!)
    lug = Part.makeBox(12.0, 10.0, 12.0, App.Vector(-6.0, 14.0, 9.0))
    stage2 = stage2.fuse(lug)

    # Inner precision guide bore (26.0 mm diameter)
    bore = Part.makeCylinder(13.0, total_z + 1.0, App.Vector(0, 0, -0.5), App.Vector(0, 0, 1))
    stage2 = stage2.cut(bore)

    # Split slot along +Y (1.5 mm wide, from Z=7 to top)
    slot = Part.makeBox(1.5, 15.0, 15.0, App.Vector(-0.75, 12.0, 7.0))
    stage2 = stage2.cut(slot)

    # PLA Stress-relief round bore at root of slot
    relief = Part.makeCylinder(1.75, 16.0, App.Vector(-8.0, 13.5, 7.0), App.Vector(1, 0, 0))
    stage2 = stage2.cut(relief)

    # M4 clamping screw hole through pinch lug along X (at Y=19, Z=15)
    th = Part.makeCylinder(2.25, 20.0, App.Vector(-10.0, 19.0, 15.0), App.Vector(1, 0, 0))
    stage2 = stage2.cut(th)

    # Counterbore on +X side
    cb = Part.makeCylinder(4.0, 6.0, App.Vector(3.0, 19.0, 15.0), App.Vector(1, 0, 0))
    stage2 = stage2.cut(cb)

    # Captive M4 hex nut on -X side
    pts = []
    r_circ = 4.16
    for i in range(7):
        ang = i * math.pi / 3.0
        pts.append(App.Vector(-8.0, 19.0 + r_circ * math.cos(ang), 15.0 + r_circ * math.sin(ang)))
    poly = Part.makePolygon(pts)
    nut_pocket = Part.Face(poly).extrude(App.Vector(3.2, 0, 0))
    stage2 = stage2.cut(nut_pocket)

    # 4 XY translation slots (8.0x8.0 mm) at (+-10.0, +-21.5) giving +-2.0 mm XY travel
    for dx in [-10.0, 10.0]:
        for dy in [-21.5, 21.5]:
            slot_xy = Part.makeBox(8.0, 8.0, flange_th + 0.4, App.Vector(dx - 4.0, dy - 4.0, -0.2))
            stage2 = stage2.cut(slot_xy)

    return stage2

def build_slim_stage3_z_carrier():
    """Slim Z-Focus Carrier: 38 mm x 44 mm flange, form-fitting pocket for PDAPC3."""
    barrel_od = 25.4
    barrel_len = 16.0
    flange_x = 38.0
    flange_y = 44.0
    flange_th = 6.0
    total_len = barrel_len + flange_th

    barrel = Part.makeCylinder(barrel_od / 2.0, barrel_len, App.Vector(0, 0, 0), App.Vector(0, 0, 1))
    flange = Part.makeBox(flange_x, flange_y, flange_th, App.Vector(-flange_x/2, -flange_y/2, barrel_len))
    stage3 = barrel.fuse(flange)

    # Central optical aperture tunnel (18.0 mm diameter)
    opt = Part.makeCylinder(9.0, total_len + 1.0, App.Vector(0, 0, -0.5), App.Vector(0, 0, 1))
    stage3 = stage3.cut(opt)

    # Form-fitting alignment pocket for Thorlabs PDAPC3 body (37.5 x 37.5 mm, 2.5 mm deep)
    pocket = Part.makeBox(37.5, 37.5, 2.6, App.Vector(-18.75, -18.75, total_len - 2.5))
    stage3 = stage3.cut(pocket)

    # 4 M4 mounting holes for PDAPC3 at (+-15.0, +-15.0) with captive M4 hex nuts
    for dx in [-15.0, 15.0]:
        for dy in [-15.0, 15.0]:
            th = Part.makeCylinder(2.25, flange_th + 1.0, App.Vector(dx, dy, barrel_len - 0.5), App.Vector(0, 0, 1))
            nut = make_hex_pocket(4.16, 3.0, App.Vector(dx, dy, barrel_len), App.Vector(0, 0, 1))
            stage3 = stage3.cut(th).cut(nut)

    return stage3

def build_slim_light_shield_hood():
    """Slim Light-Tight Hood: 40 mm x 54 mm x 68 mm, drops only 2.5 mm below casing."""
    outer_x = 40.0
    outer_y = 54.0
    length_z = 68.0

    outer = Part.makeBox(outer_x, outer_y, length_z, App.Vector(-outer_x/2, -outer_y/2, 0))
    inner = Part.makeBox(36.0, 50.0, length_z - 3.0, App.Vector(-18.0, -25.0, 0))
    hood = outer.cut(inner)

    # Mating groove at bottom matching Stage 1 tongue
    groove = Part.makeBox(36.3, 50.3, 2.7, App.Vector(-18.15, -25.15, -0.1))
    groove_core = Part.makeBox(33.0, 47.0, 3.0, App.Vector(-16.5, -23.5, -0.2))
    groove_cut = groove.cut(groove_core)
    hood = hood.cut(groove_cut)

    # Cable pass-through slot on rear wall at +Y side (width 14mm in X, depth 16mm in Y)
    slot = Part.makeBox(14.0, 16.0, 6.0, App.Vector(-7.0, 11.0, length_z - 4.0))
    hood = hood.cut(slot)

    # 2 side holes for M4 thumbscrews at X = 0, Y = +-24.5
    for dy in [-24.5, 24.5]:
        th = Part.makeCylinder(2.25, 6.0, App.Vector(0, dy, -0.1), App.Vector(0, 0, 1))
        hood = hood.cut(th)

    # 2 M3 screw tap holes for cable cap at X=0, Y=22.0 and Y=14.0
    for dy in [14.0, 22.0]:
        h = Part.makeCylinder(1.25, 8.0, App.Vector(0, dy, length_z - 5.0), App.Vector(0, 0, 1))
        hood = hood.cut(h)

    return hood

def build_slim_cable_cap():
    """Slim Baffled Cable Clamp Cap."""
    cap_x = 20.0
    cap_y = 16.0
    cap_th = 3.0

    cap_plate = Part.makeBox(cap_x, cap_y, cap_th, App.Vector(-cap_x/2, 10.0, 68.0))
    tongue = Part.makeBox(12.0, 12.0, 3.5, App.Vector(-6.0, 11.0, 64.5))
    cap = cap_plate.fuse(tongue)

    cable_slot = Part.makeBox(8.0, 14.0, 2.5, App.Vector(-4.0, 10.5, 64.5))
    cap = cap.cut(cable_slot)

    for dy in [14.0, 22.0]:
        th = Part.makeCylinder(1.7, 8.0, App.Vector(0, dy, 64.0), App.Vector(0, 0, 1))
        cap = cap.cut(th)

    return cap

def build_leveling_feet():
    """Set of 4x 5mm riser feet for the bottom of the spectrometer."""
    feet = []
    # 20mm round foot, 5mm tall with an M4 countersink hole
    for i, (fx, fy) in enumerate([(-20, -20), (20, -20), (-20, 20), (20, 20)]):
        cyl = Part.makeCylinder(10.0, 5.0, App.Vector(fx, fy, 0), App.Vector(0, 0, 1))
        th = Part.makeCylinder(2.25, 6.0, App.Vector(fx, fy, -0.5), App.Vector(0, 0, 1))
        cb = Part.makeCylinder(4.2, 3.0, App.Vector(fx, fy, 2.5), App.Vector(0, 0, 1))
        foot = cyl.cut(th).cut(cb)
        feet.append(foot)
    return Part.Compound(feet)

def export_model(shape, step_name, stl_name):
    assert shape.isValid() and shape.isClosed(), f"{step_name} must be closed"
    step_path = OUTPUT_DIR / step_name
    stl_path = OUTPUT_DIR / stl_name

    shape.exportStep(str(step_path))
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.01, AngularDeflection=0.1, Relative=False)
    mesh.write(str(stl_path))

    if DESKTOP_DIR.exists():
        shutil.copy2(step_path, DESKTOP_DIR / step_name)
        shutil.copy2(stl_path, DESKTOP_DIR / stl_name)
        print(f"Exported & Copied to Desktop: {step_name}, {stl_name}")

    return {
        "step": step_name,
        "stl": stl_name,
        "bbox": [shape.BoundBox.XLength, shape.BoundBox.YLength, shape.BoundBox.ZLength],
        "facets": mesh.CountFacets
    }

def main():
    print("Building Slim-Profile PDAPC3 XYZ Mount & Enclosure System (2.5mm table drop)...")
    results = {}

    s1 = build_slim_stage1_base()
    results["Slim_Stage1_Base"] = export_model(s1, "PDAPC3_Slim_Stage1_Base.step", "PDAPC3_Slim_Stage1_Base.stl")

    s2 = build_slim_stage2_xy_slider()
    results["Slim_Stage2_XY_Slider"] = export_model(s2, "PDAPC3_Slim_Stage2_XY_Slider.step", "PDAPC3_Slim_Stage2_XY_Slider.stl")

    s3 = build_slim_stage3_z_carrier()
    results["Slim_Stage3_Z_Carrier"] = export_model(s3, "PDAPC3_Slim_Stage3_Z_Carrier.step", "PDAPC3_Slim_Stage3_Z_Carrier.stl")

    hood = build_slim_light_shield_hood()
    results["Slim_Light_Shield_Hood"] = export_model(hood, "PDAPC3_Slim_Light_Shield_Hood.step", "PDAPC3_Slim_Light_Shield_Hood.stl")

    cap = build_slim_cable_cap()
    results["Slim_Light_Shield_Cable_Cap"] = export_model(cap, "PDAPC3_Slim_Light_Shield_Cable_Cap.step", "PDAPC3_Slim_Light_Shield_Cable_Cap.stl")

    feet = build_leveling_feet()
    feet_path = OUTPUT_DIR / "Spectrometer_Leveling_Feet_5mm.stl"
    mesh_feet = MeshPart.meshFromShape(Shape=feet, LinearDeflection=0.02, AngularDeflection=0.1, Relative=False)
    mesh_feet.write(str(feet_path))
    if DESKTOP_DIR.exists():
        shutil.copy2(feet_path, DESKTOP_DIR / "Spectrometer_Leveling_Feet_5mm.stl")
        print("Exported & Copied to Desktop: Spectrometer_Leveling_Feet_5mm.stl")

    # Complete Assembly
    s2_p = s2.copy()
    s2_p.translate(App.Vector(0, 0, 6.0))
    s3_p = s3.copy()
    s3_p.translate(App.Vector(0, 0, 6.0 + 5.0 + 4.0))
    assembly = Part.Compound([s1, s2_p, s3_p, hood, cap])
    ass_path = OUTPUT_DIR / "PDAPC3_Slim_Full_Assembly.step"
    assembly.exportStep(str(ass_path))
    if DESKTOP_DIR.exists():
        shutil.copy2(ass_path, DESKTOP_DIR / "PDAPC3_Slim_Full_Assembly.step")
        print("Exported & Copied to Desktop: PDAPC3_Slim_Full_Assembly.step")

    print("\nSlim system generated and validated successfully!")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
