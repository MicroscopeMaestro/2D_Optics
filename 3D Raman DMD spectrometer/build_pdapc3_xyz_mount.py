"""Build, validate, and export the 3D-printable 3-Axis (XYZ) Mount for Thorlabs PDAPC3.

Architecture (Option 2 - Decoupled 3-Axis Translation):
1. Stage 1 (Base Plate):
   - Mates flush to the spectrometer casing's tilted exit face.
   - 4x M4 counterbored holes (spacing 30x30 mm, dx,dy=+-15 mm).
   - 4x M4 captive hex nut pockets on the rear face for the XY clamping screws (spacing 41x41 mm, dx,dy=+-20.5 mm).
   - Central light tunnel: 20 mm diameter.

2. Stage 2 (XY Translation Slider):
   - Mates against Stage 1.
   - 4x 9.5x9.5 mm translation slots at dx,dy=+-20.5 mm (+-2.75 mm XY travel).
   - Central Z-guide sleeve: 26 mm ID, 32 mm OD, 16 mm height.
   - Integrated pinch-clamp collar with split slot (1.5 mm) and M4 clamping screw lug.

3. Stage 3 (Z-Focus Carrier & PDAPC3 Mount):
   - 25.6 mm OD barrel (0.2 mm radial sliding fit inside Stage 2 sleeve).
   - 8 mm axial focus travel range (locked by Stage 2 pinch collar).
   - Front mounting flange with 37.8x37.8 mm form-fitting pocket for Thorlabs PDAPC3 body.
   - 4x M4 mounting holes (30x30 mm cage spacing) with captive M4 hex nut pockets.
   - 18 mm clear aperture optical tunnel.

Exports STEP + STL for all stages and complete assembly, and copies to:
C:\\Users\\Juan\\Desktop\\Spectrometer_Print_Files\\
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
    """Creates a regular hexagonal prism for standard M4 nut pockets."""
    pts = []
    for i in range(7):
        ang = i * math.pi / 3.0
        pts.append(App.Vector(center.x + r_circ * math.cos(ang), center.y + r_circ * math.sin(ang), center.z))
    poly = Part.makePolygon(pts)
    face = Part.Face(poly)
    return face.extrude(normal * depth)

def build_stage1_base():
    """Stage 1: Base Mount Plate to spectrometer casing."""
    width = 52.0
    height = 52.0
    thickness = 6.0

    base = Part.makeBox(width, height, thickness, App.Vector(-width/2, -height/2, 0))

    # Central optical bore
    bore = Part.makeCylinder(10.0, thickness + 0.4, App.Vector(0, 0, -0.2), App.Vector(0, 0, 1))
    base = base.cut(bore)

    # 4 Casing mounting holes (M4 counterbored at +-15, +-15)
    for dx in [-15.0, 15.0]:
        for dy in [-15.0, 15.0]:
            th = Part.makeCylinder(2.25, thickness + 0.4, App.Vector(dx, dy, -0.2), App.Vector(0, 0, 1))
            cb = Part.makeCylinder(4.00, 3.6, App.Vector(dx, dy, 2.5), App.Vector(0, 0, 1))
            base = base.cut(th).cut(cb)

    # 4 XY stage clamping holes with captive M4 hex nut pockets on the rear face (Z=0)
    for dx in [-20.5, 20.5]:
        for dy in [-20.5, 20.5]:
            th = Part.makeCylinder(2.25, thickness + 0.4, App.Vector(dx, dy, -0.2), App.Vector(0, 0, 1))
            nut = make_hex_pocket(4.16, 3.2, App.Vector(dx, dy, 0), App.Vector(0, 0, 1))
            base = base.cut(th).cut(nut)

    return base

def build_stage2_xy_slider():
    """Stage 2: XY Translation Slider with integrated Z-guide sleeve and pinch clamp."""
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
    """Stage 3: Z-Focus Carrier & Thorlabs PDAPC3 Detector Mount."""
    barrel_od = 25.6  # 0.20 mm radial clearance inside 26.0 mm sleeve
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

def export_part(shape, step_name, stl_name):
    """Validates shape, exports STEP and high-res STL, copies to Desktop."""
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
    print("Building 3D-Printable 3-Axis (XYZ) Mount for Thorlabs PDAPC3...")
    results = {}

    # 1. Stage 1 (Base Plate)
    s1 = build_stage1_base()
    results["Stage1_Base"] = export_part(s1, "PDAPC3_XYZ_Stage1_Base.step", "PDAPC3_XYZ_Stage1_Base.stl")

    # 2. Stage 2 (XY Slider)
    s2 = build_stage2_xy_slider()
    results["Stage2_XY_Slider"] = export_part(s2, "PDAPC3_XYZ_Stage2_XY_Slider.step", "PDAPC3_XYZ_Stage2_XY_Slider.stl")

    # 3. Stage 3 (Z Carrier)
    s3 = build_stage3_z_carrier()
    results["Stage3_Z_Carrier"] = export_part(s3, "PDAPC3_XYZ_Stage3_Z_Carrier.step", "PDAPC3_XYZ_Stage3_Z_Carrier.stl")

    # 4. Assembled Complete Unit (nominal position: Stage 3 inserted 12 mm into Stage 2)
    s2_placed = s2.copy()
    s2_placed.translate(App.Vector(0, 0, 6.0))  # Sits on Stage 1 (thickness 6.0)

    s3_placed = s3.copy()
    s3_placed.translate(App.Vector(0, 0, 6.0 + 5.0 + 4.0))  # Nominal focus position (4 mm extension)

    assembly = Part.Compound([s1, s2_placed, s3_placed])
    assembly_path = OUTPUT_DIR / "PDAPC3_XYZ_Mount_Assembly.step"
    assembly.exportStep(str(assembly_path))
    if DESKTOP_DIR.exists():
        shutil.copy2(assembly_path, DESKTOP_DIR / "PDAPC3_XYZ_Mount_Assembly.step")
        print("Exported & Copied to Desktop: PDAPC3_XYZ_Mount_Assembly.step")

    report_path = OUTPUT_DIR / "pdapc3_xyz_mount_validation.json"
    report_path.write_text(json.dumps(results, indent=2))
    print("\nAll XYZ Mount parts generated and validated successfully!")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
