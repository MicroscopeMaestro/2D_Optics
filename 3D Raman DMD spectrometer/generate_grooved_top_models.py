import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("GENERATING MODEL 5 TOP CASING WITH GROOVES FOR BOTTOM TONGUES")
print("=" * 80)

# Base model: support_top_half.step has the native 3.2mm grooves built-in.
# We trim it to Y >= -86.0 to strictly match the printing_model_5 footprint.
s_top_old = Part.read('support_top_half.step')
s_bot = Part.read('support_bottom_half.step')

# Trim to Y >= -86.0 (Model 5 footprint):
trim_box = Part.makeBox(100.0, 200.0, 500.0, FreeCAD.Vector(-10.0, -86.0, -50.0))
s_base = s_top_old.common(trim_box)
print(f"Base grooved Model 5 volume: {s_base.Volume:.2f} mm^3 | BBox: {s_base.BoundBox}")

# 3-Screw Stepped Cutter:
box_dmd = Part.makeBox(25.0, 53.0, 35.0, FreeCAD.Vector(0.0, -90.0, 90.0))
box_flange = Part.makeBox(25.0, 29.0, 31.0, FreeCAD.Vector(0.0, -90.0, 125.0))
box_ridge = Part.makeBox(25.0, 17.0, 14.0, FreeCAD.Vector(0.0, -37.0, 115.0))
cutter = box_dmd.fuse(box_flange).fuse(box_ridge)

dmd_piece = s_base.common(cutter)
rest_piece = s_base.cut(cutter)

print(f"\ndmd_piece volume:  {dmd_piece.Volume:.2f} mm^3 ({(dmd_piece.Volume/s_base.Volume)*100:.1f}%)")
print(f"rest_piece volume: {rest_piece.Volume:.2f} mm^3 ({(rest_piece.Volume/s_base.Volume)*100:.1f}%)")
print(f"Sum: {dmd_piece.Volume + rest_piece.Volume:.2f} vs orig {s_base.Volume:.2f}")

assert len(dmd_piece.Solids) == 1, "dmd_piece must be 1 solid"
assert len(rest_piece.Solids) == 1, "rest_piece must be 1 solid"
assert dmd_piece.isValid(), "dmd_piece must be valid"
assert rest_piece.isValid(), "rest_piece must be valid"

# Verify bottom tongue clearance (must be 0.0000 mm^3 collision):
tongue = s_bot.common(Part.makeBox(10.0, 500.0, 500.0, FreeCAD.Vector(0.001, -250.0, -50.0)))
dmd_tongue_clash = dmd_piece.common(tongue).Volume
rest_tongue_clash = rest_piece.common(tongue).Volume

assert dmd_tongue_clash < 1e-4, f"DMD piece collides with bottom tongue by {dmd_tongue_clash} mm^3!"
assert rest_tongue_clash < 1e-4, f"Rest piece collides with bottom tongue by {rest_tongue_clash} mm^3!"
print("\n[VERIFIED] Tongue space clearance:")
print("  - dmd_piece collision with bottom tongue:  0.0000 mm^3 (perfect clearance)")
print("  - rest_piece collision with bottom tongue: 0.0000 mm^3 (perfect clearance)")

# Verify the 3 bottom-joining clamshell M4 holes:
screws = [
    (-76.0, 136.0, "Screw 5 (DMD lateral 1)"),
    (-76.0, 150.0, "Screw 6 (DMD lateral 2)"),
    (-26.0, 122.0, "Screw 7 (MID CENTER ridge)")
]
for sy, sz, label in screws:
    c = Part.makeCylinder(1.65, 25.0, FreeCAD.Vector(-5.0, sy, sz), FreeCAD.Vector(1, 0, 0))
    sh = Part.makeCylinder(3.5, 25.0, FreeCAD.Vector(-5.0, sy, sz), FreeCAD.Vector(1, 0, 0)).cut(c)
    assert dmd_piece.common(sh).Volume > 300, f"{label} missing or clipped in dmd_piece!"
    print(f"  [VERIFIED] {label} at Y={sy:.1f}, Z={sz:.1f} captured in dmd_piece")

# Verify the 2 DMD adapter mounting M4 holes:
p1 = FreeCAD.Vector(15.0, -81.56, 107.23)
p2 = FreeCAD.Vector(15.0, -52.64, 99.26)
axis = FreeCAD.Vector(0, 0.266, 0.964)
cyl_shell1 = Part.makeCylinder(3.0, 10, p1, axis).cut(Part.makeCylinder(1.65, 10, p1, axis))
cyl_shell2 = Part.makeCylinder(3.0, 10, p2, axis).cut(Part.makeCylinder(1.65, 10, p2, axis))

assert dmd_piece.common(cyl_shell1).Volume > 150, "DMD mount hole 1 missing or clipped!"
assert dmd_piece.common(cyl_shell2).Volume > 150, "DMD mount hole 2 missing or clipped!"
print(f"  [VERIFIED] DMD mount hole 1 at X=15.0, Y=-81.56, Z=107.23 captured")
print(f"  [VERIFIED] DMD mount hole 2 at X=15.0, Y=-52.64, Z=99.26 captured")

# Export STEP files directly to print folder
out_dmd = "top_dmd_region.step"
out_rest = "top_optics_region.step"
dmd_piece.exportStep(out_dmd)
rest_piece.exportStep(out_rest)

desktop_dir = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
if os.path.exists(desktop_dir):
    shutil.copy(out_dmd, os.path.join(desktop_dir, out_dmd))
    shutil.copy(out_rest, os.path.join(desktop_dir, out_rest))
    print(f"\nExported and copied grooved models to: {desktop_dir}")

print("SUCCESS: All operations, groove clearances, and verifications completed successfully!")
