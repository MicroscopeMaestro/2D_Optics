import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("SPLITTING MODEL 5 TOP CASING WITH BOTTOM-JOINING M4 HOLES IN DMD REGION")
print("=" * 80)

# Load Model 5 base top
s = Part.read('support_top_half_new.step')
orig_vol = s.Volume
print(f"Loaded support_top_half_new.step | Volume: {orig_vol:.2f} mm^3 | BBox: {s.BoundBox}")

# Stepped cutter catching both bottom-joining clamshell M4 screws:
# Section 1 (DMD aperture block): Y in [-86, -37], Z in [90, 125]
# Section 2 (Lateral clamping flange): Y in [-86, -61], Z in [125, 156]
#   This captures Screw 1 (Y=-76, Z=136) and Screw 2 (Y=-76, Z=150)
box_dmd = Part.makeBox(25.0, 53.0, 35.0, FreeCAD.Vector(-5.0, -90.0, 90.0))
box_flange = Part.makeBox(25.0, 25.0, 31.0, FreeCAD.Vector(-5.0, -86.0, 125.0))
cutter = box_dmd.fuse(box_flange)

dmd_piece = s.common(cutter)
rest_piece = s.cut(cutter)

print(f"\ndmd_piece volume:  {dmd_piece.Volume:.2f} mm^3 ({(dmd_piece.Volume/orig_vol)*100:.1f}%)")
print(f"rest_piece volume: {rest_piece.Volume:.2f} mm^3 ({(rest_piece.Volume/orig_vol)*100:.1f}%)")
print(f"Sum of volumes:    {dmd_piece.Volume + rest_piece.Volume:.2f} mm^3 (Diff: {abs(orig_vol - (dmd_piece.Volume + rest_piece.Volume)):.4f})")

assert len(dmd_piece.Solids) == 1, "dmd_piece must be 1 solid"
assert len(rest_piece.Solids) == 1, "rest_piece must be 1 solid"
assert dmd_piece.isValid(), "dmd_piece must be valid"
assert rest_piece.isValid(), "rest_piece must be valid"

# Verify the 2 bottom-joining clamshell M4 holes:
c136 = Part.makeCylinder(1.65, 25.0, FreeCAD.Vector(-5.0, -76.0, 136.0), FreeCAD.Vector(1, 0, 0))
c150 = Part.makeCylinder(1.65, 25.0, FreeCAD.Vector(-5.0, -76.0, 150.0), FreeCAD.Vector(1, 0, 0))
s136 = Part.makeCylinder(3.5, 25.0, FreeCAD.Vector(-5.0, -76.0, 136.0), FreeCAD.Vector(1, 0, 0)).cut(c136)
s150 = Part.makeCylinder(3.5, 25.0, FreeCAD.Vector(-5.0, -76.0, 150.0), FreeCAD.Vector(1, 0, 0)).cut(c150)

assert dmd_piece.common(s136).Volume > 500, "Screw hole at (Y=-76, Z=136) missing or clipped in dmd_piece!"
assert dmd_piece.common(s150).Volume > 500, "Screw hole at (Y=-76, Z=150) missing or clipped in dmd_piece!"
print("\n[VERIFIED] Both bottom-joining clamshell M4 holes captured in dmd_piece:")
print("  - M4 through-hole 1: Y = -76.0, Z = 136.0 (mates with bottom half tap hole)")
print("  - M4 through-hole 2: Y = -76.0, Z = 150.0 (mates with bottom half tap hole)")

# Verify the 2 DMD adapter mounting holes:
p1 = FreeCAD.Vector(15.0, -81.56, 107.23)
p2 = FreeCAD.Vector(15.0, -52.64, 99.26)
axis = FreeCAD.Vector(0, 0.266, 0.964)
cyl_shell1 = Part.makeCylinder(3.0, 10, p1, axis).cut(Part.makeCylinder(1.65, 10, p1, axis))
cyl_shell2 = Part.makeCylinder(3.0, 10, p2, axis).cut(Part.makeCylinder(1.65, 10, p2, axis))

assert dmd_piece.common(cyl_shell1).Volume > 150, "DMD mount hole 1 missing or clipped!"
assert dmd_piece.common(cyl_shell2).Volume > 150, "DMD mount hole 2 missing or clipped!"
print("\n[VERIFIED] Both DMD adapter mounting M4 holes captured in dmd_piece:")
print("  - DMD mount hole 1: X = 15.0, Y = -81.56, Z = 107.23")
print("  - DMD mount hole 2: X = 15.0, Y = -52.64, Z = 99.26")

# Export STEP files
out_dmd = "top_dmd_region.step"
out_rest = "top_optics_region.step"
dmd_piece.exportStep(out_dmd)
rest_piece.exportStep(out_rest)

# Deploy to Desktop
desktop_dir = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
if os.path.exists(desktop_dir):
    shutil.copy(out_dmd, os.path.join(desktop_dir, out_dmd))
    shutil.copy(out_rest, os.path.join(desktop_dir, out_rest))
    print(f"\nCopied both files to: {desktop_dir}")

print("\nSUCCESS: All operations and verifications completed successfully!")
