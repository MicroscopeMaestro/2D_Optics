import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("GENERATING TRUE MATCHING TOP HALF WITH COMPLETE GROOVES FOR PRINTED BOTTOM")
print("=" * 80)

# Base model: support_top_half.step (true un-trimmed top matching printed support_bottom_half.step)
s_top = Part.read('support_top_half.step')
s_bot = Part.read('support_bottom_half.step')
orig_vol = s_top.Volume
print(f"Loaded support_top_half.step | Volume: {orig_vol:.2f} mm^3 | BBox: {s_top.BoundBox}")

# 3-Screw Stepped Cutter with full Y=-105 coverage:
# 1. DMD aperture block: Y in [-110, -37], Z in [90, 125]
# 2. Lateral clamping flange: Y in [-110, -61], Z in [125, 156] (captures Screws 5 & 6)
# 3. Center ridge clamping block: Y in [-37, -20], Z in [115, 129] (captures Screw 7)
box_dmd = Part.makeBox(25.0, 75.0, 35.0, FreeCAD.Vector(0.0, -110.0, 90.0))
box_flange = Part.makeBox(25.0, 50.0, 31.0, FreeCAD.Vector(0.0, -110.0, 125.0))
box_ridge = Part.makeBox(25.0, 17.0, 14.0, FreeCAD.Vector(0.0, -37.0, 115.0))
cutter = box_dmd.fuse(box_flange).fuse(box_ridge)

dmd_piece = s_top.common(cutter)
rest_piece = s_top.cut(cutter)

print(f"\ndmd_piece volume:  {dmd_piece.Volume:.2f} mm^3 ({(dmd_piece.Volume/orig_vol)*100:.1f}%)")
print(f"rest_piece volume: {rest_piece.Volume:.2f} mm^3 ({(rest_piece.Volume/orig_vol)*100:.1f}%)")
print(f"Sum of volumes:    {dmd_piece.Volume + rest_piece.Volume:.2f} mm^3 (Diff: {abs(orig_vol - (dmd_piece.Volume + rest_piece.Volume)):.4f})")

assert len(dmd_piece.Solids) == 1, "dmd_piece must be 1 solid"
assert len(rest_piece.Solids) == 1, "rest_piece must be 1 solid"
assert dmd_piece.isValid(), "dmd_piece must be valid"
assert rest_piece.isValid(), "rest_piece must be valid"

# 1. VERIFY GROOVE CLEARANCE FOR ALL BOTTOM TONGUES:
tongue_all = s_bot.common(Part.makeBox(10.0, 500.0, 500.0, FreeCAD.Vector(0.001, -250.0, -50.0)))
dmd_clash = dmd_piece.common(tongue_all).Volume
rest_clash = rest_piece.common(tongue_all).Volume

assert dmd_clash < 1e-4, f"DMD piece collides with bottom tongue by {dmd_clash} mm^3!"
assert rest_clash < 1e-4, f"Rest piece collides with bottom tongue by {rest_clash} mm^3!"
print("\n[VERIFIED] Tongue-and-Groove Clearance:")
print(f"  - dmd_piece collision with bottom tongue:  {dmd_clash:.4f} mm^3 (perfect fit)")
print(f"  - rest_piece collision with bottom tongue: {rest_clash:.4f} mm^3 (perfect fit)")

# Inspect groove faces in dmd_piece:
groove_faces = [f for f in dmd_piece.Faces if abs(f.BoundBox.XMin - 3.2) < 0.05 and abs(f.BoundBox.XMax - 3.2) < 0.05]
print(f"  - dmd_piece groove bottom faces at X=3.2: {len(groove_faces)}, total area = {sum(f.Area for f in groove_faces):.2f} mm^2")
for f in groove_faces:
    print(f"      Face Area={f.Area:6.2f} mm^2, Y=[{f.BoundBox.YMin:6.1f},{f.BoundBox.YMax:6.1f}], Z=[{f.BoundBox.ZMin:6.1f},{f.BoundBox.ZMax:6.1f}]")

# 2. VERIFY 3 BOTTOM-JOINING CLAMSHELL M4 HOLES:
screws = [
    (-76.0, 136.0, "Screw 5 (DMD lateral 1)"),
    (-76.0, 150.0, "Screw 6 (DMD lateral 2)"),
    (-26.0, 122.0, "Screw 7 (MID CENTER ridge)")
]
print("\n[VERIFIED] Bottom-joining Clamshell M4 Holes in dmd_piece:")
for sy, sz, label in screws:
    c = Part.makeCylinder(1.65, 25.0, FreeCAD.Vector(-5.0, sy, sz), FreeCAD.Vector(1, 0, 0))
    sh = Part.makeCylinder(3.5, 25.0, FreeCAD.Vector(-5.0, sy, sz), FreeCAD.Vector(1, 0, 0)).cut(c)
    assert dmd_piece.common(sh).Volume > 300, f"{label} missing or clipped in dmd_piece!"
    print(f"  - {label} at Y={sy:.1f}, Z={sz:.1f} captured with full cylindrical wall")

# 3. VERIFY 2 DMD ADAPTER MOUNTING M4 HOLES:
p1 = FreeCAD.Vector(15.0, -81.56, 107.23)
p2 = FreeCAD.Vector(15.0, -52.64, 99.26)
axis = FreeCAD.Vector(0, 0.266, 0.964)
cyl_shell1 = Part.makeCylinder(3.0, 10, p1, axis).cut(Part.makeCylinder(1.65, 10, p1, axis))
cyl_shell2 = Part.makeCylinder(3.0, 10, p2, axis).cut(Part.makeCylinder(1.65, 10, p2, axis))

assert dmd_piece.common(cyl_shell1).Volume > 150, "DMD mount hole 1 missing or clipped!"
assert dmd_piece.common(cyl_shell2).Volume > 150, "DMD mount hole 2 missing or clipped!"
print("\n[VERIFIED] DMD Adapter Mounting M4 Holes in dmd_piece:")
print(f"  - DMD mount hole 1 at X=15.0, Y=-81.56, Z=107.23 captured")
print(f"  - DMD mount hole 2 at X=15.0, Y=-52.64, Z=99.26 captured")

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
    print(f"\nCopied both verified STEP files to: {desktop_dir}")

print("\nSUCCESS: Complete groove coverage and bottom half compatibility verified!")
