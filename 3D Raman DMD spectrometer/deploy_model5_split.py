import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("SPLITTING MODEL 5 BASE TOP (support_top_half_new.step) INTO TWO REGIONS")
print("=" * 80)

# Base model matching printing_model_5.stl
s = Part.read('support_top_half_new.step')
orig_vol = s.Volume
print(f"Loaded support_top_half_new.step | Volume: {orig_vol:.2f} mm^3 | BBox: {s.BoundBox}")

# Clean rectangular cut separating the DMD aperture region:
# Y: [-90, -37] (outer wall is Y=-86; dividing face is at Y=-37)
# Z: [90, 125]   (front wall is Z=98; dividing face is at Z=125)
# X: [-5, 20]    (covers full clamshell thickness X from -3 to 17.5)
cutter = Part.makeBox(25.0, 53.0, 35.0, FreeCAD.Vector(-5.0, -90.0, 90.0))

dmd_piece = s.common(cutter)
rest_piece = s.cut(cutter)

print(f"\ndmd_piece volume:  {dmd_piece.Volume:.2f} mm^3 ({(dmd_piece.Volume/orig_vol)*100:.1f}%)")
print(f"rest_piece volume: {rest_piece.Volume:.2f} mm^3 ({(rest_piece.Volume/orig_vol)*100:.1f}%)")
print(f"Sum of volumes:    {dmd_piece.Volume + rest_piece.Volume:.2f} mm^3 (Diff: {abs(orig_vol - (dmd_piece.Volume + rest_piece.Volume)):.4f})")

assert len(dmd_piece.Solids) == 1, "dmd_piece must be 1 solid"
assert len(rest_piece.Solids) == 1, "rest_piece must be 1 solid"
assert dmd_piece.isValid(), "dmd_piece must be valid"
assert rest_piece.isValid(), "rest_piece must be valid"

# Verify M4 holes in dmd_piece
p1 = FreeCAD.Vector(15.0, -81.56, 107.23)
p2 = FreeCAD.Vector(15.0, -52.64, 99.26)
axis = FreeCAD.Vector(0, 0.266, 0.964)
cyl_shell1 = Part.makeCylinder(3.0, 10, p1, axis).cut(Part.makeCylinder(1.65, 10, p1, axis))
cyl_shell2 = Part.makeCylinder(3.0, 10, p2, axis).cut(Part.makeCylinder(1.65, 10, p2, axis))

assert dmd_piece.common(cyl_shell1).Volume > 150, "Hole 1 not in dmd_piece"
assert dmd_piece.common(cyl_shell2).Volume > 150, "Hole 2 not in dmd_piece"
print("\nBoth DMD M4 holes verified intact in dmd_piece!")

# Export clean split STEP files
out_dmd = "top_dmd_region.step"
out_rest = "top_optics_region.step"
dmd_piece.exportStep(out_dmd)
rest_piece.exportStep(out_rest)

# Deploy to Desktop
desktop_dir = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
if os.path.exists(desktop_dir):
    shutil.copy(out_dmd, os.path.join(desktop_dir, out_dmd))
    shutil.copy(out_rest, os.path.join(desktop_dir, out_rest))
    # Also copy the lap-tab versions for convenience
    shutil.copy(r"model5_split_top\Top_DMD_M4.step", os.path.join(desktop_dir, "Top_DMD_M4_with_tabs.step"))
    shutil.copy(r"model5_split_top\Top_main_M4.step", os.path.join(desktop_dir, "Top_main_M4_with_pockets.step"))
    shutil.copy(r"model5_split_top\Top_matching_model5.step", os.path.join(desktop_dir, "support_top_half_model5.step"))
    print(f"\nCopied all Model 5 STEP files to: {desktop_dir}")

print("COMPLETED SUCCESSFULLY!")
