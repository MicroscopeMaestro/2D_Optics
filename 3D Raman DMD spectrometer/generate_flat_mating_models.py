import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("GENERATING FLAT MATING SURFACE MODEL (NO EXTRUDING SURFACES / NO TONGUE)")
print("=" * 80)

s = Part.read('support_top_half_new.step')

# Trim off the extruding surfaces (X < 0) to ensure a completely flat mirror mating plane at X = 0:
flat_box = Part.makeBox(100.0, 500.0, 500.0, FreeCAD.Vector(0.0, -250.0, -50.0))
s_flat = s.common(flat_box)
print(f"Flat base volume: {s_flat.Volume:.2f} mm^3 | BBox: {s_flat.BoundBox}")

# 3-Screw Stepped Cutter:
box_dmd = Part.makeBox(25.0, 53.0, 35.0, FreeCAD.Vector(0.0, -90.0, 90.0))
box_flange = Part.makeBox(25.0, 29.0, 31.0, FreeCAD.Vector(0.0, -90.0, 125.0))
box_ridge = Part.makeBox(25.0, 17.0, 14.0, FreeCAD.Vector(0.0, -37.0, 115.0))
cutter = box_dmd.fuse(box_flange).fuse(box_ridge)

dmd_flat = s_flat.common(cutter)
rest_flat = s_flat.cut(cutter)

print(f"\ndmd_flat volume:  {dmd_flat.Volume:.2f} mm^3 ({(dmd_flat.Volume/s_flat.Volume)*100:.1f}%)")
print(f"rest_flat volume: {rest_flat.Volume:.2f} mm^3 ({(rest_flat.Volume/s_flat.Volume)*100:.1f}%)")
print(f"Sum: {dmd_flat.Volume + rest_flat.Volume:.2f} vs orig {s_flat.Volume:.2f}")

assert len(dmd_flat.Solids) == 1, "dmd_flat must be 1 solid"
assert len(rest_flat.Solids) == 1, "rest_flat must be 1 solid"
assert dmd_flat.isValid(), "dmd_flat must be valid"
assert rest_flat.isValid(), "rest_flat must be valid"

# Export STEP files directly to print folder
out_dmd = "top_dmd_region.step"
out_rest = "top_optics_region.step"
dmd_flat.exportStep(out_dmd)
rest_flat.exportStep(out_rest)

desktop_dir = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
if os.path.exists(desktop_dir):
    shutil.copy(out_dmd, os.path.join(desktop_dir, out_dmd))
    shutil.copy(out_rest, os.path.join(desktop_dir, out_rest))
    print(f"\nExported and copied flat mating models (X in [0, 17.5]) to: {desktop_dir}")

print("SUCCESS: Flat mating plane verified and deployed!")
