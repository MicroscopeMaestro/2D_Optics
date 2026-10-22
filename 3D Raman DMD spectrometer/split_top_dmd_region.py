import sys
import os
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("SPLITTING TOP CASING INTO TWO REGIONS: DMD REGION & THE REST")
print("=" * 80)

# Load original top half
input_file = 'support_top_half.step'
shape = Part.Shape()
shape.read(input_file)
orig_vol = shape.Volume
print(f"Loaded {input_file} | Volume: {orig_vol:.2f} mm^3 | Solids: {len(shape.Solids)}")

# Cutter box isolating the small DMD aperture region:
# The DMD aperture sits at Y in [-88.5, -45.1], Z in [98.2, 110.2].
# Hole 1 is at (15.0, -81.56, 107.23).
# Hole 2 is at (15.0, -52.64, 99.26).
# Both M4 holes have depth 10mm along dmd_z (towards +Y, +Z).
#
# Cutter bounds:
# X: [-2.0, 22.0]  (full clamshell thickness 0 to 17.5 mm)
# Y: [-110.0, -40.0] (from outer wall Y=-105 to dividing wall Y=-40)
# Z: [94.0, 122.0]   (from front wall Z=95 to dividing wall Z=122)
box = Part.makeBox(24.0, 70.0, 28.0, FreeCAD.Vector(-2.0, -110.0, 94.0))

# Perform split
dmd_piece = shape.common(box)
rest_piece = shape.cut(box)

print(f"\ndmd_piece volume:  {dmd_piece.Volume:.2f} mm^3 ({(dmd_piece.Volume/orig_vol)*100:.1f}% of total)")
print(f"rest_piece volume: {rest_piece.Volume:.2f} mm^3 ({(rest_piece.Volume/orig_vol)*100:.1f}% of total)")
print(f"Sum of volumes:    {dmd_piece.Volume + rest_piece.Volume:.2f} mm^3 (Difference: {abs(orig_vol - (dmd_piece.Volume + rest_piece.Volume)):.4f})")

# Verify solids and validity
assert len(dmd_piece.Solids) == 1, f"Expected 1 solid for dmd_piece, got {len(dmd_piece.Solids)}"
assert len(rest_piece.Solids) == 1, f"Expected 1 solid for rest_piece, got {len(rest_piece.Solids)}"
assert dmd_piece.isValid(), "dmd_piece is not valid"
assert rest_piece.isValid(), "rest_piece is not valid"

# Verify M4 holes in dmd_piece
dmd_m4 = []
for f in dmd_piece.Faces:
    if type(f.Surface) == Part.Cylinder and 1.5 <= f.Surface.Radius <= 1.8:
        dmd_m4.append((f.Surface.Radius, f.Surface.Center))

print(f"\nM4 holes in dmd_piece: {len(dmd_m4)}")
for r, c in dmd_m4:
    print(f"  Radius = {r:.2f} mm, Center = ({c.x:.2f}, {c.y:.2f}, {c.z:.2f})")
assert len(dmd_m4) == 2, f"Expected exactly 2 M4 holes in dmd_piece, got {len(dmd_m4)}"

# Verify M4 holes in rest_piece
rest_m4 = []
for f in rest_piece.Faces:
    if type(f.Surface) == Part.Cylinder and 1.5 <= f.Surface.Radius <= 1.8:
        rest_m4.append((f.Surface.Radius, f.Surface.Center))
print(f"\nM4 holes in rest_piece: {len(rest_m4)} (entrance + sensor holes preserved)")

# Export STEP files
out_dmd = "top_dmd_region.step"
out_rest = "top_optics_region.step"

dmd_piece.exportStep(out_dmd)
rest_piece.exportStep(out_rest)
print(f"\nSuccessfully exported:\n  - {out_dmd}\n  - {out_rest}")

# Copy to print folder
desktop_dir = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
if os.path.exists(desktop_dir):
    import shutil
    shutil.copy(out_dmd, os.path.join(desktop_dir, out_dmd))
    shutil.copy(out_rest, os.path.join(desktop_dir, out_rest))
    print(f"Copied both files to {desktop_dir}")

print("\nALL VERIFICATIONS PASSED!")
