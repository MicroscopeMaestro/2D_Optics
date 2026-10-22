import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

# Read the original top half
top = Part.Shape()
top.read(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\support_top_half.step')

print('Original Top Half:')
bb = top.BoundBox
print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
print(f'  Volume: {top.Volume:.2f}')

# Based on build_spectrometer.py MANUAL_SCREW_POSITIONS:
# M4 holes that join top and bottom (through both halves):
# (-76.0, 136.0) - LEFT MID 1 (DMD lateral)
# (-76.0, 150.0) - LEFT MID 2
# These are at Y=-76, Z=136 and Z=150

# The DMD region is around Z=95-234, Y=-86 to 0
# We want to include the M4 holes at Y=-76, Z=136-150

# Create a box that covers:
# - X: full width of casing (0 to 17.5)
# - Y: from floor (-105) up to include M4 holes at Y=-76 (so up to ~-70)
# - Z: from DMD region start (95) to cover M4 holes at Z=150 (so up to ~160)

# DMD region with joining M4 holes box
dmd_m4_box = Part.makeBox(
    500,      # X width (covers full casing)
    35,       # Y height: from -105 to -70 (covers M4 holes at Y=-76)
    65,       # Z depth: from 95 to 160 (covers M4 holes at Z=136-150)
    FreeCAD.Vector(-250, -105, 95)
)

# Rest of the casing
rest_box = Part.makeBox(
    500,      # X width
    500,      # Y height (full)
    500,      # Z depth (full)
    FreeCAD.Vector(-250, -250, 0)
)

# Perform boolean operations
dmd_m4_part = top.common(dmd_m4_box)
rest_part = top.cut(dmd_m4_box)  # Use cut to get the rest

print()
print('DMD Region with Joining M4 Holes (Y=-76, Z=136-150):')
print(f'  Volume: {dmd_m4_part.Volume:.2f}')
if dmd_m4_part.Volume > 0:
    bb = dmd_m4_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    dmd_m4_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_dmd_with_joining_m4.step')
    print('  Exported: top_dmd_with_joining_m4.step')

print()
print('Rest of Casing:')
print(f'  Volume: {rest_part.Volume:.2f}')
if rest_part.Volume > 0:
    bb = rest_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    rest_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_rest_without_dmd_joining.step')
    print('  Exported: top_rest_without_dmd_joining.step')

print()
print(f'Total volume check: {dmd_m4_part.Volume + rest_part.Volume:.2f} (original: {top.Volume:.2f})')