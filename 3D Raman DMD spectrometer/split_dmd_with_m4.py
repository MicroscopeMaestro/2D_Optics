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

# Based on PROJECT_HANDOFF.md:
# - DMD center at Y ≈ -66.5 (global)
# - M4 holes at X=±15, Y=±15 in DMD local frame
# - So M4 holes at global Y ≈ -66.5 ± 15 = -51.5 and -81.5
# - The "two closest M4 holes" = the pair at Y ≈ -51.5 (X=±15)
# - DMD region Z range: ~95 to 135 (before focus lens at ~125)

# Create a box that covers:
# - X: full width of casing (0 to 17.5)
# - Y: from floor (-105) up to include the M4 holes at Y=-51.5 (so up to ~-45)
# - Z: from DMD region start (95) to before focus lens (135)

# DMD region with M4 holes box
dmd_m4_box = Part.makeBox(
    500,      # X width (covers full casing)
    60,       # Y height: from -105 to -45 (covers M4 holes at Y=-51.5)
    40,       # Z depth: from 95 to 135
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
print('DMD Region with M4 Holes (Y=-51.5 pair):')
print(f'  Volume: {dmd_m4_part.Volume:.2f}')
if dmd_m4_part.Volume > 0:
    bb = dmd_m4_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    dmd_m4_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_dmd_with_m4.step')
    print('  Exported: top_dmd_with_m4.step')

print()
print('Rest of Casing:')
print(f'  Volume: {rest_part.Volume:.2f}')
if rest_part.Volume > 0:
    bb = rest_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    rest_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_rest_without_dmd.step')
    print('  Exported: top_rest_without_dmd.step')

print()
print(f'Total volume check: {dmd_m4_part.Volume + rest_part.Volume:.2f} (original: {top.Volume:.2f})')