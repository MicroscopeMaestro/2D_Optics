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

# Based on build_spectrometer.py and PROJECT_HANDOFF.md:
# 
# 1. Joining M4 holes (through top and bottom):
#    MANUAL_SCREW_POSITIONS at (-76.0, 136.0) and (-76.0, 150.0) - LEFT MID 1, 2 (DMD lateral)
#    These are at Y=-76, Z=136 and Z=150
#
# 2. DMD Window/Aperture:
#    45x45x100 box at DMD position (label "22_CB_After_DMD_Hardware")
#    DMD center at Y ≈ -66.5, Z ≈ 115 (middle of DMD region)
#    Window spans Y: -89 to -44, X: -22.5 to 22.5, Z: ~15 to 115 (from DMD going backward)
#
# 3. DMD Mount M4 holes (for attaching DMD element):
#    At X=±15, Y=±15 in DMD local frame
#    Global: X=±15, Y=-66.5±15 = -51.5 and -81.5
#    These are on the DMD mount adapter, not in the casing itself
#    But the casing has the window opening at Y=-89 to -44

# Create a box that covers ALL of the above:
# - X: full width of casing (0 to 17.5) - but window goes to X=±22.5, so cover full
# - Y: from floor (-105) up to include DMD window top at Y=-44 and DMD mount holes at Y=-51.5
#   So Y up to ~-40
# - Z: from DMD region start (95) to cover joining M4 holes at Z=150
#   So Z up to ~160

# DMD region with joining M4 holes + DMD aperture + DMD mount holes box
dmd_full_box = Part.makeBox(
    500,      # X width (covers full casing + window)
    65,       # Y height: from -105 to -40 (covers floor to above DMD window/mount holes)
    65,       # Z depth: from 95 to 160 (covers DMD window to joining M4 holes)
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
dmd_full_part = top.common(dmd_full_box)
rest_part = top.cut(dmd_full_box)  # Use cut to get the rest

print()
print('DMD Region with Joining M4 Holes + DMD Aperture + Mount Holes:')
print(f'  Volume: {dmd_full_part.Volume:.2f}')
if dmd_full_part.Volume > 0:
    bb = dmd_full_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    dmd_full_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_dmd_full_region.step')
    print('  Exported: top_dmd_full_region.step')

print()
print('Rest of Casing:')
print(f'  Volume: {rest_part.Volume:.2f}')
if rest_part.Volume > 0:
    bb = rest_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    rest_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_rest_without_dmd_full.step')
    print('  Exported: top_rest_without_dmd_full.step')

print()
print(f'Total volume check: {dmd_full_part.Volume + rest_part.Volume:.2f} (original: {top.Volume:.2f})')

# Check for cylindrical faces (M4 holes) in this region
print()
print('Cylindrical faces in DMD region (potential M4 holes):')
for i, face in enumerate(dmd_full_part.Faces):
    surf = face.Surface
    if hasattr(surf, 'Radius') and surf.Radius > 1.5 and surf.Radius < 3.0:
        com = face.CenterOfMass
        print(f'  Face {i}: Radius={surf.Radius:.2f} at X={com.x:.2f}, Y={com.y:.2f}, Z={com.z:.2f}')