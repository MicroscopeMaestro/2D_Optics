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

# ============================================================
# STEP 1: Create diagonal cutter (same as before)
# Diagonal from floor (Y=-105, Z=95) to top of DMD window (Y=-30, Z=160)
y_floor = -105.0
y_top = -30.0
z_start = 95.0
z_end = 160.0

cut_pts = [
    FreeCAD.Vector(0, y_floor, z_start),
    FreeCAD.Vector(0, y_top, z_end),
    FreeCAD.Vector(0, y_top, 300),
    FreeCAD.Vector(0, y_floor, 300),
    FreeCAD.Vector(0, y_floor, z_start),
]

wire = Part.makePolygon(cut_pts)
cut_face = Part.Face(wire)
diagonal_cutter = cut_face.extrude(FreeCAD.Vector(50, 0, 0))
diagonal_cutter.translate(FreeCAD.Vector(-25, 0, 0))

# Get the diagonal DMD region
diagonal_dmd = top.common(diagonal_cutter)

print()
print('Diagonal DMD Region (before aperture trim):')
print(f'  Volume: {diagonal_dmd.Volume:.2f}')
bb = diagonal_dmd.BoundBox
print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')

# ============================================================
# STEP 2: Create aperture trim box
# DMD aperture is 45x45mm centered at DMD position
# From build_spectrometer.py: dmd_window_cut = Part.makeBox(45.0, 45.0, 100.0, FreeCAD.Vector(-22.5, -22.5, -100.0))
# DMD center at Y ≈ -66.5, Z ≈ 115 (middle of DMD region)
# Aperture spans: Y: -89 to -44, Z: ~15 to 115 (in DMD local coords)
# But in global casing coords, the window cut goes from DMD backward

# The aperture in the top half casing:
# - X: 0 to 17.5 (casing thickness)
# - Y: -89 to -44 (45mm window centered at Y=-66.5)
# - Z: 95 to 160 (DMD region Z range, covering the window depth)

# Add some margin around the aperture for the mounting features
margin = 5.0  # 5mm margin around aperture

aperture_box = Part.makeBox(
    50,       # X width (covers full casing + margin)
    45 + 2*margin,  # Y: -89-margin to -44+margin = -94 to -39
    65 + 2*margin,  # Z: 95-margin to 160+margin = 90 to 165
    FreeCAD.Vector(-25, -94, 90)
)

print()
print('Aperture trim box:')
print(f'  Y: -94 to -39 (aperture -89 to -44 + 5mm margin)')
print(f'  Z: 90 to 165 (DMD region 95-160 + 5mm margin)')

# ============================================================
# STEP 3: Intersect diagonal DMD region with aperture box
dmd_aperture_only = diagonal_dmd.common(aperture_box)

print()
print('DMD Region trimmed to Aperture:')
print(f'  Volume: {dmd_aperture_only.Volume:.2f}')
if dmd_aperture_only.Volume > 0:
    bb = dmd_aperture_only.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    dmd_aperture_only.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_dmd_aperture_only.step')
    print('  Exported: top_dmd_aperture_only.step')

# ============================================================
# STEP 4: The rest is everything else (original minus the aperture-trimmed DMD region)
rest_part = top.cut(dmd_aperture_only)

print()
print('Rest of Casing:')
print(f'  Volume: {rest_part.Volume:.2f}')
if rest_part.Volume > 0:
    bb = rest_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    rest_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_rest_aperture_trimmed.step')
    print('  Exported: top_rest_aperture_trimmed.step')

print()
print(f'Total volume check: {dmd_aperture_only.Volume + rest_part.Volume:.2f} (original: {top.Volume:.2f})')

# Check for cylindrical faces in DMD aperture region
print()
print('Cylindrical faces in DMD aperture region:')
for i, face in enumerate(dmd_aperture_only.Faces):
    surf = face.Surface
    if hasattr(surf, 'Radius') and surf.Radius > 1.5 and surf.Radius < 3.0:
        com = face.CenterOfMass
        print(f'  Face {i}: Radius={surf.Radius:.2f} at X={com.x:.2f}, Y={com.y:.2f}, Z={com.z:.2f}')