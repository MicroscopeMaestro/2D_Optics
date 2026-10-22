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

# Create a DIAGONAL cutting surface in YZ plane
# The cut should go from:
# - At Y = -105 (floor): Z = 95 (start of DMD region)
# - At Y = -30 (top of DMD window): Z = 160 (past joining M4 holes)
# This creates a diagonal gradient from floor to window top

# Define the diagonal cutting profile in YZ at X=0
# Points going clockwise around the cutting polygon
y_floor = -105.0
y_top = -30.0
z_start = 95.0
z_end = 160.0

# Create a polygon that defines the diagonal cut
# The cut separates: DMD region (below/left of diagonal) from rest (above/right)
pts = [
    FreeCAD.Vector(0, y_floor, z_start),      # Bottom-left: floor at DMD start
    FreeCAD.Vector(0, y_top, z_start),        # Top-left: above window at DMD start
    FreeCAD.Vector(0, y_top, 300),            # Top-right: far Z
    FreeCAD.Vector(0, y_floor, 300),          # Bottom-right: far Z at floor
    FreeCAD.Vector(0, y_floor, z_start),      # Close polygon
]

# Actually, let's make the diagonal go from (Y=-105, Z=95) to (Y=-30, Z=160)
# The DMD region is BELOW this diagonal line
# The rest is ABOVE this diagonal line

# Create the cutting face as a polygon in YZ plane
cut_pts = [
    FreeCAD.Vector(0, y_floor, z_start),      # Start at floor, DMD region start
    FreeCAD.Vector(0, y_top, z_end),          # Diagonal to top of window, past M4 holes
    FreeCAD.Vector(0, y_top, 300),            # Up to max Z
    FreeCAD.Vector(0, y_floor, 300),          # Across at floor
    FreeCAD.Vector(0, y_floor, z_start),      # Close
]

wire = Part.makePolygon(cut_pts)
cut_face = Part.Face(wire)

# Extrude along X to cover full casing thickness (0 to 17.5, plus margin)
cutter = cut_face.extrude(FreeCAD.Vector(50, 0, 0))
cutter.translate(FreeCAD.Vector(-25, 0, 0))  # Center on casing

print()
print('Diagonal cutter created:')
print(f'  From (Y={y_floor}, Z={z_start}) to (Y={y_top}, Z={z_end})')

# Perform boolean operations
dmd_part = top.common(cutter)
rest_part = top.cut(cutter)

print()
print('DMD Region (below diagonal):')
print(f'  Volume: {dmd_part.Volume:.2f}')
if dmd_part.Volume > 0:
    bb = dmd_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    dmd_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_dmd_diagonal.step')
    print('  Exported: top_dmd_diagonal.step')

print()
print('Rest Region (above diagonal):')
print(f'  Volume: {rest_part.Volume:.2f}')
if rest_part.Volume > 0:
    bb = rest_part.BoundBox
    print(f'  BBox X: {bb.XMin:.2f} to {bb.XMax:.2f}')
    print(f'  BBox Y: {bb.YMin:.2f} to {bb.YMax:.2f}')
    print(f'  BBox Z: {bb.ZMin:.2f} to {bb.ZMax:.2f}')
    rest_part.exportStep(r'C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_rest_diagonal.step')
    print('  Exported: top_rest_diagonal.step')

print()
print(f'Total volume check: {dmd_part.Volume + rest_part.Volume:.2f} (original: {top.Volume:.2f})')

# Check for cylindrical faces in DMD region
print()
print('Cylindrical faces in DMD region:')
for i, face in enumerate(dmd_part.Faces):
    surf = face.Surface
    if hasattr(surf, 'Radius') and surf.Radius > 1.5 and surf.Radius < 3.0:
        com = face.CenterOfMass
        print(f'  Face {i}: Radius={surf.Radius:.2f} at X={com.x:.2f}, Y={com.y:.2f}, Z={com.z:.2f}')