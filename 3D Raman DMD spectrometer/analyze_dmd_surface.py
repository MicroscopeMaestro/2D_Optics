import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

shape = Part.Shape()
shape.read('IPD1231G_surfaces_42_to_78.stp')

bb = shape.BoundBox
print(f"Bounding Box:")
print(f"  X: {bb.XMin:.2f} to {bb.XMax:.2f} (Width: {bb.XLength:.2f})")
print(f"  Y: {bb.YMin:.2f} to {bb.YMax:.2f} (Height: {bb.YLength:.2f})")
print(f"  Z: {bb.ZMin:.2f} to {bb.ZMax:.2f} (Depth: {bb.ZLength:.2f})")

center_x = (bb.XMin + bb.XMax) / 2.0
center_y = (bb.YMin + bb.YMax) / 2.0
center_z = (bb.ZMin + bb.ZMax) / 2.0
print(f"\nCenter: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")

holes = []
for face in shape.Faces:
    if str(face.Surface.__class__.__name__) == 'Cylinder':
        r = face.Surface.Radius
        if 0.5 < r < 5.0: # Looking for screw holes
            c = face.Surface.Center
            a = face.Surface.Axis
            holes.append((r, c, a))

unique_holes = []
for r, c, a in holes:
    is_new = True
    for ur, uc, ua in unique_holes:
        if abs(ur - r) < 0.1 and c.distanceToPoint(uc) < 0.1:
            is_new = False
            break
    if is_new:
        unique_holes.append((r, c, a))

print("\nPotential mounting holes found:")
for idx, (r, c, a) in enumerate(unique_holes):
    # Print absolute and relative to center
    dx = c.x - center_x
    dy = c.y - center_y
    dz = c.z - center_z
    print(f"  Hole {idx+1}: R={r:.2f} | Center=({c.x:.2f}, {c.y:.2f}, {c.z:.2f}) | Rel=({dx:.2f}, {dy:.2f}, {dz:.2f})")
