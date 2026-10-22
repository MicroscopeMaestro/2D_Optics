import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

shape = Part.Shape()
shape.read('PDAPC3-Step.step')

bb = shape.BoundBox
print(f"Bounding Box:")
print(f"  X: {bb.XMin:.2f} to {bb.XMax:.2f} (Width: {bb.XLength:.2f})")
print(f"  Y: {bb.YMin:.2f} to {bb.YMax:.2f} (Height: {bb.YLength:.2f})")
print(f"  Z: {bb.ZMin:.2f} to {bb.ZMax:.2f} (Depth: {bb.ZLength:.2f})")

# Let's try to find cylinders to identify mounting holes
holes = []
for face in shape.Faces:
    if str(face.Surface.__class__.__name__) == 'Cylinder':
        r = face.Surface.Radius
        # M4 threading tap holes are usually ~1.65 to 2.0 radius, M3 is ~1.25 to 1.5, M2 is ~0.8 to 1.0, 
        # Clearance holes: M4 is ~2.2, M3 is ~1.7, M2 is ~1.1
        if 1.0 < r < 3.0: 
            center = face.Surface.Center
            axis = face.Surface.Axis
            holes.append((r, center, axis))

# Deduplicate holes (many faces might share the same cylinder center/radius)
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
    print(f"  Hole {idx+1}: Radius={r:.2f}, Center=({c.x:.2f}, {c.y:.2f}, {c.z:.2f}), Axis=({a.x:.1f}, {a.y:.1f}, {a.z:.1f})")

