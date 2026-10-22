import sys
FREECAD_BIN = r"C:\Program Files\FreeCAD 1.1\bin"
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)
import FreeCAD
import Part

bot_shape = Part.Shape()
bot_shape.read("support_bottom_half.step")

pos = FreeCAD.Vector(-15.0, -46.66740226283399, 233.897739998026)

f_cyl = None
for f in bot_shape.Faces:
    if type(f.Surface) == Part.Cylinder and abs(f.Surface.Radius - 1.65) < 0.05:
        v = FreeCAD.Vector(pos.x - f.Surface.Center.x, pos.y - f.Surface.Center.y, pos.z - f.Surface.Center.z)
        radial_dist = (v - f.Surface.Axis * v.dot(f.Surface.Axis)).Length
        axial_offset = abs(v.dot(f.Surface.Axis))
        if radial_dist < 0.2 and axial_offset < 15.0:
            f_cyl = f
            break

print("Found cylinder face:", f_cyl is not None)
edges = f_cyl.Edges
non_adj = [f for f in bot_shape.Faces if f != f_cyl and not any(e1.isSame(e2) for e1 in edges for e2 in f.Edges)]

# Find closest face
min_d = float('inf')
closest_f = None
for i, f in enumerate(non_adj):
    d, pts, _ = f_cyl.distToShape(f)
    if d < min_d:
        min_d = d
        closest_f = (i, f, pts)

print(f"Closest face distance: {min_d:.6f} mm")
print(f"Closest face type: {type(closest_f[1].Surface)}")
print(f"Closest face BB: {closest_f[1].BoundBox}")
print(f"Closest points: {closest_f[2]}")
