import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Mesh

m5 = Mesh.Mesh('printing_model_5.stl')

# Find all points in printing_model_5.stl near Y=-76 and Z in [130, 160]:
pts_near = []
for f in m5.Facets:
    for p in f.Points:
        if abs(p[1] - (-76.0)) < 10.0 and 130 <= p[2] <= 160:
            pts_near.append(p)

print(f"Points near Y=-76, Z in [130, 160]: {len(pts_near)}")
if pts_near:
    xs = [p[0] for p in pts_near]
    ys = [p[1] for p in pts_near]
    zs = [p[2] for p in pts_near]
    print(f"X range: [{min(xs):.1f}, {max(xs):.1f}]")
    print(f"Y range: [{min(ys):.1f}, {max(ys):.1f}]")
    print(f"Z range: [{min(zs):.1f}, {max(zs):.1f}]")
