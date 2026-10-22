import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Mesh, Part

m5 = Mesh.Mesh('printing_model_5.stl')

# Convert the mesh facets with X > 0 into a shape or wire to see where the tongues are:
# In printing_model_5.stl, the tongue has height 3.0 mm (X in [0, 3]).
# Let's slice printing_model_5.stl at X = 1.5 mm:
slice_poly = m5.slice(FreeCAD.Vector(1, 0, 0), 1.5)
print(f"Slice at X=1.5 count: {len(slice_poly)}")
for i, poly in enumerate(slice_poly):
    pts = poly.Points
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    print(f"  Tongue loop {i}: {len(pts)} points | Y=[{min(ys):.1f}, {max(ys):.1f}], Z=[{min(zs):.1f}, {max(zs):.1f}]")
