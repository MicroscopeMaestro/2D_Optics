import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Mesh, Part

m5 = Mesh.Mesh('printing_model_5.stl')

# Let's check the clamping screw positions in printing_model_5.stl:
# Screws along X axis at:
# (-76, 136), (-76, 150), (-26, 122)
# In printing_model_5.stl, X is in [-17.5, 3.0].
# Let's test cylinders along X:
for y, z, label in [(-76.0, 136.0, "Screw 5 (DMD lateral 1)"), 
                    (-76.0, 150.0, "Screw 6 (DMD lateral 2)"), 
                    (-26.0, 122.0, "Screw 7 (MID CENTER ridge)")]:
    # Test line probe through hole center:
    p_center = FreeCAD.Vector(-8.0, y, z)
    # Check if there is a hole in printing_model_5.stl at this position
    # Let's count facets near this position
    near_facets = [f for f in m5.Facets if any((FreeCAD.Vector(p[0], p[1], p[2]) - p_center).Length < 5.0 for p in f.Points)]
    print(f"{label} at Y={y}, Z={z}: found {len(near_facets)} facets near center")
