import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Mesh, Part

m5 = Mesh.Mesh('printing_model_5.stl')
print(f"printing_model_5.stl BBox: X=[{m5.BoundBox.XMin:.2f}, {m5.BoundBox.XMax:.2f}], Y=[{m5.BoundBox.YMin:.2f}, {m5.BoundBox.YMax:.2f}], Z=[{m5.BoundBox.ZMin:.2f}, {m5.BoundBox.ZMax:.2f}]")

# 1. The body of printing_model_5.stl is at X <= 0:
# 2. The tongues of printing_model_5.stl are at X > 0:
facets_tongue = [f for f in m5.Facets if any(p[0] > 0.001 for p in f.Points)]
print(f"Total facets with X > 0 (tongue features): {len(facets_tongue)}")

# Let's inspect all points of these facets:
tongue_pts = [p for f in facets_tongue for p in f.Points if p[0] > 0.001]
print(f"Tongue X range: [{min(p[0] for p in tongue_pts):.2f}, {max(p[0] for p in tongue_pts):.2f}]")
print(f"Tongue Y range: [{min(p[1] for p in tongue_pts):.2f}, {max(p[1] for p in tongue_pts):.2f}]")
print(f"Tongue Z range: [{min(p[2] for p in tongue_pts):.2f}, {max(p[2] for p in tongue_pts):.2f}]")

# Let's check support_top_half_new.step:
s_new = Part.read('support_top_half_new.step')
print(f"\nsupport_top_half_new.step BBox: {s_new.BoundBox}")
print(f"support_top_half_new.step Volume: {s_new.Volume:.2f}")

# Where are the features in s_new with X < 0?
lip_below_0 = s_new.common(Part.makeBox(10.0, 500.0, 500.0, FreeCAD.Vector(-10.0, -250.0, -50.0)))
print(f"Volume of s_new below X=0 (the mirrored tongues): {lip_below_0.Volume:.2f} mm^3")
print(f"BBox of s_new below X=0: {lip_below_0.BoundBox}")
