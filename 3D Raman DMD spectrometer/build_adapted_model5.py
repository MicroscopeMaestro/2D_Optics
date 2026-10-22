import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("GENERATING COMPLEMENTARY TOP FOR MODEL 5 BOTTOM")
print("=" * 80)

# Base model is the Model 5 top (matches printing_model5.stl bottom)
s_top = Part.read(r'C:\Users\Juan\Desktop\Spectrometer_Print_Files\support_top_half_model5.step')

clearance = 0.25

# Enlarged cutter box matching the Y[-86, -20] Z[98, 166] cutout
# Note: Model 5 top extends to X=-3.0, so the cutter must cover that!
cutter_enl = Part.makeBox(
    30.0, 
    66.0 + 2*clearance, 
    68.0 + 2*clearance, 
    FreeCAD.Vector(-5.0, -86.0 - clearance, 98.0 - clearance)
)

rest_piece = s_top.cut(cutter_enl)
print(f"Original Model5 Top Volume: {s_top.Volume:.2f}")
print(f"Adapted Optics Region Volume: {rest_piece.Volume:.2f}")

out_step = "top_optics_region_model5_adapted.step"
rest_piece.exportStep(out_step)
print(f"Exported {out_step}")

mesh_doc = FreeCAD.newDocument()
mesh_doc.addObject("Mesh::Feature", "MeshObj")
import MeshPart
mesh_doc.MeshObj.Mesh = MeshPart.meshFromShape(rest_piece, 0.01, 0.1)
out_stl = "top_optics_region_model5_adapted.stl"
mesh_doc.MeshObj.Mesh.write(out_stl)
print(f"Exported {out_stl}")

desktop_dir = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
os.makedirs(desktop_dir, exist_ok=True)
shutil.copy(out_step, os.path.join(desktop_dir, out_step))
shutil.copy(out_stl, os.path.join(desktop_dir, out_stl))
print("Copied to Desktop.")
