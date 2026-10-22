import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("GENERATING COMPLEMENTARY TOP FOR DMD_support_top.stl")
print("=" * 80)

s_top = Part.read('support_top_half.step')

# The exact bounding box that generated DMD_support_top.stl:
# Y: [-86.0, -20.0] -> Length 66.0
# Z: [98.0, 166.0]  -> Height 68.0
# X: [-2.0, 25.0]   -> Width 27.0
# We add a 0.25mm negative clearance on all interior faces (Y and Z).
clearance = 0.25

# Enlarged cutter box
cutter_enl = Part.makeBox(
    27.0, 
    66.0 + 2*clearance, 
    68.0 + 2*clearance, 
    FreeCAD.Vector(-2.0, -86.0 - clearance, 98.0 - clearance)
)

rest_piece = s_top.cut(cutter_enl)
print(f"Original Top Volume: {s_top.Volume:.2f}")
print(f"Adapted Optics Region Volume: {rest_piece.Volume:.2f}")

out_step = "top_optics_region_adapted_to_dmd_support_top.step"
rest_piece.exportStep(out_step)
print(f"Exported {out_step}")

mesh_doc = FreeCAD.newDocument()
mesh_doc.addObject("Mesh::Feature", "MeshObj")
import MeshPart
mesh_doc.MeshObj.Mesh = MeshPart.meshFromShape(rest_piece, 0.01, 0.1)
out_stl = "top_optics_region_adapted_to_dmd_support_top.stl"
mesh_doc.MeshObj.Mesh.write(out_stl)
print(f"Exported {out_stl}")

desktop_dir = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
os.makedirs(desktop_dir, exist_ok=True)
shutil.copy(out_step, os.path.join(desktop_dir, out_step))
shutil.copy(out_stl, os.path.join(desktop_dir, out_stl))
print("Copied to Desktop.")
