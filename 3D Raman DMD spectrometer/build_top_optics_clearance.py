import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

print("=" * 80)
print("GENERATING TOP OPTICS REGION WITH 0.25MM CLEARANCE FOR PRINTED DMD PIECE")
print("=" * 80)

s_top = Part.read('support_top_half.step')

clearance = 0.25
box_dmd_enl = Part.makeBox(30.0, 75.0 + clearance*2, 35.0 + clearance*2, FreeCAD.Vector(-2.0, -110.0 - clearance, 90.0 - clearance))
box_flange_enl = Part.makeBox(30.0, 50.0 + clearance*2, 31.0 + clearance*2, FreeCAD.Vector(-2.0, -110.0 - clearance, 125.0 - clearance))
box_ridge_enl = Part.makeBox(30.0, 17.0 + clearance*2, 14.0 + clearance*2, FreeCAD.Vector(-2.0, -37.0 - clearance, 115.0 - clearance))
cutter_enlarged = box_dmd_enl.fuse(box_flange_enl).fuse(box_ridge_enl)

rest_piece = s_top.cut(cutter_enlarged)

print(f"Original Top Volume: {s_top.Volume:.2f}")
print(f"New Optics Region Volume: {rest_piece.Volume:.2f}")

out_step = "top_optics_region_clearance.step"
rest_piece.exportStep(out_step)
print(f"Exported {out_step}")

mesh_doc = FreeCAD.newDocument()
mesh_doc.addObject("Mesh::Feature", "MeshObj")
import MeshPart
mesh_doc.MeshObj.Mesh = MeshPart.meshFromShape(rest_piece, 0.01, 0.1)
out_stl = "top_optics_region_clearance.stl"
mesh_doc.MeshObj.Mesh.write(out_stl)
print(f"Exported {out_stl}")

desktop_dir = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
os.makedirs(desktop_dir, exist_ok=True)
shutil.copy(out_step, os.path.join(desktop_dir, out_step))
shutil.copy(out_stl, os.path.join(desktop_dir, out_stl))
print("Copied to Desktop.")
