import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part

doc = FreeCAD.openDocument('model1.FCStd')

def set_global_cylinder(name, z_start, thickness):
    obj = doc.getObject(name)
    cyl = Part.makeCylinder(12.7, thickness, FreeCAD.Vector(0, 0, z_start), FreeCAD.Vector(0, 0, 1))
    obj.Shape = cyl
    obj.Placement = FreeCAD.Placement()
    print(f"Set {obj.Label} to Z={z_start}..{z_start+thickness}")

# 1. 03_Collect_AC254-030-C_S1_Glass (9.5mm)
set_global_cylinder('Part__Feature002', 53.36081153406, 9.5)

# 2. 06_Collimator_AC254-030-C_S1_Glass (9.5mm)
set_global_cylinder('Part__Feature005', 100.08261100687, 9.5)

# 3. 08_Edge_Filter_1064_S1_Glass (3.0mm)
set_global_cylinder('Part__Feature007', 119.08261100687, 3.0)

# 4. 09_Edge_Filter_1064_S2_AirGap (12.0mm, shifted back by 0.5)
set_global_cylinder('Part__Feature008', 122.08261100687, 12.0)

doc.recompute()
doc.save()
