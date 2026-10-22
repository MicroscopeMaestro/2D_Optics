import sys
import os
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part

# Backup original model just in case
import shutil
shutil.copy2('model1.FCStd', 'model1_backup.FCStd')

doc = FreeCAD.openDocument('model1.FCStd')
doc.recompute()

# 1. 03_Collect_AC254-030-C_S1_Glass
obj3 = doc.getObject('Part__Feature002')
print(f"Modifying {obj3.Label} from thickness 9.0 to 9.5")
cyl3 = Part.makeCylinder(12.7, 9.5, FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1))
obj3.Shape = cyl3

# 2. 06_Collimator_AC254-030-C_S1_Glass
obj6 = doc.getObject('Part__Feature005')
print(f"Modifying {obj6.Label} from thickness 9.0 to 9.5")
cyl6 = Part.makeCylinder(12.7, 9.5, FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1))
obj6.Shape = cyl6

# 3. 08_Edge_Filter_1064_S1_Glass
obj8 = doc.getObject('Part__Feature007')
print(f"Modifying {obj8.Label} from thickness 3.5 to 3.0")
cyl8 = Part.makeCylinder(12.7, 3.0, FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1))
obj8.Shape = cyl8

# 4. 09_Edge_Filter_1064_S2_AirGap
obj9 = doc.getObject('Part__Feature008')
print(f"Modifying {obj9.Label} from length 11.5 to 12.0 and adjusting base")
cyl9 = Part.makeCylinder(12.7, 12.0, FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1))
obj9.Shape = cyl9
new_base = obj9.Placement.Base
new_base.z -= 0.5
obj9.Placement.Base = new_base

doc.recompute()
doc.save()
print("Saved model1.FCStd successfully.")
