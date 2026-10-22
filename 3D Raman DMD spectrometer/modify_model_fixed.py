import sys
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part

# Restore from backup
shutil.copy2('model1_backup.FCStd', 'model1.FCStd')

doc = FreeCAD.openDocument('model1.FCStd')
doc.recompute()

def update_lens(name, new_thickness):
    obj = doc.getObject(name)
    print(f"Modifying {obj.Label} to thickness {new_thickness}")
    # Save original placement
    orig_placement = obj.Placement
    
    # Create new cylinder
    cyl = Part.makeCylinder(12.7, new_thickness, FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1))
    
    # Assign shape and restore placement
    obj.Shape = cyl
    obj.Placement = orig_placement

update_lens('Part__Feature002', 9.5)
update_lens('Part__Feature005', 9.5)
update_lens('Part__Feature007', 3.0)

# AirGap
obj9 = doc.getObject('Part__Feature008')
print(f"Modifying {obj9.Label} from length 11.5 to 12.0 and adjusting base")
orig_placement9 = obj9.Placement
cyl9 = Part.makeCylinder(12.7, 12.0, FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1))
obj9.Shape = cyl9
orig_placement9.Base.z -= 0.5
obj9.Placement = orig_placement9

doc.recompute()
doc.save()
print("Saved model1.FCStd successfully.")
