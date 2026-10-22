import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part

doc = FreeCAD.openDocument('model1.FCStd')
optics = doc.getObject('Optical_System_Assembly')
lens = next(obj for obj in optics.Group if "15_Relay_AC254-100-C_S1_Glass" in obj.Label)
gap = next(obj for obj in optics.Group if "16_Relay_AC254-100-C_S3_AirGap" in obj.Label)

# 1. Replace Lens Shape with 6.5mm cylinder
old_placement = lens.Placement
# The radius is probably 12.7. Let's check old radius.
bb = lens.Shape.copy().transformShape(old_placement.Matrix.inverse()).BoundBox
radius = bb.XMax
new_shape = Part.makeCylinder(radius, 6.5)
lens.Shape = new_shape
lens.Placement = old_placement

# 2. Shift AirGap Base by +0.5mm along its local Z
z_vec = gap.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
gap.Placement.Base = gap.Placement.Base + z_vec * 0.5

doc.save()
print("Successfully updated 100mm lens to 6.5mm and shifted the airgap.")
