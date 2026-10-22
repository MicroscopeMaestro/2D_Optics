import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD

doc = FreeCAD.open('model1.FCStd')
optics = doc.getObject('Optical_System_Assembly').Group

print(f"{'Name':<35} | {'Size (X x Y x Z) mm':<30} | {'Global Z/Pos mm'}")
print("-" * 85)

for obj in optics:
    if not hasattr(obj, 'Shape') or obj.Shape.isNull(): continue
    
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox
    
    dx = bb.XMax - bb.XMin
    dy = bb.YMax - bb.YMin
    dz = bb.ZMax - bb.ZMin
    
    pos = obj.Placement.Base
    
    print(f"{obj.Label:<35} | {dx:5.2f} x {dy:5.2f} x {dz:5.2f} {'':<5} | Z: {pos.z:6.2f}  Y: {pos.y:6.2f}")

