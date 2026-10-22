import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part

shape = Part.Shape()
shape.read("3D_Raman_spectrometer_V2.step")

print(f"Number of Solids: {len(shape.Solids)}")
for i, s in enumerate(shape.Solids):
    print(f"Solid {i}: CenterOfMass={s.CenterOfMass}, Volume={s.Volume}")
    print(f"  BoundBox: {s.BoundBox}")

