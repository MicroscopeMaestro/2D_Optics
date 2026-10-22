import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part

shape = Part.Shape()
shape.read("3D_Raman_spectrometer_V2.step")
for f in shape.Faces:
    if isinstance(f.Surface, Part.Cylinder):
        if 1.0 < f.Surface.Radius < 2.5:
            c = f.Surface.Center
            print(f"R={f.Surface.Radius:.2f}, C=({c.x:.2f}, {c.y:.2f}, {c.z:.2f})")
