import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part

print("Loading 3D_Raman_spectrometer_V2.step...")
shape = Part.Shape()
shape.read("3D_Raman_spectrometer_V2.step")

print(f"Shape loaded. Volume: {shape.Volume}")
print(f"Bounding Box: {shape.BoundBox}")

print("Looking for cylinders (holes)...")
for f in shape.Faces:
    if isinstance(f.Surface, Part.Cylinder):
        if 1.0 < f.Surface.Radius < 2.5:
            center = f.Surface.Center
            axis = f.Surface.Axis
            # filter to holes aligned with Z axis, around Z=229
            if abs(axis.z) > 0.9 and 220 < center.z < 235:
                print(f"Hole R={f.Surface.Radius:.2f}, Center=({center.x:.2f}, {center.y:.2f}, {center.z:.2f})")
