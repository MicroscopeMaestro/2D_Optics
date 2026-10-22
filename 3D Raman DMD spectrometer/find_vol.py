import sys
import glob
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

for f in glob.glob('*.step'):
    try:
        shape = Part.Shape()
        shape.read(f)
        if abs(shape.Volume - 45193) < 1000:
            print(f'Vol Match: {f} -> {shape.Volume}')
    except Exception as e:
        pass
