import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

def verify(filename):
    shape = Part.Shape()
    shape.read(filename)
    print(f"{filename}: Solids = {len(shape.Solids)}")
    return len(shape.Solids)

verify('support_top_half.step')
verify('support_bottom_half.step')
