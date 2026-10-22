import sys
import math
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD

doc = FreeCAD.open('model1.FCStd')
asm = doc.getObject('Optical_System_Assembly')

components = []
for obj in asm.Group:
    if not hasattr(obj, 'Placement'):
        continue
    components.append({
        'label': obj.Label,
        'pos': obj.Placement.Base,
        'rot': obj.Placement.Rotation
    })

components.sort(key=lambda c: c['label'])

print("Optical Path Sequence:")
for c in components:
    print(f"{c['label']:40} -> Z={c['pos'].z:.2f}, Y={c['pos'].y:.2f}, X={c['pos'].x:.2f}")
