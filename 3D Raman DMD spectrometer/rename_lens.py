import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD

doc = FreeCAD.open('model1.FCStd')
asm = doc.getObject('Optical_System_Assembly')
changed = False
for obj in asm.Group:
    if "040" in obj.Label:
        old = obj.Label
        new = old.replace("040", "050")
        obj.Label = new
        print(f"Renamed: {old} -> {new}")
        changed = True

if changed:
    doc.save()
    print("Saved model1.FCStd.")
else:
    print("No changes needed.")
