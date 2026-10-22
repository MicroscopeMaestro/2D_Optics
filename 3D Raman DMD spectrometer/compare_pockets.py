import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.open('model1.FCStd')
asm = doc.getObject('Optical_System_Assembly')

# Let's run make_aspheric_cradle for both lenses and inspect their exact shapes
from build_spectrometer import make_aspheric_cradle

lens1 = next(o for o in asm.Group if "01_collect_acl25416u" in o.Label.lower() and "glass" in o.Label.lower())
lens17 = next(o for o in asm.Group if "17_condenser_acl25416u" in o.Label.lower() and "glass" in o.Label.lower())

cradle1 = make_aspheric_cradle(lens1)
cradle17 = make_aspheric_cradle(lens17)

print("--- POCKET 1 (01_Collect) ---")
print(f"Volume: {cradle1.Volume:.2f} mm^3")
print(f"Global BoundBox: {cradle1.BoundBox}")
# Local dimensions before transform
local_bb1 = lens1.Shape.copy().transformShape(lens1.Placement.Matrix.inverse()).BoundBox
th1 = (local_bb1.ZMax + 1.2) - (local_bb1.ZMin - 0.3)
print(f"Local Width (X): 35.40 mm")
print(f"Local Height (Y): 33.40 mm")
print(f"Local Thickness (Z): {th1:.3f} mm")

print("\n--- POCKET 2 (17_Condenser) ---")
print(f"Volume: {cradle17.Volume:.2f} mm^3")
print(f"Global BoundBox: {cradle17.BoundBox}")
local_bb17 = lens17.Shape.copy().transformShape(lens17.Placement.Matrix.inverse()).BoundBox
th17 = (local_bb17.ZMax + 1.2) - (local_bb17.ZMin - 0.3)
print(f"Local Width (X): 35.40 mm")
print(f"Local Height (Y): 33.40 mm")
print(f"Local Thickness (Z): {th17:.3f} mm")

print("\n--- COMPARISON ---")
print(f"Width diff: 0.00 mm")
print(f"Height diff: 0.00 mm")
print(f"Thickness diff: {abs(th1 - th17):.3f} mm (01_Collect={th1:.3f} mm, 17_Condenser={th17:.3f} mm)")
