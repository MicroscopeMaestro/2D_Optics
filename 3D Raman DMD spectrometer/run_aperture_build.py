import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part
import build_external_bed

doc = FreeCAD.open('model1.FCStd')
asm = doc.getObject('Optical_System_Assembly')
lens_obj = next(o for o in asm.Group if "01_collect_acl25416u" in o.Label.lower() and "glass" in o.Label.lower())
local_s = lens_obj.Shape.copy().transformShape(lens_obj.Placement.Matrix.inverse())

# Build and export standard default file
bed1 = build_external_bed.build_external_bed(output_step='external_aspheric_bed.step')

# Build and export dedicated named variation file
bed2 = build_external_bed.build_external_bed(output_step='external_aspheric_bed_10_5mm_aperture.step')

print(f"Bounding Box: {bed1.BoundBox}")
print(f"Volume: {bed1.Volume:.2f} mm^3")
print(f"Collision with lens: {bed1.common(local_s).Volume:.6f} mm^3")
