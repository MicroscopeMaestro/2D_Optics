import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import build_external_bed
build_external_bed.build_external_bed(output_step='external_aspheric_bed_13mm.step')
print("Done!")
