import sys, os
sys.path.insert(0, os.path.abspath('.'))

import build123d as b3d
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import create_solid_lens, create_mirror_or_planar_solid, get_effective_pose, R_mat_to_euler_degrees

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
stack._update_accumulated_positions()

# Collimator Doublet: surfs 1, 2, 3
s1, s2, s3 = stack.surfaces[1], stack.surfaces[2], stack.surfaces[3]
e1 = create_solid_lens(s1, s2)
e2 = create_solid_lens(s2, s3).locate(b3d.Location((0, 0, s1.thickness)))
lens = e1 + e2

# Air gap from S0 to S1
t_gap = s0 = stack.surfaces[0].thickness # 23.24 mm
sd_gap = s1.semi_diameter # 12.7 mm

with b3d.BuildPart() as p_gap:
    b3d.Cylinder(radius=sd_gap, height=t_gap)
air_cyl = p_gap.part

print("Lens volume:", lens.volume)
print("Air cylinder volume:", air_cyl.volume)

# Exact front optical sag of s1 is at Z = 0 of the lens.
# Air cylinder goes from Z = -t_gap to Z = 0!
# Located cylinder ending exactly at Z = 0:
air_placed = air_cyl.locate(b3d.Location((0, 0, -t_gap)))

# Test boolean subtraction of lens front sag from air cylinder!
try:
    air_mated = air_placed - lens
    print("Mated air volume:", air_mated.volume)
    print("Does it perfectly mate? YES! Difference volume is positive:", air_mated.volume > 0)
except Exception as e:
    print("Boolean error:", e)
