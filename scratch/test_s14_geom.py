import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.visualizer_3d import build_cylinder_geometry, get_effective_pose
import numpy as np

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
s14 = stack.surfaces[14]
s15 = stack.surfaces[15]

c14, R14 = get_effective_pose(s14)
c15, R15 = get_effective_pose(s15)

print("S14 pose center:", c14)
print("S14 R:\n", R14)
print("S14 thickness:", s14.thickness, "sd:", s14.semi_diameter)

print("S15 pose center:", c15)
print("S15 R:\n", R15)
print("S15 thickness:", s15.thickness, "sd:", s15.semi_diameter, "ap_x:", s15.aperture_x_half, "ap_y:", s15.aperture_y_half, "shape:", s15.substrate_shape)

f_mesh, r_mesh, s_mesh = build_cylinder_geometry(s14, max(s14.semi_diameter, s15.semi_diameter, 10.0), s14.thickness)
print("Front mesh Z range:", np.min(f_mesh[2]), np.max(f_mesh[2]))
print("Rear mesh Z range:", np.min(r_mesh[2]), np.max(r_mesh[2]))
