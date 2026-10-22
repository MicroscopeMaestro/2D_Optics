
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064.json')
sys_opt._update_accumulated_positions()

# Slit is at z = 81.775
slit_z = sys_opt.surfaces[6].global_origin[2]
print(f'Slit Z: {slit_z}')

for theta in [0.0, 0.02, 0.04]:
    r = Ray(pos=np.array([0., 0., slit_z]), dir=np.array([0., np.sin(theta), np.cos(theta)]), wavelength=1.087)
    tr = sys_opt.trace_ray(r)
    
    # We want the ray state BEFORE it hits the Grating. The Grating is index 11.
    # So we want the history entry just before Grating. 
    # Let's just find the first history entry where z > 115 (after collimator)
    for h in tr.history:
        if h.get('z_global', 0) > 115 and h.get('z_global',0) < 180:
            dir_y = h['dir_y']
            angle = np.arcsin(dir_y) * 180 / np.pi
            print(f'Launch theta = {theta:.3f} rad from SLIT -> Angle at Grating = {angle:.4f} degrees')
            break

