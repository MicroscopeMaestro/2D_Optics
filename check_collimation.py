
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064.json')
sys_opt._update_accumulated_positions()

print('Tracing rays from Slit to Grating...')
for theta in [0.0, 0.02, 0.04]:
    # Start ray AT the slit (which is surface 6)
    # We will launch a ray from 0,0,0 (at object) and see its angle before Grating
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., np.sin(theta), np.cos(theta)]), wavelength=1.087)
    tr = sys_opt.trace_ray(r)
    
    # Grating is surface 11
    # We want the direction of the ray AFTER the Collimator (surface 9) but BEFORE the Grating
    # history[9] is the ray state AFTER Collimator_S3
    h = tr.history[9]
    dir_z = h['dir_z']
    dir_y = h['dir_y']
    angle = np.arcsin(dir_y) * 180 / np.pi
    print(f'Launch theta = {theta:.3f} rad -> Angle at Grating = {angle:.4f} degrees')

