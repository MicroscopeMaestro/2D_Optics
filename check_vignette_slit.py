
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_75mm_stop4.json')
sys_opt._update_accumulated_positions()
slit_pos = sys_opt.surfaces[6].global_origin

wls = [1.087, 1.225, 1.350]
vignettes = 0
total = 0

# Test the NA = 0.066 cone emerging from the slit!
for wl in wls:
    for theta in np.linspace(0.001, np.arcsin(0.066), 10):
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            dir_global = sys_opt.surfaces[6].global_R @ np.array([dx, dy, dz])
            r = Ray(pos=slit_pos, dir=dir_global, wavelength=wl)
            # Start trace from surface 7 (Collimator S1)
            # Since trace_ray traces from 0, we'll just manually trace it.
            # Actually, just move the object to the slit!
            pass

# An easier way is just to see if ANY surface clips it in the trace.

