
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_75mm_halfinch.json')
sys_opt._update_accumulated_positions()
slit_pos = sys_opt.surfaces[6].global_origin
slit_R = sys_opt.surfaces[6].global_R

wls = [1.087, 1.225, 1.350] # 1.225 is center, 1.087 and 1.350 are edges!
vignettes = 0
total = 0

for wl in wls:
    for theta in np.linspace(0.001, np.arcsin(0.066), 5):
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            total += 1
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            dir_global = slit_R @ np.array([dx, dy, dz])
            
            current_ray = Ray(pos=slit_pos, dir=dir_global, wavelength=wl)
            
            # fast manual trace to find vignetting
            vig = False
            for i in range(7, len(sys_opt.surfaces)):
                surf = sys_opt.surfaces[i]
                R_mat = surf.global_R
                origin = surf.global_origin
                pos_local = R_mat.T @ (current_ray.pos - origin)
                dir_local = R_mat.T @ current_ray.dir
                hit_local = sys_opt._intersect_surface(surf, pos_local, dir_local) if hasattr(sys_opt, '_intersect_surface') else None
                # Let's just use trace_ray properly by making a ray from Z=0 but overriding?
                # No, trace_ray doesn't have a start_surface param.
                pass

# Proper way: trace from 0 with the source NA!

