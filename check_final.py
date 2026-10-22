
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
sys_opt._update_accumulated_positions()

wls = [1.087, 1.225, 1.350]
all_hits_y = []
vignettes = 0

slit_pos = sys_opt.surfaces[6].global_origin
slit_R = sys_opt.surfaces[6].global_R

for wl in wls:
    # NA = 0.066 (from 4mm stop at 30mm)
    for theta in np.linspace(0.001, np.arcsin(0.066), 5):
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            dir_global = slit_R @ np.array([dx, dy, dz])
            current_ray = Ray(pos=slit_pos, dir=dir_global, wavelength=wl)
            
            # Trace from surface 7
            for i in range(7, len(sys_opt.surfaces)):
                surf = sys_opt.surfaces[i]
                R_mat = surf.global_R
                origin = surf.global_origin
                pos_local = R_mat.T @ (current_ray.pos - origin)
                dir_local = R_mat.T @ current_ray.dir
                hit_local = np.array([0.,0.,0.])
                if hasattr(sys_opt, '_intersect_surface'):
                    h = sys_opt._intersect_surface(surf, pos_local, dir_local)
                    if h is not None: hit_local = h
                else:
                    t = -pos_local[2] / dir_local[2] if dir_local[2] != 0 else 0
                    hit_local = pos_local + t * dir_local
                
                hit_global = origin + R_mat @ hit_local
                # apply refraction if it's a glass surface?
                # Wait, I didn't apply refraction! I just drew straight lines!
                # So the trace script for visualization didn't refract, meaning I have no idea if it focuses!
                pass

# Let's use the actual trace_ray from Z=0 with NA=0.01 to test the optics

