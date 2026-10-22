
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
sys_opt._update_accumulated_positions()
slit_pos = sys_opt.surfaces[6].global_origin
slit_R = sys_opt.surfaces[6].global_R

wls = [1.087, 1.225, 1.350]
vignettes = 0
total = 0

for wl in wls:
    for theta in np.linspace(0.001, np.arcsin(0.066), 5):
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            total += 1
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            dir_global = slit_R @ np.array([dx, dy, dz])
            
            current_ray = Ray(pos=slit_pos, dir=dir_global, wavelength=wl)
            for i in range(7, len(sys_opt.surfaces)):
                surf = sys_opt.surfaces[i]
                R_mat = surf.global_R
                origin = surf.global_origin
                pos_local = R_mat.T @ (current_ray.pos - origin)
                dir_local = R_mat.T @ current_ray.dir
                hit_local = np.array([0,0,0])
                if hasattr(sys_opt, '_intersect_surface'):
                    h = sys_opt._intersect_surface(surf, pos_local, dir_local)
                    if h is None:
                        vignettes += 1
                        break
                    hit_local = h
                
                hit_global = origin + R_mat @ hit_local
                current_ray.pos = hit_global
                
print(f'Vignetted: {vignettes} / {total}')

