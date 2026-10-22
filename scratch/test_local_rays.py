import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray, ray_surface_intersect

with open('fiber_spectrometer_portable.json', 'r') as f:
    data = json.load(f)

surfaces = []
for s_dict in data["surfaces"]:
    s = Surface()
    for k, v in s_dict.items():
        if hasattr(s, k):
            setattr(s, k, v)
    surfaces.append(s)

stack = SurfaceStack()
stack.surfaces = surfaces
stack.primary_wavelength = 0.85
stack._update_accumulated_positions()

system = OpticalSystem(stack)

for wl in [0.785, 0.85, 0.90]:
    print(f"\n================ Wavelength {wl} um ================")
    # Central ray
    r0 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
    
    traced_ray = r0.copy()
    for i, surf in enumerate(system.surfaces):
        R_mat = surf.global_R
        origin = surf.global_origin
        pos_local = R_mat.T @ (traced_ray.pos - origin)
        dir_local = R_mat.T @ traced_ray.dir
        dir_local /= np.linalg.norm(dir_local)
        
        local_ray = Ray(pos=pos_local, dir=dir_local, wavelength=traced_ray.wavelength)
        res = ray_surface_intersect(surf, local_ray)
        
        print(f"Surf {i:2d} ({surf.name:18s}): pos_local = [{pos_local[0]:.3f}, {pos_local[1]:.3f}, {pos_local[2]:.3f}], dir_local = [{dir_local[0]:.3f}, {dir_local[1]:.3f}, {dir_local[2]:.3f}] -> res is None? {res is None}")
        if res is not None:
            hit_local, norm_local = res
            hit_global = origin + R_mat @ hit_local
            print(f"       hit_local = [{hit_local[0]:.3f}, {hit_local[1]:.3f}, {hit_local[2]:.3f}], norm = [{norm_local[0]:.3f}, {norm_local[1]:.3f}, {norm_local[2]:.3f}]")
            # Step forward
            traced_ray = system.trace_ray(r0) # get accurate propagation
            break
