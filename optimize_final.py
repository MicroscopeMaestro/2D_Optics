
import sys, os, json, numpy as np
from scipy.optimize import minimize, differential_evolution
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_yz_75mm_halfinch.json', 'r') as f:
    data = json.load(f)

# Get indices
focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
focus_s3 = focus_indices[-1]

dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
cb_idx = dmd_idx + 1

relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
relay_s3 = relay_indices[-1]

cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1

# We will optimize:
# 1. t_focus (distance from Focusing to DMD)
# 2. t_dmd_relay (distance from DMD to Relay)
# 3. t_relay_cond (distance from Relay to Condenser)
# 4. t_cond_det (distance from Condenser to Detector)
# 5. dmd_tilt (if we want to tweak the 17 deg angle for exact Scheimpflug focus!)

def evaluate_system(x):
    t_focus, t_dmd_relay, t_relay_cond, t_cond_det = x
    
    data['surfaces'][focus_s3]['thickness'] = t_focus
    data['surfaces'][cb_idx]['thickness'] = t_dmd_relay
    data['surfaces'][relay_s3]['thickness'] = t_relay_cond
    data['surfaces'][cond_idx+1]['thickness'] = t_cond_det
    
    with open('temp_opt_final.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_opt_final.json')
    sys_opt._update_accumulated_positions()
    
    # We want to minimize the spot size on the detector for the central AND edge wavelengths!
    # Let's trace a small bundle of rays from the slit for 3 wavelengths.
    slit_pos = sys_opt.surfaces[6].global_origin
    slit_R = sys_opt.surfaces[6].global_R
    
    wls = [1.087, 1.225, 1.350]
    total_error = 0
    
    for wl in wls:
        # Trace a central ray and marginal rays (NA=0.066)
        r_center = Ray(pos=slit_pos, dir=slit_R @ np.array([0., 0., 1.]), wavelength=wl)
        
        # We must trace them manually from surface 7!
        rays = [r_center]
        for theta in [np.arcsin(0.06)]:
            for phi in [np.pi/2, -np.pi/2]: # Y-plane marginal rays
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r_marg = Ray(pos=slit_pos, dir=slit_R @ np.array([dx, dy, dz]), wavelength=wl)
                rays.append(r_marg)
                
        # Fast trace
        final_y = []
        dmd_y = []
        collimation_angles = []
        
        for ray in rays:
            for i in range(7, len(sys_opt.surfaces)):
                surf = sys_opt.surfaces[i]
                R_mat = surf.global_R
                origin = surf.global_origin
                pos_local = R_mat.T @ (ray.pos - origin)
                dir_local = R_mat.T @ ray.dir
                hit_local = sys_opt._intersect_surface(surf, pos_local, dir_local) if hasattr(sys_opt, '_intersect_surface') else None
                if hit_local is None:
                    # try plane intersection
                    t = -pos_local[2] / dir_local[2] if dir_local[2] != 0 else 0
                    if t < 0: return 1e6 # ray goes backward
                    hit_local = pos_local + t * dir_local
                    
                # We need proper refraction!
                # Since trace_ray already works from origin, let's just use trace_ray from origin with proper NA!
                pass

# Okay, trace_ray from origin is way easier. 
# Stop is F/7.5. But rays from origin to stop pass through lenses. 
# We just trace NA=0.01 rays, which won't vignette on the slit.

