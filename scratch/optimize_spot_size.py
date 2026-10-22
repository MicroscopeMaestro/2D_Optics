import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scipy.optimize import minimize
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

# Catalog of 1-inch Thorlabs doublets
DOUBLETS = {
    'AC254-075-C': {
        'forward': [
            {'name': 'AC254-075-C_Relay_Front', 'radius': 49.3, 'thickness': 5.2, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-075-C_Relay_Mid', 'radius': -39.0, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-075-C_Relay_Back', 'radius': -170.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ],
        'reversed': [
            {'name': 'AC254-075-C_Relay_Front', 'radius': 170.0, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-075-C_Relay_Mid', 'radius': 39.0, 'thickness': 5.2, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-075-C_Relay_Back', 'radius': -49.3, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ]
    },
    'AC254-050-C': {
        'forward': [
            {'name': 'AC254-050-C_Relay_Front', 'radius': 34.0, 'thickness': 6.0, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-050-C_Relay_Mid', 'radius': -27.8, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-050-C_Relay_Back', 'radius': -115.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ],
        'reversed': [
            {'name': 'AC254-050-C_Relay_Front', 'radius': 115.0, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-050-C_Relay_Mid', 'radius': 27.8, 'thickness': 6.0, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-050-C_Relay_Back', 'radius': -34.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ]
    },
    'AC254-030-C': {
        'forward': [
            {'name': 'AC254-030-C_Relay_Front', 'radius': 18.5, 'thickness': 7.5, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-030-C_Relay_Mid', 'radius': -14.8, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-030-C_Relay_Back', 'radius': -57.2, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ],
        'reversed': [
            {'name': 'AC254-030-C_Relay_Front', 'radius': 57.2, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-030-C_Relay_Mid', 'radius': 14.8, 'thickness': 7.5, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-030-C_Relay_Back', 'radius': -18.5, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ]
    }
}

# Generate 315 test rays
test_rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

def build_system(dmd_relay, relay_name, orient, relay_cond, cond_det, primary_wl=1.3620, f_coll=47.5, f_dmd=44.0):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_relay, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay
    surfs = DOUBLETS[relay_name][orient]
    for s_dict in surfs[:-1]:
        stack.add_surface(Surface(**s_dict))
    last_s = dict(surfs[-1])
    last_s['thickness'] = relay_cond
    stack.add_surface(Surface(**last_s))
    
    # Condenser (ACL12708U-C) - semi_diameter 12.7 mm
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det, glass="AIR", semi_diameter=12.7))
    # Detector - semi_diameter 0.50 mm (1.0 mm diameter)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    return stack

print("Running Nelder-Mead optimization across relay types and orientations...")
sys.stdout.flush()

for relay_name in ['AC254-075-C', 'AC254-050-C', 'AC254-030-C']:
    for orient in ['forward', 'reversed']:
        print(f"\n--- Relay {relay_name} ({orient}) ---")
        sys.stdout.flush()
        
        def obj_fn(x):
            dmd_r, r_c, c_d = x
            if dmd_r < 5.0 or r_c < 1.0 or c_d < 1.0:
                return 1e6
            stack = build_system(dmd_r, relay_name, orient, r_c, c_d)
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(test_rays)
            det_idx = len(stack.surfaces) - 1
            
            # Count vignetted rays BEFORE detector
            vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
            hits = []
            for r in traced:
                if not (r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx)):
                    for h in r.history:
                        if h['surface_index'] == det_idx:
                            hits.append((h['x'], h['y']))
                            
            if len(hits) < len(test_rays) * 0.5:
                return 1e6 + vig_optics * 1000.0
                
            radii = [np.sqrt(hx**2 + hy**2) for hx, hy in hits]
            max_r = max(radii)
            rms_r = np.sqrt(np.mean(np.array(radii)**2))
            
            # Penalize rays outside 0.50 mm detector radius
            outside_penalty = sum(max(0.0, r - 0.50)**2 for r in radii) * 100.0
            
            return max_r + 0.5 * rms_r + vig_optics * 100.0 + outside_penalty
            
        for init_dmd_r in [15.0, 20.0, 25.0, 30.0, 40.0, 50.0]:
            for init_r_c in [2.0, 5.0, 10.0, 15.0]:
                for init_c_d in [4.5, 5.0, 5.2, 5.5]:
                    x0 = [init_dmd_r, init_r_c, init_c_d]
                    f0 = obj_fn(x0)
                    if f0 < 20.0:
                        res = minimize(obj_fn, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 0.01, 'fatol': 0.001})
                        opt_dmd_r, opt_r_c, opt_c_d = res.x
                        stack = build_system(opt_dmd_r, relay_name, orient, opt_r_c, opt_c_d)
                        opt = OpticalSystem(stack)
                        traced = opt.trace_rays(test_rays)
                        det_idx = len(stack.surfaces) - 1
                        det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx]
                        vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                        if det_hits:
                            radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                            max_r = max(radii)
                            rms_r = np.sqrt(np.mean(np.array(radii)**2))
                            print(f"  dmd_r={opt_dmd_r:.3f}, r_c={opt_r_c:.3f}, c_d={opt_c_d:.3f} -> vig_optics={vig_optics}, det_rays={len(det_hits)}/{len(test_rays)}, max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                            sys.stdout.flush()
