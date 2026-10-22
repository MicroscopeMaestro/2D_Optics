import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scipy.optimize import minimize
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

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
    }
}

# 14 fast rays for super fast optimization (chief ray + NA=0.12 marginal ray at each of the 7 wavelengths)
fast_rays = []
max_theta = np.arcsin(0.12)
rf = 0.0125
for wl in RAMAN_WAVELENGTHS_UM:
    fast_rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
    fast_rays.append(Ray(pos=np.array([rf, 0., 0.]), dir=np.array([np.sin(max_theta), 0., np.cos(max_theta)]), wavelength=wl))
    fast_rays.append(Ray(pos=np.array([-rf, 0., 0.]), dir=np.array([-np.sin(max_theta), 0., np.cos(max_theta)]), wavelength=wl))
    fast_rays.append(Ray(pos=np.array([0., rf, 0.]), dir=np.array([0., np.sin(max_theta), np.cos(max_theta)]), wavelength=wl))
    fast_rays.append(Ray(pos=np.array([0., -rf, 0.]), dir=np.array([0., -np.sin(max_theta), np.cos(max_theta)]), wavelength=wl))

# 315 full rays for verification
full_rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

def build_spectrometer(f_coll, c_g, g_f, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=c_g, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=g_f, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay
    surfs = DOUBLETS[relay_name][orient]
    stack.add_surface(Surface(**surfs[0]))
    stack.add_surface(Surface(**surfs[1]))
    s_last = dict(surfs[2])
    s_last['thickness'] = r_c
    stack.add_surface(Surface(**s_last))
    
    # Condenser (ACL12708U-C)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
    # Detector (semi_diameter=25.0)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
    
    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

if __name__ == '__main__':
    print(f"Fast rays: {len(fast_rays)}, Full rays: {len(full_rays)}")
    sys.stdout.flush()
    
    best_results = []
    
    for relay_name in ['AC254-075-C', 'AC254-050-C']:
        for orient in ['forward', 'reversed']:
            print(f"\n=======================================================")
            print(f"Optimizing for Relay: {relay_name} ({orient})")
            print(f"=======================================================")
            sys.stdout.flush()
            
            def objective(params):
                dmd_r, r_c, c_d = params
                if dmd_r < 5.0 or dmd_r > 70.0 or r_c < 1.0 or r_c > 30.0 or c_d < 3.0 or c_d > 8.0:
                    return 1e6
                stack = build_spectrometer(
                    f_coll=47.5, c_g=20.0, g_f=20.0, f_dmd=44.0,
                    dmd_r=dmd_r, relay_name=relay_name, orient=orient, r_c=r_c, c_d=c_d, pri_wl=1.3620
                )
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(fast_rays)
                det_idx = len(stack.surfaces) - 1
                
                vig_count = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx]
                
                if len(hits) < len(fast_rays):
                    return 1e5 + vig_count * 100.0
                    
                radii = [np.sqrt(x**2 + y**2) for x, y in hits]
                max_r = max(radii)
                rms_r = np.sqrt(np.mean(np.array(radii)**2))
                
                # Penalty for spot radius exceeding 0.450 mm
                rad_penalty = sum(max(0.0, r - 0.450)**2 for r in radii) * 100.0
                return max_r + 0.5 * rms_r + rad_penalty + vig_count * 500.0
                
            for init_dmd_r in [15.0, 20.0, 25.0, 30.0, 40.0]:
                for init_r_c in [2.0, 5.0, 10.0]:
                    for init_c_d in [5.0, 5.2, 5.4]:
                        x0 = [init_dmd_r, init_r_c, init_c_d]
                        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 150, 'xatol': 0.01, 'fatol': 0.001})
                        opt_dmd_r, opt_r_c, opt_c_d = res.x
                        
                        # Full ray trace verification
                        stack = build_spectrometer(
                            f_coll=47.5, c_g=20.0, g_f=20.0, f_dmd=44.0,
                            dmd_r=opt_dmd_r, relay_name=relay_name, orient=orient, r_c=opt_r_c, c_d=opt_c_d, pri_wl=1.3620
                        )
                        opt = OpticalSystem(stack)
                        traced = opt.trace_rays(full_rays)
                        det_idx = len(stack.surfaces) - 1
                        
                        vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                        hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx]
                        
                        if vig_optics == 0 and len(hits) == len(full_rays):
                            radii = [np.sqrt(x**2 + y**2) for x, y in hits]
                            max_r = max(radii)
                            rms_r = np.sqrt(np.mean(np.array(radii)**2))
                            inside_05 = sum(1 for r in radii if r <= 0.500)
                            print(f"  Result: dmd_r={opt_dmd_r:.3f}, r_c={opt_r_c:.3f}, c_d={opt_c_d:.3f} | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm, inside_05={inside_05}/{len(full_rays)}")
                            sys.stdout.flush()
                            best_results.append((max_r, rms_r, relay_name, orient, opt_dmd_r, opt_r_c, opt_c_d))

    best_results.sort(key=lambda x: x[0])
    print(f"\n=======================================================")
    print(f"Top 5 Overall Results:")
    for max_r, rms_r, rname, orient, dmd_r, r_c, c_d in best_results[:5]:
        print(f"  Max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), RMS_r={rms_r:.4f} mm | {rname} ({orient}) | dmd_r={dmd_r:.3f}, r_c={r_c:.3f}, c_d={c_d:.3f}")
    sys.stdout.flush()
