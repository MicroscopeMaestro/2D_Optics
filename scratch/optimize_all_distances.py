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

# 35 fast rays for fast optimization
fast_rays = []
max_theta = np.arcsin(0.12)
rf = 0.0125
for wl in RAMAN_WAVELENGTHS_UM:
    fast_rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
    for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
        vx = np.sin(max_theta) * np.cos(phi)
        vy = np.sin(max_theta) * np.sin(phi)
        vz = np.cos(max_theta)
        sx = rf * np.cos(phi)
        sy = rf * np.sin(phi)
        fast_rays.append(Ray(pos=np.array([sx, sy, 0.]), dir=np.array([vx, vy, vz]), wavelength=wl))

# 315 full rays for verification
full_rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

def build_stack(f_coll, c_g, g_f, f_dmd, dmd_r, relay_name, orient, r_c, c_d, primary_wl):
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
    # Detector (semi_diameter=25.0 during optimization so unclipped rays are measured)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
    
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    return stack

if __name__ == '__main__':
    print("Starting fine optimization...")
    sys.stdout.flush()
    
    best_overall = None
    min_overall_val = 1e6
    
    for relay_name in ['AC254-075-C', 'AC254-050-C']:
        for orient in ['forward', 'reversed']:
            print(f"\nOptimization for Relay: {relay_name} ({orient})")
            sys.stdout.flush()
            
            def loss_function(params):
                f_coll, f_dmd, dmd_r, r_c, c_d, pri_wl = params
                
                # Physical bounds
                if not (40.0 <= f_coll <= 70.0): return 1e5
                if not (35.0 <= f_dmd <= 65.0): return 1e5
                if not (5.0 <= dmd_r <= 60.0): return 1e5
                if not (2.0 <= r_c <= 30.0): return 1e5
                if not (3.0 <= c_d <= 8.0): return 1e5
                if not (1.350 <= pri_wl <= 1.380): return 1e5
                
                stack = build_stack(f_coll, 20.0, 20.0, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl)
                opt = OpticalSystem(stack)
                
                # Check DMD span
                r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
                r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
                x1 = r1.history[8]['x']
                x2 = r2.history[8]['x']
                span = abs(x2 - x1)
                
                # Penalty if span not in [8.00, 8.66]
                span_penalty = 0.0
                if span < 8.05: span_penalty += (8.05 - span)**2 * 500.0
                if span > 8.60: span_penalty += (span - 8.60)**2 * 500.0
                
                # Penalty if x1 or x2 outside active DMD [-4.30, +4.30]
                if abs(x1) > 4.30: span_penalty += (abs(x1) - 4.30)**2 * 500.0
                if abs(x2) > 4.30: span_penalty += (abs(x2) - 4.30)**2 * 500.0
                
                # Ray trace fast rays
                traced = opt.trace_rays(fast_rays)
                det_idx = len(stack.surfaces) - 1
                
                vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx]
                
                if len(hits) < len(fast_rays):
                    return 1e5 + vig_optics * 100.0
                    
                radii = [np.sqrt(hx**2 + hy**2) for hx, hy in hits]
                max_r = max(radii)
                rms_r = np.sqrt(np.mean(np.array(radii)**2))
                
                # Penalty for any ray exceeding 0.450 mm radius (0.90 mm diameter)
                rad_penalty = sum(max(0.0, r - 0.450)**2 for r in radii) * 200.0
                
                return max_r + 0.5 * rms_r + span_penalty + rad_penalty + vig_optics * 500.0

            # Initial guesses
            guesses = [
                [47.5, 44.0, 20.0, 5.0, 5.2, 1.3620],
                [50.0, 45.0, 25.0, 5.0, 5.2, 1.3620],
                [55.0, 48.0, 20.0, 8.0, 5.1, 1.3620],
                [60.0, 50.0, 30.0, 5.0, 5.3, 1.3630],
                [64.5, 55.0, 20.0, 10.0, 5.2, 1.3620]
            ]
            
            for g in guesses:
                res = minimize(loss_function, g, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.001})
                f_coll, f_dmd, dmd_r, r_c, c_d, pri_wl = res.x
                
                # Verify with full rays
                stack = build_stack(f_coll, 20.0, 20.0, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl)
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(full_rays)
                det_idx = len(stack.surfaces) - 1
                
                vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx]
                
                r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
                r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
                x1 = r1.history[8]['x']
                x2 = r2.history[8]['x']
                span = abs(x2 - x1)
                
                if det_hits and vig_optics == 0:
                    radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                    max_r = max(radii)
                    rms_r = np.sqrt(np.mean(np.array(radii)**2))
                    print(f"  Result: f_coll={f_coll:.2f}, f_dmd={f_dmd:.2f}, dmd_r={dmd_r:.2f}, r_c={r_c:.2f}, c_d={c_d:.2f}, wl={pri_wl:.4f} | span={span:.3f} mm [x1={x1:+.3f}, x2={x2:+.3f}] | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                    sys.stdout.flush()
                    if max_r < min_overall_val:
                        min_overall_val = max_r
                        best_overall = (f_coll, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl, span, x1, x2, max_r, rms_r)

    print(f"\n=======================================================")
    print(f"BEST OVERALL DESIGN:")
    print(f"=======================================================")
    if best_overall:
        f_coll, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl, span, x1, x2, max_r, rms_r = best_overall
        print(f"Collimator distance (t0): {f_coll:.3f} mm")
        print(f"Camera Focus distance (t7): {f_dmd:.3f} mm")
        print(f"DMD to Relay distance (t8): {dmd_r:.3f} mm")
        print(f"Relay Lens: {relay_name} ({orient})")
        print(f"Relay to Condenser distance (t11): {r_c:.3f} mm")
        print(f"Condenser to Detector distance (t13): {c_d:.3f} mm")
        print(f"Primary Wavelength: {pri_wl:.5f} um")
        print(f"DMD Chief Ray Span: {span:.4f} mm (1.1113 um at {x1:+.3f} mm, 1.6133 um at {x2:+.3f} mm)")
        print(f"Max Spot Radius at Detector: {max_r:.4f} mm (Max Diameter: {max_r*2:.4f} mm)")
        print(f"RMS Spot Radius at Detector: {rms_r:.4f} mm")
    sys.stdout.flush()
