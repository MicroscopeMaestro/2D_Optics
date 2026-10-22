import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
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

def eval_candidate(f_coll, c_g, g_f, f_dmd, dmd_relay, relay_name, orient, relay_cond, cond_det, primary_wl=1.3620):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=c_g, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=g_f, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
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
    
    # Condenser (ACL12708U-C)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det, glass="AIR", semi_diameter=12.7))
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    
    opt = OpticalSystem(stack)
    
    # 1. Chief ray check on DMD
    r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
    r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
    dmd_x1 = r1.history[8]['x']
    dmd_x2 = r2.history[8]['x']
    dmd_span = abs(dmd_x2 - dmd_x1)
    
    # 2. Phase-space ray trace
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
    det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
    
    if det_hits:
        radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
        max_r = max(radii)
        rms_r = np.sqrt(np.mean(np.array(radii)**2))
        rays_inside_05 = sum(1 for r in radii if r <= 0.500)
    else:
        max_r = 999.0
        rms_r = 999.0
        rays_inside_05 = 0
        
    return {
        "dmd_span": dmd_span,
        "dmd_x1": dmd_x1,
        "dmd_x2": dmd_x2,
        "vig_optics": vig_optics,
        "det_rays": len(det_hits),
        "total_rays": len(rays),
        "max_r": max_r,
        "rms_r": rms_r,
        "rays_inside_05": rays_inside_05,
        "pct_inside_05": rays_inside_05 / len(rays) * 100.0
    }

print("Running parameter tests...")
for f_coll in [47.5, 50.0, 52.0, 55.0]:
    for f_dmd in [44.0, 46.0, 48.0, 50.0]:
        for relay_name in ['AC254-075-C', 'AC254-050-C']:
            for orient in ['forward', 'reversed']:
                for dmd_r in [15.0, 20.0, 25.0, 30.0]:
                    for r_c in [3.0, 5.0, 8.0, 10.0]:
                        for c_d in [4.8, 5.0, 5.2, 5.4]:
                            res = eval_candidate(f_coll, 20.0, 20.0, f_dmd, dmd_r, relay_name, orient, r_c, c_d)
                            if res['vig_optics'] == 0 and res['max_r'] <= 0.550 and 8.00 <= res['dmd_span'] <= 8.66:
                                print(f"CANDIDATE: f_coll={f_coll}, f_dmd={f_dmd}, Relay={relay_name} ({orient}), dmd_r={dmd_r}, r_c={r_c}, c_d={c_d} | max_r={res['max_r']:.4f} mm (dia {res['max_r']*2:.4f} mm), inside_05={res['pct_inside_05']:.1f}%, dmd_span={res['dmd_span']:.3f} mm [x1={res['dmd_x1']:.3f}, x2={res['dmd_x2']:.3f}]")
                                sys.stdout.flush()
