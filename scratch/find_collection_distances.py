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
    'AC254-100-C': {
        'forward': [
            {'name': 'AC254-100-C_Relay_Front', 'radius': 61.2, 'thickness': 5.0, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-100-C_Relay_Mid', 'radius': -44.5, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-100-C_Relay_Back', 'radius': -198.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ],
        'reversed': [
            {'name': 'AC254-100-C_Relay_Front', 'radius': 198.0, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-100-C_Relay_Mid', 'radius': 44.5, 'thickness': 5.0, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-100-C_Relay_Back', 'radius': -61.2, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
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

def make_stack(dmd_relay, relay_name, relay_orient, relay_cond, cond_det, f_coll=47.5, f_dmd=44.0, primary_wl=1.3620):
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
    surfs = DOUBLETS[relay_name][relay_orient]
    stack.add_surface(Surface(**surfs[0]))
    stack.add_surface(Surface(**surfs[1]))
    s_last = dict(surfs[2])
    s_last['thickness'] = relay_cond
    stack.add_surface(Surface(**s_last))
    
    # Condenser (ACL12708U-C) - 12.7 mm diameter (semi_diameter 12.7 mm in optical model)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det, glass="AIR", semi_diameter=12.7))
    # Detector (semi_diameter 0.50 mm)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    return stack

if __name__ == '__main__':
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Testing rays: {len(rays)}")
    
    # Fine-grid search over collection relay
    best_results = []
    
    for relay_name in ['AC254-075-C', 'AC254-050-C', 'AC254-100-C']:
        for orient in ['forward', 'reversed']:
            for dmd_r in [15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0]:
                for r_c in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
                    for c_d in [4.8, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5]:
                        stack = make_stack(dmd_r, relay_name, orient, r_c, c_d, f_coll=47.5, f_dmd=44.0, primary_wl=1.3620)
                        opt = OpticalSystem(stack)
                        traced = opt.trace_rays(rays)
                        det_idx = len(stack.surfaces) - 1
                        
                        vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                        if vig_optics > 0:
                            continue
                            
                        det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
                        if len(det_hits) == len(rays):
                            radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                            max_r = max(radii)
                            rms_r = np.sqrt(np.mean(np.array(radii)**2))
                            best_results.append((max_r, rms_r, relay_name, orient, dmd_r, r_c, c_d))
                            if max_r <= 0.500:
                                print(f"*** FOUND STRICT PASS *** Max_r = {max_r:.4f} mm (dia = {max_r*2:.4f} mm) | Relay: {relay_name} ({orient}) | dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f}")
                                sys.stdout.flush()

    best_results.sort(key=lambda x: x[0])
    print(f"\nTotal zero-vignetting configurations: {len(best_results)}")
    print("Top 10 configurations by spot radius:")
    for max_r, rms_r, rname, orient, dmd_r, r_c, c_d in best_results[:10]:
        print(f"  Max_r = {max_r:.4f} mm (dia = {max_r*2:.4f} mm), RMS_r = {rms_r:.4f} mm | Relay: {rname} ({orient}) | dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f}")
    sys.stdout.flush()
