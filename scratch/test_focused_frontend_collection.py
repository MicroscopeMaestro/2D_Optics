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

FOCUS_LENSES = {
    'AC254-050-C': {'radius': [34.0, -27.8, -115.0], 'thickness': [6.0, 2.0], 'glass': ['N-BK7', 'SF5']},
    'AC254-040-C': {'radius': [25.1, -20.2, -84.0], 'thickness': [6.8, 2.0], 'glass': ['N-BK7', 'SF5']},
}

def build_spectrometer(f_coll, focus_name, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl=1.3620):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    
    # Focus
    f_spec = FOCUS_LENSES[focus_name]
    stack.add_surface(Surface(name=f"{focus_name}_Focus_Front", radius=f_spec['radius'][0], thickness=f_spec['thickness'][0], glass=f_spec['glass'][0], semi_diameter=12.7))
    stack.add_surface(Surface(name=f"{focus_name}_Focus_Mid", radius=f_spec['radius'][1], thickness=f_spec['thickness'][1], glass=f_spec['glass'][1], semi_diameter=12.7))
    stack.add_surface(Surface(name=f"{focus_name}_Focus_Back", radius=f_spec['radius'][2], thickness=f_dmd, glass="AIR", semi_diameter=12.7))
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
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
    
    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Generated {len(rays)} rays. Testing configs with focused front-ends...")
sys.stdout.flush()

test_cases = [
    # (f_coll, focus_name, f_dmd, pri_wl)
    (60.0, 'AC254-040-C', 48.0, 1.3620),
    (60.0, 'AC254-040-C', 50.0, 1.3620),
    (64.5, 'AC254-040-C', 48.0, 1.3620),
    (64.5, 'AC254-040-C', 50.0, 1.3620),
    (55.0, 'AC254-050-C', 50.0, 1.3620),
    (60.0, 'AC254-050-C', 55.0, 1.3620),
    (64.5, 'AC254-050-C', 64.5, 1.3620),
    (47.5, 'AC254-050-C', 44.0, 1.3620),
]

for f_coll, focus_name, f_dmd, pri_wl in test_cases:
    for relay_name in ['AC254-075-C', 'AC254-050-C', 'AC254-030-C']:
        for orient in ['forward', 'reversed']:
            for dmd_r in [15.0, 20.0, 25.0, 30.0]:
                for r_c in [3.0, 5.0, 8.0, 10.0]:
                    for c_d in [4.8, 5.0, 5.2, 5.4]:
                        stack = build_spectrometer(f_coll, focus_name, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl)
                        opt = OpticalSystem(stack)
                        
                        # Chief ray span
                        r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
                        r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
                        x1 = r1.history[8]['x']
                        x2 = r2.history[8]['x']
                        span = abs(x2 - x1)
                        if not (8.00 <= span <= 8.664 and abs(x1) <= 4.332 and abs(x2) <= 4.332):
                            continue
                            
                        traced = opt.trace_rays(rays)
                        det_idx = len(stack.surfaces) - 1
                        vig = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                        if vig > 0:
                            continue
                        det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
                        if len(det_hits) == len(rays):
                            radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                            max_r = max(radii)
                            rms_r = np.sqrt(np.mean(np.array(radii)**2))
                            if max_r <= 0.500:
                                print(f"*** STRICT PASS *** f_coll={f_coll}, Focus={focus_name} (f_dmd={f_dmd}), Relay={relay_name} ({orient}, dmd_r={dmd_r}, r_c={r_c}, c_d={c_d}) | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm, span={span:.3f} mm [x1={x1:+.3f}, x2={x2:+.3f}]")
                                sys.stdout.flush()
                            elif max_r <= 0.700:
                                print(f"  Close: f_coll={f_coll}, Focus={focus_name} (f_dmd={f_dmd}), Relay={relay_name} ({orient}, dmd_r={dmd_r}, r_c={r_c}, c_d={c_d}) | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm, span={span:.3f} mm")
                                sys.stdout.flush()

print("Search completed.")
sys.stdout.flush()
