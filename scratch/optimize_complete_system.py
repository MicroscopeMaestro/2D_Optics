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

# Focus Lens Options (Camera Lens)
FOCUS_LENSES = {
    'AC254-050-C': {
        'forward': [
            {'name': 'AC254-050-C_Focus_Front', 'radius': 34.0, 'thickness': 6.0, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-050-C_Focus_Mid', 'radius': -27.8, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-050-C_Focus_Back', 'radius': -115.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ]
    },
    'AC254-040-C': {
        'forward': [
            {'name': 'AC254-040-C_Focus_Front', 'radius': 25.1, 'thickness': 6.8, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-040-C_Focus_Mid', 'radius': -20.2, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-040-C_Focus_Back', 'radius': -84.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ]
    }
}

def build_full_system(
    f_coll,
    c_g,
    g_f,
    focus_name,
    f_dmd,
    dmd_relay,
    relay_name,
    relay_orient,
    relay_cond,
    cond_det,
    primary_wl
):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=c_g, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=g_f, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    
    # Focus
    f_surfs = FOCUS_LENSES[focus_name]['forward']
    stack.add_surface(Surface(**f_surfs[0]))
    stack.add_surface(Surface(**f_surfs[1]))
    s2 = dict(f_surfs[2])
    s2['thickness'] = f_dmd
    stack.add_surface(Surface(**s2))
    
    # DMD
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_relay, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay
    r_surfs = DOUBLETS[relay_name][relay_orient]
    stack.add_surface(Surface(**r_surfs[0]))
    stack.add_surface(Surface(**r_surfs[1]))
    s5 = dict(r_surfs[2])
    s5['thickness'] = relay_cond
    stack.add_surface(Surface(**s5))
    
    # Condenser (ACL12708U-C)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det, glass="AIR", semi_diameter=12.7))
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total test rays: {len(rays)}")
sys.stdout.flush()

for focus_name in ['AC254-050-C', 'AC254-040-C']:
    for relay_name in ['AC254-075-C', 'AC254-050-C', 'AC254-030-C']:
        for relay_orient in ['forward', 'reversed']:
            for primary_wl in [1.3620, 1.3640]:
                for f_coll in [47.5, 55.0, 64.5]:
                    for f_dmd in [44.0, 50.0, 60.0, 64.5]:
                        for dmd_relay in [15.0, 20.0, 25.0, 30.0, 40.0]:
                            for relay_cond in [2.0, 5.0, 8.0, 10.0]:
                                for cond_det in [4.8, 5.0, 5.2, 5.4, 5.6]:
                                    stack = build_full_system(
                                        f_coll=f_coll,
                                        c_g=20.0,
                                        g_f=20.0,
                                        focus_name=focus_name,
                                        f_dmd=f_dmd,
                                        dmd_relay=dmd_relay,
                                        relay_name=relay_name,
                                        relay_orient=relay_orient,
                                        relay_cond=relay_cond,
                                        cond_det=cond_det,
                                        primary_wl=primary_wl
                                    )
                                    opt = OpticalSystem(stack)
                                    
                                    # Chief ray span on DMD
                                    r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
                                    r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
                                    x1 = r1.history[8]['x']
                                    x2 = r2.history[8]['x']
                                    span = abs(x2 - x1)
                                    
                                    if not (8.00 <= span <= 8.664):
                                        continue
                                    if not (abs(x1) <= 4.332 and abs(x2) <= 4.332):
                                        continue
                                        
                                    traced = opt.trace_rays(rays)
                                    det_idx = len(stack.surfaces) - 1
                                    
                                    vig = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                                    hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
                                    
                                    if vig == 0 and len(hits) == len(rays):
                                        radii = [np.sqrt(hx**2 + hy**2) for hx, hy in hits]
                                        max_r = max(radii)
                                        if max_r <= 0.500:
                                            print(f"\n=======================================================")
                                            print(f"*** STRICT PASS FOUND! ***")
                                            print(f"Focus: {focus_name} (f_dmd={f_dmd:.1f}), Relay: {relay_name} ({relay_orient}, dmd_r={dmd_relay:.1f}, r_c={relay_cond:.1f}, c_d={cond_det:.1f}), f_coll={f_coll:.1f}")
                                            print(f"Max Spot Radius: {max_r:.4f} mm (Max Spot Diameter: {max_r*2:.4f} mm <= 1.000 mm)")
                                            print(f"DMD Chief Ray Span: {span:.3f} mm [x1={x1:+.3f}, x2={x2:+.3f}]")
                                            print(f"Vignetting: 0 / {len(rays)} (100% transmission)")
                                            print(f"=======================================================")
                                            sys.stdout.flush()
