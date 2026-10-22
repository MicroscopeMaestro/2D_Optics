import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

RELAY_CATALOG = {
    'AC254-030-C': {
        'forward': {'r1': 18.5, 't1': 7.5, 'g1': 'N-BK7', 'r2': -14.8, 't2': 2.0, 'g2': 'SF5', 'r3': -57.2},
        'reversed': {'r1': 57.2, 't1': 2.0, 'g1': 'SF5', 'r2': 14.8, 't2': 7.5, 'g2': 'N-BK7', 'r3': -18.5}
    },
    'AC254-040-C': {
        'forward': {'r1': 25.1, 't1': 6.8, 'g1': 'N-BK7', 'r2': -20.2, 't2': 2.0, 'g2': 'SF5', 'r3': -84.0},
        'reversed': {'r1': 84.0, 't1': 2.0, 'g1': 'SF5', 'r2': 20.2, 't2': 6.8, 'g2': 'N-BK7', 'r3': -25.1}
    },
    'AC254-050-C': {
        'forward': {'r1': 34.0, 't1': 6.0, 'g1': 'N-BK7', 'r2': -27.8, 't2': 2.0, 'g2': 'SF5', 'r3': -115.0},
        'reversed': {'r1': 115.0, 't1': 2.0, 'g1': 'SF5', 'r2': 27.8, 't2': 6.0, 'g2': 'N-BK7', 'r3': -34.0}
    },
}

def build_system(f_coll, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay
    p = RELAY_CATALOG[relay_name][orient]
    stack.add_surface(Surface(name=f"{relay_name}_Relay_Front", radius=p['r1'], thickness=p['t1'], glass=p['g1'], semi_diameter=12.7))
    stack.add_surface(Surface(name=f"{relay_name}_Relay_Mid", radius=p['r2'], thickness=p['t2'], glass=p['g2'], semi_diameter=12.7))
    stack.add_surface(Surface(name=f"{relay_name}_Relay_Back", radius=p['r3'], thickness=r_c, glass="AIR", semi_diameter=12.7))
    
    # Condenser (ACL12708U-C) - 12.7 mm diameter (semi_diameter=6.35 mm)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=6.35))
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Loaded {len(rays)} rays. Scanning short relay doublets...")
sys.stdout.flush()

for relay_name in ['AC254-030-C', 'AC254-040-C', 'AC254-050-C']:
    for orient in ['forward', 'reversed']:
        for pri_wl in [1.3620, 1.38919]:
            f_dmd_list = [44.0] if pri_wl == 1.3620 else [38.5, 39.0]
            for f_dmd in f_dmd_list:
                for dmd_r in [5.0, 8.0, 10.0, 12.0, 15.0]:
                    for r_c in [1.0, 2.0, 3.0, 5.0]:
                        for c_d in np.arange(4.0, 6.0, 0.2):
                            stack = build_system(47.5, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl)
                            opt = OpticalSystem(stack)
                            traced = opt.trace_rays(rays)
                            det_idx = len(stack.surfaces) - 1
                            
                            vig = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                            if vig > 0:
                                continue
                            hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx]
                            if len(hits) == len(rays):
                                rads = [np.sqrt(x**2 + y**2) for x, y in hits]
                                max_r = max(rads)
                                rms_r = np.sqrt(np.mean(np.array(rads)**2))
                                print(f"*** FOUND 100% TRANSMISSION *** {relay_name} ({orient}) | pri_wl={pri_wl:.4f}, f_dmd={f_dmd:.1f}, dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.2f} | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                                sys.stdout.flush()
