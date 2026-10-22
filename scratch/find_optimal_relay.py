import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM

# Catalog of 1-inch Thorlabs doublets (prescriptions from Thorlabs optical catalog)
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
    'AC254-040-C': {
        'forward': [
            {'name': 'AC254-040-C_Relay_Front', 'radius': 25.1, 'thickness': 6.8, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-040-C_Relay_Mid', 'radius': -20.2, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-040-C_Relay_Back', 'radius': -84.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        ],
        'reversed': [
            {'name': 'AC254-040-C_Relay_Front', 'radius': 84.0, 'thickness': 2.0, 'glass': 'SF5', 'semi_diameter': 12.7},
            {'name': 'AC254-040-C_Relay_Mid', 'radius': 20.2, 'thickness': 6.8, 'glass': 'N-BK7', 'semi_diameter': 12.7},
            {'name': 'AC254-040-C_Relay_Back', 'radius': -25.1, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 12.7},
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
    }
}

def make_fast_rays():
    # 7 wavelengths x (center ray + 4 marginal rays) = 35 rays
    rays = []
    max_theta = np.arcsin(0.12)
    rf = 0.0125
    for wl in RAMAN_WAVELENGTHS_UM:
        # Chief ray
        rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
        # Marginal spatial/angular
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            vx = np.sin(max_theta) * np.cos(phi)
            vy = np.sin(max_theta) * np.sin(phi)
            vz = np.cos(max_theta)
            sx = rf * np.cos(phi)
            sy = rf * np.sin(phi)
            rays.append(Ray(pos=np.array([sx, sy, 0.]), dir=np.array([vx, vy, vz]), wavelength=wl))
    return rays

fast_rays = make_fast_rays()

def build_system(dmd_relay, relay_name, orient, relay_cond, cond_det, primary_wl=1.3620):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
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
    return stack

print("Fast scanning all 1-inch doublets and conjugates...")
sys.stdout.flush()

results = []
for relay_name in DOUBLETS:
    for orient in ['forward', 'reversed']:
        for dmd_relay in np.arange(10.0, 60.0, 5.0):
            for relay_cond in np.arange(2.0, 25.0, 3.0):
                for cond_det in np.arange(4.5, 6.5, 0.2):
                    stack = build_system(dmd_relay, relay_name, orient, relay_cond, cond_det)
                    opt = OpticalSystem(stack)
                    traced = opt.trace_rays(fast_rays)
                    det_idx = len(stack.surfaces) - 1
                    
                    vig = 0
                    det_hits = []
                    for r in traced:
                        if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx):
                            vig += 1
                        else:
                            h = next((h for h in r.history if h['surface_index'] == det_idx), None)
                            if h:
                                det_hits.append((h['x'], h['y']))
                            else:
                                vig += 1
                    if vig == 0 and det_hits:
                        radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                        max_r = max(radii)
                        results.append((max_r, relay_name, orient, dmd_relay, relay_cond, cond_det))

results.sort(key=lambda x: x[0])
print(f"\nTop 15 candidates out of {len(results)} zero-vignetting configurations:")
for max_r, rname, orient, dmd_r, r_cond, c_det in results[:15]:
    print(f"  Max_r = {max_r:.4f} mm (dia {max_r*2:.4f} mm) | Relay: {rname} ({orient}) | dmd_relay={dmd_r:.1f} mm, relay_cond={r_cond:.1f} mm, cond_det={c_det:.1f} mm")
sys.stdout.flush()
