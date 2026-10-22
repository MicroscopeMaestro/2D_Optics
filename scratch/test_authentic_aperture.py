import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def test_ray_bundle(dmd_relay, relay_type, relay_rev, relay_cond, cond_det, primary_wl=1.3620):
    stack = SurfaceStack()
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    # 5-7. Focus lens AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    # 8. DMD4500
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_relay, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    # 9-11. Relay Lens
    if relay_type == 'AC254-075-C':
        if not relay_rev:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=170.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=39.0, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-49.3, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_type == 'AC254-050-C':
        if not relay_rev:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-34.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_type == 'AC254-030-C':
        if not relay_rev:
            stack.add_surface(Surface(name="AC254-030-C_Relay_Front", radius=18.5, thickness=7.5, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-030-C_Relay_Mid", radius=-14.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-030-C_Relay_Back", radius=-57.2, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-030-C_Relay_Front", radius=57.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-030-C_Relay_Mid", radius=14.8, thickness=7.5, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-030-C_Relay_Back", radius=-18.5, thickness=relay_cond, glass="AIR", semi_diameter=12.7))

    # 12-13. Condenser ACL12708U-C (semi-diameter 6.35 mm for authentic 0.5" optic)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det, glass="AIR", semi_diameter=6.35))
    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)
    
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    vig = 0
    det_x, det_y = [], []
    for r in traced:
        if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx):
            vig += 1
        else:
            h = next((h for h in r.history if h["surface_index"] == det_idx), None)
            if h:
                det_x.append(h["x"])
                det_y.append(h["y"])
            else:
                vig += 1
                
    if det_x:
        r_arr = np.sqrt(np.array(det_x)**2 + np.array(det_y)**2)
        max_r = float(np.max(r_arr))
        mean_x = float(np.mean(det_x))
    else:
        max_r = 999.0
        mean_x = 0.0
        
    return vig, len(rays), max_r, mean_x

print("Evaluating relay configs with authentic 6.35 mm condenser aperture:")
for relay_type in ['AC254-050-C', 'AC254-075-C', 'AC254-030-C']:
    for rev in [False, True]:
        for dmd_relay in [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0]:
            for r_cond in [2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
                for c_det in [4.5, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]:
                    vig, tot, max_r, mean_x = test_ray_bundle(dmd_relay, relay_type, rev, r_cond, c_det)
                    if vig == 0:
                        print(f"100% TRANS: Relay={relay_type} (rev={rev}) | dmd_relay={dmd_relay:.1f}, r_cond={r_cond:.1f}, c_det={c_det:.1f} | max_r={max_r:.4f} mm ({max_r*2:.4f} mm dia), mean_x={mean_x:.4f} mm")
                        sys.stdout.flush()
