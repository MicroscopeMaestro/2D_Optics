import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import json
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def test_config(
    f_coll_dist=47.5,
    c_g_dist=20.0,
    g_f_dist=20.0,
    f_dmd_dist=44.0,
    dmd_relay_dist=20.0,
    relay_type='AC254-050-C',
    relay_reversed=False,
    relay_cond_dist=5.0,
    cond_det_dist=5.2,
    primary_wl=1.3620,
    grating_tilt=12.0,
    dmd_tilt=12.0,
    dense=False
):
    stack = SurfaceStack()
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll_dist, semi_diameter=25.0))
    
    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=c_g_dist, glass="AIR", semi_diameter=12.7))
    
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=g_f_dist, glass="MIRROR", semi_diameter=12.7, tilt_y=grating_tilt, grating_lines_per_mm=300.0, grating_order=1))
    
    # 5-7. Focus lens AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd_dist, glass="AIR", semi_diameter=12.7))
    
    # 8. DMD4500
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_relay_dist, glass="MIRROR", semi_diameter=12.7, tilt_y=dmd_tilt))
    
    # 9-11. Relay Lens
    if relay_type == 'AC254-050-C':
        if not relay_reversed:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-34.0, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))
    elif relay_type == 'AC254-075-C':
        if not relay_reversed:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=170.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=39.0, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-49.3, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))
    elif relay_type == 'AC254-030-C':
        if not relay_reversed:
            stack.add_surface(Surface(name="AC254-030-C_Relay_Front", radius=18.5, thickness=7.5, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-030-C_Relay_Mid", radius=-14.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-030-C_Relay_Back", radius=-57.2, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-030-C_Relay_Front", radius=57.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-030-C_Relay_Mid", radius=14.8, thickness=7.5, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-030-C_Relay_Back", radius=-18.5, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))

    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det_dist, glass="AIR", semi_diameter=12.7))
    
    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    
    opt = OpticalSystem(stack)
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM, dense_mode=dense)
    
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    vignetted = 0
    det_hits_x = []
    det_hits_y = []
    
    for r in traced:
        if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx):
            vignetted += 1
        else:
            for h in r.history:
                if h["surface_index"] == det_idx:
                    det_hits_x.append(h["x"])
                    det_hits_y.append(h["y"])
                    
    total = len(rays)
    if det_hits_x:
        r_arr = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
        max_r = float(np.max(r_arr))
    else:
        max_r = 999.0
        
    return stack, vignetted, total, max_r

print("Starting relay exploration...")
best = None
min_score = 999999

for relay_type in ['AC254-050-C', 'AC254-075-C', 'AC254-030-C']:
    for rev in [False, True]:
        for dmd_relay in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]:
            for r_cond in [5.0, 8.0, 10.0, 15.0, 20.0]:
                for c_det in [4.5, 5.0, 5.2, 5.5, 6.0, 6.5, 7.0, 8.0]:
                    try:
                        stack, vig, tot, max_r = test_config(
                            dmd_relay_dist=dmd_relay,
                            relay_type=relay_type,
                            relay_reversed=rev,
                            relay_cond_dist=r_cond,
                            cond_det_dist=c_det,
                            primary_wl=1.3620
                        )
                        score = vig * 1000 + max_r
                        if score < min_score:
                            min_score = score
                            best = (relay_type, rev, dmd_relay, r_cond, c_det, vig, tot, max_r)
                            print(f"NEW BEST: Relay={relay_type} (rev={rev}) | DMD->Relay={dmd_relay}, R->Cond={r_cond}, Cond->Det={c_det} | Vig={vig}/{tot}, Max_r={max_r:.4f} mm ({max_r*2:.4f} mm dia)")
                            sys.stdout.flush()
                    except Exception as e:
                        pass

print(f"\nFinal Best: {best}")
