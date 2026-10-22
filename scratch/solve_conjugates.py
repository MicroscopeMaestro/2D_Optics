import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def evaluate_layout(
    f_coll_dist=47.5,
    c_g_dist=20.0,
    g_f_dist=20.0,
    f_dmd_dist=44.0,
    focus_type='AC254-050-C',
    dmd_relay_dist=72.0,
    relay_type='AC254-075-C',
    relay_reversed=False,
    relay_cond_dist=10.0,
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
    
    # 5-7. Focus lens
    if focus_type == 'AC254-050-C':
        stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd_dist, glass="AIR", semi_diameter=12.7))
    elif focus_type == 'AC254-040-C':
        stack.add_surface(Surface(name="AC254-040-C_Focus_Front", radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Focus_Mid", radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Focus_Back", radius=-84.0, thickness=f_dmd_dist, glass="AIR", semi_diameter=12.7))
        
    # 8. DMD4500
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_relay_dist, glass="MIRROR", semi_diameter=12.7, tilt_y=dmd_tilt))
    
    # 9-11. Relay Lens
    if relay_type == 'AC254-075-C':
        if not relay_reversed:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=170.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=39.0, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-49.3, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))
    elif relay_type == 'AC254-050-C':
        if not relay_reversed:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-34.0, thickness=relay_cond_dist, glass="AIR", semi_diameter=12.7))

    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det_dist, glass="AIR", semi_diameter=12.7))
    
    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    
    opt = OpticalSystem(stack)
    
    # 1. Chief ray metrics on DMD
    chief_rays = [
        Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113),
        Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133),
        Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.66708)
    ]
    tr_cr = opt.trace_rays(chief_rays)
    dmd_x1 = next(h['x'] for h in tr_cr[0].history if h['surface_index'] == 8)
    dmd_x2 = next(h['x'] for h in tr_cr[1].history if h['surface_index'] == 8)
    dmd_x3 = next(h['x'] for h in tr_cr[2].history if h['surface_index'] == 8)
    dmd_span = abs(dmd_x2 - dmd_x1)
    
    # 2. Phase-space ray trace
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM, dense_mode=dense)
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    vignetted = 0
    det_hits_x = []
    det_hits_y = []
    per_wl_max_r = {}
    
    for wl in RAMAN_WAVELENGTHS_UM:
        per_wl_max_r[wl] = 0.0
        
    for r in traced:
        wl = r.wavelength
        if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx):
            vignetted += 1
        else:
            for h in r.history:
                if h["surface_index"] == det_idx:
                    x, y = h["x"], h["y"]
                    det_hits_x.append(x)
                    det_hits_y.append(y)
                    rad = np.sqrt(x**2 + y**2)
                    if rad > per_wl_max_r[wl]:
                        per_wl_max_r[wl] = rad
                        
    total = len(rays)
    if det_hits_x:
        r_arr = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
        max_r = float(np.max(r_arr))
        mean_x = float(np.mean(det_hits_x))
        mean_y = float(np.mean(det_hits_y))
    else:
        max_r = 999.0
        mean_x, mean_y = 0.0, 0.0
        
    return {
        "stack": stack,
        "vignetted": vignetted,
        "total": total,
        "max_r": max_r,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "dmd_x1": dmd_x1,
        "dmd_x2": dmd_x2,
        "dmd_x3": dmd_x3,
        "dmd_span": dmd_span,
        "per_wl_max_r": per_wl_max_r
    }

# Test grid around 4f conjugate distances
print("Testing AC254-075-C and AC254-050-C with varying conjugates...")
for relay_type in ['AC254-075-C', 'AC254-050-C']:
    for rev in [False, True]:
        for dmd_relay in [35.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]:
            for r_cond in [5.0, 8.0, 10.0, 15.0, 20.0, 30.0]:
                for c_det in [4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]:
                    res = evaluate_layout(
                        dmd_relay_dist=dmd_relay,
                        relay_type=relay_type,
                        relay_reversed=rev,
                        relay_cond_dist=r_cond,
                        cond_det_dist=c_det,
                        primary_wl=1.3620
                    )
                    if res['vignetted'] == 0 and res['max_r'] < 1.0:
                        print(f"MATCH: Relay={relay_type} rev={rev} | dmd_relay={dmd_relay}, r_cond={r_cond}, c_det={c_det} | max_r={res['max_r']:.4f} mm (dia {res['max_r']*2:.4f} mm), DMD span={res['dmd_span']:.3f} mm [x1={res['dmd_x1']:.3f}, x2={res['dmd_x2']:.3f}]")
