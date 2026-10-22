import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scipy.optimize import minimize
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def build_spectrometer(
    f_coll=47.5,
    c_g=20.0,
    g_f=20.0,
    f_dmd=44.0,
    dmd_relay=20.0,
    relay_type='AC254-075-C',
    relay_rev=False,
    relay_cond=5.0,
    cond_det=5.2,
    primary_wl=1.3620,
    grating_tilt=12.0,
    dmd_tilt=12.0
):
    stack = SurfaceStack()
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=c_g, glass="AIR", semi_diameter=12.7))
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=g_f, glass="MIRROR", semi_diameter=12.7, tilt_y=grating_tilt, grating_lines_per_mm=300.0, grating_order=1))
    # 5-7. Focus lens AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    # 8. DMD4500
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_relay, glass="MIRROR", semi_diameter=12.7, tilt_y=dmd_tilt))
    # 9-11. Relay Lens (1-inch AC254 series)
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
    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det, glass="AIR", semi_diameter=12.7))
    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    return stack

# Generate rays once
test_rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Generated {len(test_rays)} test rays.")
sys.stdout.flush()

for relay_type in ['AC254-075-C', 'AC254-050-C']:
    for rev in [False, True]:
        print(f"\n--- Testing Relay {relay_type} (rev={rev}) ---")
        sys.stdout.flush()
        
        def objective(vars):
            dmd_relay, relay_cond, cond_det = vars
            stack = build_spectrometer(
                dmd_relay=dmd_relay,
                relay_type=relay_type,
                relay_rev=rev,
                relay_cond=relay_cond,
                cond_det=cond_det,
                primary_wl=1.3620
            )
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(test_rays)
            det_idx = len(stack.surfaces) - 1
            
            radii = []
            vig_count = 0
            for r in traced:
                if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx):
                    vig_count += 1
                else:
                    h = next((h for h in r.history if h["surface_index"] == det_idx), None)
                    if h:
                        rad = np.sqrt(h["x"]**2 + h["y"]**2)
                        radii.append(rad)
                    else:
                        vig_count += 1
            if not radii:
                return 1e6
            
            max_r = np.max(radii)
            rms_r = np.sqrt(np.mean(np.array(radii)**2))
            penalty = vig_count * 100.0
            return max_r + 0.5 * rms_r + penalty
        
        # Grid sample initial points
        best_x0 = None
        best_f0 = 1e6
        for dmd_r in [15.0, 25.0, 35.0, 50.0, 70.0]:
            for r_c in [5.0, 10.0, 20.0]:
                for c_d in [4.5, 5.0, 5.2, 5.5, 6.0]:
                    f = objective([dmd_r, r_c, c_d])
                    if f < best_f0:
                        best_f0 = f
                        best_x0 = [dmd_r, r_c, c_d]
                        print(f"  Init best: dmd_relay={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f} -> obj={f:.4f}")
                        sys.stdout.flush()
                        
        if best_x0:
            res = minimize(objective, best_x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.001})
            dmd_r, r_c, c_d = res.x
            stack = build_spectrometer(dmd_relay=dmd_r, relay_type=relay_type, relay_rev=rev, relay_cond=r_c, cond_det=c_d, primary_wl=1.3620)
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(test_rays)
            det_idx = len(stack.surfaces) - 1
            radii = [np.sqrt(h['x']**2 + h['y']**2) for r in traced for h in r.history if h['surface_index'] == det_idx]
            vig = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
            max_r = max(radii) if radii else 999.0
            print(f"  Optimized: dmd_relay={dmd_r:.3f}, r_cond={r_c:.3f}, cond_det={c_d:.3f} | vig={vig}/{len(test_rays)}, max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm)")
            sys.stdout.flush()
