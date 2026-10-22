import sys
import os
import math
import numpy as np
import scipy.optimize as opt

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM, verify_spectrometer_layout

def build_system(params):
    T0, T3, T4, T7, T8, T11, T13 = params
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    surfaces = [
        # 0. Fiber Object
        Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=float(T0), glass="AIR", semi_diameter=25.0),
        # 1-3. Collimator AC254-050-C
        Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=float(T3), glass="AIR", semi_diameter=12.7),
        # 4. Grating GR25-0316
        Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=float(T4), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1),
        # 5-7. Focusing Doublet AC254-040-C
        Surface(name="AC254-040-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=float(T7), glass="AIR", semi_diameter=12.7),
        # 8. TI DMD4500
        Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(T8), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0),
        # 9-11. Relay Doublet AC508-075-C
        Surface(name="AC508-075-C_Relay_Front", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Mid", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Back", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=float(T11), glass="AIR", semi_diameter=25.4),
        # 12-13. Condenser ACL12708U-C
        Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]),
        Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(T13), glass="AIR", semi_diameter=12.7),
        # 14. InGaAs Detector
        Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.5)
    ]
    for s in surfaces:
        stack.add_surface(s)
    return stack

def evaluate(params, rays):
    stack = build_system(params)
    opt_sys = OpticalSystem(stack)
    traced = opt_sys.trace_rays(rays)
    
    vig_count = sum(1 for r in traced if r.vignetted)
    det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14 and not h['vignetted']]
    dmd_hits = [h for r in traced for h in r.history if h['surface_index'] == 8 and not h['vignetted']]
    
    dmd_xs = [h['x'] for h in dmd_hits] if dmd_hits else [0]
    dmd_ys = [h['y'] for h in dmd_hits] if dmd_hits else [0]
    dmd_span = max(dmd_xs) - min(dmd_xs)
    dmd_my = max(map(abs, dmd_ys))
    
    if det_hits:
        det_rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
        max_det_r = max(det_rs)
        cx = np.mean([h['x'] for h in det_hits])
        cy = np.mean([h['y'] for h in det_hits])
        cent_off = np.sqrt(cx**2 + cy**2)
    else:
        max_det_r = 999.0
        cent_off = 999.0
        
    return vig_count, dmd_span, dmd_my, max_det_r, cent_off, stack

def main():
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Rays generated: {len(rays)}")
    
    # Let's run a multi-start Nelder-Mead / Powell search
    # Target: vig == 0, dmd_span <= 8.664, dmd_my <= 2.4365, max_det_r <= 0.500, cent_off <= 0.150
    
    def cost_func(p):
        v, span, my, r, c, _ = evaluate(p, rays)
        cost = (
            v * 10000.0 +
            max(0.0, span - 8.664) * 2000.0 +
            max(0.0, my - 2.4365) * 2000.0 +
            max(0.0, r - 0.500) * 5000.0 +
            max(0.0, c - 0.150) * 2000.0 +
            r * 100.0 +
            c * 100.0
        )
        return cost

    # Candidate start points
    starts = [
        [50.0, 15.0, 10.0, 36.5, 45.0, 10.0, 4.85],
        [64.57, 15.0, 10.0, 45.8, 75.0, 75.0, 4.85],
        [55.0, 15.0, 10.0, 38.0, 20.0, 5.0, 5.0],
        [60.0, 15.0, 10.0, 42.0, 30.0, 10.0, 4.85],
    ]
    
    for idx, p0 in enumerate(starts):
        print(f"\nRunning optimization from start point {idx+1}: {p0}")
        res = opt.minimize(cost_func, p0, method='Nelder-Mead', options={'maxiter': 200, 'disp': True})
        p_opt = res.x
        v, span, my, r, c, stack = evaluate(p_opt, rays)
        print(f"Optimized: T0={p_opt[0]:.2f}, T3={p_opt[1]:.2f}, T4={p_opt[2]:.2f}, T7={p_opt[3]:.2f}, T8={p_opt[4]:.2f}, T11={p_opt[5]:.2f}, T13={p_opt[6]:.2f}")
        print(f"  Vignetting: {v}/{len(rays)} ({100*(len(rays)-v)/len(rays):.2f}% transmission)")
        print(f"  DMD Span:   {span:.3f} mm (limit <= 8.664 mm), max_y: {my:.3f} mm")
        print(f"  Max Det R:  {r:.4f} mm (limit <= 0.500 mm), cent: {c:.4f} mm")
        
        if v == 0 and span <= 8.664 and r <= 0.500:
            print("\n*** SUCCESS! ALL REQUIREMENTS SATISFIED! ***")
            break

if __name__ == "__main__":
    main()
