"""
Prototype and verification script for Spot Size + Target EFL optimization.
Tests optimizing a 50 mm achromatic doublet (N-BAF10 / SF10) for minimum RMS spot radius at EFL = 50.0 mm.
"""
import math
import numpy as np

from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.matrix_tracer import MatrixTracer
from optics_app.analysis.spot import SpotDiagram
from scipy.optimize import minimize

def test_prototype():
    # Achromatic doublet surfaces:
    # S0: Object at infinity
    # S1: Crown N-BAF10 front surface (R1 variable)
    # S2: Cemented contact surface (Flat / R2 variable)
    # S3: Flint SF10 rear surface (R3 variable)
    # S4: Image / Detector surface (thickness S3 -> S4 is back focal distance)
    
    s0 = Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR")
    s1 = Surface(name="S1", radius=25.0, thickness=4.0, glass="N-BAF10")
    s2 = Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10")
    s3 = Surface(name="S3", radius=-25.0, thickness=45.0, glass="AIR")
    s4 = Surface(name="IMA", radius=0.0, thickness=0.0, glass="AIR")

    stack = SurfaceStack()
    stack.surfaces = [s0, s1, s2, s3, s4]
    stack._update_accumulated_positions()

    target_efl = 50.0
    wavelengths = [0.4861, 0.5876, 0.6563] # F, d, C lines
    pupil_radius = 5.0 # 10 mm entrance pupil diameter (F/5)

    def calc_efl(stk):
        tracer = MatrixTracer()
        # Optical surfaces up to rear of lens (S1, S2, S3)
        mat = tracer.build_from_system(stk.surfaces[1:4], wavelength_um=0.5876)
        if abs(mat.C) > 1e-12:
            return -1.0 / mat.C, -mat.A / mat.C
        return 1e6, 1e6

    efl_init, bfl_init = calc_efl(stack)
    print(f"Initial EFL: {efl_init:.2f} mm, BFL: {bfl_init:.2f} mm")

    # We want to optimize: R1, R3, and back focal distance (thickness of S3)
    # With R2 kept flat (0.0) or variable
    # And constraint: EFL = 50.0 mm
    # Target: minimize RMS spot radius across F, d, C visible wavelengths!

    def trace_and_merit(x):
        r1, r3, t3 = x
        eval_stack = SurfaceStack()
        eval_stack.surfaces = [
            Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR"),
            Surface(name="S1", radius=r1, thickness=4.0, glass="N-BAF10"),
            Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10"),
            Surface(name="S3", radius=r3, thickness=t3, glass="AIR"),
            Surface(name="IMA", radius=0.0, thickness=0.0, glass="AIR"),
        ]
        eval_stack._update_accumulated_positions()
        
        cur_efl, cur_bfl = calc_efl(eval_stack)
        efl_err = abs(cur_efl - target_efl) / target_efl
        efl_penalty = 50.0 * (efl_err + 10.0 * efl_err**2)

        # Trace rays across wavelengths
        sys = OpticalSystem(eval_stack)
        rms_list = []
        total_rays = 0
        valid_rays = 0

        for wl in wavelengths:
            # Generate rays
            rays = []
            for r_norm in np.linspace(0.1, 1.0, 5):
                for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    px = r_norm * pupil_radius * np.cos(theta)
                    py = r_norm * pupil_radius * np.sin(theta)
                    from optics_app.engine.ray import Ray
                    rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            
            traced = sys.trace_rays(rays)
            total_rays += len(traced)
            wl_hits_x = []
            wl_hits_y = []
            for ray in traced:
                if not ray.vignetted:
                    for h in ray.history:
                        if h.get("surface_index") == 4:
                            wl_hits_x.append(h["x"])
                            wl_hits_y.append(h["y"])
                            valid_rays += 1
                            break
            if wl_hits_x:
                spot = SpotDiagram(wl_hits_x, wl_hits_y)
                rms_list.append(spot.calculate_rms_radius())

        # Anti-vignetting penalty
        trans_penalty = 0.0
        if total_rays > 0:
            loss_ratio = (total_rays - valid_rays) / total_rays
            if loss_ratio > 0.01:
                trans_penalty = 500.0 * loss_ratio + 1000.0 * loss_ratio**2

        spot_rms = float(np.mean(rms_list)) if rms_list else 100.0
        return spot_rms + efl_penalty + trans_penalty

    # Initial guess
    x0 = [22.0, -22.0, 45.0]
    res = minimize(trace_and_merit, x0, method="Nelder-Mead", options={"maxiter": 300})
    print(f"Optimization Success: {res.success}")
    opt_r1, opt_r3, opt_t3 = res.x
    print(f"Optimized R1: {opt_r1:.3f} mm, R3: {opt_r3:.3f} mm, Back Thickness: {opt_t3:.3f} mm")

    final_stack = SurfaceStack()
    final_stack.surfaces = [
        Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR"),
        Surface(name="S1", radius=opt_r1, thickness=4.0, glass="N-BAF10"),
        Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10"),
        Surface(name="S3", radius=opt_r3, thickness=opt_t3, glass="AIR"),
        Surface(name="IMA", radius=0.0, thickness=0.0, glass="AIR"),
    ]
    final_efl, final_bfl = calc_efl(final_stack)
    print(f"Final EFL: {final_efl:.3f} mm (Target: {target_efl:.1f} mm)")
    print(f"Final Back Focal Distance: {opt_t3:.3f} mm (Paraxial BFL: {final_bfl:.3f} mm)")
    print(f"Final RMS merit: {res.fun:.5f} mm ({res.fun*1000:.1f} um)")

if __name__ == "__main__":
    test_prototype()
