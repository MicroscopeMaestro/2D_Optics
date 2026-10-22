import numpy as np
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.matrix_tracer import MatrixTracer
from optics_app.analysis.spot import SpotDiagram
from scipy.optimize import minimize

def test_paraxial_focus_solve():
    target_efl = 50.0
    wavelengths = [0.4861, 0.5876, 0.6563]
    pupil_radius = 5.0 # 10 mm diameter

    def calc_efl(stk):
        tracer = MatrixTracer()
        mat = tracer.build_from_system(stk.surfaces[1:4], wavelength_um=0.5876)
        if abs(mat.C) > 1e-12:
            return -1.0 / mat.C, -mat.A / mat.C
        return 1e6, 1e6

    def merit_with_focus_solve(x):
        r1, r3 = x
        eval_stack = SurfaceStack()
        eval_stack.surfaces = [
            Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR"),
            Surface(name="S1", radius=r1, thickness=4.0, glass="N-BAF10"),
            Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10"),
            Surface(name="S3", radius=r3, thickness=0.0, glass="AIR"), # thickness solved below
            Surface(name="IMA", radius=0.0, thickness=0.0, glass="AIR"),
        ]
        cur_efl, cur_bfl = calc_efl(eval_stack)
        if cur_bfl <= 5.0 or cur_bfl > 200.0 or cur_efl <= 0:
            return 1e5
        
        # Marginal ray solve: back thickness = paraxial BFL!
        eval_stack.surfaces[3].thickness = cur_bfl
        eval_stack._update_accumulated_positions()

        efl_err = abs(cur_efl - target_efl) / target_efl
        efl_penalty = 1000.0 * (efl_err + 10.0 * efl_err**2)

        sys = OpticalSystem(eval_stack)
        rms_list = []
        for wl in wavelengths:
            rays = []
            for r_norm in [0.2, 0.4, 0.6, 0.8, 1.0]:
                for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    px = r_norm * pupil_radius * np.cos(theta)
                    py = r_norm * pupil_radius * np.sin(theta)
                    from optics_app.engine.ray import Ray
                    rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            
            traced = sys.trace_rays(rays)
            wl_hits_x, wl_hits_y = [], []
            for ray in traced:
                if not ray.vignetted:
                    for h in ray.history:
                        if h.get("surface_index") == 4:
                            wl_hits_x.append(h["x"])
                            wl_hits_y.append(h["y"])
                            break
            if len(wl_hits_x) == len(rays):
                spot = SpotDiagram(wl_hits_x, wl_hits_y)
                rms_list.append(spot.calculate_rms_radius())
            else:
                return 1e5 # ray loss!

        spot_rms = float(np.mean(rms_list))
        return spot_rms + efl_penalty

    # Optimize with Nelder-Mead
    x0 = [25.0, -25.0]
    res = minimize(merit_with_focus_solve, x0, method="Nelder-Mead", options={"maxiter": 200})
    print(f"Success: {res.success}")
    opt_r1, opt_r3 = res.x
    
    stk = SurfaceStack()
    stk.surfaces = [
        Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR"),
        Surface(name="S1", radius=opt_r1, thickness=4.0, glass="N-BAF10"),
        Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10"),
        Surface(name="S3", radius=opt_r3, thickness=0.0, glass="AIR"),
        Surface(name="IMA", radius=0.0, thickness=0.0, glass="AIR"),
    ]
    efl, bfl = calc_efl(stk)
    stk.surfaces[3].thickness = bfl
    stk._update_accumulated_positions()
    print(f"Optimal R1 = {opt_r1:.3f} mm, R3 = {opt_r3:.3f} mm")
    print(f"System EFL = {efl:.3f} mm, BFL = {bfl:.3f} mm")

    # Trace rays to evaluate spot size
    sys = OpticalSystem(stk)
    for wl, name in [(0.4861, "F"), (0.5876, "d"), (0.6563, "C")]:
        rays = []
        for r_norm in [0.2, 0.4, 0.6, 0.8, 1.0]:
            for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
                px = r_norm * pupil_radius * np.cos(theta)
                py = r_norm * pupil_radius * np.sin(theta)
                from optics_app.engine.ray import Ray
                rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
        traced = sys.trace_rays(rays)
        hx = [h["x"] for r in traced if not r.vignetted for h in r.history if h.get("surface_index") == 4]
        hy = [h["y"] for r in traced if not r.vignetted for h in r.history if h.get("surface_index") == 4]
        spot = SpotDiagram(hx, hy)
        print(f"Wavelength {name} ({wl*1000:.1f} nm): RMS Spot Radius = {spot.calculate_rms_radius()*1000:.2f} um, Geo Radius = {spot.calculate_geo_radius()*1000:.2f} um")

if __name__ == "__main__":
    test_paraxial_focus_solve()
