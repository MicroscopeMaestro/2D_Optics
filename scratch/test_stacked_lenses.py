"""
Test stacked 2x f=60 doublets as condenser
"""

import sys, os, numpy as np, math
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def build_system(t_gap=1.0) -> SurfaceStack:
    stack = SurfaceStack()
    stack.primary_wavelength = 0.785

    # 0. Fiber
    stack.add_surface(Surface(name="Fiber", surface_type=SurfaceType.OBJECT, thickness=31.5, glass="AIR"))

    # 1. L1 (AC254-030-B)
    stack.add_surface(Surface(name="L1_F", surface_type=SurfaceType.STANDARD, radius=57.2, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L1_M", surface_type=SurfaceType.STANDARD, radius=14.8, thickness=7.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L1_B", surface_type=SurfaceType.STANDARD, radius=-18.5, thickness=30.0, glass="AIR"))

    # 2. Grating
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=30.0, glass="MIRROR", tilt_y=18.2, grating_lines_per_mm=600.0, grating_order=1))

    # 3. L2 (AC254-040-B)
    stack.add_surface(Surface(name="L2_F", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7"))
    stack.add_surface(Surface(name="L2_M", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L2_B", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=46.25, glass="AIR"))

    # 4. DMD
    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=187.0, glass="MIRROR", tilt_y=12.0))

    # 5. L3 (AC254-150-B)
    stack.add_surface(Surface(name="L3_F", surface_type=SurfaceType.STANDARD, radius=298.0, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L3_M", surface_type=SurfaceType.STANDARD, radius=66.8, thickness=4.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L3_B", surface_type=SurfaceType.STANDARD, radius=-92.5, thickness=25.0, glass="AIR"))

    # 6. L4a (1st stacked AC854-060-B)
    stack.add_surface(Surface(name="L4a_F", surface_type=SurfaceType.STANDARD, radius=39.48, thickness=4.0, glass="N-BK7"))
    stack.add_surface(Surface(name="L4a_M", surface_type=SurfaceType.STANDARD, radius=-33.00, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L4a_B", surface_type=SurfaceType.STANDARD, radius=-165.20, thickness=1.0, glass="AIR"))

    # 7. L4b (2nd stacked AC854-060-B)
    stack.add_surface(Surface(name="L4b_F", surface_type=SurfaceType.STANDARD, radius=39.48, thickness=4.0, glass="N-BK7"))
    stack.add_surface(Surface(name="L4b_M", surface_type=SurfaceType.STANDARD, radius=-33.00, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L4b_B", surface_type=SurfaceType.STANDARD, radius=-165.20, thickness=t_gap, glass="AIR"))

    # 8. Detector
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

    stack._update_accumulated_positions()
    return stack

MIN_GAP = 0.0
min_r = 1000.0
for t_gap in np.linspace(10, 30, 21):
    stack = build_system(t_gap)
    opt_sys = OpticalSystem(stack)
    rays = []
    zdist = 50.0
    theta_max = math.asin(0.10)
    for wl in [0.785, 0.880]:
        rays.append(Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=wl))
        for ring_frac in [0.5, 1.0]:
            r_pupil = zdist * math.tan(theta_max * ring_frac)
            npts = int(12 * ring_frac)
            for p in range(max(6, npts)):
                phi = 2.0 * np.pi * p / max(6, npts)
                dir_v = np.array([r_pupil * np.cos(phi), r_pupil * np.sin(phi), zdist])
                dir_v /= np.linalg.norm(dir_v)
                rays.append(Ray(pos=np.array([0.,0.,0.]), dir=dir_v, wavelength=wl))
    tr = opt_sys.trace_rays(rays)
    xs = []
    ys = []
    for r in tr:
        if len(r.history) > 17:
            xs.append(r.history[17]['x'])
            ys.append(r.history[17]['y'])
    if xs:
        r = max(np.max(np.abs(xs)), np.max(np.abs(ys)))
        if r < min_r:
            min_r = r
            MIN_GAP = t_gap
        print(f'Gap {t_gap:.2f} -> {r:.3f}')
print(f"\n\nBest Gap: {MIN_GAP} (MaxR: {1000*min_r:.2f} um)")
