import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

def build_front_end(t0=47.5, t3=20.0, grat_tilt=12.0, t4=20.0, t7=44.0, primary_wl=1.3625):
    stack = SurfaceStack()
    stack.primary_wavelength = primary_wl
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=t0, semi_diameter=25.0))
    # 1-3. Collimator
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=t3, glass="AIR", semi_diameter=12.7))
    # 4. Grating
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, grating_lines_per_mm=300.0, tilt_y=grat_tilt, thickness=t4, glass="MIRROR", semi_diameter=12.7))
    # 5-7. Camera Focus
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=t7, glass="AIR", semi_diameter=12.7))
    # 8. DMD
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=12.7))

    stack._update_accumulated_positions()
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

print("Scanning t0 and t7 for minimum monochromatic spot blur on DMD...")
best = None
min_spot_size = 999.0

for t0 in np.linspace(44.0, 52.0, 17):
    for t7 in np.linspace(40.0, 50.0, 21):
        stack = build_front_end(t0=t0, t7=t7)
        opt = OpticalSystem(stack)
        traced = opt.trace_rays(rays)
        
        vig = any(r.vignetted and (not r.history or r.history[-1]["surface_index"] < 8) for r in traced)
        if vig:
            continue
            
        # Check spot size for center wavelength (1.38919 um)
        c_hits = [h for r in traced if abs(r.wavelength - 1.38919) < 1e-4 for h in r.history if h["surface_index"] == 8]
        if len(c_hits) == 45:
            xs = [h["x"] for h in c_hits]
            ys = [h["y"] for h in c_hits]
            cx, cy = np.mean(xs), np.mean(ys)
            spot_r = max(np.sqrt((x - cx)**2 + (y - cy)**2) for x, y in zip(xs, ys))
            if spot_r < min_spot_size:
                min_spot_size = spot_r
                best = (t0, t7, spot_r)
                print(f"t0={t0:.2f} mm, t7={t7:.2f} mm -> Monochromatic Spot Radius at DMD: {spot_r*1000.0:.1f} um ({spot_r:.4f} mm)")

print(f"\nBest Front End: t0={best[0]:.2f} mm, t7={best[1]:.2f} mm, Spot Radius={best[2]*1000.0:.1f} um")
