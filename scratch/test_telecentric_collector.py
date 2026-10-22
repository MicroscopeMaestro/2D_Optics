import sys
import os
import math
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM, verify_spectrometer_layout

def test_telecentric_collector():
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Total rays to trace: {len(rays)}")
    
    # We want T0 ~ 64.57, T3 = 15.0, T4 = 10.0, T7 ~ 45.8
    # We sweep T8, T11, T13 to find where 100% of rays have det_r <= 0.50 and vig == 0
    t0 = 64.57
    t7 = 45.8
    
    for t8 in np.linspace(15.0, 45.0, 7):
        for t11 in np.linspace(2.0, 20.0, 10):
            for t13 in np.linspace(4.5, 5.5, 11):
                stack = SurfaceStack()
                stack.primary_wavelength = 1.38919
                surfaces = [
                    Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=t0, glass="AIR", semi_diameter=25.0),
                    Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7),
                    Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7),
                    Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=15.0, glass="AIR", semi_diameter=12.7),
                    Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=10.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1),
                    Surface(name="AC254-040-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7),
                    Surface(name="AC254-040-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7),
                    Surface(name="AC254-040-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=t7, glass="AIR", semi_diameter=12.7),
                    Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=t8, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0),
                    Surface(name="AC508-075-C_Relay_Front", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4),
                    Surface(name="AC508-075-C_Relay_Mid", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4),
                    Surface(name="AC508-075-C_Relay_Back", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=t11, glass="AIR", semi_diameter=25.4),
                    Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]),
                    Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=t13, glass="AIR", semi_diameter=12.7),
                    Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.5)
                ]
                for s in surfaces:
                    stack.add_surface(s)
                    
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(rays)
                
                vigs = sum(1 for r in traced if r.vignetted)
                det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14 and not h['vignetted']]
                
                if len(det_hits) == len(rays):
                    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                    mr = max(rs)
                    cx = np.mean([h['x'] for h in det_hits])
                    cy = np.mean([h['y'] for h in det_hits])
                    cent = np.sqrt(cx**2 + cy**2)
                    print(f"PERFECT! 315/315 unvignetted rays on detector: t8={t8:4.1f}, t11={t11:4.1f}, t13={t13:4.2f} -> max_det_r={mr:.4f} mm, cent={cent:.4f} mm")
                    return t8, t11, t13

if __name__ == "__main__":
    test_telecentric_collector()
