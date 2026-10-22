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

def create_system(T0, T3, T4, T7, T8, T11, T13, grat_tilt=12.0, dmd_tilt=12.0):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    surfaces = [
        # 0. Fiber
        Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=float(T0), glass="AIR", semi_diameter=25.0),
        # 1-3. Collimator AC254-050-C
        Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=float(T3), glass="AIR", semi_diameter=12.7),
        # 4. Grating GR25-0316
        Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=float(T4), glass="MIRROR", semi_diameter=12.7, tilt_y=float(grat_tilt), grating_lines_per_mm=300.0, grating_order=1),
        # 5-7. Focusing Doublet AC254-040-C
        Surface(name="AC254-040-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=float(T7), glass="AIR", semi_diameter=12.7),
        # 8. TI DMD4500
        Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(T8), glass="MIRROR", semi_diameter=12.7, tilt_y=float(dmd_tilt)),
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

def solve():
    # We want:
    # 1. 100% transmission across all 315 rays
    # 2. DMD span <= 8.664 mm
    # 3. Detector max spot radius <= 0.500 mm
    # 4. Detector centroid <= 0.150 mm
    
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Rays count: {len(rays)}")
    
    best_loss = 1e9
    best_p = None
    
    # Let's search over T0, T7, T8, T11, T13
    for T0 in np.linspace(46.0, 52.0, 7):
        for T7 in np.linspace(35.0, 42.0, 8):
            for T8 in np.linspace(10.0, 25.0, 4):
                for T11 in [1.0, 2.0, 4.0]:
                    for T13 in np.linspace(4.0, 5.5, 7):
                        stack = create_system(T0, 15.0, 10.0, T7, T8, T11, T13)
                        opt_sys = OpticalSystem(stack)
                        traced = opt_sys.trace_rays(rays)
                        
                        vig = sum(1 for r in traced if r.vignetted)
                        det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14 and not h['vignetted']]
                        dmd_hits = [h for r in traced for h in r.history if h['surface_index'] == 8 and not h['vignetted']]
                        
                        if len(det_hits) == len(rays): # 0% vignetting!
                            dmd_xs = [h['x'] for h in dmd_hits]
                            dmd_span = max(dmd_xs) - min(dmd_xs)
                            det_radii = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                            max_r = max(det_radii)
                            cx = np.mean([h['x'] for h in det_hits])
                            cy = np.mean([h['y'] for h in det_hits])
                            cent = np.sqrt(cx**2 + cy**2)
                            
                            loss = max_r + max(0, dmd_span - 8.664) * 10.0 + cent
                            if loss < best_loss:
                                best_loss = loss
                                best_p = (T0, T7, T8, T11, T13, dmd_span, max_r, cent)
                                print(f"NEW BEST: T0={T0:.2f}, T7={T7:.2f}, T8={T8:.2f}, T11={T11:.2f}, T13={T13:.2f} -> Vig=0/315, DMD={dmd_span:.3f} mm, Det_R={max_r:.4f} mm, Cent={cent:.4f} mm")

if __name__ == "__main__":
    solve()
