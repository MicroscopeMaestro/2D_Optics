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

def build_spectrometer(t0=64.57, t3=15.0, t4=10.0, t7=45.8, t8=2.0, t11=1.0, t13=4.85):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    surfaces = [
        # 0. Fiber Object
        Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=float(t0), glass="AIR", semi_diameter=25.0),
        # 1-3. Collimator AC254-050-C
        Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=float(t3), glass="AIR", semi_diameter=12.7),
        # 4. Grating GR25-0316
        Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=float(t4), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1),
        # 5-7. Focusing Doublet AC254-040-C
        Surface(name="AC254-040-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=float(t7), glass="AIR", semi_diameter=12.7),
        # 8. TI DMD4500
        Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(t8), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0),
        # 9-11. Relay Doublet AC508-075-C
        Surface(name="AC508-075-C_Relay_Front", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Mid", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Back", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=float(t11), glass="AIR", semi_diameter=25.4),
        # 12-13. Condenser ACL12708U-C
        Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]),
        Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(t13), glass="AIR", semi_diameter=12.7),
        # 14. InGaAs Detector
        Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.5)
    ]
    for s in surfaces:
        stack.add_surface(s)
    return stack

def main():
    stack = build_spectrometer()
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    opt_sys = OpticalSystem(stack)
    traced = opt_sys.trace_rays(rays)
    
    print(f"Traced {len(rays)} rays.")
    for i, s in enumerate(stack.surfaces):
        hits = [h for r in traced for h in r.history if h['surface_index'] == i]
        vigs = [h for h in hits if h['vignetted']]
        xs = [h['x'] for h in hits]
        ys = [h['y'] for h in hits]
        print(f"Surf {i:2d} ({s.name:25s}): hits={len(hits):3d}, x in [{min(xs):+.3f}, {max(xs):+.3f}], y in [{min(ys):+.3f}, {max(ys):+.3f}], vig={len(vigs)}")

if __name__ == "__main__":
    main()
