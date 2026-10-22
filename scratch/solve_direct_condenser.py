import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

# Let's trace from DMD to a single focusing / condenser lens, or doublet + condenser
# Let's find the exact combination of Thorlabs lenses that gives spot diameter <= 1.0 mm (radius <= 0.50 mm)
# with 0% vignetting across all 315 rays!

def test_config(relay_name, cond_name, d1, d2, d3):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=47.5, glass="AIR", semi_diameter=25.0))
    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    # 5-7. Camera AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    # 8. DMD
    stack.add_surface(Surface(name="TI_DMD4500", radius=0.0, thickness=float(d1), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay Lens
    if relay_name == "AC254-050-C":
        stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=float(d2), glass="AIR", semi_diameter=12.7))
    elif relay_name == "AC254-040-C":
        stack.add_surface(Surface(name="AC254-040-C_Relay_Front", radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Relay_Mid", radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Relay_Back", radius=-84.0, thickness=float(d2), glass="AIR", semi_diameter=12.7))
    elif relay_name == "AC254-030-C":
        stack.add_surface(Surface(name="AC254-030-C_Relay_Front", radius=18.2, thickness=8.5, glass="N-LAK22", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Relay_Mid", radius=-15.4, thickness=2.0, glass="N-SF6", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Relay_Back", radius=-60.0, thickness=float(d2), glass="AIR", semi_diameter=12.7))
    elif relay_name == "AC508-075-C":
        stack.add_surface(Surface(name="AC508-075-C_Relay_Front", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Mid", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Back", radius=-170.0, thickness=float(d2), glass="AIR", semi_diameter=25.4))
        
    # Condenser / Collector
    if cond_name == "ACL12708U-C":
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=float(d3), glass="AIR", semi_diameter=12.7))
    elif cond_name == "ACL25416U-C":
        stack.add_surface(Surface(name="ACL25416U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=8.32, thickness=12.0, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.8e-6]))
        stack.add_surface(Surface(name="ACL25416U-C_Cond_Back", radius=0.0, thickness=float(d3), glass="AIR", semi_diameter=12.7))
    elif cond_name == "AC254-030-C":
        stack.add_surface(Surface(name="AC254-030-C_Cond_Front", radius=18.2, thickness=8.5, glass="N-LAK22", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond_Mid", radius=-15.4, thickness=2.0, glass="N-SF6", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond_Back", radius=-60.0, thickness=float(d3), glass="AIR", semi_diameter=12.7))
    elif cond_name == "None":
        # Direct focus onto detector! (Relay alone acts as collector!)
        pass

    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Rays ready: {len(rays)}")

# Let's test direct single-lens collector (e.g. AC254-030-C or AC254-040-C directly after DMD)
print("\n--- Testing Single-Doublet Collector directly after DMD ---")
for r_name in ["AC254-030-C", "AC254-040-C", "AC254-050-C"]:
    for d1 in [10.0, 15.0, 20.0, 25.0, 30.0]:
        for d_det in np.linspace(15.0, 45.0, 31):
            stack = test_config(r_name, "None", d1, d_det, 0.0)
            stack.surfaces[-1].semi_diameter = 10.0
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1]
            if len(vig_optics) == 0 and len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                if max_r < 0.50:
                    print(f"PASS Single Doublet: {r_name} | d1={d1:.1f}, d_det={d_det:.2f} -> max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm)")
