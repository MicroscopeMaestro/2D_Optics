import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays, verify_spectrometer_layout

def make_full_system(relay_f=50, cond_type="8mm_asph", d_dmd_relay=20.0, d_relay_cond=5.0, d_cond_det=5.0):
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
    stack.add_surface(Surface(name="TI_DMD4500", radius=0.0, thickness=float(d_dmd_relay), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # 9-11. Relay Doublet
    if relay_f == 50:
        stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=12.7))
    elif relay_f == 75:
        stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=12.7))
    elif relay_f == 40:
        stack.add_surface(Surface(name="AC254-040-C_Relay_Front", radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Relay_Mid", radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Relay_Back", radius=-84.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=12.7))
        
    # 12-13. Condenser / Collector Lens
    if cond_type == "8mm_asph":
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
    elif cond_type == "16mm_asph":
        stack.add_surface(Surface(name="ACL25416U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=8.32, thickness=12.0, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.8e-6]))
        stack.add_surface(Surface(name="ACL25416U-C_Cond_Back", radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
    elif cond_type == "30mm_doublet":
        # Strictly Thorlabs 1-inch achromatic doublet AC254-030-C
        stack.add_surface(Surface(name="AC254-030-C_Cond_Front", radius=18.2, thickness=8.5, glass="N-LAK22", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond_Mid", radius=-15.4, thickness=2.0, glass="N-SF6", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond_Back", radius=-60.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
    elif cond_type == "40mm_doublet":
        stack.add_surface(Surface(name="AC254-040-C_Cond_Front", radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Cond_Mid", radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Cond_Back", radius=-84.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))

    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

print("Testing full 315-ray trace across candidate architectures:")
for rf in [50, 40, 75]:
    for ct in ["30mm_doublet", "40mm_doublet", "8mm_asph", "16mm_asph"]:
        for d1 in [10.0, 15.0, 20.0, 25.0, 30.0]:
            for d2 in [5.0, 10.0, 15.0]:
                for d3 in [5.0, 8.0, 12.0, 18.0, 22.0, 25.0, 28.0]:
                    stack = make_full_system(rf, ct, d1, d2, d3)
                    opt = OpticalSystem(stack)
                    traced = opt.trace_rays(rays)
                    
                    vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
                    if len(vig_optics) > 0:
                        continue
                        
                    det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1 and not r.vignetted]
                    if len(det_hits) == len(rays):
                        rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                        max_r = max(rs)
                        cx = np.mean([h['x'] for h in det_hits])
                        cy = np.mean([h['y'] for h in det_hits])
                        c_off = np.sqrt(cx**2 + cy**2)
                        if max_r <= 0.500:
                            print(f"PASS: Relay=AC254-{rf:03d}-C, Cond={ct:12s} | d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.1f} -> vig=0, max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm), cent_off={c_off:.4f}mm", flush=True)
