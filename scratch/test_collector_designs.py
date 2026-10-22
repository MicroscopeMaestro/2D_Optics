import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

# Let's test different relay / condenser configurations
# 1. 75mm doublet + 8mm asphere (current)
# 2. 50mm doublet + 8mm asphere
# 3. 100mm doublet + 8mm asphere
# 4. 75mm doublet + 16mm asphere (ACL25416U)
# 5. Reversed 75mm doublet + 8mm asphere

def make_stack(relay_type="75mm", cond_type="8mm_asph", d_dmd_relay=20.0, d_relay_cond=5.0, d_cond_det=5.0):
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
    
    # Relay
    if relay_type == "75mm":
        stack.add_surface(Surface(name="AC508-075-C_Relay_Front", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Mid", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Back", radius=-170.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=25.4))
    elif relay_type == "75mm_rev":
        stack.add_surface(Surface(name="AC508-075-C_Relay_Front", radius=170.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Mid", radius=39.0, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Back", radius=-49.3, thickness=float(d_relay_cond), glass="AIR", semi_diameter=25.4))
    elif relay_type == "100mm":
        stack.add_surface(Surface(name="AC254-100-C_Relay_Front", radius=61.2, thickness=5.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-100-C_Relay_Mid", radius=-44.5, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-100-C_Relay_Back", radius=-198.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=12.7))
    elif relay_type == "50mm":
        stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=12.7))
        
    # Condenser
    if cond_type == "8mm_asph":
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
    elif cond_type == "16mm_asph":
        stack.add_surface(Surface(name="ACL25416U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=8.32, thickness=12.0, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.8e-6]))
        stack.add_surface(Surface(name="ACL25416U-C_Cond_Back", radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
    elif cond_type == "30mm_doublet":
        stack.add_surface(Surface(name="AC254-030-C_Cond_Front", radius=18.2, thickness=8.5, glass="N-LAK22", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond_Mid", radius=-15.4, thickness=2.0, glass="N-SF6", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond_Back", radius=-60.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
        
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=10.0))
    
    return stack

# Test all configurations with chief rays
chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in RAMAN_WAVELENGTHS_UM]

print("Testing relay & condenser configurations (chief rays):", flush=True)

for r_type in ["75mm", "75mm_rev", "50mm", "100mm"]:
    for c_type in ["8mm_asph", "16mm_asph", "30mm_doublet"]:
        for d1 in [10.0, 20.0, 30.0, 50.0, 70.0]:
            for d2 in [2.0, 5.0, 10.0]:
                for d3 in [4.0, 5.0, 6.0, 8.0, 12.0, 20.0]:
                    stack = make_stack(r_type, c_type, d1, d2, d3)
                    opt = OpticalSystem(stack)
                    traced = opt.trace_rays(chief_rays)
                    det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1]
                    if len(det_hits) == 7:
                        rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                        max_r = max(rs)
                        if max_r < 0.45:
                            print(f"PASS: {r_type:8s} + {c_type:12s} | d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.1f} -> max_r={max_r:.4f} mm (diam={2*max_r:.4f} mm)", flush=True)
