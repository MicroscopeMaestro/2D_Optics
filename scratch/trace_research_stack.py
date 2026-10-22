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

def build_research_stack(t_dmd_relay=25.0, t_relay_cond=10.0, t_cond_det=5.2, primary_wl=1.3625):
    stack = SurfaceStack()
    stack.primary_wavelength = primary_wl
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_Input_Slit", surface_type=SurfaceType.OBJECT, thickness=46.50, semi_diameter=25.0))
    # 1-3. Collimator
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Rear", radius=-115.0, thickness=40.0, glass="AIR", semi_diameter=12.7))
    # 4. Grating
    stack.add_surface(Surface(name="GR25-0316_Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=300.0, tilt_y=12.0, thickness=40.0, glass="MIRROR", semi_diameter=12.7))
    # 5-7. Camera Focus
    stack.add_surface(Surface(name="AC254-050-C_Foc_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Foc_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Foc_Rear", radius=-115.0, thickness=44.1, glass="AIR", semi_diameter=12.7))
    # 8. DMD
    stack.add_surface(Surface(name="TI_DLP4500_DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=t_dmd_relay, glass="MIRROR", tilt_y=12.0, semi_diameter=12.7))

    # 9-11. Relay AC254-075-C
    stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Rear", radius=-170.0, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))

    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, conic=-0.6265, asphere_coeffs=[1.5e-5], thickness=4.5, glass="B270", semi_diameter=12.7))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Rear", radius=0.0, thickness=t_cond_det, glass="AIR", semi_diameter=12.7))

    # 14. Detector
    stack.add_surface(Surface(name="InGaAs_Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))

    stack._update_accumulated_positions()
    return stack

if __name__ == "__main__":
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Tracing research stack with {len(rays)} rays...")
    stack = build_research_stack()
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    
    det_hits = [h for r in traced for h in r.history if h["surface_index"] == 14]
    vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < 14)]
    if det_hits:
        rs = [np.sqrt(h["x"]**2 + h["y"]**2) for h in det_hits]
        max_r = max(rs)
        cx = np.mean([h["x"] for h in det_hits])
        cy = np.mean([h["y"] for h in det_hits])
        print(f"Max Spot Radius: {max_r:.4f} mm (Diameter: {2*max_r:.4f} mm)")
        print(f"Hits: {len(det_hits)}/{len(rays)}, Vignetted Optics: {len(vig_optics)}")
        print(f"Centroid: ({cx:.4f}, {cy:.4f}) mm")
