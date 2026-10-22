import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays, verify_spectrometer_layout

def build_100_8_system(d_dmd_relay=95.0, d_relay_cond=10.0, d_cond_det=4.85):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    # 0. Fiber (25 um core, NA=0.12)
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=47.5, glass="AIR", semi_diameter=25.0))
    # 1-3. Collimator AC254-050-C (f=50mm, Ø25.4mm)
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    # 4. Grating GR25-0316 (300 l/mm, Gold, Blaze 1600nm, alpha=12.0 deg)
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    # 5-7. Camera AC254-050-C (f=50mm, Ø25.4mm)
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    # 8. DMD TI_DMD4500 (8.664mm active width, tilt_y=12.0 deg)
    stack.add_surface(Surface(name="TI_DMD4500", radius=0.0, thickness=float(d_dmd_relay), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    # 9-11. Relay Collimator AC254-100-C (f=100mm, Ø25.4mm, R1=61.2, R2=-44.5, R3=-198.0)
    stack.add_surface(Surface(name="AC254-100-C_Relay_Front", radius=61.2, thickness=5.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-100-C_Relay_Mid", radius=-44.5, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-100-C_Relay_Back", radius=-198.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=12.7))
    # 12-13. Condenser ACL12708U-C (f=8.0mm, NA=0.78 Asphere, R=4.16, k=-0.6265)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
    # 14. Detector InGaAs 1mm
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    return stack

if __name__ == '__main__':
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Testing AC254-100-C + ACL12708U-C with {len(rays)} rays...")
    
    for d1 in np.linspace(85.0, 105.0, 5):
        for d2 in [5.0, 10.0, 15.0]:
            for d3 in np.linspace(4.5, 5.2, 8):
                stack = build_100_8_system(d1, d2, d3)
                stack.surfaces[-1].semi_diameter = 10.0
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(rays)
                
                vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
                det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1]
                
                if len(vig_optics) == 0 and len(det_hits) == len(rays):
                    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                    max_r = max(rs)
                    print(f"d1={d1:5.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> vig=0, hits=315/315, max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm)")
