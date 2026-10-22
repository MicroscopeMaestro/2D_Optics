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

def build_arch_a(d1=20.0, d2=5.0, d3=20.0):
    # Arch A: Collimator AC254-050-C, Grating 300l/mm, Camera AC254-050-C, DMD, Relay AC254-050-C, Condenser AC254-030-C (1-inch doublets throughout!)
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=47.5, glass="AIR", semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", radius=0.0, thickness=float(d1), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=float(d2), glass="AIR", semi_diameter=12.7))
    
    # Condenser AC254-030-C (f=30mm NIR doublet, N-LAK22 / N-SF6)
    stack.add_surface(Surface(name="AC254-030-C_Cond_Front", radius=18.2, thickness=8.5, glass="N-LAK22", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-030-C_Cond_Mid", radius=-15.4, thickness=2.0, glass="N-SF6", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-030-C_Cond_Back", radius=-60.0, thickness=float(d3), glass="AIR", semi_diameter=12.7))
    
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    return stack

def build_arch_b(d1=10.0, d2=5.0, d3=5.0):
    # Arch B: Collimator AC254-050-C, Grating 300l/mm, Camera AC254-050-C, DMD, Relay AC254-050-C, Condenser ACL12708U-C
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=47.5, glass="AIR", semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", radius=0.0, thickness=float(d1), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=float(d2), glass="AIR", semi_diameter=12.7))
    
    # Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=float(d3), glass="AIR", semi_diameter=12.7))
    
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    return stack

if __name__ == '__main__':
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Loaded {len(rays)} phase-space rays.")
    
    print("\n--- Testing Architecture A (All 1-inch doublets: AC254-050-C + AC254-030-C) ---")
    for d1 in [18.0, 20.0, 22.0]:
        for d2 in [5.0, 10.0]:
            for d3 in [18.0, 20.0, 22.0]:
                stack = build_arch_a(d1, d2, d3)
                stack.surfaces[-1].semi_diameter = 10.0
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(rays)
                det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1 and not r.vignetted]
                vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
                if det_hits:
                    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                    max_r = max(rs)
                    print(f"Arch A: d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.1f} -> vig_optics={len(vig_optics)}, hits={len(det_hits)}/315, max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm)")

    print("\n--- Testing Architecture B (Relay AC254-050-C + Condenser ACL12708U-C) ---")
    for d1 in [8.0, 10.0, 12.0]:
        for d2 in [3.0, 5.0]:
            for d3 in [4.5, 4.8, 5.0]:
                stack = build_arch_b(d1, d2, d3)
                stack.surfaces[-1].semi_diameter = 10.0
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(rays)
                det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1 and not r.vignetted]
                vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
                if det_hits:
                    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                    max_r = max(rs)
                    print(f"Arch B: d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.1f} -> vig_optics={len(vig_optics)}, hits={len(det_hits)}/315, max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm)")
