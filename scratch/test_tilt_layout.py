"""
Test 3D clearance and ray propagation with 30 deg fold geometry.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def test_30deg():
    stack = SurfaceStack()
    
    # 0. Object: Multimode Fiber (50 um core, NA = 0.22)
    stack.add_surface(Surface(
        name="Fiber Core Output",
        surface_type=SurfaceType.OBJECT,
        radius=0.0,
        thickness=25.7,
        glass="AIR",
        semi_diameter=0.025,
        thickness_variable=True,
        thickness_min=10.0,
        thickness_max=45.0
    ))
    
    # 1. Collimating Doublet L1 (Thorlabs AC127-030-B, f = 30.0 mm, 1/2" dia)
    stack.add_surface(Surface(
        name="L1_Front",
        surface_type=SurfaceType.STANDARD,
        radius=19.1,
        thickness=2.5,
        glass="SF5",
        semi_diameter=6.35
    ))
    stack.add_surface(Surface(
        name="L1_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-17.8,
        thickness=1.0,
        glass="N-BK7",
        semi_diameter=6.35
    ))
    stack.add_surface(Surface(
        name="L1_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-82.3,
        thickness=25.0, # Air gap L1 -> Grating
        glass="AIR",
        semi_diameter=6.35,
        thickness_variable=True,
        thickness_min=10.0,
        thickness_max=50.0
    ))
    
    # 2. Diffraction Grating (600 lines/mm, 30 deg tilt)
    stack.add_surface(Surface(
        name="Diffraction_Grating",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=35.0, # Air gap Grating -> L2
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=30.0,
        grating_lines_per_mm=600.0,
        grating_order=1,
        thickness_variable=True,
        thickness_min=15.0,
        thickness_max=60.0
    ))
    
    # 3. Focusing Doublet L2 (Thorlabs AC254-050-B, f = 50.0 mm, 1" dia)
    stack.add_surface(Surface(
        name="L2_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.0,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="L2_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="L2_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-115.0,
        thickness=45.0, # Air gap L2 -> DMD
        glass="AIR",
        semi_diameter=12.7,
        thickness_variable=True,
        thickness_min=25.0,
        thickness_max=65.0
    ))
    
    # 4. DLP4500 DMD (30 deg tilt)
    stack.add_surface(Surface(
        name="DMD4500_Mirror",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=45.0, # Air gap DMD -> L3
        glass="MIRROR",
        semi_diameter=10.0,
        tilt_x=30.0,
        thickness_variable=True,
        thickness_min=20.0,
        thickness_max=65.0
    ))
    
    # 5. Relay Collimator L3 (Thorlabs AC254-050-B, f = 50.0 mm, 1" dia)
    stack.add_surface(Surface(
        name="L3_Front",
        surface_type=SurfaceType.STANDARD,
        radius=115.0,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="L3_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=27.8,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="L3_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-34.0,
        thickness=20.0, # Air gap L3 -> L4
        glass="AIR",
        semi_diameter=12.7,
        thickness_variable=True,
        thickness_min=5.0,
        thickness_max=40.0
    ))
    
    # 6. Condenser L4 (Thorlabs ACL12708U Aspheric Condenser, f = 8.0 mm, NA = 0.78, 1/2" dia)
    stack.add_surface(Surface(
        name="L4_Condenser_Front",
        surface_type=SurfaceType.ASPHERIC,
        radius=4.16,
        thickness=4.5,
        glass="B270",
        semi_diameter=6.35,
        conic=-0.6265,
        asphere_coeffs=[1.5e-5]
    ))
    stack.add_surface(Surface(
        name="L4_Condenser_Back",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=5.5, # Air gap L4 -> Detector
        glass="AIR",
        semi_diameter=6.35,
        thickness_variable=True,
        thickness_min=2.0,
        thickness_max=12.0
    ))
    
    # 7. Detector: 1.0 mm x 1.0 mm
    stack.add_surface(Surface(
        name="Detector_1mm2",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.5,
        aperture_type="Rectangular",
        aperture_x_half=0.5,
        aperture_y_half=0.5
    ))
    
    stack.primary_wavelength = 0.850
    stack._update_accumulated_positions()
    
    print("SURFACE POSITIONS:")
    for i, s in enumerate(stack.surfaces):
        print(f"  [{i:02d}] {s.name:<22} Origin: ({s.global_origin[0]:7.2f}, {s.global_origin[1]:7.2f}, {s.global_origin[2]:7.2f}) | Thickness: {s.thickness:6.2f} | TiltX: {s.tilt_x:4.1f}")
        
    sys_opt = OpticalSystem(stack)
    for wl in [0.785, 0.850, 0.900, 0.950]:
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        traced = sys_opt.trace_ray(ray)
        print(f"\nWl {wl*1000:.1f} nm: Vignetted={traced.vignetted}, Interference={getattr(traced, 'interference', False)}, History Steps={len(traced.history)}/15")
        if len(traced.history) > 0:
            det_hit = traced.history[-1]
            print(f"   Final Hit: surf={det_hit['surface_index']} ({stack.surfaces[det_hit['surface_index']].name}), local=({det_hit['x']:+.3f}, {det_hit['y']:+.3f})")

if __name__ == "__main__":
    test_30deg()
