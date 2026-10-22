"""
Analyze DMD dispersion width and Detector spot size for the 30deg folded design.
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
from scratch.test_tilt_layout import test_30deg

def inspect_dmd_and_detector():
    stack = SurfaceStack()
    
    # 0. Fiber
    stack.add_surface(Surface(
        name="Fiber Core Output", surface_type=SurfaceType.OBJECT, radius=0.0,
        thickness=25.7, glass="AIR", semi_diameter=0.025, thickness_variable=True
    ))
    
    # 1. L1 (f = 30.0 mm, AC127-030-B)
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=19.1, thickness=2.5, glass="SF5", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Mid", surface_type=SurfaceType.STANDARD, radius=-17.8, thickness=1.0, glass="N-BK7", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-82.3, thickness=20.0, glass="AIR", semi_diameter=6.35, thickness_variable=True))
    
    # 2. Grating (600 lp/mm, 30 deg tilt)
    stack.add_surface(Surface(name="Diffraction_Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=35.0, glass="MIRROR", semi_diameter=12.7, tilt_x=30.0, grating_lines_per_mm=600.0, grating_order=1, thickness_variable=True))
    
    # 3. L2 (f = 50.0 mm, AC254-050-B)
    stack.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=45.0, glass="AIR", semi_diameter=12.7, thickness_variable=True))
    
    # 4. DMD4500 (30 deg tilt)
    stack.add_surface(Surface(name="DMD4500_Mirror", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=45.0, glass="MIRROR", semi_diameter=10.0, tilt_x=30.0, thickness_variable=True))
    
    # 5. L3 (f = 50.0 mm, AC254-050-B in reverse)
    stack.add_surface(Surface(name="L3_Front", surface_type=SurfaceType.STANDARD, radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L3_Mid", surface_type=SurfaceType.STANDARD, radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L3_Back", surface_type=SurfaceType.STANDARD, radius=-34.0, thickness=20.0, glass="AIR", semi_diameter=12.7, thickness_variable=True))
    
    # 6. L4 Condenser (Thorlabs ACL12708U, f = 8.0 mm, NA = 0.78)
    stack.add_surface(Surface(name="L4_Condenser_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="L4_Condenser_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=5.5, glass="AIR", semi_diameter=6.35, thickness_variable=True))
    
    # 7. Detector: 1.0 mm x 1.0 mm (semi-diameter = 0.5 mm)
    stack.add_surface(Surface(name="Detector_1mm2", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.5, aperture_type="Rectangular", aperture_x_half=0.5, aperture_y_half=0.5))
    
    stack.primary_wavelength = 0.850
    stack._update_accumulated_positions()
    
    sys_opt = OpticalSystem(stack)
    wls = [0.785, 0.820, 0.850, 0.880, 0.920]
    
    print("=" * 70)
    print("DMD4500 & DETECTOR SPATIAL COVERAGE ANALYSIS:")
    print("=" * 70)
    
    dmd_hits = {}
    det_hits = {}
    
    for wl in wls:
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = sys_opt.trace_ray(ray)
        for h in tr.history:
            if h["surface_index"] == 8:
                dmd_hits[wl] = (h["x"], h["y"])
            if h["surface_index"] == 14:
                det_hits[wl] = (h["x"], h["y"])
                
        print(f"  Lambda = {wl*1000:5.1f} nm | DMD (X,Y) = ({dmd_hits.get(wl, (0,0))[0]:+6.3f}, {dmd_hits.get(wl, (0,0))[1]:+6.3f}) mm | Det (X,Y) = ({det_hits.get(wl, (0,0))[0]:+6.3f}, {det_hits.get(wl, (0,0))[1]:+6.3f}) mm")

    dmd_x = [pos[0] for pos in dmd_hits.values()]
    dmd_y = [pos[1] for pos in dmd_hits.values()]
    dmd_span_x = max(dmd_x) - min(dmd_x)
    dmd_span_y = max(dmd_y) - min(dmd_y)
    dmd_total_span = np.sqrt(dmd_span_x**2 + dmd_span_y**2)
    
    det_x = [pos[0] for pos in det_hits.values()]
    det_y = [pos[1] for pos in det_hits.values()]
    det_span_x = max(det_x) - min(det_x)
    det_span_y = max(det_y) - min(det_y)
    det_total_span = np.sqrt(det_span_x**2 + det_span_y**2)
    
    print("\n" + "-" * 70)
    print(f"DMD4500 Active Size: 9.855 mm (width) x 6.160 mm (height)")
    print(f"  - Total Dispersed Length on DMD:  {dmd_total_span:.3f} mm")
    print(f"  - DMD Width Coverage Percentage:  {(dmd_total_span / 9.855) * 100.0:.1f}%  (Constraint: ~50% / Half Area)")
    print(f"\n1 mm² Detector (1.0 mm x 1.0 mm):")
    print(f"  - Total Dispersed Length on Det:  {det_total_span:.3f} mm")
    print(f"  - Net Demagnification M:          {det_total_span / dmd_total_span:.3f} ({dmd_total_span / det_total_span:.1f}x demagnification)")
    print(f"  - Conforms to 1.0 mm² Detector:   {'YES (Completely inside 1 mm detector)' if det_total_span <= 0.85 else 'NO'}")
    print("-" * 70)

if __name__ == "__main__":
    inspect_dmd_and_detector()
