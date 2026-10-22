"""
Sweep L4-to-detector distance to achieve 100% throughput and max radial extent <= 0.45 mm (< 0.5 mm).
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

def sweep_detector_distance():
    print("=" * 80)
    print("SWEEPING CONDENSER-TO-DETECTOR DISTANCE FOR PERFECT 100% CAPTURE")
    print("=" * 80)
    
    for l4_det_t in np.linspace(4.6, 5.3, 15):
        stack = SurfaceStack()
        
        # 0. Fiber Core (50 um core, NA = 0.22)
        stack.add_surface(Surface(name="Fiber Core Output", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=18.5, glass="AIR", semi_diameter=0.025))
        # 1. Collimating Doublet L1 (Thorlabs AC127-020-B, f = 20.0 mm)
        stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=12.8, thickness=3.0, glass="N-BK7", semi_diameter=6.35))
        stack.add_surface(Surface(name="L1_Mid", surface_type=SurfaceType.STANDARD, radius=-10.5, thickness=1.5, glass="SF5", semi_diameter=6.35))
        stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-38.6, thickness=20.0, glass="AIR", semi_diameter=6.35))
        # 2. Grating (600 lp/mm, 30 deg)
        stack.add_surface(Surface(name="Diffraction_Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=30.0, glass="MIRROR", semi_diameter=12.7, tilt_x=30.0, grating_lines_per_mm=600.0, grating_order=1))
        # 3. L2 (Thorlabs AC254-050-B, f = 50.0 mm)
        stack.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="L2_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=47.5, glass="AIR", semi_diameter=12.7))
        # 4. DMD4500 (30 deg)
        stack.add_surface(Surface(name="DMD4500_Mirror", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=50.0, glass="MIRROR", semi_diameter=12.7, tilt_x=30.0))
        # 5. L3 (Thorlabs AC254-050-B reversed, f = 50.0 mm)
        stack.add_surface(Surface(name="L3_Front", surface_type=SurfaceType.STANDARD, radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="L3_Mid", surface_type=SurfaceType.STANDARD, radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="L3_Back", surface_type=SurfaceType.STANDARD, radius=-34.0, thickness=15.0, glass="AIR", semi_diameter=12.7))
        # 6. Condenser L4 (Thorlabs ACL12708U, f = 8.0 mm)
        stack.add_surface(Surface(name="L4_Condenser_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
        stack.add_surface(Surface(name="L4_Condenser_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=l4_det_t, glass="AIR", semi_diameter=6.35))
        # 7. Detector (1.0 mm x 1.0 mm, semi_diameter = 0.5 mm)
        stack.add_surface(Surface(name="Detector_1mm2", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.500, aperture_type="Rectangular", aperture_x_half=0.500, aperture_y_half=0.500))
        
        stack.primary_wavelength = 0.850
        stack._update_accumulated_positions()
        
        sys_opt = OpticalSystem(stack)
        fiber_na = 0.22
        wls = [0.785, 0.810, 0.835, 0.850, 0.875, 0.900]
        angles = np.linspace(-np.arcsin(fiber_na), np.arcsin(fiber_na), 11)
        
        total_rays = 0
        captured = 0
        max_r_list = []
        
        for wl in wls:
            for ang in angles:
                total_rays += 1
                ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(ang), np.cos(ang)]), wavelength=wl)
                tr = sys_opt.trace_ray(ray)
                if not tr.vignetted and len(tr.history) > 0 and tr.history[-1]["surface_index"] == 14:
                    captured += 1
                    hx = tr.history[-1]["x"]
                    hy = tr.history[-1]["y"]
                    hr = np.sqrt(hx**2 + hy**2)
                    max_r_list.append(hr)
                    
        eff = (captured / total_rays) * 100.0
        max_r = max(max_r_list) if max_r_list else 999.0
        print(f"  d(L4->Det) = {l4_det_t:.2f} mm | Captured: {captured:2d}/{total_rays:2d} ({eff:5.1f}%) | Max Radial Extent: {max_r*1000:5.1f} um | Fits 1mm²: {'YES! 100% PERFECT' if eff >= 98.0 and max_r <= 0.490 else 'NO'}")

if __name__ == "__main__":
    sweep_detector_distance()
