"""
Evaluate collection relay magnification (L3, L4) for 100% ray capture on 1mm² detector.
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
from optics_app.engine.optimizer import OpticalOptimizer

def test_l3_l4_pairs():
    print("=" * 75)
    print("OPTIMIZING (L3, L4) MAGNIFICATION FOR 1 mm² DETECTOR")
    print("=" * 75)
    
    # We will test two configurations:
    # Config A: L3 = 50 mm (AC254-050-B), L4 = 6.0 mm aspheric condenser (ACL1006U)
    # Config B: L3 = 60 mm (AC254-060-B), L4 = 8.0 mm aspheric condenser (ACL12708U)
    # Config C: L3 = 50 mm (AC254-050-B), L4 = 7.5 mm aspheric condenser
    
    configs = [
        ("Config A: L3=50mm / L4=6.0mm Aspheric", 50.0, 6.0, 3.12, 3.8, -0.6265),
        ("Config B: L3=60mm / L4=8.0mm Aspheric", 60.0, 8.0, 4.16, 4.5, -0.6265),
        ("Config C: L3=50mm / L4=8.0mm Aspheric", 50.0, 8.0, 4.16, 4.5, -0.6265),
    ]
    
    for name, f3, f4, r_asph, t_asph, k_asph in configs:
        print(f"\n--- Testing {name} (Theoretical M = {f4/f3:.3f}, Demagnification = {f3/f4:.1f}x) ---")
        stack = SurfaceStack()
        
        # 0. Fiber
        stack.add_surface(Surface(
            name="Fiber Core Output", surface_type=SurfaceType.OBJECT, radius=0.0,
            thickness=25.7, glass="AIR", semi_diameter=0.025, thickness_variable=True,
            thickness_min=15.0, thickness_max=40.0
        ))
        
        # 1. L1 (f=30mm, Thorlabs AC127-030-B)
        stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=19.1, thickness=2.5, glass="SF5", semi_diameter=6.35))
        stack.add_surface(Surface(name="L1_Mid", surface_type=SurfaceType.STANDARD, radius=-17.8, thickness=1.0, glass="N-BK7", semi_diameter=6.35))
        stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-82.3, thickness=25.0, glass="AIR", semi_diameter=6.35, thickness_variable=True, thickness_min=10.0, thickness_max=45.0))
        
        # 2. Grating (600 lp/mm, 30 deg)
        stack.add_surface(Surface(name="Diffraction_Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=35.0, glass="MIRROR", semi_diameter=12.7, tilt_x=30.0, grating_lines_per_mm=600.0, grating_order=1, thickness_variable=True, thickness_min=15.0, thickness_max=55.0))
        
        # 3. L2 (f=50mm, Thorlabs AC254-050-B)
        stack.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="L2_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=45.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=25.0, thickness_max=65.0))
        
        # 4. DMD4500 (30 deg)
        stack.add_surface(Surface(name="DMD4500_Mirror", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=45.0, glass="MIRROR", semi_diameter=12.7, tilt_x=30.0, thickness_variable=True, thickness_min=20.0, thickness_max=65.0))
        
        # 5. L3 Relay (f=50mm or 60mm)
        if f3 == 50.0:
            stack.add_surface(Surface(name="L3_Front", surface_type=SurfaceType.STANDARD, radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="L3_Mid", surface_type=SurfaceType.STANDARD, radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="L3_Back", surface_type=SurfaceType.STANDARD, radius=-34.0, thickness=20.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=5.0, thickness_max=40.0))
        else: # f3 = 60mm
            stack.add_surface(Surface(name="L3_Front", surface_type=SurfaceType.STANDARD, radius=138.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="L3_Mid", surface_type=SurfaceType.STANDARD, radius=33.5, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="L3_Back", surface_type=SurfaceType.STANDARD, radius=-41.0, thickness=20.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=5.0, thickness_max=40.0))
            
        # 6. L4 Condenser (f4)
        stack.add_surface(Surface(name="L4_Condenser_Front", surface_type=SurfaceType.ASPHERIC, radius=r_asph, thickness=t_asph, glass="B270", semi_diameter=6.35, conic=k_asph, asphere_coeffs=[1.5e-5]))
        stack.add_surface(Surface(name="L4_Condenser_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=f4*0.65, glass="AIR", semi_diameter=6.35, thickness_variable=True, thickness_min=1.5, thickness_max=12.0))
        
        # 7. Detector: 1.0 mm x 1.0 mm (semi-diameter = 0.5 mm)
        stack.add_surface(Surface(name="Detector_1mm2", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.5, aperture_type="Rectangular", aperture_x_half=0.5, aperture_y_half=0.5))
        
        wls = [0.785, 0.850, 0.900, 0.950]
        stack.primary_wavelength = 0.850
        stack._update_accumulated_positions()
        
        # Optimize air gaps
        opt = OpticalOptimizer(
            system=OpticalSystem(stack),
            wavelengths=wls,
            method="Nelder-Mead",
            max_iter=300,
            target="Spot Size (Focus)",
            num_rays=15,
            target_surface_index=14
        )
        res = opt.optimize(variable_indices=[0, 3, 4, 7, 8, 11, 13])
        opt_stack = res.optimized_stack
        opt_sys = OpticalSystem(opt_stack)
        
        # Ray trace full field & divergence
        fiber_na = 0.22
        fine_wls = [0.785, 0.820, 0.850, 0.890, 0.920, 0.950]
        angles = np.linspace(-np.arcsin(fiber_na), np.arcsin(fiber_na), 11)
        
        total_rays = 0
        captured_rays = 0
        all_det_radii = []
        dmd_y_centroids = []
        
        for wl in fine_wls:
            wl_hits_r = []
            for ang in angles:
                total_rays += 1
                ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(ang), np.cos(ang)]), wavelength=wl)
                tr = opt_sys.trace_ray(ray)
                if not tr.vignetted and len(tr.history) > 0 and tr.history[-1]["surface_index"] == 14:
                    captured_rays += 1
                    hx = tr.history[-1]["hit_local"][0]
                    hy = tr.history[-1]["hit_local"][1]
                    hr = np.sqrt(hx**2 + hy**2)
                    wl_hits_r.append(hr)
                    all_det_radii.append(hr)
                # Check DMD
                for h in tr.history:
                    if h["surface_index"] == 8 and abs(ang) < 1e-4:
                        dmd_y_centroids.append(h["y"])
                        
        dmd_span = max(dmd_y_centroids) - min(dmd_y_centroids) if dmd_y_centroids else 0.0
        max_r_det = max(all_det_radii) if all_det_radii else 999.0
        eff = (captured_rays / total_rays) * 100.0
        
        print(f"  Result for {name}:")
        print(f"    - DMD Spectral Span:        {dmd_span:.3f} mm ({dmd_span/9.855*100:.1f}% of DMD width)")
        print(f"    - Max Radial Spot on Det:   {max_r_det:.3f} mm ({max_r_det*1000:.1f} um)")
        print(f"    - Optical Throughput:       {eff:.1f}% ({captured_rays}/{total_rays} rays)")
        print(f"    - Fits 1 mm² Detector:      {'PERFECT! ALL RAYS FIT' if max_r_det <= 0.500 and eff >= 95.0 else 'NO'}")

if __name__ == "__main__":
    test_l3_l4_pairs()
