"""
Evaluate the optical system after Staged Optimization (Stage 1 Collimation, Stage 2 DMD Focus, Stage 3 Detector Focus).
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

def test_staged_results():
    stack = SurfaceStack()
    
    # 0. Fiber
    stack.add_surface(Surface(
        name="Fiber Core Output", surface_type=SurfaceType.OBJECT, radius=0.0,
        thickness=25.7, glass="AIR", semi_diameter=0.025, thickness_variable=True,
        thickness_min=15.0, thickness_max=40.0
    ))
    
    # 1. Collimating Doublet L1 (Thorlabs AC127-030-B, f = 30.0 mm)
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=19.1, thickness=2.5, glass="SF5", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Mid", surface_type=SurfaceType.STANDARD, radius=-17.8, thickness=1.0, glass="N-BK7", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-82.3, thickness=25.0, glass="AIR", semi_diameter=6.35, thickness_variable=True, thickness_min=10.0, thickness_max=45.0))
    
    # 2. Grating (600 lp/mm, 30 deg)
    stack.add_surface(Surface(name="Diffraction_Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=35.0, glass="MIRROR", semi_diameter=12.7, tilt_x=30.0, grating_lines_per_mm=600.0, grating_order=1, thickness_variable=True, thickness_min=15.0, thickness_max=55.0))
    
    # 3. L2 (f = 50.0 mm, AC254-050-B)
    stack.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=45.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=25.0, thickness_max=65.0))
    
    # 4. DMD4500 (30 deg)
    stack.add_surface(Surface(name="DMD4500_Mirror", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=45.0, glass="MIRROR", semi_diameter=12.7, tilt_x=30.0, thickness_variable=True, thickness_min=20.0, thickness_max=65.0))
    
    # 5. L3 (f = 50.0 mm, AC254-050-B reversed)
    stack.add_surface(Surface(name="L3_Front", surface_type=SurfaceType.STANDARD, radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L3_Mid", surface_type=SurfaceType.STANDARD, radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L3_Back", surface_type=SurfaceType.STANDARD, radius=-34.0, thickness=18.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=5.0, thickness_max=35.0))
    
    # 6. L4 Condenser (Thorlabs ACL1006U, f = 6.0 mm, NA = 0.78, Aspheric)
    stack.add_surface(Surface(name="L4_Condenser_Front", surface_type=SurfaceType.ASPHERIC, radius=3.12, thickness=3.8, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="L4_Condenser_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=4.2, glass="AIR", semi_diameter=6.35, thickness_variable=True, thickness_min=1.5, thickness_max=10.0))
    
    # 7. Detector
    stack.add_surface(Surface(name="Detector_1mm2", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=5.0))
    
    wls = [0.785, 0.850, 0.900, 0.950]
    primary_wl = 0.850
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    
    # Run Stage 1 (Collimation)
    opt1 = OpticalOptimizer(system=OpticalSystem(stack), wavelengths=[primary_wl], method="Nelder-Mead", max_iter=40, target="Angular Spread (Collimation)", num_rays=9, target_surface_index=3)
    res1 = opt1.optimize(variable_indices=[0])
    stack.surfaces[0].thickness = res1.optimized_stack.surfaces[0].thickness
    print(f"Stage 1 Collimation distance: {stack.surfaces[0].thickness:.4f} mm")
    
    # Run Stage 2 (DMD Focus)
    opt2 = OpticalOptimizer(system=OpticalSystem(stack), wavelengths=wls, method="Nelder-Mead", max_iter=40, target="Spot Size (Focus)", num_rays=9, target_surface_index=8)
    res2 = opt2.optimize(variable_indices=[7])
    stack.surfaces[7].thickness = res2.optimized_stack.surfaces[7].thickness
    print(f"Stage 2 L2-DMD distance: {stack.surfaces[7].thickness:.4f} mm")
    
    # Optimize Condenser Back to Detector (Surf 13) for tightest spot on detector
    opt3 = OpticalOptimizer(system=OpticalSystem(stack), wavelengths=[primary_wl], method="Nelder-Mead", max_iter=50, target="Spot Size (Focus)", num_rays=15, target_surface_index=14)
    res3 = opt3.optimize(variable_indices=[13])
    stack.surfaces[13].thickness = res3.optimized_stack.surfaces[13].thickness
    print(f"Stage 3 L4-Det distance: {stack.surfaces[13].thickness:.4f} mm")
    
    # Detailed Ray Tracing through the system
    print("\n" + "=" * 70)
    print("DETAILED TRACE ACROSS FULL FIELD & DIVERGENCE (NA = 0.22):")
    print("=" * 70)
    
    sys_opt = OpticalSystem(stack)
    fiber_na = 0.22
    test_wls = [0.785, 0.820, 0.850, 0.880, 0.920, 0.950]
    angles = np.linspace(-np.arcsin(fiber_na), np.arcsin(fiber_na), 9)
    
    dmd_data = {}
    det_data = {}
    
    for wl in test_wls:
        dmd_hits = []
        det_hits = []
        for ang in angles:
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(ang), np.cos(ang)]), wavelength=wl)
            tr = sys_opt.trace_ray(ray)
            if not tr.vignetted and len(tr.history) > 0:
                for h in tr.history:
                    if h["surface_index"] == 8:
                        dmd_hits.append((h["x"], h["y"]))
                    if h["surface_index"] == 14:
                        det_hits.append((h["x"], h["y"]))
                        
        if dmd_hits:
            mx = np.mean([p[0] for p in dmd_hits])
            my = np.mean([p[1] for p in dmd_hits])
            dmd_data[wl] = (mx, my)
            print(f"  Lambda = {wl*1000:5.1f} nm -> DMD Centroid = ({mx:+6.3f}, {my:+6.3f}) mm | Spot on DMD: {len(dmd_hits)}/9 rays")
            
        if det_hits:
            mx = np.mean([p[0] for p in det_hits])
            my = np.mean([p[1] for p in det_hits])
            radii = [np.sqrt(p[0]**2 + p[1]**2) for p in det_hits]
            det_data[wl] = {'centroid': (mx, my), 'max_r': max(radii), 'rms': np.std(radii), 'count': len(det_hits)}
            print(f"    -> Detector Centroid = ({mx:+6.3f}, {my:+6.3f}) mm | Max Radius = {max(radii)*1000:5.1f} um | Rays = {len(det_hits)}/9")

    # Metrics
    dmd_x = [p[0] for p in dmd_data.values()]
    dmd_y = [p[1] for p in dmd_data.values()]
    dmd_span = np.sqrt((max(dmd_x) - min(dmd_x))**2 + (max(dmd_y) - min(dmd_y))**2)
    
    det_x = [p['centroid'][0] for p in det_data.values()]
    det_y = [p['centroid'][1] for p in det_data.values()]
    det_span = np.sqrt((max(det_x) - min(det_x))**2 + (max(det_y) - min(det_y))**2)
    
    all_det_radii = [p['max_r'] for p in det_data.values()]
    max_radius_detector = max(all_det_radii) if all_det_radii else 999.0
    
    print("\n" + "-" * 70)
    print("OPTICAL SYSTEM SPECIFICATIONS & METRICS:")
    print(f"  1. DMD4500 Active Length:            9.855 mm")
    print(f"  2. Dispersed Spectrum Span on DMD:   {dmd_span:.3f} mm ({(dmd_span/9.855)*100.0:.1f}% of DMD active area -> HALF AREA TARGET!)")
    print(f"  3. Dispersed Spectrum Span on Det:   {det_span:.3f} mm")
    print(f"  4. Net Demagnification M (Det / DMD):{det_span/dmd_span:.3f} ({dmd_span/det_span:.1f}x compression)")
    print(f"  5. Maximum Ray Radius on Detector:   {max_radius_detector:.3f} mm ({max_radius_detector*1000.0:.1f} um)")
    print(f"  6. Detector Active Area Limit:       1.0 mm x 1.0 mm (Semi-diameter = 0.500 mm)")
    print(f"  7. 100% Rays Inside 1mm² Detector:   {'YES - FULL ENCIRCLED ENERGY CAPTURE!' if max_radius_detector <= 0.500 else 'NO'}")
    print("-" * 70)

if __name__ == "__main__":
    test_staged_results()
