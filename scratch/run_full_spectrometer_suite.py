"""
Full Optical Design, Optimization, and Verification Suite for Portable DMD Spectrometer.
"""
import sys
import os
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.analysis.spot import SpotDiagram

def build_portable_dmd_spectrometer():
    stack = SurfaceStack()
    
    # 0. Object: Multimode Fiber (50 um core, NA = 0.22)
    stack.add_surface(Surface(
        name="Fiber Core Output",
        surface_type=SurfaceType.OBJECT,
        radius=0.0,
        thickness=25.7, # Collimating distance to L1
        glass="AIR",
        semi_diameter=0.025,
        thickness_variable=True,
        thickness_min=10.0,
        thickness_max=45.0,
        aperture_type="Circular",
        aperture_x_half=15.0,
        aperture_y_half=15.0
    ))
    
    # 1. Collimating Achromatic Doublet L1 (Thorlabs AC127-030-B, f = 30.0 mm, Dia = 12.7 mm)
    # Prescription: R1 = 19.1 mm, t1 = 2.5 mm (SF5), R2 = -17.8 mm, t2 = 1.0 mm (N-BK7), R3 = -82.3 mm
    stack.add_surface(Surface(
        name="L1_Front",
        surface_type=SurfaceType.STANDARD,
        radius=19.1,
        thickness=2.5,
        glass="SF5",
        semi_diameter=6.35,
        aperture_type="Circular",
        aperture_x_half=6.35,
        aperture_y_half=6.35
    ))
    stack.add_surface(Surface(
        name="L1_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-17.8,
        thickness=1.0,
        glass="N-BK7",
        semi_diameter=6.35,
        aperture_type="Circular",
        aperture_x_half=6.35,
        aperture_y_half=6.35
    ))
    stack.add_surface(Surface(
        name="L1_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-82.3,
        thickness=20.0, # Air gap L1 -> Grating
        glass="AIR",
        semi_diameter=6.35,
        thickness_variable=True,
        thickness_min=5.0,
        thickness_max=40.0,
        aperture_type="Circular",
        aperture_x_half=6.35,
        aperture_y_half=6.35
    ))
    
    # 2. Reflection Diffraction Grating (600 lines/mm, 15 deg incidence tilt)
    stack.add_surface(Surface(
        name="Diffraction_Grating",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=20.0, # Air gap Grating -> L2
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=15.0,
        grating_lines_per_mm=600.0,
        grating_order=1,
        thickness_variable=True,
        thickness_min=5.0,
        thickness_max=50.0,
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7
    ))
    
    # 3. Focusing Achromatic Doublet L2 (Thorlabs AC254-050-B, f = 50.0 mm, Dia = 25.4 mm)
    # Prescription: R1 = 34.0 mm, t1 = 6.0 mm (N-BK7), R2 = -27.8 mm, t2 = 2.0 mm (SF5), R3 = -115.0 mm
    stack.add_surface(Surface(
        name="L2_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.0,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7,
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7
    ))
    stack.add_surface(Surface(
        name="L2_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7,
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7
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
        thickness_max=65.0,
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7
    ))
    
    # 4. DLP4500 DMD Modulator (Active size: 9.855 x 6.16 mm, 12 deg tilt ON-state)
    stack.add_surface(Surface(
        name="DMD4500_Mirror",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=45.0, # Air gap DMD -> L3
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=12.0, # ON-state tilt
        thickness_variable=True,
        thickness_min=20.0,
        thickness_max=65.0,
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7
    ))
    
    # 5. Collection Relay Collimator Doublet L3 (Thorlabs AC254-050-B in reverse conjugate, f = 50.0 mm, Dia = 25.4 mm)
    # Prescription reversed: R1 = 115.0 mm, t1 = 2.0 mm (SF5), R2 = 27.8 mm, t2 = 6.0 mm (N-BK7), R3 = -34.0 mm
    stack.add_surface(Surface(
        name="L3_Front",
        surface_type=SurfaceType.STANDARD,
        radius=115.0,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7,
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7
    ))
    stack.add_surface(Surface(
        name="L3_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=27.8,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7,
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7
    ))
    stack.add_surface(Surface(
        name="L3_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-34.0,
        thickness=15.0, # Air gap L3 -> L4
        glass="AIR",
        semi_diameter=12.7,
        thickness_variable=True,
        thickness_min=5.0,
        thickness_max=35.0,
        aperture_type="Circular",
        aperture_x_half=12.7,
        aperture_y_half=12.7
    ))
    
    # 6. High-NA Condenser Lens L4 (Thorlabs ACL12708U Aspheric Condenser, f = 8.0 mm, NA = 0.78, Dia = 12.7 mm)
    stack.add_surface(Surface(
        name="L4_Condenser_Front",
        surface_type=SurfaceType.ASPHERIC,
        radius=4.16,
        thickness=4.5,
        glass="B270",
        semi_diameter=6.35,
        conic=-0.6265,
        asphere_coeffs=[1.5e-5],
        aperture_type="Circular",
        aperture_x_half=6.35,
        aperture_y_half=6.35
    ))
    stack.add_surface(Surface(
        name="L4_Condenser_Back",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=5.5, # Air gap L4 -> Detector (WD ~ 4.3 - 6.0 mm)
        glass="AIR",
        semi_diameter=6.35,
        thickness_variable=True,
        thickness_min=2.0,
        thickness_max=12.0,
        aperture_type="Circular",
        aperture_x_half=6.35,
        aperture_y_half=6.35
    ))
    
    # 7. Detector: 1.0 mm x 1.0 mm active area (1 mm² single detector)
    stack.add_surface(Surface(
        name="Detector_1mm2",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.5, # 1.0 mm active diameter / 1 mm² active area
        aperture_type="Rectangular",
        aperture_x_half=0.5,
        aperture_y_half=0.5
    ))
    
    return stack

def main():
    print("=" * 75)
    print("PORTABLE FIBER DMD SPECTROMETER OPTIMIZATION & VERIFICATION")
    print("=" * 75)
    
    stack = build_portable_dmd_spectrometer()
    wls = [0.785, 0.850, 0.900, 0.950]
    primary_wl = 0.850
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    
    # Stage 1: Optimize Collimation (Surf 0: Fiber to L1)
    print("\n--- STAGE 1: Fiber Collimation (Surf 0) ---")
    opt1 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[primary_wl],
        method="Nelder-Mead",
        max_iter=200,
        target="Angular Spread (Collimation)",
        num_rays=15,
        target_surface_index=3 # After L1 Back
    )
    res1 = opt1.optimize(variable_indices=[0])
    stack.surfaces[0].thickness = res1.optimized_stack.surfaces[0].thickness
    print(f"  Collimation Success: {res1.success} | Fiber-L1 distance: {stack.surfaces[0].thickness:.4f} mm")

    # Stage 2: Optimize DMD Focus (Surf 7: L2 Back to DMD)
    print("\n--- STAGE 2: DMD Spectral Focus (Surf 7) ---")
    opt2 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=wls,
        method="Nelder-Mead",
        max_iter=250,
        target="Spot Size (Focus)",
        num_rays=15,
        target_surface_index=8 # DMD Surface
    )
    res2 = opt2.optimize(variable_indices=[7])
    stack.surfaces[7].thickness = res2.optimized_stack.surfaces[7].thickness
    print(f"  DMD Focus Success: {res2.success} | L2-DMD distance: {stack.surfaces[7].thickness:.4f} mm")

    # Measure dispersion on DMD
    stack._update_accumulated_positions()
    dmd_sys = OpticalSystem(stack)
    fiber_na = 0.22
    dmd_hits = {}
    print("\n  Spectral Line Position on DMD4500 (Target: ~5.0 mm = Half Area):")
    for wl in wls:
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = dmd_sys.trace_ray(chief_ray)
        for h in tr.history:
            if h["surface_index"] == 8:
                dmd_hits[wl] = h["y"]
                print(f"    Lambda = {wl*1000:5.1f} nm -> Chief Ray Y on DMD = {h['y']:+7.3f} mm")
                break
                
    if len(dmd_hits) >= 2:
        dmd_span = max(dmd_hits.values()) - min(dmd_hits.values())
        print(f"  -> Total Spectral Dispersion Span on DMD: {dmd_span:.3f} mm ({dmd_span/9.855*100.0:.1f}% of DMD 9.855 mm active length)")

    # Stage 3: Optimize Relay & Condenser to Detector (Surf 8, 11, 13)
    print("\n--- STAGE 3: Relay & Condenser Focus on Detector (Surfs 8, 11, 13) ---")
    opt3 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=wls,
        method="Nelder-Mead",
        max_iter=350,
        target="Spot Size (Focus)",
        num_rays=15,
        target_surface_index=14 # Detector
    )
    res3 = opt3.optimize(variable_indices=[8, 11, 13])
    for idx in [8, 11, 13]:
        stack.surfaces[idx].thickness = res3.optimized_stack.surfaces[idx].thickness
    print(f"  Detector Focus Success: {res3.success}")
    print(f"  DMD-L3 distance: {stack.surfaces[8].thickness:.4f} mm")
    print(f"  L3-L4 distance:  {stack.surfaces[11].thickness:.4f} mm")
    print(f"  L4-Det distance: {stack.surfaces[13].thickness:.4f} mm")

    # Stage 4: Global Fine-Tuning of All Air Gaps
    print("\n--- STAGE 4: Global System Fine-Tuning ---")
    all_var_indices = [i for i, s in enumerate(stack.surfaces) if getattr(s, 'thickness_variable', False)]
    opt4 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=wls,
        method="Nelder-Mead",
        max_iter=400,
        target="Spot Size (Focus)",
        num_rays=21,
        target_surface_index=14
    )
    res4 = opt4.optimize(variable_indices=all_var_indices)
    stack = res4.optimized_stack
    print(f"  Global Optimization Success: {res4.success}")
    print(f"  Initial RMS: {res4.initial_rms:.4f} mm -> Final RMS: {res4.final_rms:.4f} mm ({res4.rms_reduction_pct:.1f}% reduction)")

    # Full Ray Tracing Verification
    print("\n" + "=" * 75)
    print("DETAILED VERIFICATION ON 1 mm² DETECTOR (Active radius = 0.500 mm)")
    print("=" * 75)
    
    final_sys = OpticalSystem(stack)
    test_wls = np.linspace(0.785, 0.950, 9)
    test_angles = np.linspace(-np.arcsin(fiber_na), np.arcsin(fiber_na), 15)
    
    total_traced = 0
    total_passed = 0
    all_det_hits_y = []
    all_det_radii = []
    
    for wl in test_wls:
        wl_hits_y = []
        wl_hits_r = []
        for ang in test_angles:
            total_traced += 1
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(ang), np.cos(ang)]), wavelength=wl)
            tr = final_sys.trace_ray(ray)
            if not tr.vignetted and len(tr.history) > 0 and tr.history[-1]["surface_index"] == 14:
                total_passed += 1
                hy = tr.history[-1]["hit_local"][1]
                hx = tr.history[-1]["hit_local"][0]
                hr = np.sqrt(hx**2 + hy**2)
                wl_hits_y.append(hy)
                wl_hits_r.append(hr)
                all_det_hits_y.append(hy)
                all_det_radii.append(hr)
        
        if wl_hits_y:
            cy = np.mean(wl_hits_y)
            rms_r = np.std(wl_hits_y)
            max_r = np.max(wl_hits_r)
            print(f"  Lambda = {wl*1000:5.1f} nm | Rays: {len(wl_hits_y):2d}/15 | Centroid Y: {cy:+6.3f} mm | RMS Spot: {rms_r*1000:5.1f} um | Max Extent: {max_r*1000:5.1f} um")

    det_span_y = max(all_det_hits_y) - min(all_det_hits_y)
    det_max_radius = max(all_det_radii)
    
    print("\n" + "-" * 75)
    print("FINAL SUMMARY PERFORMANCE METRICS:")
    print(f"  - Total Dispersed Centroid Span on DMD4500:  {dmd_span:.3f} mm (~{dmd_span/9.855*100:.1f}% of DMD active width)")
    print(f"  - Total Dispersed Centroid Span on Detector: {det_span_y:.3f} mm")
    print(f"  - Net Optical Demagnification M (Det / DMD): {det_span_y / dmd_span:.3f} ({dmd_span / det_span_y:.1f}x demagnification)")
    print(f"  - Maximum Ray Radial Extent on Detector:     {det_max_radius:.3f} mm ({det_max_radius*1000:.1f} um)")
    print(f"  - Detector Active Radius Limit:              0.500 mm (1.0 mm x 1.0 mm)")
    print(f"  - System Optical Throughput / Transmission:  {total_passed / total_traced * 100.0:.1f}%")
    print(f"  - Conformance to 1 mm² Detector Limit:       {'EXCELLENT (100% rays fit within 1 mm detector)' if det_max_radius <= 0.500 else 'CLIPPED'}")
    print("-" * 75)

    # Save to JSON
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portable_fiber_spectrometer_dmd4500_1mm.json")
    stack.save_to_json(json_path, [0.785, 0.850, 0.900, 0.950])
    print(f"\nFinal optical prescription saved to:\n  {json_path}")

    # Also update fiber_spectrometer_portable_optimized.json
    portable_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fiber_spectrometer_portable_optimized.json")
    stack.save_to_json(portable_path, [0.785, 0.850, 0.900, 0.950])
    print(f"Updated portable model:\n  {portable_path}")

if __name__ == "__main__":
    main()
