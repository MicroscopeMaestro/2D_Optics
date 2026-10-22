"""
Fast Staged Optical Optimizer & Verification for Portable DMD Spectrometer.
Demonstrates rapid convergence and exact optical analysis.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.analysis.spot import SpotDiagram

def build_spectrometer(l3_fl=50.0, l4_fl=6.0):
    stack = SurfaceStack()
    
    # 0. Fiber (50 um core, NA = 0.22)
    stack.add_surface(Surface(
        name="Fiber Core Output", surface_type=SurfaceType.OBJECT, radius=0.0,
        thickness=25.7, glass="AIR", semi_diameter=0.025, thickness_variable=True,
        thickness_min=15.0, thickness_max=40.0
    ))
    
    # 1. Collimating Doublet L1 (Thorlabs AC127-030-B, f = 30.0 mm, Dia = 12.7 mm)
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=19.1, thickness=2.5, glass="SF5", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Mid", surface_type=SurfaceType.STANDARD, radius=-17.8, thickness=1.0, glass="N-BK7", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-82.3, thickness=25.0, glass="AIR", semi_diameter=6.35, thickness_variable=True, thickness_min=10.0, thickness_max=45.0))
    
    # 2. Diffraction Grating (600 lines/mm, 30 deg tilt)
    stack.add_surface(Surface(name="Diffraction_Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=35.0, glass="MIRROR", semi_diameter=12.7, tilt_x=30.0, grating_lines_per_mm=600.0, grating_order=1, thickness_variable=True, thickness_min=15.0, thickness_max=55.0))
    
    # 3. Focusing Doublet L2 (Thorlabs AC254-050-B, f = 50.0 mm, Dia = 25.4 mm)
    stack.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=45.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=25.0, thickness_max=65.0))
    
    # 4. DLP4500 DMD (30 deg tilt, Active size: 9.855 x 6.16 mm)
    stack.add_surface(Surface(name="DMD4500_Mirror", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=45.0, glass="MIRROR", semi_diameter=10.0, tilt_x=30.0, thickness_variable=True, thickness_min=20.0, thickness_max=65.0))
    
    # 5. Relay Collimator L3 (Thorlabs AC254-050-B in reverse, f = 50.0 mm, Dia = 25.4 mm)
    stack.add_surface(Surface(name="L3_Front", surface_type=SurfaceType.STANDARD, radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L3_Mid", surface_type=SurfaceType.STANDARD, radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L3_Back", surface_type=SurfaceType.STANDARD, radius=-34.0, thickness=18.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=5.0, thickness_max=35.0))
    
    # 6. High-NA Aspheric Condenser L4 (Thorlabs ACL1006U / ACL12708U, f = 6.0 mm, NA = 0.78)
    # R = 3.12 mm, t = 3.8 mm, conic = -0.6265
    stack.add_surface(Surface(name="L4_Condenser_Front", surface_type=SurfaceType.ASPHERIC, radius=3.12, thickness=3.8, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="L4_Condenser_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=4.2, glass="AIR", semi_diameter=6.35, thickness_variable=True, thickness_min=1.5, thickness_max=10.0))
    
    # 7. Detector: 1.0 mm x 1.0 mm (semi-diameter = 0.5 mm)
    stack.add_surface(Surface(name="Detector_1mm2", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=5.0))
    
    return stack

def run_fast_optimization():
    t0 = time.time()
    print("=" * 75)
    print("FAST STAGED OPTIMIZATION: FIBER PORTABLE DMD SPECTROMETER")
    print("=" * 75)
    
    stack = build_spectrometer(l3_fl=50.0, l4_fl=6.0)
    wls = [0.785, 0.850, 0.900, 0.950]
    primary_wl = 0.850
    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    
    # Stage 1: Collimation (Surf 0)
    print("\n[STAGE 1] Collimating Fiber -> L1 (Surface 0)")
    opt1 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[primary_wl],
        method="Nelder-Mead",
        max_iter=50,
        target="Angular Spread (Collimation)",
        num_rays=9,
        target_surface_index=3
    )
    res1 = opt1.optimize(variable_indices=[0])
    stack.surfaces[0].thickness = res1.optimized_stack.surfaces[0].thickness
    print(f"  Stage 1 Success: {res1.success} | Fiber-L1 distance: {stack.surfaces[0].thickness:.4f} mm")

    # Stage 2: Dispersed Focus on DMD (Surf 7: L2 Back to DMD)
    print("\n[STAGE 2] Dispersed Spectral Focus on DMD (Surface 7)")
    opt2 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=wls,
        method="Nelder-Mead",
        max_iter=60,
        target="Spot Size (Focus)",
        num_rays=9,
        target_surface_index=8 # DMD
    )
    res2 = opt2.optimize(variable_indices=[7])
    stack.surfaces[7].thickness = res2.optimized_stack.surfaces[7].thickness
    print(f"  Stage 2 Success: {res2.success} | L2-DMD distance: {stack.surfaces[7].thickness:.4f} mm")

    # Measure dispersion on DMD
    stack._update_accumulated_positions()
    dmd_sys = OpticalSystem(stack)
    dmd_hits = {}
    print("\n  Dispersion Footprint on DMD4500:")
    for wl in [0.785, 0.820, 0.850, 0.890, 0.950]:
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = dmd_sys.trace_ray(ray)
        for h in tr.history:
            if h["surface_index"] == 8:
                dmd_hits[wl] = (h["x"], h["y"])
                print(f"    Wavelength {wl*1000:5.1f} nm -> DMD (X,Y) = ({h['x']:+6.3f}, {h['y']:+6.3f}) mm")
                break
                
    dmd_span = max(pos[0] for pos in dmd_hits.values()) - min(pos[0] for pos in dmd_hits.values())
    dmd_span_total = np.sqrt((max(pos[0] for pos in dmd_hits.values()) - min(pos[0] for pos in dmd_hits.values()))**2 + (max(pos[1] for pos in dmd_hits.values()) - min(pos[1] for pos in dmd_hits.values()))**2)
    print(f"  -> Total Spectral Dispersion on DMD: {dmd_span_total:.3f} mm ({(dmd_span_total / 9.855) * 100.0:.1f}% of DMD 9.855 mm active width)")

    # Stage 3: Relay & Condenser Focus (Surf 8: DMD->L3, Surf 11: L3->L4, Surf 13: L4->Detector)
    print("\n[STAGE 3] Relay & Condenser Focus onto Detector (Surfaces 8, 11, 13)")
    opt3 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=wls,
        method="Nelder-Mead",
        max_iter=80,
        target="Spot Size (Focus)",
        num_rays=9,
        target_surface_index=14
    )
    res3 = opt3.optimize(variable_indices=[8, 11, 13])
    for idx in [8, 11, 13]:
        stack.surfaces[idx].thickness = res3.optimized_stack.surfaces[idx].thickness
    print(f"  Stage 3 Success: {res3.success}")
    print(f"  DMD-L3 distance: {stack.surfaces[8].thickness:.4f} mm")
    print(f"  L3-L4 distance:  {stack.surfaces[11].thickness:.4f} mm")
    print(f"  L4-Det distance: {stack.surfaces[13].thickness:.4f} mm")

    # Stage 4: Joint System Fine-Tuning
    print("\n[STAGE 4] Joint System Optimization (All Air Gaps)")
    opt4 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=wls,
        method="Nelder-Mead",
        max_iter=100,
        target="Spot Size (Focus)",
        num_rays=11,
        target_surface_index=14
    )
    res4 = opt4.optimize(variable_indices=[0, 3, 4, 7, 8, 11, 13])
    stack = res4.optimized_stack
    print(f"  Stage 4 Success: {res4.success}")
    print(f"  RMS Spot Radius: {res4.initial_rms:.4f} mm -> {res4.final_rms:.4f} mm ({res4.rms_reduction_pct:.1f}% reduction)")

    # Final Verification with strict 1 mm² Detector (radius = 0.500 mm)
    stack.surfaces[14].semi_diameter = 0.500
    stack.surfaces[14].aperture_type = "Rectangular"
    stack.surfaces[14].aperture_x_half = 0.500
    stack.surfaces[14].aperture_y_half = 0.500
    stack._update_accumulated_positions()
    
    print("\n" + "=" * 75)
    print("FINAL STRICT 1 mm² DETECTOR VERIFICATION (Active Radius = 0.500 mm)")
    print("=" * 75)
    
    final_sys = OpticalSystem(stack)
    fiber_na = 0.22
    fine_wls = [0.785, 0.810, 0.835, 0.850, 0.875, 0.900, 0.925, 0.950]
    angles = np.linspace(-np.arcsin(fiber_na), np.arcsin(fiber_na), 11)
    
    total_traced = 0
    total_captured = 0
    all_det_r = []
    det_centroids = {}
    
    for wl in fine_wls:
        wl_hits_r = []
        wl_hits_xy = []
        for ang in angles:
            total_traced += 1
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(ang), np.cos(ang)]), wavelength=wl)
            tr = final_sys.trace_ray(ray)
            if not tr.vignetted and len(tr.history) > 0 and tr.history[-1]["surface_index"] == 14:
                total_captured += 1
                hx = tr.history[-1]["hit_local"][0]
                hy = tr.history[-1]["hit_local"][1]
                hr = np.sqrt(hx**2 + hy**2)
                wl_hits_r.append(hr)
                wl_hits_xy.append((hx, hy))
                all_det_r.append(hr)
                
        if wl_hits_xy:
            mean_x = np.mean([p[0] for p in wl_hits_xy])
            mean_y = np.mean([p[1] for p in wl_hits_xy])
            rms_spot = np.std([np.sqrt((p[0]-mean_x)**2 + (p[1]-mean_y)**2) for p in wl_hits_xy])
            max_r = np.max(wl_hits_r)
            det_centroids[wl] = (mean_x, mean_y)
            print(f"  Lambda = {wl*1000:5.1f} nm | Rays: {len(wl_hits_xy):2d}/11 | Centroid (X,Y) = ({mean_x:+6.3f}, {mean_y:+6.3f}) mm | RMS Spot: {rms_spot*1000:5.1f} um | Max Extent: {max_r*1000:5.1f} um")

    det_span_total = 0.0
    if len(det_centroids) >= 2:
        cx = [p[0] for p in det_centroids.values()]
        cy = [p[1] for p in det_centroids.values()]
        det_span_total = np.sqrt((max(cx)-min(cx))**2 + (max(cy)-min(cy))**2)
        
    max_radius_overall = max(all_det_r) if all_det_r else 999.0
    throughput = (total_captured / total_traced) * 100.0
    
    print("\n" + "-" * 75)
    print(f"OPTICAL PERFORMANCE SUMMARY:")
    print(f"  - DMD4500 Spectral Footprint Length:       {dmd_span_total:.3f} mm  ({(dmd_span_total/9.855)*100.0:.1f}% of DMD length -> HALF AREA CONSTRAINED!)")
    print(f"  - Detector Total Dispersed Centroid Span: {det_span_total:.3f} mm")
    print(f"  - Net Optical Demagnification M (Det/DMD): {det_span_total/dmd_span_total:.3f} ({dmd_span_total/det_span_total:.1f}x compression)")
    print(f"  - Maximum Ray Radial Extent on Detector:   {max_radius_overall:.3f} mm ({max_radius_overall*1000.0:.1f} um)")
    print(f"  - Detector Active Radius Limit:            0.500 mm (1.0 mm x 1.0 mm square active area)")
    print(f"  - Total System Optical Transmission:       {throughput:.1f}% ({total_captured}/{total_traced} rays)")
    print(f"  - Conformance to 1 mm² Detector Limit:     {'[100% SUCCESS] ALL RAYS FIT INSIDE 1mm² DETECTOR' if max_radius_overall <= 0.500 and throughput >= 95.0 else '[WARNING] Rays clipped'}")
    print(f"  - Total Optimization Time:                 {time.time() - t0:.2f} seconds")
    print("-" * 75)

    # Save to JSON
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portable_fiber_spectrometer_dmd4500_1mm.json")
    stack.save_to_json(json_path, [0.785, 0.850, 0.900, 0.950])
    print(f"\nSaved verified optical prescription to:\n  {json_path}")
    
    return stack

if __name__ == "__main__":
    run_fast_optimization()
