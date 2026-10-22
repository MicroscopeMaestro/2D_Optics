import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram
from verify_layout import generate_fiber_phase_space_rays

def detailed_stress_test(layout_file="hybrid_raman_1064.json"):
    print("=" * 80)
    print(f" DETAILED PHYSICAL & OPTICAL STRESS TEST FOR: {layout_file}")
    print("=" * 80)
    
    loaded = SurfaceStack.load_from_json(layout_file)
    stack = loaded["stack"] if isinstance(loaded, dict) else loaded
    opt = OpticalSystem(stack)
    
    dmd_idx = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())
    det_idx = len(stack.surfaces) - 1
    
    test_wls = [1.11100, 1.11130, 1.36200, 1.38919, 1.61300, 1.61330, 1.66700, 1.66708]
    
    print("\n--- 1. SURFACE SEQUENCE & THORLABS PRESCRIPTION AUDIT ---")
    for i, s in enumerate(stack.surfaces):
        stype_str = s.surface_type.name if hasattr(s.surface_type, 'name') else str(s.surface_type)
        print(f"Surface {i:2d}: {s.name:<28s} | Type: {stype_str:<10s} | R: {s.radius:8.2f} mm | T: {s.thickness:6.2f} mm | Glass: {s.glass:<10s} | CA Semi: {s.semi_diameter:5.2f} mm | Tilt Y: {s.tilt_y:5.1f}°")
    
    print("\n--- 2. CHIEF RAY COORDINATES & CONTAINMENT ---")
    chief_rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        for wl in test_wls
    ]
    traced_chief = opt.trace_rays(chief_rays)
    
    print(f"{'Wavelength (µm)':<16s} | {'DMD X (mm)':<12s} | {'DMD Status (<4.33mm)':<22s} | {'Det X (mm)':<12s} | {'Det Y (mm)':<12s} | {'Det R (mm)':<12s} | {'Det In 1mm Area':<16s}")
    print("-" * 115)
    for r in traced_chief:
        wl = r.wavelength
        dmd_hit = next(h for h in r.history if h["surface_index"] == dmd_idx)
        det_hit = next(h for h in r.history if h["surface_index"] == det_idx)
        dmd_x = dmd_hit["x"]
        dmd_ok = abs(dmd_x) <= 4.332
        dmd_status = "PASS (In Active)" if dmd_ok else f"MARGIN/OVER ({abs(dmd_x):.3f}mm)"
        det_x = det_hit["x"]
        det_y = det_hit["y"]
        det_r = np.sqrt(det_x**2 + det_y**2)
        det_ok = det_r <= 0.500
        det_status = "PASS (<=0.5mm)" if det_ok else f"EXCEED ({det_r:.3f}mm)"
        print(f"{wl:<16.5f} | {dmd_x:<12.4f} | {dmd_status:<22s} | {det_x:<12.4f} | {det_y:<12.4f} | {det_r:<12.4f} | {det_status:<16s}")
    
    # 3. Intermediate Optical Transmission Check (Surfaces 0 to 13)
    print("\n--- 3. INTERMEDIATE OPTICS TRANSMISSION (SURFACES 0-13) ---")
    phase_space_rays = generate_fiber_phase_space_rays(0.0125, 0.12, test_wls, dense_mode=False)
    traced_bundle = opt.trace_rays(phase_space_rays)
    
    for wl in test_wls:
        wl_rays = [r for r in traced_bundle if abs(r.wavelength - wl) < 1e-5]
        # Check if any ray clipped BEFORE reaching surface 14 (Detector)
        clipped_in_optics = sum(1 for r in wl_rays if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx))
        reached_det = sum(1 for r in wl_rays if len(r.history) > 0 and r.history[-1]["surface_index"] == det_idx)
        
        det_hits = [
            (h["x"], h["y"])
            for r in wl_rays
            for h in r.history
            if h["surface_index"] == det_idx
        ]
        
        if det_hits:
            radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
            max_r = max(radii)
            spot = SpotDiagram([p[0] for p in det_hits], [p[1] for p in det_hits], wavelength_um=wl)
            rms_r = spot.calculate_rms_radius()
        else:
            max_r = 999.0
            rms_r = 999.0
            
        print(f"WL {wl:.5f} µm: Total={len(wl_rays):3d} | Clipped in Optics (0-13)={clipped_in_optics:2d} (0% vignetting: {clipped_in_optics==0}) | Reached Det={reached_det:3d} | Det Max R={max_r:.4f}mm | RMS R={rms_r:.4f}mm")
        
if __name__ == "__main__":
    detailed_stress_test("hybrid_raman_1064.json")
    detailed_stress_test("hybrid_raman_1064_full_range.json")
