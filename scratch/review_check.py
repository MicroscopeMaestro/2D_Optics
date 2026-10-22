import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.surface import SurfaceType
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

def main():
    stack = SurfaceStack.load_from_json('hybrid_raman_1064.json')['stack']
    opt = OpticalSystem(stack)

    wls = [1.1113, 1.2000, 1.3000, 1.3620, 1.38919, 1.5000, 1.6000, 1.6133, 1.66708]
    print("=" * 80)
    print("CHIEF RAY TRACE RESULTS")
    print("=" * 80)
    print(f"{'WL (um)':<10} | {'DMD X (mm)':<12} | {'DMD Y (mm)':<12} | {'DET X (mm)':<12} | {'DET Y (mm)':<12} | {'DET R (mm)':<12}")
    print("-" * 80)
    
    chief_dmd_x = {}
    chief_det_r = {}
    for wl in wls:
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = opt.trace_rays([r])[0]
        dmd_h = next(h for h in tr.history if h['surface_index'] == 8)
        det_h = next(h for h in tr.history if h['surface_index'] == 14)
        det_r = np.sqrt(det_h['x']**2 + det_h['y']**2)
        chief_dmd_x[wl] = dmd_h['x']
        chief_det_r[wl] = det_r
        print(f"{wl:<10.4f} | {dmd_h['x']:<12.4f} | {dmd_h['y']:<12.4f} | {det_h['x']:<12.4f} | {det_h['y']:<12.4f} | {det_r:<12.4f}")

    print("\n" + "=" * 80)
    print("DMD DISPERSION SPAN & CENTERING CHECKS")
    print("=" * 80)
    span_1613 = abs(chief_dmd_x[1.6133] - chief_dmd_x[1.1113])
    center_1613 = (chief_dmd_x[1.6133] + chief_dmd_x[1.1113]) / 2.0
    print(f"1.1113 um to 1.6133 um: Span = {span_1613:.4f} mm, Center = {center_1613:.4f} mm")
    print(f"Requirement: Span in [8.00, 8.664] mm -> {'PASS' if 8.00 <= span_1613 <= 8.664 else 'FAIL'}")
    print(f"Active window bounds: [-4.332, +4.332] mm")
    print(f"Min DMD X (1.1113 um): {chief_dmd_x[1.1113]:.4f} mm -> {'PASS' if chief_dmd_x[1.1113] >= -4.332 else 'FAIL'}")
    print(f"Max DMD X (1.6133 um): {chief_dmd_x[1.6133]:.4f} mm -> {'PASS' if chief_dmd_x[1.6133] <= 4.332 else 'FAIL'}")

    span_1600 = abs(chief_dmd_x[1.6000] - chief_dmd_x[1.1113])
    print(f"1.1113 um to 1.6000 um: Span = {span_1600:.4f} mm")

    print("\n" + "=" * 80)
    print("DETECTOR SPOT RADIUS (CHIEF RAYS)")
    print("=" * 80)
    max_chief_r = max(chief_det_r[wl] for wl in [1.1113, 1.2, 1.3, 1.362, 1.38919, 1.5, 1.6, 1.6133])
    print(f"Max chief ray radius (1.1113 - 1.6133 um): {max_chief_r:.4f} mm (Diameter = {max_chief_r*2:.4f} mm)")
    print(f"Requirement: Spot radius <= 0.500 mm (diameter <= 1.0 mm) -> {'PASS' if max_chief_r <= 0.500 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("FULL RAY BUNDLE TRACE (315 rays: 25 um fiber, NA=0.12, 7 Raman wavelengths)")
    print("=" * 80)
    dense_rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM, dense_mode=True)
    print(f"Total rays generated (dense mode): {len(dense_rays)}")
    traced_dense = opt.trace_rays(dense_rays)
    vignetted = [r for r in traced_dense if r.vignetted and (not r.history or r.history[-1]['surface_index'] < 14)]
    print(f"Vignetted before detector: {len(vignetted)} / {len(traced_dense)} ({len(vignetted)/len(traced_dense)*100:.2f}%)")
    
    det_hits = [r.history[-1] for r in traced_dense if len(r.history) > 1 and r.history[-1]['surface_index'] == 14]
    print(f"Rays reaching detector: {len(det_hits)} / {len(traced_dense)}")
    
    print("\n" + "=" * 80)
    print("SURFACES AUDIT IN hybrid_raman_1064.json")
    print("=" * 80)
    for i, s in enumerate(stack.surfaces):
        st_name = s.surface_type.name if hasattr(s.surface_type, 'name') else str(s.surface_type)
        print(f"Surface {i:2d}: {s.name:<25} | Type: {st_name:<10} | R: {s.radius:8.2f} | CT: {s.thickness:6.2f} | Glass: {s.glass:<10} | SemiDia: {s.semi_diameter:5.2f} mm | TiltY: {s.tilt_y:5.1f} deg")

if __name__ == '__main__':
    main()
