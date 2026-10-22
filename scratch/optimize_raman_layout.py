import sys
import os
import math
import numpy as np
import scipy.optimize as opt

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.source import SourceConfig, SourceType
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM, verify_spectrometer_layout

def build_spectrometer(p):
    t0, t3, t4, t7, t8, t11, t13 = p
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    surfaces = [
        # 0. Fiber Object
        Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=float(t0), glass="AIR", semi_diameter=25.0),
        # 1-3. Collimator AC254-050-C
        Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=float(t3), glass="AIR", semi_diameter=12.7),
        # 4. Grating GR25-0316
        Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=float(t4), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1),
        # 5-7. Focusing Doublet AC254-040-C
        Surface(name="AC254-040-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=float(t7), glass="AIR", semi_diameter=12.7),
        # 8. TI DMD4500
        Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(t8), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0),
        # 9-11. Relay Doublet AC508-075-C
        Surface(name="AC508-075-C_Relay_Front", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Mid", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Back", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=float(t11), glass="AIR", semi_diameter=25.4),
        # 12-13. Condenser ACL12708U-C
        Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]),
        Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(t13), glass="AIR", semi_diameter=12.7),
        # 14. InGaAs Detector
        Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.5)
    ]
    for s in surfaces:
        stack.add_surface(s)
    return stack

def main():
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Loaded {len(rays)} rays.")
    
    # Let's define the merit function
    def merit(p):
        stack = build_spectrometer(p)
        opt = OpticalSystem(stack)
        traced = opt.trace_rays(rays)
        
        # Count vignetting & measure metrics
        vigs = sum(1 for r in traced if r.vignetted)
        dmd_hits = [h for r in traced for h in r.history if h['surface_index'] == 8 and not h['vignetted']]
        det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14 and not h['vignetted']]
        
        dmd_xs = [h['x'] for h in dmd_hits] if dmd_hits else [0]
        dmd_ys = [h['y'] for h in dmd_hits] if dmd_hits else [0]
        dmd_span = max(dmd_xs) - min(dmd_xs)
        dmd_my = max(map(abs, dmd_ys))
        
        if det_hits:
            det_rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
            max_r = max(det_rs)
            cx = np.mean([h['x'] for h in det_hits])
            cy = np.mean([h['y'] for h in det_hits])
            cent = np.sqrt(cx**2 + cy**2)
        else:
            max_r = 999.0
            cent = 999.0
            
        score = (
            vigs * 10000.0 +
            max(0.0, dmd_span - 8.664) * 5000.0 +
            max(0.0, dmd_my - 2.4365) * 5000.0 +
            max(0.0, max_r - 0.500) * 10000.0 +
            max(0.0, cent - 0.150) * 5000.0 +
            max_r * 100.0 +
            cent * 100.0
        )
        return score

    # Test initial point
    p0 = [64.57, 15.0, 10.0, 44.0, 10.0, 5.0, 5.20]
    print(f"Initial score: {merit(p0)}")
    
    # Run Powell optimization
    res = opt.minimize(merit, p0, method='Powell', options={'maxiter': 50, 'disp': True})
    p_opt = res.x
    print(f"\nOptimal parameters: {p_opt}")
    
    # Save to hybrid_raman_1064_full_range.json and verify
    stack = build_spectrometer(p_opt)
    sc = SourceConfig()
    sc.source_type = SourceType.FIBER
    sc.input_aperture_width = 0.025
    sc.input_aperture_height = 1.0
    sc.input_aperture_shape = "Circular"
    sc.beam_radius = 0.0125
    sc.set_na(0.12)
    sc.num_rays = 7
    sc.wavelengths = [1.11130, 1.20000, 1.30000, 1.38919, 1.50000, 1.60000, 1.66708]
    stack.source_config = sc
    stack.save_to_json("hybrid_raman_1064_full_range.json", wavelengths=sc.wavelengths)
    
    passed, report = verify_spectrometer_layout("hybrid_raman_1064_full_range.json", json_out_path="verification_results.json", verbose=True)
    print(f"Final Verification: {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    main()
