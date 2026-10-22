import sys, os
sys.path.insert(0, os.path.abspath('.'))
import json
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.cad_export import CADExporter
from verify_layout import verify_spectrometer_layout

def create_final_1064_spectrometer_stack():
    stack = SurfaceStack()
    
    # 0. Fiber / Slit Source (25 um core, NA=0.12)
    stack.add_surface(Surface(
        name="Fiber_FG025LJA",
        surface_type=SurfaceType.OBJECT,
        thickness=47.5,
        semi_diameter=25.0,
        glass="AIR"
    ))
    
    # 1-3. Collimator: Thorlabs AC254-050-C (1-inch doublet, f=50.0mm, N-BK7 / SF5)
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.0,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-115.0,
        thickness=20.0,
        glass="AIR",
        semi_diameter=12.7
    ))
    
    # 4. Diffraction Grating: Thorlabs GR25-0316 (300 l/mm, Gold, Blaze 1600nm, alpha=12.0 deg)
    stack.add_surface(Surface(
        name="Grating_GR25-0316",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=20.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_y=12.0,
        grating_lines_per_mm=300.0,
        grating_order=1
    ))
    
    # 5-7. Spectrograph Camera Focus: Thorlabs AC254-050-C (1-inch doublet, f=50.0mm)
    stack.add_surface(Surface(
        name="AC254-050-C_Focus_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.0,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Focus_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Focus_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-115.0,
        thickness=44.0,
        glass="AIR",
        semi_diameter=12.7
    ))
    
    # 8. Spatial Light Modulator: Texas Instruments DLP4500 DMD (8.664mm x 4.873mm, +12.0 deg ON tilt)
    stack.add_surface(Surface(
        name="TI_DMD4500",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=20.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_y=12.0
    ))
    
    # 9-11. Post-DMD Collection Relay Collimator: Thorlabs AC254-075-C (1-inch doublet, f=75.0mm, N-BK7 / SF5)
    stack.add_surface(Surface(
        name="AC254-075-C_Relay_Front",
        surface_type=SurfaceType.STANDARD,
        radius=49.3,
        thickness=5.2,
        glass="N-BK7",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-075-C_Relay_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-39.0,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-075-C_Relay_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-170.0,
        thickness=5.0,
        glass="AIR",
        semi_diameter=12.7
    ))
    
    # 12-13. High-NA Molded Aspheric Condenser: Thorlabs ACL12708U-C (0.5-inch / 12.7mm dia, f=8.0mm, NA=0.78, B270)
    stack.add_surface(Surface(
        name="ACL12708U-C_Cond_Front",
        surface_type=SurfaceType.ASPHERIC,
        radius=4.16,
        thickness=4.5,
        glass="B270",
        semi_diameter=12.7,
        conic=-0.6265,
        asphere_coeffs=[1.5e-5]
    ))
    stack.add_surface(Surface(
        name="ACL12708U-C_Cond_Back",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=5.2,
        glass="AIR",
        semi_diameter=12.7
    ))
    
    # 14. Single-Pixel InGaAs Photodiode Detector (1.0 mm active area, semi-diameter 0.50 mm)
    stack.add_surface(Surface(
        name="Detector_InGaAs_1mm",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.50
    ))
    
    stack.primary_wavelength = 1.3620
    stack.wavelengths = [1.1113, 1.2, 1.3, 1.3620, 1.5, 1.6, 1.6133]
    stack._update_accumulated_positions()
    return stack

if __name__ == '__main__':
    stack = create_final_1064_spectrometer_stack()
    
    # Save to both target json paths
    stack.save_to_json("hybrid_raman_1064.json")
    stack.save_to_json("hybrid_raman_1064_full_range.json")
    print("[INFO] Successfully saved hybrid_raman_1064.json and hybrid_raman_1064_full_range.json")
    
    # Export CAD models
    print("[INFO] Exporting CAD deliverables...")
    CADExporter.export_to_step(stack, "hybrid_raman_1064.step")
    CADExporter.export_to_dxf_2d(stack, "hybrid_raman_1064.dxf")
    print("[INFO] CAD exports completed successfully.")
    
    # Run verification harness
    print("[INFO] Running verification harness...")
    passed, report = verify_spectrometer_layout("hybrid_raman_1064.json", verbose=True)
    print(f"\nFinal Verification Status: {'PASS' if passed else 'FAIL'}")
