import sys
import os
import json
import math
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.source import SourceConfig, SourceType
from verify_layout import verify_spectrometer_layout, generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def create_and_verify(t0, t3, t4, t7, t8, t11, t13, relay_tilt=0.0, cond_tilt=0.0, output_path="hybrid_raman_1064_full_range.json"):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    surfaces = [
        Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=float(t0), glass="AIR", semi_diameter=25.0),
        Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=float(t3), glass="AIR", semi_diameter=12.7),
        Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=float(t4), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1),
        Surface(name="AC254-040-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=float(t7), glass="AIR", semi_diameter=12.7),
        Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(t8), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0),
        Surface(name="AC508-075-C_Relay_Front", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4, tilt_y=float(relay_tilt)),
        Surface(name="AC508-075-C_Relay_Mid", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Back", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=float(t11), glass="AIR", semi_diameter=25.4),
        Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5], tilt_y=float(cond_tilt)),
        Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(t13), glass="AIR", semi_diameter=12.7),
        Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.5)
    ]
    for s in surfaces:
        stack.add_surface(s)
        
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
    stack.save_to_json(output_path, wavelengths=sc.wavelengths)
    
    passed, report = verify_spectrometer_layout(output_path, json_out_path=None, verbose=False)
    return passed, report

def main():
    print("Testing configurations...")
    # Test catalog default first
    p, r = create_and_verify(50.0, 15.0, 10.0, 36.5, 45.0, 10.0, 4.85)
    print(f"Catalog Default: passed={p}, transmission={r.get('assertions', [{}])[8].get('actual')}")
    for a in r.get('assertions', []):
        if not a['passed']:
            print(f"  FAIL: {a['name']} -> expected {a['expected']}, got {a['actual']}")

if __name__ == "__main__":
    main()
