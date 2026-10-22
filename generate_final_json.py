#!/usr/bin/env python3
"""
generate_final_json.py: Authoritative Layout Generator for the Hybrid Raman Spectrometer (1064 nm).
Builds the 15-surface Thorlabs COTS optical layout adhering strictly to SurfaceStack version 2 schema.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.source import SourceConfig, SourceType
from verify_layout import RAMAN_WAVELENGTHS_UM

def build_hybrid_raman_layout() -> SurfaceStack:
    stack = SurfaceStack()
    stack.primary_wavelength = 1.3620
    
    surfaces = [
        # 0. Fiber Object: Thorlabs FG025LJA (25 um core, NA=0.12)
        Surface(
            name="Fiber_FG025LJA",
            surface_type=SurfaceType.OBJECT,
            radius=0.0,
            thickness=47.50,
            glass="AIR",
            semi_diameter=25.00,
            thickness_variable=False
        ),
        # 1-3. Collimating Doublet: Thorlabs AC254-050-C (f=50.0 mm, NIR C-coated)
        Surface(
            name="AC254-050-C_Coll_Front",
            surface_type=SurfaceType.STANDARD,
            radius=34.00,
            thickness=6.00,
            glass="N-BK7",
            semi_diameter=12.70,
            thickness_variable=False
        ),
        Surface(
            name="AC254-050-C_Coll_Mid",
            surface_type=SurfaceType.STANDARD,
            radius=-27.80,
            thickness=2.00,
            glass="SF5",
            semi_diameter=12.70,
            thickness_variable=False
        ),
        Surface(
            name="AC254-050-C_Coll_Back",
            surface_type=SurfaceType.STANDARD,
            radius=-115.00,
            thickness=20.00,
            glass="AIR",
            semi_diameter=12.70,
            thickness_variable=True
        ),
        # 4. Diffraction Grating: Thorlabs GR25-0316 (300 lines/mm, Blaze 1600 nm, 12 deg tilt)
        Surface(
            name="Grating_GR25-0316",
            surface_type=SurfaceType.GRATING,
            radius=0.0,
            thickness=20.00,
            glass="MIRROR",
            semi_diameter=12.70,
            tilt_y=12.0,
            grating_lines_per_mm=300.0,
            grating_order=1,
            thickness_variable=True
        ),
        # 5-7. Focusing Doublet: Thorlabs AC254-050-C (f=50.0 mm, NIR C-coated)
        Surface(
            name="AC254-050-C_Focus_Front",
            surface_type=SurfaceType.STANDARD,
            radius=34.00,
            thickness=6.00,
            glass="N-BK7",
            semi_diameter=12.70,
            thickness_variable=False
        ),
        Surface(
            name="AC254-050-C_Focus_Mid",
            surface_type=SurfaceType.STANDARD,
            radius=-27.80,
            thickness=2.00,
            glass="SF5",
            semi_diameter=12.70,
            thickness_variable=False
        ),
        Surface(
            name="AC254-050-C_Focus_Back",
            surface_type=SurfaceType.STANDARD,
            radius=-115.00,
            thickness=44.00,
            glass="AIR",
            semi_diameter=12.70,
            thickness_variable=True
        ),
        # 8. Texas Instruments DLP4500 DMD (8.664 x 4.873 mm active area, +12 deg tilt)
        Surface(
            name="TI_DMD4500",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=20.00,
            glass="MIRROR",
            semi_diameter=12.70,
            tilt_y=12.0,
            thickness_variable=True
        ),
        # 9-11. Collector Relay Doublet: Thorlabs AC254-075-C (1" doublet, f=75.0 mm, NIR C-coated)
        Surface(
            name="AC254-075-C_Relay_Front",
            surface_type=SurfaceType.STANDARD,
            radius=49.30,
            thickness=5.20,
            glass="N-BK7",
            semi_diameter=12.70,
            thickness_variable=False
        ),
        Surface(
            name="AC254-075-C_Relay_Mid",
            surface_type=SurfaceType.STANDARD,
            radius=-39.00,
            thickness=2.00,
            glass="SF5",
            semi_diameter=12.70,
            thickness_variable=False
        ),
        Surface(
            name="AC254-075-C_Relay_Back",
            surface_type=SurfaceType.STANDARD,
            radius=-170.00,
            thickness=5.00,
            glass="AIR",
            semi_diameter=12.70,
            thickness_variable=True
        ),
        # 12-13. Aspheric Condenser Lens: Thorlabs ACL12708U-C (f=8.0 mm, NA=0.67)
        Surface(
            name="ACL12708U-C_Cond_Front",
            surface_type=SurfaceType.ASPHERIC,
            radius=4.16,
            thickness=4.50,
            glass="B270",
            semi_diameter=12.70,
            conic=-0.6265,
            asphere_coeffs=[1.5e-5],
            thickness_variable=False
        ),
        Surface(
            name="ACL12708U-C_Cond_Back",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=5.20,
            glass="AIR",
            semi_diameter=12.70,
            thickness_variable=True
        ),
        # 14. Single-Element InGaAs Detector (1.0 mm active diameter / 0.50 mm semi-diameter)
        Surface(
            name="Detector_InGaAs_1mm",
            surface_type=SurfaceType.DETECTOR,
            radius=0.0,
            thickness=0.0,
            glass="AIR",
            semi_diameter=0.50,
            thickness_variable=False
        )
    ]
    for s in surfaces:
        stack.add_surface(s)
        
    # Source Configuration for 25 um Core, NA=0.12 Fiber
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
    return stack

def main():
    stack = build_hybrid_raman_layout()
    output_full = "hybrid_raman_1064_full_range.json"
    output_primary = "hybrid_raman_1064.json"
    stack.save_to_json(output_full, wavelengths=RAMAN_WAVELENGTHS_UM)
    stack.save_to_json(output_primary, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Successfully generated {output_full} and {output_primary} with {len(stack.surfaces)} Thorlabs catalog surfaces.")

if __name__ == "__main__":
    main()
