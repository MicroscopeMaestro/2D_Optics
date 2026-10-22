"""
Full Optical Design, Simulation & Optimization Suite for Portable DMD Spectrometer.
"""
import sys
import os
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.analysis.spot import SpotDiagram

def create_dmd4500_portable_spectrometer_stack():
    """
    Builds the complete optical surface stack for the Portable DMD Spectrometer:
    - Fiber Source (NA = 0.22, core = 50 um)
    - L1 Collimator: Thorlabs AC127-030-B (f = 30 mm achromatic doublet)
    - Diffraction Grating: 600 lines/mm, 15 deg tilt
    - L2 Focusing Lens: Thorlabs AC254-050-B (f = 50 mm achromatic doublet)
    - DMD4500 Modulator: Active size 9.855 x 6.16 mm, 12 deg tilt ON-state
    - L3 Relay Collimator: Thorlabs AC254-050-B (f = 50 mm achromatic doublet)
    - L4 Condenser: Thorlabs High-NA Aspheric Condenser ACL12708U (f = 8 mm, NA = 0.78)
    - Detector: 1.0 mm x 1.0 mm (1 mm² active area)
    """
    stack = SurfaceStack()
    
    # 0. Object: Multimode Fiber (core 50 um, semi_diameter = 0.025 mm)
    stack.add_surface(Surface(
        name="Fiber Core Output",
        surface_type=SurfaceType.OBJECT,
        radius=0.0,
        thickness=26.5, # Initial guess for f=30mm doublet BFL
        glass="AIR",
        semi_diameter=0.025,
        thickness_variable=True,
        thickness_min=10.0,
        thickness_max=50.0
    ))
    
    # 1. Collimator Doublet L1 (Thorlabs AC127-030-B, f=30mm, 1/2" diameter)
    # R1 = 19.1 mm (SF5, t=2.5mm), R2 = -17.8 mm (N-BK7, t=1.0mm), R3 = -82.3 mm
    stack.add_surface(Surface(
        name="L1_Front",
        surface_type=SurfaceType.STANDARD,
        radius=19.1,
        thickness=2.5,
        glass="SF5",
        semi_diameter=6.35
    ))
    stack.add_surface(Surface(
        name="L1_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-17.8,
        thickness=1.0,
        glass="N-BK7",
        semi_diameter=6.35
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
        thickness_max=50.0
    ))
    
    # 2. Diffraction Grating (600 lines/mm, 15 deg tilt)
    stack.add_surface(Surface(
        name="Diffraction_Grating",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=25.0, # Air gap Grating -> L2
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=15.0,
        grating_lines_per_mm=600.0,
        grating_order=1,
        thickness_variable=True,
        thickness_min=10.0,
        thickness_max=60.0
    ))
    
    # 3. Focusing Doublet L2 (Thorlabs AC254-050-B, f=50mm, 1" diameter)
    # Radii for 50mm doublet: R1 = 34.0, t=6.0 (N-BK7), R2 = -27.8, t=2.0 (SF5), R3 = -115.0
    stack.add_surface(Surface(
        name="L2_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.0,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="L2_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
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
        thickness_max=70.0
    ))
    
    # 4. DMD4500 Spatial Light Modulator (Active Area: 9.855 x 6.16 mm, ON-state tilt = 12 deg)
    stack.add_surface(Surface(
        name="DMD4500_Mirror",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=45.0, # Air gap DMD -> L3
        glass="MIRROR",
        semi_diameter=6.0, # Covers > 11.6 mm diagonal active array
        tilt_x=12.0, # 12 deg micromirror ON-state tilt
        thickness_variable=True,
        thickness_min=20.0,
        thickness_max=70.0
    ))
    
    # 5. Relay Collimator Doublet L3 (Thorlabs AC254-050-B, f=50mm, 1" diameter)
    # Oriented in reverse conjugate (flint facing DMD)
    stack.add_surface(Surface(
        name="L3_Front",
        surface_type=SurfaceType.STANDARD,
        radius=115.0,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="L3_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=27.8,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7
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
        thickness_max=40.0
    ))
    
    # 6. High-NA Condenser Lens L4 (Thorlabs ACL12708U, f=8mm, NA=0.78, Aspheric)
    stack.add_surface(Surface(
        name="L4_Condenser_Front",
        surface_type=SurfaceType.ASPHERIC,
        radius=4.16,
        thickness=4.5,
        glass="B270",
        semi_diameter=6.35,
        conic=-0.6265,
        asphere_coeffs=[1.5e-5]
    ))
    stack.add_surface(Surface(
        name="L4_Condenser_Back",
        surface_type=SurfaceType.STANDARD,
        radius=0.0, # Plano back
        thickness=6.0, # Air gap L4 -> Detector (WD ~ 4.3 - 6 mm)
        glass="AIR",
        semi_diameter=6.35,
        thickness_variable=True,
        thickness_min=2.0,
        thickness_max=15.0
    ))
    
    # 7. Detector: 1.0 mm x 1.0 mm (1 mm² active area, semi-diameter = 0.5 mm)
    stack.add_surface(Surface(
        name="Detector_1mm2",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.5, # 1.0 mm active diameter / 1mm² active square
        aperture_type="Rectangular",
        aperture_x_half=0.5,
        aperture_y_half=0.5
    ))
    
    return stack

if __name__ == "__main__":
    stack = create_dmd4500_portable_spectrometer_stack()
    wls = [0.785, 0.850, 0.900, 0.950]
    stack.primary_wavelength = 0.850
    stack._update_accumulated_positions()
    print(f"Created DMD4500 Portable Spectrometer Stack with {len(stack.surfaces)} surfaces.")
