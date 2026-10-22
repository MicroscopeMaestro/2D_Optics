import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType

def build_hybrid_spectrometer() -> SurfaceStack:
    stack = SurfaceStack()
    
    # 0. Fiber Object
    stack.add_surface(Surface(
        name="Fiber Object",
        surface_type=SurfaceType.OBJECT,
        thickness=35.0, # Approximate back focal length of 40mm doublet
        semi_diameter=0.0125,
        thickness_variable=True
    ))
    
    # Collimator L1: AC254-040-A Doublet (replaces single mirror)
    stack.add_surface(Surface(
        name="L1 Front",
        surface_type=SurfaceType.STANDARD,
        radius=57.7,
        thickness=2.5,
        glass="SF5",
        semi_diameter=12.7,
    ))
    stack.add_surface(Surface(
        name="L1 Middle",
        surface_type=SurfaceType.STANDARD,
        radius=20.1,
        thickness=10.0,
        glass="N-BK7",
        semi_diameter=12.7,
    ))
    stack.add_surface(Surface(
        name="L1 Back",
        surface_type=SurfaceType.STANDARD,
        radius=-23.7,
        thickness=5.0,  # Distance to the Aperture Stop
        thickness_variable=True,
        glass="AIR",
        semi_diameter=12.7,
    ))
    
    # Field / Aperture Stop (To prevent overfilling)
    stack.add_surface(Surface(
        name="Aperture Stop",
        surface_type=SurfaceType.STOP,
        radius=float('inf'),
        thickness=35.0, # Distance to grating
        glass="AIR",
        semi_diameter=11.0, # Slightly smaller than the 12.7mm optical elements
        thickness_variable=True
    ))
    
    # Diffraction Grating (Reflective)
    stack.add_surface(Surface(
        name="Diffraction Grating",
        surface_type=SurfaceType.GRATING,
        radius=float('inf'),
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        grating_lines_per_mm=600,
        grating_order=1,
        tilt_x=15.0, # Fold the beam
        thickness_variable=True
    ))
    
    # Focusing Mirror (Concave)
    stack.add_surface(Surface(
        name="Focusing Mirror",
        surface_type=SurfaceType.STANDARD,
        radius=-100.0,  # f=50
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=-15.0, # Fold the beam
        radius_variable=True,
        thickness_variable=True
    ))
    
    # DMD Mirror (Flat)
    stack.add_surface(Surface(
        name="DMD Mirror",
        surface_type=SurfaceType.STANDARD,
        radius=float('inf'),
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=10.0,
        tilt_x=45.0,
        thickness_variable=True
    ))
    
    # Imaging Mirror 1 (Concave)
    stack.add_surface(Surface(
        name="Imaging Mirror 1",
        surface_type=SurfaceType.STANDARD,
        radius=-100.0,  # f=50
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=-15.0,
        radius_variable=True,
        thickness_variable=True
    ))
    
    # Detector Plane
    stack.add_surface(Surface(
        name="Detector",
        surface_type=SurfaceType.DETECTOR,
        radius=float('inf'),
        thickness=0.0,
        glass="AIR",
        semi_diameter=10.0,
        tilt_x=0.0
    ))
    
    return stack

if __name__ == "__main__":
    stack = build_hybrid_spectrometer()
    wavelengths = [0.785, 0.850, 0.900]
    stack.primary_wavelength = 0.850
    out_file = "hybrid_spectrometer.json"
    stack.save_to_json(out_file, wavelengths=wavelengths)
    print(f"Saved {out_file}")
