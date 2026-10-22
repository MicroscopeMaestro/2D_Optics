import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack

def build_mirror_spectrometer():
    stack = SurfaceStack()
    
    # 0. Fiber Object
    stack.add_surface(Surface(
        name="Fiber Object",
        surface_type=SurfaceType.OBJECT,
        thickness=50.0,
        semi_diameter=0.0125,
        thickness_variable=True
    ))
    
    # 1. Collimating Mirror (replaces L1)
    stack.add_surface(Surface(
        name="Collimating Mirror",
        surface_type=SurfaceType.STANDARD,
        radius=-100.0,  # f=50
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=-15.0, # Fold the beam
        radius_variable=True,
        thickness_variable=True
    ))
    
    # 2. Diffraction Grating (Reflective)
    stack.add_surface(Surface(
        name="Diffraction Grating",
        surface_type=SurfaceType.GRATING,
        radius=float('inf'),
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        grating_lines_per_mm=600,
        grating_order=1,
        tilt_x=15.0, # Bring the beam back
        thickness_variable=True
    ))
    
    # 3. Focusing Mirror (replaces L2)
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
    
    # 4. DMD Mirror
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
    
    # 5. Imaging Mirror 1 (replaces L3)
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
    
    # 6. Imaging Mirror 2 (replaces Condenser)
    stack.add_surface(Surface(
        name="Imaging Mirror 2",
        surface_type=SurfaceType.STANDARD,
        radius=-100.0,  # Concave
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=15.0,
        radius_variable=True,
        thickness_variable=True
    ))
    
    # 7. Detector
    stack.add_surface(Surface(
        name="Detector",
        surface_type=SurfaceType.DETECTOR,
        radius=float('inf'),
        thickness=0.0,
        semi_diameter=25.0
    ))
    
    return stack

if __name__ == "__main__":
    stack = build_mirror_spectrometer()
    wavelengths = [0.785, 0.850, 0.900]
    stack.primary_wavelength = 0.850  # Use center wavelength for diffraction path
    out_file = "fiber_spectrometer_mirrors.json"
    stack.save_to_json(out_file, wavelengths=wavelengths)
    print(f"Saved {out_file}")
