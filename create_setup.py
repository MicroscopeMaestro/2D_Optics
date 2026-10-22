import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack

def build_fiber_setup():
    stack = SurfaceStack()
    
    # 0. Object: 25 um core fiber output
    # Semi-diameter is 12.5 um (0.0125 mm). Light travels 19 mm to the first lens.
    stack.add_surface(Surface(name="Fiber Core Output", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=19.0, glass="AIR", semi_diameter=0.0125))
    
    # 1. Lens 1 (19 mm focal length doublet approximation)
    # R ~ 19.6 mm for N-BK7 to get f=19mm
    stack.add_surface(Surface(name="L1 Front", surface_type=SurfaceType.STANDARD, radius=19.6, thickness=3.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1 Back", surface_type=SurfaceType.STANDARD, radius=-19.6, thickness=20.0, glass="AIR", semi_diameter=12.7))
    
    # 2. Diffraction Grating (600 lines/mm, 13 deg tilt, reflective)
    stack.add_surface(Surface(
        name="Diffraction Grating", 
        surface_type=SurfaceType.GRATING, 
        radius=0.0, 
        thickness=19.0, 
        glass="MIRROR", 
        semi_diameter=12.7, 
        tilt_x=13.0, 
        grating_lines_per_mm=600.0, 
        grating_order=1
    ))
    
    # 3. Lens 2 (19 mm focal length)
    stack.add_surface(Surface(name="L2 Front", surface_type=SurfaceType.STANDARD, radius=19.6, thickness=3.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2 Back", surface_type=SurfaceType.STANDARD, radius=-19.6, thickness=19.0, glass="AIR", semi_diameter=12.7))
    
    # 4. Digital Mirror Device (DMD) - 15 deg tilt, reflective
    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=10.0, tilt_x=15.0))
    
    # 5. Lens 3 (19 mm focal length)
    stack.add_surface(Surface(name="L3 Front", surface_type=SurfaceType.STANDARD, radius=19.6, thickness=3.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L3 Back", surface_type=SurfaceType.STANDARD, radius=-19.6, thickness=10.0, glass="AIR", semi_diameter=12.7))
    
    # 6. Condenser Lens (20 mm focal length)
    # R ~ 20.6 mm for N-BK7 to get f=20mm
    stack.add_surface(Surface(name="Condenser Front", surface_type=SurfaceType.STANDARD, radius=20.6, thickness=4.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Condenser Back", surface_type=SurfaceType.STANDARD, radius=-20.6, thickness=20.0, glass="AIR", semi_diameter=12.7))
    
    # 7. Single Detector
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=5.0))
    
    # Save the setup
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fiber_spectrometer_setup.json")
    stack.save_to_json(file_path)
    print(f"Setup saved to: {file_path}")

if __name__ == "__main__":
    build_fiber_setup()
