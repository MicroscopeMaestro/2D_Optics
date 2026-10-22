import json
import math
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import save_system

stack = SurfaceStack()

# Object (Fiber)
stack.add_surface(Surface(
    name="Fiber Object",
    surface_type=SurfaceType.STANDARD,
    radius=0.0,
    thickness=33.8,  # Approximate BFL of AC254-040-A
    thickness_variable=True,  # WE WILL OPTIMIZE THIS
    glass_name="AIR",
    semi_diameter=0.025
))

# Collimator Doublet: Thorlabs AC254-040-A
# Materials: N-BK7 and SF5. Radii: 23.7, -20.1, -57.7. Thicknesses: 10.0, 2.5.
stack.add_surface(Surface(
    name="Coll_Front",
    surface_type=SurfaceType.STANDARD,
    radius=23.7,
    thickness=10.0,
    glass_name="N-BK7",
    semi_diameter=12.7
))
stack.add_surface(Surface(
    name="Coll_Mid",
    surface_type=SurfaceType.STANDARD,
    radius=-20.1,
    thickness=2.5,
    glass_name="SF5",
    semi_diameter=12.7
))
stack.add_surface(Surface(
    name="Coll_Back",
    surface_type=SurfaceType.STANDARD,
    radius=-57.7,
    thickness=15.0,  # Air gap to grating
    glass_name="AIR",
    semi_diameter=12.7
))

# Grating
stack.add_surface(Surface(
    name="Diffraction_Grating",
    surface_type=SurfaceType.GRATING,
    radius=0.0,
    thickness=15.0,  # Air gap to focuser
    glass_name="AIR",
    semi_diameter=12.7,
    grating_lines_per_mm=600.0,
    grating_order=1
))

# Focuser Doublet: Thorlabs AC254-050-A
# Actually, the search showed R1=43.4, R2=33.3, R3=-22.3 for N-BAF10 / N-SF10. Let's use N-BAF10 and N-SF6 as an approx, or we can use another AC254-040-A.
# A 40mm focal length for focusing will work perfectly and provide 1:1 magnification, meaning the 25um fiber spot will map to a 25um spot on the DMD.
# Wait, let's use the AC254-040-A for the focuser too. BUT flipped! 
# A focusing doublet should be oriented with the steeper curve toward the collimated beam (infinite conjugate).
# For AC254-040-A, R1=23.7 is convex, R3=-57.7 is concave. The 23.7 side should face the collimated beam.
stack.add_surface(Surface(
    name="Focuser_Front",
    surface_type=SurfaceType.STANDARD,
    radius=23.7,
    thickness=10.0,
    glass_name="N-BK7",
    semi_diameter=12.7
))
stack.add_surface(Surface(
    name="Focuser_Mid",
    surface_type=SurfaceType.STANDARD,
    radius=-20.1,
    thickness=2.5,
    glass_name="SF5",
    semi_diameter=12.7
))
stack.add_surface(Surface(
    name="Focuser_Back",
    surface_type=SurfaceType.STANDARD,
    radius=-57.7,
    thickness=33.8,  # Air gap to DMD, we will optimize this
    thickness_variable=True,
    glass_name="AIR",
    semi_diameter=12.7
))

# DMD (Detector)
# The user wants to optimize the inclination angle (tilt) of the DMD.
stack.add_surface(Surface(
    name="DMD_Detector",
    surface_type=SurfaceType.DETECTOR,
    radius=0.0,
    thickness=0.0,
    glass_name="AIR",
    semi_diameter=10.0,
    tilt_x=0.0, # We will optimize this!
    tilt_x_variable=True
))

wls = [0.5, 0.588, 0.65] # Blue, Yellow, Red

save_system(stack, wls, 'thorlabs_spectrometer.json')
print("Saved thorlabs_spectrometer.json")
