import json

system_data = {
    "surfaces": [
        {
            "name": "Fiber Object",
            "surface_type": "Standard",
            "radius": 0.0,
            "thickness": 33.8,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 0.025,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        # Collimator
        {
            "name": "Coll_Front",
            "surface_type": "Standard",
            "radius": 57.7,
            "thickness": 2.5,
            "thickness_variable": False,
            "glass": "SF5",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        {
            "name": "Coll_Mid",
            "surface_type": "Standard",
            "radius": 20.1,
            "thickness": 10.0,
            "thickness_variable": False,
            "glass": "N-BK7",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        {
            "name": "Coll_Back",
            "surface_type": "Standard",
            "radius": -23.7,
            "thickness": 15.0,
            "thickness_variable": False,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        # Grating
        {
            "name": "Diffraction_Grating",
            "surface_type": "Grating",
            "radius": 0.0,
            "thickness": 15.0,
            "thickness_variable": False,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "grating_lines_per_mm": 600.0,
            "grating_order": 1,
            "tilt_x": 0.0
        },
        # Focuser 1
        {
            "name": "Foc1_Front",
            "surface_type": "Standard",
            "radius": 23.7,
            "thickness": 10.0,
            "thickness_variable": False,
            "glass": "N-BK7",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        {
            "name": "Foc1_Mid",
            "surface_type": "Standard",
            "radius": -20.1,
            "thickness": 2.5,
            "thickness_variable": False,
            "glass": "SF5",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        {
            "name": "Foc1_Back",
            "surface_type": "Standard",
            "radius": -57.7,
            "thickness": 33.8,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        # DMD Mirror (FOLDED GEOMETRY)
        # We tilt it by 12 degrees to simulate the 'ON' state of the micromirrors
        # glass="MIRROR" triggers the tracer and LDE to physically reflect the ray path!
        {
            "name": "DMD_Mirror",
            "surface_type": "Standard",
            "radius": 0.0,
            "thickness": 33.8,
            "thickness_variable": True,
            "glass": "MIRROR",
            "semi_diameter": 10.0,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 12.0
        },
        # Focuser 2 (Collector lens for Single Detector)
        # Path is now traveling along the reflected axis!
        {
            "name": "Foc2_Front",
            "surface_type": "Standard",
            "radius": 57.7,
            "thickness": 2.5,
            "thickness_variable": False,
            "glass": "SF5",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        {
            "name": "Foc2_Mid",
            "surface_type": "Standard",
            "radius": 20.1,
            "thickness": 10.0,
            "thickness_variable": False,
            "glass": "N-BK7",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        {
            "name": "Foc2_Back",
            "surface_type": "Standard",
            "radius": -23.7,
            "thickness": 33.8,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        # Final Single Detector
        {
            "name": "Single_Detector",
            "surface_type": "Detector",
            "radius": 0.0,
            "thickness": 0.0,
            "thickness_variable": False,
            "glass": "AIR",
            "semi_diameter": 5.0,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        }
    ],
    "wavelengths": [0.5, 0.588, 0.65]
}

with open("full_spectrometer_dmd_detector.json", "w") as f:
    json.dump(system_data, f, indent=4)
print("Saved FOLDED full_spectrometer_dmd_detector.json")
