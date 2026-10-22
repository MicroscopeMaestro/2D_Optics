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
        # FLIPPED COLLIMATOR: Flat side (Flint) faces the fiber
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
        # FORWARD FOCUSER: Curved side (Crown) faces the grating
        {
            "name": "Focuser_Front",
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
            "name": "Focuser_Mid",
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
            "name": "Focuser_Back",
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
        {
            "name": "DMD_Detector",
            "surface_type": "Detector",
            "radius": 0.0,
            "thickness": 0.0,
            "thickness_variable": False,
            "glass": "AIR",
            "semi_diameter": 10.0,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        }
    ],
    "wavelengths": [0.5, 0.588, 0.65]
}

with open("thorlabs_spectrometer.json", "w") as f:
    json.dump(system_data, f, indent=4)
print("Saved corrected thorlabs_spectrometer.json")
