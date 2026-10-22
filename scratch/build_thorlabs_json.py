import json

system_data = {
    "surfaces": [
        {
            "name": "Fiber Object",
            "type": "STANDARD",
            "radius": 0.0,
            "thickness": 33.8,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 0.025,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0,
            "tilt_x_variable": False
        },
        {
            "name": "Coll_Front",
            "type": "STANDARD",
            "radius": 23.7,
            "thickness": 10.0,
            "thickness_variable": False,
            "glass": "N-BK7",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0,
            "tilt_x_variable": False
        },
        {
            "name": "Coll_Mid",
            "type": "STANDARD",
            "radius": -20.1,
            "thickness": 2.5,
            "thickness_variable": False,
            "glass": "SF5",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0,
            "tilt_x_variable": False
        },
        {
            "name": "Coll_Back",
            "type": "STANDARD",
            "radius": -57.7,
            "thickness": 15.0,
            "thickness_variable": False,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0,
            "tilt_x_variable": False
        },
        {
            "name": "Diffraction_Grating",
            "type": "GRATING",
            "radius": 0.0,
            "thickness": 15.0,
            "thickness_variable": False,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "grating_lines_per_mm": 600.0,
            "grating_order": 1,
            "tilt_x": 0.0,
            "tilt_x_variable": False
        },
        {
            "name": "Focuser_Front",
            "type": "STANDARD",
            "radius": 23.7,
            "thickness": 10.0,
            "thickness_variable": False,
            "glass": "N-BK7",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0,
            "tilt_x_variable": False
        },
        {
            "name": "Focuser_Mid",
            "type": "STANDARD",
            "radius": -20.1,
            "thickness": 2.5,
            "thickness_variable": False,
            "glass": "SF5",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0,
            "tilt_x_variable": False
        },
        {
            "name": "Focuser_Back",
            "type": "STANDARD",
            "radius": -57.7,
            "thickness": 33.8,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0,
            "tilt_x_variable": False
        },
        {
            "name": "DMD_Detector",
            "type": "DETECTOR",
            "radius": 0.0,
            "thickness": 0.0,
            "thickness_variable": False,
            "glass": "AIR",
            "semi_diameter": 10.0,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0,
            "tilt_x_variable": True
        }
    ],
    "wavelengths": [0.5, 0.588, 0.65]
}

with open("thorlabs_spectrometer.json", "w") as f:
    json.dump(system_data, f, indent=4)
print("Saved thorlabs_spectrometer.json")
