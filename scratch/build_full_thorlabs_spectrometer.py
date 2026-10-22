import json

# Original thicknesses
t_fiber = 33.8  # Focal length of AC254-040-A
t_l1_back = 21.02
t_grating = 13.33
t_l2_back = 20.85
t_dmd = 22.90
t_l3_back = 35.10
t_condenser_back = 21.21

system_data = {
    "surfaces": [
        {
            "name": "Fiber Object",
            "surface_type": "Standard",
            "radius": 0.0,
            "thickness": t_fiber,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 0.025,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        # Collimator L1: AC254-040-A (Reversed: Flat side faces fiber)
        {
            "name": "L1 Front",
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
            "name": "L1 Middle",
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
            "name": "L1 Back",
            "surface_type": "Standard",
            "radius": -23.7,
            "thickness": t_l1_back,
            "thickness_variable": True,
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
            "thickness": t_grating,
            "thickness_variable": True,
            "glass": "MIRROR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "grating_lines_per_mm": 600.0,
            "grating_order": 1,
            "tilt_x": 13.0
        },
        # Focusing lens L2: AC254-040-A (Forward: Curved side faces grating)
        {
            "name": "L2 Front",
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
            "name": "L2 Middle",
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
            "name": "L2 Back",
            "surface_type": "Standard",
            "radius": -57.7,
            "thickness": t_l2_back,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        # DMD Mirror (FOLDED)
        {
            "name": "DMD_Mirror",
            "surface_type": "Standard",
            "radius": 0.0,
            "thickness": t_dmd,
            "thickness_variable": True,
            "glass": "MIRROR",
            "semi_diameter": 10.0,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 15.0
        },
        # Relay lens L3: AC254-040-A 
        {
            "name": "L3 Front",
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
            "name": "L3 Middle",
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
            "name": "L3 Back",
            "surface_type": "Standard",
            "radius": -57.7,
            "thickness": t_l3_back,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        # Condenser (From original setup)
        {
            "name": "Condenser Front",
            "surface_type": "Aspheric",
            "radius": 10.462,
            "thickness": 9.83,
            "thickness_variable": False,
            "glass": "B270",
            "semi_diameter": 12.5,
            "radius_variable": False,
            "conic": -0.6265,
            "asphere_coeffs": [1.5e-05],
            "tilt_x": 0.0
        },
        {
            "name": "Condenser Back",
            "surface_type": "Standard",
            "radius": 0.0,
            "thickness": t_condenser_back,
            "thickness_variable": True,
            "glass": "AIR",
            "semi_diameter": 12.5,
            "radius_variable": False,
            "conic": 0.0,
            "tilt_x": 0.0
        },
        # Final Single Detector
        {
            "name": "Detector",
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

with open("fiber_spectrometer_thorlabs_full.json", "w") as f:
    json.dump(system_data, f, indent=4)
print("Saved exact match fiber_spectrometer_thorlabs_full.json")
