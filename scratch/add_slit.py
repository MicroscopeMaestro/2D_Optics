import json
import copy

with open('spectrometer_yz_40mm_100mm_optimized.json', 'r') as f:
    config = json.load(f)

slit_surface = {
    "name": "Entrance_Slit",
    "is_active": True,
    "surface_type": "Stop",
    "is_coord_break": False,
    "radius": 0.0,
    "thickness": 19.31,
    "glass": "AIR",
    "semi_diameter": 12.7,
    "conic": 0.0,
    "asphere_coeffs": [],
    "decenter_x": 0.0,
    "decenter_y": 0.0,
    "decenter_z": 0.0,
    "tilt_x": 0.0,
    "tilt_y": 0.0,
    "tilt_z": 0.0,
    "grating_lines_per_mm": 0.0,
    "grating_order": 1,
    "grating_primary_wavelength_um": None,
    "cad_profile": None,
    "substrate_thickness": 6.0,
    "substrate_shape": "Auto",
    "thickness_variable": False,
    "thickness_min": 0.0,
    "thickness_max": 1000.0,
    "radius_variable": False,
    "radius_min": -1000.0,
    "radius_max": 1000.0,
    "tilt_x_variable": False,
    "tilt_x_min": -45.0,
    "tilt_x_max": 45.0,
    "tilt_y_variable": False,
    "tilt_y_min": -45.0,
    "tilt_y_max": 45.0,
    "tilt_z_variable": False,
    "tilt_z_min": -45.0,
    "tilt_z_max": 45.0,
    "decenter_x_variable": False,
    "decenter_x_min": -25.0,
    "decenter_x_max": 25.0,
    "decenter_y_variable": False,
    "decenter_y_min": -25.0,
    "decenter_y_max": 25.0,
    "decenter_z_variable": False,
    "decenter_z_min": -25.0,
    "decenter_z_max": 25.0,
    "extinction_coeff": 0.0,
    "aperture_type": "Rectangular",
    "aperture_x_half": 0.5,
    "aperture_y_half": 0.0125,
    "aperture_a_semi": 0.0,
    "aperture_b_semi": 0.0,
    "aperture_r_inner": 0.0,
    "aperture_r_outer": 0.0,
    "aperture_center_radius": 5.0,
    "aperture_spider_width": 1.0,
    "aperture_spider_count": 4,
    "aperture_spider_angle": 0.0,
    "jones_matrix": None,
    "element_id": None,
    "group_name": "Slit",
    "is_locked": False
}

# Insert slit after surface 5
config['surfaces'].insert(6, slit_surface)

# Optional: ensure we have multiple field points to simulate a line source
# A 25um slit means y goes from -0.0125 to +0.0125 mm
config['system_config']['fields'] = {
    "type": "Object Height (mm)",
    "points": [
        {"x": 0.0, "y": 0.0, "weight": 1.0},
        {"x": 0.0, "y": 0.0125, "weight": 1.0},
        {"x": 0.0, "y": -0.0125, "weight": 1.0}
    ]
}

config['system_config']['aperture'] = {
    "beam_type": "Diverging (Point/Slit Source)",
    "type": "Entrance Pupil Diameter",
    "value": 25.4,
    "slit_width_um": 25.0,
    "apodization_type": "Uniform",
    "apodization_factor": 0.0,
    "telecentric": False
}

with open('spectrometer_yz_40mm_100mm_slit.json', 'w') as f:
    json.dump(config, f, indent=4)

print("Saved spectrometer_yz_40mm_100mm_slit.json")
