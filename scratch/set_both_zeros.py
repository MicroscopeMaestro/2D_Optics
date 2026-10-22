import json

path = r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

surfs = d['surfaces']

# We need:
# S20: DMD4500_Micromirrors with thickness = 0.0
# S21: Air_after_DMD (Air Space) with thickness = 45.0
# followed by the rest of the optics (ACL25416U, Detector)

new_surfs = []
for i, s in enumerate(surfs):
    if s.get('name') == 'DMD4500_Micromirrors':
        s['thickness'] = 0.0
        new_surfs.append(s)
        
        air_after_dmd = {
            "name": "Air_after_DMD",
            "is_active": True,
            "surface_type": "Air Space",
            "is_coord_break": False,
            "radius": 0.0,
            "thickness": 45.0,
            "glass": "AIR",
            "semi_diameter": 12.7,
            "conic": 0.0,
            "asphere_coeffs": [],
            "decenter_x": 0.0, "decenter_y": 0.0, "decenter_z": 0.0,
            "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0,
            "grating_lines_per_mm": 0.0, "grating_order": 1,
            "substrate_thickness": 0.0, "substrate_shape": "Circular",
            "thickness_variable": False, "thickness_min": 0.0, "thickness_max": 1000.0,
            "radius_variable": False, "radius_min": -1000.0, "radius_max": 1000.0,
            "tilt_x_variable": False, "tilt_x_min": -45.0, "tilt_x_max": 45.0,
            "tilt_y_variable": False, "tilt_y_min": -45.0, "tilt_y_max": 45.0,
            "tilt_z_variable": False, "tilt_z_min": -45.0, "tilt_z_max": 45.0,
            "decenter_x_variable": False, "decenter_x_min": -25.0, "decenter_x_max": 25.0,
            "decenter_y_variable": False, "decenter_y_min": -25.0, "decenter_y_max": 25.0,
            "decenter_z_variable": False, "decenter_z_min": -25.0, "decenter_z_max": 25.0,
            "extinction_coeff": 0.0,
            "aperture_type": "Circular",
            "aperture_x_half": 12.7,
            "aperture_y_half": 12.7,
            "aperture_a_semi": 20.0, "aperture_b_semi": 15.0,
            "aperture_r_inner": 0.0, "aperture_r_outer": 12.7,
            "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
            "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
            "jones_matrix": None, "element_id": None, "group_name": "", "is_locked": False
        }
        new_surfs.append(air_after_dmd)
    else:
        new_surfs.append(s)

d['surfaces'] = new_surfs
with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=4)

print("Updated desktop layout: Grating t=0 + Air_after_grating (20mm), DMD t=0 + Air_after_DMD (45mm)")
