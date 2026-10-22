import json
import os

path = "mirror_spectrometer/Spectrometer_All_Mirror_Optimized.json"
out_path = "mirror_spectrometer/Spectrometer_All_Mirror_With_AirGaps.json"

with open(path, 'r') as f:
    data = json.load(f)

new_surfaces = []
for i, surf in enumerate(data['surfaces']):
    # If the surface has a thickness > 0 and is not air, we will extract the thickness to an explicit air gap
    thickness = surf.get('thickness', 0.0)
    
    if surf.get('surface_type') in ['Mirror', 'Grating', 'Detector'] or surf.get('glass') == 'MIRROR':
        surf['thickness'] = 0.0
        new_surfaces.append(surf)
        
        if thickness > 0:
            air_gap = {
                "name": f"Air_Gap_After_{surf['name']}",
                "is_active": True,
                "surface_type": "Standard",
                "is_coord_break": False,
                "radius": 0.0,
                "thickness": thickness,
                "glass": "AIR",
                "semi_diameter": surf.get("semi_diameter", 12.7),
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
                "aperture_x_half": surf.get("semi_diameter", 12.7),
                "aperture_y_half": surf.get("semi_diameter", 12.7),
                "aperture_a_semi": 20.0, "aperture_b_semi": 15.0,
                "aperture_r_inner": 0.0, "aperture_r_outer": surf.get("semi_diameter", 12.7),
                "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
                "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
                "jones_matrix": None, "element_id": None, "group_name": "", "is_locked": False
            }
            new_surfaces.append(air_gap)
    else:
        new_surfaces.append(surf)

data['surfaces'] = new_surfaces

with open(out_path, 'w') as f:
    json.dump(data, f, indent=4)
print("Saved:", out_path)
