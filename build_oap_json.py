import json
import uuid
import copy
from pathlib import Path
import numpy as np

def build_oap_system():
    src_path = Path("spectrometer_raman_50mm_100mm_manual.json")
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Update system metadata
    data["system_config"]["title"] = "1064nm Raman DMD Spectrometer with Thorlabs MPD119-P01 OAP Mirror"
    data["system_config"]["description"] = (
        "Manual optimized 1064 nm excitation Raman spectrometer featuring a Thorlabs MPD119-P01 "
        "90-degree off-axis parabolic mirror (EFL=25.4mm, 1-inch diameter, protected silver, upward +Y fold) "
        "replacing the chromatic ACL25416U aspheric condenser lens."
    )
    
    # 2. Update spectrometer calibration metadata
    cal = data.setdefault("spectrometer_calibration", {})
    cal["instrument_type"] = "1064 nm excitation Raman spectrometer (All-Reflective Detection Arm)"
    cal["condenser_type"] = "Thorlabs 90 deg Off-Axis Parabolic Mirror"
    cal["condenser_part_number"] = "MPD119-P01"
    cal["condenser_focal_length_mm"] = 25.4
    cal["condenser_diameter_mm"] = 25.4
    cal["condenser_coating"] = "Protected Silver (P01: 450 nm - 20 um)"
    oap_uid = uuid.uuid4().hex
    cal["condenser_component_uid"] = oap_uid
    
    cal["warnings"] = [
        "DMD part DLP4500NIR is an explicit assumption; verify the chip marking and package orientation.",
        "The GR25-0610 catalog blaze is 1.0 um, not the 1.064 um laser line; no measured diffraction-efficiency curve is loaded.",
        "DMD fill factor, window double pass, mirror reflectivity and pixel diffraction are documented but not applied to ray intensity.",
        "OFF rays are rejected by the sequential detector arm; a physical beam dump is not yet placed.",
        "Condenser replaced: Thorlabs MPD119-P01 90-deg OAP mirror (+Y fold) eliminates chromatic aberration and glass absorption in the NIR detection arm."
    ]

    # 3. Fix Air Gap 1, AC254-050-C_S3, Aperture_3, and Air Gap 2
    for s in data["surfaces"]:
        if s.get("name") == "Air Gap 1":
            s["surface_type"] = "Air Space"
            s["glass"] = "AIR"
        elif s.get("name") == "AC254-050-C_S3":
            s["thickness"] = 0.5
        elif s.get("name") == "Aperture_3":
            s["thickness"] = 41.5
        elif s.get("name") == "DMD4500_Micromirrors":
            s["thickness"] = 0.0
        elif s.get("name") == "Air Gap 2":
            s["surface_type"] = "Air Space"
            s["thickness"] = 25.4
            s["glass"] = "AIR"

    # 4. Locate ACL25416U surfaces
    acl1_idx = None
    acl2_idx = None
    for i, s in enumerate(data["surfaces"]):
        name = s.get("name", "")
        if "ACL25416U_S1" in name:
            acl1_idx = i
        elif "ACL25416U_S2" in name:
            acl2_idx = i

    if acl1_idx is None or acl2_idx is None:
        raise ValueError(f"Could not find ACL25416U surfaces in {src_path}")

    # 5. Construct Thorlabs MPD119-P01 OAP Mirror surface
    # EFL = 25.4 mm (1.00 inch), Diameter = 25.4 mm, R = -50.8 mm, conic = -1.0, tilt_x = +45.0 (Upward +Y fold)
    oap_mirror_surface = {
        "name": "MPD119-P01_S1",
        "is_active": True,
        "surface_type": "Mirror",
        "is_coord_break": False,
        "radius": -50.8,
        "thickness": 25.4,
        "glass": "MIRROR",
        "semi_diameter": 12.7,
        "conic": -1.0,
        "asphere_coeffs": [],
        "decenter_x": 0.0,
        "decenter_y": 0.0,
        "decenter_z": 0.0,
        "tilt_x": 45.0,
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
        "aperture_type": "Circular",
        "aperture_x_half": 12.7,
        "aperture_y_half": 12.7,
        "aperture_a_semi": 12.7,
        "aperture_b_semi": 12.7,
        "aperture_r_inner": 0.0,
        "aperture_r_outer": 0.0,
        "aperture_center_radius": 5.0,
        "aperture_spider_width": 1.0,
        "aperture_spider_count": 4,
        "aperture_spider_angle": 0.0,
        "jones_matrix": None,
        "element_id": None,
        "component_uid": oap_uid,
        "catalog_metadata": {
            "manufacturer": "Thorlabs",
            "part_number": "MPD119-P01",
            "section": "parabolic_mirrors",
            "family": "Concave Parabolic Mirror",
            "type": "Parabolic Mirror (Protected Silver (P01))",
            "source_url": "https://www.thorlabs.com/item/MPD119-P01",
            "retrieved_date": "2026-10-15",
            "diameter_mm": 25.4,
            "semi_diameter_mm": 12.7,
            "nominal_focal_length_mm": 25.4,
            "reflected_focal_length_mm": 25.4,
            "parent_focal_length_mm": 12.7,
            "off_axis_angle_deg": 90.0,
            "radius_mm": -50.8,
            "conic_k": -1.0,
            "coating_code": "P01",
            "coating": "Protected Silver",
            "coating_range_nm": [450.0, 20000.0],
            "reflectivity_1064nm": 0.98,
            "center_thickness_mm": 6.0,
            "clear_aperture_percent": 90.0,
            "surface_quality": "40-20 Scratch-Dig",
            "wavefront_error": "< lambda / 2 at 633 nm"
        },
        "group_name": "MPD119-P01",
        "is_locked": True,
        "mechanism_id": "",
        "mechanism_type": "",
        "mechanical_state": "",
        "mechanical_states": {}
    }

    new_surfaces = data["surfaces"][:acl1_idx] + [oap_mirror_surface] + data["surfaces"][acl2_idx + 1:]
    data["surfaces"] = new_surfaces

    # Update optical route if present
    if "optical_routes" in data:
        for route_name, r_info in data.get("optical_routes", {}).items():
            if isinstance(r_info, dict) and "steps" in r_info:
                for step in r_info["steps"]:
                    if step.get("component_uid") == "42a6beb86c8c4d13ae4df32bea31a6d2":
                        step["component_uid"] = oap_uid

    target1 = Path("spectrometer_raman_50mm_100mm_manual_oap.json")
    target2 = Path("spectrometer_raman_50mm_100mm_oap.json")
    
    with open(target1, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {target1} ({len(data['surfaces'])} surfaces)")

    with open(target2, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {target2} ({len(data['surfaces'])} surfaces)")

if __name__ == "__main__":
    build_oap_system()
