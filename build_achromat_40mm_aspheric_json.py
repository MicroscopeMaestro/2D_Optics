import json
import uuid
import copy
from pathlib import Path

def create_spectrometer_40mm_aspheric_json():
    src_path = Path("spectrometer_raman_50mm_100mm_manual.json")
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Update system metadata
    data["system_config"]["title"] = "1064nm Raman DMD Spectrometer with Thorlabs AC254-040-C Relay & ACL25416U Condenser"
    data["system_config"]["description"] = (
        "Manual optimized 1064 nm excitation Raman spectrometer featuring a 2-lens detection arm: "
        "a Thorlabs AC254-040-C achromatic doublet (EFL=40mm, 1-inch diameter, NIR C-coated 1050-1700nm) "
        "as a collimating relay after the DMD, followed by a Thorlabs ACL25416U aspheric condenser lens (EFL=16mm) "
        "focusing onto the single-element detector."
    )
    
    # 2. Update spectrometer calibration metadata
    cal = data.setdefault("spectrometer_calibration", {})
    cal["instrument_type"] = "1064 nm excitation Raman spectrometer (Achromat Relay + Aspheric Condenser Detection Arm)"
    cal["relay_type"] = "Thorlabs Achromatic Doublet"
    cal["relay_part_number"] = "AC254-040-C"
    cal["relay_focal_length_mm"] = 40.0
    cal["relay_diameter_mm"] = 25.4
    cal["relay_coating"] = "C-Coating (1050 - 1700 nm)"
    relay_uid = uuid.uuid4().hex
    cal["relay_component_uid"] = relay_uid

    cal["condenser_type"] = "Thorlabs Aspheric Condenser Lens"
    cal["condenser_part_number"] = "ACL25416U"
    cal["condenser_focal_length_mm"] = 16.0
    cal["condenser_diameter_mm"] = 25.4
    condenser_uid = uuid.uuid4().hex
    cal["condenser_component_uid"] = condenser_uid
    
    cal["warnings"] = [
        "DMD part DLP4500NIR is an explicit assumption; verify the chip marking and package orientation.",
        "The GR25-0610 catalog blaze is 1.0 um, not the 1.064 um laser line; no measured diffraction-efficiency curve is loaded.",
        "DMD fill factor, window double pass, mirror reflectivity and pixel diffraction are documented but not applied to ray intensity.",
        "OFF rays are rejected by the sequential detector arm; a physical beam dump is not yet placed.",
        "Detection Arm: Thorlabs AC254-040-C collimating relay + ACL25416U condenser provides tight achromatic focus at 1064-1350 nm onto the detector."
    ]

    # 3. Ensure ray transmission fixes before the DMD
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

    # 4. Construct Detection Arm: Air Gap 2 (DMD to Relay), AC254-040-C, Relay Gap, ACL25416U, Detector
    # Locate DMD surface index
    dmd_idx = next(i for i, s in enumerate(data["surfaces"]) if "DMD" in s.get("name", ""))
    
    air_gap_2 = {
        "name": "Air Gap 2",
        "is_active": True,
        "surface_type": "Air Space",
        "is_coord_break": False,
        "radius": 0.0,
        "thickness": 32.8,  # Thorlabs AC254-040-C BFL to DMD focus
        "glass": "AIR",
        "semi_diameter": 12.7,
        "conic": 0.0,
        "asphere_coeffs": [],
        "decenter_x": 0.0, "decenter_y": 0.0, "decenter_z": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0,
        "grating_lines_per_mm": 0.0, "grating_order": 1, "grating_primary_wavelength_um": None,
        "cad_profile": None, "substrate_thickness": 6.0, "substrate_shape": "Auto",
        "extinction_coeff": 0.0, "aperture_type": "Circular",
        "aperture_x_half": 12.7, "aperture_y_half": 12.7,
        "aperture_a_semi": 12.7, "aperture_b_semi": 12.7,
        "aperture_r_inner": 0.0, "aperture_r_outer": 0.0,
        "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
        "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
        "jones_matrix": None, "element_id": None, "component_uid": "",
        "group_name": "Relay Spacing", "is_locked": False,
        "mechanism_id": "", "mechanism_type": "", "mechanical_state": "", "mechanical_states": {}
    }

    # AC254-040-C (Flipped for short-conjugate collimation: Flint S1 R=143.9, Crown S2 R=21.1, S3 R=-24.4)
    ac_s1 = {
        "name": "Relay_AC254-040-C_S1",
        "is_active": True,
        "surface_type": "Standard",
        "is_coord_break": False,
        "radius": 143.9,
        "thickness": 1.8,
        "glass": "N-SF6HT",
        "semi_diameter": 12.7,
        "conic": 0.0,
        "asphere_coeffs": [],
        "decenter_x": 0.0, "decenter_y": 0.0, "decenter_z": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0,
        "grating_lines_per_mm": 0.0, "grating_order": 1, "grating_primary_wavelength_um": None,
        "cad_profile": None, "substrate_thickness": 6.0, "substrate_shape": "Auto",
        "extinction_coeff": 0.0, "aperture_type": "Circular",
        "aperture_x_half": 12.7, "aperture_y_half": 12.7,
        "aperture_a_semi": 12.7, "aperture_b_semi": 12.7,
        "aperture_r_inner": 0.0, "aperture_r_outer": 0.0,
        "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
        "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
        "jones_matrix": None, "element_id": None, "component_uid": relay_uid,
        "catalog_metadata": {
            "manufacturer": "Thorlabs",
            "part_number": "AC254-040-C",
            "section": "nir_achromats_c",
            "family": "NIR Achromatic Doublets",
            "type": "Achromatic Doublet (C-Coated: 1050 - 1700 nm)",
            "source_url": "https://www.thorlabs.com/item/AC254-040-C",
            "diameter_mm": 25.4,
            "semi_diameter_mm": 12.7,
            "focal_length_mm": 40.0,
            "back_focal_length_mm": 32.8,
            "center_thickness_mm": 11.8,
            "clear_aperture_percent": 90.0,
            "coating": "C-Coating (1050-1700 nm)"
        },
        "group_name": "AC254-040-C", "is_locked": True,
        "mechanism_id": "", "mechanism_type": "", "mechanical_state": "", "mechanical_states": {}
    }

    ac_s2 = {
        "name": "Relay_AC254-040-C_S2",
        "is_active": True,
        "surface_type": "Standard",
        "is_coord_break": False,
        "radius": 21.1,
        "thickness": 10.0,
        "glass": "N-LAK22",
        "semi_diameter": 12.7,
        "conic": 0.0,
        "asphere_coeffs": [],
        "decenter_x": 0.0, "decenter_y": 0.0, "decenter_z": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0,
        "grating_lines_per_mm": 0.0, "grating_order": 1, "grating_primary_wavelength_um": None,
        "cad_profile": None, "substrate_thickness": 6.0, "substrate_shape": "Auto",
        "extinction_coeff": 0.0, "aperture_type": "Circular",
        "aperture_x_half": 12.7, "aperture_y_half": 12.7,
        "aperture_a_semi": 12.7, "aperture_b_semi": 12.7,
        "aperture_r_inner": 0.0, "aperture_r_outer": 0.0,
        "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
        "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
        "jones_matrix": None, "element_id": None, "component_uid": relay_uid,
        "group_name": "AC254-040-C", "is_locked": True,
        "mechanism_id": "", "mechanism_type": "", "mechanical_state": "", "mechanical_states": {}
    }

    ac_s3 = {
        "name": "Relay_AC254-040-C_S3",
        "is_active": True,
        "surface_type": "Standard",
        "is_coord_break": False,
        "radius": -24.4,
        "thickness": 2.0,  # Air gap between 40mm achromat and aspheric condenser
        "glass": "AIR",
        "semi_diameter": 12.7,
        "conic": 0.0,
        "asphere_coeffs": [],
        "decenter_x": 0.0, "decenter_y": 0.0, "decenter_z": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0,
        "grating_lines_per_mm": 0.0, "grating_order": 1, "grating_primary_wavelength_um": None,
        "cad_profile": None, "substrate_thickness": 6.0, "substrate_shape": "Auto",
        "extinction_coeff": 0.0, "aperture_type": "Circular",
        "aperture_x_half": 12.7, "aperture_y_half": 12.7,
        "aperture_a_semi": 12.7, "aperture_b_semi": 12.7,
        "aperture_r_inner": 0.0, "aperture_r_outer": 0.0,
        "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
        "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
        "jones_matrix": None, "element_id": None, "component_uid": relay_uid,
        "group_name": "AC254-040-C", "is_locked": True,
        "mechanism_id": "", "mechanism_type": "", "mechanical_state": "", "mechanical_states": {}
    }

    # ACL25416U Aspheric Condenser Lens
    acl_s1 = {
        "name": "Condenser_ACL25416U_S1",
        "is_active": True,
        "surface_type": "Aspheric",
        "is_coord_break": False,
        "radius": 8.3,
        "thickness": 14.0,
        "glass": "B270",
        "semi_diameter": 12.7,
        "conic": -0.6265,
        "asphere_coeffs": [],
        "decenter_x": 0.0, "decenter_y": 0.0, "decenter_z": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0,
        "grating_lines_per_mm": 0.0, "grating_order": 1, "grating_primary_wavelength_um": None,
        "cad_profile": None, "substrate_thickness": 6.0, "substrate_shape": "Auto",
        "extinction_coeff": 0.0, "aperture_type": "Circular",
        "aperture_x_half": 12.7, "aperture_y_half": 12.7,
        "aperture_a_semi": 12.7, "aperture_b_semi": 12.7,
        "aperture_r_inner": 0.0, "aperture_r_outer": 0.0,
        "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
        "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
        "jones_matrix": None, "element_id": None, "component_uid": condenser_uid,
        "catalog_metadata": {
            "manufacturer": "Thorlabs",
            "part_number": "ACL25416U",
            "section": "aspheric_condenser_lenses",
            "family": "Aspheric Condenser Lenses",
            "type": "Precision Aspheric Condenser Lens",
            "source_url": "https://www.thorlabs.com/item/ACL25416U",
            "diameter_mm": 25.4,
            "semi_diameter_mm": 12.7,
            "focal_length_mm": 16.0,
            "back_focal_length_mm": 8.5,
            "center_thickness_mm": 14.0,
            "clear_aperture_percent": 90.0,
            "conic_k": -0.6265,
            "material": "B270"
        },
        "group_name": "ACL25416U", "is_locked": True,
        "mechanism_id": "", "mechanism_type": "", "mechanical_state": "", "mechanical_states": {}
    }

    acl_s2 = {
        "name": "Condenser_ACL25416U_S2",
        "is_active": True,
        "surface_type": "Standard",
        "is_coord_break": False,
        "radius": 0.0,
        "thickness": 4.5,  # Focus onto detector
        "glass": "AIR",
        "semi_diameter": 12.7,
        "conic": 0.0,
        "asphere_coeffs": [],
        "decenter_x": 0.0, "decenter_y": 0.0, "decenter_z": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0,
        "grating_lines_per_mm": 0.0, "grating_order": 1, "grating_primary_wavelength_um": None,
        "cad_profile": None, "substrate_thickness": 6.0, "substrate_shape": "Auto",
        "extinction_coeff": 0.0, "aperture_type": "Circular",
        "aperture_x_half": 12.7, "aperture_y_half": 12.7,
        "aperture_a_semi": 12.7, "aperture_b_semi": 12.7,
        "aperture_r_inner": 0.0, "aperture_r_outer": 0.0,
        "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
        "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
        "jones_matrix": None, "element_id": None, "component_uid": condenser_uid,
        "group_name": "ACL25416U", "is_locked": True,
        "mechanism_id": "", "mechanism_type": "", "mechanical_state": "", "mechanical_states": {}
    }

    detector = {
        "name": "S25 (IMA)",
        "is_active": True,
        "surface_type": "Detector",
        "is_coord_break": False,
        "radius": 0.0,
        "thickness": 0.0,
        "glass": "AIR",
        "semi_diameter": 0.4,
        "conic": 0.0,
        "asphere_coeffs": [],
        "decenter_x": 0.0, "decenter_y": 0.0, "decenter_z": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0,
        "grating_lines_per_mm": 0.0, "grating_order": 1, "grating_primary_wavelength_um": None,
        "cad_profile": None, "substrate_thickness": 6.0, "substrate_shape": "Auto",
        "extinction_coeff": 0.0, "aperture_type": "Circular",
        "aperture_x_half": 12.7, "aperture_y_half": 12.7,
        "aperture_a_semi": 12.7, "aperture_b_semi": 12.7,
        "aperture_r_inner": 0.0, "aperture_r_outer": 0.0,
        "aperture_center_radius": 5.0, "aperture_spider_width": 1.0,
        "aperture_spider_count": 4, "aperture_spider_angle": 0.0,
        "jones_matrix": None, "element_id": None, "component_uid": "",
        "group_name": "Detector", "is_locked": False,
        "mechanism_id": "", "mechanism_type": "", "mechanical_state": "", "mechanical_states": {}
    }

    # Reconstruct surfaces: 0..21 (DMD), air_gap_2, ac_s1, ac_s2, ac_s3, acl_s1, acl_s2, detector
    new_surfaces = data["surfaces"][:dmd_idx + 1] + [
        air_gap_2, ac_s1, ac_s2, ac_s3, acl_s1, acl_s2, detector
    ]
    data["surfaces"] = new_surfaces

    target1 = Path("spectrometer_raman_50mm_100mm_manual_40mm_aspheric.json")
    target2 = Path("spectrometer_raman_40mm_relay_aspheric.json")

    with open(target1, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {target1} ({len(data['surfaces'])} surfaces)")

    with open(target2, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {target2} ({len(data['surfaces'])} surfaces)")

if __name__ == "__main__":
    create_spectrometer_40mm_aspheric_json()
