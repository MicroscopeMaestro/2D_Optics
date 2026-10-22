import re

with open("optics_app/catalogs/thorlabs.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. New helper definitions to replace _polarization_cube
new_helpers = '''def _amplitude_cube(
    part_number: str,
    dimension_mm: float,
    coating_range_nm: Tuple[float, float],
    split_ratio_R: float = 0.50,
    split_ratio_T: float = 0.50,
    substrate_glass: str = "N-BK7",
    coating_code: str = "VIS",
    design_wavelengths_nm: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if design_wavelengths_nm is None:
        mid = 0.5 * (coating_range_nm[0] + coating_range_nm[1])
        design_wavelengths_nm = [mid]

    cfg = {
        "splitter_type": "Cube",
        "coating_type": "NPBS_50_50" if (abs(split_ratio_R - 0.5) < 1e-4 and abs(split_ratio_T - 0.5) < 1e-4) else "Custom",
        "split_ratio_R": float(split_ratio_R),
        "split_ratio_T": float(split_ratio_T),
        "substrate_glass": substrate_glass,
        "substrate_thickness_mm": float(dimension_mm) / 2.0,
        "cube_side_length_mm": float(dimension_mm),
        "clear_aperture_mm": float(dimension_mm),
        "orientation_angle_deg": 45.0,
        "is_polarizing": False,
        "Rs": float(split_ratio_R),
        "Rp": float(split_ratio_R),
        "Ts": float(split_ratio_T),
        "Tp": float(split_ratio_T),
        "extinction_ratio": 1.0,
        "part_number": part_number,
    }

    ratio_str = f"{int(round(split_ratio_R*100))}:{int(round(split_ratio_T*100))}"
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "amplitude_beamsplitters",
        "family": f"Non-Polarizing Beamsplitter Cube ({ratio_str})",
        "type": f"{ratio_str} Beamsplitter Cube ({part_number})",
        "coating_code": coating_code,
        "coating_range_nm": list(coating_range_nm),
        "design_wavelengths_nm": design_wavelengths_nm,
        "diameter_mm": float(dimension_mm),
        "semi_diameter": float(dimension_mm) / 2.0,
        "dimensions_mm": f"{dimension_mm} x {dimension_mm} x {dimension_mm}",
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(dimension_mm),
        "split_ratio_R": float(split_ratio_R),
        "split_ratio_T": float(split_ratio_T),
        "is_beam_splitter": True,
        "is_polarizing": False,
        "beamsplitter_config": cfg,
        "surfaces": [
            {"radius": 0.0, "thickness": float(dimension_mm) / 2.0, "glass": substrate_glass},
            {"radius": 0.0, "thickness": float(dimension_mm) / 2.0, "glass": substrate_glass, "tilt_x": 45.0, "is_beam_splitter": True, "beamsplitter_config": cfg},
        ],
    }


def _amplitude_plate(
    part_number: str,
    diameter_mm: float,
    thickness_mm: float,
    coating_range_nm: Tuple[float, float],
    split_ratio_R: float = 0.50,
    split_ratio_T: float = 0.50,
    substrate_glass: str = "FUSED_SILICA",
    coating_code: str = "VIS",
    aoi_deg: float = 45.0,
    design_wavelengths_nm: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if design_wavelengths_nm is None:
        mid = 0.5 * (coating_range_nm[0] + coating_range_nm[1])
        design_wavelengths_nm = [mid]

    cfg = {
        "splitter_type": "Plate",
        "coating_type": "NPBS_50_50" if (abs(split_ratio_R - 0.5) < 1e-4 and abs(split_ratio_T - 0.5) < 1e-4) else "Custom",
        "split_ratio_R": float(split_ratio_R),
        "split_ratio_T": float(split_ratio_T),
        "substrate_glass": substrate_glass,
        "substrate_thickness_mm": float(thickness_mm),
        "cube_side_length_mm": float(diameter_mm),
        "clear_aperture_mm": float(diameter_mm),
        "orientation_angle_deg": float(aoi_deg),
        "is_polarizing": False,
        "Rs": float(split_ratio_R),
        "Rp": float(split_ratio_R),
        "Ts": float(split_ratio_T),
        "Tp": float(split_ratio_T),
        "extinction_ratio": 1.0,
        "part_number": part_number,
    }

    ratio_str = f"{int(round(split_ratio_R*100))}:{int(round(split_ratio_T*100))}"
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "amplitude_beamsplitters",
        "family": f"Non-Polarizing Plate Beamsplitter ({ratio_str})",
        "type": f"{ratio_str} Plate Beamsplitter ({part_number})",
        "coating_code": coating_code,
        "coating_range_nm": list(coating_range_nm),
        "design_wavelengths_nm": design_wavelengths_nm,
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(thickness_mm),
        "split_ratio_R": float(split_ratio_R),
        "split_ratio_T": float(split_ratio_T),
        "angle_of_incidence_deg": float(aoi_deg),
        "is_beam_splitter": True,
        "is_polarizing": False,
        "beamsplitter_config": cfg,
        "surfaces": [
            {"radius": 0.0, "thickness": float(thickness_mm), "glass": substrate_glass, "tilt_x": float(aoi_deg), "is_beam_splitter": True, "beamsplitter_config": cfg},
            {"radius": 0.0, "thickness": 25.0, "glass": "AIR", "tilt_x": -float(aoi_deg)},
        ],
    }


def _pellicle_beamsplitter(
    part_number: str,
    diameter_mm: float = 25.4,
    coating_range_nm: Tuple[float, float] = (400.0, 700.0),
    split_ratio_R: float = 0.50,
    split_ratio_T: float = 0.50,
    membrane_thickness_um: float = 5.0,
) -> Dict[str, Any]:
    cfg = {
        "splitter_type": "Pellicle",
        "coating_type": "NPBS_50_50",
        "split_ratio_R": float(split_ratio_R),
        "split_ratio_T": float(split_ratio_T),
        "substrate_glass": "AIR",
        "substrate_thickness_mm": membrane_thickness_um * 1e-3,
        "cube_side_length_mm": float(diameter_mm),
        "clear_aperture_mm": float(diameter_mm),
        "orientation_angle_deg": 45.0,
        "is_polarizing": False,
        "Rs": float(split_ratio_R),
        "Rp": float(split_ratio_R),
        "Ts": float(split_ratio_T),
        "Tp": float(split_ratio_T),
        "extinction_ratio": 1.0,
        "part_number": part_number,
    }
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "amplitude_beamsplitters",
        "family": "Pellicle Beamsplitter (Zero Ghosting)",
        "type": f"Pellicle Beamsplitter ({part_number})",
        "coating_code": "PELLICLE",
        "coating_range_nm": list(coating_range_nm),
        "design_wavelengths_nm": [0.5 * (coating_range_nm[0] + coating_range_nm[1])],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": membrane_thickness_um * 1e-3,
        "split_ratio_R": float(split_ratio_R),
        "split_ratio_T": float(split_ratio_T),
        "is_beam_splitter": True,
        "is_polarizing": False,
        "beamsplitter_config": cfg,
        "surfaces": [
            {"radius": 0.0, "thickness": membrane_thickness_um * 1e-3, "glass": "AIR", "tilt_x": 45.0, "is_beam_splitter": True, "beamsplitter_config": cfg},
            {"radius": 0.0, "thickness": 25.0, "glass": "AIR", "tilt_x": -45.0},
        ],
    }


def _polarization_cube(
    part_number: str,
    dimension_mm: float = 25.4,
    coating_range_nm: Tuple[float, float] = (680.0, 1080.0),
    glass: str = "N-SF11",
    extinction_ratio: float = 1000.0,
    transmission_tp: float = 0.92,
    reflection_rs: float = 0.995,
    reflection_rp: float = 0.01,
    transmission_ts: float = 0.005,
    coating_code: str = "B",
    section: str = "polarizing_beamsplitters",
) -> Dict[str, Any]:
    cfg = {
        "splitter_type": "Cube",
        "coating_type": "PBS",
        "split_ratio_R": 0.5 * (reflection_rs + reflection_rp),
        "split_ratio_T": 0.5 * (transmission_ts + transmission_tp),
        "substrate_glass": glass,
        "substrate_thickness_mm": float(dimension_mm) / 2.0,
        "cube_side_length_mm": float(dimension_mm),
        "clear_aperture_mm": float(dimension_mm),
        "orientation_angle_deg": 45.0,
        "is_polarizing": True,
        "Rs": float(reflection_rs),
        "Rp": float(reflection_rp),
        "Ts": float(transmission_ts),
        "Tp": float(transmission_tp),
        "extinction_ratio": float(extinction_ratio),
        "part_number": part_number,
    }
    mid = 0.5 * (coating_range_nm[0] + coating_range_nm[1])
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": section,
        "family": "Polarizing Beamsplitter Cube (PBS)",
        "type": f"Polarizing Beamsplitter Cube ({part_number})",
        "coating_code": coating_code,
        "coating_range_nm": list(coating_range_nm),
        "design_wavelengths_nm": [mid],
        "diameter_mm": float(dimension_mm),
        "semi_diameter": float(dimension_mm) / 2.0,
        "dimensions_mm": f"{dimension_mm} x {dimension_mm} x {dimension_mm}",
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(dimension_mm),
        "extinction_ratio": float(extinction_ratio),
        "transmission_tp": float(transmission_tp),
        "reflection_rs": float(reflection_rs),
        "transmission_ts": float(transmission_ts),
        "reflection_rp": float(reflection_rp),
        "is_beam_splitter": True,
        "is_polarizing": True,
        "beamsplitter_config": cfg,
        "surfaces": [
            {"radius": 0.0, "thickness": float(dimension_mm) / 2.0, "glass": glass},
            {"radius": 0.0, "thickness": float(dimension_mm) / 2.0, "glass": glass, "tilt_x": 45.0, "is_beam_splitter": True, "beamsplitter_config": cfg},
        ],
    }


def _polarization_plate(
    part_number: str,
    diameter_mm: float = 25.4,
    thickness_mm: float = 2.0,
    coating_range_nm: Tuple[float, float] = (400.0, 700.0),
    glass: str = "FUSED_SILICA",
    extinction_ratio: float = 800.0,
    transmission_tp: float = 0.88,
    reflection_rs: float = 0.98,
    reflection_rp: float = 0.02,
    transmission_ts: float = 0.01,
    coating_code: str = "VIS",
    aoi_deg: float = 45.0,
) -> Dict[str, Any]:
    cfg = {
        "splitter_type": "Plate",
        "coating_type": "PBS",
        "split_ratio_R": 0.5 * (reflection_rs + reflection_rp),
        "split_ratio_T": 0.5 * (transmission_ts + transmission_tp),
        "substrate_glass": glass,
        "substrate_thickness_mm": float(thickness_mm),
        "cube_side_length_mm": float(diameter_mm),
        "clear_aperture_mm": float(diameter_mm),
        "orientation_angle_deg": float(aoi_deg),
        "is_polarizing": True,
        "Rs": float(reflection_rs),
        "Rp": float(reflection_rp),
        "Ts": float(transmission_ts),
        "Tp": float(transmission_tp),
        "extinction_ratio": float(extinction_ratio),
        "part_number": part_number,
    }
    mid = 0.5 * (coating_range_nm[0] + coating_range_nm[1])
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "polarizing_beamsplitters",
        "family": "Wire Grid Polarizing Plate Beamsplitter",
        "type": f"Wire Grid Polarizer ({part_number})",
        "coating_code": coating_code,
        "coating_range_nm": list(coating_range_nm),
        "design_wavelengths_nm": [mid],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(thickness_mm),
        "extinction_ratio": float(extinction_ratio),
        "transmission_tp": float(transmission_tp),
        "reflection_rs": float(reflection_rs),
        "transmission_ts": float(transmission_ts),
        "reflection_rp": float(reflection_rp),
        "is_beam_splitter": True,
        "is_polarizing": True,
        "beamsplitter_config": cfg,
        "surfaces": [
            {"radius": 0.0, "thickness": float(thickness_mm), "glass": glass, "tilt_x": float(aoi_deg), "is_beam_splitter": True, "beamsplitter_config": cfg},
            {"radius": 0.0, "thickness": 25.0, "glass": "AIR", "tilt_x": -float(aoi_deg)},
        ],
    }
'''

# Find def _polarization_cube and replace it up to def _waveplate
pat = r"def _polarization_cube\(.*?\n(?=def _waveplate)"
content, count = re.subn(pat, new_helpers + "\n", content, flags=re.DOTALL)
assert count == 1, f"Failed to replace _polarization_cube, count={count}"

# 2. Add the parts dictionaries
parts_block = '''
_AMPLITUDE_BEAMSPLITTERS: Dict[str, Dict[str, Any]] = {
    # 50:50 Cubes: Miniature
    "BS004": _amplitude_cube("BS004", 5.0, (400.0, 700.0), 0.50, 0.50, "N-BK7", "VIS"),
    "BS007": _amplitude_cube("BS007", 10.0, (400.0, 700.0), 0.50, 0.50, "N-BK7", "VIS"),
    "BS008": _amplitude_cube("BS008", 10.0, (700.0, 1100.0), 0.50, 0.50, "N-BK7", "NIR"),
    # 50:50 Cubes: 1/2" (12.7 mm)
    "BS013": _amplitude_cube("BS013", 12.7, (400.0, 700.0), 0.50, 0.50, "N-BK7", "VIS"),
    "BS014": _amplitude_cube("BS014", 12.7, (700.0, 1100.0), 0.50, 0.50, "N-BK7", "NIR"),
    "BS015": _amplitude_cube("BS015", 12.7, (1100.0, 1600.0), 0.50, 0.50, "N-BK7", "IR"),
    # 50:50 Cubes: 1" (25.4 mm) - Core Lab Standard
    "BS016": _amplitude_cube("BS016", 25.4, (400.0, 700.0), 0.50, 0.50, "N-BK7", "VIS"),
    "BS017": _amplitude_cube("BS017", 25.4, (700.0, 1100.0), 0.50, 0.50, "N-BK7", "NIR"),
    "BS018": _amplitude_cube("BS018", 25.4, (1100.0, 1600.0), 0.50, 0.50, "N-BK7", "IR"),
    # 50:50 Cubes: 2" (50.8 mm)
    "BS025": _amplitude_cube("BS025", 50.8, (400.0, 700.0), 0.50, 0.50, "N-BK7", "VIS"),
    "BS026": _amplitude_cube("BS026", 50.8, (700.0, 1100.0), 0.50, 0.50, "N-BK7", "NIR"),
    "BS027": _amplitude_cube("BS027", 50.8, (1100.0, 1600.0), 0.50, 0.50, "N-BK7", "IR"),
    # Unbalanced Cubes: 1" (25.4 mm)
    "BS010": _amplitude_cube("BS010", 25.4, (400.0, 700.0), 0.10, 0.90, "N-BK7", "VIS"),
    "BS011": _amplitude_cube("BS011", 25.4, (400.0, 700.0), 0.30, 0.70, "N-BK7", "VIS"),
    "BS012": _amplitude_cube("BS012", 25.4, (400.0, 700.0), 0.70, 0.30, "N-BK7", "VIS"),
    "BS019": _amplitude_cube("BS019", 25.4, (400.0, 700.0), 0.90, 0.10, "N-BK7", "VIS"),
    # 50:50 Plates: 1/2" (12.7 mm)
    "BSW04": _amplitude_plate("BSW04", 12.7, 2.0, (400.0, 700.0), 0.50, 0.50, "FUSED_SILICA", "VIS"),
    "BSW05": _amplitude_plate("BSW05", 12.7, 2.0, (700.0, 1100.0), 0.50, 0.50, "FUSED_SILICA", "NIR"),
    "BSW06": _amplitude_plate("BSW06", 12.7, 2.0, (1100.0, 1600.0), 0.50, 0.50, "FUSED_SILICA", "IR"),
    # 50:50 Plates: 1" (25.4 mm)
    "BSW10": _amplitude_plate("BSW10", 25.4, 3.0, (400.0, 700.0), 0.50, 0.50, "FUSED_SILICA", "VIS"),
    "BSW11": _amplitude_plate("BSW11", 25.4, 3.0, (700.0, 1100.0), 0.50, 0.50, "FUSED_SILICA", "NIR"),
    "BSW12": _amplitude_plate("BSW12", 25.4, 3.0, (1100.0, 1600.0), 0.50, 0.50, "FUSED_SILICA", "IR"),
    # 50:50 Plates: 2" (50.8 mm)
    "BSW26": _amplitude_plate("BSW26", 50.8, 5.0, (400.0, 700.0), 0.50, 0.50, "FUSED_SILICA", "VIS"),
    "BSW27": _amplitude_plate("BSW27", 50.8, 5.0, (700.0, 1100.0), 0.50, 0.50, "FUSED_SILICA", "NIR"),
    "BSW28": _amplitude_plate("BSW28", 50.8, 5.0, (1100.0, 1600.0), 0.50, 0.50, "FUSED_SILICA", "IR"),
    # Unbalanced Plates: 1" (25.4 mm)
    "BSW29": _amplitude_plate("BSW29", 25.4, 3.0, (400.0, 700.0), 0.70, 0.30, "FUSED_SILICA", "VIS"),
    "BSW30": _amplitude_plate("BSW30", 25.4, 3.0, (400.0, 700.0), 0.30, 0.70, "FUSED_SILICA", "VIS"),
    "BSW31": _amplitude_plate("BSW31", 25.4, 3.0, (400.0, 700.0), 0.90, 0.10, "FUSED_SILICA", "VIS"),
    "BSW32": _amplitude_plate("BSW32", 25.4, 3.0, (400.0, 700.0), 0.10, 0.90, "FUSED_SILICA", "VIS"),
    # Pellicle Beamsplitters
    "BP108": _pellicle_beamsplitter("BP108", 25.4, (400.0, 700.0), 0.50, 0.50, 5.0),
    "BP145B2": _pellicle_beamsplitter("BP145B2", 25.4, (700.0, 1100.0), 0.50, 0.50, 5.0),
}

_POLARIZING_BEAMSPLITTERS: Dict[str, Dict[str, Any]] = {
    # 10 mm Cubes
    "PBS101": _polarization_cube("PBS101", 10.0, (420.0, 680.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "VIS"),
    "PBS102": _polarization_cube("PBS102", 10.0, (620.0, 1000.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "NIR"),
    "PBS103": _polarization_cube("PBS103", 10.0, (900.0, 1300.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR1"),
    "PBS104": _polarization_cube("PBS104", 10.0, (1200.0, 1600.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR2"),
    # 1/2" (12.7 mm) Cubes
    "PBS121": _polarization_cube("PBS121", 12.7, (420.0, 680.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "VIS"),
    "PBS122": _polarization_cube("PBS122", 12.7, (620.0, 1000.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "NIR"),
    "PBS123": _polarization_cube("PBS123", 12.7, (900.0, 1300.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR1"),
    "PBS124": _polarization_cube("PBS124", 12.7, (1200.0, 1600.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR2"),
    # 1" (25.4 mm) Cubes - Research Workhorse
    "PBS251": _polarization_cube("PBS251", 25.4, (420.0, 680.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "VIS"),
    "PBS252": _polarization_cube("PBS252", 25.4, (680.0, 1080.0), "N-SF11", 1000.0, 0.92, 0.995, 0.01, 0.005, "B"),
    "PBS253": _polarization_cube("PBS253", 25.4, (900.0, 1300.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR1"),
    "PBS254": _polarization_cube("PBS254", 25.4, (1200.0, 1600.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR2"),
    "PBS255": _polarization_cube("PBS255", 25.4, (1500.0, 1700.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR3"),
    # 2" (50.8 mm) Cubes
    "PBS511": _polarization_cube("PBS511", 50.8, (420.0, 680.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "VIS"),
    "PBS512": _polarization_cube("PBS512", 50.8, (620.0, 1000.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "NIR"),
    "PBS513": _polarization_cube("PBS513", 50.8, (900.0, 1300.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR1"),
    "PBS514": _polarization_cube("PBS514", 50.8, (1200.0, 1600.0), "N-SF11", 1000.0, 0.90, 0.995, 0.01, 0.005, "IR2"),
    # Wire Grid Polarizing Plates
    "WPBS25-VIS": _polarization_plate("WPBS25-VIS", 25.4, 2.0, (400.0, 700.0), "FUSED_SILICA", 800.0, 0.85, 0.95, 0.02, 0.01, "VIS"),
    "WPBS25-NIR": _polarization_plate("WPBS25-NIR", 25.4, 2.0, (650.0, 2000.0), "FUSED_SILICA", 1000.0, 0.88, 0.95, 0.02, 0.01, "NIR"),
    "WPBS50-VIS": _polarization_plate("WPBS50-VIS", 50.8, 3.0, (400.0, 700.0), "FUSED_SILICA", 800.0, 0.85, 0.95, 0.02, 0.01, "VIS"),
    "WPBS50-NIR": _polarization_plate("WPBS50-NIR", 50.8, 3.0, (650.0, 2000.0), "FUSED_SILICA", 1000.0, 0.88, 0.95, 0.02, 0.01, "NIR"),
}
'''

# Insert parts_block right before _POLARIZATION_OPTICS
content = content.replace("_POLARIZATION_OPTICS: Dict[str, Dict[str, Any]] = {", parts_block + "\n_POLARIZATION_OPTICS: Dict[str, Dict[str, Any]] = {")

# 3. Add sections to THORLABS_CATALOG_SECTIONS
sections_insertion = '''    "amplitude_beamsplitters": {
        "label": "Amplitude Beam Splitters (50:50, 70:30, 90:10 Cubes & Plates)",
        "description": "Thorlabs broadband non-polarizing cube and plate beam splitters (BS013-BS027, BSW04-BSW32, BP108)",
        "parts": _AMPLITUDE_BEAMSPLITTERS,
    },
    "polarizing_beamsplitters": {
        "label": "Polarizing Beam Splitters (PBS Cubes & Wire Grid Plates)",
        "description": "Thorlabs polarizing cube and plate beam splitters (PBS101-PBS514, WPBS25/50 Series)",
        "parts": _POLARIZING_BEAMSPLITTERS,
    },
'''

content = content.replace('"polarization_optics": {', sections_insertion + '    "polarization_optics": {')

# 4. Update query_catalog filters
query_filter_old = '''    elif sec_filter in ("polarization", "polarizer", "pbs", "hwp", "waveplate", "eom", "polarization_optics"):
        sec_filter = "polarization_optics"'''

query_filter_new = '''    elif sec_filter in ("amplitude_beamsplitter", "amplitude_beamsplitters", "beamsplitter", "beamsplitters", "bs", "npbs"):
        sec_filter = "amplitude_beamsplitters"
    elif sec_filter in ("polarizing_beamsplitter", "polarizing_beamsplitters", "pbs", "polarizing_cube", "wpbs"):
        sec_filter = "polarizing_beamsplitters"
    elif sec_filter in ("polarization", "polarizer", "hwp", "waveplate", "eom", "polarization_optics"):
        sec_filter = "polarization_optics"'''

content = content.replace(query_filter_old, query_filter_new)

with open("optics_app/catalogs/thorlabs.py", "w", encoding="utf-8") as f:
    f.write(content)

with open("v2/catalogs/thorlabs.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated optics_app/catalogs/thorlabs.py and v2/catalogs/thorlabs.py!")
