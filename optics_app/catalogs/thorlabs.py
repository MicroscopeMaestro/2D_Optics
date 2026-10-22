"""Curated Thorlabs component prescriptions used by the optical model.

Contains commercial Thorlabs optical components including:
- IR Achromatic Doublets (C-Coating: 1050-1700 nm) across miniature, 1/2", 1", 30mm, and 2" diameters
- NIR Achromatic Doublets (B-Coating: 650-1050 nm) across miniature, 1/2", 1", 30mm, 2", and 3" diameters
- Visible Achromatic Doublets (A-Coating: 400-700 nm) across miniature, 1/2", 1", 30mm, 2", and 3" diameters
- Precision Aspheric Condensers (ACL Series) across 12mm, 15mm, 20mm, 25.4mm, 30mm, and 50.8mm diameters
- Plano-Convex Singlets (LA Series) across 1/2", 1", and 2" diameters
- Bi-Convex Singlets (LB Series) across standard diameters
- Plano-Concave Singlets (LC Series) across standard diameters
- Concave Spherical Mirrors (CM Series) across 1" and 2" diameters

Prescription values are transcribed from Thorlabs optical catalog / Zemax support data.
Radii follow the standard optical design sign convention for light travelling from surface 1 to rear.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


def _achromat(
    part_number: str,
    diameter_mm: float,
    nominal_focal_length_mm: float,
    catalog_efl_mm: float,
    back_focal_length_mm: float,
    radii_mm: Tuple[float, float, float],
    center_thicknesses_mm: Tuple[float, float],
    glasses: Tuple[str, str],
    coating_code: str = "C",
    coating_range_nm: Optional[List[float]] = None,
    design_wavelengths_nm: Optional[List[float]] = None,
    section: str = "achromatic_doublets",
    family: str = "IR Achromatic Doublets",
) -> Dict[str, Any]:
    if coating_range_nm is None:
        coating_range_nm = [1050.0, 1700.0] if coating_code == "C" else ([650.0, 1050.0] if coating_code == "B" else [400.0, 700.0])
    if design_wavelengths_nm is None:
        design_wavelengths_nm = [1016.0, 1330.0, 1550.0] if coating_code == "C" else ([650.0, 785.0, 1050.0] if coating_code == "B" else [400.0, 587.6, 700.0])

    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": section,
        "family": family,
        "type": "Cemented Achromatic Doublet",
        "coating_code": coating_code,
        "coating_range_nm": coating_range_nm,
        "design_wavelengths_nm": design_wavelengths_nm,
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": float(nominal_focal_length_mm),
        "focal_length_mm": float(catalog_efl_mm),
        "optical_power_diopters": 1000.0 / float(catalog_efl_mm) if float(catalog_efl_mm) != 0 else 0.0,
        "back_focal_length_mm": float(back_focal_length_mm),
        "center_thickness_mm": float(center_thicknesses_mm[0] + center_thicknesses_mm[1]),
        # The final thickness is an editable air gap, not part of the physical glass element.
        "surfaces": [
            {"radius": radii_mm[0], "thickness": center_thicknesses_mm[0], "glass": glasses[0]},
            {"radius": radii_mm[1], "thickness": center_thicknesses_mm[1], "glass": glasses[1]},
            {"radius": radii_mm[2], "thickness": float(back_focal_length_mm), "glass": "AIR"},
        ],
    }


def _singlet(
    part_number: str,
    diameter_mm: float,
    catalog_efl_mm: float,
    back_focal_length_mm: float,
    radius_mm: float,
    center_thickness_mm: float,
    glass: str = "N-BK7",
    coating_code: str = "Uncoated",
    section: str = "singlets",
    family: str = "Plano-Convex Lens",
    lens_type: str = "N-BK7 Plano-Convex Singlet",
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": section,
        "family": family,
        "type": lens_type,
        "coating_code": coating_code,
        "coating_range_nm": [350.0, 2000.0],
        "design_wavelengths_nm": [587.6],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": float(catalog_efl_mm),
        "focal_length_mm": float(catalog_efl_mm),
        "optical_power_diopters": 1000.0 / float(catalog_efl_mm) if float(catalog_efl_mm) != 0 else 0.0,
        "back_focal_length_mm": float(back_focal_length_mm),
        "center_thickness_mm": float(center_thickness_mm),
        "surfaces": [
            {"radius": float(radius_mm), "thickness": float(center_thickness_mm), "glass": glass},
            {"radius": 0.0, "thickness": float(back_focal_length_mm), "glass": "AIR"},
        ],
    }


def _biconvex(
    part_number: str,
    diameter_mm: float,
    catalog_efl_mm: float,
    back_focal_length_mm: float,
    radius_mm: float,
    center_thickness_mm: float,
    glass: str = "N-BK7",
    coating_code: str = "Uncoated",
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "biconvex_singlets",
        "family": "Bi-Convex Lens",
        "type": "N-BK7 Bi-Convex Singlet",
        "coating_code": coating_code,
        "coating_range_nm": [350.0, 2000.0],
        "design_wavelengths_nm": [587.6],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": float(catalog_efl_mm),
        "focal_length_mm": float(catalog_efl_mm),
        "optical_power_diopters": 1000.0 / float(catalog_efl_mm) if float(catalog_efl_mm) != 0 else 0.0,
        "back_focal_length_mm": float(back_focal_length_mm),
        "center_thickness_mm": float(center_thickness_mm),
        "surfaces": [
            {"radius": float(radius_mm), "thickness": float(center_thickness_mm), "glass": glass},
            {"radius": -float(radius_mm), "thickness": float(back_focal_length_mm), "glass": "AIR"},
        ],
    }


def _planoconcave(
    part_number: str,
    diameter_mm: float,
    catalog_efl_mm: float,
    back_focal_length_mm: float,
    radius_mm: float,
    center_thickness_mm: float,
    glass: str = "N-BK7",
    coating_code: str = "Uncoated",
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "planoconcave_singlets",
        "family": "Plano-Concave Lens",
        "type": "N-BK7 Plano-Concave Singlet",
        "coating_code": coating_code,
        "coating_range_nm": [350.0, 2000.0],
        "design_wavelengths_nm": [587.6],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": float(catalog_efl_mm),
        "focal_length_mm": float(catalog_efl_mm),
        "optical_power_diopters": 1000.0 / float(catalog_efl_mm) if float(catalog_efl_mm) != 0 else 0.0,
        "back_focal_length_mm": float(back_focal_length_mm),
        "center_thickness_mm": float(center_thickness_mm),
        "surfaces": [
            {"radius": float(radius_mm), "thickness": float(center_thickness_mm), "glass": glass},
            {"radius": 0.0, "thickness": float(back_focal_length_mm), "glass": "AIR"},
        ],
    }


def _mirror(
    part_number: str,
    diameter_mm: float,
    focal_length_mm: float,
    radius_mm: float,
    substrate_thickness_mm: float = 6.0,
    coating: str = "Protected Silver (P01)",
    coating_code: str = "P01",
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "spherical_mirrors",
        "family": "Concave Spherical Mirror",
        "type": f"Spherical Mirror ({coating})",
        "coating_code": coating_code,
        "coating_range_nm": [450.0, 20000.0],
        "design_wavelengths_nm": [587.6],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": float(focal_length_mm),
        "focal_length_mm": float(focal_length_mm),
        "radius_mm": float(radius_mm),
        "optical_power_diopters": 1000.0 / float(focal_length_mm) if float(focal_length_mm) != 0 else 0.0,
        "back_focal_length_mm": float(focal_length_mm),
        "center_thickness_mm": float(substrate_thickness_mm),
        "surfaces": [
            {"radius": float(radius_mm), "thickness": float(substrate_thickness_mm), "glass": "MIRROR"},
        ],
    }


def _asphere(
    part_number: str,
    diameter_mm: float,
    catalog_efl_mm: float,
    back_focal_length_mm: float,
    center_thickness_mm: float,
    radius_mm: float,
    conic: float,
    asphere_coeffs: List[float],
    glass: str = "B270",
    coating_code: str = "Uncoated",
    coating_range_nm: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if coating_range_nm is None:
        coating_range_nm = [350.0, 2000.0] if coating_code == "Uncoated" else ([650.0, 1050.0] if coating_code == "B" else ([1050.0, 1700.0] if coating_code == "C" else [400.0, 700.0]))
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "aspheric_condensers",
        "family": "Aspheric Condenser",
        "type": f"{glass} Precision Aspheric Condenser Lens",
        "coating_code": coating_code,
        "coating_range_nm": coating_range_nm,
        "design_wavelengths_nm": [587.6],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": float(catalog_efl_mm),
        "focal_length_mm": float(catalog_efl_mm),
        "optical_power_diopters": 1000.0 / float(catalog_efl_mm),
        "back_focal_length_mm": float(back_focal_length_mm),
        "center_thickness_mm": float(center_thickness_mm),
        "surfaces": [
            {"radius": float(radius_mm), "thickness": float(center_thickness_mm), "glass": glass, "conic": float(conic), "asphere_coeffs": asphere_coeffs},
            {"radius": 0.0, "thickness": float(back_focal_length_mm), "glass": "AIR"},
        ],
    }


def _parabolic_mirror(
    part_number: str,
    diameter_mm: float,
    focal_length_mm: float,
    substrate_thickness_mm: float = 6.0,
    coating: str = "Protected Silver (P01)",
    coating_code: str = "P01",
    coating_range_nm: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if coating_range_nm is None:
        if coating_code == "P01":
            coating_range_nm = [450.0, 20000.0]
        elif coating_code == "M01":
            coating_range_nm = [800.0, 20000.0]
        elif coating_code == "G01":
            coating_range_nm = [250.0, 450.0]
        else:
            coating_range_nm = [400.0, 20000.0]

    r = -2.0 * float(focal_length_mm)
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "parabolic_mirrors",
        "family": "Concave Parabolic Mirror",
        "type": f"Parabolic Mirror ({coating})",
        "coating_code": coating_code,
        "coating_range_nm": coating_range_nm,
        "design_wavelengths_nm": [587.6],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": float(focal_length_mm),
        "focal_length_mm": float(focal_length_mm),
        "radius_mm": r,
        "conic_k": -1.0,
        "conic": -1.0,
        "optical_power_diopters": 1000.0 / float(focal_length_mm) if float(focal_length_mm) != 0 else 0.0,
        "back_focal_length_mm": float(focal_length_mm),
        "center_thickness_mm": float(substrate_thickness_mm),
        "surfaces": [
            {
                "radius": r,
                "thickness": float(substrate_thickness_mm),
                "glass": "MIRROR",
                "conic": -1.0,
            },
        ],
    }


def _grating(
    part_number: str,
    grooves_per_mm: float,
    blaze_wavelength_nm: float,
    blaze_angle_deg: float,
    dimension_mm: float = 25.0,
    grating_type: str = "Reflective Ruled Diffraction Grating",
    family: str = "Ruled Reflective Grating",
    is_transmission: bool = False,
    coating: str = "Aluminum",
    coating_code: str = "AL",
) -> Dict[str, Any]:
    blaze_um = float(blaze_wavelength_nm) / 1000.0
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "diffraction_gratings",
        "family": family,
        "type": grating_type,
        "coating_code": coating_code,
        "coating_range_nm": [max(190.0, blaze_wavelength_nm * 0.6), blaze_wavelength_nm * 1.8],
        "design_wavelengths_nm": [blaze_wavelength_nm],
        "diameter_mm": float(dimension_mm),
        "semi_diameter": float(dimension_mm) / 2.0,
        "dimensions_mm": f"{dimension_mm} x {dimension_mm} x 6.0",
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": 6.0 if not is_transmission else 3.0,
        "grooves_per_mm": float(grooves_per_mm),
        "blaze_wavelength_um": blaze_um,
        "blaze_angle_deg": float(blaze_angle_deg),
        "is_transmission": is_transmission,
        "surfaces": [
            {
                "radius": 0.0,
                "thickness": 6.0 if not is_transmission else 3.0,
                "glass": "AIR" if is_transmission else "MIRROR",
                "surface_type": "Grating",
                "grating_lines_per_mm": float(grooves_per_mm),
                "grating_order": 1,
                "grating_primary_wavelength_um": blaze_um,
                "grating_blaze_wavelength_um": blaze_um,
                "grating_blaze_angle_deg": float(blaze_angle_deg),
                "grating_type": "Blazed Reflective" if not is_transmission else "Transmission",
                "is_transmission_grating": is_transmission,
            }
        ],
    }


def _amplitude_cube(
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

def _waveplate(
    part_number: str,
    diameter_mm: float = 25.4,
    design_wavelength_nm: float = 780.0,
    retardation: str = "Half-Wave (pi)",
    coating_range_nm: Tuple[float, float] = (650.0, 1050.0),
    glass: str = "N-BK7",
    thickness_mm: float = 5.0,
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "polarization_optics",
        "family": "Zero-Order Half-Wave Plate",
        "type": f"Zero-Order Waveplate ({part_number})",
        "coating_code": "B" if design_wavelength_nm < 900 else "C",
        "coating_range_nm": list(coating_range_nm),
        "design_wavelengths_nm": [float(design_wavelength_nm)],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(thickness_mm),
        "retardation": retardation,
        "retardation_rad": math.pi,
        "surfaces": [
            {"radius": 0.0, "thickness": float(thickness_mm), "glass": glass},
        ],
    }


def _eom_device(
    part_number: str = "EO-AM-NR-C1",
    crystal: str = "RTP",
    freq_mhz: float = 10.0,
    v_pi_1040: float = 240.0,
    aperture_mm: float = 3.0,
    length_mm: float = 15.0,
    insertion_loss: float = 0.02,
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "polarization_optics",
        "family": "Resonant Electro-Optic Amplitude Modulator",
        "type": f"Resonant EOM ({part_number})",
        "coating_code": "C",
        "coating_range_nm": [1000.0, 1600.0],
        "design_wavelengths_nm": [1040.0],
        "diameter_mm": float(aperture_mm),
        "semi_diameter": float(aperture_mm) / 2.0,
        "clear_aperture_mm": float(aperture_mm),
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(length_mm),
        "modulation_frequency_mhz": float(freq_mhz),
        "half_wave_voltage_v": float(v_pi_1040),
        "crystal": crystal,
        "insertion_loss_fraction": float(insertion_loss),
        "surfaces": [
            {"radius": 0.0, "thickness": float(length_mm), "glass": crystal},
        ],
    }


def _optical_filter(
    part_number: str,
    filter_type: str,
    cwl_or_cutoff_nm: float,
    fwhm_nm: Optional[float] = None,
    od_blocking: float = 6.0,
    diameter_mm: float = 25.4,
    thickness_mm: float = 5.0,
    glass: str = "FUSED_SILICA",
    transmission: float = 0.92,
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "optical_filters",
        "family": f"Premium Optical Filter ({filter_type})",
        "type": f"{filter_type} Filter ({part_number})",
        "coating_code": "FILT",
        "coating_range_nm": [200.0, 1650.0],
        "design_wavelengths_nm": [float(cwl_or_cutoff_nm)],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(thickness_mm),
        "filter_type": filter_type,
        "cwl_or_cutoff_nm": float(cwl_or_cutoff_nm),
        "fwhm_nm": float(fwhm_nm) if fwhm_nm is not None else 0.0,
        "optical_density": float(od_blocking),
        "transmission": float(transmission),
        "surfaces": [
            {"radius": 0.0, "thickness": float(thickness_mm), "glass": glass},
        ],
    }


def _dichroic_mirror(
    part_number: str = "DMLP900",
    cutoff_wavelength_nm: float = 900.0,
    aoi_deg: float = 45.0,
    diameter_mm: float = 25.4,
    thickness_mm: float = 3.0,
    glass: str = "FUSED_SILICA",
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "dichroic_mirrors",
        "family": "Longpass Dichroic Mirror",
        "type": f"Dichroic Combiner ({part_number})",
        "coating_code": "DICHROIC",
        "coating_range_nm": [680.0, 1300.0],
        "design_wavelengths_nm": [797.0, 1040.0],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(thickness_mm),
        "cutoff_wavelength_nm": float(cutoff_wavelength_nm),
        "angle_of_incidence_deg": float(aoi_deg),
        "reflection_band_nm": [680.0, 880.0],
        "transmission_band_nm": [920.0, 1300.0],
        "reflection_pump_797": 0.98,
        "transmission_stokes_1040": 0.92,
        "surfaces": [
            {"radius": 0.0, "thickness": float(thickness_mm), "glass": "MIRROR", "tilt_x": -float(aoi_deg)},
        ],
    }


def _flat_mirror(
    part_number: str = "PF10-03-P01",
    coating: str = "Protected Silver",
    coating_code: str = "P01",
    diameter_mm: float = 25.4,
    thickness_mm: float = 6.0,
    glass: str = "FUSED_SILICA",
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "flat_mirrors",
        "family": "Precision Flat Mirror",
        "type": f"Flat Mirror ({coating})",
        "coating_code": coating_code,
        "coating_range_nm": [450.0, 20000.0],
        "design_wavelengths_nm": [797.0, 1040.0],
        "diameter_mm": float(diameter_mm),
        "semi_diameter": float(diameter_mm) / 2.0,
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": float(thickness_mm),
        "reflectance": 0.99,
        "surfaces": [
            {"radius": 0.0, "thickness": float(thickness_mm), "glass": "MIRROR"},
        ],
    }


def _beam_trap(
    part_number: str = "BT610",
    aperture_mm: float = 10.0,
    max_power_w: float = 5.0,
    max_energy_density_j_cm2: float = 10.0,
    backscatter_fraction: float = 1.0e-5,
) -> Dict[str, Any]:
    return {
        "part_number": part_number,
        "manufacturer": "Thorlabs",
        "section": "beam_traps",
        "family": "Optical Beam Trap / Dump",
        "type": f"Beam Trap ({part_number})",
        "coating_code": "ABSORB",
        "coating_range_nm": [200.0, 20000.0],
        "design_wavelengths_nm": [797.0, 1040.0],
        "diameter_mm": float(aperture_mm),
        "semi_diameter": float(aperture_mm) / 2.0,
        "nominal_focal_length_mm": 0.0,
        "focal_length_mm": 0.0,
        "optical_power_diopters": 0.0,
        "back_focal_length_mm": 0.0,
        "center_thickness_mm": 0.0,
        "max_power_w": float(max_power_w),
        "max_energy_density_j_cm2": float(max_energy_density_j_cm2),
        "backscatter_fraction": float(backscatter_fraction),
        "surfaces": [
            {"radius": 0.0, "thickness": 0.0, "glass": "AIR", "surface_type": "DETECTOR"},
        ],
    }


# =============================================================================
# 1. IR ACHROMATIC DOUBLETS (C-Coating: 1050 - 1700 nm)
# =============================================================================
_IR_ACHROMATIC_DOUBLETS: Dict[str, Dict[str, Any]] = {
    # Miniature diameters (10 mm and under)
    "AC050-008-C": _achromat("AC050-008-C", 5.0, 7.5, 7.5, 5.2, (4.6, -3.9, -23.9), (2.5, 1.5), ("N-LAK22", "N-SF6")),
    "AC050-010-C": _achromat("AC050-010-C", 5.0, 10.0, 10.0, 6.9, (4.6, -4.6, 36.0), (2.5, 1.5), ("N-LAK22", "N-SF6")),
    "AC050-015-C": _achromat("AC050-015-C", 5.0, 15.0, 15.0, 11.6, (5.3, -5.5, 15.2), (2.0, 1.3), ("N-BAF10", "N-SF6")),
    "AC060-010-C": _achromat("AC060-010-C", 6.0, 10.0, 10.0, 8.5, (10.4, -3.6, -9.2), (3.5, 1.3), ("N-LAK22", "N-SF6")),
    "AC064-013-C": _achromat("AC064-013-C", 6.35, 12.7, 12.7, 11.4, (13.2, -4.9, -12.4), (2.8, 1.3), ("N-LAK22", "N-SF6")),
    "AC064-015-C": _achromat("AC064-015-C", 6.35, 15.0, 15.0, 14.4, (22.7, -4.9, -11.3), (2.3, 1.3), ("N-LAK22", "N-SF6")),
    "AC080-010-C": _achromat("AC080-010-C", 8.0, 10.0, 10.0, 7.2, (7.1, -4.9, -20.9), (4.2, 1.3), ("N-BAF10", "N-SF6HT")),
    "AC080-016-C": _achromat("AC080-016-C", 8.0, 16.0, 16.0, 12.3, (7.5, -7.8, 68.5), (3.5, 1.3), ("N-LAK22", "N-SF6HT")),
    # Diameter 12.7 mm (1/2 inch)
    "AC127-019-C": _achromat("AC127-019-C", 12.7, 19.0, 19.0, 15.4, (12.4, -10.0, -48.8), (5.0, 1.5), ("N-LAK22", "N-SF6HT")),
    "AC127-025-C": _achromat("AC127-025-C", 12.7, 25.0, 25.0, 20.3, (12.0, -12.9, 151.7), (4.7, 1.5), ("N-LAK22", "N-SF6HT")),
    "AC127-030-C": _achromat("AC127-030-C", 12.7, 30.0, 30.0, 24.5, (12.4, -14.0, 65.3), (4.7, 1.5), ("N-LAK22", "N-SF6HT")),
    "AC127-050-C": _achromat("AC127-050-C", 12.7, 50.0, 49.9, 43.5, (16.0, -18.4, 44.6), (4.0, 1.5), ("N-BAF10", "N-SF6HT")),
    "AC127-075-C": _achromat("AC127-075-C", 12.7, 75.0, 75.1, 69.8, (23.2, -27.9, 66.7), (3.0, 1.5), ("N-BAF10", "N-SF6HT")),
    # Diameter 25.4 mm (1 inch)
    "AC254-030-C": _achromat("AC254-030-C", 25.4, 30.0, 30.4, 22.2, (21.1, -15.2, -71.1), (13.0, 1.8), ("N-BAF10", "N-SF6HT")),
    "AC254-040-C": _achromat("AC254-040-C", 25.4, 40.0, 40.0, 32.8, (24.4, -21.1, -143.9), (10.0, 1.8), ("N-LAK22", "N-SF6HT")),
    "AC254-050-C": _achromat("AC254-050-C", 25.4, 50.0, 50.0, 41.2, (22.9, -25.9, 194.5), (9.0, 1.8), ("N-LAK22", "N-SF6HT")),
    "AC254-075-C": _achromat("AC254-075-C", 25.4, 75.0, 75.1, 65.0, (26.4, -29.4, 84.9), (7.6, 1.8), ("N-BAF10", "N-SF6HT")),
    "AC254-100-C": _achromat("AC254-100-C", 25.4, 100.0, 100.1, 90.4, (32.1, -38.0, 93.5), (6.5, 1.8), ("N-BAF10", "N-SF6HT")),
    "AC254-125-C": _achromat("AC254-125-C", 25.4, 125.0, 125.0, 115.35, (36.9, -47.5, 108.6), (5.0, 3.0), ("N-LAK22", "N-SF6HT")),
    "AC254-150-C": _achromat("AC254-150-C", 25.4, 150.0, 150.5, 140.8, (42.7, -52.0, 111.5), (5.0, 2.5), ("N-BAF10", "N-SF6HT")),
    "AC254-200-C": _achromat("AC254-200-C", 25.4, 200.0, 200.0, 193.1, (70.0, -95.9, 274.3), (4.0, 3.0), ("N-LAK22", "N-SF6HT")),
    # Diameter 30.0 mm
    "AC300-050-C": _achromat("AC300-050-C", 30.0, 50.0, 50.1, 44.7, (41.7, -22.7, -75.7), (10.0, 2.0), ("N-BAF10", "N-SF6HT")),
    "AC300-080-C": _achromat("AC300-080-C", 30.0, 80.0, 80.5, 68.5, (29.4, -33.9, 97.7), (9.5, 2.0), ("N-BAF10", "N-SF6HT")),
    "AC300-100-C": _achromat("AC300-100-C", 30.0, 100.0, 99.9, 87.8, (33.5, -39.2, 100.7), (8.5, 2.2), ("N-BAF10", "N-SF6HT")),
    # Diameter 50.8 mm (2 inch)
    "AC508-075-C": _achromat("AC508-075-C", 50.8, 75.0, 75.4, 63.0, (49.9, -39.1, -230.7), (19.0, 2.5), ("N-BAF10", "N-SF6HT")),
    "AC508-080-C": _achromat("AC508-080-C", 50.8, 80.0, 80.3, 66.9, (47.2, -43.2, -640.7), (18.0, 2.5), ("N-BAF10", "N-SF6HT")),
    "AC508-100-C": _achromat("AC508-100-C", 50.8, 100.0, 99.7, 83.0, (44.7, -48.3, 259.4), (17.0, 2.5), ("N-BAF10", "N-SF6HT")),
    "AC508-150-C": _achromat("AC508-150-C", 50.8, 150.0, 150.2, 117.7, (39.5, -49.9, 83.6), (18.0, 5.0), ("N-LAK22", "N-SF6HT")),
    "AC508-200-C": _achromat("AC508-200-C", 50.8, 200.0, 199.2, 182.7, (67.1, -87.6, 234.3), (12.0, 3.0), ("N-LAK22", "N-SF6HT")),
}


# =============================================================================
# 2. NIR ACHROMATIC DOUBLETS (B-Coating: 650 - 1050 nm)
# =============================================================================
_VISIBLE_ACHROMATS_B: Dict[str, Dict[str, Any]] = {
    # Miniature (≤ 10 mm)
    "AC050-008-B": _achromat("AC050-008-B", 5.0, 7.5, 7.5, 5.2, (4.6, -3.9, -23.9), (2.5, 1.5), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC050-010-B": _achromat("AC050-010-B", 5.0, 10.0, 10.0, 6.9, (4.6, -4.6, 36.0), (2.5, 1.5), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC050-015-B": _achromat("AC050-015-B", 5.0, 15.0, 15.0, 11.6, (5.3, -5.5, 15.2), (2.0, 1.3), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC060-010-B": _achromat("AC060-010-B", 6.0, 10.0, 10.0, 8.5, (10.4, -3.6, -9.2), (3.5, 1.3), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC064-013-B": _achromat("AC064-013-B", 6.35, 12.7, 12.7, 11.4, (13.2, -4.9, -12.4), (2.8, 1.3), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC064-015-B": _achromat("AC064-015-B", 6.35, 15.0, 15.0, 14.4, (22.7, -4.9, -11.3), (2.3, 1.3), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC080-010-B": _achromat("AC080-010-B", 8.0, 10.0, 10.0, 7.2, (7.1, -4.9, -20.9), (4.2, 1.3), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC080-016-B": _achromat("AC080-016-B", 8.0, 16.0, 16.0, 12.3, (7.5, -7.8, 68.5), (3.5, 1.3), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC080-020-B": _achromat("AC080-020-B", 8.0, 20.0, 20.0, 16.5, (11.1, -9.2, -34.8), (3.0, 1.3), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    # Diameter 12.7 mm (1/2 inch)
    "AC127-019-B": _achromat("AC127-019-B", 12.7, 19.0, 19.0, 14.8, (12.4, -10.0, -48.8), (5.0, 1.5), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC127-025-B": _achromat("AC127-025-B", 12.7, 25.0, 25.0, 20.8, (12.0, -12.9, 151.7), (4.7, 1.5), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC127-030-B": _achromat("AC127-030-B", 12.7, 30.0, 30.0, 25.8, (12.4, -14.0, 65.3), (4.7, 1.5), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC127-040-B": _achromat("AC127-040-B", 12.7, 40.0, 40.0, 35.8, (26.12, -21.28, -137.09), (3.5, 1.5), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC127-050-B": _achromat("AC127-050-B", 12.7, 50.0, 50.0, 45.8, (16.0, -18.4, 44.6), (4.0, 1.5), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC127-075-B": _achromat("AC127-075-B", 12.7, 75.0, 75.0, 70.8, (23.2, -27.9, 66.7), (3.0, 1.5), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    # Diameter 25.4 mm (1 inch)
    "AC254-030-B": _achromat("AC254-030-B", 25.4, 30.0, 30.0, 22.0, (21.09, -16.18, -79.08), (12.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-035-B": _achromat("AC254-035-B", 25.4, 35.0, 35.0, 27.2, (24.0, -19.1, -102.1), (10.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-040-B": _achromat("AC254-040-B", 25.4, 40.0, 40.0, 32.0, (26.12, -21.28, -137.09), (10.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-045-B": _achromat("AC254-045-B", 25.4, 45.0, 45.0, 37.5, (31.2, -25.9, -130.6), (9.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-050-B": _achromat("AC254-050-B", 25.4, 50.0, 50.0, 43.3, (33.55, -27.05, -125.6), (9.0, 2.0), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-060-B": _achromat("AC254-060-B", 25.4, 60.0, 60.0, 53.5, (41.7, -25.9, -230.7), (8.0, 2.5), ("N-BAF10", "N-SF10"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-075-B": _achromat("AC254-075-B", 25.4, 75.0, 75.0, 68.0, (36.90, -42.17, 417.8), (7.6, 1.8), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-100-B": _achromat("AC254-100-B", 25.4, 100.0, 100.0, 93.0, (66.68, -53.7, -259.41), (4.0, 1.5), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-125-B": _achromat("AC254-125-B", 25.4, 125.0, 125.0, 118.5, (36.9, -47.5, 108.6), (5.0, 3.0), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-150-B": _achromat("AC254-150-B", 25.4, 150.0, 150.0, 143.0, (83.6, -89.33, -1330.5), (5.0, 2.5), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-200-B": _achromat("AC254-200-B", 25.4, 200.0, 200.0, 193.5, (106.4, -96.6, 2000.0), (4.0, 3.0), ("N-LAK22", "N-SF10"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-250-B": _achromat("AC254-250-B", 25.4, 250.0, 250.0, 244.0, (137.1, -111.5, -459.2), (4.0, 2.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-300-B": _achromat("AC254-300-B", 25.4, 300.0, 300.0, 294.5, (165.2, -137.1, -557.4), (4.0, 2.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-400-B": _achromat("AC254-400-B", 25.4, 400.0, 400.0, 395.0, (219.8, -181.6, -738.5), (3.5, 2.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC254-500-B": _achromat("AC254-500-B", 25.4, 500.0, 500.0, 495.5, (337.3, -186.8, -557.4), (3.5, 2.0), ("N-BK7", "SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    # Diameter 30.0 mm
    "AC300-050-B": _achromat("AC300-050-B", 30.0, 50.0, 50.0, 41.5, (41.7, -22.7, -75.7), (10.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC300-080-B": _achromat("AC300-080-B", 30.0, 80.0, 80.0, 72.0, (29.4, -33.9, 97.7), (9.5, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC300-100-B": _achromat("AC300-100-B", 30.0, 100.0, 100.0, 92.5, (33.5, -39.2, 100.7), (8.5, 2.2), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    # Diameter 50.8 mm (2 inch)
    "AC508-075-B": _achromat("AC508-075-B", 50.8, 75.0, 75.0, 61.2, (49.9, -39.1, -230.7), (19.0, 2.5), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-080-B": _achromat("AC508-080-B", 50.8, 80.0, 80.0, 66.5, (47.2, -43.2, -640.7), (18.0, 2.5), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-100-B": _achromat("AC508-100-B", 50.8, 100.0, 100.0, 86.8, (44.7, -48.3, 259.4), (17.0, 2.5), ("N-BAF10", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-150-B": _achromat("AC508-150-B", 50.8, 150.0, 150.0, 138.2, (39.5, -49.9, 83.6), (18.0, 5.0), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-200-B": _achromat("AC508-200-B", 50.8, 200.0, 200.0, 189.5, (67.1, -87.6, 234.3), (12.0, 3.0), ("N-LAK22", "N-SF6HT"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-250-B": _achromat("AC508-250-B", 50.8, 250.0, 250.0, 240.5, (131.2, -116.0, -538.1), (9.5, 3.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-300-B": _achromat("AC508-300-B", 50.8, 300.0, 300.0, 291.5, (153.7, -139.9, -706.8), (8.8, 3.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-400-B": _achromat("AC508-400-B", 50.8, 400.0, 400.0, 392.5, (292.3, -148.9, -398.5), (8.0, 3.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-500-B": _achromat("AC508-500-B", 50.8, 500.0, 500.0, 493.5, (382.5, -182.1, -471.2), (7.5, 3.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-750-B": _achromat("AC508-750-B", 50.8, 750.0, 750.0, 744.0, (398.6, -343.7, -1544.5), (6.8, 3.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC508-1000-B": _achromat("AC508-1000-B", 50.8, 1000.0, 1000.0, 994.5, (757.9, -364.7, -954.2), (6.2, 3.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    # Diameter 75.0 mm (3 inch)
    "AC750-200-B": _achromat("AC750-200-B", 75.0, 200.0, 200.0, 185.0, (106.8, -92.0, -410.0), (16.0, 4.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC750-300-B": _achromat("AC750-300-B", 75.0, 300.0, 300.0, 286.0, (154.5, -139.5, -700.0), (13.0, 4.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
    "AC750-500-B": _achromat("AC750-500-B", 75.0, 500.0, 500.0, 488.0, (382.5, -182.1, -471.2), (11.0, 4.0), ("N-BK7", "N-SF2"), coating_code="B", section="visible_achromats_b", family="NIR Achromatic Doublets"),
}


# =============================================================================
# 3. VISIBLE ACHROMATIC DOUBLETS (A-Coating: 400 - 700 nm)
# =============================================================================
_VISIBLE_ACHROMATS_A: Dict[str, Dict[str, Any]] = {
    # Miniature (≤ 10 mm)
    "AC050-008-A": _achromat("AC050-008-A", 5.0, 7.5, 7.5, 5.2, (5.3, -3.9, -17.1), (2.8, 1.7), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC050-010-A": _achromat("AC050-010-A", 5.0, 10.0, 10.0, 6.9, (6.6, -4.3, -15.4), (2.5, 1.9), ("N-SSK8", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC050-015-A": _achromat("AC050-015-A", 5.0, 15.0, 15.0, 11.6, (12.5, -5.3, -12.1), (2.7, 2.1), ("N-BK7", "SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC060-010-A": _achromat("AC060-010-A", 6.0, 10.0, 10.0, 8.5, (6.2, -4.6, -19.6), (3.5, 1.3), ("N-SSK8", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC064-013-A": _achromat("AC064-013-A", 6.35, 12.7, 12.7, 11.4, (7.1, -5.9, -40.4), (2.8, 1.3), ("N-SSK8", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC064-015-A": _achromat("AC064-015-A", 6.35, 15.0, 15.0, 14.4, (8.8, -6.6, -21.7), (2.3, 1.3), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC080-010-A": _achromat("AC080-010-A", 8.0, 10.0, 10.0, 7.2, (7.1, -5.3, -22.7), (4.2, 1.3), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC080-016-A": _achromat("AC080-016-A", 8.0, 16.0, 16.0, 12.3, (11.0, -9.2, -46.8), (3.5, 1.3), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC080-020-A": _achromat("AC080-020-A", 8.0, 20.0, 20.0, 16.5, (11.1, -9.2, -34.8), (3.0, 1.3), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    # Diameter 12.7 mm (1/2 inch)
    "AC127-019-A": _achromat("AC127-019-A", 12.7, 19.0, 19.0, 14.8, (12.9, -11.0, -59.3), (4.5, 1.5), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC127-025-A": _achromat("AC127-025-A", 12.7, 25.0, 25.0, 20.8, (18.8, -10.6, -68.1), (4.0, 1.5), ("N-BAF10", "N-SF10"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC127-030-A": _achromat("AC127-030-A", 12.7, 30.0, 30.0, 25.8, (17.9, -13.5, -44.2), (3.8, 1.5), ("N-BK7", "SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC127-040-A": _achromat("AC127-040-A", 12.7, 40.0, 40.0, 35.8, (23.7, -20.1, -57.7), (3.5, 1.5), ("N-BK7", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC127-050-A": _achromat("AC127-050-A", 12.7, 50.0, 50.0, 45.8, (27.4, -22.5, -101.4), (3.2, 1.5), ("N-BK7", "SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC127-075-A": _achromat("AC127-075-A", 12.7, 75.0, 75.0, 70.8, (46.5, -33.9, -95.5), (2.8, 1.5), ("N-BK7", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    # Diameter 25.4 mm (1 inch)
    "AC254-030-A": _achromat("AC254-030-A", 25.4, 30.0, 30.0, 22.0, (20.9, -16.7, -79.8), (12.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-035-A": _achromat("AC254-035-A", 25.4, 35.0, 35.0, 27.2, (24.0, -19.1, -102.1), (10.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-040-A": _achromat("AC254-040-A", 25.4, 40.0, 40.0, 32.0, (23.7, -20.1, -57.7), (10.0, 2.5), ("N-BK7", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-045-A": _achromat("AC254-045-A", 25.4, 45.0, 45.0, 37.5, (31.2, -25.9, -130.6), (7.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-050-A": _achromat("AC254-050-A", 25.4, 50.0, 50.0, 43.3, (33.3, -22.3, -291.1), (9.0, 2.5), ("N-BAF10", "N-SF10"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-060-A": _achromat("AC254-060-A", 25.4, 60.0, 60.0, 53.5, (41.7, -25.9, -230.7), (8.0, 2.5), ("N-BAF10", "N-SF10"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-075-A": _achromat("AC254-075-A", 25.4, 75.0, 75.0, 68.0, (46.5, -33.9, -95.5), (7.0, 2.5), ("N-BK7", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-100-A": _achromat("AC254-100-A", 25.4, 100.0, 100.0, 93.0, (62.8, -45.7, -128.2), (4.0, 2.5), ("N-BK7", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-125-A": _achromat("AC254-125-A", 25.4, 125.0, 125.0, 118.5, (77.6, -55.9, -160.8), (4.0, 2.8), ("N-BK7", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-150-A": _achromat("AC254-150-A", 25.4, 150.0, 150.0, 143.0, (91.6, -66.7, -197.7), (5.7, 2.2), ("N-BK7", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-200-A": _achromat("AC254-200-A", 25.4, 200.0, 200.0, 193.5, (77.4, -87.6, 291.1), (4.0, 2.5), ("N-SSK8", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-250-A": _achromat("AC254-250-A", 25.4, 250.0, 250.0, 244.0, (137.1, -111.5, -459.2), (4.0, 2.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-300-A": _achromat("AC254-300-A", 25.4, 300.0, 300.0, 294.5, (165.2, -137.1, -557.4), (4.0, 2.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-400-A": _achromat("AC254-400-A", 25.4, 400.0, 400.0, 395.0, (219.8, -181.6, -738.5), (3.5, 2.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC254-500-A": _achromat("AC254-500-A", 25.4, 500.0, 500.0, 495.5, (337.3, -186.8, -557.4), (3.5, 2.0), ("N-BK7", "SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    # Diameter 30.0 mm
    "AC300-050-A": _achromat("AC300-050-A", 30.0, 50.0, 50.0, 41.5, (41.7, -22.7, -75.7), (10.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC300-080-A": _achromat("AC300-080-A", 30.0, 80.0, 80.0, 72.0, (29.4, -33.9, 97.7), (9.5, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC300-100-A": _achromat("AC300-100-A", 30.0, 100.0, 100.0, 92.5, (33.5, -39.2, 100.7), (8.5, 2.2), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    # Diameter 50.8 mm (2 inch)
    "AC508-075-A": _achromat("AC508-075-A", 50.8, 75.0, 75.0, 61.2, (50.8, -41.7, -247.7), (20.0, 3.0), ("N-BAF10", "N-SF11"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-080-A": _achromat("AC508-080-A", 50.8, 80.0, 80.0, 66.5, (54.9, -46.4, -247.2), (16.0, 2.0), ("N-BAF10", "N-SF6HT"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-100-A": _achromat("AC508-100-A", 50.8, 100.0, 100.0, 86.8, (71.1, -44.2, -363.1), (16.0, 4.0), ("N-BAF10", "N-SF10"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-150-A": _achromat("AC508-150-A", 50.8, 150.0, 150.0, 138.2, (83.2, -72.1, -247.7), (12.0, 3.0), ("N-BK7", "SF5"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-200-A": _achromat("AC508-200-A", 50.8, 200.0, 200.0, 189.5, (106.2, -92.1, -409.4), (10.5, 3.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-250-A": _achromat("AC508-250-A", 50.8, 250.0, 250.0, 240.5, (131.2, -116.0, -538.1), (9.5, 3.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-300-A": _achromat("AC508-300-A", 50.8, 300.0, 300.0, 291.5, (153.7, -139.9, -706.8), (8.8, 3.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-400-A": _achromat("AC508-400-A", 50.8, 400.0, 400.0, 392.5, (292.3, -148.9, -398.5), (8.0, 3.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-500-A": _achromat("AC508-500-A", 50.8, 500.0, 500.0, 493.5, (382.5, -182.1, -471.2), (7.5, 3.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-750-A": _achromat("AC508-750-A", 50.8, 750.0, 750.0, 744.0, (398.6, -343.7, -1544.5), (6.8, 3.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC508-1000-A": _achromat("AC508-1000-A", 50.8, 1000.0, 1000.0, 994.5, (757.9, -364.7, -954.2), (6.2, 3.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    # Diameter 75.0 mm (3 inch)
    "AC750-200-A": _achromat("AC750-200-A", 75.0, 200.0, 200.0, 185.0, (106.8, -92.0, -410.0), (16.0, 4.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC750-300-A": _achromat("AC750-300-A", 75.0, 300.0, 300.0, 286.0, (154.5, -139.5, -700.0), (13.0, 4.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
    "AC750-500-A": _achromat("AC750-500-A", 75.0, 500.0, 500.0, 488.0, (382.5, -182.1, -471.2), (11.0, 4.0), ("N-BK7", "N-SF2"), coating_code="A", section="visible_achromats_a", family="Visible Achromatic Doublets"),
}


# =============================================================================
# 4. PRECISION ASPHERIC CONDENSERS (ACL Series)
# =============================================================================
_ASPHERIC_CONDENSERS: Dict[str, Dict[str, Any]] = {
    "ACL1210U": _asphere("ACL1210U", 12.0, 10.5, 6.0, 6.5, 5.5, -0.65, [2.1e-5]),
    "ACL1512U": _asphere("ACL1512U", 15.0, 12.0, 7.0, 7.5, 6.25, -0.64, [1.5e-5]),
    "ACL2018U": _asphere("ACL2018U", 20.0, 18.0, 11.0, 10.0, 9.38, -0.63, [8.2e-6]),
    "ACL25416U": _asphere("ACL25416U", 25.4, 16.0, 7.3, 14.0, 8.32, -0.6265, [1.8e-6], coating_code="Uncoated"),
    "ACL25416U-A": _asphere("ACL25416U-A", 25.4, 16.0, 7.3, 14.0, 8.32, -0.6265, [1.8e-6], coating_code="A"),
    "ACL25416U-B": _asphere("ACL25416U-B", 25.4, 16.0, 8.5, 12.0, 8.32, -0.6265, [1.8e-6], coating_code="B"),
    "ACL25416U-C": _asphere("ACL25416U-C", 25.4, 16.0, 9.2, 12.0, 8.32, -0.6265, [1.8e-6], coating_code="C"),
    "ACL3026U": _asphere("ACL3026U", 30.0, 26.0, 17.0, 12.0, 13.52, -0.61, [3.5e-6]),
    "ACL5040U": _asphere("ACL5040U", 50.0, 40.0, 23.0, 20.0, 20.8, -0.60, [1.1e-6]),
    "ACL50832U": _asphere("ACL50832U", 50.8, 32.0, 19.5, 24.0, 16.64, -0.62, [1.2e-6]),
}


# =============================================================================
# 5. PLANO-CONVEX SINGLETS (LA Series)
# =============================================================================
_PLANO_CONVEX_SINGLETS: Dict[str, Dict[str, Any]] = {
    # Diameter 12.7 mm (1/2 inch)
    "LA1074": _singlet("LA1074", 12.7, 15.0, 12.5, 7.7, 4.0),
    "LA1213": _singlet("LA1213", 12.7, 20.0, 17.8, 10.3, 3.5),
    "LA1255": _singlet("LA1255", 12.7, 25.4, 23.3, 13.1, 3.2),
    "LA1256": _singlet("LA1256", 12.7, 30.0, 28.1, 15.5, 3.0),
    "LA1257": _singlet("LA1257", 12.7, 40.0, 38.3, 20.6, 2.8),
    "LA1258": _singlet("LA1258", 12.7, 50.0, 48.4, 25.8, 2.6),
    "LA1289": _singlet("LA1289", 12.7, 75.0, 73.5, 38.6, 2.4),
    "LA1200": _singlet("LA1200", 12.7, 100.0, 98.6, 51.5, 2.2),
    # Diameter 25.4 mm (1 inch)
    "LA1951": _singlet("LA1951", 25.4, 25.4, 19.6, 13.1, 8.8),
    "LA1805": _singlet("LA1805", 25.4, 30.0, 24.8, 15.5, 7.9),
    "LA1608": _singlet("LA1608", 25.4, 40.0, 35.6, 20.6, 6.4),
    "LA1131": _singlet("LA1131", 25.4, 50.0, 46.5, 25.8, 5.3),
    "LA1134": _singlet("LA1134", 25.4, 60.0, 56.8, 30.9, 4.8),
    "LA1401": _singlet("LA1401", 25.4, 75.0, 72.2, 38.6, 4.3),
    "LA1509": _singlet("LA1509", 25.4, 100.0, 97.5, 51.5, 3.8),
    "LA1979": _singlet("LA1979", 25.4, 150.0, 147.8, 77.3, 3.3),
    "LA1708": _singlet("LA1708", 25.4, 200.0, 198.0, 103.0, 3.1),
    "LA1251": _singlet("LA1251", 25.4, 250.0, 248.1, 128.8, 3.0),
    "LA1433": _singlet("LA1433", 25.4, 300.0, 298.2, 154.5, 2.8),
    "LA1725": _singlet("LA1725", 25.4, 400.0, 398.3, 206.0, 2.6),
    "LA1417": _singlet("LA1417", 25.4, 500.0, 498.4, 257.5, 2.5),
    "LA1265": _singlet("LA1265", 25.4, 1000.0, 998.5, 515.0, 2.4),
    # Diameter 50.8 mm (2 inch)
    "LA1304": _singlet("LA1304", 50.8, 100.0, 89.5, 51.5, 16.0),
    "LA1401-2": _singlet("LA1401-2", 50.8, 150.0, 142.5, 77.3, 11.5),
    "LA1461": _singlet("LA1461", 50.8, 200.0, 194.0, 103.0, 9.2),
    "LA1253": _singlet("LA1253", 50.8, 250.0, 244.8, 128.8, 8.0),
    "LA1422": _singlet("LA1422", 50.8, 300.0, 295.2, 154.5, 7.3),
    "LA1399": _singlet("LA1399", 50.8, 400.0, 395.8, 206.0, 6.4),
    "LA1464": _singlet("LA1464", 50.8, 500.0, 496.2, 257.5, 5.8),
    "LA1229": _singlet("LA1229", 50.8, 1000.0, 996.8, 515.0, 4.9),
}


# =============================================================================
# 6. BI-CONVEX SINGLETS (LB Series)
# =============================================================================
_BICONVEX_SINGLETS: Dict[str, Dict[str, Any]] = {
    "LB1761": _biconvex("LB1761", 25.4, 25.4, 21.2, 25.0, 8.2),
    "LB1757": _biconvex("LB1757", 25.4, 30.0, 26.4, 29.8, 7.1),
    "LB1844": _biconvex("LB1844", 25.4, 40.0, 37.1, 40.2, 5.7),
    "LB1471": _biconvex("LB1471", 25.4, 50.0, 47.6, 50.6, 4.9),
    "LB1676": _biconvex("LB1676", 25.4, 60.0, 57.9, 61.0, 4.4),
    "LB1901": _biconvex("LB1901", 25.4, 75.0, 73.3, 76.5, 4.0),
    "LB1630": _biconvex("LB1630", 25.4, 100.0, 98.5, 102.3, 3.6),
    "LB1779": _biconvex("LB1779", 25.4, 150.0, 148.8, 153.8, 3.2),
    "LB1945": _biconvex("LB1945", 25.4, 200.0, 199.0, 205.2, 3.0),
}


# =============================================================================
# 7. PLANO-CONCAVE SINGLETS (LC Series)
# =============================================================================
_PLANOCONCAVE_SINGLETS: Dict[str, Dict[str, Any]] = {
    "LC1715": _planoconcave("LC1715", 25.4, -30.0, -32.0, -15.5, 3.0),
    "LC1582": _planoconcave("LC1582", 25.4, -50.0, -52.0, -25.8, 3.0),
    "LC1611": _planoconcave("LC1611", 25.4, -75.0, -77.0, -38.6, 3.0),
    "LC1120": _planoconcave("LC1120", 25.4, -100.0, -102.0, -51.5, 3.0),
}


# =============================================================================
# 8. CONCAVE SPHERICAL MIRRORS (CM Series)
# =============================================================================
_SPHERICAL_MIRRORS: Dict[str, Dict[str, Any]] = {
    # 1 inch (25.4 mm)
    "CM254-025-P01": _mirror("CM254-025-P01", 25.4, 25.0, -50.0, 6.0),
    "CM254-050-P01": _mirror("CM254-050-P01", 25.4, 50.0, -100.0, 6.0),
    "CM254-075-P01": _mirror("CM254-075-P01", 25.4, 75.0, -150.0, 6.0),
    "CM254-100-P01": _mirror("CM254-100-P01", 25.4, 100.0, -200.0, 6.0),
    "CM254-150-P01": _mirror("CM254-150-P01", 25.4, 150.0, -300.0, 6.0),
    "CM254-200-P01": _mirror("CM254-200-P01", 25.4, 200.0, -400.0, 6.0),
    # 2 inch (50.8 mm)
    "CM508-050-P01": _mirror("CM508-050-P01", 50.8, 50.0, -100.0, 10.0),
    "CM508-100-P01": _mirror("CM508-100-P01", 50.8, 100.0, -200.0, 10.0),
    "CM508-200-P01": _mirror("CM508-200-P01", 50.8, 200.0, -400.0, 10.0),
}


# =============================================================================
# 9. CONCAVE PARABOLIC MIRRORS (MPD Series, Conic k = -1.0)
# =============================================================================
_PARABOLIC_MIRRORS: Dict[str, Dict[str, Any]] = {
    # 1/2 inch (12.7 mm) - Protected Silver (-P01: 450 nm - 20 µm)
    "MPD050-015-P01": _parabolic_mirror("MPD050-015-P01", 12.7, 15.0, 6.0, "Protected Silver (P01)", "P01"),
    "MPD050-025-P01": _parabolic_mirror("MPD050-025-P01", 12.7, 25.4, 6.0, "Protected Silver (P01)", "P01"),
    "MPD050-050-P01": _parabolic_mirror("MPD050-050-P01", 12.7, 50.8, 6.0, "Protected Silver (P01)", "P01"),

    # 1/2 inch (12.7 mm) - Protected Gold (-M01: 800 nm - 20 µm)
    "MPD050-015-M01": _parabolic_mirror("MPD050-015-M01", 12.7, 15.0, 6.0, "Protected Gold (M01)", "M01"),
    "MPD050-025-M01": _parabolic_mirror("MPD050-025-M01", 12.7, 25.4, 6.0, "Protected Gold (M01)", "M01"),

    # 1 inch (25.4 mm) - Protected Silver (-P01: 450 nm - 20 µm)
    "MPD129-P01": _parabolic_mirror("MPD129-P01", 25.4, 25.4, 6.0, "Protected Silver (P01)", "P01"),
    "MPD133-P01": _parabolic_mirror("MPD133-P01", 25.4, 33.0, 6.0, "Protected Silver (P01)", "P01"),
    "MPD149-P01": _parabolic_mirror("MPD149-P01", 25.4, 50.8, 6.0, "Protected Silver (P01)", "P01"),
    "MPD169-P01": _parabolic_mirror("MPD169-P01", 25.4, 76.2, 6.0, "Protected Silver (P01)", "P01"),
    "MPD1016-P01": _parabolic_mirror("MPD1016-P01", 25.4, 101.6, 6.0, "Protected Silver (P01)", "P01"),
    "MPD1152-P01": _parabolic_mirror("MPD1152-P01", 25.4, 152.4, 6.0, "Protected Silver (P01)", "P01"),
    "MPD1203-P01": _parabolic_mirror("MPD1203-P01", 25.4, 203.2, 6.0, "Protected Silver (P01)", "P01"),

    # 1 inch (25.4 mm) - Protected Gold (-M01: 800 nm - 20 µm)
    "MPD129-M01": _parabolic_mirror("MPD129-M01", 25.4, 25.4, 6.0, "Protected Gold (M01)", "M01"),
    "MPD149-M01": _parabolic_mirror("MPD149-M01", 25.4, 50.8, 6.0, "Protected Gold (M01)", "M01"),
    "MPD169-M01": _parabolic_mirror("MPD169-M01", 25.4, 76.2, 6.0, "Protected Gold (M01)", "M01"),
    "MPD1016-M01": _parabolic_mirror("MPD1016-M01", 25.4, 101.6, 6.0, "Protected Gold (M01)", "M01"),

    # 1 inch (25.4 mm) - Protected Aluminum (-G01: 250 - 450 nm UV/VIS)
    "MPD129-G01": _parabolic_mirror("MPD129-G01", 25.4, 25.4, 6.0, "Protected Aluminum (G01)", "G01"),
    "MPD149-G01": _parabolic_mirror("MPD149-G01", 25.4, 50.8, 6.0, "Protected Aluminum (G01)", "G01"),
    "MPD169-G01": _parabolic_mirror("MPD169-G01", 25.4, 76.2, 6.0, "Protected Aluminum (G01)", "G01"),

    # 2 inch (50.8 mm) - Protected Silver (-P01: 450 nm - 20 µm)
    "MPD229-P01": _parabolic_mirror("MPD229-P01", 50.8, 50.8, 10.0, "Protected Silver (P01)", "P01"),
    "MPD249-P01": _parabolic_mirror("MPD249-P01", 50.8, 101.6, 10.0, "Protected Silver (P01)", "P01"),
    "MPD269-P01": _parabolic_mirror("MPD269-P01", 50.8, 152.4, 10.0, "Protected Silver (P01)", "P01"),
    "MPD289-P01": _parabolic_mirror("MPD289-P01", 50.8, 203.2, 10.0, "Protected Silver (P01)", "P01"),

    # 2 inch (50.8 mm) - Protected Gold (-M01: 800 nm - 20 µm)
    "MPD229-M01": _parabolic_mirror("MPD229-M01", 50.8, 50.8, 10.0, "Protected Gold (M01)", "M01"),
    "MPD249-M01": _parabolic_mirror("MPD249-M01", 50.8, 101.6, 10.0, "Protected Gold (M01)", "M01"),
    "MPD269-M01": _parabolic_mirror("MPD269-M01", 50.8, 152.4, 10.0, "Protected Gold (M01)", "M01"),

    # 3 inch (75.0 mm / 76.2 mm) - Large Aperture
    "MPD349-P01": _parabolic_mirror("MPD349-P01", 75.0, 152.4, 15.0, "Protected Silver (P01)", "P01"),
    "MPD389-P01": _parabolic_mirror("MPD389-P01", 75.0, 254.0, 15.0, "Protected Silver (P01)", "P01"),
}


# =============================================================================
# 10. DIFFRACTION GRATINGS (GR / GH / GT Series)
# =============================================================================
_DIFFRACTION_GRATINGS: Dict[str, Dict[str, Any]] = {
    # 200 lines/mm
    "GR25-0213": _grating("GR25-0213", 200.0, 1300.0, 7.47, 25.0, "Reflective Ruled Grating (200 l/mm, 1300nm)", "Ruled Reflective Grating"),

    # 300 lines/mm
    "GR25-0305": _grating("GR25-0305", 300.0, 500.0, 4.3, 25.0, "Reflective Ruled Grating (300 l/mm, 500nm)", "Ruled Reflective Grating"),
    "GR25-0310": _grating("GR25-0310", 300.0, 1000.0, 8.6, 25.0, "Reflective Ruled Grating (300 l/mm, 1000nm)", "Ruled Reflective Grating"),
    "GR25-0330": _grating("GR25-0330", 300.0, 3000.0, 26.7, 25.0, "Reflective Ruled Grating (300 l/mm, 3.0µm IR)", "Ruled Reflective Grating"),
    "GR50-0305": _grating("GR50-0305", 300.0, 500.0, 4.3, 50.0, "Reflective Ruled Grating 50mm (300 l/mm, 500nm)", "Ruled Reflective Grating"),
    "GR50-0310": _grating("GR50-0310", 300.0, 1000.0, 8.6, 50.0, "Reflective Ruled Grating 50mm (300 l/mm, 1000nm)", "Ruled Reflective Grating"),

    # 600 lines/mm
    "GR25-0605": _grating("GR25-0605", 600.0, 500.0, 8.6, 25.0, "Reflective Ruled Grating (600 l/mm, 500nm VIS)", "Ruled Reflective Grating"),
    "GR25-0608": _grating("GR25-0608", 600.0, 750.0, 13.0, 25.0, "Reflective Ruled Grating (600 l/mm, 750nm NIR)", "Ruled Reflective Grating"),
    "GR25-0610": _grating("GR25-0610", 600.0, 1000.0, 17.5, 25.0, "Reflective Ruled Grating (600 l/mm, 1000nm NIR)", "Ruled Reflective Grating"),
    "GR25-0616": _grating("GR25-0616", 600.0, 1600.0, 28.7, 25.0, "Reflective Ruled Grating (600 l/mm, 1600nm IR)", "Ruled Reflective Grating"),
    "GR50-0605": _grating("GR50-0605", 600.0, 500.0, 8.6, 50.0, "Reflective Ruled Grating 50mm (600 l/mm, 500nm)", "Ruled Reflective Grating"),
    "GR50-0610": _grating("GR50-0610", 600.0, 1000.0, 17.5, 50.0, "Reflective Ruled Grating 50mm (600 l/mm, 1000nm)", "Ruled Reflective Grating"),
    "GR50-0616": _grating("GR50-0616", 600.0, 1600.0, 28.7, 50.0, "Reflective Ruled Grating 50mm (600 l/mm, 1600nm)", "Ruled Reflective Grating"),

    # 830 lines/mm
    "GR25-0808": _grating("GR25-0808", 830.0, 800.0, 19.4, 25.0, "Reflective Ruled Grating (830 l/mm, 800nm NIR)", "Ruled Reflective Grating"),
    "GR25-0812": _grating("GR25-0812", 830.0, 1200.0, 29.9, 25.0, "Reflective Ruled Grating (830 l/mm, 1200nm IR)", "Ruled Reflective Grating"),

    # 1200 lines/mm
    "GR25-1204": _grating("GR25-1204", 1200.0, 400.0, 13.9, 25.0, "Reflective Ruled Grating (1200 l/mm, 400nm UV)", "Ruled Reflective Grating"),
    "GR25-1205": _grating("GR25-1205", 1200.0, 500.0, 17.5, 25.0, "Reflective Ruled Grating (1200 l/mm, 500nm VIS)", "Ruled Reflective Grating"),
    "GR25-1208": _grating("GR25-1208", 1200.0, 750.0, 26.7, 25.0, "Reflective Ruled Grating (1200 l/mm, 750nm NIR)", "Ruled Reflective Grating"),
    "GR25-1210": _grating("GR25-1210", 1200.0, 1000.0, 36.9, 25.0, "Reflective Ruled Grating (1200 l/mm, 1000nm NIR)", "Ruled Reflective Grating"),
    "GR50-1205": _grating("GR50-1205", 1200.0, 500.0, 17.5, 50.0, "Reflective Ruled Grating 50mm (1200 l/mm, 500nm)", "Ruled Reflective Grating"),
    "GR50-1208": _grating("GR50-1208", 1200.0, 750.0, 26.7, 50.0, "Reflective Ruled Grating 50mm (1200 l/mm, 750nm)", "Ruled Reflective Grating"),
    "GR50-1210": _grating("GR50-1210", 1200.0, 1000.0, 36.9, 50.0, "Reflective Ruled Grating 50mm (1200 l/mm, 1000nm)", "Ruled Reflective Grating"),

    # 1800 lines/mm
    "GR25-1805": _grating("GR25-1805", 1800.0, 500.0, 26.7, 25.0, "Reflective Ruled Grating (1800 l/mm, 500nm VIS)", "Ruled Reflective Grating"),
    "GR50-1805": _grating("GR50-1805", 1800.0, 500.0, 26.7, 50.0, "Reflective Ruled Grating 50mm (1800 l/mm, 500nm)", "Ruled Reflective Grating"),

    # Holographic Reflective Gratings (GH Series - low stray light)
    "GH13-12U": _grating("GH13-12U", 1200.0, 250.0, 0.0, 12.7, "Holographic Grating 1/2\" (1200 l/mm, UV 190-400nm)", "Holographic Grating", coating="Aluminum", coating_code="UV"),
    "GH13-18V": _grating("GH13-18V", 1800.0, 500.0, 0.0, 12.7, "Holographic Grating 1/2\" (1800 l/mm, VIS 350-800nm)", "Holographic Grating", coating="Aluminum", coating_code="VIS"),
    "GH25-12V": _grating("GH25-12V", 1200.0, 600.0, 0.0, 25.0, "Holographic Grating (1200 l/mm, VIS 400-1000nm)", "Holographic Grating", coating="Aluminum", coating_code="VIS"),
    "GH25-18V": _grating("GH25-18V", 1800.0, 500.0, 0.0, 25.0, "Holographic Grating (1800 l/mm, VIS 350-800nm)", "Holographic Grating", coating="Aluminum", coating_code="VIS"),
    "GH25-24U": _grating("GH25-24U", 2400.0, 250.0, 0.0, 25.0, "Holographic Grating (2400 l/mm, UV 190-450nm)", "Holographic Grating", coating="Aluminum", coating_code="UV"),
    "GH25-24V": _grating("GH25-24V", 2400.0, 500.0, 0.0, 25.0, "Holographic Grating (2400 l/mm, VIS 350-700nm)", "Holographic Grating", coating="Aluminum", coating_code="VIS"),
    "GH50-12V": _grating("GH50-12V", 1200.0, 600.0, 0.0, 50.0, "Holographic Grating 50mm (1200 l/mm, VIS 400-1000nm)", "Holographic Grating", coating="Aluminum", coating_code="VIS"),
    "GH50-24V": _grating("GH50-24V", 2400.0, 500.0, 0.0, 50.0, "Holographic Grating 50mm (2400 l/mm, VIS 350-700nm)", "Holographic Grating", coating="Aluminum", coating_code="VIS"),

    # Transmission Diffraction Gratings (GT Series)
    "GT13-03": _grating("GT13-03", 300.0, 550.0, 0.0, 12.7, "Transmission Grating 1/2\" (300 l/mm, VIS 350-800nm)", "Transmission Grating", is_transmission=True, coating="Uncoated", coating_code="TRANS"),
    "GT25-03": _grating("GT25-03", 300.0, 550.0, 0.0, 25.0, "Transmission Grating (300 l/mm, VIS 350-800nm)", "Transmission Grating", is_transmission=True, coating="Uncoated", coating_code="TRANS"),
    "GT25-06": _grating("GT25-06", 600.0, 600.0, 0.0, 25.0, "Transmission Grating (600 l/mm, VIS 400-1000nm)", "Transmission Grating", is_transmission=True, coating="Uncoated", coating_code="TRANS"),
    "GT25-12": _grating("GT25-12", 1200.0, 600.0, 0.0, 25.0, "Transmission Grating (1200 l/mm, VIS 400-1000nm)", "Transmission Grating", is_transmission=True, coating="Uncoated", coating_code="TRANS"),
    "GT50-06": _grating("GT50-06", 600.0, 600.0, 0.0, 50.0, "Transmission Grating 50mm (600 l/mm, VIS 400-1000nm)", "Transmission Grating", is_transmission=True, coating="Uncoated", coating_code="TRANS"),
}


# =============================================================================
# 11. MULTIPHOTON MICROSCOPY ASSEMBLIES
# =============================================================================
# These are effective assembly records, not disclosed internal prescriptions.
_MICROSCOPE_OBJECTIVES: Dict[str, Dict[str, Any]] = {
    "TL10X-2P": {
        "part_number": "TL10X-2P", "manufacturer": "Thorlabs", "section": "microscope_objectives",
        "family": "Multiphoton Microscope Objective", "type": "Effective Objective Model",
        "coating_code": "2P", "coating_range_nm": [450.0, 1300.0], "design_wavelengths_nm": [550.0, 800.0, 1300.0],
        "diameter_mm": 20.0, "semi_diameter": 10.0, "nominal_focal_length_mm": 20.0,
        "focal_length_mm": 20.0, "optical_power_diopters": 50.0, "back_focal_length_mm": 7.77,
        "center_thickness_mm": 0.0, "numerical_aperture": 0.50, "magnification": 10.0,
        "design_tube_lens_focal_length_mm": 200.0, "effective_component": True,
        "surfaces": [{"radius": 0.0, "thickness": 7.77, "glass": "AIR", "surface_type": "STOP"}],
    },
    "TL15X-2P": {
        "part_number": "TL15X-2P", "manufacturer": "Thorlabs", "section": "microscope_objectives",
        "family": "Multiphoton Microscope Objective", "type": "Effective Objective Model",
        "coating_code": "2P", "coating_range_nm": [450.0, 1300.0], "design_wavelengths_nm": [550.0, 800.0, 1300.0],
        "diameter_mm": 18.6666667, "semi_diameter": 9.33333335, "nominal_focal_length_mm": 13.3333333,
        "focal_length_mm": 13.3333333, "optical_power_diopters": 75.0, "back_focal_length_mm": 2.6,
        "center_thickness_mm": 0.0, "numerical_aperture": 0.70, "magnification": 15.0,
        "design_tube_lens_focal_length_mm": 200.0, "effective_component": True,
        "surfaces": [{"radius": 0.0, "thickness": 2.6, "glass": "AIR", "surface_type": "STOP"}],
    },
}

_MULTIPHOTON_SCAN_TUBE_LENSES: Dict[str, Dict[str, Any]] = {
    "SL50-2P2": {
        "part_number": "SL50-2P2", "manufacturer": "Thorlabs", "section": "multiphoton_scan_tube_lenses",
        "family": "Multiphoton Scan Lens", "type": "Effective Scan Lens Model", "coating_code": "2P",
        "coating_range_nm": [680.0, 1400.0], "design_wavelengths_nm": [800.0, 1064.0, 1300.0],
        "diameter_mm": 25.4, "semi_diameter": 12.7, "nominal_focal_length_mm": 50.0,
        "focal_length_mm": 50.0, "optical_power_diopters": 20.0, "back_focal_length_mm": 45.0,
        "center_thickness_mm": 6.0, "effective_component": True,
        "surfaces": [
            {"radius": 35.0, "thickness": 6.0, "glass": "N-BK7"},
            {"radius": -35.0, "thickness": 45.0, "glass": "AIR"},
        ],
    },
    "TTL200MP": {
        "part_number": "TTL200MP", "manufacturer": "Thorlabs", "section": "multiphoton_scan_tube_lenses",
        "family": "Multiphoton Tube Lens", "type": "Effective Tube Lens Model", "coating_code": "MP",
        "coating_range_nm": [680.0, 1300.0], "design_wavelengths_nm": [800.0, 1064.0, 1300.0],
        "diameter_mm": 50.8, "semi_diameter": 25.4, "nominal_focal_length_mm": 200.0,
        "focal_length_mm": 200.0, "optical_power_diopters": 5.0, "back_focal_length_mm": 190.0,
        "center_thickness_mm": 8.0, "effective_component": True,
        "surfaces": [
            {"radius": 105.0, "thickness": 8.0, "glass": "N-BK7"},
            {"radius": -105.0, "thickness": 190.0, "glass": "AIR"},
        ],
    },
}



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

_POLARIZATION_OPTICS: Dict[str, Dict[str, Any]] = {
    "WPH10M-780": _waveplate("WPH10M-780", 25.4, 780.0, "Half-Wave (pi)", (650.0, 1050.0), "N-BK7", 5.0),
    "WPH10M-1064": _waveplate("WPH10M-1064", 25.4, 1064.0, "Half-Wave (pi)", (1050.0, 1700.0), "N-BK7", 5.0),
    "PBS252": _polarization_cube("PBS252", 25.4, (680.0, 1080.0), "N-SF11", 1000.0, 0.92, 0.995),
    "EO-AM-NR-C1": _eom_device("EO-AM-NR-C1", "RTP", 10.0, 240.0, 3.0, 15.0, 0.02),
}

_OPTICAL_FILTERS: Dict[str, Dict[str, Any]] = {
    "FBH790-10": _optical_filter("FBH790-10", "Bandpass", 790.0, 10.0, 6.0, 25.4, 5.0, "FUSED_SILICA", 0.92),
    "FELH1000": _optical_filter("FELH1000", "Longpass", 1000.0, None, 5.0, 25.4, 5.0, "FUSED_SILICA", 0.92),
    "FESH0850": _optical_filter("FESH0850", "Shortpass", 850.0, None, 6.0, 25.4, 3.5, "FUSED_SILICA", 0.90),
    "FESH0800": _optical_filter("FESH0800", "Shortpass", 800.0, None, 6.0, 25.4, 3.5, "FUSED_SILICA", 0.85),
}

_DICHROIC_MIRRORS: Dict[str, Dict[str, Any]] = {
    "DMLP900": _dichroic_mirror("DMLP900", 900.0, 45.0, 25.4, 3.0, "FUSED_SILICA"),
}

_FLAT_MIRRORS: Dict[str, Dict[str, Any]] = {
    "PF10-03-P01": _flat_mirror("PF10-03-P01", "Protected Silver", "P01", 25.4, 6.0, "FUSED_SILICA"),
}

_BEAM_TRAPS: Dict[str, Dict[str, Any]] = {
    "BT610": _beam_trap("BT610", 10.0, 5.0, 10.0, 1.0e-5),
}


# =============================================================================
# THORLABS CATALOG SECTION REGISTRY
# =============================================================================
THORLABS_CATALOG_SECTIONS: Dict[str, Dict[str, Any]] = {
    "achromatic_doublets": {
        "label": "IR Achromatic Doublets (C-Coating: 1050-1700 nm)",
        "description": "C-coated cemented IR achromats (1050-1700 nm)",
        "parts": _IR_ACHROMATIC_DOUBLETS,
    },
    "visible_achromats_b": {
        "label": "NIR Achromatic Doublets (B-Coating: 650-1050 nm)",
        "description": "B-coated cemented NIR achromats (650-1050 nm)",
        "parts": _VISIBLE_ACHROMATS_B,
    },
    "visible_achromats_a": {
        "label": "Visible Achromatic Doublets (A-Coating: 400-700 nm)",
        "description": "A-coated cemented visible achromats (400-700 nm)",
        "parts": _VISIBLE_ACHROMATS_A,
    },
    "aspheric_condensers": {
        "label": "Precision Aspheric Condensers (ACL Series)",
        "description": "Precision molded aspheric condenser lenses (380-2100 nm)",
        "parts": _ASPHERIC_CONDENSERS,
    },
    "singlets": {
        "label": "Plano-Convex Singlets (LA Series)",
        "description": "N-BK7 plano-convex spherical singlets",
        "parts": _PLANO_CONVEX_SINGLETS,
    },
    "biconvex_singlets": {
        "label": "Bi-Convex Singlets (LB Series)",
        "description": "N-BK7 bi-convex spherical singlets",
        "parts": _BICONVEX_SINGLETS,
    },
    "planoconcave_singlets": {
        "label": "Plano-Concave Singlets (LC Series)",
        "description": "N-BK7 plano-concave spherical singlets",
        "parts": _PLANOCONCAVE_SINGLETS,
    },
    "spherical_mirrors": {
        "label": "Concave Spherical Mirrors (CM Series)",
        "description": "Protected silver concave spherical mirrors",
        "parts": _SPHERICAL_MIRRORS,
    },
    "parabolic_mirrors": {
        "label": "Parabolic Mirrors (MPD Series, k=-1.0)",
        "description": "Concave parabolic mirrors across 15mm-254mm focal lengths (Protected Silver, Gold, Aluminum)",
        "parts": _PARABOLIC_MIRRORS,
    },
    "diffraction_gratings": {
        "label": "Diffraction Gratings (GR / GH / GT Series)",
        "description": "Ruled reflective, holographic, and transmission gratings (200-2400 lines/mm)",
        "parts": _DIFFRACTION_GRATINGS,
    },
    "microscope_objectives": {
        "label": "Multiphoton Microscope Objectives (Effective Models)",
        "description": "TL10X-2P and TL15X-2P effective NA/pupil/working-distance models; internal prescriptions are not implied",
        "parts": _MICROSCOPE_OBJECTIVES,
    },
    "multiphoton_scan_tube_lenses": {
        "label": "Multiphoton Scan & Tube Lenses",
        "description": "Effective models for Thorlabs multiphoton scan/tube-lens assemblies",
        "parts": _MULTIPHOTON_SCAN_TUBE_LENSES,
    },
        "amplitude_beamsplitters": {
        "label": "Amplitude Beam Splitters (50:50, 70:30, 90:10 Cubes & Plates)",
        "description": "Thorlabs broadband non-polarizing cube and plate beam splitters (BS013-BS027, BSW04-BSW32, BP108)",
        "parts": _AMPLITUDE_BEAMSPLITTERS,
    },
    "polarizing_beamsplitters": {
        "label": "Polarizing Beam Splitters (PBS Cubes & Wire Grid Plates)",
        "description": "Thorlabs polarizing cube and plate beam splitters (PBS101-PBS514, WPBS25/50 Series)",
        "parts": _POLARIZING_BEAMSPLITTERS,
    },
    "polarization_optics": {
        "label": "Polarization Optics & Modulators (HWP, PBS, EOM)",
        "description": "Zero-order half-wave plates (WPH10M), polarizing cubes (PBS252), and resonant EOMs (EO-AM-NR-C1)",
        "parts": _POLARIZATION_OPTICS,
    },
    "optical_filters": {
        "label": "Laser Clean-Up & Rejection Filters (FBH, FELH, FESH)",
        "description": "Precision bandpass (FBH790-10), longpass (FELH1000), and shortpass (FESH0850/800) filters",
        "parts": _OPTICAL_FILTERS,
    },
    "dichroic_mirrors": {
        "label": "Dichroic Beam Combiners (DMLP Series)",
        "description": "Longpass 45-degree dichroic mirrors for collinear pulse combining (DMLP900)",
        "parts": _DICHROIC_MIRRORS,
    },
    "flat_mirrors": {
        "label": "Precision Flat Steering Mirrors (PF Series)",
        "description": "Protected silver precision flat steering mirrors (PF10-03-P01)",
        "parts": _FLAT_MIRRORS,
    },
    "beam_traps": {
        "label": "Optical Beam Dumps & Traps (BT Series)",
        "description": "High-absorption optical beam dumps absorbing rejected polarization and laser leakage (BT610)",
        "parts": _BEAM_TRAPS,
    },
}


def get_catalog_sections() -> Dict[str, Dict[str, Any]]:
    """Return a defensive copy of the sectioned Thorlabs catalog."""
    return deepcopy(THORLABS_CATALOG_SECTIONS)


def get_lens_catalog() -> Dict[str, Dict[str, Any]]:
    """Return a flat part-number mapping for existing insert/replace callers."""
    parts: Dict[str, Dict[str, Any]] = {}
    for section in THORLABS_CATALOG_SECTIONS.values():
        parts.update(section["parts"])
    return deepcopy(parts)


def get_catalog_lens(part_number: str) -> Optional[Dict[str, Any]]:
    """Retrieve full prescription record for a specific part number."""
    key = part_number.strip().upper()
    all_lenses = get_lens_catalog()
    if key in all_lenses:
        return deepcopy(all_lenses[key])
    for k, record in all_lenses.items():
        if k.upper() == key:
            return deepcopy(record)
    return None


def get_amplitude_beamsplitter_catalog() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of all Thorlabs amplitude / non-polarizing beamsplitters."""
    sec = THORLABS_CATALOG_SECTIONS.get("amplitude_beamsplitters", {})
    return deepcopy(sec.get("parts", {}))


def get_polarizing_beamsplitter_catalog() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of all Thorlabs polarizing beamsplitters (cubes and plates)."""
    sec = THORLABS_CATALOG_SECTIONS.get("polarizing_beamsplitters", {})
    return deepcopy(sec.get("parts", {}))


def get_beamsplitter_catalog() -> Dict[str, Dict[str, Any]]:
    """Return combined dictionary of all amplitude and polarizing beamsplitters."""
    combined = {}
    combined.update(get_amplitude_beamsplitter_catalog())
    combined.update(get_polarizing_beamsplitter_catalog())
    return combined


def query_catalog(
    section: Optional[str] = None,
    diameter_mm: Optional[float] = None,
    focal_length_mm: Optional[float] = None,
    coating_code: Optional[str] = None,
    search_text: Optional[str] = None,
    tolerance_mm: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Filter catalog records by section, diameter, nominal focal length, coating, or search text."""
    records: List[Dict[str, Any]] = []
    sec_filter = section.strip().lower() if section else None
    if sec_filter in ("ir_achromats_c", "ir_achromats", "achromats_c", "c"):
        sec_filter = "achromatic_doublets"
    elif sec_filter in ("nir_achromats_b", "nir_achromats", "b", "nir"):
        sec_filter = "visible_achromats_b"
    elif sec_filter in ("vis_achromats_a", "vis_achromats", "a", "vis"):
        sec_filter = "visible_achromats_a"
    elif sec_filter in ("aspheric", "asphere", "acl", "aspheric_condensers"):
        sec_filter = "aspheric_condensers"
    elif sec_filter in ("singlet", "singlets", "plano_convex", "la"):
        sec_filter = "singlets"
    elif sec_filter in ("biconvex", "biconvex_singlets", "lb"):
        sec_filter = "biconvex_singlets"
    elif sec_filter in ("planoconcave", "planoconcave_singlets", "lc"):
        sec_filter = "planoconcave_singlets"
    elif sec_filter in ("spherical_mirrors", "mirrors", "mirror", "cm"):
        sec_filter = "spherical_mirrors"
    elif sec_filter in ("parabolic_mirrors", "parabolic", "parabola", "oap", "mpd"):
        sec_filter = "parabolic_mirrors"
    elif sec_filter in ("diffraction_gratings", "gratings", "grating", "diffraction", "gr", "gh", "gt"):
        sec_filter = "diffraction_gratings"
    elif sec_filter in ("microscope_objectives", "objective", "objectives", "multiphoton_objectives", "2p_objectives"):
        sec_filter = "microscope_objectives"
    elif sec_filter in ("multiphoton_scan_tube_lenses", "scan_lens", "tube_lens", "multiphoton_lenses", "2p_lenses"):
        sec_filter = "multiphoton_scan_tube_lenses"
    elif sec_filter in ("amplitude_beamsplitter", "amplitude_beamsplitters", "bs", "npbs"):
        sec_filter = "amplitude_beamsplitters"
    elif sec_filter in ("polarizing_beamsplitter", "polarizing_beamsplitters", "pbs", "polarizing_cube", "wpbs"):
        sec_filter = "polarizing_beamsplitters"
    elif sec_filter in ("beamsplitter", "beamsplitters", "all_beamsplitters"):
        sec_filter = "beamsplitters"
    elif sec_filter in ("polarization", "polarizer", "hwp", "waveplate", "eom", "polarization_optics"):
        sec_filter = "polarization_optics"
    elif sec_filter in ("filter", "filters", "bandpass", "longpass", "shortpass", "fbh", "felh", "fesh", "optical_filters"):
        sec_filter = "optical_filters"
    elif sec_filter in ("dichroic", "combiner", "dmlp", "dichroics", "dichroic_mirrors"):
        sec_filter = "dichroic_mirrors"
    elif sec_filter in ("flat_mirror", "steering_mirror", "pf", "flat_mirrors"):
        sec_filter = "flat_mirrors"
    elif sec_filter in ("beam_trap", "beam_traps", "dump", "dumps", "bt", "beam_dump", "beam_dumps"):
        sec_filter = "beam_traps"

    coat_filter = coating_code.strip().upper() if coating_code else None
    query_terms = search_text.strip().lower().split() if search_text else []

    for section_key, section_data in THORLABS_CATALOG_SECTIONS.items():
        if sec_filter:
            if sec_filter == "beamsplitters" and "beamsplitter" in section_key:
                pass
            elif section_key != sec_filter and sec_filter not in section_key:
                continue
        for record in section_data["parts"].values():
            if diameter_mm is not None and abs(record["diameter_mm"] - float(diameter_mm)) > tolerance_mm:
                continue
            if focal_length_mm is not None and abs(record["nominal_focal_length_mm"] - float(focal_length_mm)) > tolerance_mm:
                continue
            if coat_filter and coat_filter != "ALL":
                rec_coat = str(record.get("coating_code", "")).upper()
                if coat_filter == "MIRROR" and any(c in rec_coat for c in ("P01", "M01", "G01", "AL")):
                    pass
                elif coat_filter == "GRATING" and record.get("section") == "diffraction_gratings":
                    pass
                elif coat_filter not in rec_coat and rec_coat != coat_filter:
                    continue
            if query_terms:
                haystack = f"{record['part_number']} {record.get('family', '')} {record.get('type', '')} {record['diameter_mm']} {record['nominal_focal_length_mm']} {record.get('coating_code', '')}".lower()
                if not all(term in haystack for term in query_terms):
                    continue
            records.append(deepcopy(record))

    return sorted(records, key=lambda item: (item["diameter_mm"], item["nominal_focal_length_mm"], item["part_number"]))
