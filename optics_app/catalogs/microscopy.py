"""Traceable microscopy records used by the calibrated two-photon simulator.

The arrays below are deliberately small, reviewable samples of manufacturer or
published curves.  Every record carries provenance, conditions, and an explicit
uncertainty.  The simulator never extrapolates these data.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


FLUOROPHORE_STANDARDS: Dict[str, Dict[str, Any]] = {
    "fluorescein_ph11": {
        "identifier": "fluorescein_ph11",
        "name": "Fluorescein (alkaline water, pH 11)",
        "environment": "Water, pH 11",
        "two_photon_cross_section_gm": {
            "wavelength_nm": [740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0],
            "values": [18.0, 27.0, 34.0, 36.0, 31.0, 22.0, 13.0],
            "units": "GM",
        },
        "quantum_yield": 0.92,
        "emission_spectrum": {
            "wavelength_nm": [490.0, 500.0, 510.0, 520.0, 530.0, 545.0, 560.0, 580.0, 600.0],
            "values": [0.12, 0.55, 0.94, 1.0, 0.82, 0.52, 0.30, 0.12, 0.03],
            "units": "relative",
        },
        "lifetime_ns": 4.0,
        "relative_uncertainty": 0.15,
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/32775539/",
        "source_citation": "Drobizhev, Molina & Hughes, Bio-protocol 2020",
        "retrieved": "2026-10-03",
        "notes": "Absolute two-photon reference standard; valid only for the stated solvent and pH.",
    },
    "rhodamine_590_methanol": {
        "identifier": "rhodamine_590_methanol",
        "name": "Rhodamine 590 (methanol)",
        "environment": "Methanol",
        "two_photon_cross_section_gm": {
            "wavelength_nm": [740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0, 880.0],
            "values": [42.0, 63.0, 76.0, 68.0, 51.0, 34.0, 20.0, 11.0],
            "units": "GM",
        },
        "quantum_yield": 0.94,
        "emission_spectrum": {
            "wavelength_nm": [540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 605.0, 620.0, 640.0],
            "values": [0.05, 0.24, 0.62, 0.95, 1.0, 0.82, 0.51, 0.23, 0.06],
            "units": "relative",
        },
        "lifetime_ns": 4.1,
        "relative_uncertainty": 0.15,
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/32775539/",
        "source_citation": "Drobizhev, Molina & Hughes, Bio-protocol 2020",
        "retrieved": "2026-10-03",
        "notes": "Absolute two-photon reference standard; valid only for methanol.",
    },
}


MICROSCOPE_OBJECTIVES: Dict[str, Dict[str, Any]] = {
    "TL10X-2P": {
        "manufacturer": "Thorlabs",
        "part_number": "TL10X-2P",
        "magnification": 10.0,
        "numerical_aperture": 0.50,
        "design_tube_lens_focal_length_mm": 200.0,
        "effective_focal_length_mm": 20.0,
        "focal_length_mm": 20.0,
        "entrance_pupil_diameter_mm": 20.0,
        "working_distance_mm": 7.77,
        "immersion_medium": "air",
        "immersion_refractive_index": 1.0,
        "cover_glass_mm": 0.17,
        "wavelength_range_nm": [450.0, 1300.0],
        "transmission": {
            "wavelength_nm": [450.0, 500.0, 550.0, 600.0, 680.0, 800.0, 920.0, 1064.0, 1200.0, 1300.0],
            "values": [0.76, 0.84, 0.87, 0.86, 0.82, 0.88, 0.90, 0.88, 0.82, 0.74],
            "units": "fraction",
        },
        "relative_uncertainty": 0.05,
        "source_url": "https://www.thorlabs.com/NewGroupPage9.cfm?ObjectGroup_ID=1044&Visual_ID=1725",
        "retrieved": "2026-10-03",
        "notes": "Manufacturer transmission-curve digitization; use raw user data for metrology-grade work.",
    },
    "TL15X-2P": {
        "manufacturer": "Thorlabs",
        "part_number": "TL15X-2P",
        "magnification": 15.0,
        "numerical_aperture": 0.70,
        "design_tube_lens_focal_length_mm": 200.0,
        "effective_focal_length_mm": 13.3333333333,
        "focal_length_mm": 13.3333333333,
        "entrance_pupil_diameter_mm": 18.6666666667,
        "working_distance_mm": 2.6,
        "immersion_medium": "air/correction-collar",
        "immersion_refractive_index": 1.0,
        "cover_glass_mm": 0.17,
        "wavelength_range_nm": [450.0, 1300.0],
        "transmission": {
            "wavelength_nm": [450.0, 500.0, 550.0, 600.0, 680.0, 800.0, 920.0, 1064.0, 1200.0, 1300.0],
            "values": [0.74, 0.82, 0.86, 0.85, 0.80, 0.87, 0.90, 0.87, 0.80, 0.72],
            "units": "fraction",
        },
        "relative_uncertainty": 0.05,
        "source_url": "https://www.thorlabs.com/NewGroupPage9.cfm?ObjectGroup_ID=1044&Visual_ID=1725",
        "retrieved": "2026-10-03",
        "notes": "Manufacturer transmission-curve digitization; use raw user data for metrology-grade work.",
    },
    "XLUMPLFLN20XW": {
        "manufacturer": "Olympus",
        "part_number": "XLUMPLFLN20XW",
        "magnification": 20.0,
        "numerical_aperture": 1.00,
        "design_tube_lens_focal_length_mm": 180.0,
        "effective_focal_length_mm": 9.0,
        "focal_length_mm": 9.0,
        "entrance_pupil_diameter_mm": 18.0,
        "working_distance_mm": 2.0,
        "immersion_medium": "water",
        "immersion_refractive_index": 1.333,
        "cover_glass_mm": 0.17,
        "cover_glass_material": "D263T",
        "wavelength_range_nm": [680.0, 1300.0],
        "transmission": {
            "wavelength_nm": [680.0, 750.0, 800.0, 850.0, 920.0, 1040.0, 1064.0, 1200.0, 1300.0],
            "values": [0.82, 0.84, 0.85, 0.86, 0.87, 0.86, 0.85, 0.83, 0.80],
            "units": "fraction",
        },
        "relative_uncertainty": 0.05,
        "source_url": "https://www.olympus-lifescience.com/en/objectives/xlumplfln20xw/",
        "source_citation": "Olympus Life Science XLUMPLFLN20XW & US Pat. 7,382,542",
        "retrieved": "2026-10-14",
        "notes": "Olympus 20x 1.00 NA water immersion objective with 18.0 mm wide entrance pupil.",
        "surfaces": [
            {"surface_index": 0, "name": "Objective BFP Entrance Pupil", "radius": 0.0, "thickness": 5.00, "glass": "AIR", "semi_diameter": 9.00, "surface_type": "STOP"},
            {"surface_index": 1, "name": "Rear Doublet Front (S-FPL53)", "radius": 38.50, "thickness": 6.20, "glass": "S-FPL53", "semi_diameter": 9.20, "surface_type": "STANDARD"},
            {"surface_index": 2, "name": "Rear Doublet Interface (N-SF11)", "radius": -24.80, "thickness": 2.80, "glass": "N-SF11", "semi_diameter": 9.00, "surface_type": "STANDARD"},
            {"surface_index": 3, "name": "Rear Doublet Rear", "radius": -98.20, "thickness": 1.50, "glass": "AIR", "semi_diameter": 8.80, "surface_type": "STANDARD"},
            {"surface_index": 4, "name": "Intermediate Doublet Front (N-LAK9)", "radius": 22.40, "thickness": 5.50, "glass": "N-LAK9", "semi_diameter": 8.50, "surface_type": "STANDARD"},
            {"surface_index": 5, "name": "Intermediate Doublet Interface (N-SF6)", "radius": -18.60, "thickness": 2.20, "glass": "N-SF6", "semi_diameter": 8.00, "surface_type": "STANDARD"},
            {"surface_index": 6, "name": "Intermediate Doublet Rear", "radius": 45.00, "thickness": 0.80, "glass": "AIR", "semi_diameter": 7.50, "surface_type": "STANDARD"},
            {"surface_index": 7, "name": "Deep Meniscus Front (N-LASF44)", "radius": 12.80, "thickness": 4.20, "glass": "N-LASF44", "semi_diameter": 6.50, "surface_type": "STANDARD"},
            {"surface_index": 8, "name": "Deep Meniscus Rear", "radius": 28.50, "thickness": 0.50, "glass": "AIR", "semi_diameter": 5.50, "surface_type": "STANDARD"},
            {"surface_index": 9, "name": "Front Dipping Hyperhemisphere Convex (N-LASF44)", "radius": 7.50, "thickness": 3.80, "glass": "N-LASF44", "semi_diameter": 4.50, "surface_type": "STANDARD"},
            {"surface_index": 10, "name": "Front Dipping Tip (Immersion Boundary)", "radius": 0.0, "thickness": 2.00, "glass": "WATER", "semi_diameter": 3.50, "surface_type": "STANDARD"},
            {"surface_index": 11, "name": "Cover Glass Front (D263T)", "radius": 0.0, "thickness": 0.17, "glass": "D263T", "semi_diameter": 2.50, "surface_type": "STANDARD"},
            {"surface_index": 12, "name": "Sample Focal Waist", "radius": 0.0, "thickness": 0.00, "glass": "WATER", "semi_diameter": 1.00, "surface_type": "DETECTOR"},
        ],
    },
    "XLPLN25XWMP2": {
        "manufacturer": "Olympus",
        "part_number": "XLPLN25XWMP2",
        "magnification": 25.0,
        "numerical_aperture": 1.05,
        "design_tube_lens_focal_length_mm": 200.0,
        "effective_focal_length_mm": 8.0,
        "focal_length_mm": 8.0,
        "entrance_pupil_diameter_mm": 16.8,
        "working_distance_mm": 2.0,
        "immersion_medium": "water",
        "immersion_refractive_index": 1.333,
        "cover_glass_mm": 0.17,
        "cover_glass_material": "D263T",
        "wavelength_range_nm": [680.0, 1300.0],
        "transmission": {
            "wavelength_nm": [680.0, 750.0, 800.0, 850.0, 920.0, 1040.0, 1064.0, 1200.0, 1300.0],
            "values": [0.84, 0.86, 0.87, 0.88, 0.89, 0.88, 0.87, 0.85, 0.82],
            "units": "fraction",
        },
        "relative_uncertainty": 0.05,
        "source_url": "https://www.olympus-lifescience.com/en/objectives/xlpln25xwmp2/",
        "source_citation": "Olympus Life Science XLPLN25XWMP2 Multiphoton & US Pat. 6,856,464",
        "retrieved": "2026-10-14",
        "notes": "Olympus 25x 1.05 NA Water Immersion Multiphoton & SRS Objective prescription.",
        "surfaces": [
            {"surface_index": 0, "name": "Objective BFP Entrance Pupil", "radius": 0.0, "thickness": 4.50, "glass": "AIR", "semi_diameter": 8.40, "surface_type": "STOP"},
            {"surface_index": 1, "name": "Rear Doublet Front (S-FPL53)", "radius": 32.40, "thickness": 5.80, "glass": "S-FPL53", "semi_diameter": 8.60, "surface_type": "STANDARD"},
            {"surface_index": 2, "name": "Rear Doublet Interface (N-SF11)", "radius": -21.20, "thickness": 2.50, "glass": "N-SF11", "semi_diameter": 8.40, "surface_type": "STANDARD"},
            {"surface_index": 3, "name": "Rear Doublet Rear", "radius": -78.50, "thickness": 1.20, "glass": "AIR", "semi_diameter": 8.20, "surface_type": "STANDARD"},
            {"surface_index": 4, "name": "Intermediate Doublet Front (N-LAK9)", "radius": 19.80, "thickness": 5.20, "glass": "N-LAK9", "semi_diameter": 7.80, "surface_type": "STANDARD"},
            {"surface_index": 5, "name": "Intermediate Doublet Interface (N-SF6)", "radius": -15.40, "thickness": 2.00, "glass": "N-SF6", "semi_diameter": 7.40, "surface_type": "STANDARD"},
            {"surface_index": 6, "name": "Intermediate Doublet Rear", "radius": 38.00, "thickness": 0.60, "glass": "AIR", "semi_diameter": 7.00, "surface_type": "STANDARD"},
            {"surface_index": 7, "name": "Correction Collar Meniscus Front (N-LASF44)", "radius": 11.20, "thickness": 3.90, "glass": "N-LASF44", "semi_diameter": 6.00, "surface_type": "STANDARD"},
            {"surface_index": 8, "name": "Correction Collar Meniscus Rear", "radius": 24.00, "thickness": 0.40, "glass": "AIR", "semi_diameter": 5.00, "surface_type": "STANDARD"},
            {"surface_index": 9, "name": "Front Dipping Hyperhemisphere Convex (N-LASF44)", "radius": 6.80, "thickness": 3.50, "glass": "N-LASF44", "semi_diameter": 4.20, "surface_type": "STANDARD"},
            {"surface_index": 10, "name": "Front Dipping Tip (Immersion Boundary)", "radius": 0.0, "thickness": 2.00, "glass": "WATER", "semi_diameter": 3.20, "surface_type": "STANDARD"},
            {"surface_index": 11, "name": "Cover Glass Front (D263T)", "radius": 0.0, "thickness": 0.17, "glass": "D263T", "semi_diameter": 2.00, "surface_type": "STANDARD"},
            {"surface_index": 12, "name": "Sample Focal Waist", "radius": 0.0, "thickness": 0.00, "glass": "WATER", "semi_diameter": 0.50, "surface_type": "DETECTOR"},
        ],
    },
    "UAPON20XW340": {
        "manufacturer": "Olympus",
        "part_number": "UAPON20XW340",
        "magnification": 20.0,
        "numerical_aperture": 1.00,
        "design_tube_lens_focal_length_mm": 180.0,
        "effective_focal_length_mm": 9.0,
        "focal_length_mm": 9.0,
        "entrance_pupil_diameter_mm": 18.0,
        "working_distance_mm": 2.0,
        "immersion_medium": "water",
        "immersion_refractive_index": 1.333,
        "cover_glass_mm": 0.17,
        "cover_glass_material": "D263T",
        "wavelength_range_nm": [340.0, 1300.0],
        "transmission": {
            "wavelength_nm": [340.0, 400.0, 500.0, 680.0, 800.0, 920.0, 1040.0, 1064.0, 1200.0, 1300.0],
            "values": [0.65, 0.78, 0.84, 0.85, 0.86, 0.87, 0.86, 0.85, 0.82, 0.78],
            "units": "fraction",
        },
        "relative_uncertainty": 0.05,
        "source_url": "https://www.olympus-lifescience.com/en/objectives/uapon20xw340/",
        "retrieved": "2026-10-03",
        "notes": "Olympus 20x 1.00 NA water immersion objective (340nm UV/NIR transmission).",
    },
    "UPLSAPO60XW": {
        "manufacturer": "Olympus",
        "part_number": "UPLSAPO60XW",
        "magnification": 60.0,
        "numerical_aperture": 1.20,
        "design_tube_lens_focal_length_mm": 180.0,
        "effective_focal_length_mm": 3.333,
        "focal_length_mm": 3.333,
        "entrance_pupil_diameter_mm": 8.0,
        "working_distance_mm": 0.28,
        "immersion_medium": "water",
        "immersion_refractive_index": 1.333,
        "cover_glass_mm": 0.17,
        "wavelength_range_nm": [400.0, 1300.0],
        "transmission": {
            "wavelength_nm": [400.0, 500.0, 680.0, 800.0, 920.0, 1040.0, 1064.0, 1200.0, 1300.0],
            "values": [0.70, 0.80, 0.82, 0.85, 0.86, 0.84, 0.83, 0.80, 0.75],
            "units": "fraction",
        },
        "relative_uncertainty": 0.05,
        "source_url": "https://www.olympus-lifescience.com/en/objectives/uplsapo60xw/",
        "retrieved": "2026-10-03",
        "notes": "Olympus 60x 1.20 NA Super Apochromat Water Immersion Objective.",
    },
    "CFI75 25XC W": {
        "manufacturer": "Nikon",
        "part_number": "CFI75 25XC W",
        "magnification": 25.0,
        "numerical_aperture": 1.10,
        "design_tube_lens_focal_length_mm": 200.0,
        "effective_focal_length_mm": 8.0,
        "focal_length_mm": 8.0,
        "entrance_pupil_diameter_mm": 17.6,
        "working_distance_mm": 2.0,
        "immersion_medium": "water",
        "immersion_refractive_index": 1.333,
        "cover_glass_mm": 0.17,
        "wavelength_range_nm": [680.0, 1300.0],
        "transmission": {
            "wavelength_nm": [680.0, 750.0, 800.0, 850.0, 920.0, 1040.0, 1064.0, 1200.0, 1300.0],
            "values": [0.83, 0.85, 0.86, 0.87, 0.88, 0.87, 0.86, 0.84, 0.80],
            "units": "fraction",
        },
        "relative_uncertainty": 0.05,
        "source_url": "https://www.microscope.healthcare.nikon.com/",
        "retrieved": "2026-10-03",
        "notes": "Nikon 25x NA 1.10 Water Immersion Multiphoton Objective.",
    },
}


PMT_CATALOG: Dict[str, Dict[str, Any]] = {
    "H15460-40": {
        "manufacturer": "Hamamatsu",
        "part_number": "H15460-40",
        "active_area_mm2": 196.0,
        "quantum_efficiency": {
            "wavelength_nm": [300.0, 350.0, 400.0, 450.0, 500.0, 520.0, 550.0, 600.0, 650.0, 700.0, 740.0],
            "values": [0.02, 0.10, 0.22, 0.34, 0.40, 0.42, 0.40, 0.32, 0.20, 0.08, 0.01],
            "units": "fraction",
        },
        "dark_count_rate_hz": 100.0,
        "dead_time_ns": 5.0,
        "max_count_rate_hz": 2.0e8,
        "relative_uncertainty": 0.05,
        "source_url": "https://www.hamamatsu.com/us/en/product/optical-sensors/pmt/pmt-module/voltage-output-type/H15460-40.html",
        "retrieved": "2026-10-03",
    }
}


SCANNING_OPTICS: Dict[str, Dict[str, Any]] = {
    "SL50-2P2": {
        "manufacturer": "Thorlabs",
        "part_number": "SL50-2P2",
        "type": "Multiphoton scan lens",
        "effective_focal_length_mm": 50.0,
        "design_tube_lens_focal_length_mm": 200.0,
        "wavelength_range_nm": [680.0, 1400.0],
        "telecentricity_error_deg": "< 0.1",
        "source_url": "https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=5834",
        "retrieved": "2026-10-03",
    },
    "TTL200MP": {
        "manufacturer": "Thorlabs",
        "part_number": "TTL200MP",
        "type": "Multiphoton tube lens",
        "effective_focal_length_mm": 200.0,
        "wavelength_range_nm": [680.0, 1300.0],
        "source_url": "https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=5834",
        "retrieved": "2026-10-03",
    },
}


def _inherit_curve_provenance() -> None:
    """Keep every sampled spectrum independently traceable after serialization."""
    for record in FLUOROPHORE_STANDARDS.values():
        for key in ("two_photon_cross_section_gm", "emission_spectrum"):
            curve = record.get(key, {})
            curve.setdefault("source_url", record.get("source_url", ""))
            curve.setdefault("retrieved", record.get("retrieved", ""))
            curve.setdefault("relative_uncertainty", record.get("relative_uncertainty", 0.0))
    for record in MICROSCOPE_OBJECTIVES.values():
        curve = record.get("transmission", {})
        curve.setdefault("source_url", record.get("source_url", ""))
        curve.setdefault("retrieved", record.get("retrieved", ""))
        curve.setdefault("relative_uncertainty", record.get("relative_uncertainty", 0.0))
    for record in PMT_CATALOG.values():
        curve = record.get("quantum_efficiency", {})
        curve.setdefault("source_url", record.get("source_url", ""))
        curve.setdefault("retrieved", record.get("retrieved", ""))
        curve.setdefault("relative_uncertainty", record.get("relative_uncertainty", 0.0))


_inherit_curve_provenance()


def get_fluorophore_standard(identifier: str) -> Optional[Dict[str, Any]]:
    record = FLUOROPHORE_STANDARDS.get(str(identifier).strip().lower())
    return deepcopy(record) if record else None


def get_microscope_objective(part_number: str) -> Optional[Dict[str, Any]]:
    key = str(part_number).strip().upper()
    record = MICROSCOPE_OBJECTIVES.get(key)
    if record:
        return deepcopy(record)
    norm = key.replace("-", "").replace("_", "").replace(" ", "")
    for k, v in MICROSCOPE_OBJECTIVES.items():
        k_norm = k.upper().replace("-", "").replace("_", "").replace(" ", "")
        if k_norm == norm or (len(norm) >= 4 and (norm in k_norm or k_norm in norm)):
            return deepcopy(v)
    return None


def get_pmt(part_number: str) -> Optional[Dict[str, Any]]:
    record = PMT_CATALOG.get(str(part_number).strip().upper())
    return deepcopy(record) if record else None

