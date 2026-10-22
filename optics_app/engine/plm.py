"""
Texas Instruments 4-Bit Phase Light Modulator (TI-PLM) Physics & Pattern Engine.

Simulates reflective MEMS piston micromirror Phase Light Modulators such as the
TI DLP670S / DLP6750Q1 (0.67" diagonal, 1358 x 800 phase pixels, 10.8 um pitch,
4-bit = 16 discrete piston/phase levels in [0, 2*pi * 15/16]).

Capabilities:
1. 4-Bit Phase / Piston Quantization (16 discrete states: 0..15, linear or custom LUT).
2. Zernike Polynomial Phase Synthesis (Noll / OSA-ANSI modes Z1..Z15 with analytical
   spatial gradients for exact 3D geometric + wave-optics ray tracing).
3. Programmable Phase Gratings (Blazed 4-bit staircase, Binary 0/pi, Sinusoidal,
   Chirped Variable-Line-Spacing VLS, and Hadamard Coded Spectral Masks).
4. 2D Pattern Loading & Export (.npy, .csv, .png, .bmp, .json) plus built-in
   synthetic holograms (Fresnel zone lens, optical vortex, Airy cubic mask,
   Walsh-Hadamard spectral masks, checkerboard).
5. Far-field PSF, 4-bit quantization efficiency, and diffraction order efficiency calculator.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# TI-PLM Hardware Specifications (TI DLP670S / DLP6750Q1 0.67" 4-Bit PLM)
# ============================================================================

TI_PLM_MODELS: Dict[str, Dict[str, Any]] = {
    "DLP670S": {
        "part_number": "DLP670S",
        "manufacturer": "Texas Instruments",
        "modulator_type": "PLM",
        "description": "TI DLP670S 0.67-inch 4-Bit Phase Light Modulator (1358x800, 10.8 um pitch)",
        "array_cols": 1358,
        "array_rows": 800,
        "pixel_pitch_um": 10.8,
        "active_width_mm": 14.6664,   # 1358 * 0.0108 mm
        "active_height_mm": 8.6400,   # 800 * 0.0108 mm
        "semi_diameter_mm": 8.5112,   # sqrt(7.3332^2 + 4.32^2)
        "bit_depth": 4,
        "num_states": 16,
        "fill_factor": 0.92,
        "mirror_reflectivity": 0.88,
        "switching_time_us": 50.0,
        "max_pattern_rate_hz": 1440.0,
        "window_transmission": 0.97,
    },
    "DLP6750Q1": {
        "part_number": "DLP6750Q1",
        "manufacturer": "Texas Instruments",
        "modulator_type": "PLM",
        "description": "TI DLP6750Q1 0.67-inch 4-Bit Automotive/Visible PLM (1358x800, 10.8 um pitch)",
        "array_cols": 1358,
        "array_rows": 800,
        "pixel_pitch_um": 10.8,
        "active_width_mm": 14.6664,
        "active_height_mm": 8.6400,
        "semi_diameter_mm": 8.5112,
        "bit_depth": 4,
        "num_states": 16,
        "fill_factor": 0.92,
        "mirror_reflectivity": 0.89,
        "switching_time_us": 50.0,
        "max_pattern_rate_hz": 1440.0,
        "window_transmission": 0.98,
    },
}

TI_DMD_MODELS: Dict[str, Dict[str, Any]] = {
    "DLP4500": {
        "part_number": "DLP4500",
        "manufacturer": "Texas Instruments",
        "modulator_type": "DMD",
        "description": "TI DLP4500 0.45-inch WXGA DMD (1140x912, 7.6 um pitch, +/-12 deg tilt)",
        "array_cols": 1140,
        "array_rows": 912,
        "pixel_pitch_um": 7.6,
        "active_width_mm": 8.664,
        "active_height_mm": 6.9312,
        "semi_diameter_mm": 5.547,
        "tilt_angle_deg": 12.0,
        "bit_depth": 1,
        "num_states": 2,
        "fill_factor": 0.92,
        "mirror_reflectivity": 0.88,
        "switching_time_us": 15.0,
        "max_pattern_rate_hz": 4225.0,
        "window_transmission": 0.97,
    },
    "DLP6500": {
        "part_number": "DLP6500",
        "manufacturer": "Texas Instruments",
        "modulator_type": "DMD",
        "description": "TI DLP6500 0.65-inch 1080p DMD (1920x1080, 7.56 um pitch, +/-12 deg tilt)",
        "array_cols": 1920,
        "array_rows": 1080,
        "pixel_pitch_um": 7.56,
        "active_width_mm": 14.5152,
        "active_height_mm": 8.1648,
        "semi_diameter_mm": 8.324,
        "tilt_angle_deg": 12.0,
        "bit_depth": 1,
        "num_states": 2,
        "fill_factor": 0.92,
        "mirror_reflectivity": 0.88,
        "switching_time_us": 10.0,
        "max_pattern_rate_hz": 9500.0,
        "window_transmission": 0.97,
    },
    "DLP9000": {
        "part_number": "DLP9000",
        "manufacturer": "Texas Instruments",
        "modulator_type": "DMD",
        "description": "TI DLP9000 0.9-inch WQXGA DMD (2560x1600, 7.56 um pitch, +/-12 deg tilt)",
        "array_cols": 2560,
        "array_rows": 1600,
        "pixel_pitch_um": 7.56,
        "active_width_mm": 19.3536,
        "active_height_mm": 12.096,
        "semi_diameter_mm": 11.411,
        "tilt_angle_deg": 12.0,
        "bit_depth": 1,
        "num_states": 2,
        "fill_factor": 0.92,
        "mirror_reflectivity": 0.88,
        "switching_time_us": 10.0,
        "max_pattern_rate_hz": 9500.0,
        "window_transmission": 0.97,
    },
    "DLP2010": {
        "part_number": "DLP2010",
        "manufacturer": "Texas Instruments",
        "modulator_type": "DMD",
        "description": "TI DLP2010 0.2-inch WVGA DMD (854x480, 5.4 um pitch, +/-17 deg tilt)",
        "array_cols": 854,
        "array_rows": 480,
        "pixel_pitch_um": 5.4,
        "active_width_mm": 4.6116,
        "active_height_mm": 2.592,
        "semi_diameter_mm": 2.645,
        "tilt_angle_deg": 17.0,
        "bit_depth": 1,
        "num_states": 2,
        "fill_factor": 0.90,
        "mirror_reflectivity": 0.88,
        "switching_time_us": 20.0,
        "max_pattern_rate_hz": 2880.0,
        "window_transmission": 0.97,
    },
}

TI_ALL_MODELS: Dict[str, Dict[str, Any]] = {**TI_PLM_MODELS, **TI_DMD_MODELS}
TI_PLM_DLP670S_SPECS = TI_PLM_MODELS["DLP670S"]
TI_DMD_DLP4500_SPECS = TI_DMD_MODELS["DLP4500"]


# Standard Zernike mode metadata (Noll 1..15)
ZERNIKE_MODES_INFO: List[Dict[str, Any]] = [
    {"noll": 1,  "n": 0, "m": 0,  "key": "z1_piston",          "name": "Z1: Piston",                       "formula": "1"},
    {"noll": 2,  "n": 1, "m": 1,  "key": "z2_tilt_x",          "name": "Z2: Tip / Tilt X",                 "formula": "2*r*cos(t)"},
    {"noll": 3,  "n": 1, "m": -1, "key": "z3_tilt_y",          "name": "Z3: Tilt Y",                       "formula": "2*r*sin(t)"},
    {"noll": 4,  "n": 2, "m": 0,  "key": "z4_defocus",         "name": "Z4: Defocus",                      "formula": "sqrt(3)*(2*r^2 - 1)"},
    {"noll": 5,  "n": 2, "m": -2, "key": "z5_astig_oblique",   "name": "Z5: Oblique Astigmatism (45°)",    "formula": "sqrt(6)*r^2*sin(2t)"},
    {"noll": 6,  "n": 2, "m": 2,  "key": "z6_astig_vertical",  "name": "Z6: Vertical Astigmatism (0°)",    "formula": "sqrt(6)*r^2*cos(2t)"},
    {"noll": 7,  "n": 3, "m": -1, "key": "z7_coma_vertical",   "name": "Z7: Vertical Coma (Y)",            "formula": "sqrt(8)*(3*r^3 - 2*r)*sin(t)"},
    {"noll": 8,  "n": 3, "m": 1,  "key": "z8_coma_horizontal", "name": "Z8: Horizontal Coma (X)",          "formula": "sqrt(8)*(3*r^3 - 2*r)*cos(t)"},
    {"noll": 9,  "n": 3, "m": -3, "key": "z9_trefoil_vert",    "name": "Z9: Vertical Trefoil",             "formula": "sqrt(8)*r^3*sin(3t)"},
    {"noll": 10, "n": 3, "m": 3,  "key": "z10_trefoil_obl",    "name": "Z10: Oblique Trefoil",             "formula": "sqrt(8)*r^3*cos(3t)"},
    {"noll": 11, "n": 4, "m": 0,  "key": "z11_spherical",      "name": "Z11: Primary Spherical",           "formula": "sqrt(5)*(6*r^4 - 6*r^2 + 1)"},
    {"noll": 12, "n": 4, "m": 2,  "key": "z12_sec_astig_vert", "name": "Z12: Secondary Astigmatism (0°)",  "formula": "sqrt(10)*(4*r^4 - 3*r^2)*cos(2t)"},
    {"noll": 13, "n": 4, "m": -2, "key": "z13_sec_astig_obl",  "name": "Z13: Secondary Astigmatism (45°)", "formula": "sqrt(10)*(4*r^4 - 3*r^2)*sin(2t)"},
    {"noll": 14, "n": 4, "m": 4,  "key": "z14_tetrafoil",      "name": "Z14: Tetrafoil (0°)",              "formula": "sqrt(10)*r^4*cos(4t)"},
    {"noll": 15, "n": 6, "m": 0,  "key": "z15_sec_spherical",  "name": "Z15: Secondary Spherical",         "formula": "sqrt(7)*(20*r^6 - 30*r^4 + 12*r^2 - 1)"},
]


def build_4bit_state_table(design_wavelength_um: float = 1.225, lut_fractions: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """
    Build the 16 discrete 4-bit piston/phase states (0..15) for the TI-PLM.
    In reflection at normal incidence, optical path difference OPD = 2 * h_s,
    so a full 2*pi wave corresponds to mirror piston travel lambda_0 / 2.
    State s in {0..15} has fractional phase f_s = s / 16 (or custom LUT value).
    """
    wl_nm = float(design_wavelength_um) * 1000.0
    max_piston_half_wave_nm = wl_nm / 2.0
    table: List[Dict[str, Any]] = []
    for s in range(16):
        frac = float(lut_fractions[s]) if (lut_fractions and len(lut_fractions) >= 16) else (s / 16.0)
        phase_rad = 2.0 * math.pi * frac
        phase_deg = 360.0 * frac
        piston_nm = frac * max_piston_half_wave_nm
        opd_nm = 2.0 * piston_nm
        table.append({
            "state": s,
            "name": f"STATE_{s}",
            "binary": format(s, "04b"),
            "hex": f"0x{s:X}",
            "phase_fraction_2pi": round(frac, 6),
            "phase_rad": round(phase_rad, 6),
            "phase_deg": round(phase_deg, 3),
            "piston_nm": round(piston_nm, 3),
            "opd_nm": round(opd_nm, 3),
        })
    return table


def get_dmd_mechanical_states(tilt_angle_deg: float = 12.0) -> Dict[str, Dict[str, Any]]:
    """
    Return the dictionary of discrete mechanical states for a Texas Instruments DMD.
    In a DMD, micromirrors operate in binary tilt (+/- tilt_angle_deg):
    - ON: Reflected light follows the designed detector path (+tilt_angle_deg)
    - OFF: Micromirrors tilted into beam dump / light trap (-tilt_angle_deg)
    - FLAT: Unpowered flat parking state (0.0 deg)
    - HADAMARD_A: Active binary Walsh-Hadamard mask (+1 bands ON, -1 bands OFF)
    - HADAMARD_B: Complementary binary Walsh-Hadamard mask (-1 bands ON, +1 bands OFF)
    """
    tilt = float(tilt_angle_deg)
    return {
        "ON": {
            "tilt_x": tilt,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "ACTIVE_PATTERN",
            "route": "detector_path",
            "tilt_angle_deg": tilt,
            "description": f"TI-DMD Active Binary Pattern (+{tilt:.1f}° Tilt -> Detector Path)",
        },
        "OFF": {
            "tilt_x": -tilt,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "FLAT_ZERO",
            "uniform_state": 0,
            "route": "beam_dump",
            "tilt_angle_deg": -tilt,
            "description": f"TI-DMD OFF State (-{tilt:.1f}° Tilt -> Beam Dump / Light Trap)",
        },
        "FLAT": {
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "FLAT_ZERO",
            "uniform_state": 0,
            "route": "parking",
            "tilt_angle_deg": 0.0,
            "description": "TI-DMD Unpowered Parking State (0.0° Tilt -> Specular Reflection)",
        },
        "HADAMARD_A": {
            "tilt_x": tilt,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "HADAMARD_A",
            "route": "detector_path",
            "tilt_angle_deg": tilt,
            "description": "TI-DMD Binary Walsh-Hadamard Mask A (+1 Bands ON -> Detector, -1 Bands OFF -> Dump)",
        },
        "HADAMARD_B": {
            "tilt_x": tilt,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "HADAMARD_B",
            "route": "detector_path",
            "tilt_angle_deg": tilt,
            "description": "TI-DMD Binary Walsh-Hadamard Complement Mask B (-1 Bands ON -> Detector, +1 Bands OFF -> Dump)",
        },
    }


def get_plm_mechanical_states(
    design_wavelength_um: float = 1.225,
    modulator_type: str = "PLM",
    tilt_angle_deg: float = 12.0
) -> Dict[str, Dict[str, Any]]:
    """
    Return the dictionary of discrete states exposed to MechanicalStateController
    for a TI-PLM or TI-DMD surface.
    """
    if str(modulator_type).upper() == "DMD":
        return get_dmd_mechanical_states(tilt_angle_deg=tilt_angle_deg)

    state_table = build_4bit_state_table(design_wavelength_um)
    states: Dict[str, Dict[str, Any]] = {
        "ON": {
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "ACTIVE_PATTERN",
            "route": "detector_path",
            "description": "TI-PLM Active 4-Bit Phase Pattern (Zernike + Grating + Custom Pattern -> +1st Order Detector Path)",
        },
        "OFF": {
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "FLAT_ZERO",
            "uniform_state": 0,
            "route": "beam_dump",
            "description": "TI-PLM Flat 0th-Order Specular Bypass (All Pixels State 0000 = 0 rad -> 0th Order Dump)",
        },
        "HADAMARD_A": {
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "HADAMARD_A",
            "route": "detector_path",
            "description": "TI-PLM 4-Bit Hadamard Mask A (+1 Bands Blazed to Detector, -1 Bands Flat to Dump)",
        },
        "HADAMARD_B": {
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "HADAMARD_B",
            "route": "detector_path",
            "description": "TI-PLM 4-Bit Hadamard Complement Mask B (-1 Bands Blazed to Detector, +1 Bands Flat to Dump)",
        },
    }
    for entry in state_table:
        s = entry["state"]
        states[f"STATE_{s}"] = {
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
            "transmission": 1.0,
            "plm_mode": "UNIFORM_STATE",
            "uniform_state": s,
            "piston_nm": entry["piston_nm"],
            "phase_rad": entry["phase_rad"],
            "phase_deg": entry["phase_deg"],
            "binary": entry["binary"],
            "route": "specular_0th_order",
            "description": (
                f"TI-PLM Uniform 4-Bit State {s} ({entry['binary']} / {entry['hex']}): "
                f"Phase = {entry['phase_deg']:.1f} deg ({entry['phase_rad']:.3f} rad), "
                f"Piston = {entry['piston_nm']:.1f} nm"
            ),
        }
    return states


# ============================================================================
# Dataclasses for Zernike, Grating, Pattern, and Full PLM Configuration
# ============================================================================

@dataclass
class ZernikePhaseConfig:
    """Configuration for Zernike wavefront phase synthesis on the TI-PLM."""
    enabled: bool = True
    normalization_radius_mm: float = 4.32      # Reference pupil radius R_norm (mm)
    center_x_mm: float = 0.0                   # Pupil center X offset on PLM (mm)
    center_y_mm: float = 0.0                   # Pupil center Y offset on PLM (mm)
    units: str = "waves"                       # 'waves' (multiples of design_wavelength_um) or 'radians'
    # Noll Z1..Z15 coefficients (in waves at design_wavelength_um by default)
    coeffs: Dict[str, float] = field(default_factory=lambda: {
        "z1_piston": 0.0,
        "z2_tilt_x": 0.0,
        "z3_tilt_y": 0.0,
        "z4_defocus": 0.0,
        "z5_astig_oblique": 0.0,
        "z6_astig_vertical": 0.0,
        "z7_coma_vertical": 0.0,
        "z8_coma_horizontal": 0.0,
        "z9_trefoil_vert": 0.0,
        "z10_trefoil_obl": 0.0,
        "z11_spherical": 0.0,
        "z12_sec_astig_vert": 0.0,
        "z13_sec_astig_obl": 0.0,
        "z14_tetrafoil": 0.0,
        "z15_sec_spherical": 0.0,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "normalization_radius_mm": float(self.normalization_radius_mm),
            "center_x_mm": float(self.center_x_mm),
            "center_y_mm": float(self.center_y_mm),
            "units": str(self.units),
            "coeffs": {str(k): float(v) for k, v in (self.coeffs or {}).items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZernikePhaseConfig":
        if not isinstance(data, dict):
            return cls()
        default_coeffs = cls().coeffs
        raw_coeffs = data.get("coeffs", {})
        if isinstance(raw_coeffs, dict):
            for k, v in raw_coeffs.items():
                try:
                    default_coeffs[str(k)] = float(v)
                except (ValueError, TypeError):
                    pass
        return cls(
            enabled=bool(data.get("enabled", True)),
            normalization_radius_mm=float(data.get("normalization_radius_mm", 4.32)),
            center_x_mm=float(data.get("center_x_mm", 0.0)),
            center_y_mm=float(data.get("center_y_mm", 0.0)),
            units=str(data.get("units", "waves")),
            coeffs=default_coeffs,
        )

    def has_nonzero_aberration(self) -> bool:
        if not self.enabled or not self.coeffs:
            return False
        return any(abs(float(v)) > 1e-9 for k, v in self.coeffs.items() if k != "z1_piston")


@dataclass
class PLMGratingConfig:
    """Configuration for programmable diffractive phase gratings on the TI-PLM."""
    enabled: bool = True
    grating_type: str = "Blazed"               # 'Blazed', 'Binary', '4-Level', 'Sinusoidal', 'Chirped_VLS', 'Hadamard_Spectral'
    pitch_um: float = 86.4                     # Grating period Lambda in um (e.g., 8 pixels * 10.8 um = 86.4 um -> 11.574 l/mm)
    lines_per_mm: float = 11.574074            # 1000.0 / pitch_um
    pixels_per_period: float = 8.0             # pitch_um / 10.8
    orientation_deg: float = 0.0               # 0 deg = grooves along X (disperses/steers along Y); 90 deg = steers along X
    diffraction_order: int = 1                 # Active tracked order m (typically +1 or -1 or 0)
    phase_depth_waves: float = 1.0             # Peak-to-peak phase modulation depth in units of 2*pi at design_wavelength_um
    duty_cycle: float = 0.5                    # Duty cycle for Binary grating (0.05..0.95)
    chirp_lines_per_mm2: float = 0.0           # Linear frequency chirp d(lines_per_mm)/du for VLS focusing/flattening
    fold_optical_axis: bool = True             # Whether sequential optical axis follows the PLM diffracted order when ON
    reference_lines_per_mm: Optional[float] = None  # Fixed mechanical arm reference frequency (if None, uses lines_per_mm)
    reference_order: int = 1                   # Fixed mechanical arm reference diffraction order

    def sync_from_pitch(self, pixel_pitch_um: float = 10.8) -> None:
        self.pitch_um = max(float(self.pitch_um), 2.0 * pixel_pitch_um)
        self.lines_per_mm = 1000.0 / self.pitch_um
        self.pixels_per_period = self.pitch_um / max(pixel_pitch_um, 1e-6)

    def sync_from_lines_per_mm(self, pixel_pitch_um: float = 10.8) -> None:
        max_lpmm = 1000.0 / (2.0 * max(pixel_pitch_um, 1e-6))
        self.lines_per_mm = max(0.0, min(float(self.lines_per_mm), max_lpmm))
        if self.lines_per_mm > 1e-6:
            self.pitch_um = 1000.0 / self.lines_per_mm
            self.pixels_per_period = self.pitch_um / max(pixel_pitch_um, 1e-6)

    def sync_from_pixels_per_period(self, pixel_pitch_um: float = 10.8) -> None:
        self.pixels_per_period = max(2.0, float(self.pixels_per_period))
        self.pitch_um = self.pixels_per_period * pixel_pitch_um
        self.lines_per_mm = 1000.0 / self.pitch_um

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "enabled": bool(self.enabled),
            "grating_type": str(self.grating_type),
            "pitch_um": float(self.pitch_um),
            "lines_per_mm": float(self.lines_per_mm),
            "pixels_per_period": float(self.pixels_per_period),
            "orientation_deg": float(self.orientation_deg),
            "diffraction_order": int(self.diffraction_order),
            "phase_depth_waves": float(self.phase_depth_waves),
            "duty_cycle": float(self.duty_cycle),
            "chirp_lines_per_mm2": float(self.chirp_lines_per_mm2),
            "fold_optical_axis": bool(self.fold_optical_axis),
            "reference_order": int(self.reference_order),
        }
        if self.reference_lines_per_mm is not None:
            d["reference_lines_per_mm"] = float(self.reference_lines_per_mm)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PLMGratingConfig":
        if not isinstance(data, dict):
            return cls()
        ref_lpmm = data.get("reference_lines_per_mm", None)
        return cls(
            enabled=bool(data.get("enabled", True)),
            grating_type=str(data.get("grating_type", "Blazed")),
            pitch_um=float(data.get("pitch_um", 86.4)),
            lines_per_mm=float(data.get("lines_per_mm", 11.574074)),
            pixels_per_period=float(data.get("pixels_per_period", 8.0)),
            orientation_deg=float(data.get("orientation_deg", 0.0)),
            diffraction_order=int(data.get("diffraction_order", 1)),
            phase_depth_waves=float(data.get("phase_depth_waves", 1.0)),
            duty_cycle=float(data.get("duty_cycle", 0.5)),
            chirp_lines_per_mm2=float(data.get("chirp_lines_per_mm2", 0.0)),
            fold_optical_axis=bool(data.get("fold_optical_axis", True)),
            reference_lines_per_mm=float(ref_lpmm) if ref_lpmm is not None else None,
            reference_order=int(data.get("reference_order", 1)),
        )


@dataclass
class PLMPatternConfig:
    """Configuration for 2D synthetic or file-loaded phase patterns on the TI-PLM."""
    enabled: bool = False
    pattern_type: str = "None"                 # 'None', 'File', 'Hadamard', 'Column_Range', 'Fresnel_Lens', 'Vortex', 'Cubic_Airy', 'Checkerboard'
    file_path: str = ""                        # Path to loaded pattern (.npy, .csv, .png, .bmp, .json)
    hadamard_order: int = 32                   # Number of spectral bands for Hadamard mask (16, 32, 64, 128)
    hadamard_row_index: int = 5                # Active Walsh-Hadamard row index (0..hadamard_order-1)
    hadamard_axis: str = "X"                   # Dispersion axis along which spectral columns are masked ('X' or 'Y')
    column_start: int = 0                      # Start column index for Column_Range slit / bandpass (0..array_cols-1)
    column_end: int = 50                       # End column index for Column_Range slit / bandpass (0..array_cols-1)
    column_invert: bool = False                # If True, selected column range is OFF and rest is ON
    pattern_angle_deg: float = 0.0             # Rotation angle theta of pattern on array (0=vertical cols, 90=horizontal rows, +/-45=diagonal)
    column_state: int = 1                      # Discrete state value (0..1 for DMD, 0..15 for PLM) for active column range
    column_background_state: int = 0           # Discrete state value (0..1 for DMD, 0..15 for PLM) for unselected columns
    column_ranges: List[Dict[str, int]] = field(default_factory=list) # Multi-range list: [{"start": c0, "end": c1, "state": s}, ...]
    fresnel_focal_length_mm: float = 100.0     # Focal length of synthetic Fresnel zone lens (mm)
    vortex_charge: int = 1                     # Topological charge l for spiral phase mask
    cubic_alpha_waves: float = 2.0             # Cubic phase mask strength in waves at edge
    checkerboard_period_pixels: int = 32       # Checkerboard cell size in PLM pixels
    phase_scale: float = 1.0                   # Multiplier on loaded pattern phase

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "pattern_type": str(self.pattern_type),
            "file_path": str(self.file_path),
            "hadamard_order": int(self.hadamard_order),
            "hadamard_row_index": int(self.hadamard_row_index),
            "hadamard_axis": str(self.hadamard_axis),
            "column_start": int(self.column_start),
            "column_end": int(self.column_end),
            "column_invert": bool(self.column_invert),
            "pattern_angle_deg": float(self.pattern_angle_deg),
            "column_state": int(self.column_state),
            "column_background_state": int(self.column_background_state),
            "column_ranges": [
                {"start": int(r.get("start", 0)), "end": int(r.get("end", 0)), "state": int(r.get("state", 1))}
                for r in (self.column_ranges or []) if isinstance(r, dict)
            ],
            "fresnel_focal_length_mm": float(self.fresnel_focal_length_mm),
            "vortex_charge": int(self.vortex_charge),
            "cubic_alpha_waves": float(self.cubic_alpha_waves),
            "checkerboard_period_pixels": int(self.checkerboard_period_pixels),
            "phase_scale": float(self.phase_scale),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PLMPatternConfig":
        if not isinstance(data, dict):
            return cls()
        raw_ranges = data.get("column_ranges", [])
        parsed_ranges: List[Dict[str, int]] = []
        if isinstance(raw_ranges, list):
            for r in raw_ranges:
                if isinstance(r, dict):
                    parsed_ranges.append({
                        "start": int(r.get("start", 0)),
                        "end": int(r.get("end", 0)),
                        "state": int(r.get("state", 1)),
                    })
        return cls(
            enabled=bool(data.get("enabled", False)),
            pattern_type=str(data.get("pattern_type", "None")),
            file_path=str(data.get("file_path", "")),
            hadamard_order=int(data.get("hadamard_order", 32)),
            hadamard_row_index=int(data.get("hadamard_row_index", 5)),
            hadamard_axis=str(data.get("hadamard_axis", "X")),
            column_start=int(data.get("column_start", 0)),
            column_end=int(data.get("column_end", 50)),
            column_invert=bool(data.get("column_invert", False)),
            pattern_angle_deg=float(data.get("pattern_angle_deg", 0.0)),
            column_state=int(data.get("column_state", 1)),
            column_background_state=int(data.get("column_background_state", 0)),
            column_ranges=parsed_ranges,
            fresnel_focal_length_mm=float(data.get("fresnel_focal_length_mm", 100.0)),
            vortex_charge=int(data.get("vortex_charge", 1)),
            cubic_alpha_waves=float(data.get("cubic_alpha_waves", 2.0)),
            checkerboard_period_pixels=int(data.get("checkerboard_period_pixels", 32)),
            phase_scale=float(data.get("phase_scale", 1.0)),
        )


@dataclass
class PLMConfig:
    """
    Complete configuration for a Texas Instruments Phase Light Modulator (TI-PLM)
    or Digital Micromirror Device (TI-DMD).
    Stored on Surface.plm_config (and OpticalComponent.plm_config).
    """
    model: str = "DLP670S"
    modulator_type: str = "PLM"                # "PLM" (4-bit phase piston) or "DMD" (1-bit binary tilt)
    array_cols: int = 1358
    array_rows: int = 800
    pixel_pitch_um: float = 10.8
    active_width_mm: float = 14.6664
    active_height_mm: float = 8.6400
    bit_depth: int = 4                         # 4-bit = 16 states (0..15) for PLM; 1-bit = 2 states for DMD
    tilt_angle_deg: float = 12.0               # Micromirror tilt angle (+/- deg) for DMD
    panel_rotation_deg: float = 0.0            # Physical mounting / clocking rotation angle around optical axis Z (deg, tilt_z)
    design_wavelength_um: float = 1.225        # Reference wavelength where state 16/16 = 2*pi
    quantize_4bit: bool = True                 # Apply discrete state quantization
    compensate_incidence_angle: bool = True    # Adjust piston scaling by 1/cos(theta_inc) at nominal tilt
    lut_fractions: List[float] = field(default_factory=lambda: [i / 16.0 for i in range(16)])
    zernike: ZernikePhaseConfig = field(default_factory=ZernikePhaseConfig)
    grating: PLMGratingConfig = field(default_factory=PLMGratingConfig)
    pattern: PLMPatternConfig = field(default_factory=PLMPatternConfig)
    # Optional runtime-cached 2D custom pattern array (not serialized in full if large; summary/file stored)
    _custom_pattern_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def is_dmd(self) -> bool:
        return str(self.modulator_type).upper() == "DMD" or int(self.bit_depth) == 1

    @property
    def num_states(self) -> int:
        if self.is_dmd:
            return 2
        return max(2, 1 << int(max(1, min(8, self.bit_depth))))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": str(self.model),
            "modulator_type": str(self.modulator_type),
            "array_cols": int(self.array_cols),
            "array_rows": int(self.array_rows),
            "pixel_pitch_um": float(self.pixel_pitch_um),
            "active_width_mm": float(self.active_width_mm),
            "active_height_mm": float(self.active_height_mm),
            "bit_depth": int(self.bit_depth),
            "tilt_angle_deg": float(self.tilt_angle_deg),
            "panel_rotation_deg": float(self.panel_rotation_deg),
            "design_wavelength_um": float(self.design_wavelength_um),
            "quantize_4bit": bool(self.quantize_4bit),
            "compensate_incidence_angle": bool(self.compensate_incidence_angle),
            "lut_fractions": [float(x) for x in (self.lut_fractions or [i / 16.0 for i in range(16)])],
            "zernike": self.zernike.to_dict(),
            "grating": self.grating.to_dict(),
            "pattern": self.pattern.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "PLMConfig":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        lut = data.get("lut_fractions")
        if not isinstance(lut, list) or len(lut) < 2:
            lut = [i / 16.0 for i in range(16)]
        model_name = str(data.get("model", "DLP670S"))
        is_dmd_model = any(model_name.startswith(p) for p in ("DLP4", "DLP65", "DLP9", "DLP20"))
        mod_type = str(data.get("modulator_type", "DMD" if is_dmd_model else "PLM"))
        tilt = float(data.get("tilt_angle_deg", 12.0))
        rot_deg = float(data.get("panel_rotation_deg", 0.0))
        bit_d = int(data.get("bit_depth", 1 if mod_type == "DMD" else 4))
        return cls(
            model=model_name,
            modulator_type=mod_type,
            array_cols=int(data.get("array_cols", 1358)),
            array_rows=int(data.get("array_rows", 800)),
            pixel_pitch_um=float(data.get("pixel_pitch_um", 10.8)),
            active_width_mm=float(data.get("active_width_mm", 14.6664)),
            active_height_mm=float(data.get("active_height_mm", 8.6400)),
            bit_depth=bit_d,
            tilt_angle_deg=tilt,
            panel_rotation_deg=rot_deg,
            design_wavelength_um=float(data.get("design_wavelength_um", 1.225)),
            quantize_4bit=bool(data.get("quantize_4bit", True)),
            compensate_incidence_angle=bool(data.get("compensate_incidence_angle", True)),
            lut_fractions=[float(x) for x in lut],
            zernike=ZernikePhaseConfig.from_dict(data.get("zernike", {})),
            grating=PLMGratingConfig.from_dict(data.get("grating", {})),
            pattern=PLMPatternConfig.from_dict(data.get("pattern", {})),
        )


def is_dmd(surface: Any) -> bool:
    """Return True when surface represents a Texas Instruments Digital Micromirror Device (DMD)."""
    if surface is None:
        return False
    if bool(getattr(surface, "is_air_space", False)):
        return False
    st = str(getattr(surface, "surface_type", "") or "")
    if st in ("Air Space", "Object", "Detector"):
        return False
    name = str(getattr(surface, "name", "") or "")
    if name.startswith("Air_after"):
        return False
    if bool(getattr(surface, "is_dmd", False)):
        return True
    cfg = getattr(surface, "plm_config", None)
    if isinstance(cfg, PLMConfig) and cfg.is_dmd:
        return True
    if isinstance(cfg, dict) and cfg.get("modulator_type") == "DMD":
        return True
    name_up = name.upper()
    mech_up = str(getattr(surface, "mechanism_type", "") or "").upper()
    if mech_up in ("DMD", "DIGITAL MICROMIRROR DEVICE"):
        return True
    if "DMD" in name_up or any(tag in name_up for tag in ("DLP4500", "DLP6500", "DLP9000", "DLP2010")):
        return True
    metadata = getattr(surface, "catalog_metadata", None)
    if isinstance(metadata, dict):
        pn = str(metadata.get("part_number", "")).upper()
        if pn.startswith("DLP") and "670" not in pn and "6750" not in pn and "PLM" not in pn:
            return True
    return False


def is_plm(surface: Any) -> bool:
    """Return True when surface represents a Texas Instruments Phase Light Modulator (PLM)."""
    if surface is None:
        return False
    if is_dmd(surface):
        return False
    st = str(getattr(surface, "surface_type", "") or "")
    if st in ("Air Space", "Object", "Detector"):
        return False
    name = str(getattr(surface, "name", "") or "")
    if name.startswith("Air_after"):
        return False
    if bool(getattr(surface, "is_air_space", False)):
        return False
    if bool(getattr(surface, "is_plm", False)) or bool(getattr(surface, "plm_enabled", False)):
        return True
    name_up = name.upper()
    group_up = str(getattr(surface, "group_name", "") or "").upper()
    mech_up = str(getattr(surface, "mechanism_type", "") or "").upper()
    if any(tag in name_up for tag in ("PLM", "DLP670S", "DLP6750")):
        return True
    if any(tag in group_up for tag in ("PLM", "DLP670S", "DLP6750")):
        return True
    if mech_up in ("PLM", "TI_PLM", "PHASE LIGHT MODULATOR", "DLP670S"):
        return True
    cfg = getattr(surface, "plm_config", None)
    if cfg is not None:
        if isinstance(cfg, PLMConfig) and not cfg.is_dmd:
            return True
        if isinstance(cfg, dict) and cfg.get("modulator_type") != "DMD":
            return True
    return False


def ensure_plm_config(surface: Any) -> PLMConfig:
    """Retrieve or initialize a PLMConfig instance on the given surface."""
    raw = getattr(surface, "plm_config", None)
    if isinstance(raw, PLMConfig):
        if surface is not None and hasattr(surface, "tilt_z"):
            raw.panel_rotation_deg = float(getattr(surface, "tilt_z", 0.0))
        return raw
    if isinstance(raw, dict):
        cfg = PLMConfig.from_dict(raw)
        if surface is not None and hasattr(surface, "tilt_z"):
            cfg.panel_rotation_deg = float(getattr(surface, "tilt_z", 0.0))
        surface.plm_config = cfg
        return cfg
    wl = float(getattr(surface, "grating_primary_wavelength_um", None) or 1.225)

    name_up = str(getattr(surface, "name", "") or "").upper()
    mech_up = str(getattr(surface, "mechanism_type", "") or "").upper()
    is_dmd_surf = "DMD" in name_up or mech_up == "DMD"

    if is_dmd_surf:
        specs = TI_DMD_MODELS["DLP4500"]
        cfg = PLMConfig(
            model="DLP4500",
            modulator_type="DMD",
            array_cols=specs["array_cols"],
            array_rows=specs["array_rows"],
            pixel_pitch_um=specs["pixel_pitch_um"],
            active_width_mm=specs["active_width_mm"],
            active_height_mm=specs["active_height_mm"],
            bit_depth=1,
            design_wavelength_um=wl,
            tilt_angle_deg=specs.get("tilt_angle_deg", 12.0),
            panel_rotation_deg=float(getattr(surface, "tilt_z", 0.0)) if surface is not None else 0.0,
        )
        cfg.grating.enabled = False
    else:
        cfg = PLMConfig(
            design_wavelength_um=wl,
            panel_rotation_deg=float(getattr(surface, "tilt_z", 0.0)) if surface is not None else 0.0,
        )
    if surface is not None:
        surface.plm_config = cfg
    return cfg


# ============================================================================
# Zernike Polynomial Evaluation (2D Grid + Exact Point & Analytical Gradient)
# ============================================================================

def evaluate_zernike_waves_and_grad_xy(
    x_mm: Union[float, np.ndarray],
    y_mm: Union[float, np.ndarray],
    zcfg: ZernikePhaseConfig,
    design_wavelength_um: float = 1.225,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Evaluate Zernike wavefront phase W(x, y) in waves (multiples of 2*pi) AND its exact
    spatial derivatives (dW/dx, dW/dy) in waves/mm at coordinates (x_mm, y_mm).

    Using Cartesian analytic forms of Noll Z1..Z15 avoids polar singularities at r=0!
    """
    if not zcfg.enabled or not zcfg.coeffs:
        zero = np.zeros_like(x_mm, dtype=float) if isinstance(x_mm, np.ndarray) else 0.0
        return zero, zero, zero

    R = max(float(zcfg.normalization_radius_mm), 1e-6)
    inv_R = 1.0 / R
    u = (x_mm - float(zcfg.center_x_mm)) * inv_R
    v = (y_mm - float(zcfg.center_y_mm)) * inv_R
    r2 = u * u + v * v
    r4 = r2 * r2
    r6 = r4 * r2

    c = zcfg.coeffs
    scale = (1.0 / (2.0 * math.pi)) if str(zcfg.units).lower().startswith("rad") else 1.0

    W = np.zeros_like(u, dtype=float) if isinstance(u, np.ndarray) else 0.0
    dW_du = np.zeros_like(u, dtype=float) if isinstance(u, np.ndarray) else 0.0
    dW_dv = np.zeros_like(u, dtype=float) if isinstance(u, np.ndarray) else 0.0

    # Z1: Piston = 1
    c1 = float(c.get("z1_piston", 0.0)) * scale
    if abs(c1) > 1e-12:
        W = W + c1

    # Z2: Tip X = 2 * u
    c2 = float(c.get("z2_tilt_x", 0.0)) * scale
    if abs(c2) > 1e-12:
        W = W + c2 * (2.0 * u)
        dW_du = dW_du + c2 * 2.0

    # Z3: Tilt Y = 2 * v
    c3 = float(c.get("z3_tilt_y", 0.0)) * scale
    if abs(c3) > 1e-12:
        W = W + c3 * (2.0 * v)
        dW_dv = dW_dv + c3 * 2.0

    # Z4: Defocus = sqrt(3) * (2*(u^2 + v^2) - 1)
    c4 = float(c.get("z4_defocus", 0.0)) * scale
    if abs(c4) > 1e-12:
        s3 = math.sqrt(3.0)
        W = W + c4 * s3 * (2.0 * r2 - 1.0)
        dW_du = dW_du + c4 * s3 * (4.0 * u)
        dW_dv = dW_dv + c4 * s3 * (4.0 * v)

    # Z5: Oblique Astigmatism (45 deg) = 2 * sqrt(6) * u * v
    c5 = float(c.get("z5_astig_oblique", 0.0)) * scale
    if abs(c5) > 1e-12:
        s6_2 = 2.0 * math.sqrt(6.0)
        W = W + c5 * s6_2 * (u * v)
        dW_du = dW_du + c5 * s6_2 * v
        dW_dv = dW_dv + c5 * s6_2 * u

    # Z6: Vertical Astigmatism (0 deg) = sqrt(6) * (u^2 - v^2)
    c6 = float(c.get("z6_astig_vertical", 0.0)) * scale
    if abs(c6) > 1e-12:
        s6 = math.sqrt(6.0)
        W = W + c6 * s6 * (u * u - v * v)
        dW_du = dW_du + c6 * s6 * (2.0 * u)
        dW_dv = dW_dv - c6 * s6 * (2.0 * v)

    # Z7: Vertical Coma (Y) = sqrt(8) * v * (3*(u^2 + v^2) - 2)
    c7 = float(c.get("z7_coma_vertical", 0.0)) * scale
    if abs(c7) > 1e-12:
        s8 = math.sqrt(8.0)
        W = W + c7 * s8 * v * (3.0 * r2 - 2.0)
        dW_du = dW_du + c7 * s8 * (6.0 * u * v)
        dW_dv = dW_dv + c7 * s8 * (3.0 * u * u + 9.0 * v * v - 2.0)

    # Z8: Horizontal Coma (X) = sqrt(8) * u * (3*(u^2 + v^2) - 2)
    c8 = float(c.get("z8_coma_horizontal", 0.0)) * scale
    if abs(c8) > 1e-12:
        s8 = math.sqrt(8.0)
        W = W + c8 * s8 * u * (3.0 * r2 - 2.0)
        dW_du = dW_du + c8 * s8 * (9.0 * u * u + 3.0 * v * v - 2.0)
        dW_dv = dW_dv + c8 * s8 * (6.0 * u * v)

    # Z9: Vertical Trefoil = sqrt(8) * (3*u^2*v - v^3)
    c9 = float(c.get("z9_trefoil_vert", 0.0)) * scale
    if abs(c9) > 1e-12:
        s8 = math.sqrt(8.0)
        W = W + c9 * s8 * (3.0 * u * u * v - v * v * v)
        dW_du = dW_du + c9 * s8 * (6.0 * u * v)
        dW_dv = dW_dv + c9 * s8 * (3.0 * u * u - 3.0 * v * v)

    # Z10: Oblique Trefoil = sqrt(8) * (u^3 - 3*u*v^2)
    c10 = float(c.get("z10_trefoil_obl", 0.0)) * scale
    if abs(c10) > 1e-12:
        s8 = math.sqrt(8.0)
        W = W + c10 * s8 * (u * u * u - 3.0 * u * v * v)
        dW_du = dW_du + c10 * s8 * (3.0 * u * u - 3.0 * v * v)
        dW_dv = dW_dv - c10 * s8 * (6.0 * u * v)

    # Z11: Primary Spherical = sqrt(5) * (6*r^4 - 6*r^2 + 1)
    c11 = float(c.get("z11_spherical", 0.0)) * scale
    if abs(c11) > 1e-12:
        s5 = math.sqrt(5.0)
        W = W + c11 * s5 * (6.0 * r4 - 6.0 * r2 + 1.0)
        dW_dr2 = c11 * s5 * (24.0 * r2 - 12.0)
        dW_du = dW_du + dW_dr2 * u
        dW_dv = dW_dv + dW_dr2 * v

    # Z12: Secondary Astigmatism (0 deg) = sqrt(10) * (4*r^2 - 3) * (u^2 - v^2)
    c12 = float(c.get("z12_sec_astig_vert", 0.0)) * scale
    if abs(c12) > 1e-12:
        s10 = math.sqrt(10.0)
        u2_v2 = u * u - v * v
        W = W + c12 * s10 * (4.0 * r2 - 3.0) * u2_v2
        dW_du = dW_du + c12 * s10 * (8.0 * u * u2_v2 + (4.0 * r2 - 3.0) * 2.0 * u)
        dW_dv = dW_dv + c12 * s10 * (8.0 * v * u2_v2 - (4.0 * r2 - 3.0) * 2.0 * v)

    # Z13: Secondary Astigmatism (45 deg) = 2 * sqrt(10) * (4*r^2 - 3) * u * v
    c13 = float(c.get("z13_sec_astig_obl", 0.0)) * scale
    if abs(c13) > 1e-12:
        s10_2 = 2.0 * math.sqrt(10.0)
        W = W + c13 * s10_2 * (4.0 * r2 - 3.0) * u * v
        dW_du = dW_du + c13 * s10_2 * (8.0 * u * u * v + (4.0 * r2 - 3.0) * v)
        dW_dv = dW_dv + c13 * s10_2 * (8.0 * u * v * v + (4.0 * r2 - 3.0) * u)

    # Z14: Tetrafoil (0 deg) = sqrt(10) * (u^4 - 6*u^2*v^2 + v^4)
    c14 = float(c.get("z14_tetrafoil", 0.0)) * scale
    if abs(c14) > 1e-12:
        s10 = math.sqrt(10.0)
        u2 = u * u
        v2 = v * v
        W = W + c14 * s10 * (u2 * u2 - 6.0 * u2 * v2 + v2 * v2)
        dW_du = dW_du + c14 * s10 * (4.0 * u * u2 - 12.0 * u * v2)
        dW_dv = dW_dv + c14 * s10 * (4.0 * v * v2 - 12.0 * u2 * v)

    # Z15: Secondary Spherical = sqrt(7) * (20*r^6 - 30*r^4 + 12*r^2 - 1)
    c15 = float(c.get("z15_sec_spherical", 0.0)) * scale
    if abs(c15) > 1e-12:
        s7 = math.sqrt(7.0)
        W = W + c15 * s7 * (20.0 * r6 - 30.0 * r4 + 12.0 * r2 - 1.0)
        dW_dr2 = c15 * s7 * (120.0 * r4 - 120.0 * r2 + 24.0)
        dW_du = dW_du + dW_dr2 * u
        dW_dv = dW_dv + dW_dr2 * v

    dW_dx = dW_du * inv_R
    dW_dy = dW_dv * inv_R
    return W, dW_dx, dW_dy


# ============================================================================
# Walsh-Hadamard Generator & 2D Pattern Synthesis / File I/O
# ============================================================================

def generate_hadamard_matrix(n: int) -> np.ndarray:
    """
    Generate an N x N Sylvester / Walsh-Hadamard matrix with entries in {+1, -1},
    ordered by sequency (number of zero crossings) for spectral multiplexing.
    N is rounded up to the nearest power of 2.
    """
    p = max(1, int(math.ceil(math.log2(max(2, n)))))
    size = 1 << p
    H = np.array([[1]], dtype=int)
    while H.shape[0] < size:
        H = np.block([[H, H], [H, -H]])
    # Sort rows by sequency (number of sign changes)
    sign_changes = np.sum(np.abs(np.diff(H, axis=1)) > 0, axis=1)
    order = np.argsort(sign_changes)
    return H[order]


def load_pattern_from_file(file_path: str) -> np.ndarray:
    """
    Load a 2D pattern from disk (.npy, .npz, .csv, .txt, .json, .png, .bmp, .jpg, .tif)
    and return a 2D float array of phase values in waves [0, 1) (where 1 wave = 2*pi rad).
    If the input contains integer 4-bit states in 0..15, maps state s -> s / 16.0.
    If the input contains 8-bit image values 0..255, maps v -> (v / 256.0).
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"PLM pattern file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    arr: Optional[np.ndarray] = None

    if ext == ".npy":
        arr = np.load(file_path)
    elif ext == ".npz":
        data = np.load(file_path)
        first_key = list(data.keys())[0]
        arr = data[first_key]
    elif ext in (".csv", ".txt"):
        try:
            arr = np.loadtxt(file_path, delimiter=",")
        except Exception:
            arr = np.loadtxt(file_path)
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)
        if isinstance(jdata, dict):
            for key in ("states_4bit", "pattern", "phase_waves", "phase_rad", "data"):
                if key in jdata:
                    arr = np.array(jdata[key], dtype=float)
                    if key == "states_4bit":
                        return np.mod(arr / 16.0, 1.0)
                    elif key == "phase_rad":
                        return np.mod(arr / (2.0 * math.pi), 1.0)
                    break
        elif isinstance(jdata, list):
            arr = np.array(jdata, dtype=float)
    elif ext in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(file_path)
            if img.ndim == 3:
                # Convert RGB(A) to luminance
                img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            arr = np.asarray(img, dtype=float)
        except Exception:
            from PyQt6.QtGui import QImage
            qimg = QImage(file_path)
            if qimg.isNull():
                raise ValueError(f"Unable to decode image file: {file_path}")
            qimg = qimg.convertToFormat(QImage.Format.Format_Grayscale8)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(h * qimg.bytesPerLine())
            raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, qimg.bytesPerLine()))[:, :w]
            arr = raw.astype(float)

    if arr is None:
        raise ValueError(f"Unsupported pattern file format: {ext}")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim > 2:
        arr = arr[..., 0]

    max_val = float(np.nanmax(arr)) if arr.size > 0 else 0.0
    min_val = float(np.nanmin(arr)) if arr.size > 0 else 0.0

    # Auto-detect encoding:
    # 1. Integer 4-bit states 0..15
    if min_val >= 0.0 and max_val <= 15.0 and np.allclose(arr, np.round(arr)):
        if max_val > 1.0:
            return np.mod(arr / 16.0, 1.0)
    # 2. 8-bit image 0..255
    if max_val > 15.0:
        return np.clip(arr / 256.0, 0.0, 15.0 / 16.0)
    # 3. Radians 0..2*pi
    if max_val > 1.0001 and max_val <= 2.0 * math.pi + 0.1:
        return np.mod(arr / (2.0 * math.pi), 1.0)
    # 4. Waves [0, 1)
    return np.mod(arr, 1.0)


def save_pattern_to_file(
    file_path: str,
    states_4bit: np.ndarray,
    phase_rad: Optional[np.ndarray] = None,
    plm_config: Optional[PLMConfig] = None,
) -> str:
    """
    Export a 2D 4-bit quantized TI-PLM pattern (values in 0..15) to .npy, .csv, .json, or .png.
    """
    ext = os.path.splitext(file_path)[1].lower()
    states_uint8 = np.clip(np.asarray(states_4bit, dtype=int), 0, 15).astype(np.uint8)

    if ext == ".npy":
        np.save(file_path, states_uint8)
    elif ext in (".csv", ".txt"):
        np.savetxt(file_path, states_uint8, fmt="%d", delimiter=",")
    elif ext == ".json":
        payload = {
            "device": plm_config.model if plm_config else "DLP670S",
            "bit_depth": 4,
            "num_states": 16,
            "shape": list(states_uint8.shape),
            "design_wavelength_um": plm_config.design_wavelength_um if plm_config else 1.225,
            "plm_config": plm_config.to_dict() if plm_config else {},
            "states_4bit": states_uint8.tolist(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    elif ext in (".png", ".bmp", ".jpg"):
        import matplotlib.pyplot as plt
        # Scale 0..15 -> 0..255 grayscale or viridis colormap
        plt.imsave(file_path, states_uint8, cmap="twilight", vmin=0, vmax=15)
    else:
        np.save(file_path, states_uint8)
    return file_path


# ============================================================================
# Combined PLM Phase, 4-Bit Quantization, and Ray Deflection Evaluation
# ============================================================================

def quantize_phase_waves_4bit(
    phase_waves: Union[float, np.ndarray],
    bit_depth: int = 4,
    lut_fractions: Optional[List[float]] = None,
) -> Tuple[Union[int, np.ndarray], Union[float, np.ndarray]]:
    """
    Wrap continuous phase (in waves, where 1.0 = 2*pi rad) into [0, 1) and quantize
    to 4-bit discrete TI-PLM mirror states s in {0, 1, ..., 15}.

    Returns:
        (state_index, quantized_phase_waves)
    """
    n_levels = max(2, 1 << int(max(1, min(8, bit_depth))))
    wrapped = np.mod(phase_waves, 1.0)
    states = np.floor(wrapped * n_levels + 1e-9).astype(int)
    states = np.clip(states, 0, n_levels - 1)

    if lut_fractions and len(lut_fractions) >= n_levels:
        lut_arr = np.asarray(lut_fractions[:n_levels], dtype=float)
        q_waves = lut_arr[states]
    else:
        q_waves = states / float(n_levels)

    if np.isscalar(phase_waves):
        return int(states), float(q_waves)
    return states, q_waves


def resolve_column_state_map(
    c_indices: Union[int, np.ndarray],
    pcfg: PLMPatternConfig,
    max_state: int = 15,
) -> Union[int, np.ndarray]:
    """
    Given 0-based column indices on the modulator array, resolve the discrete
    integer state (0..max_state) for each column based on configured ranges.

    If pcfg.column_ranges has entries:
      - Starts with pcfg.column_background_state.
      - For each range {start, end, state}:
          applies state to all columns in [start, end].
      - Clamps state to [0, max_state].
    If pcfg.column_ranges is empty:
      - Uses pcfg.column_start, pcfg.column_end, pcfg.column_state, pcfg.column_background_state,
        and respects pcfg.column_invert.
    """
    bg_st = int(np.clip(int(getattr(pcfg, "column_background_state", 0)), 0, max_state))
    is_arr = isinstance(c_indices, np.ndarray)

    ranges = getattr(pcfg, "column_ranges", None)
    if ranges and len(ranges) > 0:
        if is_arr:
            out_st = np.full_like(c_indices, bg_st, dtype=int)
            for r in ranges:
                if isinstance(r, dict):
                    r_s = int(r.get("start", 0))
                    r_e = int(r.get("end", 0))
                    if r_s > r_e:
                        r_s, r_e = r_e, r_s
                    r_val = int(np.clip(int(r.get("state", 1)), 0, max_state))
                    mask = (c_indices >= r_s) & (c_indices <= r_e)
                    out_st[mask] = r_val
            return out_st
        else:
            c_val = int(c_indices)
            curr_st = bg_st
            for r in ranges:
                if isinstance(r, dict):
                    r_s = int(r.get("start", 0))
                    r_e = int(r.get("end", 0))
                    if r_s > r_e:
                        r_s, r_e = r_e, r_s
                    if r_s <= c_val <= r_e:
                        curr_st = int(np.clip(int(r.get("state", 1)), 0, max_state))
            return curr_st
    else:
        # Single range fallback
        r_s = int(getattr(pcfg, "column_start", 0))
        r_e = int(getattr(pcfg, "column_end", 50))
        if r_s > r_e:
            r_s, r_e = r_e, r_s
        act_st = int(np.clip(int(getattr(pcfg, "column_state", 1)), 0, max_state))
        invert = bool(getattr(pcfg, "column_invert", False))
        if is_arr:
            in_range = (c_indices >= r_s) & (c_indices <= r_e)
            if invert:
                return np.where(in_range, bg_st, act_st)
            else:
                return np.where(in_range, act_st, bg_st)
        else:
            in_range = (r_s <= int(c_indices) <= r_e)
            if invert:
                return bg_st if in_range else act_st
            else:
                return act_st if in_range else bg_st


def evaluate_hadamard_band_gate(
    x_mm: Union[float, np.ndarray],
    y_mm: Union[float, np.ndarray],
    plm_cfg: PLMConfig,
    complement: bool = False,
) -> Union[float, np.ndarray]:
    """
    Evaluate binary {0.0, 1.0} spectral band gate for Walsh-Hadamard compressive sensing.
    When gate == 1.0, the local PLM pixels display the active blazed diffraction grating
    (routing that wavelength column into the +1st order detector).
    When gate == 0.0, the local PLM pixels stay flat at STATE_0 (routing that wavelength
    column to the 0th-order specular dump).
    """
    pcfg = plm_cfg.pattern
    n_bands = max(2, int(pcfg.hadamard_order))
    H = generate_hadamard_matrix(n_bands)
    n_actual = H.shape[0]
    row_idx = int(pcfg.hadamard_row_index) % n_actual
    row_bits = H[row_idx]  # values in {+1, -1}
    if complement:
        row_bits = -row_bits

    half_span = (
        0.5 * float(plm_cfg.active_width_mm)
        if str(pcfg.hadamard_axis).upper() == "X"
        else 0.5 * float(plm_cfg.active_height_mm)
    )
    coord = x_mm if str(pcfg.hadamard_axis).upper() == "X" else y_mm
    norm_u = np.clip((coord + half_span) / max(2.0 * half_span, 1e-6), 0.0, 0.999999)
    band_idx = np.floor(norm_u * n_actual).astype(int)
    gate = (row_bits[band_idx] > 0).astype(float)
    if np.isscalar(coord):
        return float(gate)
    return gate


def evaluate_plm_phase_and_gradient(
    x_mm: Union[float, np.ndarray],
    y_mm: Union[float, np.ndarray],
    plm_cfg: PLMConfig,
    mechanical_state: str = "ON",
    wavelength_um: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate the complete TI-PLM 2D phase modulation at local coordinates (x_mm, y_mm):
    - Combines Zernike phase + Programmable Grating phase + Synthetic/Loaded 2D Pattern.
    - Applies 4-bit state quantization (states 0..15).
    - Computes the local transverse wavefront slope (dW/dx, dW/dy) in waves/mm for
      exact 3D ray deflection in OpticalSystem.trace_ray().

    Returns dict with:
        'continuous_waves': unquantized wrapped phase in [0, 1)
        'state_4bit': integer state(s) in 0..15
        'quantized_waves': 4-bit quantized phase in [0, 15/16]
        'phase_rad': quantized phase in radians at wavelength_um
        'piston_nm': physical micromirror piston height in nm
        'grad_x_waves_per_mm': local dW/dx (waves/mm at design_wavelength_um)
        'grad_y_waves_per_mm': local dW/dy (waves/mm at design_wavelength_um)
        'diffraction_efficiency': theoretical 4-bit diffraction efficiency factor
    """
    st_up = str(mechanical_state or "ON").strip().upper()
    wl_design = max(float(plm_cfg.design_wavelength_um), 0.1)
    wl_actual = float(wavelength_um) if (wavelength_um is not None and wavelength_um > 0) else wl_design
    chromatic_ratio = wl_design / wl_actual

    is_arr = isinstance(x_mm, np.ndarray) or isinstance(y_mm, np.ndarray)
    zero_val = np.zeros_like(x_mm, dtype=float) if is_arr else 0.0

    # 1. Check if in uniform piston state (OFF, FLAT, or STATE_0..STATE_15)
    if st_up in ("OFF", "FLAT") or st_up.startswith("STATE_"):
        s_idx = 0
        if st_up.startswith("STATE_"):
            try:
                s_idx = max(0, min(plm_cfg.num_states - 1, int(st_up.split("_")[1])))
            except Exception:
                s_idx = 0
        frac = s_idx / float(plm_cfg.num_states)
        state_out = np.full_like(x_mm, s_idx, dtype=int) if is_arr else int(s_idx)
        q_waves = np.full_like(x_mm, frac, dtype=float) if is_arr else float(frac)
        piston_nm = q_waves * (wl_design * 500.0)
        phase_rad = q_waves * 2.0 * math.pi * chromatic_ratio
        specs = TI_ALL_MODELS.get(plm_cfg.model, TI_PLM_DLP670S_SPECS)
        refl = float(specs.get("mirror_reflectivity", 0.88))
        return {
            "continuous_waves": q_waves,
            "state_4bit": state_out,
            "quantized_waves": q_waves,
            "phase_rad": phase_rad,
            "piston_nm": piston_nm,
            "grad_x_waves_per_mm": zero_val,
            "grad_y_waves_per_mm": zero_val,
            "diffraction_efficiency": refl,
            "grating_efficiency_only": 1.0,
        }

    # 1b. Check if device is in DMD Binary Mode (limited strictly to binary patterns)
    if plm_cfg.is_dmd:
        pcfg = plm_cfg.pattern
        gcfg = plm_cfg.grating
        hadamard_gate = 1.0
        if st_up in ("HADAMARD_A", "HADAMARD_B") or (
            pcfg.enabled and "HADAMARD" in str(pcfg.pattern_type).upper()
        ) or (
            gcfg.enabled and "HADAMARD" in str(gcfg.grating_type).upper()
        ):
            hadamard_gate = evaluate_hadamard_band_gate(
                x_mm, y_mm, plm_cfg, complement=(st_up == "HADAMARD_B")
            )

        # Determine 2D spatial pattern mask
        pat_mask = np.ones_like(x_mm, dtype=float) if is_arr else 1.0
        pat_ang = float(getattr(pcfg, "pattern_angle_deg", 0.0))
        if abs(pat_ang) > 1e-4:
            rad_p = math.radians(pat_ang)
            c_p, s_p = math.cos(rad_p), math.sin(rad_p)
            xp_mm = x_mm * c_p + y_mm * s_p
            yp_mm = -x_mm * s_p + y_mm * c_p
        else:
            c_p, s_p = 1.0, 0.0
            xp_mm, yp_mm = x_mm, y_mm

        if st_up in ("HADAMARD_A", "HADAMARD_B") or (pcfg.enabled and "HADAMARD" in str(pcfg.pattern_type).upper()):
            pat_mask = np.asarray(hadamard_gate, dtype=float) if is_arr else float(hadamard_gate)
        elif pcfg.enabled:
            ptype = str(pcfg.pattern_type).strip().upper()
            if "COLUMN" in ptype or ptype == "COLUMN_RANGE":
                W_tot = float(plm_cfg.active_width_mm)
                H_tot = float(plm_cfg.active_height_mm)
                half_span = (0.5 * W_tot * abs(c_p) + 0.5 * H_tot * abs(s_p)) if abs(pat_ang) > 1e-4 else (0.5 * W_tot)
                cols = max(1, int(plm_cfg.array_cols))
                if is_arr:
                    norm_x = np.clip((xp_mm + half_span) / max(2.0 * half_span, 1e-6), 0.0, 0.999999)
                    c_idx = np.floor(norm_x * cols).astype(int)
                    resolved_st = resolve_column_state_map(c_idx, pcfg, max_state=1)
                    pat_mask = (resolved_st >= 1).astype(float)
                else:
                    norm_x = min(max((float(xp_mm) + half_span) / max(2.0 * half_span, 1e-6), 0.0), 0.999999)
                    c_idx = int(math.floor(norm_x * cols))
                    resolved_st = resolve_column_state_map(c_idx, pcfg, max_state=1)
                    pat_mask = 1.0 if resolved_st >= 1 else 0.0
            elif "CHECKER" in ptype:
                period_mm = max(1, int(pcfg.checkerboard_period_pixels)) * (plm_cfg.pixel_pitch_um * 1e-3)
                if is_arr:
                    cx = np.floor(xp_mm / period_mm).astype(int)
                    cy = np.floor(yp_mm / period_mm).astype(int)
                    pat_mask = ((cx + cy) & 1).astype(float)
                else:
                    cx = int(math.floor(xp_mm / period_mm))
                    cy = int(math.floor(yp_mm / period_mm))
                    pat_mask = float((cx + cy) & 1)
            elif "SLIT" in ptype:
                slit_p = 1000.0 / max(float(gcfg.lines_per_mm if gcfg.lines_per_mm > 0 else 10.0), 1.0) * 1e-3
                if is_arr:
                    pat_mask = (np.mod(xp_mm, slit_p) < 0.5 * slit_p).astype(float)
                else:
                    pat_mask = 1.0 if (xp_mm % slit_p) < 0.5 * slit_p else 0.0
            elif ptype == "FILE" and pcfg.file_path:
                if plm_cfg._custom_pattern_cache is None and os.path.exists(pcfg.file_path):
                    try:
                        plm_cfg._custom_pattern_cache = load_pattern_from_file(pcfg.file_path)
                    except Exception:
                        plm_cfg._custom_pattern_cache = None
                if plm_cfg._custom_pattern_cache is not None:
                    pat = plm_cfg._custom_pattern_cache
                    rows, cols = pat.shape
                    hw = 0.5 * max(float(plm_cfg.active_width_mm), 1e-3)
                    hh = 0.5 * max(float(plm_cfg.active_height_mm), 1e-3)
                    if is_arr:
                        col_idx = np.clip(((x_mm + hw) / (2.0 * hw) * cols).astype(int), 0, cols - 1)
                        row_idx = np.clip(((y_mm + hh) / (2.0 * hh) * rows).astype(int), 0, rows - 1)
                        raw_val = pat[row_idx, col_idx]
                        pat_mask = (raw_val >= 0.5).astype(float)
                    else:
                        col_idx = int(np.clip(int((x_mm + hw) / (2.0 * hw) * cols), 0, cols - 1))
                        row_idx = int(np.clip(int((y_mm + hh) / (2.0 * hh) * rows), 0, rows - 1))
                        raw_val = pat[row_idx, col_idx]
                        pat_mask = 1.0 if float(raw_val) >= 0.5 else 0.0

        if gcfg.enabled and gcfg.lines_per_mm > 0:
            psi_rad = math.radians(float(gcfg.orientation_deg))
            sin_p = math.sin(psi_rad)
            cos_p = math.cos(psi_rad)
            u_mm = x_mm * sin_p + y_mm * cos_p
            f0 = float(gcfg.lines_per_mm)
            chirp = float(gcfg.chirp_lines_per_mm2)
            order_m = int(gcfg.diffraction_order)
            phase_cycles = f0 * u_mm + 0.5 * chirp * (u_mm * u_mm)
            f_inst = (f0 + chirp * u_mm) * order_m
            duty = max(0.05, min(0.95, float(gcfg.duty_cycle)))
            # Hadamard A/B encodings and Column Range masks must maintain direct mirror state control:
            # the carrier grating only steers the ON mirrors when active and must not gate the mask itself.
            hadamard_mode = st_up in ("HADAMARD_A", "HADAMARD_B") or (
                pcfg.enabled and "HADAMARD" in str(pcfg.pattern_type).upper()
            )
            col_mode = pcfg.enabled and ("COLUMN" in str(pcfg.pattern_type).upper())
            steer_only = hadamard_mode or col_mode
            if is_arr:
                bin_grat = (np.mod(phase_cycles, 1.0) < duty).astype(float)
                binary_mask = np.asarray(pat_mask, dtype=float) if steer_only else bin_grat * pat_mask
            else:
                bin_grat = 1.0 if float(phase_cycles % 1.0) < duty else 0.0
                binary_mask = float(pat_mask) if steer_only else bin_grat * float(pat_mask)
            f_eff = f_inst * binary_mask
            grad_x = f_eff * sin_p
            grad_y = f_eff * cos_p
        else:
            binary_mask = pat_mask
            grad_x = np.zeros_like(x_mm, dtype=float) if is_arr else 0.0
            grad_y = np.zeros_like(x_mm, dtype=float) if is_arr else 0.0

        if is_arr:
            state_binary = (binary_mask >= 0.5).astype(int)
            q_waves = state_binary.astype(float)
            phase_rad = state_binary.astype(float) * math.pi
            piston_nm = state_binary.astype(float) * (wl_design * 250.0)
        else:
            state_binary = 1 if float(binary_mask) >= 0.5 else 0
            q_waves = float(state_binary)
            phase_rad = float(state_binary) * math.pi
            piston_nm = float(state_binary) * (wl_design * 250.0)

        specs = TI_ALL_MODELS.get(plm_cfg.model, TI_DMD_MODELS["DLP4500"])
        refl = float(specs.get("mirror_reflectivity", 0.88))
        fill = float(specs.get("fill_factor", 0.92))
        eta = refl * fill
        if gcfg.enabled:
            eta *= (2.0 / math.pi) ** 2

        if is_arr:
            total_eff = eta * (binary_mask >= 0.5).astype(float)
        else:
            total_eff = float(eta) if float(binary_mask) >= 0.5 else 0.0

        return {
            "continuous_waves": q_waves,
            "state_4bit": state_binary,
            "quantized_waves": q_waves,
            "phase_rad": phase_rad,
            "piston_nm": piston_nm,
            "grad_x_waves_per_mm": grad_x,
            "grad_y_waves_per_mm": grad_y,
            "diffraction_efficiency": total_eff if is_arr else float(total_eff),
            "grating_efficiency_only": (2.0 / math.pi) ** 2 if gcfg.enabled else 1.0,
        }

    # 2. Active pattern mode ("ON", "HADAMARD_A", "HADAMARD_B")
    total_waves = np.zeros_like(x_mm, dtype=float) if is_arr else 0.0
    grad_x = np.zeros_like(x_mm, dtype=float) if is_arr else 0.0
    grad_y = np.zeros_like(x_mm, dtype=float) if is_arr else 0.0

    # 2a. Zernike Polynomial Phase + Analytical Gradient
    if plm_cfg.zernike.enabled:
        z_w, z_gx, z_gy = evaluate_zernike_waves_and_grad_xy(x_mm, y_mm, plm_cfg.zernike, wl_design)
        total_waves = total_waves + z_w
        grad_x = grad_x + z_gx
        grad_y = grad_y + z_gy

    # 2b. Synthetic / File-loaded 2D Pattern
    pcfg = plm_cfg.pattern
    hadamard_gate: Union[float, np.ndarray] = 1.0
    if st_up in ("HADAMARD_A", "HADAMARD_B") or (
        pcfg.enabled and pcfg.pattern_type.upper() == "HADAMARD"
    ) or (
        plm_cfg.grating.enabled and plm_cfg.grating.grating_type.upper() == "HADAMARD_SPECTRAL"
    ):
        hadamard_gate = evaluate_hadamard_band_gate(
            x_mm, y_mm, plm_cfg, complement=(st_up == "HADAMARD_B")
        )

    if pcfg.enabled:
        ptype = pcfg.pattern_type.strip().upper()
        scale_p = float(pcfg.phase_scale)
        if ptype == "FRESNEL_LENS":
            f_mm = float(pcfg.fresnel_focal_length_mm)
            if abs(f_mm) > 1e-3:
                # Parabolic thin-lens phase: W(x,y) = -(x^2 + y^2) / (2 * f * lambda_0)
                wl_mm = wl_design * 1e-3
                r2 = x_mm * x_mm + y_mm * y_mm
                w_fzp = -scale_p * r2 / (2.0 * f_mm * wl_mm)
                gx_fzp = -scale_p * x_mm / (f_mm * wl_mm)
                gy_fzp = -scale_p * y_mm / (f_mm * wl_mm)
                total_waves = total_waves + w_fzp
                grad_x = grad_x + gx_fzp
                grad_y = grad_y + gy_fzp
        elif ptype == "VORTEX":
            charge = int(pcfg.vortex_charge)
            if charge != 0:
                theta = np.arctan2(y_mm, x_mm)
                r2 = np.maximum(x_mm * x_mm + y_mm * y_mm, 1e-6)
                w_vort = scale_p * charge * (theta / (2.0 * math.pi))
                gx_vort = scale_p * charge * (-y_mm / (2.0 * math.pi * r2))
                gy_vort = scale_p * charge * (x_mm / (2.0 * math.pi * r2))
                total_waves = total_waves + w_vort
                grad_x = grad_x + gx_vort
                grad_y = grad_y + gy_vort
        elif ptype == "CUBIC_AIRY":
            alpha = float(pcfg.cubic_alpha_waves) * scale_p
            R_norm = max(float(plm_cfg.zernike.normalization_radius_mm), 1.0)
            u = x_mm / R_norm
            v = y_mm / R_norm
            total_waves = total_waves + alpha * (u * u * u + v * v * v)
            grad_x = grad_x + alpha * 3.0 * (u * u) / R_norm
            grad_y = grad_y + alpha * 3.0 * (v * v) / R_norm
        elif "COLUMN" in ptype or ptype == "COLUMN_RANGE":
            pat_ang = float(getattr(pcfg, "pattern_angle_deg", 0.0))
            if abs(pat_ang) > 1e-4:
                rad_p = math.radians(pat_ang)
                c_p, s_p = math.cos(rad_p), math.sin(rad_p)
                xp_mm = x_mm * c_p + y_mm * s_p
            else:
                c_p, s_p = 1.0, 0.0
                xp_mm = x_mm
            W_tot = float(plm_cfg.active_width_mm)
            H_tot = float(plm_cfg.active_height_mm)
            half_span = (0.5 * W_tot * abs(c_p) + 0.5 * H_tot * abs(s_p)) if abs(pat_ang) > 1e-4 else (0.5 * W_tot)
            cols = max(1, int(plm_cfg.array_cols))
            n_levels = max(2, int(plm_cfg.num_states))
            max_st = n_levels - 1
            if is_arr:
                norm_x = np.clip((xp_mm + half_span) / max(2.0 * half_span, 1e-6), 0.0, 0.999999)
                c_idx = np.floor(norm_x * cols).astype(int)
                resolved_st = resolve_column_state_map(c_idx, pcfg, max_state=max_st)
                if plm_cfg.lut_fractions and len(plm_cfg.lut_fractions) >= n_levels:
                    lut_arr = np.asarray(plm_cfg.lut_fractions[:n_levels], dtype=float)
                    st_waves = lut_arr[resolved_st]
                else:
                    st_waves = resolved_st.astype(float) / float(n_levels)
            else:
                norm_x = min(max((float(xp_mm) + half_span) / max(2.0 * half_span, 1e-6), 0.0), 0.999999)
                c_idx = int(math.floor(norm_x * cols))
                resolved_st = resolve_column_state_map(c_idx, pcfg, max_state=max_st)
                if plm_cfg.lut_fractions and len(plm_cfg.lut_fractions) >= n_levels:
                    st_waves = float(plm_cfg.lut_fractions[resolved_st])
                else:
                    st_waves = float(resolved_st) / float(n_levels)
            total_waves = total_waves + scale_p * st_waves
        elif ptype == "CHECKERBOARD":
            period_mm = max(1, int(pcfg.checkerboard_period_pixels)) * (plm_cfg.pixel_pitch_um * 1e-3)
            cx = np.floor(x_mm / period_mm).astype(int)
            cy = np.floor(y_mm / period_mm).astype(int)
            chk = ((cx + cy) & 1) * 0.5 * scale_p
            total_waves = total_waves + chk
        elif ptype == "FILE" and pcfg.file_path:
            if plm_cfg._custom_pattern_cache is None and os.path.exists(pcfg.file_path):
                try:
                    plm_cfg._custom_pattern_cache = load_pattern_from_file(pcfg.file_path)
                except Exception:
                    plm_cfg._custom_pattern_cache = None
            if plm_cfg._custom_pattern_cache is not None:
                pat = plm_cfg._custom_pattern_cache
                rows, cols = pat.shape
                hw = 0.5 * max(float(plm_cfg.active_width_mm), 1e-3)
                hh = 0.5 * max(float(plm_cfg.active_height_mm), 1e-3)
                col_idx = np.clip(((x_mm + hw) / (2.0 * hw) * cols).astype(int), 0, cols - 1)
                row_idx = np.clip(((y_mm + hh) / (2.0 * hh) * rows).astype(int), 0, rows - 1)
                total_waves = total_waves + scale_p * pat[row_idx, col_idx]

    # 2c. Programmable Phase Grating
    gcfg = plm_cfg.grating
    eta_grating = 1.0
    if gcfg.enabled and gcfg.lines_per_mm > 0:
        psi_rad = math.radians(float(gcfg.orientation_deg))
        # Dispersion / periodicity coordinate u_mm (for orientation_deg=0, grating modulates along Y, grooves along X)
        sin_p = math.sin(psi_rad)
        cos_p = math.cos(psi_rad)
        u_mm = x_mm * sin_p + y_mm * cos_p

        f0 = float(gcfg.lines_per_mm)
        chirp = float(gcfg.chirp_lines_per_mm2)
        order_m = int(gcfg.diffraction_order)
        depth = float(gcfg.phase_depth_waves)
        gtype = gcfg.grating_type.strip().upper()

        # Instantaneous spatial phase (in cycles) and spatial frequency f_inst(u) = f0 + chirp * u
        phase_cycles = f0 * u_mm + 0.5 * chirp * (u_mm * u_mm)
        f_inst = (f0 + chirp * u_mm) * order_m

        if gtype in ("BLAZED", "CHIRPED_VLS", "HADAMARD_SPECTRAL"):
            w_grat = depth * np.mod(phase_cycles * np.sign(order_m if order_m != 0 else 1), 1.0)
        elif gtype == "4-LEVEL":
            w_grat = depth * (np.floor(np.mod(phase_cycles, 1.0) * 4.0) / 4.0)
        elif gtype == "BINARY":
            duty = max(0.05, min(0.95, float(gcfg.duty_cycle)))
            w_grat = np.where(np.mod(phase_cycles, 1.0) < duty, 0.5 * depth, 0.0)
        elif gtype == "SINUSOIDAL":
            w_grat = 0.5 * depth * (1.0 + np.sin(2.0 * math.pi * phase_cycles))
        else:
            w_grat = depth * np.mod(phase_cycles, 1.0)

        # Apply Hadamard spectral mask gate if active (gated bands get the grating; blocked bands stay flat 0th order)
        w_grat = w_grat * hadamard_gate
        f_eff = f_inst * hadamard_gate

        total_waves = total_waves + w_grat
        grad_x = grad_x + f_eff * sin_p
        grad_y = grad_y + f_eff * cos_p

        # Compute theoretical diffraction efficiency for the selected grating profile & quantization
        n_levels = plm_cfg.num_states if plm_cfg.quantize_4bit else 256
        if gtype == "BINARY":
            eta_grating = (2.0 / math.pi) ** 2 * (math.sin(0.5 * math.pi * depth) ** 2) if abs(order_m) == 1 else 0.25
        elif gtype == "4-LEVEL":
            eta_grating = (math.sin(math.pi / 4.0) / (math.pi / 4.0)) ** 2
        elif gtype == "SINUSOIDAL":
            # Bessel J_1^2(pi * depth / 2) approximation
            arg = 0.5 * math.pi * depth
            j1_approx = 0.5 * arg * (1.0 - (arg * arg) / 8.0 + (arg ** 4) / 192.0)
            eta_grating = float(max(0.05, min(0.34, j1_approx * j1_approx)))
        else:
            # N-level quantized blazed staircase + pixel pitch envelope sinc^2(1/N_eff)
            ppp = max(2.0, float(gcfg.pixels_per_period))
            n_eff = min(float(n_levels), ppp)
            sinc_quant = (math.sin(math.pi / n_eff) / (math.pi / n_eff)) ** 2
            # Envelope for phase depth mismatch from 1.0 wave at actual wavelength
            depth_eff = depth * chromatic_ratio
            mismatch = abs(depth_eff - float(abs(order_m) if order_m != 0 else 1))
            sinc_depth = (math.sin(math.pi * mismatch) / (math.pi * mismatch)) ** 2 if mismatch > 1e-4 else 1.0
            eta_grating = sinc_quant * sinc_depth

    # 3. Wrap & Quantize to 4-bit (16 discrete states 0..15)
    cont_wrapped = np.mod(total_waves, 1.0)
    if plm_cfg.quantize_4bit:
        state_4bit, q_waves = quantize_phase_waves_4bit(
            cont_wrapped, bit_depth=plm_cfg.bit_depth, lut_fractions=plm_cfg.lut_fractions
        )
    else:
        state_4bit, _ = quantize_phase_waves_4bit(cont_wrapped, bit_depth=4)
        q_waves = cont_wrapped

    piston_nm = q_waves * (wl_design * 500.0)
    phase_rad = q_waves * (2.0 * math.pi) * chromatic_ratio

    refl = float(TI_PLM_MODELS.get(plm_cfg.model, TI_PLM_DLP670S_SPECS)["mirror_reflectivity"])
    fill = float(TI_PLM_MODELS.get(plm_cfg.model, TI_PLM_DLP670S_SPECS)["fill_factor"])
    total_eff = refl * fill * eta_grating

    return {
        "continuous_waves": cont_wrapped,
        "state_4bit": state_4bit,
        "quantized_waves": q_waves,
        "phase_rad": phase_rad,
        "piston_nm": piston_nm,
        "grad_x_waves_per_mm": grad_x,
        "grad_y_waves_per_mm": grad_y,
        "diffraction_efficiency": float(total_eff),
        "grating_efficiency_only": float(eta_grating),
    }


def generate_plm_2d_maps(
    plm_cfg: PLMConfig,
    mechanical_state: str = "ON",
    nx: int = 340,
    ny: int = 200,
) -> Dict[str, Any]:
    """
    Generate 2D preview grids over the active TI-PLM aperture (14.6664 mm x 8.6400 mm)
    for interactive GUI plotting, 1D cross-sections, and far-field FFT PSF / diffraction analysis.
    """
    hw = 0.5 * max(float(plm_cfg.active_width_mm), 1.0)
    hh = 0.5 * max(float(plm_cfg.active_height_mm), 1.0)
    x_1d = np.linspace(-hw, hw, nx)
    y_1d = np.linspace(-hh, hh, ny)
    X, Y = np.meshgrid(x_1d, y_1d)

    res = evaluate_plm_phase_and_gradient(X, Y, plm_cfg, mechanical_state=mechanical_state)

    # Compute 2D Far-Field PSF / Diffraction Pattern via 2D FFT of complex pupil
    pupil_phase = res["phase_rad"]
    # Apply super-Gaussian soft circular/elliptical illumination apodization
    illum_amp = np.exp(-0.5 * ((X / (0.7 * hw)) ** 2 + (Y / (0.7 * hh)) ** 2))
    if plm_cfg.is_dmd:
        pupil_trans = (res["state_4bit"] > 0).astype(float)
        field_pupil = illum_amp * pupil_trans * np.exp(1j * pupil_phase)
    else:
        field_pupil = illum_amp * np.exp(1j * pupil_phase)
    pad_y, pad_x = ny * 2, nx * 2
    fft_field = np.fft.fftshift(np.fft.fft2(field_pupil, s=(pad_y, pad_x)))
    psf_intensity = np.abs(fft_field) ** 2
    max_i = float(np.max(psf_intensity)) if psf_intensity.size > 0 else 1.0
    if max_i > 0:
        psf_intensity = psf_intensity / max_i

    # State histogram (fraction of pixels in each of the 16 4-bit states 0..15)
    states_flat = res["state_4bit"].ravel()
    hist_counts = np.bincount(np.clip(states_flat, 0, 15), minlength=16)
    hist_fracs = hist_counts / max(1, states_flat.size)

    dx_val = float(x_1d[1] - x_1d[0]) if len(x_1d) > 1 else 1.0
    dy_val = float(y_1d[1] - y_1d[0]) if len(y_1d) > 1 else 1.0
    wl_um_val = float(getattr(plm_cfg, "design_wavelength_um", 1.225) or 1.225)
    max_theta_x_mrad = float(0.5 * wl_um_val / dx_val)
    max_theta_y_mrad = float(0.5 * wl_um_val / dy_val)

    return {
        "x_mm": x_1d,
        "y_mm": y_1d,
        "X": X,
        "Y": Y,
        "state_4bit": res["state_4bit"],
        "quantized_waves": res["quantized_waves"],
        "continuous_waves": res["continuous_waves"],
        "phase_rad": res["phase_rad"],
        "piston_nm": res["piston_nm"],
        "psf_intensity": psf_intensity,
        "psf_extent_mrad": [-max_theta_x_mrad, max_theta_x_mrad, -max_theta_y_mrad, max_theta_y_mrad],
        "state_histogram": hist_fracs,
        "diffraction_efficiency": res.get("diffraction_efficiency", 1.0),
        "grating_efficiency_only": res.get("grating_efficiency_only", 1.0),
    }


def deflect_ray_on_plm_3d(
    ray_dir_in: np.ndarray,
    surf_normal: np.ndarray,
    surf_global_R: np.ndarray,
    local_x_mm: float,
    local_y_mm: float,
    plm_cfg: PLMConfig,
    mechanical_state: str = "ON",
    wavelength_um: float = 1.225,
) -> Tuple[np.ndarray, float, float, int]:
    """
    Compute the 3D reflected + phase-deflected ray direction, accumulated phase shift (rad),
    efficiency weight, and local 4-bit state index (0..15) when a ray hits a TI-PLM surface
    at local coordinates (local_x_mm, local_y_mm).

    Uses the Generalized Snell / Grating Wavevector Law in 3D:
        k_out_tangential = k_in_tangential + (lambda_mm) * (dW/dx * e_x + dW/dy * e_y)
        k_out = k_out_tangential + sqrt(1 - |k_out_tangential|^2) * n_refl
    where dW/dx and dW/dy (in waves/mm) include both the programmable PLM Grating carrier
    and the local Zernike / Fresnel / Vortex phase gradients!
    """
    d_in = np.asarray(ray_dir_in, dtype=float)
    d_norm = np.linalg.norm(d_in)
    if d_norm > 1e-12:
        d_in = d_in / d_norm

    n_vec = np.asarray(surf_normal, dtype=float)
    n_norm = np.linalg.norm(n_vec)
    if n_norm > 1e-12:
        n_vec = n_vec / n_norm

    # Ensure normal points into the reflected half-space (opposing incident ray)
    if np.dot(d_in, n_vec) > 0:
        n_refl = -n_vec
    else:
        n_refl = n_vec

    eval_res = evaluate_plm_phase_and_gradient(
        float(local_x_mm),
        float(local_y_mm),
        plm_cfg,
        mechanical_state=mechanical_state,
        wavelength_um=wavelength_um,
    )

    gx_waves_mm = float(eval_res["grad_x_waves_per_mm"])
    gy_waves_mm = float(eval_res["grad_y_waves_per_mm"])
    phase_rad = float(eval_res["phase_rad"])
    eff = float(eval_res["diffraction_efficiency"])
    state_4bit = int(eval_res["state_4bit"])

    # If DMD pixel is OFF (state_4bit == 0) while array is ON:
    # That micromirror is tilted to -12° (deflecting light -48° away from the ON channel into the dump).
    if plm_cfg.is_dmd and state_4bit == 0 and str(mechanical_state).upper() in ("ON", "HADAMARD_A", "HADAMARD_B") and plm_cfg.pattern.enabled:
        is_y_hinge = abs(float(surf_global_R[0, 2])) > abs(float(surf_global_R[1, 2]))
        rad_off = math.radians(-24.0)
        c, s = math.cos(rad_off), math.sin(rad_off)
        if is_y_hinge:
            R_off = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
        else:
            R_off = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
        n_dump = surf_global_R @ (R_off @ np.array([0.0, 0.0, 1.0]))
        if np.dot(d_in, n_dump) > 0:
            n_dump = -n_dump
        d_out = d_in - 2.0 * np.dot(d_in, n_dump) * n_dump
        d_out = d_out / max(np.linalg.norm(d_out), 1e-12)
        return d_out, phase_rad, 0.0, 0

    # Local tangent unit vectors in global 3D coordinates
    ex_global = surf_global_R[:, 0]
    ey_global = surf_global_R[:, 1]

    # Project onto surface tangent plane orthogonal to n_refl
    ex_tan = ex_global - np.dot(ex_global, n_refl) * n_refl
    ey_tan = ey_global - np.dot(ey_global, n_refl) * n_refl
    if np.linalg.norm(ex_tan) > 1e-9:
        ex_tan = ex_tan / np.linalg.norm(ex_tan)
    if np.linalg.norm(ey_tan) > 1e-9:
        ey_tan = ey_tan / np.linalg.norm(ey_tan)

    wl_mm = max(float(wavelength_um), 0.1) * 1e-3
    delta_k_tan = wl_mm * (gx_waves_mm * ex_tan + gy_waves_mm * ey_tan)

    # Specular tangential component of incident ray
    d_in_tan = d_in - np.dot(d_in, n_refl) * n_refl
    d_out_tan = d_in_tan + delta_k_tan

    tan_sq = float(np.dot(d_out_tan, d_out_tan))
    if tan_sq >= 1.0:
        # Evanescent fallback to specular reflection
        d_out = d_in - 2.0 * np.dot(d_in, n_refl) * n_refl
    else:
        d_out = d_out_tan + math.sqrt(max(0.0, 1.0 - tan_sq)) * n_refl

    d_out = d_out / max(np.linalg.norm(d_out), 1e-12)
    return d_out, phase_rad, eff, state_4bit
