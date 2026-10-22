"""
Generic Optical Components Architecture & Actuation Models.

Provides unified abstractions and configurations for:
- OpticalComponent: Base polymorphic component container grouping physical surfaces.
- GalvoScannerConfig: 1D/2D galvanometer scanner with continuous angle sweep,
  rotation axis definitions, and off-vertex pivot offset kinematics.
- BeamSplitterConfig: Plate and cube beam splitters with R/T power division,
  polarizing splitting (PBS), and dichroic wavelength cutoffs.
- PrismConfig: Wedge, equilateral, and right-angle prisms with exact multi-interface
  facet geometry, minimum deviation, and chromatic dispersion tracking.
- ActiveMechanism & MechanismChannel: Arbitrary continuous and discrete optomechanical actuators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import numpy as np


class ComponentType:
    GENERIC = "generic"
    LENS = "lens"
    MIRROR = "mirror"
    GALVO = "galvo"
    BEAMSPLITTER = "beamsplitter"
    PRISM = "prism"
    MECHANISM = "mechanism"
    AOM = "aom"
    EOM = "eom"
    PLM = "plm"


class MechanismType:
    GENERIC = "generic"
    GALVO = "galvo"
    DMD = "dmd"
    PLM = "plm"
    FILTER_WHEEL = "filter_wheel"
    PIEZO_STAGE = "piezo_stage"
    TUNABLE_LENS = "tunable_lens"
    AOM = "aom"
    EOM = "eom"


class BeamSplitterType:
    PLATE = "Plate"
    CUBE = "Cube"
    PELLICLE = "Pellicle"


class SplitterCoatingType:
    NON_POLARIZING_50_50 = "NPBS_50_50"
    POLARIZING = "PBS"
    DICHROIC = "Dichroic"
    CUSTOM = "Custom"


@dataclass
class BeamSplitterConfig:
    """
    Configuration for an optical beam splitter (plate, cube, or pellicle).
    Governs energy division (R/T ratio), polarization handling (Jones calculus), and physical geometry.
    """
    splitter_type: str = BeamSplitterType.PLATE
    coating_type: str = SplitterCoatingType.NON_POLARIZING_50_50
    split_ratio_R: float = 0.50                # Power reflectance (0.0 to 1.0)
    split_ratio_T: float = 0.50                # Power transmittance (0.0 to 1.0)
    substrate_glass: str = "N-BK7"
    substrate_thickness_mm: float = 3.0        # Plate thickness or cube half-width
    clear_aperture_mm: float = 25.4
    dichroic_cutoff_wavelength_um: Optional[float] = None
    dichroic_transition_width_um: float = 0.02
    cube_side_length_mm: float = 20.0
    orientation_angle_deg: float = 45.0
    is_polarizing: bool = False
    Rs: float = 0.50
    Rp: float = 0.50
    Ts: float = 0.50
    Tp: float = 0.50
    extinction_ratio: float = 1000.0
    part_number: str = ""

    def evaluate_split(
        self,
        wavelength_um: float = 0.532,
        polarization: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Evaluate power reflectance R and transmittance T for given wavelength and polarization.
        Guarantees strict energy conservation: R + T <= 1.0.
        """
        if self.coating_type == SplitterCoatingType.DICHROIC and self.dichroic_cutoff_wavelength_um is not None:
            # Sigmoid transition centered at cutoff wavelength: short wavelengths reflected, long transmitted
            cut = float(self.dichroic_cutoff_wavelength_um)
            w = max(1e-4, float(self.dichroic_transition_width_um))
            arg = np.clip((wavelength_um - cut) / w, -50.0, 50.0)
            R = float(1.0 / (1.0 + np.exp(arg)))
            T = max(0.0, 1.0 - R)
            return (R, T)

        if self.coating_type == SplitterCoatingType.POLARIZING or self.is_polarizing:
            if polarization == "s":
                R_val, T_val = float(self.Rs), float(self.Ts)
            elif polarization == "p":
                R_val, T_val = float(self.Rp), float(self.Tp)
            else:
                R_val = 0.5 * (float(self.Rs) + float(self.Rp))
                T_val = 0.5 * (float(self.Ts) + float(self.Tp))
            total = R_val + T_val
            if total > 1.0:
                R_val /= total
                T_val /= total
            return (R_val, T_val)

        R = max(0.0, min(1.0, float(self.split_ratio_R)))
        T = max(0.0, min(1.0, float(self.split_ratio_T)))
        total = R + T
        if total > 1.0:
            R /= total
            T /= total
        return (R, T)

    def evaluate_jones(
        self,
        jones_in: Optional[np.ndarray],
        wavelength_um: float = 0.532,
        intensity_in: float = 1.0,
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """
        Evaluate output transmitted and reflected Jones vectors and power intensities
        for an incoming Jones vector [E_s, E_p] expressed in the local interface basis.
        Returns:
            (jones_trans, intensity_trans, jones_refl, intensity_refl)
        Strictly satisfies energy conservation: intensity_trans + intensity_refl <= intensity_in.
        """
        if intensity_in <= 1e-18:
            z = np.array([0.0 + 0j, 0.0 + 0j], dtype=complex)
            return z, 0.0, z, 0.0

        if jones_in is None or len(jones_in) < 2:
            amp = math.sqrt(max(0.0, intensity_in) * 0.5)
            Es, Ep = complex(amp, 0.0), complex(amp, 0.0)
        else:
            j_arr = np.asarray(jones_in, dtype=complex)
            raw_mag2 = float(abs(j_arr[0])**2 + abs(j_arr[1])**2)
            if raw_mag2 > 1e-15:
                scale = math.sqrt(max(0.0, intensity_in) / raw_mag2)
                Es, Ep = complex(j_arr[0] * scale), complex(j_arr[1] * scale)
            else:
                amp = math.sqrt(max(0.0, intensity_in) * 0.5)
                Es, Ep = complex(amp, 0.0), complex(amp, 0.0)

        if self.coating_type == SplitterCoatingType.DICHROIC and self.dichroic_cutoff_wavelength_um is not None:
            R_frac, T_frac = self.evaluate_split(wavelength_um)
            j_trans = np.array([math.sqrt(T_frac) * Es, math.sqrt(T_frac) * Ep], dtype=complex)
            j_refl = np.array([math.sqrt(R_frac) * Es, -math.sqrt(R_frac) * Ep], dtype=complex)
            I_trans = float(abs(j_trans[0])**2 + abs(j_trans[1])**2)
            I_refl = float(abs(j_refl[0])**2 + abs(j_refl[1])**2)
            return j_trans, I_trans, j_refl, I_refl

        if self.coating_type == SplitterCoatingType.POLARIZING or self.is_polarizing:
            rs_val = float(np.clip(self.Rs, 0.0, 1.0))
            rp_val = float(np.clip(self.Rp, 0.0, 1.0))
            ts_val = float(np.clip(self.Ts, 0.0, 1.0))
            tp_val = float(np.clip(self.Tp, 0.0, 1.0))

            if rs_val + ts_val > 1.0:
                tot_s = rs_val + ts_val
                rs_val /= tot_s
                ts_val /= tot_s
            if rp_val + tp_val > 1.0:
                tot_p = rp_val + tp_val
                rp_val /= tot_p
                tp_val /= tot_p

            j_trans = np.array([math.sqrt(ts_val) * Es, math.sqrt(tp_val) * Ep], dtype=complex)
            j_refl = np.array([math.sqrt(rs_val) * Es, -math.sqrt(rp_val) * Ep], dtype=complex)
            I_trans = float(abs(j_trans[0])**2 + abs(j_trans[1])**2)
            I_refl = float(abs(j_refl[0])**2 + abs(j_refl[1])**2)
            return j_trans, I_trans, j_refl, I_refl

        # Non-polarizing / amplitude division
        R_frac, T_frac = self.evaluate_split(wavelength_um)
        j_trans = np.array([math.sqrt(T_frac) * Es, math.sqrt(T_frac) * Ep], dtype=complex)
        j_refl = np.array([math.sqrt(R_frac) * Es, -math.sqrt(R_frac) * Ep], dtype=complex)
        I_trans = float(abs(j_trans[0])**2 + abs(j_trans[1])**2)
        I_refl = float(abs(j_refl[0])**2 + abs(j_refl[1])**2)
        return j_trans, I_trans, j_refl, I_refl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "splitter_type": self.splitter_type,
            "coating_type": self.coating_type,
            "split_ratio_R": float(self.split_ratio_R),
            "split_ratio_T": float(self.split_ratio_T),
            "substrate_glass": self.substrate_glass,
            "substrate_thickness_mm": float(self.substrate_thickness_mm),
            "clear_aperture_mm": float(self.clear_aperture_mm),
            "dichroic_cutoff_wavelength_um": self.dichroic_cutoff_wavelength_um,
            "dichroic_transition_width_um": float(self.dichroic_transition_width_um),
            "cube_side_length_mm": float(self.cube_side_length_mm),
            "orientation_angle_deg": float(self.orientation_angle_deg),
            "is_polarizing": bool(self.is_polarizing),
            "Rs": float(self.Rs),
            "Rp": float(self.Rp),
            "Ts": float(self.Ts),
            "Tp": float(self.Tp),
            "extinction_ratio": float(self.extinction_ratio),
            "part_number": str(self.part_number),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeamSplitterConfig":
        clean = {k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}
        return cls(**clean)


class PrismGeometryType:
    EQUILATERAL = "Equilateral"        # Apex angle alpha = 60 deg, high dispersion
    RIGHT_ANGLE = "Right_Angle"        # Apex angle alpha = 90 deg (45-90-45 TIR fold)
    WEDGE = "Wedge"                    # Small angle beam steering (alpha <= 10 deg)
    DOVE = "Dove"                      # Image rotator
    PELLIN_BROCA = "Pellin_Broca"      # Constant 90 deg deviation dispersing prism


@dataclass
class PrismConfig:
    """
    Configuration for an optical dispersing or steering prism.
    Defines apex geometry, dimensions, material glass, and exact refraction/dispersion.
    """
    geometry_type: str = PrismGeometryType.EQUILATERAL
    apex_angle_deg: float = 60.0               # Apex angle alpha (deg)
    base_width_mm: float = 25.0
    height_mm: float = 25.0
    length_mm: float = 25.0                    # Extrusion width (Y or transversal)
    glass: str = "N-SF11"
    orientation_angle_deg: float = 0.0

    def _get_refractive_index(self, wavelength_um: float) -> float:
        try:
            from optics_app.utils.materials import get_refractive_index
            return float(get_refractive_index(self.glass, wavelength_um))
        except Exception:
            try:
                from v2.engine.materials import get_refractive_index
                return float(get_refractive_index(self.glass, wavelength_um))
            except Exception:
                # Fallback estimate for standard flint/crown glass
                if "SF11" in self.glass.upper():
                    return float(1.78 + 0.02 / (wavelength_um**2))
                return float(1.51 + 0.01 / (wavelength_um**2))

    def calculate_deviation(self, incident_angle_deg: float, wavelength_um: float = 0.532) -> float:
        """
        Calculate total angular deviation delta (deg) through the prism:
        sin(theta1) = n * sin(theta1') => theta1' = arcsin(sin(theta1) / n)
        theta2 = alpha - theta1'
        sin(theta2') = n * sin(theta2) => theta2' = arcsin(n * sin(theta2))
        delta = theta1 + theta2' - alpha
        """
        n = self._get_refractive_index(wavelength_um)
        alpha_rad = math.radians(self.apex_angle_deg)
        th1_rad = math.radians(incident_angle_deg)

        sin_th1_prime = math.sin(th1_rad) / n
        if abs(sin_th1_prime) > 1.0:
            raise ValueError(f"Total internal reflection at prism entrance face for {incident_angle_deg}°")
        th1_prime_rad = math.asin(sin_th1_prime)

        th2_rad = alpha_rad - th1_prime_rad
        sin_th2_prime = n * math.sin(th2_rad)
        if abs(sin_th2_prime) > 1.0:
            raise ValueError("Total internal reflection (TIR) at prism exit facet")
        th2_prime_rad = math.asin(sin_th2_prime)

        delta_rad = th1_rad + th2_prime_rad - alpha_rad
        return math.degrees(delta_rad)

    def calculate_minimum_deviation(self, wavelength_um: float = 0.532) -> float:
        """
        Calculate minimum deviation angle delta_min (deg):
        delta_min = 2 * arcsin(n * sin(alpha / 2)) - alpha
        """
        n = self._get_refractive_index(wavelength_um)
        alpha_rad = math.radians(self.apex_angle_deg)
        arg = n * math.sin(alpha_rad / 2.0)
        if abs(arg) > 1.0:
            raise ValueError(f"Apex angle {self.apex_angle_deg}° too large for index {n:.4f} without TIR")
        delta_min_rad = 2.0 * math.asin(arg) - alpha_rad
        return math.degrees(delta_min_rad)

    def calculate_dispersion(self, wavelengths_um: Sequence[float]) -> Dict[float, float]:
        """
        Calculate minimum deviation angle for each specified wavelength.
        """
        res = {}
        for wl in wavelengths_um:
            res[float(wl)] = self.calculate_minimum_deviation(float(wl))
        return res

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometry_type": self.geometry_type,
            "apex_angle_deg": float(self.apex_angle_deg),
            "base_width_mm": float(self.base_width_mm),
            "height_mm": float(self.height_mm),
            "length_mm": float(self.length_mm),
            "glass": self.glass,
            "orientation_angle_deg": float(self.orientation_angle_deg),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrismConfig":
        clean = {k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}
        return cls(**clean)


@dataclass
class GalvoScannerConfig:
    """
    Configuration for a Galvanometer Scanning Mirror actuator.
    Supports continuous angle sweep, rotation axis definition,
    and 3D off-vertex pivot offset kinematics d_pivot = [x_p, y_p, z_p].
    """
    scan_axis: str = "X"                       # "X", "Y", or "Z"
    min_angle_deg: float = -10.0               # Mechanical minimum scan angle (deg)
    max_angle_deg: float = 10.0                # Mechanical maximum scan angle (deg)
    step_size_deg: float = 1.0                 # Scan step size (deg)
    current_angle_deg: float = 0.0             # Live mechanical angle (deg)
    optical_deflection_factor: float = 2.0     # 2.0 for specular reflection
    pivot_offset_mm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # [dx, dy, dz] in mount coordinates
    mirror_thickness_mm: float = 2.0
    clear_aperture_xy_mm: Tuple[float, float] = (10.0, 15.0) # Clear aperture dimensions (width, height)
    max_voltage_v: float = 10.0                # Command scaling (+/-10V)
    volts_to_deg: float = 1.0                  # 1 V = 1 deg mechanical
    surface_index: Optional[int] = None
    surface_uid: str = ""

    @property
    def current_optical_angle_deg(self) -> float:
        return float(self.current_angle_deg * self.optical_deflection_factor)

    @property
    def optical_min_deg(self) -> float:
        return float(self.min_angle_deg * self.optical_deflection_factor)

    @property
    def optical_max_deg(self) -> float:
        return float(self.max_angle_deg * self.optical_deflection_factor)

    def set_mechanical_angle(self, angle_deg: float) -> None:
        """Set current mechanical scan angle, clamped to [min_angle_deg, max_angle_deg]."""
        self.current_angle_deg = float(np.clip(angle_deg, self.min_angle_deg, self.max_angle_deg))

    def set_voltage(self, voltage_v: float) -> None:
        """Drive galvo angle via voltage command."""
        v = float(np.clip(voltage_v, -self.max_voltage_v, self.max_voltage_v))
        self.set_mechanical_angle(v * self.volts_to_deg)

    def get_mechanical_angles(self, num_steps: Optional[int] = None) -> np.ndarray:
        """Generate discrete mechanical scan angle array."""
        if num_steps is not None and num_steps > 1:
            return np.linspace(self.min_angle_deg, self.max_angle_deg, num_steps, dtype=float)
        span = self.max_angle_deg - self.min_angle_deg
        step = max(0.01, abs(self.step_size_deg))
        n = max(3, int(round(span / step)) + 1)
        return np.linspace(self.min_angle_deg, self.max_angle_deg, n, dtype=float)

    def get_optical_angles(self, num_steps: Optional[int] = None) -> np.ndarray:
        """Generate discrete optical scan angle array."""
        return self.get_mechanical_angles(num_steps) * self.optical_deflection_factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_axis": self.scan_axis,
            "min_angle_deg": float(self.min_angle_deg),
            "max_angle_deg": float(self.max_angle_deg),
            "step_size_deg": float(self.step_size_deg),
            "current_angle_deg": float(self.current_angle_deg),
            "optical_deflection_factor": float(self.optical_deflection_factor),
            "pivot_offset_mm": [float(x) for x in self.pivot_offset_mm],
            "mirror_thickness_mm": float(self.mirror_thickness_mm),
            "clear_aperture_xy_mm": [float(x) for x in self.clear_aperture_xy_mm],
            "max_voltage_v": float(self.max_voltage_v),
            "volts_to_deg": float(self.volts_to_deg),
            "surface_index": self.surface_index,
            "surface_uid": self.surface_uid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GalvoScannerConfig":
        clean = {k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}
        if "clear_aperture_xy_mm" in clean and isinstance(clean["clear_aperture_xy_mm"], (list, tuple)):
            clean["clear_aperture_xy_mm"] = tuple(float(x) for x in clean["clear_aperture_xy_mm"][:2])
        if "pivot_offset_mm" in clean and isinstance(clean["pivot_offset_mm"], (list, tuple)):
            clean["pivot_offset_mm"] = [float(x) for x in clean["pivot_offset_mm"][:3]]
        return cls(**clean)


@dataclass
class AOMConfig:
    """
    Configuration for an Acousto-Optic Modulator (AOM) device.
    Governs acoustic material, RF driving parameters, active aperture,
    Bragg diffraction angles, and diffraction efficiency.
    """
    material: str = "TeO2_slow"
    rf_frequency_mhz: float = 80.0
    rf_power_w: float = 1.0
    active_aperture_mm: float = 2.0
    interaction_length_mm: float = 10.0
    bragg_angle_mrad: float = 0.0
    diffraction_efficiency: float = 0.0

    def copy(self) -> "AOMConfig":
        """Return an independent copy of this configuration."""
        import copy
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "material": str(self.material),
            "rf_frequency_mhz": float(self.rf_frequency_mhz),
            "rf_power_w": float(self.rf_power_w),
            "active_aperture_mm": float(self.active_aperture_mm),
            "interaction_length_mm": float(self.interaction_length_mm),
            "bragg_angle_mrad": float(self.bragg_angle_mrad),
            "diffraction_efficiency": float(self.diffraction_efficiency),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AOMConfig":
        """Deserialize configuration from a dictionary."""
        clean = {k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}
        return cls(
            material=str(clean.get("material", "TeO2_slow")),
            rf_frequency_mhz=float(clean.get("rf_frequency_mhz", 80.0)),
            rf_power_w=float(clean.get("rf_power_w", 1.0)),
            active_aperture_mm=float(clean.get("active_aperture_mm", 2.0)),
            interaction_length_mm=float(clean.get("interaction_length_mm", 10.0)),
            bragg_angle_mrad=float(clean.get("bragg_angle_mrad", 0.0)),
            diffraction_efficiency=float(clean.get("diffraction_efficiency", 0.0)),
        )


@dataclass
class EOMConfig:
    """
    Configuration for an Electro-Optic Modulator (EOM) / Pockels cell device.
    Governs electro-optic crystal, half-wave voltage (V_pi), applied/bias voltages,
    modulation frequency for lock-in detection, extinction ratio, and optical transmission.
    """
    crystal: str = "LiNbO3"
    v_pi_volts: float = 200.0
    applied_voltage_volts: float = 0.0
    bias_voltage_volts: float = 0.0
    modulation_freq_mhz: float = 10.0
    extinction_ratio: float = 1000.0
    transmission: float = 0.0

    def copy(self) -> "EOMConfig":
        """Return an independent copy of this configuration."""
        import copy
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "crystal": str(self.crystal),
            "v_pi_volts": float(self.v_pi_volts),
            "applied_voltage_volts": float(self.applied_voltage_volts),
            "bias_voltage_volts": float(self.bias_voltage_volts),
            "modulation_freq_mhz": float(self.modulation_freq_mhz),
            "extinction_ratio": float(self.extinction_ratio),
            "transmission": float(self.transmission),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EOMConfig":
        """Deserialize configuration from a dictionary."""
        clean = {k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}
        return cls(
            crystal=str(clean.get("crystal", "LiNbO3")),
            v_pi_volts=float(clean.get("v_pi_volts", 200.0)),
            applied_voltage_volts=float(clean.get("applied_voltage_volts", 0.0)),
            bias_voltage_volts=float(clean.get("bias_voltage_volts", 0.0)),
            modulation_freq_mhz=float(clean.get("modulation_freq_mhz", 10.0)),
            extinction_ratio=float(clean.get("extinction_ratio", 1000.0)),
            transmission=float(clean.get("transmission", 0.0)),
        )


class ActuationType:
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass
class MechanismChannel:
    """A single controllable degree of freedom in an active mechanism."""
    channel_name: str
    target_surface_uid: str = ""
    target_property: str = "tilt_x"            # "tilt_x", "tilt_y", "tilt_z", "decenter_x", etc.
    unit: str = "deg"                          # "deg", "mm", "V", "um"
    min_value: float = -10.0
    max_value: float = 10.0
    current_value: float = 0.0
    scale_factor: float = 1.0
    offset: float = 0.0

    def set_value(self, val: float) -> float:
        self.current_value = float(np.clip(val, self.min_value, self.max_value))
        return self.effective_property_value()

    def effective_property_value(self) -> float:
        return float(self.current_value * self.scale_factor + self.offset)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_name": self.channel_name,
            "target_surface_uid": self.target_surface_uid,
            "target_property": self.target_property,
            "unit": self.unit,
            "min_value": float(self.min_value),
            "max_value": float(self.max_value),
            "current_value": float(self.current_value),
            "scale_factor": float(self.scale_factor),
            "offset": float(self.offset),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanismChannel":
        clean = {k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}
        return cls(**clean)


@dataclass
class ActiveMechanism:
    """
    Generalized mechanism model for active actuators.
    Supports continuous parameter actuation (e.g. galvo, piezo stage, tunable lens)
    and discrete named states (e.g. DMD ON/OFF/FLAT, filter wheels, flip mounts).
    """
    mechanism_id: str = ""
    mechanism_type: str = "generic"            # "galvo", "dmd", "filter_wheel", "piezo_stage", "tunable_lens"
    actuation_type: str = ActuationType.CONTINUOUS
    channels: Dict[str, MechanismChannel] = field(default_factory=dict)
    discrete_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_state_name: str = ""

    def set_channel_value(self, channel_name: str, value: float) -> float:
        if channel_name not in self.channels:
            raise KeyError(f"Channel '{channel_name}' not found in mechanism '{self.mechanism_id}'")
        return self.channels[channel_name].set_value(value)

    def set_state(self, state_name: str) -> Dict[str, Any]:
        st = str(state_name).upper()
        if st not in self.discrete_states:
            raise ValueError(f"State '{state_name}' not found in discrete states {list(self.discrete_states.keys())}")
        self.current_state_name = st
        return self.discrete_states[st]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "mechanism_type": self.mechanism_type,
            "actuation_type": self.actuation_type,
            "channels": {k: v.to_dict() for k, v in self.channels.items()},
            "discrete_states": self.discrete_states,
            "current_state_name": self.current_state_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActiveMechanism":
        channels = {}
        for k, cd in data.get("channels", {}).items():
            channels[k] = MechanismChannel.from_dict(cd)
        return cls(
            mechanism_id=str(data.get("mechanism_id", "")),
            mechanism_type=str(data.get("mechanism_type", "generic")),
            actuation_type=str(data.get("actuation_type", ActuationType.CONTINUOUS)),
            channels=channels,
            discrete_states=dict(data.get("discrete_states", {})),
            current_state_name=str(data.get("current_state_name", "")),
        )


@dataclass
class OpticalComponent:
    """
    Generic polymorphic optical component container.
    Groups one or more physical surfaces linked by component_uid and
    retains dedicated hardware configurations (galvo, beam splitter, prism, active mechanism).
    """
    component_uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Optical Component"
    component_type: str = "generic"            # "lens", "mirror", "galvo", "beamsplitter", "prism", "mechanism", "plm"
    surface_uids: List[str] = field(default_factory=list)
    is_active: bool = True
    is_locked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    mechanism: Optional[ActiveMechanism] = None
    galvo_config: Optional[GalvoScannerConfig] = None
    beamsplitter_config: Optional[BeamSplitterConfig] = None
    prism_config: Optional[PrismConfig] = None
    aom_config: Optional[AOMConfig] = None
    eom_config: Optional[EOMConfig] = None
    plm_config: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "component_uid": self.component_uid,
            "name": self.name,
            "component_type": self.component_type,
            "surface_uids": list(self.surface_uids),
            "is_active": bool(self.is_active),
            "is_locked": bool(self.is_locked),
            "metadata": dict(self.metadata),
        }
        if self.mechanism is not None:
            d["mechanism"] = self.mechanism.to_dict()
        if self.galvo_config is not None:
            d["galvo_config"] = self.galvo_config.to_dict()
        if self.beamsplitter_config is not None:
            d["beamsplitter_config"] = self.beamsplitter_config.to_dict()
        if self.prism_config is not None:
            d["prism_config"] = self.prism_config.to_dict()
        if self.aom_config is not None:
            d["aom_config"] = self.aom_config.to_dict()
        if self.eom_config is not None:
            d["eom_config"] = self.eom_config.to_dict()
        if self.plm_config is not None:
            d["plm_config"] = self.plm_config.to_dict() if hasattr(self.plm_config, "to_dict") else self.plm_config
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpticalComponent":
        mech = ActiveMechanism.from_dict(data["mechanism"]) if "mechanism" in data and data["mechanism"] else None
        galvo = GalvoScannerConfig.from_dict(data["galvo_config"]) if "galvo_config" in data and data["galvo_config"] else None
        bs = BeamSplitterConfig.from_dict(data["beamsplitter_config"]) if "beamsplitter_config" in data and data["beamsplitter_config"] else None
        prism = PrismConfig.from_dict(data["prism_config"]) if "prism_config" in data and data["prism_config"] else None
        aom = AOMConfig.from_dict(data["aom_config"]) if "aom_config" in data and data["aom_config"] else None
        eom = EOMConfig.from_dict(data["eom_config"]) if "eom_config" in data and data["eom_config"] else None
        plm = None
        if "plm_config" in data and data["plm_config"]:
            try:
                from optics_app.engine.plm import PLMConfig
                plm = PLMConfig.from_dict(data["plm_config"])
            except Exception:
                try:
                    from v2.engine.plm import PLMConfig
                    plm = PLMConfig.from_dict(data["plm_config"])
                except Exception:
                    plm = data["plm_config"]

        return cls(
            component_uid=str(data.get("component_uid", str(uuid.uuid4()))),
            name=str(data.get("name", "Optical Component")),
            component_type=str(data.get("component_type", "generic")),
            surface_uids=list(data.get("surface_uids", [])),
            is_active=bool(data.get("is_active", True)),
            is_locked=bool(data.get("is_locked", False)),
            metadata=dict(data.get("metadata", {})),
            mechanism=mech,
            galvo_config=galvo,
            beamsplitter_config=bs,
            prism_config=prism,
            aom_config=aom,
            eom_config=eom,
            plm_config=plm,
        )


def create_galvo_component(
    name: str = "Galvo Scanner X",
    scan_axis: str = "X",
    min_angle_deg: float = -10.0,
    max_angle_deg: float = 10.0,
    current_angle_deg: float = 0.0,
    pivot_offset_mm: Optional[List[float]] = None,
    component_uid: Optional[str] = None,
) -> Tuple[OpticalComponent, Any]:
    """
    Factory creating a Galvo OpticalComponent along with its configured Surface.
    """
    try:
        from optics_app.engine.surface import Surface, SurfaceType
    except ImportError:
        from v2.engine.surface import Surface, SurfaceType

    c_uid = component_uid or str(uuid.uuid4())
    s_uid = f"surf_{c_uid[:8]}"
    p_offset = pivot_offset_mm if pivot_offset_mm is not None else [0.0, 0.0, -1.5]

    config = GalvoScannerConfig(
        scan_axis=scan_axis,
        min_angle_deg=min_angle_deg,
        max_angle_deg=max_angle_deg,
        current_angle_deg=current_angle_deg,
        pivot_offset_mm=p_offset,
        surface_uid=s_uid,
    )

    surface = Surface(
        name=f"{name} Mirror Face",
        surface_type=SurfaceType.MIRROR,
        radius=0.0,
        thickness=25.0,
        glass="MIRROR",
        semi_diameter=10.0,
        component_uid=c_uid,
        mechanism_type="GALVO",
        galvo_config=config,
    )
    # Also attach galvo_config attribute directly
    surface.galvo_config = config

    comp = OpticalComponent(
        component_uid=c_uid,
        name=name,
        component_type="galvo",
        surface_uids=[s_uid],
        galvo_config=config,
    )
    return comp, surface


def create_beamsplitter_component(
    name: str = "50:50 Plate Beamsplitter",
    splitter_type: str = BeamSplitterType.PLATE,
    coating_type: str = SplitterCoatingType.NON_POLARIZING_50_50,
    split_ratio_R: float = 0.50,
    split_ratio_T: float = 0.50,
    substrate_glass: str = "N-BK7",
    substrate_thickness_mm: float = 3.0,
    cube_side_length_mm: float = 25.4,
    clear_aperture_mm: float = 25.4,
    is_polarizing: bool = False,
    Rs: float = 0.50,
    Rp: float = 0.50,
    Ts: float = 0.50,
    Tp: float = 0.50,
    extinction_ratio: float = 1000.0,
    part_number: str = "",
    component_uid: Optional[str] = None,
) -> Tuple[OpticalComponent, List[Any]]:
    """
    Factory creating a BeamSplitter OpticalComponent with its physical surfaces.
    Accurately supports Cubes (entrance face, internal 45° hypotenuse interface, exit face),
    Plates (front R/T face tilted 45°, rear AR face tilted -45°), and Pellicles.
    """
    try:
        from optics_app.engine.surface import Surface, SurfaceType
    except ImportError:
        from v2.engine.surface import Surface, SurfaceType

    c_uid = component_uid or str(uuid.uuid4())
    config = BeamSplitterConfig(
        splitter_type=splitter_type,
        coating_type=coating_type,
        split_ratio_R=split_ratio_R,
        split_ratio_T=split_ratio_T,
        substrate_glass=substrate_glass,
        substrate_thickness_mm=substrate_thickness_mm,
        cube_side_length_mm=cube_side_length_mm,
        clear_aperture_mm=clear_aperture_mm,
        is_polarizing=is_polarizing,
        Rs=Rs,
        Rp=Rp,
        Ts=Ts,
        Tp=Tp,
        extinction_ratio=extinction_ratio,
        part_number=part_number,
    )

    semi_dia = float(clear_aperture_mm) / 2.0 if clear_aperture_mm > 0 else 12.7

    if splitter_type == BeamSplitterType.CUBE:
        half_side = float(cube_side_length_mm) / 2.0
        s1_uid = f"surf_{c_uid[:8]}_front"
        s2_uid = f"surf_{c_uid[:8]}_hypotenuse"
        s3_uid = f"surf_{c_uid[:8]}_rear"

        surf_front = Surface(
            name=f"{name} (Entrance Face AR)",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=half_side,
            glass=substrate_glass,
            semi_diameter=semi_dia,
            component_uid=c_uid,
        )
        surf_hypotenuse = Surface(
            name=f"{name} (Hypotenuse Interface R/T)",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=half_side,
            glass=substrate_glass,
            semi_diameter=semi_dia,
            component_uid=c_uid,
            tilt_x=45.0,
            is_beam_splitter=True,
            beamsplitter_config=config,
        )
        surf_rear = Surface(
            name=f"{name} (Exit Face AR)",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=25.0,
            glass="AIR",
            semi_diameter=semi_dia,
            component_uid=c_uid,
        )
        surfaces = [surf_front, surf_hypotenuse, surf_rear]
        surface_uids = [s1_uid, s2_uid, s3_uid]
    else:
        # Plate or Pellicle
        t_sub = 0.005 if splitter_type == BeamSplitterType.PELLICLE else float(substrate_thickness_mm)
        g_sub = "AIR" if splitter_type == BeamSplitterType.PELLICLE else substrate_glass

        s1_uid = f"surf_{c_uid[:8]}_front"
        s2_uid = f"surf_{c_uid[:8]}_rear"

        surf_front = Surface(
            name=f"{name} (Front Face R/T)",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=t_sub,
            glass=g_sub,
            semi_diameter=semi_dia,
            component_uid=c_uid,
            tilt_x=45.0,
            is_beam_splitter=True,
            beamsplitter_config=config,
        )
        surf_rear = Surface(
            name=f"{name} (Rear Face BBAR)",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=25.0,
            glass="AIR",
            semi_diameter=semi_dia,
            component_uid=c_uid,
            tilt_x=-45.0,
        )
        surfaces = [surf_front, surf_rear]
        surface_uids = [s1_uid, s2_uid]

    comp = OpticalComponent(
        component_uid=c_uid,
        name=name,
        component_type="beamsplitter",
        surface_uids=surface_uids,
        beamsplitter_config=config,
    )
    return comp, surfaces


def create_amplitude_beamsplitter_component(
    name: str = "Thorlabs 50:50 Beamsplitter",
    part_number: str = "BS016",
    splitter_type: str = BeamSplitterType.CUBE,
    split_ratio_R: float = 0.50,
    split_ratio_T: float = 0.50,
    dimension_mm: float = 25.4,
    substrate_glass: str = "N-BK7",
    thickness_mm: float = 3.0,
    component_uid: Optional[str] = None,
) -> Tuple[OpticalComponent, List[Any]]:
    """Factory for standard amplitude / non-polarizing Thorlabs beam splitters."""
    coating = SplitterCoatingType.NON_POLARIZING_50_50 if (abs(split_ratio_R - 0.5) < 1e-4 and abs(split_ratio_T - 0.5) < 1e-4) else SplitterCoatingType.CUSTOM
    return create_beamsplitter_component(
        name=name,
        splitter_type=splitter_type,
        coating_type=coating,
        split_ratio_R=split_ratio_R,
        split_ratio_T=split_ratio_T,
        substrate_glass=substrate_glass,
        substrate_thickness_mm=thickness_mm,
        cube_side_length_mm=dimension_mm,
        clear_aperture_mm=dimension_mm,
        is_polarizing=False,
        Rs=split_ratio_R,
        Rp=split_ratio_R,
        Ts=split_ratio_T,
        Tp=split_ratio_T,
        part_number=part_number,
        component_uid=component_uid,
    )


def create_polarizing_beamsplitter_component(
    name: str = "Thorlabs Polarizing Beamsplitter",
    part_number: str = "PBS252",
    splitter_type: str = BeamSplitterType.CUBE,
    dimension_mm: float = 25.4,
    substrate_glass: str = "N-SF11",
    extinction_ratio: float = 1000.0,
    Tp: float = 0.92,
    Rs: float = 0.995,
    Rp: float = 0.01,
    Ts: float = 0.005,
    thickness_mm: float = 3.0,
    component_uid: Optional[str] = None,
) -> Tuple[OpticalComponent, List[Any]]:
    """Factory for standard polarizing Thorlabs beam splitters (PBS)."""
    return create_beamsplitter_component(
        name=name,
        splitter_type=splitter_type,
        coating_type=SplitterCoatingType.POLARIZING,
        split_ratio_R=0.5 * (Rs + Rp),
        split_ratio_T=0.5 * (Ts + Tp),
        substrate_glass=substrate_glass,
        substrate_thickness_mm=thickness_mm,
        cube_side_length_mm=dimension_mm,
        clear_aperture_mm=dimension_mm,
        is_polarizing=True,
        Rs=Rs,
        Rp=Rp,
        Ts=Ts,
        Tp=Tp,
        extinction_ratio=extinction_ratio,
        part_number=part_number,
        component_uid=component_uid,
    )


def create_prism_component(
    name: str = "Equilateral Dispersing Prism",
    geometry_type: str = PrismGeometryType.EQUILATERAL,
    apex_angle_deg: float = 60.0,
    glass: str = "N-SF11",
    base_width_mm: float = 25.0,
    component_uid: Optional[str] = None,
) -> Tuple[OpticalComponent, List[Any]]:
    """
    Factory creating a Prism OpticalComponent with entrance and exit facet surfaces.
    """
    try:
        from optics_app.engine.surface import Surface, SurfaceType
    except ImportError:
        from v2.engine.surface import Surface, SurfaceType

    c_uid = component_uid or str(uuid.uuid4())
    config = PrismConfig(
        geometry_type=geometry_type,
        apex_angle_deg=apex_angle_deg,
        glass=glass,
        base_width_mm=base_width_mm,
    )

    s1_uid = f"surf_{c_uid[:8]}_entrance"
    s2_uid = f"surf_{c_uid[:8]}_exit"

    # Facet 1 tilted relative to input
    surf1 = Surface(
        name=f"{name} (Entrance Facet)",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=base_width_mm * 0.8,
        glass=glass,
        semi_diameter=15.0,
        component_uid=c_uid,
        tilt_x=apex_angle_deg / 2.0,
        prism_config=config,
    )

    # Facet 2 exit face back into air
    surf2 = Surface(
        name=f"{name} (Exit Facet)",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=50.0,
        glass="AIR",
        semi_diameter=15.0,
        component_uid=c_uid,
        tilt_x=-apex_angle_deg / 2.0,
        prism_config=config,
    )

    comp = OpticalComponent(
        component_uid=c_uid,
        name=name,
        component_type="prism",
        surface_uids=[s1_uid, s2_uid],
        prism_config=config,
    )
    return comp, [surf1, surf2]


def create_aom_component(
    name: str = "Acousto-Optic Modulator",
    material: str = "TeO2_slow",
    rf_frequency_mhz: float = 80.0,
    rf_power_w: float = 1.0,
    active_aperture_mm: float = 2.0,
    interaction_length_mm: float = 10.0,
    bragg_angle_mrad: float = 0.0,
    diffraction_efficiency: float = 0.0,
    component_uid: Optional[str] = None,
) -> Tuple[OpticalComponent, Any]:
    """
    Factory creating an AOM OpticalComponent along with its configured Surface.
    """
    try:
        from optics_app.engine.surface import Surface, SurfaceType
    except ImportError:
        from v2.engine.surface import Surface, SurfaceType

    c_uid = component_uid or str(uuid.uuid4())
    s_uid = f"surf_{c_uid[:8]}"

    config = AOMConfig(
        material=material,
        rf_frequency_mhz=rf_frequency_mhz,
        rf_power_w=rf_power_w,
        active_aperture_mm=active_aperture_mm,
        interaction_length_mm=interaction_length_mm,
        bragg_angle_mrad=bragg_angle_mrad,
        diffraction_efficiency=diffraction_efficiency,
    )

    surface = Surface(
        name=f"{name} Cell",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=interaction_length_mm,
        glass="AIR",
        semi_diameter=active_aperture_mm / 2.0,
        component_uid=c_uid,
        mechanism_type="AOM",
        aom_config=config,
    )
    surface.aom_config = config

    comp = OpticalComponent(
        component_uid=c_uid,
        name=name,
        component_type="aom",
        surface_uids=[s_uid],
        aom_config=config,
    )
    return comp, surface


def create_eom_component(
    name: str = "Electro-Optic Modulator",
    crystal: str = "LiNbO3",
    v_pi_volts: float = 200.0,
    applied_voltage_volts: float = 0.0,
    bias_voltage_volts: float = 0.0,
    modulation_freq_mhz: float = 10.0,
    extinction_ratio: float = 1000.0,
    transmission: float = 0.0,
    crystal_length_mm: float = 15.0,
    component_uid: Optional[str] = None,
) -> Tuple[OpticalComponent, Any]:
    """
    Factory creating an EOM OpticalComponent along with its configured Surface.
    """
    try:
        from optics_app.engine.surface import Surface, SurfaceType
    except ImportError:
        from v2.engine.surface import Surface, SurfaceType

    c_uid = component_uid or str(uuid.uuid4())
    s_uid = f"surf_{c_uid[:8]}"

    config = EOMConfig(
        crystal=crystal,
        v_pi_volts=v_pi_volts,
        applied_voltage_volts=applied_voltage_volts,
        bias_voltage_volts=bias_voltage_volts,
        modulation_freq_mhz=modulation_freq_mhz,
        extinction_ratio=extinction_ratio,
        transmission=transmission,
    )

    surface = Surface(
        name=f"{name} Cell",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=crystal_length_mm,
        glass="AIR",
        semi_diameter=5.0,
        component_uid=c_uid,
        mechanism_type="EOM",
        eom_config=config,
    )
    surface.eom_config = config

    comp = OpticalComponent(
        component_uid=c_uid,
        name=name,
        component_type="eom",
        surface_uids=[s_uid],
        eom_config=config,
    )
    return comp, surface


def create_aom_mechanism(
    mechanism_id: str = "AOM_RF_Driver",
    target_surface_uid: str = "",
    min_rf_power_w: float = 0.0,
    max_rf_power_w: float = 2.5,
    current_rf_power_w: float = 1.0,
) -> ActiveMechanism:
    """Create an active mechanism controlling an AOM RF power channel."""
    ch = MechanismChannel(
        channel_name="rf_power_w",
        target_surface_uid=target_surface_uid,
        target_property="rf_power_w",
        unit="W",
        min_value=min_rf_power_w,
        max_value=max_rf_power_w,
        current_value=current_rf_power_w,
    )
    return ActiveMechanism(
        mechanism_id=mechanism_id,
        mechanism_type="aom",
        actuation_type=ActuationType.CONTINUOUS,
        channels={"rf_power_w": ch},
    )


def create_eom_mechanism(
    mechanism_id: str = "EOM_Voltage_Driver",
    target_surface_uid: str = "",
    min_voltage_v: float = -300.0,
    max_voltage_v: float = 300.0,
    current_voltage_v: float = 0.0,
) -> ActiveMechanism:
    """Create an active mechanism controlling an EOM voltage channel."""
    ch = MechanismChannel(
        channel_name="applied_voltage_volts",
        target_surface_uid=target_surface_uid,
        target_property="applied_voltage_volts",
        unit="V",
        min_value=min_voltage_v,
        max_value=max_voltage_v,
        current_value=current_voltage_v,
    )
    return ActiveMechanism(
        mechanism_id=mechanism_id,
        mechanism_type="eom",
        actuation_type=ActuationType.CONTINUOUS,
        channels={"applied_voltage_volts": ch},
    )

