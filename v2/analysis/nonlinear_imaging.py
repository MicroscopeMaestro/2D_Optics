"""Dual-pulse nonlinear microscope model.

This is a thin-sample, undepleted, Gaussian-beam model. Absolute predictions
require measured sample coefficients and optical/detector response curves.
SRS is a transfer of energy between existing beams, not a new spectral line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import math

import numpy as np

from optics_app.analysis.pulse_width import gaussian_pulse_fwhm_fs
from optics_app.analysis.two_photon import (
    LIGHT_M_S, PLANCK_J_S, ObjectiveSpec, PMTSpec, SimulationIssue,
    SpectralResponse, UltrafastLaserConfig,
)
from optics_app.engine.routes import OpticalRouteSet


EPSILON_0 = 8.8541878128e-12


@dataclass
class BeamCombinerConfig:
    pump_to_sample_fraction: Optional[float] = None
    stokes_to_sample_fraction: Optional[float] = None
    delay_fs: float = 0.0
    lateral_offset_um: float = 0.0
    dump_power_rating_w: Optional[float] = None
    calibration_source_url: str = ""
    calibration_date: str = ""


@dataclass
class NonlinearSampleSpec:
    name: str = "Uncalibrated sample"
    thickness_um: float = 1.0
    refractive_index: float = 1.0
    d_eff_pm_per_v: Optional[float] = None
    chi3_nonres_m2_per_v2: Optional[float] = None
    chi3_resonant_m2_per_v2: Optional[float] = None
    raman_shift_cm1: Optional[float] = None
    raman_linewidth_cm1: Optional[float] = None
    raman_gain_m_per_w: Optional[float] = None
    phase_mismatch_sfg_m1: Optional[float] = None
    phase_mismatch_dfg_m1: Optional[float] = None
    phase_mismatch_cars_m1: Optional[float] = None
    source_url: str = ""
    measurement_conditions: str = ""
    relative_uncertainty: Optional[float] = None


@dataclass
class NonlinearImagingConfig:
    pump: UltrafastLaserConfig = field(default_factory=UltrafastLaserConfig)
    stokes: UltrafastLaserConfig = field(default_factory=lambda: UltrafastLaserConfig(
        center_wavelength_nm=1040.0, average_power_w=0.020))
    combiner: BeamCombinerConfig = field(default_factory=BeamCombinerConfig)
    sample: NonlinearSampleSpec = field(default_factory=NonlinearSampleSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    pmt: PMTSpec = field(default_factory=PMTSpec)
    routes: OpticalRouteSet = field(default_factory=OpticalRouteSet)
    pump_path_transmission: Optional[SpectralResponse] = None
    stokes_path_transmission: Optional[SpectralResponse] = None
    detection_transmission: Optional[SpectralResponse] = None
    collection_fraction: Optional[float] = None
    detection_direction: str = "forward"
    tube_lens_focal_length_mm: float = 200.0
    dwell_time_us: float = 2.0
    sample_x_um: float = 0.0
    sample_y_um: float = 0.0
    sample_z_um: float = 0.0
    pump_path_gdd_fs2: float = 0.0
    stokes_path_gdd_fs2: float = 0.0
    scan_range_um: float = 3.0
    scan_samples: int = 61

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["routes"] = self.routes.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "NonlinearImagingConfig":
        d = data or {}
        return cls(
            pump=UltrafastLaserConfig(**d.get("pump", {})),
            stokes=UltrafastLaserConfig(**d.get("stokes", {
                "center_wavelength_nm": 1040.0})),
            combiner=BeamCombinerConfig(**d.get("combiner", {})),
            sample=NonlinearSampleSpec(**d.get("sample", {})),
            objective=ObjectiveSpec.from_catalog(d.get("objective", {})),
            pmt=PMTSpec.from_catalog(d.get("pmt", {})),
            routes=OpticalRouteSet.from_dict(d.get("routes")),
            pump_path_transmission=SpectralResponse.from_dict(d.get("pump_path_transmission")),
            stokes_path_transmission=SpectralResponse.from_dict(d.get("stokes_path_transmission")),
            detection_transmission=SpectralResponse.from_dict(d.get("detection_transmission")),
            collection_fraction=d.get("collection_fraction"),
            detection_direction=str(d.get("detection_direction", "forward")),
            tube_lens_focal_length_mm=float(d.get("tube_lens_focal_length_mm", 200.0)),
            dwell_time_us=float(d.get("dwell_time_us", 2.0)),
            sample_x_um=float(d.get("sample_x_um", 0.0)),
            sample_y_um=float(d.get("sample_y_um", 0.0)),
            sample_z_um=float(d.get("sample_z_um", 0.0)),
            pump_path_gdd_fs2=float(d.get("pump_path_gdd_fs2", 0.0)),
            stokes_path_gdd_fs2=float(d.get("stokes_path_gdd_fs2", 0.0)),
            scan_range_um=float(d.get("scan_range_um", 3.0)),
            scan_samples=int(d.get("scan_samples", 61)),
        )


@dataclass
class NonlinearChannelResult:
    process: str
    wavelength_nm: Optional[float]
    valid: bool = False
    generated_power_w: Optional[float] = None
    photons_per_pulse: Optional[float] = None
    pmt_incident_power_w: Optional[float] = None
    pmt_photoelectron_rate_hz: Optional[float] = None
    pmt_counts_per_second: Optional[float] = None
    pmt_counts_per_dwell: Optional[float] = None
    pmt_baseline_counts_per_second: Optional[float] = None
    stokes_gain_w: Optional[float] = None
    pump_loss_w: Optional[float] = None
    issues: List[SimulationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return value


@dataclass
class NonlinearImagingResult:
    valid: bool
    channels: Dict[str, NonlinearChannelResult]
    issues: List[SimulationIssue]
    pump_sample_power_w: float = 0.0
    stokes_sample_power_w: float = 0.0
    rejected_power_w: float = 0.0
    pump_pupil_transmission: float = 0.0
    stokes_pupil_transmission: float = 0.0
    pump_pulse_width_fs: float = 0.0
    stokes_pulse_width_fs: float = 0.0
    pump_peak_intensity_w_m2: float = 0.0
    stokes_peak_intensity_w_m2: float = 0.0
    raman_detuning_cm1: float = 0.0
    delay_fs: np.ndarray = field(default_factory=lambda: np.array([]))
    temporal_overlap: np.ndarray = field(default_factory=lambda: np.array([]))
    scan_positions_um: np.ndarray = field(default_factory=lambda: np.array([]))
    scan_response: np.ndarray = field(default_factory=lambda: np.array([]))
    scan_profiles: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        result = {
            "valid": self.valid,
            "channels": {name: channel.to_dict() for name, channel in self.channels.items()},
            "issues": [asdict(issue) for issue in self.issues],
            "pump_sample_power_w": self.pump_sample_power_w,
            "stokes_sample_power_w": self.stokes_sample_power_w,
            "rejected_power_w": self.rejected_power_w,
            "pump_pupil_transmission": self.pump_pupil_transmission,
            "stokes_pupil_transmission": self.stokes_pupil_transmission,
            "pump_pulse_width_fs": self.pump_pulse_width_fs,
            "stokes_pulse_width_fs": self.stokes_pulse_width_fs,
            "pump_peak_intensity_w_m2": self.pump_peak_intensity_w_m2,
            "stokes_peak_intensity_w_m2": self.stokes_peak_intensity_w_m2,
            "raman_detuning_cm1": self.raman_detuning_cm1,
        }
        if include_arrays:
            result.update({
                "delay_fs": self.delay_fs.tolist(),
                "temporal_overlap": self.temporal_overlap.tolist(),
                "scan_positions_um": self.scan_positions_um.tolist(),
                "scan_response": self.scan_response.tolist(),
                "scan_profiles": {name: values.tolist() for name, values in self.scan_profiles.items()},
            })
        return result


def _issue(level: str, code: str, message: str) -> SimulationIssue:
    return SimulationIssue(level, code, message)


def _response(curve: Optional[SpectralResponse], wavelength_nm: float, label: str,
              issues: List[SimulationIssue]) -> Optional[float]:
    if curve is None or not curve.traceable:
        issues.append(_issue("error", "missing_" + label, f"{label} needs a sourced spectral curve."))
        return None
    value = curve.sample(wavelength_nm)
    if value is None:
        issues.append(_issue("error", label + "_range", f"{label} has no data at {wavelength_nm:.1f} nm."))
        return None
    if not 0 <= value <= 1:
        issues.append(_issue("error", label + "_fraction", f"{label} must be between zero and one."))
        return None
    return value


class NonlinearImagingSimulator:
    """Numerical pulse-overlap and thin-sample nonlinear signal calculation."""

    @staticmethod
    def run(config: NonlinearImagingConfig) -> NonlinearImagingResult:
        p, s, sample, obj = config.pump, config.stokes, config.sample, config.objective
        issues: List[SimulationIssue] = []
        channels = {name: NonlinearChannelResult(name, None) for name in ("SFG", "DFG", "CARS", "SRS")}
        result = NonlinearImagingResult(False, channels, issues)
        if p.center_wavelength_nm > 0 and s.center_wavelength_nm > 0:
            inverse_p, inverse_s = 1 / p.center_wavelength_nm, 1 / s.center_wavelength_nm
            channels["SFG"].wavelength_nm = 1 / (inverse_p + inverse_s)
            channels["DFG"].wavelength_nm = 1 / abs(inverse_p - inverse_s) if abs(inverse_p - inverse_s) > 1e-15 else None
            channels["CARS"].wavelength_nm = 1 / (2 * inverse_p - inverse_s) if 2 * inverse_p > inverse_s else None
            channels["SRS"].wavelength_nm = s.center_wavelength_nm
        if min(p.center_wavelength_nm, s.center_wavelength_nm, p.repetition_rate_hz,
               s.repetition_rate_hz, p.transform_limit_fwhm_fs,
               s.transform_limit_fwhm_fs) <= 0:
            issues.append(_issue("error", "laser_parameters", "Both lasers need positive wavelength, repetition rate, and pulse width."))
            return result
        if p.average_power_w < 0 or s.average_power_w < 0:
            issues.append(_issue("error", "laser_power", "Laser powers cannot be negative."))
            return result
        if abs(p.repetition_rate_hz - s.repetition_rate_hz) / max(p.repetition_rate_hz, s.repetition_rate_hz) > 1e-6:
            issues.append(_issue("error", "synchronization", "Pump and Stokes repetition rates must be synchronized."))
            return result
        if (config.combiner.pump_to_sample_fraction is None or
                config.combiner.stokes_to_sample_fraction is None or
                not config.combiner.calibration_source_url or not config.combiner.calibration_date):
            issues.append(_issue("error", "combiner_calibration", "Both measured combiner fractions and their source/date are required."))
            return result
        if not (0 <= config.combiner.pump_to_sample_fraction <= 1 and
                0 <= config.combiner.stokes_to_sample_fraction <= 1):
            issues.append(_issue("error", "combiner_fraction", "Combiner path fractions must be between zero and one."))
            return result
        if obj.numerical_aperture <= 0 or obj.numerical_aperture > sample.refractive_index:
            issues.append(_issue("error", "objective_na", "Objective NA must be positive and not exceed sample refractive index."))
            return result
        if abs(obj.immersion_refractive_index - sample.refractive_index) > 0.05:
            issues.append(_issue("error", "immersion_mismatch", "Homogeneous sample index does not match objective immersion; interface aberration is not calibrated by this solver."))
            return result
        if obj.design_tube_lens_focal_length_mm > 0 and abs(config.tube_lens_focal_length_mm - obj.design_tube_lens_focal_length_mm) > 0.01:
            issues.append(_issue("warning", "tube_lens_mismatch", "Tube-lens focal length differs from objective design; scan/focus mapping may be incorrect."))
        if sample.thickness_um <= 0 or sample.refractive_index <= 0:
            issues.append(_issue("error", "sample_geometry", "Sample thickness and refractive index must be positive."))
            return result
        if config.collection_fraction is not None and not 0 <= config.collection_fraction <= 1:
            issues.append(_issue("error", "collection_fraction", "Collection fraction must be between zero and one."))
            return result
        if config.detection_direction not in ("forward", "epi"):
            issues.append(_issue("error", "detector_direction", "Detection direction must be forward or epi."))
            return result
        if p.polarization != s.polarization:
            issues.append(_issue("warning", "polarization_model", "Scalar nonlinear model assumes co-polarized beams; differing polarization needs tensor calibration."))
        if obj.wavelength_range_nm:
            for laser, label in ((p, "pump"), (s, "Stokes")):
                if not obj.wavelength_range_nm[0] <= laser.center_wavelength_nm <= obj.wavelength_range_nm[-1]:
                    issues.append(_issue("warning", "objective_range", f"{label} wavelength is outside the objective's specified band."))

        pump_path = _response(config.pump_path_transmission, p.center_wavelength_nm, "pump_path_transmission", issues)
        stokes_path = _response(config.stokes_path_transmission, s.center_wavelength_nm, "stokes_path_transmission", issues)
        obj_p = _response(obj.transmission, p.center_wavelength_nm, "objective_transmission", issues)
        obj_s = _response(obj.transmission, s.center_wavelength_nm, "objective_transmission", issues)
        if any(value is None for value in (pump_path, stokes_path, obj_p, obj_s)):
            return result
        result.pump_sample_power_w = p.average_power_w * config.combiner.pump_to_sample_fraction * pump_path * obj_p
        result.stokes_sample_power_w = s.average_power_w * config.combiner.stokes_to_sample_fraction * stokes_path * obj_s
        result.rejected_power_w = (p.average_power_w * (1 - config.combiner.pump_to_sample_fraction)
                                   + s.average_power_w * (1 - config.combiner.stokes_to_sample_fraction))
        pupil_diameter = max(obj.entrance_pupil_diameter_mm, 1e-9)
        result.pump_pupil_transmission = 1 - math.exp(-2 * (pupil_diameter / max(p.beam_diameter_mm, 1e-9))**2)
        result.stokes_pupil_transmission = 1 - math.exp(-2 * (pupil_diameter / max(s.beam_diameter_mm, 1e-9))**2)
        result.pump_sample_power_w *= result.pump_pupil_transmission
        result.stokes_sample_power_w *= result.stokes_pupil_transmission
        if min(result.pump_pupil_transmission, result.stokes_pupil_transmission) < 0.95:
            issues.append(_issue("warning", "pupil_vignetting", "The objective pupil clips at least 5% of one beam's power."))
        if (config.combiner.dump_power_rating_w is not None and
                result.rejected_power_w > config.combiner.dump_power_rating_w):
            issues.append(_issue("warning", "dump_rating", "Rejected power exceeds the beam-dump rating."))

        tp = gaussian_pulse_fwhm_fs(p.transform_limit_fwhm_fs, p.input_gdd_fs2 + config.pump_path_gdd_fs2)
        ts = gaussian_pulse_fwhm_fs(s.transform_limit_fwhm_fs, s.input_gdd_fs2 + config.stokes_path_gdd_fs2)
        if not np.isfinite(tp) or not np.isfinite(ts):
            issues.append(_issue("error", "pulse_propagation", "Pulse propagation did not yield finite widths."))
            return result
        result.pump_pulse_width_fs, result.stokes_pulse_width_fs = tp, ts
        for width, laser in ((tp, p), (ts, s)):
            if width > 2 * laser.transform_limit_fwhm_fs:
                issues.append(_issue("warning", "pulse_broadening", "Dispersion more than doubled a pulse width."))
        # Gaussian waist, corrected for finite pupil fill. This remains a paraxial
        # approximation and is flagged for high-NA objectives.
        pupil = max(obj.entrance_pupil_diameter_mm, 1e-9)
        wp = p.center_wavelength_nm * 1e-9 / (math.pi * obj.numerical_aperture) * max(1, pupil / max(p.beam_diameter_mm, 1e-9)) * max(p.m2, 1)
        ws = s.center_wavelength_nm * 1e-9 / (math.pi * obj.numerical_aperture) * max(1, pupil / max(s.beam_diameter_mm, 1e-9)) * max(s.m2, 1)
        zp = math.pi * sample.refractive_index * wp**2 / (p.center_wavelength_nm * 1e-9)
        zs = math.pi * sample.refractive_index * ws**2 / (s.center_wavelength_nm * 1e-9)
        axial_p = 1 + (config.sample_z_um * 1e-6 / zp)**2
        axial_s = 1 + (config.sample_z_um * 1e-6 / zs)**2
        radial_p = (config.sample_x_um**2 + config.sample_y_um**2) * 1e-12
        radial_s = ((config.sample_x_um - config.combiner.lateral_offset_um)**2 + config.sample_y_um**2) * 1e-12
        sample_factor_p = math.exp(-2 * radial_p / (wp**2 * axial_p)) / axial_p
        sample_factor_s = math.exp(-2 * radial_s / (ws**2 * axial_s)) / axial_s
        if min(sample_factor_p, sample_factor_s) < 0.1:
            issues.append(_issue("warning", "sample_position", "Sample lies far from the common pump/Stokes focus."))
        if obj.numerical_aperture > 0.6:
            issues.append(_issue("warning", "high_na_approximation", "Nonlinear fields use Gaussian paraxial focusing; high-NA polarization effects are not modeled."))
        for diameter, label in ((p.beam_diameter_mm, "pump"), (s.beam_diameter_mm, "Stokes")):
            if diameter < 0.7 * pupil or diameter > 1.5 * pupil:
                issues.append(_issue("warning", "pupil_fill", f"{label} beam underfills or overfills the objective pupil."))

        fs = 1e-15
        duration = max(tp, ts, abs(config.combiner.delay_fs), 1.0) * 7
        time = np.linspace(-duration, duration, 6001) * fs
        envelope_p = np.exp(-4 * math.log(2) * (time / (tp * fs))**2)
        envelope_s = np.exp(-4 * math.log(2) * ((time - config.combiner.delay_fs * fs) / (ts * fs))**2)
        ep = result.pump_sample_power_w / p.repetition_rate_hz
        es = result.stokes_sample_power_w / s.repetition_rate_hz
        ip = ep * sample_factor_p * envelope_p / (math.pi * wp**2 / 2 * np.trapezoid(envelope_p, time))
        ist = es * sample_factor_s * envelope_s / (math.pi * ws**2 / 2 * np.trapezoid(envelope_s, time))
        result.pump_peak_intensity_w_m2 = float(np.max(ip))
        result.stokes_peak_intensity_w_m2 = float(np.max(ist))
        overlap_area = math.pi / (2 * (1 / wp**2 + 1 / ws**2))
        spatial = math.exp(-2 * (config.combiner.lateral_offset_um * 1e-6)**2 / (wp**2 + ws**2))
        if spatial < 0.1:
            issues.append(_issue("warning", "spatial_overlap", "Pump and Stokes beams overlap poorly at the sample."))
        overlap = float(np.trapezoid(ip * ist, time))
        if abs(config.combiner.delay_fs) > 2 * math.sqrt(tp**2 + ts**2):
            issues.append(_issue("warning", "temporal_overlap", "Pump and Stokes pulses overlap poorly."))
        result.raman_detuning_cm1 = 1e7 * (1 / p.center_wavelength_nm - 1 / s.center_wavelength_nm)
        result.delay_fs = np.linspace(-3 * max(tp, ts), 3 * max(tp, ts), 121)
        result.temporal_overlap = np.exp(-4 * math.log(2) * result.delay_fs**2 / (tp**2 + ts**2))
        result.scan_positions_um = np.linspace(-abs(config.scan_range_um), abs(config.scan_range_um), max(3, config.scan_samples))
        displacement = result.scan_positions_um - config.sample_x_um
        sum_profile = np.exp(-2 * displacement**2 * (1/wp**2 + 1/ws**2) * 1e-12)
        cars_profile = np.exp(-2 * displacement**2 * (2/wp**2 + 1/ws**2) * 1e-12)
        result.scan_profiles = {"SFG": sum_profile, "DFG": sum_profile,
                                "CARS": cars_profile, "SRS": sum_profile}
        result.scan_response = cars_profile

        length = sample.thickness_um * 1e-6
        calibrated = bool(sample.source_url and sample.measurement_conditions and
                          sample.relative_uncertainty is not None)
        for name, channel in channels.items():
            if channel.wavelength_nm is None:
                channel.issues.append(_issue("error", "frequency_order", f"{name} output frequency is not positive."))
                continue
            if name in ("SFG", "DFG"):
                if sample.d_eff_pm_per_v is None:
                    channel.issues.append(_issue("error", "missing_d_eff", "Measured d_eff is required for absolute SFG/DFG power."))
                    continue
                if not calibrated:
                    channel.issues.append(_issue("error", "sample_provenance", "Sample coefficient needs source, conditions, and uncertainty."))
                    continue
                omega = 2 * math.pi * LIGHT_M_S / (channel.wavelength_nm * 1e-9)
                mismatch = sample.phase_mismatch_sfg_m1 if name == "SFG" else sample.phase_mismatch_dfg_m1
                if mismatch is None:
                    channel.issues.append(_issue("error", "missing_phase_mismatch", f"Measured {name} phase mismatch is required; zero cannot be assumed."))
                    continue
                phase = float(np.sinc(mismatch * length / (2 * math.pi))**2)
                if phase < 0.1:
                    channel.issues.append(_issue("warning", "phase_mismatch", "Strong phase mismatch suppresses conversion."))
                factor = (2 * omega**2 * (sample.d_eff_pm_per_v * 1e-12)**2 * length**2
                          / (sample.refractive_index**3 * EPSILON_0 * LIGHT_M_S**3))
                # DFG is driven by E_p E_s*, just as SFG uses E_p E_s.
                generated_energy = max(0.0, factor * overlap * overlap_area * phase)
            elif name == "CARS":
                if sample.chi3_nonres_m2_per_v2 is None or sample.chi3_resonant_m2_per_v2 is None:
                    channel.issues.append(_issue("error", "missing_chi3", "Measured resonant and nonresonant chi3 are required for CARS."))
                    continue
                if not calibrated:
                    channel.issues.append(_issue("error", "sample_provenance", "Sample coefficient needs source, conditions, and uncertainty."))
                    continue
                if sample.raman_shift_cm1 is None or not sample.raman_linewidth_cm1 or sample.raman_linewidth_cm1 <= 0:
                    channel.issues.append(_issue("error", "raman_resonance", "Raman line center and positive linewidth are required."))
                    continue
                gamma = sample.raman_linewidth_cm1
                chi = (sample.chi3_nonres_m2_per_v2 + sample.chi3_resonant_m2_per_v2
                       * gamma / complex(sample.raman_shift_cm1 - result.raman_detuning_cm1, -gamma))
                n = sample.refractive_index
                e_p_sq = 2 * ip / (n * EPSILON_0 * LIGHT_M_S)
                e_s = np.sqrt(2 * ist / (n * EPSILON_0 * LIGHT_M_S))
                polarization_sq = np.abs(0.75 * EPSILON_0 * chi * e_p_sq * e_s)**2
                omega = 2 * math.pi * LIGHT_M_S / (channel.wavelength_nm * 1e-9)
                if sample.phase_mismatch_cars_m1 is None:
                    channel.issues.append(_issue("error", "missing_phase_mismatch", "Measured CARS phase mismatch is required; zero cannot be assumed."))
                    continue
                phase = float(np.sinc(sample.phase_mismatch_cars_m1 * length / (2 * math.pi))**2)
                if phase < 0.1:
                    channel.issues.append(_issue("warning", "phase_mismatch", "Strong phase mismatch suppresses CARS."))
                generated_intensity = omega**2 * length**2 * polarization_sq / (8 * n * LIGHT_M_S * EPSILON_0)
                generated_energy = float(np.trapezoid(generated_intensity, time) * overlap_area * phase)
            else:
                if sample.raman_gain_m_per_w is None:
                    channel.issues.append(_issue("error", "missing_raman_gain", "Measured Raman gain is required for SRS modulation."))
                    continue
                if not calibrated:
                    channel.issues.append(_issue("error", "sample_provenance", "Raman gain needs source, conditions, and uncertainty."))
                    continue
                if sample.raman_shift_cm1 is None or not sample.raman_linewidth_cm1 or sample.raman_linewidth_cm1 <= 0:
                    channel.issues.append(_issue("error", "raman_resonance", "Raman line center and positive linewidth are required."))
                    continue
                if result.raman_detuning_cm1 <= 0:
                    channel.issues.append(_issue("error", "srs_order", "SRS requires pump frequency above Stokes frequency."))
                    continue
                lorentz = sample.raman_linewidth_cm1**2 / (
                    (result.raman_detuning_cm1 - sample.raman_shift_cm1)**2 + sample.raman_linewidth_cm1**2)
                gain = np.expm1(np.clip(sample.raman_gain_m_per_w * ip * length * lorentz, 0, 30))
                delta_stokes_j = float(np.trapezoid(ist * gain, time) * math.pi * ws**2 / 2)
                delta_stokes_j = min(delta_stokes_j, ep * p.center_wavelength_nm / s.center_wavelength_nm)
                channel.stokes_gain_w = delta_stokes_j * p.repetition_rate_hz
                channel.pump_loss_w = channel.stokes_gain_w * s.center_wavelength_nm / p.center_wavelength_nm
                if delta_stokes_j >= 0.1 * ep * p.center_wavelength_nm / s.center_wavelength_nm:
                    channel.issues.append(_issue("warning", "pump_depletion", "SRS is outside the small-signal approximation; gain is capped by energy conservation."))
                channel.issues.append(_issue("warning", "srs_detection", "SRS is a small modulation on a bright beam; use modulated balanced/lock-in detection to resolve it."))
                channel.generated_power_w = None  # No separate generated line.
                channel.valid = True
                NonlinearImagingSimulator._detect(
                    config, channel, channel.stokes_gain_w, s.center_wavelength_nm,
                    baseline_power_w=result.stokes_sample_power_w,
                )
                continue

            # Undepleted model must never predict more converted energy than available.
            input_energy = min(ep, es) if name == "SFG" else ep
            if generated_energy > 0.1 * input_energy:
                channel.issues.append(_issue("warning", "depletion", "Thin-sample undepleted approximation exceeded 10% conversion; output is capped."))
            generated_energy = min(generated_energy, input_energy)
            channel.generated_power_w = generated_energy * p.repetition_rate_hz
            channel.photons_per_pulse = generated_energy / (PLANCK_J_S * LIGHT_M_S / (channel.wavelength_nm * 1e-9))
            channel.valid = True
            NonlinearImagingSimulator._detect(config, channel, channel.generated_power_w, channel.wavelength_nm)

        result.valid = any(channel.valid for channel in channels.values()) and not any(issue.level == "error" for issue in issues)
        return result

    @staticmethod
    def _detect(config: NonlinearImagingConfig, channel: NonlinearChannelResult,
                optical_power_w: float, wavelength_nm: float,
                baseline_power_w: float = 0.0) -> None:
        if config.collection_fraction is None:
            channel.issues.append(_issue("warning", "unknown_collection", "Specify forward collection or measured epi backscatter fraction for PMT prediction."))
            return
        transmission = _response(config.detection_transmission, wavelength_nm, "detection_transmission", channel.issues)
        qe = _response(config.pmt.quantum_efficiency, wavelength_nm, "pmt_qe", channel.issues)
        if transmission is None or qe is None:
            return
        energy = PLANCK_J_S * LIGHT_M_S / (wavelength_nm * 1e-9)
        channel.pmt_incident_power_w = optical_power_w * config.collection_fraction * transmission
        incident_signal_rate = optical_power_w * config.collection_fraction * transmission * qe / energy
        channel.pmt_photoelectron_rate_hz = incident_signal_rate
        baseline_rate = baseline_power_w * config.collection_fraction * transmission * qe / energy
        if config.pmt.part_number.strip().upper() == "H15460-40":
            channel.issues.append(_issue(
                "warning", "voltage_output_pmt",
                "H15460-40 is voltage-output; counts/s need a calibrated photon-counting readout. "
                "Reported photocathode-electron rate is not measured pulse counts.",
            ))
            return
        incident_count_rate = incident_signal_rate + baseline_rate + config.pmt.dark_count_rate_hz
        if config.pmt.dead_time_ns < 0 or config.pmt.max_count_rate_hz <= 0:
            channel.issues.append(_issue("error", "pmt_limits", "PMT dead time and maximum count rate are required."))
            return
        dead_time = config.pmt.dead_time_ns * 1e-9
        count_rate = incident_count_rate / (1 + incident_count_rate * dead_time)
        baseline_incident = baseline_rate + config.pmt.dark_count_rate_hz
        baseline_count = baseline_incident / (1 + baseline_incident * dead_time)
        channel.pmt_baseline_counts_per_second = baseline_count if channel.process == "SRS" else None
        channel.pmt_counts_per_second = count_rate - baseline_count if channel.process == "SRS" else count_rate
        channel.pmt_counts_per_dwell = channel.pmt_counts_per_second * config.dwell_time_us * 1e-6
        if incident_count_rate > 0.1 * config.pmt.max_count_rate_hz:
            channel.issues.append(_issue("warning", "pmt_nonlinearity", "PMT photon rate exceeds 10% of its specified maximum; a linear counts prediction is unreliable."))


def build_nonlinear_surface_template() -> Any:
    """Routed two-input imaging layout; surfaces remain legacy-trace compatible."""
    from optics_app.engine.lde import SurfaceStack
    from optics_app.engine.routes import OpticalRoute, RouteStep
    from optics_app.engine.surface import Surface, SurfaceType
    from optics_app.catalogs.microscopy import get_microscope_objective, get_pmt

    stack = SurfaceStack()
    components = [
        Surface(name="Pump Laser", group_name="Pump Laser", surface_type=SurfaceType.OBJECT, thickness=15, semi_diameter=5),
        Surface(name="Beam Combiner", group_name="Beam Combiner", surface_type=SurfaceType.STOP, thickness=25, semi_diameter=8),
        Surface(name="Galvo X", group_name="Galvo X", surface_type=SurfaceType.MIRROR, glass="MIRROR", tilt_y=45, thickness=20, semi_diameter=5),
        Surface(name="Galvo Y", group_name="Galvo Y", surface_type=SurfaceType.MIRROR, glass="MIRROR", tilt_x=-45, thickness=25, semi_diameter=5),
        Surface(name="Scan Lens", group_name="Scan Lens", radius=35, glass="N-BK7", thickness=50, semi_diameter=12.5),
        Surface(name="Tube Lens", group_name="Tube Lens", radius=105, glass="N-BK7", thickness=45, semi_diameter=25),
        Surface(name="Objective", group_name="Objective", surface_type=SurfaceType.STOP, thickness=3, semi_diameter=9.333),
        Surface(name="Nonlinear Sample", group_name="Sample", surface_type=SurfaceType.DETECTOR, thickness=30, semi_diameter=1),
        # Branch-only hardware follows the terminal sample. Ordinary sequential
        # traces can reach the sample; routed rays describe the other arms.
        Surface(name="Pump Rejection Dump", group_name="Pump Dump", surface_type=SurfaceType.DETECTOR, decenter_x=-25, decenter_y=-45, decenter_z=-153, thickness=0, semi_diameter=8),
        Surface(name="Stokes Rejection Dump", group_name="Stokes Dump", surface_type=SurfaceType.DETECTOR, decenter_x=-25, decenter_y=5, decenter_z=-153, thickness=0, semi_diameter=8),
        Surface(name="Spectral Filter", group_name="Spectral Filter", surface_type=SurfaceType.STOP, decenter_z=-3, thickness=20, semi_diameter=10),
        Surface(name="GaAsP PMT", group_name="PMT", surface_type=SurfaceType.DETECTOR, thickness=0, semi_diameter=7, substrate_shape="Rectangular"),
        Surface(name="Stokes Laser", group_name="Stokes Laser", surface_type=SurfaceType.OBJECT, decenter_x=-60, decenter_y=-45, decenter_z=-173, thickness=0, semi_diameter=5),
    ]
    for component in components:
        stack.add_surface(component)
    uid = {surface.group_name: surface.component_uid for surface in stack.surfaces}
    common = [RouteStep(uid[n], "transmit") for n in ("Beam Combiner", "Galvo X", "Galvo Y", "Scan Lens", "Tube Lens", "Objective", "Sample")]
    routes = OpticalRouteSet(
        excitation=OpticalRoute("pump", [RouteStep(uid["Pump Laser"], "emit"), *common], "#E53935"),
        dump=OpticalRoute("pump rejection", [RouteStep(uid["Beam Combiner"], "reflect"), RouteStep(uid["Pump Dump"], "absorb")], "#FB8C00"),
        epi_detection=OpticalRoute("CARS to PMT", [RouteStep(uid["Sample"], "emit"), RouteStep(uid["Spectral Filter"], "transmit"), RouteStep(uid["PMT"], "absorb")], "#E53935"),
        extra_routes={
            "stokes": OpticalRoute("stokes", [RouteStep(uid["Stokes Laser"], "emit"), *common], "#7E57C2"),
            "stokes_rejection": OpticalRoute("stokes rejection", [RouteStep(uid["Beam Combiner"], "reflect"), RouteStep(uid["Stokes Dump"], "absorb")], "#FFB300"),
            "sfg": OpticalRoute("SFG", [RouteStep(uid["Sample"], "emit"), RouteStep(uid["Spectral Filter"], "transmit"), RouteStep(uid["PMT"], "absorb")], "#1976D2"),
            "dfg": OpticalRoute("DFG (IR)", [RouteStep(uid["Sample"], "emit"), RouteStep(uid["Spectral Filter"], "transmit"), RouteStep(uid["PMT"], "absorb")], "#5D4037"),
            "srs": OpticalRoute("SRS modulation (Stokes)", [RouteStep(uid["Sample"], "transmit"), RouteStep(uid["Spectral Filter"], "transmit"), RouteStep(uid["PMT"], "absorb")], "#9575CD"),
        },
    )
    stack.optical_routes = routes
    config = NonlinearImagingConfig(
        objective=ObjectiveSpec.from_catalog(get_microscope_objective("TL15X-2P") or {}),
        pmt=PMTSpec.from_catalog(get_pmt("H15460-40") or {}), routes=routes,
    )
    stack.nonlinear_imaging_config = config
    return stack


NonlinearImagingAnalysis = NonlinearImagingSimulator

__all__ = [
    "EPSILON_0",
    "BeamCombinerConfig",
    "NonlinearSampleSpec",
    "NonlinearImagingConfig",
    "NonlinearChannelResult",
    "NonlinearImagingResult",
    "NonlinearImagingSimulator",
    "NonlinearImagingAnalysis",
    "build_nonlinear_surface_template",
]

