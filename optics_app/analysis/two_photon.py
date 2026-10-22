"""Calibrated point-emitter simulation for a two-photon scanning microscope."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np

from optics_app.engine.routes import OpticalRouteSet


PLANCK_J_S = 6.62607015e-34
LIGHT_M_S = 299792458.0
GM_TO_M4_S_PER_PHOTON = 1.0e-58


@dataclass
class SpectralResponse:
    wavelength_nm: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    units: str = "fraction"
    source_url: str = ""
    retrieved: str = ""
    relative_uncertainty: float = 0.0

    def __post_init__(self) -> None:
        self.wavelength_nm = [float(v) for v in self.wavelength_nm]
        self.values = [float(v) for v in self.values]
        if len(self.wavelength_nm) != len(self.values):
            raise ValueError("Spectral wavelength and value arrays must have equal length.")
        if len(self.wavelength_nm) > 1:
            order = np.argsort(self.wavelength_nm)
            self.wavelength_nm = [self.wavelength_nm[i] for i in order]
            self.values = [self.values[i] for i in order]

    @property
    def valid(self) -> bool:
        return len(self.wavelength_nm) >= 2 and all(np.isfinite(self.wavelength_nm)) and all(np.isfinite(self.values))

    @property
    def traceable(self) -> bool:
        return self.valid and bool(str(self.source_url).strip()) and bool(str(self.retrieved).strip())

    def sample(self, wavelength_nm: float, *, strict: bool = True) -> Optional[float]:
        if not self.valid:
            return None
        wl = float(wavelength_nm)
        if strict and (wl < self.wavelength_nm[0] or wl > self.wavelength_nm[-1]):
            return None
        return float(np.interp(wl, self.wavelength_nm, self.values))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["SpectralResponse"]:
        if not data:
            return None
        return cls(
            wavelength_nm=data.get("wavelength_nm", []), values=data.get("values", []),
            units=str(data.get("units", "fraction")), source_url=str(data.get("source_url", "")),
            retrieved=str(data.get("retrieved", "")),
            relative_uncertainty=float(data.get("relative_uncertainty", 0.0)),
        )


@dataclass
class UltrafastLaserConfig:
    center_wavelength_nm: float = 800.0
    average_power_w: float = 0.020
    repetition_rate_hz: float = 80.0e6
    transform_limit_fwhm_fs: float = 100.0
    input_gdd_fs2: float = 0.0
    beam_diameter_mm: float = 8.0
    m2: float = 1.0
    polarization: str = "linear_x"
    average_power_relative_uncertainty: float = 0.02
    pulse_width_relative_uncertainty: float = 0.05
    repetition_rate_relative_uncertainty: float = 0.001

    @property
    def pulse_energy_j(self) -> float:
        return self.average_power_w / self.repetition_rate_hz if self.repetition_rate_hz > 0.0 else 0.0


@dataclass
class PockelsPBSConfig:
    voltage_v: float = 0.0
    half_wave_voltage_v: float = 200.0
    maximum_microscope_fraction: float = 0.95
    minimum_microscope_fraction: float = 0.001
    insertion_loss_fraction: float = 0.02
    dump_power_rating_w: float = 1.0
    input_polarization_angle_deg: float = 0.0

    def split(self, input_power_w: float) -> Tuple[float, float, float]:
        if self.half_wave_voltage_v <= 0.0:
            return 0.0, 0.0, max(0.0, input_power_w)
        phase = math.pi * self.voltage_v / self.half_wave_voltage_v
        input_angle = math.radians(self.input_polarization_angle_deg)
        controlled = self.minimum_microscope_fraction + (
            self.maximum_microscope_fraction - self.minimum_microscope_fraction
        ) * math.cos(input_angle + phase / 2.0) ** 2
        available = max(0.0, input_power_w) * max(0.0, 1.0 - self.insertion_loss_fraction)
        microscope = available * float(np.clip(controlled, 0.0, 1.0))
        dump = available - microscope
        loss = max(0.0, input_power_w - available)
        return microscope, dump, loss


@dataclass
class ObjectiveSpec:
    manufacturer: str = ""
    part_number: str = ""
    magnification: float = 0.0
    numerical_aperture: float = 0.0
    design_tube_lens_focal_length_mm: float = 0.0
    effective_focal_length_mm: float = 0.0
    entrance_pupil_diameter_mm: float = 0.0
    working_distance_mm: float = 0.0
    immersion_medium: str = "air"
    immersion_refractive_index: float = 1.0
    cover_glass_mm: float = 0.0
    wavelength_range_nm: List[float] = field(default_factory=list)
    transmission: Optional[SpectralResponse] = None
    relative_uncertainty: float = 0.0
    source_url: str = ""

    @classmethod
    def from_catalog(cls, record: Dict[str, Any]) -> "ObjectiveSpec":
        payload = dict(record)
        payload["transmission"] = SpectralResponse.from_dict(payload.get("transmission"))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: payload[k] for k in payload if k in allowed})


@dataclass
class FluorophoreSpec:
    identifier: str = ""
    name: str = ""
    environment: str = ""
    two_photon_cross_section_gm: Optional[SpectralResponse] = None
    quantum_yield: float = 0.0
    emission_spectrum: Optional[SpectralResponse] = None
    lifetime_ns: float = 0.0
    relative_uncertainty: float = 0.0
    source_url: str = ""
    source_citation: str = ""

    @classmethod
    def from_catalog(cls, record: Dict[str, Any]) -> "FluorophoreSpec":
        payload = dict(record)
        payload["two_photon_cross_section_gm"] = SpectralResponse.from_dict(payload.get("two_photon_cross_section_gm"))
        payload["emission_spectrum"] = SpectralResponse.from_dict(payload.get("emission_spectrum"))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: payload[k] for k in payload if k in allowed})


@dataclass
class PointEmitterConfig:
    x_um: float = 0.0
    y_um: float = 0.0
    z_um: float = 0.0
    molecule_count: float = 1.0
    medium_refractive_index: float = 1.0
    orientation_model: str = "isotropic"


@dataclass
class PMTSpec:
    manufacturer: str = ""
    part_number: str = ""
    active_area_mm2: float = 0.0
    quantum_efficiency: Optional[SpectralResponse] = None
    dark_count_rate_hz: float = 0.0
    dead_time_ns: float = 0.0
    max_count_rate_hz: float = 0.0
    relative_uncertainty: float = 0.0
    source_url: str = ""

    @classmethod
    def from_catalog(cls, record: Dict[str, Any]) -> "PMTSpec":
        payload = dict(record)
        payload["quantum_efficiency"] = SpectralResponse.from_dict(payload.get("quantum_efficiency"))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: payload[k] for k in payload if k in allowed})


@dataclass
class SimulationIssue:
    level: str
    code: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class TwoPhotonMicroscopeConfig:
    laser: UltrafastLaserConfig = field(default_factory=UltrafastLaserConfig)
    pockels_pbs: PockelsPBSConfig = field(default_factory=PockelsPBSConfig)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    fluorophore: FluorophoreSpec = field(default_factory=FluorophoreSpec)
    emitter: PointEmitterConfig = field(default_factory=PointEmitterConfig)
    pmt: PMTSpec = field(default_factory=PMTSpec)
    routes: OpticalRouteSet = field(default_factory=OpticalRouteSet)
    relay_transmission: Optional[SpectralResponse] = None
    dichroic_reflectance: Optional[SpectralResponse] = None
    emission_filter_transmission: Optional[SpectralResponse] = None
    collection_optics_transmission: Optional[SpectralResponse] = None
    tube_lens_focal_length_mm: float = 200.0
    scan_lens_focal_length_mm: float = 50.0
    dwell_time_us: float = 2.0
    pulse_width_at_sample_fs: Optional[float] = None
    solver_mode: str = "fast"
    scan_range_um: float = 3.0
    scan_samples: int = 61
    damage_threshold_peak_w_m2: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        def item(value: Any) -> Any:
            if value is None:
                return None
            if hasattr(value, "to_dict"):
                return value.to_dict()
            if hasattr(value, "__dataclass_fields__"):
                return {k: item(v) for k, v in asdict(value).items()}
            return value
        return {field_name: item(getattr(self, field_name)) for field_name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TwoPhotonMicroscopeConfig":
        d = data or {}
        return cls(
            laser=UltrafastLaserConfig(**d.get("laser", {})),
            pockels_pbs=PockelsPBSConfig(**d.get("pockels_pbs", {})),
            objective=ObjectiveSpec.from_catalog(d.get("objective", {})),
            fluorophore=FluorophoreSpec.from_catalog(d.get("fluorophore", {})),
            emitter=PointEmitterConfig(**d.get("emitter", {})),
            pmt=PMTSpec.from_catalog(d.get("pmt", {})),
            routes=OpticalRouteSet.from_dict(d.get("routes")),
            relay_transmission=SpectralResponse.from_dict(d.get("relay_transmission")),
            dichroic_reflectance=SpectralResponse.from_dict(d.get("dichroic_reflectance")),
            emission_filter_transmission=SpectralResponse.from_dict(d.get("emission_filter_transmission")),
            collection_optics_transmission=SpectralResponse.from_dict(d.get("collection_optics_transmission")),
            tube_lens_focal_length_mm=float(d.get("tube_lens_focal_length_mm", 200.0)),
            scan_lens_focal_length_mm=float(d.get("scan_lens_focal_length_mm", 50.0)),
            dwell_time_us=float(d.get("dwell_time_us", 2.0)),
            pulse_width_at_sample_fs=d.get("pulse_width_at_sample_fs"),
            solver_mode=str(d.get("solver_mode", "fast")),
            scan_range_um=float(d.get("scan_range_um", 3.0)),
            scan_samples=int(d.get("scan_samples", 61)),
            damage_threshold_peak_w_m2=d.get("damage_threshold_peak_w_m2"),
        )


@dataclass
class FocusFieldResult:
    x_um: np.ndarray
    y_um: np.ndarray
    z_um: np.ndarray
    xy_intensity: np.ndarray
    xz_intensity: np.ndarray
    waist_um: float
    rayleigh_range_um: float
    effective_na: float
    pupil_fill: float
    peak_intensity_w_m2: float


@dataclass
class TwoPhotonSimulationResult:
    valid: bool
    mode: str
    issues: List[SimulationIssue]
    microscope_power_w: float = 0.0
    dump_power_w: float = 0.0
    pockels_loss_w: float = 0.0
    sample_average_power_w: float = 0.0
    pulse_energy_at_sample_j: float = 0.0
    pulse_width_at_sample_fs: float = 0.0
    peak_intensity_w_m2: float = 0.0
    excitation_probability_per_molecule_per_pulse: float = 0.0
    emitted_photons_per_pulse: float = 0.0
    collection_efficiency: float = 0.0
    spectral_detection_efficiency: float = 0.0
    detected_counts_per_second: float = 0.0
    detected_counts_per_dwell: float = 0.0
    shot_noise_sigma_per_dwell: float = 0.0
    calibration_relative_uncertainty: float = 0.0
    calibration_sigma_counts_per_second: float = 0.0
    focus: Optional[FocusFieldResult] = None
    scan_positions_um: np.ndarray = field(default_factory=lambda: np.array([]))
    scan_mechanical_angles_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    scan_counts_per_second: np.ndarray = field(default_factory=lambda: np.array([]))
    emission_wavelength_nm: np.ndarray = field(default_factory=lambda: np.array([]))
    emission_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    detected_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def warnings(self) -> List[str]:
        return [issue.message for issue in self.issues if issue.level != "error"]

    @property
    def errors(self) -> List[str]:
        return [issue.message for issue in self.issues if issue.level == "error"]

    def to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "valid": self.valid, "mode": self.mode,
            "issues": [issue.to_dict() for issue in self.issues],
            "microscope_power_w": self.microscope_power_w, "dump_power_w": self.dump_power_w,
            "pockels_loss_w": self.pockels_loss_w, "sample_average_power_w": self.sample_average_power_w,
            "pulse_energy_at_sample_j": self.pulse_energy_at_sample_j,
            "pulse_width_at_sample_fs": self.pulse_width_at_sample_fs,
            "peak_intensity_w_m2": self.peak_intensity_w_m2,
            "excitation_probability_per_molecule_per_pulse": self.excitation_probability_per_molecule_per_pulse,
            "emitted_photons_per_pulse": self.emitted_photons_per_pulse,
            "collection_efficiency": self.collection_efficiency,
            "spectral_detection_efficiency": self.spectral_detection_efficiency,
            "detected_counts_per_second": self.detected_counts_per_second,
            "detected_counts_per_dwell": self.detected_counts_per_dwell,
            "shot_noise_sigma_per_dwell": self.shot_noise_sigma_per_dwell,
            "calibration_relative_uncertainty": self.calibration_relative_uncertainty,
            "calibration_sigma_counts_per_second": self.calibration_sigma_counts_per_second,
        }
        if include_arrays:
            data.update({
                "scan_positions_um": self.scan_positions_um.tolist(),
                "scan_mechanical_angles_deg": self.scan_mechanical_angles_deg.tolist(),
                "scan_counts_per_second": self.scan_counts_per_second.tolist(),
                "emission_wavelength_nm": self.emission_wavelength_nm.tolist(),
                "emission_spectrum": self.emission_spectrum.tolist(),
                "detected_spectrum": self.detected_spectrum.tolist(),
            })
        return data


class TwoPhotonMicroscopeSimulator:
    """Absolute point-emitter photon budget with fast and vectorial focal solvers."""

    @staticmethod
    def _issue(issues: List[SimulationIssue], level: str, code: str, message: str) -> None:
        issues.append(SimulationIssue(level, code, message))

    @classmethod
    def _validate(cls, config: TwoPhotonMicroscopeConfig, stack: Optional[Any]) -> List[SimulationIssue]:
        issues: List[SimulationIssue] = []
        laser, obj, fluor, pmt = config.laser, config.objective, config.fluorophore, config.pmt
        required = [
            (laser.average_power_w > 0, "laser_power", "Laser average power must be greater than zero."),
            (laser.repetition_rate_hz > 0, "repetition_rate", "Laser repetition rate must be greater than zero."),
            (laser.transform_limit_fwhm_fs > 0, "pulse_width", "Laser pulse width must be greater than zero."),
            (laser.beam_diameter_mm > 0, "beam_diameter", "Laser beam diameter must be greater than zero."),
            (0 < obj.numerical_aperture <= max(1.0, obj.immersion_refractive_index), "objective_na", "Objective NA is missing or invalid."),
            (obj.entrance_pupil_diameter_mm > 0, "objective_pupil", "Objective entrance pupil diameter is required."),
            (obj.transmission is not None and obj.transmission.traceable, "objective_transmission", "A traceable objective transmission curve with source and retrieval date is required."),
            (config.relay_transmission is not None and config.relay_transmission.traceable, "relay_transmission", "A traceable excitation relay transmission curve is required."),
            (fluor.two_photon_cross_section_gm is not None and fluor.two_photon_cross_section_gm.traceable, "cross_section", "A traceable wavelength-dependent two-photon cross-section is required."),
            (fluor.emission_spectrum is not None and fluor.emission_spectrum.traceable, "emission_spectrum", "A traceable fluorophore emission spectrum is required."),
            (0 < fluor.quantum_yield <= 1, "quantum_yield", "Fluorophore quantum yield must be in (0, 1]."),
            (config.emitter.molecule_count > 0, "molecule_count", "Point-emitter molecule count must be greater than zero."),
            (config.dichroic_reflectance is not None and config.dichroic_reflectance.traceable, "dichroic_curve", "A traceable dichroic reflectance curve is required."),
            (config.emission_filter_transmission is not None and config.emission_filter_transmission.traceable, "filter_curve", "A traceable emission-filter transmission curve is required."),
            (config.collection_optics_transmission is not None and config.collection_optics_transmission.traceable, "collection_curve", "A traceable collection-optics transmission curve is required."),
            (pmt.quantum_efficiency is not None and pmt.quantum_efficiency.traceable, "pmt_qe", "A traceable PMT quantum-efficiency curve is required."),
        ]
        for ok, code, message in required:
            if not ok:
                cls._issue(issues, "error", code, message)
        wl = laser.center_wavelength_nm
        if fluor.two_photon_cross_section_gm and fluor.two_photon_cross_section_gm.sample(wl) is None:
            cls._issue(issues, "error", "excitation_out_of_range", f"{wl:.1f} nm is outside the fluorophore cross-section data range; extrapolation is disabled.")
        if obj.wavelength_range_nm and not (obj.wavelength_range_nm[0] <= wl <= obj.wavelength_range_nm[-1]):
            cls._issue(issues, "error", "objective_wavelength", f"{wl:.1f} nm is outside the objective wavelength range.")
        if obj.design_tube_lens_focal_length_mm > 0 and abs(config.tube_lens_focal_length_mm - obj.design_tube_lens_focal_length_mm) > 0.01 * obj.design_tube_lens_focal_length_mm:
            cls._issue(issues, "warning", "tube_lens_mismatch", "Tube-lens focal length differs from the objective design value by more than 1%.")
        if abs(config.emitter.z_um) / 1000.0 > obj.working_distance_mm > 0:
            cls._issue(issues, "warning", "working_distance", "Emitter axial position exceeds the objective working distance.")
        if abs(config.emitter.medium_refractive_index - obj.immersion_refractive_index) > 0.02:
            cls._issue(issues, "warning", "immersion_mismatch", "Sample-medium index does not match the objective immersion model; spherical aberration is not included in the effective objective.")
        if config.emitter.orientation_model.lower() != "isotropic":
            cls._issue(issues, "error", "orientation_model", "Version one supports only an isotropic freely rotating emitter.")
        if stack is not None:
            uids = [getattr(surface, "component_uid", "") for surface in getattr(stack, "surfaces", [])]
            for problem in config.routes.validate(uids):
                cls._issue(issues, "error", "route_reference", problem)
        reference_curves = [
            curve for curve in (
                config.relay_transmission,
                config.dichroic_reflectance,
                config.emission_filter_transmission,
                config.collection_optics_transmission,
            )
            if curve is not None and str(curve.source_url).startswith("embedded://")
        ]
        if reference_curves:
            cls._issue(
                issues, "warning", "reference_response_curves",
                "The template uses embedded reference throughput curves for relay/filter optics; replace them with measured or exact vendor curves for instrument-specific calibration.",
            )
        return issues

    @staticmethod
    def _pulse_width(config: TwoPhotonMicroscopeConfig, stack: Optional[Any], issues: List[SimulationIssue]) -> float:
        if config.pulse_width_at_sample_fs is not None:
            return float(config.pulse_width_at_sample_fs)
        if stack is None or not getattr(stack, "surfaces", None):
            return float(config.laser.transform_limit_fwhm_fs)
        try:
            from optics_app.analysis.pulse_width import PulseAnalysisConfig, PulseWidthAnalysis
            pulse = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig(
                center_wavelength_nm=config.laser.center_wavelength_nm,
                transform_limit_fwhm_fs=config.laser.transform_limit_fwhm_fs,
                initial_gdd_fs2=config.laser.input_gdd_fs2,
            ))
            issues.extend(SimulationIssue("warning", "pulse_analysis", msg) for msg in pulse.warnings)
            valid = [probe for probe in pulse.probes if probe.valid]
            if valid:
                return float(valid[-1].pulse_fwhm_fs)
        except Exception as exc:
            issues.append(SimulationIssue("warning", "pulse_analysis_failed", f"Pulse analysis failed; using source duration: {exc}"))
        return float(config.laser.transform_limit_fwhm_fs)

    @staticmethod
    def _fast_focus(config: TwoPhotonMicroscopeConfig, pulse_energy_j: float, pulse_width_fs: float) -> FocusFieldResult:
        laser, obj = config.laser, config.objective
        pupil_fill = laser.beam_diameter_mm / max(obj.entrance_pupil_diameter_mm, 1e-15)
        effective_na = obj.numerical_aperture * min(1.0, pupil_fill)
        wl_m = laser.center_wavelength_nm * 1e-9
        waist_m = max(laser.m2, 1.0) * wl_m / (math.pi * max(effective_na, 1e-12))
        zr_m = math.pi * waist_m**2 * max(config.emitter.medium_refractive_index, 1e-12) / (max(laser.m2, 1.0) * wl_m)
        span_xy = max(3.5 * waist_m, 0.5e-6)
        span_z = max(3.5 * zr_m, 1.0e-6)
        x = np.linspace(-span_xy, span_xy, 81)
        y = x.copy()
        z = np.linspace(-span_z, span_z, 81)
        xx, yy = np.meshgrid(x, y)
        xy = np.exp(-2.0 * (xx**2 + yy**2) / waist_m**2)
        xz = np.empty((len(z), len(x)))
        for iz, zv in enumerate(z):
            wz = waist_m * math.sqrt(1.0 + (zv / zr_m) ** 2)
            xz[iz, :] = (waist_m / wz) ** 2 * np.exp(-2.0 * x**2 / wz**2)
        temporal_peak_per_s = 2.0 * math.sqrt(math.log(2.0)) / (pulse_width_fs * 1e-15 * math.sqrt(math.pi))
        spatial_peak_per_m2 = 2.0 / (math.pi * waist_m**2)
        peak = pulse_energy_j * temporal_peak_per_s * spatial_peak_per_m2
        return FocusFieldResult(x * 1e6, y * 1e6, z * 1e6, xy, xz, waist_m * 1e6, zr_m * 1e6, effective_na, pupil_fill, peak)

    @classmethod
    def _vectorial_focus(cls, config: TwoPhotonMicroscopeConfig, pulse_energy_j: float, pulse_width_fs: float) -> FocusFieldResult:
        base = cls._fast_focus(config, pulse_energy_j, pulse_width_fs)
        n_medium = max(config.emitter.medium_refractive_index, 1e-12)
        alpha = math.asin(min(0.999999, base.effective_na / n_medium))
        theta = (np.arange(24) + 0.5) * alpha / 24.0
        phi = (np.arange(48) + 0.5) * 2.0 * math.pi / 48.0
        th, ph = np.meshgrid(theta, phi, indexing="ij")
        pupil_radius = max(config.objective.entrance_pupil_diameter_mm / 2.0, 1e-12)
        beam_radius = config.laser.beam_diameter_mm / 2.0
        fill = beam_radius / pupil_radius
        radial = np.sin(th) / max(math.sin(alpha), 1e-12)
        amp = np.exp(-(radial / max(fill, 1e-6)) ** 2) * np.sqrt(np.cos(th)) * np.sin(th)
        ex = amp * (np.cos(th) * np.cos(ph) ** 2 + np.sin(ph) ** 2)
        ey = amp * ((np.cos(th) - 1.0) * np.cos(ph) * np.sin(ph))
        ez = -amp * np.sin(th) * np.cos(ph)
        sx = np.sin(th) * np.cos(ph)
        sy = np.sin(th) * np.sin(ph)
        sz = np.cos(th)
        k = 2.0 * math.pi * n_medium / (config.laser.center_wavelength_nm * 1e-9)

        x_m = base.x_um * 1e-6
        y_m = base.y_um * 1e-6
        z_m = base.z_um * 1e-6
        xy = np.empty((len(y_m), len(x_m)))
        xz = np.empty((len(z_m), len(x_m)))

        def intensity(xv: float, yv: float, zv: float) -> float:
            phase = np.exp(1j * k * (xv * sx + yv * sy + zv * sz))
            return float(abs(np.sum(ex * phase)) ** 2 + abs(np.sum(ey * phase)) ** 2 + abs(np.sum(ez * phase)) ** 2)

        for iy, yv in enumerate(y_m):
            for ix, xv in enumerate(x_m):
                xy[iy, ix] = intensity(float(xv), float(yv), 0.0)
        for iz, zv in enumerate(z_m):
            for ix, xv in enumerate(x_m):
                xz[iz, ix] = intensity(float(xv), 0.0, float(zv))
        dx = abs(x_m[1] - x_m[0])
        dy = abs(y_m[1] - y_m[0])
        norm = float(np.sum(xy) * dx * dy)
        if norm <= 0 or not np.isfinite(norm):
            return base
        xy_norm = xy / norm
        peak_spatial = float(np.max(xy_norm))
        xy_display = xy / max(float(np.max(xy)), 1e-30)
        xz_display = xz / max(float(np.max(xz)), 1e-30)
        temporal_peak_per_s = 2.0 * math.sqrt(math.log(2.0)) / (pulse_width_fs * 1e-15 * math.sqrt(math.pi))
        peak = pulse_energy_j * temporal_peak_per_s * peak_spatial
        return FocusFieldResult(base.x_um, base.y_um, base.z_um, xy_display, xz_display, base.waist_um, base.rayleigh_range_um, base.effective_na, base.pupil_fill, peak)

    @staticmethod
    def _spectral_efficiency(config: TwoPhotonMicroscopeConfig) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        emission = config.fluorophore.emission_spectrum
        assert emission is not None
        wl = np.asarray(emission.wavelength_nm, dtype=float)
        raw = np.clip(np.asarray(emission.values, dtype=float), 0.0, None)
        if len(wl) < 2 or np.trapezoid(raw, wl) <= 0:
            return 0.0, wl, raw, np.zeros_like(raw)
        normalized = raw / np.trapezoid(raw, wl)
        combined = np.ones_like(wl)
        for curve in (
            config.objective.transmission,
            config.dichroic_reflectance,
            config.emission_filter_transmission,
            config.collection_optics_transmission,
            config.pmt.quantum_efficiency,
        ):
            assert curve is not None
            vals = np.array([curve.sample(value) if curve.sample(value) is not None else 0.0 for value in wl])
            combined *= np.clip(vals, 0.0, 1.0)
        detected = normalized * combined
        return float(np.trapezoid(detected, wl)), wl, normalized, detected

    @classmethod
    def run(cls, config: TwoPhotonMicroscopeConfig, mode: Optional[str] = None, stack: Optional[Any] = None) -> TwoPhotonSimulationResult:
        selected_mode = str(mode or config.solver_mode or "fast").lower()
        if selected_mode not in ("fast", "vectorial"):
            raise ValueError("Two-photon solver mode must be 'fast' or 'vectorial'.")
        issues = cls._validate(config, stack)
        if any(issue.level == "error" for issue in issues):
            return TwoPhotonSimulationResult(False, selected_mode, issues)

        laser, obj, emitter = config.laser, config.objective, config.emitter
        microscope_power, dump_power, pockels_loss = config.pockels_pbs.split(laser.average_power_w)
        if dump_power > config.pockels_pbs.dump_power_rating_w:
            cls._issue(issues, "warning", "dump_overload", f"Beam dump receives {dump_power:.4g} W, above its {config.pockels_pbs.dump_power_rating_w:.4g} W rating.")
        relay_t = config.relay_transmission.sample(laser.center_wavelength_nm)  # type: ignore[union-attr]
        objective_t = obj.transmission.sample(laser.center_wavelength_nm)  # type: ignore[union-attr]
        if relay_t is None or objective_t is None:
            cls._issue(issues, "error", "excitation_transmission_range", "Excitation wavelength is outside a required transmission curve.")
            return TwoPhotonSimulationResult(False, selected_mode, issues, microscope_power_w=microscope_power, dump_power_w=dump_power, pockels_loss_w=pockels_loss)
        sample_power = microscope_power * max(0.0, relay_t) * max(0.0, objective_t)
        pulse_width = cls._pulse_width(config, stack, issues)
        pulse_energy = sample_power / laser.repetition_rate_hz
        focus = cls._vectorial_focus(config, pulse_energy, pulse_width) if selected_mode == "vectorial" else cls._fast_focus(config, pulse_energy, pulse_width)

        if focus.pupil_fill < 0.8:
            cls._issue(issues, "warning", "pupil_underfill", f"Objective pupil is underfilled ({focus.pupil_fill:.2f}×); effective NA is {focus.effective_na:.3f}.")
        elif focus.pupil_fill > 1.2:
            cls._issue(issues, "warning", "pupil_overfill", f"Objective pupil is overfilled ({focus.pupil_fill:.2f}×); clipping/vignetting losses may be underestimated.")
        if pulse_width > 1.5 * laser.transform_limit_fwhm_fs:
            cls._issue(issues, "warning", "pulse_broadening", f"Pulse broadened from {laser.transform_limit_fwhm_fs:.1f} fs to {pulse_width:.1f} fs at the sample.")
        if config.damage_threshold_peak_w_m2 and focus.peak_intensity_w_m2 > config.damage_threshold_peak_w_m2:
            cls._issue(issues, "warning", "damage_threshold", "Calculated focal peak intensity exceeds the user-provided threshold; this is not a biological safety determination.")

        delta_gm = config.fluorophore.two_photon_cross_section_gm.sample(laser.center_wavelength_nm)  # type: ignore[union-attr]
        assert delta_gm is not None
        photon_energy = PLANCK_J_S * LIGHT_M_S / (laser.center_wavelength_nm * 1e-9)
        tau_s = pulse_width * 1e-15
        squared_time_integral = tau_s * math.sqrt(math.pi / (8.0 * math.log(2.0)))
        photon_flux_peak = focus.peak_intensity_w_m2 / photon_energy
        exponent = max(0.0, delta_gm * GM_TO_M4_S_PER_PHOTON * photon_flux_peak**2 * squared_time_integral)
        excitation_probability = 1.0 - math.exp(-min(exponent, 700.0))
        if excitation_probability > 0.1:
            cls._issue(issues, "warning", "saturation", f"Excitation probability is {excitation_probability:.1%} per pulse; quadratic scaling is no longer reliable.")
        emitted_per_pulse = emitter.molecule_count * excitation_probability * config.fluorophore.quantum_yield

        n_medium = max(emitter.medium_refractive_index, 1e-12)
        theta = math.asin(min(1.0, obj.numerical_aperture / n_medium))
        collection = (1.0 - math.cos(theta)) / 2.0
        spectral_eff, em_wl, em_spec, detected_spec = cls._spectral_efficiency(config)
        signal_rate = emitted_per_pulse * laser.repetition_rate_hz * collection * spectral_eff
        dead_time_s = max(0.0, config.pmt.dead_time_ns) * 1e-9
        detected_rate = signal_rate / (1.0 + signal_rate * dead_time_s) + max(0.0, config.pmt.dark_count_rate_hz)
        if config.pmt.max_count_rate_hz > 0 and detected_rate > config.pmt.max_count_rate_hz:
            cls._issue(issues, "warning", "pmt_nonlinearity", f"Expected {detected_rate:.3g} counts/s exceeds the PMT rated count rate.")
        dwell_s = max(0.0, config.dwell_time_us) * 1e-6
        counts_dwell = detected_rate * dwell_s

        uncertainty_terms = [
            2.0 * max(0.0, laser.average_power_relative_uncertainty),
            max(0.0, laser.pulse_width_relative_uncertainty),
            max(0.0, laser.repetition_rate_relative_uncertainty),
            max(0.0, config.fluorophore.relative_uncertainty),
            max(0.0, obj.relative_uncertainty),
            max(0.0, config.pmt.relative_uncertainty),
        ]
        for curve in (config.relay_transmission, config.dichroic_reflectance, config.emission_filter_transmission, config.collection_optics_transmission):
            if curve is not None:
                uncertainty_terms.append(max(0.0, curve.relative_uncertainty))
        calibration_uncertainty = math.sqrt(sum(value * value for value in uncertainty_terms))

        scan_positions = np.linspace(-abs(config.scan_range_um), abs(config.scan_range_um), max(3, int(config.scan_samples)))
        scan_scale_mm_per_rad = (
            2.0 * max(config.scan_lens_focal_length_mm, 1e-12)
            * max(obj.effective_focal_length_mm, 1e-12)
            / max(config.tube_lens_focal_length_mm, 1e-12)
        )
        scan_angles = np.degrees(np.arctan((scan_positions * 1e-3) / scan_scale_mm_per_rad))
        wx = max(focus.waist_um, 1e-12)
        scan_profile = np.exp(-4.0 * ((scan_positions - emitter.x_um) ** 2 + emitter.y_um**2) / wx**2)
        scan_signal = max(0.0, detected_rate - config.pmt.dark_count_rate_hz) * scan_profile + config.pmt.dark_count_rate_hz

        return TwoPhotonSimulationResult(
            valid=not any(issue.level == "error" for issue in issues), mode=selected_mode, issues=issues,
            microscope_power_w=microscope_power, dump_power_w=dump_power, pockels_loss_w=pockels_loss,
            sample_average_power_w=sample_power, pulse_energy_at_sample_j=pulse_energy,
            pulse_width_at_sample_fs=pulse_width, peak_intensity_w_m2=focus.peak_intensity_w_m2,
            excitation_probability_per_molecule_per_pulse=excitation_probability,
            emitted_photons_per_pulse=emitted_per_pulse, collection_efficiency=collection,
            spectral_detection_efficiency=spectral_eff, detected_counts_per_second=detected_rate,
            detected_counts_per_dwell=counts_dwell, shot_noise_sigma_per_dwell=math.sqrt(max(counts_dwell, 0.0)),
            calibration_relative_uncertainty=calibration_uncertainty,
            calibration_sigma_counts_per_second=max(0.0, detected_rate - config.pmt.dark_count_rate_hz) * calibration_uncertainty,
            focus=focus, scan_positions_um=scan_positions, scan_mechanical_angles_deg=scan_angles, scan_counts_per_second=scan_signal,
            emission_wavelength_nm=em_wl, emission_spectrum=em_spec, detected_spectrum=detected_spec,
        )


def build_calibrated_template() -> TwoPhotonMicroscopeConfig:
    """Create an immediately runnable reference configuration with traceable catalog records."""
    from optics_app.catalogs.microscopy import get_fluorophore_standard, get_microscope_objective, get_pmt

    objective = ObjectiveSpec.from_catalog(get_microscope_objective("TL15X-2P") or {})
    fluorophore = FluorophoreSpec.from_catalog(get_fluorophore_standard("fluorescein_ph11") or {})
    pmt = PMTSpec.from_catalog(get_pmt("H15460-40") or {})
    source = "embedded://two-photon-reference-response"
    retrieved = "2026-10-03"
    return TwoPhotonMicroscopeConfig(
        objective=objective, fluorophore=fluorophore, pmt=pmt,
        relay_transmission=SpectralResponse([700, 800, 900, 1000, 1100, 1200, 1300], [0.86, 0.90, 0.91, 0.90, 0.88, 0.84, 0.78], source_url=source, retrieved=retrieved, relative_uncertainty=0.05),
        dichroic_reflectance=SpectralResponse([480, 500, 520, 550, 580, 620, 700], [0.88, 0.94, 0.96, 0.96, 0.94, 0.90, 0.05], source_url=source, retrieved=retrieved, relative_uncertainty=0.03),
        emission_filter_transmission=SpectralResponse([480, 500, 520, 540, 560, 580, 600], [0.0, 0.80, 0.92, 0.92, 0.85, 0.15, 0.0], source_url=source, retrieved=retrieved, relative_uncertainty=0.03),
        collection_optics_transmission=SpectralResponse([480, 520, 560, 600, 650, 700], [0.88, 0.90, 0.90, 0.89, 0.87, 0.84], source_url=source, retrieved=retrieved, relative_uncertainty=0.03),
    )


def build_two_photon_surface_template() -> Any:
    """Build a routed microscope layout while keeping the legacy main stack sequential."""
    from optics_app.engine.lde import SurfaceStack
    from optics_app.engine.routes import OpticalRoute, OpticalRouteSet, RouteStep
    from optics_app.engine.surface import Surface, SurfaceType

    stack = SurfaceStack()
    components = [
        Surface(name="Femtosecond Laser Output", group_name="Femtosecond Laser", surface_type=SurfaceType.OBJECT, thickness=30.0, semi_diameter=5.0),
        Surface(name="Pockels Cell Front", group_name="Pockels Cell", radius=0.0, thickness=20.0, glass="FUSED_SILICA", semi_diameter=6.0),
        Surface(name="Pockels Cell Rear", group_name="Pockels Cell", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=6.0),
        Surface(name="Polarizing Beamsplitter", group_name="PBS", surface_type=SurfaceType.STOP, thickness=25.0, semi_diameter=8.0),
        Surface(name="Galvo X", group_name="Galvo X", surface_type=SurfaceType.MIRROR, glass="MIRROR", tilt_y=45.0, thickness=20.0, semi_diameter=5.0),
        Surface(name="Galvo Y", group_name="Galvo Y", surface_type=SurfaceType.MIRROR, glass="MIRROR", tilt_x=-45.0, thickness=35.0, semi_diameter=5.0),
        Surface(name="Scan Lens Front", group_name="Scan Lens", radius=35.0, thickness=6.0, glass="N-BK7", semi_diameter=12.5),
        Surface(name="Scan Lens Rear", group_name="Scan Lens", radius=-35.0, thickness=150.0, glass="AIR", semi_diameter=12.5),
        Surface(name="Tube Lens Front", group_name="Tube Lens", radius=105.0, thickness=8.0, glass="N-BK7", semi_diameter=25.0),
        Surface(name="Tube Lens Rear", group_name="Tube Lens", radius=-105.0, thickness=40.0, glass="AIR", semi_diameter=25.0),
        Surface(name="Primary Dichroic", group_name="Primary Dichroic", surface_type=SurfaceType.STOP, thickness=20.0, semi_diameter=15.0),
        Surface(name="TL15X-2P Effective Objective", group_name="Objective", surface_type=SurfaceType.STOP, thickness=2.6, semi_diameter=9.333),
        Surface(name="Point Fluorophore", group_name="Point Emitter", surface_type=SurfaceType.DETECTOR, thickness=20.0, semi_diameter=0.002),
        # Branch-only hardware follows the terminal sample so ordinary sequential
        # traces stop before it.  Route metadata supplies the physical branch.
        Surface(name="Beam Dump", group_name="Beam Dump", surface_type=SurfaceType.DETECTOR, decenter_x=35.0, tilt_y=90.0, thickness=10.0, semi_diameter=12.5, substrate_shape="Rectangular"),
        Surface(name="Emission Filter", group_name="Emission Filter", surface_type=SurfaceType.STOP, decenter_x=-30.0, tilt_y=-90.0, thickness=15.0, semi_diameter=12.5),
        Surface(name="Collection Lens", group_name="Collection Lens", radius=25.0, glass="N-BK7", decenter_x=-30.0, tilt_y=-90.0, thickness=30.0, semi_diameter=12.5),
        Surface(name="GaAsP PMT H15460-40", group_name="PMT", surface_type=SurfaceType.DETECTOR, decenter_x=-30.0, tilt_y=-90.0, thickness=0.0, semi_diameter=7.0, substrate_shape="Rectangular"),
    ]
    for surface in components:
        stack.add_surface(surface)
    uid: Dict[str, str] = {}
    for surface in stack.surfaces:
        uid.setdefault(surface.group_name, surface.component_uid)
    stack.optical_routes = OpticalRouteSet(
        excitation=OpticalRoute("excitation", [
            RouteStep(uid[name], "transmit") for name in (
                "Femtosecond Laser", "Pockels Cell", "PBS", "Galvo X", "Galvo Y",
                "Scan Lens", "Tube Lens", "Primary Dichroic", "Objective", "Point Emitter",
            )
        ], color="#E53935"),
        dump=OpticalRoute("dump", [RouteStep(uid["PBS"], "reflect"), RouteStep(uid["Beam Dump"], "absorb")], color="#FB8C00"),
        epi_detection=OpticalRoute("epi_detection", [
            RouteStep(uid["Point Emitter"], "emit", "reverse"),
            RouteStep(uid["Objective"], "collect", "reverse"),
            RouteStep(uid["Primary Dichroic"], "reflect", "reverse"),
            RouteStep(uid["Emission Filter"], "transmit"),
            RouteStep(uid["Collection Lens"], "transmit"),
            RouteStep(uid["PMT"], "absorb"),
        ], color="#43A047"),
    )
    config = build_calibrated_template()
    config.routes = stack.optical_routes
    stack.two_photon_config = config
    return stack


TwoPhotonMicroscopeAnalysis = TwoPhotonMicroscopeSimulator

__all__ = [
    "PLANCK_J_S",
    "LIGHT_M_S",
    "GM_TO_M4_S_PER_PHOTON",
    "SpectralResponse",
    "UltrafastLaserConfig",
    "PockelsPBSConfig",
    "ObjectiveSpec",
    "FluorophoreSpec",
    "PointEmitterConfig",
    "PMTSpec",
    "TwoPhotonMicroscopeConfig",
    "TwoPhotonSimulationResult",
    "SimulationIssue",
    "FocusFieldResult",
    "TwoPhotonMicroscopeSimulator",
    "TwoPhotonMicroscopeAnalysis",
    "build_calibrated_template",
    "build_two_photon_surface_template",
]


