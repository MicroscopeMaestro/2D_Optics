"""Femtosecond pulse-width and dispersion analysis along sequential systems.

The analyzer traces a narrow frequency comb around the pulse carrier.  At each
surface it reconstructs the accumulated spectral phase from the optical path,
fits GD/GDD/TOD, and propagates a Gaussian spectral field to report the temporal
intensity FWHM.  Wavelength-dependent geometric paths (gratings/prisms/folds)
and ordinary material dispersion are both included and also reported
separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.utils.materials import get_refractive_index


C_MM_PER_FS = 0.000299792458
C_NM_PER_FS = 299.792458


@dataclass
class PulseAnalysisConfig:
    center_wavelength_nm: float = 800.0
    transform_limit_fwhm_fs: float = 100.0
    initial_gdd_fs2: float = 0.0
    include_tod: bool = True
    spectral_samples: int = 13
    spectral_span_fwhm: float = 2.5
    spatial_overlap_tolerance_mm: float = 0.05
    probe_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_wavelength_nm": float(self.center_wavelength_nm),
            "transform_limit_fwhm_fs": float(self.transform_limit_fwhm_fs),
            "initial_gdd_fs2": float(self.initial_gdd_fs2),
            "include_tod": bool(self.include_tod),
            "spectral_samples": int(self.spectral_samples),
            "spectral_span_fwhm": float(self.spectral_span_fwhm),
            "spatial_overlap_tolerance_mm": float(self.spatial_overlap_tolerance_mm),
            "probe_indices": [int(i) for i in self.probe_indices],
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PulseAnalysisConfig":
        data = data or {}
        return cls(
            center_wavelength_nm=float(data.get("center_wavelength_nm", 800.0)),
            transform_limit_fwhm_fs=float(data.get("transform_limit_fwhm_fs", 100.0)),
            initial_gdd_fs2=float(data.get("initial_gdd_fs2", 0.0)),
            include_tod=bool(data.get("include_tod", True)),
            spectral_samples=max(7, int(data.get("spectral_samples", 13))),
            spectral_span_fwhm=max(0.5, float(data.get("spectral_span_fwhm", 2.5))),
            spatial_overlap_tolerance_mm=max(0.0, float(data.get("spatial_overlap_tolerance_mm", 0.05))),
            probe_indices=[int(i) for i in data.get("probe_indices", [])],
        )

    @property
    def gaussian_bandwidth_thz(self) -> float:
        return 0.441 / max(self.transform_limit_fwhm_fs, 1e-9) * 1000.0

    @property
    def gaussian_bandwidth_nm(self) -> float:
        return (self.center_wavelength_nm ** 2 / C_NM_PER_FS) * (self.gaussian_bandwidth_thz / 1000.0)


@dataclass
class PulseProbeResult:
    surface_index: int
    surface_name: str
    path_length_mm: float
    glass_path_mm: float
    group_delay_fs: float
    material_gdd_fs2: float
    geometric_gdd_fs2: float
    system_gdd_fs2: float
    total_gdd_fs2: float
    system_tod_fs3: float
    pulse_fwhm_fs: float
    spatial_chirp_mm: float
    locally_overlapped: bool
    reached_samples: int
    total_samples: int

    @property
    def valid(self) -> bool:
        return self.reached_samples >= 5 and np.isfinite(self.pulse_fwhm_fs)


@dataclass
class PulseAnalysisResult:
    config: PulseAnalysisConfig
    probes: List[PulseProbeResult]
    wavelengths_nm: np.ndarray
    warnings: List[str] = field(default_factory=list)

    def probe(self, surface_index: int) -> Optional[PulseProbeResult]:
        return next((p for p in self.probes if p.surface_index == surface_index), None)


def gaussian_pulse_fwhm_fs(
    transform_limit_fwhm_fs: float,
    gdd_fs2: float = 0.0,
    tod_fs3: float = 0.0,
    include_tod: bool = True,
    points: int = 16384,
) -> float:
    """Numerically propagate a transform-limited Gaussian pulse spectral field."""
    tau = max(float(transform_limit_fwhm_fs), 1e-6)
    n = max(2048, int(points))
    omega_max = 64.0 / tau
    omega = np.linspace(-omega_max, omega_max, n, endpoint=False)
    spectral_field = np.exp(-(omega * omega * tau * tau) / (8.0 * np.log(2.0)))
    phase = 0.5 * float(gdd_fs2) * omega**2
    if include_tod:
        phase += (float(tod_fs3) / 6.0) * omega**3
    field_t = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spectral_field * np.exp(1j * phase))))
    intensity = np.abs(field_t) ** 2
    peak = float(np.max(intensity))
    if peak <= 0.0 or not np.isfinite(peak):
        return float("nan")
    time_fs = np.fft.fftshift(np.fft.fftfreq(n, d=(omega[1] - omega[0]) / (2.0 * np.pi)))
    above = np.flatnonzero(intensity >= 0.5 * peak)
    if len(above) < 2:
        return float("nan")

    left, right = int(above[0]), int(above[-1])

    def crossing(i0: int, i1: int) -> float:
        x0, x1 = float(time_fs[i0]), float(time_fs[i1])
        y0, y1 = float(intensity[i0]), float(intensity[i1])
        if abs(y1 - y0) < 1e-30:
            return x0
        return x0 + (0.5 * peak - y0) * (x1 - x0) / (y1 - y0)

    t_left = crossing(max(0, left - 1), left)
    t_right = crossing(right, min(n - 1, right + 1))
    return max(0.0, t_right - t_left)


class PulseWidthAnalysis:
    """Spectral-phase analysis with a virtual probe at every optical surface."""

    @staticmethod
    def _is_propagating_glass(name: str) -> bool:
        return str(name or "").strip().upper() not in ("", "AIR", "VACUUM", "NONE", "MIRROR", "MIRR", "REFL")

    @staticmethod
    def _frequency_samples(config: PulseAnalysisConfig) -> Tuple[np.ndarray, np.ndarray]:
        n = max(7, int(config.spectral_samples))
        if n % 2 == 0:
            n += 1
        omega0 = 2.0 * np.pi * C_NM_PER_FS / float(config.center_wavelength_nm)
        delta_omega_fwhm = 4.0 * np.log(2.0) / max(float(config.transform_limit_fwhm_fs), 1e-6)
        offsets = np.linspace(
            -config.spectral_span_fwhm * delta_omega_fwhm,
            config.spectral_span_fwhm * delta_omega_fwhm,
            n,
        )
        omega = omega0 + offsets
        if np.any(omega <= 0.0):
            raise ValueError("Pulse bandwidth extends to non-positive optical frequency; increase pulse duration.")
        wavelengths_nm = 2.0 * np.pi * C_NM_PER_FS / omega
        return omega, wavelengths_nm

    @classmethod
    def analyze(cls, system_or_stack: Any, config: Optional[PulseAnalysisConfig] = None) -> PulseAnalysisResult:
        config = config or PulseAnalysisConfig()
        stack = getattr(system_or_stack, "stack", system_or_stack)
        if stack is None or not getattr(stack, "surfaces", None):
            return PulseAnalysisResult(config=config, probes=[], wavelengths_nm=np.array([]), warnings=["No optical system loaded."])

        stack._update_accumulated_positions()
        system = system_or_stack if isinstance(system_or_stack, OpticalSystem) else OpticalSystem(stack)
        surfaces = list(system.surfaces)
        omega, wavelengths_nm = cls._frequency_samples(config)
        center_sample = int(np.argmin(np.abs(wavelengths_nm - config.center_wavelength_nm)))

        # Per-frequency phase/time data accumulated at every reached surface.
        phases: List[List[Tuple[float, float]]] = [[] for _ in surfaces]
        material_phases: List[List[Tuple[float, float]]] = [[] for _ in surfaces]
        path_lengths: List[List[float]] = [[] for _ in surfaces]
        glass_lengths: List[List[float]] = [[] for _ in surfaces]
        hit_positions: List[List[np.ndarray]] = [[] for _ in surfaces]
        central_segment_lengths: Dict[int, float] = {}

        traced_by_sample = []
        for sample_index, wl_nm in enumerate(wavelengths_nm):
            ray = system.trace_ray(Ray(pos=np.zeros(3), dir=np.array([0.0, 0.0, 1.0]), wavelength=float(wl_nm) / 1000.0))
            traced_by_sample.append(ray)
            hits = {int(h["surface_index"]): np.array([h["x_global"], h["y_global"], h["z_global"]], dtype=float)
                    for h in ray.history if not bool(h.get("vignetted", False))}
            if sample_index == center_sample:
                previous = np.zeros(3, dtype=float)  # Source reference is the global origin.
                for idx in range(len(surfaces)):
                    if idx not in hits:
                        break
                    central_segment_lengths[idx] = float(np.linalg.norm(hits[idx] - previous))
                    previous = hits[idx]

        # Fixed central-ray lengths give material-only phase; traced lengths give total phase.
        for sample_index, (om, wl_nm, ray) in enumerate(zip(omega, wavelengths_nm, traced_by_sample)):
            hits = {int(h["surface_index"]): np.array([h["x_global"], h["y_global"], h["z_global"]], dtype=float)
                    for h in ray.history if not bool(h.get("vignetted", False))}
            previous = np.zeros(3, dtype=float)
            optical_path_mm = 0.0
            fixed_material_opl_mm = 0.0
            geometric_path_mm = 0.0
            glass_path_mm = 0.0
            for idx, surf in enumerate(surfaces):
                if idx not in hits:
                    break
                segment_mm = float(np.linalg.norm(hits[idx] - previous))
                previous = hits[idx]
                medium = "AIR" if idx == 0 else str(getattr(surfaces[idx - 1], "glass", "AIR") or "AIR")
                if medium.upper() in ("MIRROR", "MIRR", "REFL"):
                    medium = "AIR"
                n_wl = get_refractive_index(medium, float(wl_nm) / 1000.0)
                optical_path_mm += n_wl * segment_mm
                geometric_path_mm += segment_mm
                if cls._is_propagating_glass(medium):
                    glass_path_mm += segment_mm

                fixed_len = central_segment_lengths.get(idx, segment_mm)
                fixed_material_opl_mm += n_wl * fixed_len
                phase = float(om) * (optical_path_mm / C_MM_PER_FS)
                material_phase = float(om) * (fixed_material_opl_mm / C_MM_PER_FS)
                phases[idx].append((float(om), phase))
                material_phases[idx].append((float(om), material_phase))
                path_lengths[idx].append(geometric_path_mm)
                glass_lengths[idx].append(glass_path_mm)
                hit_positions[idx].append(hits[idx])

        probes: List[PulseProbeResult] = []
        warnings: List[str] = []
        omega0 = 2.0 * np.pi * C_NM_PER_FS / float(config.center_wavelength_nm)

        def fit_phase(samples: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
            if len(samples) < 5:
                return float("nan"), float("nan"), float("nan")
            arr = np.asarray(samples, dtype=float)
            order = np.argsort(arr[:, 0])
            x = arr[order, 0] - omega0
            y = arr[order, 1]
            # Remove the large constant before fitting to improve conditioning.
            y = y - y[int(np.argmin(np.abs(x)))]
            coeff = np.polynomial.polynomial.polyfit(x, y, 3)
            return float(coeff[1]), float(2.0 * coeff[2]), float(6.0 * coeff[3])

        for idx, surf in enumerate(surfaces):
            gd, system_gdd, system_tod = fit_phase(phases[idx])
            _, material_gdd, _ = fit_phase(material_phases[idx])
            geometric_gdd = system_gdd - material_gdd
            total_gdd = float(config.initial_gdd_fs2) + system_gdd
            width = gaussian_pulse_fwhm_fs(
                config.transform_limit_fwhm_fs,
                gdd_fs2=total_gdd,
                tod_fs3=system_tod,
                include_tod=config.include_tod,
            ) if np.isfinite(total_gdd) else float("nan")
            positions = hit_positions[idx]
            if len(positions) >= 2:
                pos_arr = np.asarray(positions, dtype=float)
                spatial_chirp_mm = float(np.max(np.linalg.norm(pos_arr[:, None, :] - pos_arr[None, :, :], axis=2)))
            else:
                spatial_chirp_mm = 0.0
            locally_overlapped = spatial_chirp_mm <= config.spatial_overlap_tolerance_mm
            probes.append(PulseProbeResult(
                surface_index=idx,
                surface_name=str(getattr(surf, "name", f"Surface {idx}")),
                path_length_mm=float(np.median(path_lengths[idx])) if path_lengths[idx] else float("nan"),
                glass_path_mm=float(np.median(glass_lengths[idx])) if glass_lengths[idx] else float("nan"),
                group_delay_fs=gd,
                material_gdd_fs2=material_gdd,
                geometric_gdd_fs2=geometric_gdd,
                system_gdd_fs2=system_gdd,
                total_gdd_fs2=total_gdd,
                system_tod_fs3=system_tod,
                pulse_fwhm_fs=width,
                spatial_chirp_mm=spatial_chirp_mm,
                locally_overlapped=locally_overlapped,
                reached_samples=len(phases[idx]),
                total_samples=len(omega),
            ))
            if len(phases[idx]) < len(omega):
                warnings.append(
                    f"Surface #{idx} '{getattr(surf, 'name', '')}': only {len(phases[idx])}/{len(omega)} spectral samples reached the probe."
                )
            if not locally_overlapped:
                warnings.append(
                    f"Surface #{idx} '{getattr(surf, 'name', '')}': spectral chief rays span "
                    f"{spatial_chirp_mm:.3f} mm, above the {config.spatial_overlap_tolerance_mm:.3f} mm "
                    "point-probe overlap tolerance. Reported duration is a coherent-recombination equivalent."
                )

        return PulseAnalysisResult(config=config, probes=probes, wavelengths_nm=wavelengths_nm, warnings=warnings)


PulseWidthConfig = PulseAnalysisConfig
PulseWidthResult = PulseAnalysisResult

__all__ = [
    "PulseAnalysisConfig",
    "PulseWidthConfig",
    "PulseProbeResult",
    "PulseAnalysisResult",
    "PulseWidthResult",
    "PulseWidthAnalysis",
    "gaussian_pulse_fwhm_fs",
]


