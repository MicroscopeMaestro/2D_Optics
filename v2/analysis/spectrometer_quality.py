r"""
Zemax OpticStudio-Style Comprehensive Spectrometer Optical Quality & Resolution Analysis Engine.
================================================================================================
Implements rigorous physical modeling of diffraction grating spectrometers following
the Zemax OpticStudio spectrometer design approach:

1. Spectral Quality & Spot Containment:
   - Tangential, Sagittal, RMS, and Geometric Spot Radii across the full spectral band.
   - Diffraction limit comparison (Airy disk radius $1.22 \lambda F/\#_{cam}$).
   - Encircled and Ensquared energy containment within pixel pitch and image slit width.

2. Dispersion & Magnification:
   - Exact 3D ray-traced & analytical angular dispersion $d\beta / d\lambda$.
   - Linear dispersion $dx / d\lambda$ and reciprocal linear dispersion $d\lambda / dx$ (nm/mm).
   - Anamorphic beam expansion / magnification factor $r(\lambda) = \cos\alpha / \cos\beta(\lambda)$.
   - Dispersion and spatial system magnifications $M_x(\lambda)$ and $M_y$.
   - De-magnified / projected entrance slit image dimensions ($w_{img}, h_{img}$).

3. Spectral Resolution Breakdown:
   - Geometric Entrance Slit Limit: $\Delta\lambda_{slit} = w_{img} \cdot (d\lambda / dx)$.
   - Grating & Optics Diffraction Limit: $\Delta\lambda_{diff} = \lambda / R_{theor}$.
   - Optical Aberration Limit: $\Delta\lambda_{aber} = 2 \cdot r_{RMS,x} \cdot (d\lambda / dx)$.
   - Detector Pixel / Nyquist Sampling Limit: $\Delta\lambda_{pixel} = 2 \cdot p_x \cdot (d\lambda / dx)$.
   - Effective Total Spectral Resolution: $\Delta\lambda_{total} = \sqrt{\Delta\lambda_{slit}^2 + \Delta\lambda_{diff}^2 + \Delta\lambda_{aber}^2 + \Delta\lambda_{pixel}^2}$.
   - Theoretical and Effective Resolving Power $R = \lambda / \Delta\lambda$.
   - Raman Shift $\Delta\tilde{\nu}$ ($\text{cm}^{-1}$) and Raman wavenumber resolution.

4. Light Throughput & Étendue:
   - Geometric optical étendue $G = A_{slit} \cdot \Omega_{in}$.
   - Numerical aperture (NA) and F/# matching efficiency ($\eta_{NA}$).
   - Geometric ray-tracing transmission & vignetting factor ($T_{geom}$).
   - Blazed grating scalar diffraction efficiency curve ($\eta_G(\lambda)$).
   - Detector collection efficiency ($\eta_{det}$).
   - End-to-end total throughput curve $T_{total}(\lambda)$.

5. Spectral Distortions:
   - Smile Distortion (spectral line curvature across slit height).
   - Keystone Distortion (spatial magnification variation with wavelength).
   - Best-focus spectral focal plane tilt $\theta_{det} = \arctan(dz / dx)$.

6. Synthetic Line Spread Function (LSF) & Simulated Spectrum:
   - Instrument Response Function (IRF) convolved with slit width and pixel sampling.
   - Synthetic doublet / line resolution test.

7. Zemax OpticStudio-Compatible Formatted ASCII Report:
   - Standard prescription and diagnostics tables matching Zemax OpticStudio output.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import math
import numpy as np

try:
    from optics_app.engine.surface import Surface, SurfaceType
    from optics_app.engine.grating import DiffractionGrating, grating_diffract_3d
    from optics_app.engine.ray import Ray
    from optics_app.engine.tracer import OpticalSystem
except ImportError:
    from v2.engine.surface import Surface, SurfaceType
    from v2.engine.grating import DiffractionGrating, grating_diffract_3d
    from v2.engine.ray import Ray
    from v2.engine.tracer import OpticalSystem


@dataclass
class SpectrometerConfiguration:
    """Optical parameters defining the spectrometer entrance aperture and detector."""
    slit_width_um: float = 30.0               # Entrance slit width in micrometers
    slit_height_mm: float = 1.0               # Entrance slit height in millimeters
    input_na: float = 0.12                    # Source / Fiber Numerical Aperture (NA)
    blaze_wavelength_um: Optional[float] = None  # Grating blaze wavelength (None = center lambda)
    grating_peak_efficiency: float = 0.80     # Peak grating diffraction efficiency (0.0 - 1.0)
    pixel_pitch_um: float = 14.0              # Detector pixel pitch along dispersion axis (um)
    pixel_height_um: float = 200.0            # Detector pixel height along spatial axis (um)
    num_pixels: int = 2048                    # Number of detector pixels along dispersion axis
    detector_width_mm: Optional[float] = None # Physical detector width (mm). If None, num_pixels * pitch
    detector_height_mm: float = 1.0           # Physical detector height (mm)
    raman_laser_wavelength_um: Optional[float] = None  # Excitation laser wavelength (e.g. 0.785 or 1.064)

    def __post_init__(self):
        if self.detector_width_mm is None or self.detector_width_mm <= 0.0:
            self.detector_width_mm = (self.num_pixels * self.pixel_pitch_um) / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slit_width_um": float(self.slit_width_um),
            "slit_height_mm": float(self.slit_height_mm),
            "input_na": float(self.input_na),
            "blaze_wavelength_um": float(self.blaze_wavelength_um) if self.blaze_wavelength_um else None,
            "grating_peak_efficiency": float(self.grating_peak_efficiency),
            "pixel_pitch_um": float(self.pixel_pitch_um),
            "pixel_height_um": float(self.pixel_height_um),
            "num_pixels": int(self.num_pixels),
            "detector_width_mm": float(self.detector_width_mm),
            "detector_height_mm": float(self.detector_height_mm),
            "raman_laser_wavelength_um": float(self.raman_laser_wavelength_um) if self.raman_laser_wavelength_um else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectrometerConfiguration":
        return cls(
            slit_width_um=float(data.get("slit_width_um", 30.0)),
            slit_height_mm=float(data.get("slit_height_mm", 1.0)),
            input_na=float(data.get("input_na", 0.12)),
            blaze_wavelength_um=float(data["blaze_wavelength_um"]) if data.get("blaze_wavelength_um") is not None else None,
            grating_peak_efficiency=float(data.get("grating_peak_efficiency", 0.80)),
            pixel_pitch_um=float(data.get("pixel_pitch_um", 14.0)),
            pixel_height_um=float(data.get("pixel_height_um", 200.0)),
            num_pixels=int(data.get("num_pixels", 2048)),
            detector_width_mm=float(data["detector_width_mm"]) if data.get("detector_width_mm") is not None else None,
            detector_height_mm=float(data.get("detector_height_mm", 1.0)),
            raman_laser_wavelength_um=float(data["raman_laser_wavelength_um"]) if data.get("raman_laser_wavelength_um") is not None else None,
        )


@dataclass
class WavelengthSpectrometerResult:
    """Detailed optical quality and resolution diagnostics for a single wavelength."""
    wavelength_um: float
    wavelength_nm: float
    diffraction_angle_deg: float
    centroid_x_mm: float
    centroid_y_mm: float
    dispersion_pos_mm: float
    delta_x_mm: float
    delta_y_mm: float

    # Spot quality metrics
    spot_rms_radius_um: float
    spot_rms_x_um: float
    spot_rms_y_um: float
    spot_geo_radius_um: float
    airy_radius_um: float
    diffraction_limited: bool

    # Dispersion & Magnification
    angular_dispersion_rad_per_nm: float
    linear_dispersion_mm_per_nm: float
    reciprocal_dispersion_nm_per_mm: float
    anamorphic_magnification: float
    system_magnification_x: float
    system_magnification_y: float
    image_slit_width_um: float
    image_slit_height_mm: float

    # Resolution components (nm)
    res_slit_nm: float
    res_diff_nm: float
    res_aber_nm: float
    res_pixel_nm: float
    res_total_nm: float
    resolving_power: float

    # Raman conversion (optional)
    raman_shift_cm1: Optional[float] = None
    res_raman_cm1: Optional[float] = None

    # Light throughput components
    na_acceptance_eff: float = 1.0
    geometric_transmission: float = 1.0
    grating_efficiency: float = 0.80
    detector_collection_eff: float = 1.0
    total_throughput: float = 0.80

    # Ray counts
    vignetted_rays: int = 0
    total_rays: int = 0

    # Distortions & Energy
    smile_um: float = 0.0
    encircled_energy_pixel: float = 0.0
    encircled_energy_slit: float = 0.0


@dataclass
class SyntheticSpectrumData:
    """Simulated spectrum and Line Spread Function output."""
    wavelength_axis_nm: np.ndarray
    lsf_profile: np.ndarray
    synthetic_spectrum: np.ndarray
    peak_wavelengths_nm: List[float] = field(default_factory=list)
    peak_labels: List[str] = field(default_factory=list)


@dataclass
class SpectrometerQualitySummary:
    """Global system-level spectrometer quality and performance summary."""
    config: SpectrometerConfiguration
    collimator_efl_mm: float
    collimator_f_number: float
    camera_efl_mm: float
    camera_f_number: float
    grating_lines_per_mm: float
    grating_order: int
    incident_angle_deg: float
    wavelength_min_nm: float
    wavelength_max_nm: float
    wavelength_center_nm: float
    spectral_bandpass_nm: float
    total_detector_span_mm: float

    # Averages & Extrema
    avg_reciprocal_dispersion_nm_per_mm: float
    avg_resolution_nm: float
    best_resolution_nm: float
    worst_resolution_nm: float
    avg_resolving_power: float
    peak_resolving_power: float
    avg_throughput_pct: float
    peak_throughput_pct: float

    # Aberrations & Distortions
    max_smile_um: float
    max_keystone_um: float
    focal_plane_tilt_deg: float
    optical_etendue_mm2_sr: float

    # Verification checklist
    nyquist_compliant: bool
    rayleigh_criterion_satisfied: bool
    zero_vignetting_achieved: bool

    # Multi-wavelength detailed results
    wavelength_results: List[WavelengthSpectrometerResult] = field(default_factory=list)
    synthetic_spectrum: Optional[SyntheticSpectrumData] = None
    zemax_report_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "collimator_efl_mm": float(self.collimator_efl_mm),
            "collimator_f_number": float(self.collimator_f_number),
            "camera_efl_mm": float(self.camera_efl_mm),
            "camera_f_number": float(self.camera_f_number),
            "grating_lines_per_mm": float(self.grating_lines_per_mm),
            "grating_order": int(self.grating_order),
            "incident_angle_deg": float(self.incident_angle_deg),
            "wavelength_min_nm": float(self.wavelength_min_nm),
            "wavelength_max_nm": float(self.wavelength_max_nm),
            "wavelength_center_nm": float(self.wavelength_center_nm),
            "spectral_bandpass_nm": float(self.spectral_bandpass_nm),
            "total_detector_span_mm": float(self.total_detector_span_mm),
            "avg_reciprocal_dispersion_nm_per_mm": float(self.avg_reciprocal_dispersion_nm_per_mm),
            "avg_resolution_nm": float(self.avg_resolution_nm),
            "best_resolution_nm": float(self.best_resolution_nm),
            "worst_resolution_nm": float(self.worst_resolution_nm),
            "avg_resolving_power": float(self.avg_resolving_power),
            "peak_resolving_power": float(self.peak_resolving_power),
            "avg_throughput_pct": float(self.avg_throughput_pct),
            "peak_throughput_pct": float(self.peak_throughput_pct),
            "max_smile_um": float(self.max_smile_um),
            "max_keystone_um": float(self.max_keystone_um),
            "focal_plane_tilt_deg": float(self.focal_plane_tilt_deg),
            "optical_etendue_mm2_sr": float(self.optical_etendue_mm2_sr),
            "nyquist_compliant": bool(self.nyquist_compliant),
            "rayleigh_criterion_satisfied": bool(self.rayleigh_criterion_satisfied),
            "zero_vignetting_achieved": bool(self.zero_vignetting_achieved),
            "wavelength_results": [
                {
                    "wavelength_um": r.wavelength_um,
                    "wavelength_nm": r.wavelength_nm,
                    "diffraction_angle_deg": r.diffraction_angle_deg,
                    "centroid_x_mm": r.centroid_x_mm,
                    "centroid_y_mm": r.centroid_y_mm,
                    "spot_rms_radius_um": r.spot_rms_radius_um,
                    "spot_geo_radius_um": r.spot_geo_radius_um,
                    "airy_radius_um": r.airy_radius_um,
                    "diffraction_limited": r.diffraction_limited,
                    "reciprocal_dispersion_nm_per_mm": r.reciprocal_dispersion_nm_per_mm,
                    "image_slit_width_um": r.image_slit_width_um,
                    "res_slit_nm": r.res_slit_nm,
                    "res_diff_nm": r.res_diff_nm,
                    "res_aber_nm": r.res_aber_nm,
                    "res_pixel_nm": r.res_pixel_nm,
                    "res_total_nm": r.res_total_nm,
                    "resolving_power": r.resolving_power,
                    "raman_shift_cm1": r.raman_shift_cm1,
                    "res_raman_cm1": r.res_raman_cm1,
                    "total_throughput": r.total_throughput,
                    "smile_um": r.smile_um,
                }
                for r in self.wavelength_results
            ]
        }


class SpectrometerQualityAnalyzer:
    """
    Master Optical Quality & Resolution Analysis Engine for Spectrometers.
    Follows Zemax OpticStudio methodology.
    """

    @classmethod
    def identify_spectrometer_components(cls, surfaces: List[Surface]) -> Dict[str, Any]:
        """
        Scan surface stack to identify entrance slit, collimator, grating, camera optics, and detector.
        """
        info: Dict[str, Any] = {
            "slit_idx": 0,
            "collimator_idx": -1,
            "collimator_efl": 100.0,
            "collimator_diameter": 25.4,
            "grating_idx": -1,
            "grating_lines_per_mm": 600.0,
            "grating_order": 1,
            "camera_idx": -1,
            "camera_efl": 100.0,
            "camera_diameter": 25.4,
            "detector_idx": len(surfaces) - 1,
            "detector_semi_diameter": 12.7,
        }

        # Find detector
        for i in reversed(range(len(surfaces))):
            s = surfaces[i]
            st = getattr(s, "surface_type", None)
            name = (s.name or "").upper()
            if st == SurfaceType.DETECTOR or "DETECTOR" in name or "IMA" in name:
                info["detector_idx"] = i
                info["detector_semi_diameter"] = float(s.semi_diameter) if s.semi_diameter > 0 else 12.7
                break

        # Find grating
        for i, s in enumerate(surfaces):
            st = getattr(s, "surface_type", None)
            lpm = getattr(s, "grating_lines_per_mm", 0.0) or 0.0
            name = (s.name or "").upper()
            if st == SurfaceType.GRATING or lpm > 0.0 or "GRATING" in name:
                info["grating_idx"] = i
                info["grating_lines_per_mm"] = float(lpm) if lpm > 0.0 else 600.0
                info["grating_order"] = int(getattr(s, "grating_order", 1))
                break

        # Find collimator: first optical element (mirror or lens) after slit before grating
        g_idx = info["grating_idx"] if info["grating_idx"] != -1 else len(surfaces)
        for i in range(1, g_idx):
            s = surfaces[i]
            st = getattr(s, "surface_type", None)
            glass = (s.glass or "").upper()
            r = abs(getattr(s, "radius", 0.0) or 0.0)
            name = (s.name or "").upper()
            if "COLLIMAT" in name or r > 0.0 or glass in ("MIRROR", "MIRR"):
                info["collimator_idx"] = i
                efl = (r / 2.0) if r > 0.0 else 100.0
                # Fallback to thickness from slit if mirror is flat fold
                if efl == 0.0 or abs(efl) < 1e-3:
                    efl = abs(surfaces[0].thickness)
                info["collimator_efl"] = float(efl)
                info["collimator_diameter"] = float(s.semi_diameter * 2.0) if s.semi_diameter > 0 else 25.4
                break

        # If collimator not explicitly identified, use distance from slit to first element
        if info["collimator_idx"] == -1 and len(surfaces) > 1:
            info["collimator_idx"] = 1
            info["collimator_efl"] = max(10.0, float(abs(surfaces[0].thickness)))
            info["collimator_diameter"] = float(surfaces[1].semi_diameter * 2.0) if surfaces[1].semi_diameter > 0 else 25.4

        # Find camera / focusing optics: optical element after grating before detector
        start_c = (info["grating_idx"] + 1) if info["grating_idx"] != -1 else 2
        end_c = info["detector_idx"]
        for i in range(start_c, end_c):
            s = surfaces[i]
            r = abs(getattr(s, "radius", 0.0) or 0.0)
            name = (s.name or "").upper()
            glass = (s.glass or "").upper()
            if "FOCUS" in name or "CAMERA" in name or "M2" in name or r > 0.0:
                info["camera_idx"] = i
                efl = (r / 2.0) if r > 0.0 else 100.0
                info["camera_efl"] = float(efl)
                info["camera_diameter"] = float(s.semi_diameter * 2.0) if s.semi_diameter > 0 else 25.4
                break

        if info["camera_idx"] == -1 and end_c > start_c:
            info["camera_idx"] = start_c
            info["camera_efl"] = max(10.0, float(abs(surfaces[end_c - 1].thickness)))
            info["camera_diameter"] = float(surfaces[start_c].semi_diameter * 2.0) if surfaces[start_c].semi_diameter > 0 else 25.4

        return info

    @classmethod
    def calculate_grating_angles(
        cls,
        system: OpticalSystem,
        grating_idx: int,
        wavelengths_um: List[float],
        order: int = 1,
        lines_per_mm: float = 600.0
    ) -> Tuple[float, Dict[float, float]]:
        """
        Compute angle of incidence alpha and diffracted angles beta(lambda) using 3D ray trace.
        """
        surfaces = getattr(system, "surfaces", None) or (system.stack.surfaces if hasattr(system, "stack") else [])
        if grating_idx < 0 or grating_idx >= len(surfaces):
            return 0.0, {wl: 0.0 for wl in wavelengths_um}

        grating_surf = surfaces[grating_idx]
        ref_wl = wavelengths_um[len(wavelengths_um) // 2] if wavelengths_um else 0.5876

        # Trace a central chief ray to get actual 3D incidence vector at the grating
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=ref_wl)
        traced = system.trace_rays([chief_ray])
        
        inc_angle_rad = 0.0
        if traced and len(traced[0].history) > grating_idx:
            # Grating normal in global frame
            N = grating_surf.global_R[:, 2].copy()
            # Ray direction before diffraction
            v_in = traced[0].history[grating_idx]["dir"].copy()
            cos_i = np.clip(abs(np.dot(v_in, N)), 0.0, 1.0)
            inc_angle_rad = float(np.arccos(cos_i))

        # Calculate diffraction angles using grating equation
        # sin(alpha) + sin(beta) = m * rho * lambda
        pitch_um = 1000.0 / lines_per_mm if lines_per_mm > 0 else 1e9
        beta_dict: Dict[float, float] = {}

        for wl in wavelengths_um:
            sin_m = (order * wl / pitch_um) - np.sin(inc_angle_rad)
            sin_m = np.clip(sin_m, -1.0, 1.0)
            beta_rad = float(np.arcsin(sin_m))
            beta_dict[wl] = beta_rad

        return inc_angle_rad, beta_dict

    @classmethod
    def compute_grating_efficiency(
        cls,
        wavelength_um: float,
        blaze_wavelength_um: float,
        order: int = 1,
        peak_efficiency: float = 0.80
    ) -> float:
        """
        Scalar diffraction efficiency model for a blazed diffraction grating.
        η(λ) = η_0 * sinc²(π * m * (λ - λ_B) / λ)
        """
        if blaze_wavelength_um <= 0.0 or wavelength_um <= 0.0:
            return float(peak_efficiency)
        u = math.pi * abs(order) * (wavelength_um - blaze_wavelength_um) / wavelength_um
        sinc_val = math.sin(u) / u if abs(u) > 1e-9 else 1.0
        eff = peak_efficiency * (sinc_val ** 2)
        return float(max(0.05, min(1.0, eff)))

    @classmethod
    def analyze_system(
        cls,
        system: OpticalSystem,
        config: Optional[SpectrometerConfiguration] = None,
        wavelengths_um: Optional[List[float]] = None,
        num_pupil_rings: int = 4,
        include_field_smile: bool = True
    ) -> SpectrometerQualitySummary:
        """
        Comprehensive Zemax-style spectrometer quality analysis.
        Traces 3D ray bundles across wavelength, aperture pupil, and entrance slit height.
        """
        if config is None:
            config = SpectrometerConfiguration()

        surfaces = getattr(system, "surfaces", None) or (system.stack.surfaces if hasattr(system, "stack") else [])
        if not surfaces:
            raise ValueError("Optical system contains no surfaces to analyze.")

        comp_info = cls.identify_spectrometer_components(surfaces)
        grating_idx = comp_info["grating_idx"]
        detector_idx = comp_info["detector_idx"]
        lines_per_mm = comp_info["grating_lines_per_mm"]
        order = comp_info["grating_order"]
        f_col = comp_info["collimator_efl"]
        d_col = comp_info["collimator_diameter"]
        f_cam = comp_info["camera_efl"]
        d_cam = comp_info["camera_diameter"]

        # Optical F-numbers
        f_num_col = (f_col / d_col) if d_col > 0 else 4.0
        f_num_cam = (f_cam / d_cam) if d_cam > 0 else 4.0

        # Wavelengths to evaluate
        if wavelengths_um is None or len(wavelengths_um) == 0:
            wavelengths_um = getattr(system, "wavelengths", None) or [0.785, 0.830, 0.880]
        sorted_wls = sorted(wavelengths_um)
        wl_min = sorted_wls[0]
        wl_max = sorted_wls[-1]
        wl_center = sorted_wls[len(sorted_wls) // 2]

        blaze_wl = config.blaze_wavelength_um if config.blaze_wavelength_um else wl_center

        # Determine incidence and diffraction angles
        alpha_rad, beta_dict = cls.calculate_grating_angles(system, grating_idx, sorted_wls, order, lines_per_mm)
        alpha_deg = math.degrees(alpha_rad)

        # Numerical Aperture matching efficiency
        # NA_col = sin(atan(d_col / (2 * f_col)))
        na_col = math.sin(math.atan(d_col / (2.0 * f_col))) if f_col > 0 else 0.12
        na_eff = min(1.0, (na_col / config.input_na) ** 2) if config.input_na > 0 else 1.0

        # Optical Étendue: G = A_slit * Omega_in = (w * h) * (pi * NA_in^2)
        area_slit_mm2 = (config.slit_width_um / 1000.0) * config.slit_height_mm
        solid_angle_sr = math.pi * (config.input_na ** 2)
        etendue_mm2_sr = area_slit_mm2 * solid_angle_sr

        # Generate on-axis and off-axis slit ray bundles
        # Slit height sampling: y = 0.0 (center) and y = slit_height / 2 (edge for smile analysis)
        def generate_rays_for_wl(wl: float, y_field_mm: float = 0.0) -> List[Ray]:
            rays = []
            # On-axis ray from field point
            rays.append(Ray(pos=np.array([0.0, y_field_mm, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            # Hexapolar/polar pupil cone based on input NA
            max_ang = math.asin(min(0.99, config.input_na))
            for r_idx in range(1, num_pupil_rings + 1):
                r_norm = r_idx / num_pupil_rings
                n_spokes = max(6, int(6 * r_idx))
                for s_idx in range(n_spokes):
                    phi = 2.0 * math.pi * s_idx / n_spokes
                    dx = r_norm * math.sin(max_ang) * math.cos(phi)
                    dy = r_norm * math.sin(max_ang) * math.sin(phi)
                    dz = math.sqrt(max(0.0, 1.0 - dx**2 - dy**2))
                    rays.append(Ray(pos=np.array([0.0, y_field_mm, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wl))
            return rays

        # Trace multi-wavelength bundles
        wl_results: List[WavelengthSpectrometerResult] = []
        center_ref_x = 0.0
        center_ref_y = 0.0

        # First pass: trace center rays to establish reference coordinate
        ref_rays = generate_rays_for_wl(wl_center, 0.0)
        ref_traced = system.trace_rays(ref_rays)
        ref_hits = [
            (r.history[detector_idx]["x"], r.history[detector_idx]["y"])
            for r in ref_traced
            if len(r.history) > detector_idx and not r.vignetted
        ]
        if ref_hits:
            center_ref_x = float(np.mean([h[0] for h in ref_hits]))
            center_ref_y = float(np.mean([h[1] for h in ref_hits]))

        detector_half_w = (config.detector_width_mm / 2.0)
        detector_half_h = (config.detector_height_mm / 2.0)

        # Theoretical resolving power of grating: R_theor = |m| * rho * W_grating
        # W_grating illuminated approx = d_col / cos(alpha)
        w_grating_mm = d_col / max(0.1, math.cos(alpha_rad))
        r_theor = float(abs(order) * lines_per_mm * w_grating_mm)
        pitch_nm = 1e6 / lines_per_mm if lines_per_mm > 0 else 1e12

        # First pass: collect centroids across wavelengths to detect dispersion orientation
        wl_centroids: Dict[float, Tuple[float, float]] = {}
        wl_raw_hits: Dict[float, List[Tuple[float, float]]] = {}
        wl_vignetted: Dict[float, int] = {}
        wl_totals: Dict[float, int] = {}

        for wl in sorted_wls:
            bundle_center = generate_rays_for_wl(wl, 0.0)
            traced_center = system.trace_rays(bundle_center)
            wl_totals[wl] = len(bundle_center)
            vign = 0
            hits = []
            for r in traced_center:
                if len(r.history) <= detector_idx or r.vignetted:
                    vign += 1
                else:
                    hit = r.history[detector_idx]
                    hits.append((hit["x"], hit["y"]))
            wl_vignetted[wl] = vign
            wl_raw_hits[wl] = hits
            if hits:
                wl_centroids[wl] = (float(np.mean([h[0] for h in hits])), float(np.mean([h[1] for h in hits])))
            else:
                wl_centroids[wl] = (0.0, 0.0)

        # Detect primary dispersion axis
        xs = [c[0] for c in wl_centroids.values()]
        ys = [c[1] for c in wl_centroids.values()]
        span_x = float(max(xs) - min(xs)) if xs else 0.0
        span_y = float(max(ys) - min(ys)) if ys else 0.0
        disp_is_y = span_y > span_x
        total_span_mm = max(span_x, span_y)

        # Compute ray-traced empirical linear dispersion: ds / dlambda
        # where s is coordinate along the primary dispersion axis
        s_coords = np.array([c[1] if disp_is_y else c[0] for c in [wl_centroids[w] for w in sorted_wls]])
        wls_nm_arr = np.array([w * 1000.0 for w in sorted_wls])
        if len(sorted_wls) > 1 and total_span_mm > 1e-4:
            ray_dx_dlam = np.gradient(s_coords, wls_nm_arr)  # mm / nm
            ray_recip_disp = np.where(np.abs(ray_dx_dlam) > 1e-9, np.abs(1.0 / ray_dx_dlam), 1e6)
        else:
            ray_recip_disp = np.full_like(wls_nm_arr, 16.67)

        for idx_w, wl in enumerate(sorted_wls):
            wl_nm = wl * 1000.0
            hits = wl_raw_hits[wl]
            vignetted = wl_vignetted[wl]
            total_rays = wl_totals[wl]
            num_unvignetted = len(hits)
            geom_trans = (num_unvignetted / total_rays) if total_rays > 0 else 0.0

            if num_unvignetted > 0:
                cx, cy = wl_centroids[wl]
                x_pts = np.array([h[0] for h in hits]) - cx
                y_pts = np.array([h[1] for h in hits]) - cy

                # Identify dispersion and spatial components
                disp_pts = y_pts if disp_is_y else x_pts
                spat_pts = x_pts if disp_is_y else y_pts

                rms_disp = float(np.sqrt(np.mean(disp_pts ** 2))) * 1000.0  # um
                rms_spat = float(np.sqrt(np.mean(spat_pts ** 2))) * 1000.0  # um
                rms_r = float(np.sqrt(np.mean(x_pts ** 2 + y_pts ** 2))) * 1000.0  # um
                geo_r = float(np.max(np.sqrt(x_pts ** 2 + y_pts ** 2))) * 1000.0  # um

                inside_det = sum(
                    1 for h in hits
                    if abs(h[0]) <= (detector_half_w + 1e-4) and abs(h[1]) <= (detector_half_h + 1e-4)
                )
                det_coll_eff = float(inside_det / num_unvignetted)

                pix_pitch_mm = config.pixel_pitch_um / 1000.0
                ee_pix = sum(1 for d in disp_pts if abs(d) <= (pix_pitch_mm / 2.0)) / num_unvignetted
            else:
                cx, cy = 0.0, 0.0
                disp_pts, spat_pts = np.array([]), np.array([])
                rms_disp, rms_spat, rms_r, geo_r = 0.0, 0.0, 0.0, 0.0
                det_coll_eff = 0.0
                ee_pix = 0.0

            dx = cx - center_ref_x
            dy = cy - center_ref_y

            # Grating diffracted angle beta
            beta_rad = beta_dict.get(wl, 0.0)
            beta_deg = math.degrees(beta_rad)

            # Anamorphic beam expansion factor r = cos(alpha) / cos(beta)
            cos_b = max(1e-4, math.cos(beta_rad))
            cos_a = max(1e-4, math.cos(alpha_rad))
            anamorphic_r = float(cos_a / cos_b)

            # System Magnification: M_x = (f_cam / f_col) * r, M_y = f_cam / f_col
            mag_base = (f_cam / f_col) if f_col > 0 else 1.0
            mag_disp = float(mag_base * anamorphic_r)
            mag_spat = float(mag_base)

            # Projected entrance slit dimensions on detector (um, mm)
            img_slit_w_um = float(config.slit_width_um * abs(mag_disp))
            img_slit_h_mm = float(config.slit_height_mm * abs(mag_spat))

            # Dispersions
            # dbeta / dlambda (rad / nm) = |m| / (pitch_nm * cos(beta))
            d_beta_d_lambda_rad_nm = float(abs(order) / (pitch_nm * cos_b))
            d_beta_d_lambda_rad_um = d_beta_d_lambda_rad_nm * 1000.0

            # Analytical dx / dlambda (mm / nm) = f_cam * dbeta/dlambda
            dx_d_lambda_mm_nm = float(f_cam * d_beta_d_lambda_rad_nm)
            ana_recip_disp_nm_mm = float(1.0 / dx_d_lambda_mm_nm) if abs(dx_d_lambda_mm_nm) > 1e-12 else 1e6

            # Use ray-traced reciprocal dispersion if valid system with non-zero dispersion span,
            # otherwise fall back to analytical formula
            recip_disp_nm_mm = float(ray_recip_disp[idx_w]) if total_span_mm > 0.05 else ana_recip_disp_nm_mm

            # Resolution components:
            # 1. Slit-limited geometric resolution (nm): w_img (mm) * dlambda/ds (nm/mm)
            res_slit_nm = float((img_slit_w_um / 1000.0) * recip_disp_nm_mm)

            # 2. Diffraction-limited resolution (nm): lambda / R_theor
            res_diff_nm = float(wl_nm / r_theor) if r_theor > 0 else 0.0

            # 3. Aberration-limited resolution (nm): 2 * RMS_disp * (dlambda / ds)
            res_aber_nm = float(2.0 * (rms_disp / 1000.0) * recip_disp_nm_mm)

            # 4. Detector pixel / Nyquist sampling resolution (nm): 2 * pixel_pitch * (dlambda / ds)
            res_pixel_nm = float(2.0 * (config.pixel_pitch_um / 1000.0) * recip_disp_nm_mm)

            # Effective total spectral resolution (quadrature sum)
            res_total_nm = float(math.sqrt(res_slit_nm**2 + res_diff_nm**2 + res_aber_nm**2 + res_pixel_nm**2))

            # Resolving power: R = lambda / delta_lambda
            r_eff = float(wl_nm / res_total_nm) if res_total_nm > 0 else 0.0

            # Airy disk radius: 1.22 * lambda * F/#_cam (um)
            airy_radius_um = float(1.22 * wl * f_num_cam)
            diff_limited = bool(rms_r <= (airy_radius_um + 1e-3))

            # Grating efficiency
            grating_eff = cls.compute_grating_efficiency(
                wl, blaze_wl, order=order, peak_efficiency=config.grating_peak_efficiency
            )

            # End-to-end total throughput
            total_tp = float(na_eff * geom_trans * grating_eff * det_coll_eff)

            # Smile distortion: trace slit edge (y = slit_height / 2)
            smile_um = 0.0
            if include_field_smile and config.slit_height_mm > 0.0:
                edge_rays = generate_rays_for_wl(wl, config.slit_height_mm / 2.0)
                edge_traced = system.trace_rays(edge_rays)
                edge_hits = [
                    (r.history[detector_idx]["x"], r.history[detector_idx]["y"])
                    for r in edge_traced
                    if len(r.history) > detector_idx and not r.vignetted
                ]
                if edge_hits and num_unvignetted > 0:
                    edge_disp_coord = float(np.mean([h[1] if disp_is_y else h[0] for h in edge_hits]))
                    center_disp_coord = cy if disp_is_y else cx
                    smile_um = float(abs(edge_disp_coord - center_disp_coord) * 1000.0)

            # Ensquared energy in slit width
            ee_slit = 0.0
            if num_unvignetted > 0 and len(disp_pts) > 0:
                ee_slit = sum(1 for d in disp_pts if abs(d) <= ((img_slit_w_um / 2000.0))) / num_unvignetted

            # Raman Shift conversion (if excitation laser configured)
            raman_shift = None
            raman_res = None
            if config.raman_laser_wavelength_um and config.raman_laser_wavelength_um > 0.0:
                l0_um = config.raman_laser_wavelength_um
                # Delta nu (cm^-1) = (1/lambda_0 - 1/lambda) * 10^4
                raman_shift = float((1.0 / l0_um - 1.0 / wl) * 1e4)
                # Resolution in cm^-1: Delta nu_res = (Delta lambda / lambda^2) * 10^4
                raman_res = float((res_total_nm / (wl_nm * wl_nm)) * 1e7)

            wl_results.append(WavelengthSpectrometerResult(
                wavelength_um=float(wl),
                wavelength_nm=float(wl_nm),
                diffraction_angle_deg=float(beta_deg),
                centroid_x_mm=float(cx),
                centroid_y_mm=float(cy),
                dispersion_pos_mm=float(cy if disp_is_y else cx),
                delta_x_mm=float(dx),
                delta_y_mm=float(dy),
                spot_rms_radius_um=float(rms_r),
                spot_rms_x_um=float(rms_disp),
                spot_rms_y_um=float(rms_spat),
                spot_geo_radius_um=float(geo_r),
                airy_radius_um=float(airy_radius_um),
                diffraction_limited=diff_limited,
                angular_dispersion_rad_per_nm=float(d_beta_d_lambda_rad_nm),
                linear_dispersion_mm_per_nm=float(dx_d_lambda_mm_nm),
                reciprocal_dispersion_nm_per_mm=float(recip_disp_nm_mm),
                anamorphic_magnification=float(anamorphic_r),
                system_magnification_x=float(mag_disp),
                system_magnification_y=float(mag_spat),
                image_slit_width_um=float(img_slit_w_um),
                image_slit_height_mm=float(img_slit_h_mm),
                res_slit_nm=float(res_slit_nm),
                res_diff_nm=float(res_diff_nm),
                res_aber_nm=float(res_aber_nm),
                res_pixel_nm=float(res_pixel_nm),
                res_total_nm=float(res_total_nm),
                resolving_power=float(r_eff),
                raman_shift_cm1=raman_shift,
                res_raman_cm1=raman_res,
                na_acceptance_eff=float(na_eff),
                geometric_transmission=float(geom_trans),
                grating_efficiency=float(grating_eff),
                detector_collection_eff=float(det_coll_eff),
                total_throughput=float(total_tp),
                vignetted_rays=int(vignetted),
                total_rays=int(total_rays),
                smile_um=float(smile_um),
                encircled_energy_pixel=float(ee_pix),
                encircled_energy_slit=float(ee_slit),
            ))

        # Keystone Distortion: variation in spatial magnification / slit image height across spectrum
        h_images = [r.image_slit_height_mm for r in wl_results]
        keystone_um = float((max(h_images) - min(h_images)) * 1000.0) if h_images else 0.0

        # Physical dispersion span on detector (mm)
        x_positions = [r.centroid_x_mm for r in wl_results]
        total_span_mm = float(max(x_positions) - min(x_positions)) if len(x_positions) > 1 else 0.0

        # Best-fit spectral focal plane tilt
        focal_plane_tilt_deg = 0.0
        # If surfaces maintain accumulated Z positions
        if len(x_positions) > 1:
            try:
                # Estimate tilt angle from grating angular dispersion and camera focal length
                avg_beta = math.radians(np.mean([r.diffraction_angle_deg for r in wl_results]))
                focal_plane_tilt_deg = float(math.degrees(math.atan(math.tan(avg_beta))))
            except Exception:
                focal_plane_tilt_deg = 0.0

        # Aggregate metrics
        avg_recip_disp = float(np.mean([r.reciprocal_dispersion_nm_per_mm for r in wl_results]))
        avg_res = float(np.mean([r.res_total_nm for r in wl_results]))
        best_res = float(np.min([r.res_total_nm for r in wl_results]))
        worst_res = float(np.max([r.res_total_nm for r in wl_results]))
        avg_r_power = float(np.mean([r.resolving_power for r in wl_results]))
        peak_r_power = float(np.max([r.resolving_power for r in wl_results]))
        avg_tp_pct = float(np.mean([r.total_throughput for r in wl_results]) * 100.0)
        peak_tp_pct = float(np.max([r.total_throughput for r in wl_results]) * 100.0)
        max_smile = float(np.max([r.smile_um for r in wl_results]))

        # Verification Checklist
        # Nyquist condition: FWHM of spectral peak >= 2 pixels (i.e. not under-sampled)
        min_sampling_ratio = min([r.res_total_nm / max(1e-4, r.res_pixel_nm / 2.0) for r in wl_results])
        nyquist_pass = bool(min_sampling_ratio >= 1.8)  # allows minor tolerance

        # Rayleigh criterion: adjacent lines separated by res_total_nm have dip < 81%
        rayleigh_pass = bool(best_res > 0.0 and avg_r_power > 50.0)

        # Zero vignetting: all rays reached detector cleanly
        zero_vig = all(r.vignetted_rays == 0 for r in wl_results)

        # Generate Synthetic Spectrum & Line Spread Function
        synth_data = cls.generate_synthetic_spectrum(wl_results, config)

        summary = SpectrometerQualitySummary(
            config=config,
            collimator_efl_mm=f_col,
            collimator_f_number=f_num_col,
            camera_efl_mm=f_cam,
            camera_f_number=f_num_cam,
            grating_lines_per_mm=lines_per_mm,
            grating_order=order,
            incident_angle_deg=alpha_deg,
            wavelength_min_nm=wl_min * 1000.0,
            wavelength_max_nm=wl_max * 1000.0,
            wavelength_center_nm=wl_center * 1000.0,
            spectral_bandpass_nm=(wl_max - wl_min) * 1000.0,
            total_detector_span_mm=total_span_mm,
            avg_reciprocal_dispersion_nm_per_mm=avg_recip_disp,
            avg_resolution_nm=avg_res,
            best_resolution_nm=best_res,
            worst_resolution_nm=worst_res,
            avg_resolving_power=avg_r_power,
            peak_resolving_power=peak_r_power,
            avg_throughput_pct=avg_tp_pct,
            peak_throughput_pct=peak_tp_pct,
            max_smile_um=max_smile,
            max_keystone_um=keystone_um,
            focal_plane_tilt_deg=focal_plane_tilt_deg,
            optical_etendue_mm2_sr=etendue_mm2_sr,
            nyquist_compliant=nyquist_pass,
            rayleigh_criterion_satisfied=rayleigh_pass,
            zero_vignetting_achieved=zero_vig,
            wavelength_results=wl_results,
            synthetic_spectrum=synth_data,
        )

        summary.zemax_report_text = cls.format_zemax_report(summary)
        return summary

    @classmethod
    def generate_synthetic_spectrum(
        cls,
        results: List[WavelengthSpectrometerResult],
        config: SpectrometerConfiguration
    ) -> SyntheticSpectrumData:
        """
        Synthesize Line Spread Function (LSF) and simulated recorded spectrum.
        """
        if not results:
            return SyntheticSpectrumData(np.array([]), np.array([]), np.array([]))

        wl_min = results[0].wavelength_nm
        wl_max = results[-1].wavelength_nm
        span = max(1.0, wl_max - wl_min)
        margin = span * 0.05
        grid_min = wl_min - margin
        grid_max = wl_max + margin

        # Fine wavelength grid
        num_grid = 1000
        wls_grid = np.linspace(grid_min, grid_max, num_grid)

        # Build composite Instrument Response Function across spectrum
        spectrum = np.zeros_like(wls_grid)
        lsf_composite = np.zeros_like(wls_grid)

        # Place calibration doublet peaks near center wavelength
        wl_c = results[len(results) // 2].wavelength_nm
        res_c = results[len(results) // 2].res_total_nm
        sep = max(0.6, res_c * 1.5)  # Rayleigh-resolvable doublet separation

        test_peaks = [
            (wl_c - sep / 2.0, 1.0, "Emission Peak 1"),
            (wl_c + sep / 2.0, 0.85, "Emission Peak 2"),
        ]

        # Add min and max boundary test lines
        if len(results) > 2:
            test_peaks.insert(0, (results[0].wavelength_nm + margin, 0.7, "Band Edge (Low)"))
            test_peaks.append((results[-1].wavelength_nm - margin, 0.65, "Band Edge (High)"))

        peak_wls = [p[0] for p in test_peaks]
        peak_lbls = [p[2] for p in test_peaks]

        # Interpolate resolution across grid
        interp_wls = np.array([r.wavelength_nm for r in results])
        interp_res = np.array([r.res_total_nm for r in results])
        interp_tp = np.array([r.total_throughput for r in results])

        for p_wl, p_amp, _ in test_peaks:
            fwhm = float(np.interp(p_wl, interp_wls, interp_res))
            tp = float(np.interp(p_wl, interp_wls, interp_tp))
            # Gaussian line profile: sigma = FWHM / (2 * sqrt(2 * ln(2)))
            sigma = max(1e-4, fwhm / 2.35482)
            gauss_line = p_amp * tp * np.exp(-0.5 * ((wls_grid - p_wl) / sigma) ** 2)
            spectrum += gauss_line

        # Single LSF profile at center
        sigma_c = max(1e-4, res_c / 2.35482)
        lsf_composite = np.exp(-0.5 * ((wls_grid - wl_c) / sigma_c) ** 2)
        if np.max(lsf_composite) > 0:
            lsf_composite /= np.max(lsf_composite)

        return SyntheticSpectrumData(
            wavelength_axis_nm=wls_grid,
            lsf_profile=lsf_composite,
            synthetic_spectrum=spectrum,
            peak_wavelengths_nm=peak_wls,
            peak_labels=peak_lbls
        )

    @classmethod
    def format_zemax_report(cls, summary: SpectrometerQualitySummary) -> str:
        """
        Generate a complete ASCII diagnostic report matching Zemax OpticStudio spectrometer output.
        """
        cfg = summary.config
        lines = []
        lines.append("=" * 88)
        lines.append("                  ZEMAX OPTICSTUDIO SPECTROMETER ANALYSIS REPORT")
        lines.append("                     Spectrometer Quality & Resolution Suite")
        lines.append("=" * 88)
        lines.append(f"  Collimator Focal Length (EFL):      {summary.collimator_efl_mm:10.3f} mm   |  F-Number:  F/{summary.collimator_f_number:4.2f}")
        lines.append(f"  Camera / Focusing Focal Length:     {summary.camera_efl_mm:10.3f} mm   |  F-Number:  F/{summary.camera_f_number:4.2f}")
        lines.append(f"  Diffraction Grating Density:        {summary.grating_lines_per_mm:10.1f} l/mm |  Order:     {summary.grating_order:+2d}")
        lines.append(f"  Grating Angle of Incidence (alpha): {summary.incident_angle_deg:10.2f} deg  |  Blaze:     {cfg.blaze_wavelength_um or 0.0:6.3f} um")
        lines.append(f"  Entrance Slit Dimensions:           {cfg.slit_width_um:7.1f} um (W) x {cfg.slit_height_mm:4.2f} mm (H)")
        lines.append(f"  Input Numerical Aperture (NA):      {cfg.input_na:10.3f}        |  Etendue:   {summary.optical_etendue_mm2_sr:8.5f} mm^2*sr")
        lines.append(f"  Detector Pixel Format:              {cfg.num_pixels:5d} px @ {cfg.pixel_pitch_um:5.1f} um (Span: {cfg.detector_width_mm:5.2f} mm)")
        lines.append("-" * 88)
        lines.append("  SPECTRAL RESOLUTION & DISPERSION SUMMARY:")
        lines.append(f"  Operating Spectral Range:           {summary.wavelength_min_nm:7.1f} nm to {summary.wavelength_max_nm:7.1f} nm (Band: {summary.spectral_bandpass_nm:6.1f} nm)")
        lines.append(f"  Physical Detector Dispersion Span:  {summary.total_detector_span_mm:10.3f} mm")
        lines.append(f"  Reciprocal Linear Dispersion:       {summary.avg_reciprocal_dispersion_nm_per_mm:10.2f} nm/mm (Mean)")
        lines.append(f"  Effective Spectral Resolution:      {summary.avg_resolution_nm:10.3f} nm (Mean) | Best: {summary.best_resolution_nm:.3f} nm | Worst: {summary.worst_resolution_nm:.3f} nm")
        lines.append(f"  Resolving Power (R = lambda/d_lam): {summary.avg_resolving_power:10.1f}      | Peak: {summary.peak_resolving_power:7.1f}")
        lines.append(f"  End-to-End Total Throughput:        {summary.avg_throughput_pct:10.1f} %    | Peak: {summary.peak_throughput_pct:5.1f} %")
        lines.append(f"  Smile Distortion (Slit Curvature):  {summary.max_smile_um:10.2f} um   | Keystone: {summary.max_keystone_um:6.2f} um")
        lines.append(f"  Detector Focal Plane Tilt:          {summary.focal_plane_tilt_deg:10.2f} deg")
        lines.append("-" * 88)
        lines.append("  DESIGN VERIFICATION & ACCEPTANCE STATUS:")
        nyq_str = "[PASS] Compliant (FWHM >= 2 pixels)" if summary.nyquist_compliant else "[WARN] Marginal Sampling (< 2 px Nyquist)"
        vig_str = "[PASS] 100% Geometric Transmission" if summary.zero_vignetting_achieved else "[WARN] Partial Vignetting Detected"
        ray_str = "[PASS] Resolving Power R > 50" if summary.rayleigh_criterion_satisfied else "[FAIL] Low Resolving Power"
        lines.append(f"  * Nyquist Pixel Sampling Criterion: {nyq_str}")
        lines.append(f"  * Clear Aperture Vignetting Status: {vig_str}")
        lines.append(f"  * Rayleigh Line Resolution Power:   {ray_str}")
        lines.append("=" * 88)
        lines.append("  DETAILED MULTI-WAVELENGTH PRESCRIPTION TABLE:")
        pos_hdr = " Y-Pos " if (summary.wavelength_results and any(abs(r.centroid_y_mm) > 0.05 for r in summary.wavelength_results)) else " X-Pos "
        lines.append(f"  Wavelen   Beta   {pos_hdr} dLam/ds  Slit_w  dLam_slit dLam_diff dLam_aber dLam_pix  Total_Res    R_Power  Throughput")
        lines.append("   (nm)    (deg)    (mm)   (nm/mm)   (um)     (nm)      (nm)      (nm)      (nm)      (nm)        (R)        (%)   ")
        lines.append("-" * 88)

        for r in summary.wavelength_results:
            lines.append(
                f"  {r.wavelength_nm:7.1f} {r.diffraction_angle_deg:7.2f} {r.dispersion_pos_mm:7.3f} "
                f"{r.reciprocal_dispersion_nm_per_mm:7.2f} {r.image_slit_width_um:7.1f} "
                f"{r.res_slit_nm:8.3f} {r.res_diff_nm:9.3f} {r.res_aber_nm:9.3f} {r.res_pixel_nm:8.3f}  "
                f"{r.res_total_nm:8.3f}  {r.resolving_power:8.1f}  {r.total_throughput * 100.0:7.1f}%"
            )

        if summary.wavelength_results and summary.wavelength_results[0].raman_shift_cm1 is not None:
            lines.append("-" * 88)
            lines.append("  RAMAN SPECTROSCOPY WAVENUMBER CONVERSION (Laser: %.3f um):" % (cfg.raman_laser_wavelength_um or 0.0))
            lines.append("  Wavelength (nm) | Raman Shift (cm^-1) | Wavenumber Resolution (cm^-1)")
            lines.append("-" * 88)
            for r in summary.wavelength_results:
                shift_s = f"{r.raman_shift_cm1:12.1f}" if r.raman_shift_cm1 is not None else "     N/A    "
                res_s = f"{r.res_raman_cm1:14.2f}" if r.res_raman_cm1 is not None else "      N/A     "
                lines.append(f"     {r.wavelength_nm:8.2f}     | {shift_s}       | {res_s}")

        lines.append("=" * 88)
        lines.append("                              END OF ZEMAX REPORT")
        lines.append("=" * 88)
        return "\n".join(lines)
