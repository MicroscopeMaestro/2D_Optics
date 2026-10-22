"""
Laser Scanning System Simulation & Field Aberrations Engine.
Simulates galvanometer scanning mirrors, laser beam propagation, and field-dependent
aberrations across the scan field (RMS spot radius, Petzval field curvature, astigmatism,
and f*theta scan linearity distortion).
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.tracer import OpticalSystem, build_rotation_matrix
from optics_app.engine.mechanisms import (
    MechanicalStateController,
    active_state,
    available_states,
    is_dmd,
)


def is_galvo(surface: Any) -> bool:
    """Return True if surface is designated as a galvanometer scanning mirror."""
    name_upper = str(getattr(surface, "name", "")).upper()
    mech_type = str(getattr(surface, "mechanism_type", "")).upper()
    stype = str(getattr(surface, "surface_type", "")).upper()
    return (
        "GALVO" in name_upper
        or "SCANNER" in name_upper
        or "GALVANOMETER" in name_upper
        or mech_type in ("GALVO", "SCANNER", "GALVANOMETER")
        or "GALVO" in stype
    )


@dataclass
class GalvoScannerConfig:
    """Configuration for a Galvanometer Scanning Mirror actuator."""
    surface_index: int = 1
    scan_axis: str = "X"                       # "X", "Y", or "XY"
    min_angle_deg: float = -10.0               # Mechanical minimum scan angle (deg)
    max_angle_deg: float = 10.0                # Mechanical maximum scan angle (deg)
    step_size_deg: float = 1.0                 # Scan step size (deg)
    current_angle_deg: float = 0.0             # Live mechanical angle (deg)
    optical_deflection_factor: float = 2.0     # Optical angle = factor * mechanical angle (2.0 for reflection)
    mirror_separation_mm: float = 10.0         # Separation between X and Y galvos if dual-axis
    clear_aperture_mm: float = 10.0

    @property
    def optical_min_deg(self) -> float:
        return self.min_angle_deg * self.optical_deflection_factor

    @property
    def optical_max_deg(self) -> float:
        return self.max_angle_deg * self.optical_deflection_factor

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


@dataclass
class LaserBeamConfig:
    """Input Laser Beam parameters for scan field simulation."""
    wavelength_um: float = 0.532               # Default green laser line (532 nm)
    beam_diameter_mm: float = 5.0              # 1/e^2 or clear beam aperture diameter (mm)
    num_pupil_rings: int = 3                   # Concentric pupil rings for bundle sampling
    rays_per_ring: int = 8                     # Ray azimuths per ring (plus chief ray)
    numerical_aperture: Optional[float] = None # Optional input NA

    @property
    def beam_radius_mm(self) -> float:
        return self.beam_diameter_mm / 2.0


@dataclass
class ScanPointResult:
    """Detailed optical performance and aberration metrics at a single scan field point."""
    scan_angle_x_deg: float                    # Mechanical scan angle X (deg)
    scan_angle_y_deg: float                    # Mechanical scan angle Y (deg)
    optical_angle_x_deg: float                 # Reflected optical scan angle X (deg)
    optical_angle_y_deg: float                 # Reflected optical scan angle Y (deg)
    total_optical_angle_deg: float             # Total optical off-axis angle sqrt(theta_x^2 + theta_y^2)

    # Image Plane Coordinates
    chief_ray_x_mm: float                      # Spot centroid / chief ray intercept X (mm)
    chief_ray_y_mm: float                      # Spot centroid / chief ray intercept Y (mm)
    field_radius_mm: float                     # Radial image height on detector (mm)

    # Spot Quality Metrics
    rms_spot_radius_um: float                  # Geometric RMS spot radius (um)
    rms_x_um: float                            # RMS spot width X (um)
    rms_y_um: float                            # RMS spot height Y (um)
    geo_spot_radius_um: float                  # 100% enclosed geometric spot radius (um)
    airy_radius_um: float                      # Diffraction-limited Airy disk radius (um)
    is_diffraction_limited: bool               # True if RMS <= Airy disk radius
    strehl_ratio_estimate: float               # Approximate Strehl ratio [0.0, 1.0]

    # Field Aberrations
    f_theta_ideal_radius_mm: float             # Target f*theta scan height = f * theta_rad (mm)
    f_theta_error_percent: float               # Linearity distortion error (%) = (actual - ideal)/ideal * 100
    tangential_focus_shift_mm: float           # Longitudinal focus shift for tangential fan (mm)
    sagittal_focus_shift_mm: float             # Longitudinal focus shift for sagittal fan (mm)
    astigmatism_mm: float                      # Longitudinal astigmatic difference Delta_zT - Delta_zS (mm)
    field_curvature_mm: float                  # Mean Petzval field curvature (Delta_zT + Delta_zS) / 2 (mm)
    spot_ellipticity: float                    # Ellipticity ratio max(rms_x, rms_y) / min(rms_x, rms_y)

    # Transmission & Vignetting
    launched_rays: int
    transmitted_rays: int
    vignetting_factor: float                   # 0.0 = unvignetted, 1.0 = fully clipped


@dataclass
class LaserScanSimulationResult:
    """Complete laser scanning system field aberration evaluation across the scan field."""
    scan_axis: str
    scan_points: List[ScanPointResult]
    scan_lens_efl_mm: float
    wavelength_um: float
    beam_diameter_mm: float

    # Extracted 1D Arrays for plotting and analysis
    angles_deg: np.ndarray = field(init=False)
    optical_angles_deg: np.ndarray = field(init=False)
    field_heights_mm: np.ndarray = field(init=False)
    rms_spots_um: np.ndarray = field(init=False)
    airy_radii_um: np.ndarray = field(init=False)
    tangential_foci_mm: np.ndarray = field(init=False)
    sagittal_foci_mm: np.ndarray = field(init=False)
    astigmatisms_mm: np.ndarray = field(init=False)
    field_curvatures_mm: np.ndarray = field(init=False)
    f_theta_errors_percent: np.ndarray = field(init=False)

    def __post_init__(self):
        n = len(self.scan_points)
        self.angles_deg = np.array([p.scan_angle_x_deg if self.scan_axis == "X" else p.scan_angle_y_deg for p in self.scan_points], dtype=float) if n else np.array([])
        self.optical_angles_deg = np.array([p.total_optical_angle_deg for p in self.scan_points], dtype=float) if n else np.array([])
        self.field_heights_mm = np.array([p.field_radius_mm for p in self.scan_points], dtype=float) if n else np.array([])
        self.rms_spots_um = np.array([p.rms_spot_radius_um for p in self.scan_points], dtype=float) if n else np.array([])
        self.airy_radii_um = np.array([p.airy_radius_um for p in self.scan_points], dtype=float) if n else np.array([])
        self.tangential_foci_mm = np.array([p.tangential_focus_shift_mm for p in self.scan_points], dtype=float) if n else np.array([])
        self.sagittal_foci_mm = np.array([p.sagittal_focus_shift_mm for p in self.scan_points], dtype=float) if n else np.array([])
        self.astigmatisms_mm = np.array([p.astigmatism_mm for p in self.scan_points], dtype=float) if n else np.array([])
        self.field_curvatures_mm = np.array([p.field_curvature_mm for p in self.scan_points], dtype=float) if n else np.array([])
        self.f_theta_errors_percent = np.array([p.f_theta_error_percent for p in self.scan_points], dtype=float) if n else np.array([])

    @property
    def max_rms_spot_um(self) -> float:
        return float(np.max(self.rms_spots_um)) if len(self.rms_spots_um) else 0.0

    @property
    def min_rms_spot_um(self) -> float:
        return float(np.min(self.rms_spots_um)) if len(self.rms_spots_um) else 0.0

    @property
    def mean_rms_spot_um(self) -> float:
        return float(np.mean(self.rms_spots_um)) if len(self.rms_spots_um) else 0.0

    @property
    def max_astigmatism_mm(self) -> float:
        return float(np.max(np.abs(self.astigmatisms_mm))) if len(self.astigmatisms_mm) else 0.0

    @property
    def field_curvature_span_mm(self) -> float:
        if not len(self.field_curvatures_mm):
            return 0.0
        return float(np.max(self.field_curvatures_mm) - np.min(self.field_curvatures_mm))

    @property
    def max_f_theta_error_percent(self) -> float:
        if not len(self.f_theta_errors_percent):
            return 0.0
        return float(np.max(np.abs(self.f_theta_errors_percent)))

    @property
    def scan_field_diameter_mm(self) -> float:
        return float(2.0 * np.max(self.field_heights_mm)) if len(self.field_heights_mm) else 0.0

    @property
    def diffraction_limited_fraction(self) -> float:
        if not len(self.scan_points):
            return 0.0
        diff_lim_count = sum(1 for p in self.scan_points if p.is_diffraction_limited)
        return float(diff_lim_count) / len(self.scan_points)


class LaserScannerSimulator:
    """
    Precision Laser Scanning System Simulator.
    Simulates galvanometer actuators and computes field aberrations across arbitrary scan fields.
    """

    def __init__(self, stack: Any):
        self.stack = stack

    def find_galvo_surface_index(self) -> Optional[int]:
        """Automatically identify the galvanometer mirror surface in the system."""
        surfaces = getattr(self.stack, "surfaces", [])
        # 1. Explicit Galvo check
        for i, s in enumerate(surfaces):
            if is_galvo(s):
                return i
        # 2. Reflective mirror surface that is not a grating
        for i, s in enumerate(surfaces):
            gl = str(getattr(s, "glass", "")).upper()
            st = getattr(s, "surface_type", None)
            if (st == SurfaceType.MIRROR or gl in ("MIRROR", "MIRR", "REFL")) and st != SurfaceType.GRATING:
                return i
        # 3. Stop plane (often scanner pupil)
        for i, s in enumerate(surfaces):
            if getattr(s, "surface_type", None) == SurfaceType.STOP or getattr(s, "is_stop", False):
                return i
        # Default fallback to surface 1 if available
        return 1 if len(surfaces) > 1 else None

    def estimate_scan_lens_efl(self, galvo_idx: int) -> float:
        """
        Estimate the effective focal length (EFL) of the scan lens assembly
        downstream of the galvanometer scanner.
        """
        surfaces = getattr(self.stack, "surfaces", [])
        if not surfaces or galvo_idx >= len(surfaces) - 1:
            return 100.0

        # Method 1: Trace small paraxial chief rays deflected by small angles +/- 0.5 deg
        delta_deg = 0.5
        trial = deepcopy(self.stack)
        system = OpticalSystem(trial)
        det_idx = len(trial.surfaces) - 1

        # Trace zero angle chief ray
        r0 = Ray(pos=np.array([0.0, 0.0, 0.0], dtype=float), dir=np.array([0.0, 0.0, 1.0], dtype=float), wavelength=0.532)
        t0 = system.trace_ray(r0)
        p0 = t0.pos if not t0.vignetted else np.array([0.0, 0.0, 0.0])

        # Temporarily tilt galvo by delta_deg
        surf_g = trial.surfaces[galvo_idx]
        orig_tx = float(getattr(surf_g, "tilt_x", 0.0))
        surf_g.tilt_x = orig_tx + delta_deg
        trial._update_accumulated_positions()
        system_tilted = OpticalSystem(trial)

        r_tilted = Ray(pos=np.array([0.0, 0.0, 0.0], dtype=float), dir=np.array([0.0, 0.0, 1.0], dtype=float), wavelength=0.532)
        t_tilted = system_tilted.trace_ray(r_tilted)

        if not t_tilted.vignetted and not t0.vignetted:
            dy = float(np.linalg.norm(t_tilted.pos[:2] - p0[:2]))
            theta_opt_rad = np.radians(2.0 * delta_deg) # 2x mechanical angle for reflection
            if theta_opt_rad > 1e-6:
                efl_est = dy / theta_opt_rad
                if 5.0 < efl_est < 5000.0:
                    return float(efl_est)

        # Method 2: Distance from galvo to detector
        p_galvo = surfaces[galvo_idx].global_origin
        p_det = surfaces[-1].global_origin
        dist = float(np.linalg.norm(p_det - p_galvo))
        return max(10.0, dist)

    def generate_collimated_laser_bundle(
        self,
        beam_config: LaserBeamConfig,
        origin_z: float = 0.0,
    ) -> List[Ray]:
        """
        Generate a collimated circular laser bundle with concentric rings.
        """
        rays: List[Ray] = []
        wl = beam_config.wavelength_um
        r_max = beam_config.beam_radius_mm

        # 1. Chief ray at center
        rays.append(Ray(pos=np.array([0.0, 0.0, origin_z], dtype=float), dir=np.array([0.0, 0.0, 1.0], dtype=float), wavelength=wl, intensity=1.0))

        # 2. Concentric rings
        num_rings = max(1, beam_config.num_pupil_rings)
        base_azimuths = max(4, beam_config.rays_per_ring)

        for ring_idx in range(1, num_rings + 1):
            r = (ring_idx / num_rings) * r_max
            n_azimuth = base_azimuths * ring_idx
            for az_idx in range(n_azimuth):
                phi = 2.0 * np.pi * az_idx / n_azimuth
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                # Gaussian apodization weight: exp(-2*(r/w0)^2)
                intensity = float(np.exp(-2.0 * (r / max(1e-6, r_max))**2))
                rays.append(Ray(
                    pos=np.array([x, y, origin_z], dtype=float),
                    dir=np.array([0.0, 0.0, 1.0], dtype=float),
                    wavelength=wl,
                    intensity=intensity
                ))

        return rays

    def evaluate_scan_point(
        self,
        galvo_idx: int,
        angle_x_deg: float,
        angle_y_deg: float,
        beam_config: LaserBeamConfig,
        efl_mm: float,
        target_surface_idx: int = -1,
    ) -> ScanPointResult:
        """
        Evaluate optical performance and field aberrations at a specific (angle_x, angle_y) scan point.
        """
        trial = deepcopy(self.stack)
        if not (0 <= galvo_idx < len(trial.surfaces)):
            raise IndexError(f"Galvo surface index {galvo_idx} is out of bounds.")

        surf_g = trial.surfaces[galvo_idx]
        base_tx = float(getattr(surf_g, "tilt_x", 0.0))
        base_ty = float(getattr(surf_g, "tilt_y", 0.0))

        # Apply mechanical galvo tilts
        surf_g.tilt_x = base_tx + float(angle_y_deg)
        surf_g.tilt_y = base_ty + float(angle_x_deg)

        trial._update_accumulated_positions()
        system = OpticalSystem(trial)
        det_idx = target_surface_idx if target_surface_idx >= 0 else len(trial.surfaces) - 1

        # Generate collimated laser ray bundle
        # Origin slightly upstream of surface 0
        z_start = float(trial.surfaces[0].global_origin[2]) - 1.0
        bundle = self.generate_collimated_laser_bundle(beam_config, origin_z=z_start)

        # Trace ray bundle
        traced_bundle = [system.trace_ray(r) for r in bundle]

        # Extract ray intercepts on detector plane
        hits_x: List[float] = []
        hits_y: List[float] = []
        slopes_x: List[float] = []
        slopes_y: List[float] = []
        weights: List[float] = []

        chief_hit_x = 0.0
        chief_hit_y = 0.0

        for idx, r in enumerate(traced_bundle):
            if r.vignetted:
                continue
            # Extract hit coordinates from history on detector
            det_hits = [h for h in r.history if int(h.get("surface_index", -1)) == det_idx]
            if not det_hits:
                # Use terminal ray position
                hx = float(r.pos[0])
                hy = float(r.pos[1])
                kz = float(r.dir[2]) if abs(r.dir[2]) > 1e-9 else 1.0
                kx = float(r.dir[0]) / kz
                ky = float(r.dir[1]) / kz
            else:
                h_rec = det_hits[-1]
                hx = float(h_rec.get("x_global", r.pos[0]))
                hy = float(h_rec.get("y_global", r.pos[1]))
                kz = float(r.dir[2]) if abs(r.dir[2]) > 1e-9 else 1.0
                kx = float(r.dir[0]) / kz
                ky = float(r.dir[1]) / kz

            hits_x.append(hx)
            hits_y.append(hy)
            slopes_x.append(kx)
            slopes_y.append(ky)
            weights.append(float(getattr(r, "intensity", 1.0)))

            if idx == 0:
                chief_hit_x = hx
                chief_hit_y = hy

        launched = len(bundle)
        transmitted = len(hits_x)
        vignetting = (1.0 - (transmitted / launched)) if launched > 0 else 1.0

        opt_angle_x = float(angle_x_deg * 2.0)
        opt_angle_y = float(angle_y_deg * 2.0)
        total_opt_angle = float(math.sqrt(opt_angle_x**2 + opt_angle_y**2))

        # Compute spot statistics
        if transmitted >= 2:
            hx_arr = np.array(hits_x)
            hy_arr = np.array(hits_y)
            w_arr = np.array(weights)
            w_sum = np.sum(w_arr)
            if w_sum > 0:
                cx = float(np.sum(hx_arr * w_arr) / w_sum)
                cy = float(np.sum(hy_arr * w_arr) / w_sum)
            else:
                cx = float(np.mean(hx_arr))
                cy = float(np.mean(hy_arr))

            dx = hx_arr - cx
            dy = hy_arr - cy
            r2 = dx**2 + dy**2
            rms_spot_um = float(math.sqrt(np.mean(r2)) * 1000.0) # mm -> um
            rms_x_um = float(math.sqrt(np.mean(dx**2)) * 1000.0)
            rms_y_um = float(math.sqrt(np.mean(dy**2)) * 1000.0)
            geo_spot_um = float(np.max(np.sqrt(r2)) * 1000.0)

            # Focus shift / astigmatism via covariance minimization
            # Delta_z_tangential = -Cov(y, ky) / Var(ky)
            # Delta_z_sagittal   = -Cov(x, kx) / Var(kx)
            kx_arr = np.array(slopes_x)
            ky_arr = np.array(slopes_y)

            var_kx = float(np.var(kx_arr))
            var_ky = float(np.var(ky_arr))

            if var_ky > 1e-9:
                cov_y_ky = float(np.mean(dy * (ky_arr - np.mean(ky_arr))))
                focus_shift_T = -cov_y_ky / var_ky
            else:
                focus_shift_T = 0.0

            if var_kx > 1e-9:
                cov_x_kx = float(np.mean(dx * (kx_arr - np.mean(kx_arr))))
                focus_shift_S = -cov_x_kx / var_kx
            else:
                focus_shift_S = 0.0

            astigmatism_mm = float(focus_shift_T - focus_shift_S)
            field_curv_mm = float((focus_shift_T + focus_shift_S) / 2.0)
            ellipticity = float(max(rms_x_um, rms_y_um) / max(1e-6, min(rms_x_um, rms_y_um)))

        else:
            cx = chief_hit_x
            cy = chief_hit_y
            rms_spot_um = 0.0
            rms_x_um = 0.0
            rms_y_um = 0.0
            geo_spot_um = 0.0
            focus_shift_T = 0.0
            focus_shift_S = 0.0
            astigmatism_mm = 0.0
            field_curv_mm = 0.0
            ellipticity = 1.0

        # Airy disk radius: r_Airy = 1.22 * lambda * f / D
        airy_radius_um = float(1.22 * beam_config.wavelength_um * efl_mm / max(1e-3, beam_config.beam_diameter_mm))
        is_diff_lim = (rms_spot_um <= airy_radius_um) and (transmitted > 0)
        # Strehl estimate via Maréchal approx: S ~ exp(-(2*pi*sigma/lambda)^2)
        sigma_um = rms_spot_um / 3.4
        strehl_est = float(max(0.01, min(1.0, math.exp(-(2.0 * math.pi * sigma_um / beam_config.wavelength_um)**2))))

        field_radius_mm = float(math.sqrt(cx**2 + cy**2))

        # f*theta distortion error calculation
        # Ideal height = f * theta_opt_rad
        theta_rad = math.radians(total_opt_angle)
        f_theta_ideal = float(efl_mm * theta_rad)
        if theta_rad > 1e-5 and f_theta_ideal > 1e-6:
            f_theta_error_pct = float(((field_radius_mm - f_theta_ideal) / f_theta_ideal) * 100.0)
        else:
            f_theta_error_pct = 0.0

        return ScanPointResult(
            scan_angle_x_deg=float(angle_x_deg),
            scan_angle_y_deg=float(angle_y_deg),
            optical_angle_x_deg=opt_angle_x,
            optical_angle_y_deg=opt_angle_y,
            total_optical_angle_deg=total_opt_angle,
            chief_ray_x_mm=float(cx),
            chief_ray_y_mm=float(cy),
            field_radius_mm=field_radius_mm,
            rms_spot_radius_um=rms_spot_um,
            rms_x_um=rms_x_um,
            rms_y_um=rms_y_um,
            geo_spot_radius_um=geo_spot_um,
            airy_radius_um=airy_radius_um,
            is_diffraction_limited=is_diff_lim,
            strehl_ratio_estimate=strehl_est,
            f_theta_ideal_radius_mm=f_theta_ideal,
            f_theta_error_percent=f_theta_error_pct,
            tangential_focus_shift_mm=float(focus_shift_T),
            sagittal_focus_shift_mm=float(focus_shift_S),
            astigmatism_mm=astigmatism_mm,
            field_curvature_mm=field_curv_mm,
            spot_ellipticity=ellipticity,
            launched_rays=launched,
            transmitted_rays=transmitted,
            vignetting_factor=float(vignetting),
        )

    def simulate_1d_scan(
        self,
        galvo_idx: Optional[int] = None,
        scan_axis: str = "X",
        min_angle_deg: float = -10.0,
        max_angle_deg: float = 10.0,
        step_size_deg: float = 1.0,
        beam_config: Optional[LaserBeamConfig] = None,
        target_surface_idx: int = -1,
    ) -> LaserScanSimulationResult:
        """
        Perform a 1D sweep across galvanometer mechanical angles and compute field aberrations.
        """
        if galvo_idx is None:
            galvo_idx = self.find_galvo_surface_index()
            if galvo_idx is None:
                raise ValueError("No galvanometer or scanning surface found in the optical system.")

        if beam_config is None:
            beam_config = LaserBeamConfig()

        efl = self.estimate_scan_lens_efl(galvo_idx)
        config = GalvoScannerConfig(
            surface_index=galvo_idx,
            scan_axis=scan_axis,
            min_angle_deg=min_angle_deg,
            max_angle_deg=max_angle_deg,
            step_size_deg=step_size_deg
        )
        angles = config.get_mechanical_angles()

        points: List[ScanPointResult] = []
        for ang in angles:
            if scan_axis.upper() == "Y":
                pt = self.evaluate_scan_point(
                    galvo_idx=galvo_idx,
                    angle_x_deg=0.0,
                    angle_y_deg=ang,
                    beam_config=beam_config,
                    efl_mm=efl,
                    target_surface_idx=target_surface_idx
                )
            else:
                pt = self.evaluate_scan_point(
                    galvo_idx=galvo_idx,
                    angle_x_deg=ang,
                    angle_y_deg=0.0,
                    beam_config=beam_config,
                    efl_mm=efl,
                    target_surface_idx=target_surface_idx
                )
            points.append(pt)

        return LaserScanSimulationResult(
            scan_axis=scan_axis.upper(),
            scan_points=points,
            scan_lens_efl_mm=efl,
            wavelength_um=beam_config.wavelength_um,
            beam_diameter_mm=beam_config.beam_diameter_mm
        )

    def simulate_2d_raster_scan(
        self,
        galvo_idx: Optional[int] = None,
        angle_range_x: Tuple[float, float] = (-8.0, 8.0),
        angle_range_y: Tuple[float, float] = (-8.0, 8.0),
        step_size_deg: float = 2.0,
        beam_config: Optional[LaserBeamConfig] = None,
        target_surface_idx: int = -1,
    ) -> LaserScanSimulationResult:
        """
        Perform a 2D raster grid scan across (theta_x, theta_y) to evaluate 2D scan field distortion and spot size.
        """
        if galvo_idx is None:
            galvo_idx = self.find_galvo_surface_index()
            if galvo_idx is None:
                raise ValueError("No galvanometer or scanning surface found in the optical system.")

        if beam_config is None:
            beam_config = LaserBeamConfig()

        efl = self.estimate_scan_lens_efl(galvo_idx)
        step = max(0.1, abs(step_size_deg))
        nx = max(3, int(round((angle_range_x[1] - angle_range_x[0]) / step)) + 1)
        ny = max(3, int(round((angle_range_y[1] - angle_range_y[0]) / step)) + 1)

        x_angles = np.linspace(angle_range_x[0], angle_range_x[1], nx)
        y_angles = np.linspace(angle_range_y[0], angle_range_y[1], ny)

        points: List[ScanPointResult] = []
        for y_ang in y_angles:
            for x_ang in x_angles:
                pt = self.evaluate_scan_point(
                    galvo_idx=galvo_idx,
                    angle_x_deg=float(x_ang),
                    angle_y_deg=float(y_ang),
                    beam_config=beam_config,
                    efl_mm=efl,
                    target_surface_idx=target_surface_idx
                )
                points.append(pt)

        return LaserScanSimulationResult(
            scan_axis="XY",
            scan_points=points,
            scan_lens_efl_mm=efl,
            wavelength_um=beam_config.wavelength_um,
            beam_diameter_mm=beam_config.beam_diameter_mm
        )


def configure_galvo_scanner(
    surface: Surface,
    scan_axis: str = "X",
    min_angle_deg: float = -10.0,
    max_angle_deg: float = 10.0,
    step_size_deg: float = 1.0,
) -> GalvoScannerConfig:
    """
    Attach galvanometer scanner properties and mechanical actuation metadata to an optical surface.
    """
    surface.mechanism_type = "GALVO"
    if not is_galvo(surface):
        surface.name = f"{surface.name} (Galvo {scan_axis})"

    config = GalvoScannerConfig(
        scan_axis=scan_axis,
        min_angle_deg=min_angle_deg,
        max_angle_deg=max_angle_deg,
        step_size_deg=step_size_deg
    )
    setattr(surface, "galvo_config", config)

    # Register in mechanical_states
    surface.mechanical_states = {
        "CENTER": {
            "label": "Center (0°)",
            "optical_tilt_offset_deg": [0.0, 0.0, 0.0],
            "route": "axial_path",
            "description": "On-axis central scan position."
        },
        "MIN": {
            "label": f"Min Scan ({min_angle_deg:.1f}°)",
            "optical_tilt_offset_deg": [0.0, float(min_angle_deg * 2.0), 0.0] if scan_axis == "X" else [float(min_angle_deg * 2.0), 0.0, 0.0],
            "route": "negative_scan",
            "description": "Negative scan field limit."
        },
        "MAX": {
            "label": f"Max Scan ({max_angle_deg:.1f}°)",
            "optical_tilt_offset_deg": [0.0, float(max_angle_deg * 2.0), 0.0] if scan_axis == "X" else [float(max_angle_deg * 2.0), 0.0, 0.0],
            "route": "positive_scan",
            "description": "Positive scan field limit."
        }
    }
    surface.mechanical_state = "CENTER"
    return config


__all__ = [
    "is_galvo",
    "GalvoScannerConfig",
    "LaserBeamConfig",
    "ScanPointResult",
    "LaserScanSimulationResult",
    "LaserScannerSimulator",
    "configure_galvo_scanner",
]
