"""
Grating Dispersion Analysis Subsystem.
Provides spatial dispersion, angular/linear dispersion, spectral resolution metrics, and CCD sensor integration.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from optics_app.engine.grating import DiffractionGrating
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.ccd import CCDSensor
from optics_app.analysis.spot import SpotDiagram


@dataclass
class WavelengthSpotResult:
    """Diagnostic result for a single wavelength component in a dispersion analysis."""
    wavelength_um: float
    diffraction_angle_rad: Optional[float]
    centroid_x_mm: float
    centroid_y_mm: float
    delta_x_mm: float
    delta_y_mm: float
    delta_r_mm: float
    rms_spot_radius_mm: float
    geo_spot_radius_mm: float
    airy_radius_um: float
    vignetted_count: int
    total_rays: int
    is_evanescent: bool = False


@dataclass
class DispersionMetrics:
    """Summary dispersion and resolution metrics for a grating optical system."""
    reference_wavelength_um: float
    angular_dispersion_rad_per_um: float
    angular_dispersion_rad_per_nm: float
    linear_dispersion_mm_per_um: float
    linear_dispersion_mm_per_nm: float
    reciprocal_linear_dispersion_nm_per_mm: float
    theoretical_resolving_power: float
    diffraction_resolution_nm: float
    instrumental_resolution_nm: float
    pixel_limited_resolution_nm: Optional[float]
    effective_resolution_nm: float
    effective_resolving_power: float
    total_spectral_span_mm: float


@dataclass
class GratingDispersionResult:
    """Complete container for multi-wavelength grating dispersion analysis."""
    metrics: DispersionMetrics
    spots: List[WavelengthSpotResult]
    wavelengths_um: np.ndarray
    delta_x_array: np.ndarray
    delta_y_array: np.ndarray
    linear_dispersion_curve: np.ndarray        # dx/dlambda vs lambda
    reciprocal_dispersion_curve: np.ndarray   # dlambda/dx vs lambda


class GratingDispersionAnalysis:
    """
    Grating Dispersion Analysis Subsystem.
    Provides analytical and ray-trace based spatial dispersion, angular/linear dispersion,
    spectral resolution metrics, and CCD image sensor integration.
    """

    @staticmethod
    def calculate_spatial_dispersion(
        grating: DiffractionGrating,
        wavelengths_um: List[float],
        focal_length_mm: float = 100.0,
        incident_angle_rad: float = 0.0,
        reference_wavelength_um: Optional[float] = None
    ) -> List[Tuple[float, Optional[float], Optional[float]]]:
        """
        Analytical 1D spatial dispersion calculator.
        Returns list of tuples: (wavelength_um, theta_m_rad, x_pos_mm).
        """
        spots = []
        for wl in wavelengths_um:
            try:
                theta_m = grating.calculate_diffraction_angle(incident_angle_rad, wl)
                x_pos = focal_length_mm * np.tan(theta_m)
                spots.append((float(wl), float(theta_m), float(x_pos)))
            except ValueError:
                spots.append((float(wl), None, None))
        return spots

    @staticmethod
    def compute_dispersion_metrics(
        grating_lines_per_mm: float,
        order: int,
        diffraction_angle_rad: float,
        focal_length_mm: float,
        wavelength_um: float,
        beam_diameter_mm: float = 20.0,
        rms_spot_radius_mm: float = 0.0,
        pixel_pitch_mm: Optional[float] = None
    ) -> DispersionMetrics:
        """
        Compute theoretical and instrumental dispersion and spectral resolution metrics.
        """
        cos_theta = np.cos(diffraction_angle_rad)
        if grating_lines_per_mm == 0.0 or order == 0:
            d_theta_d_lambda_um = 0.0
        elif abs(cos_theta) > 1e-6:
            pitch_um = 1000.0 / grating_lines_per_mm
            d_theta_d_lambda_um = float(order / (pitch_um * cos_theta))
        else:
            d_theta_d_lambda_um = 1e9

        d_theta_d_lambda_nm = d_theta_d_lambda_um / 1000.0
        dx_d_lambda_um = focal_length_mm * d_theta_d_lambda_um
        dx_d_lambda_nm = dx_d_lambda_um / 1000.0

        reciprocal_disp_nm_mm = (1.0 / dx_d_lambda_nm) if abs(dx_d_lambda_nm) > 1e-12 else 1e9

        theoretical_R = float(abs(order) * grating_lines_per_mm * beam_diameter_mm)
        diffraction_res_nm = (wavelength_um * 1000.0 / theoretical_R) if theoretical_R > 0 else 0.0

        instrumental_res_nm = (2.0 * rms_spot_radius_mm / dx_d_lambda_nm) if abs(dx_d_lambda_nm) > 1e-12 else 0.0

        if pixel_pitch_mm is not None and abs(dx_d_lambda_nm) > 1e-12:
            pixel_res_nm = float(2.0 * pixel_pitch_mm / dx_d_lambda_nm)
        else:
            pixel_res_nm = None

        eff_res_components = [diffraction_res_nm, abs(instrumental_res_nm)]
        if pixel_res_nm is not None:
            eff_res_components.append(abs(pixel_res_nm))
        effective_res_nm = float(max(eff_res_components))

        effective_R = (wavelength_um * 1000.0 / effective_res_nm) if effective_res_nm > 0 else 0.0

        return DispersionMetrics(
            reference_wavelength_um=float(wavelength_um),
            angular_dispersion_rad_per_um=float(d_theta_d_lambda_um),
            angular_dispersion_rad_per_nm=float(d_theta_d_lambda_nm),
            linear_dispersion_mm_per_um=float(dx_d_lambda_um),
            linear_dispersion_mm_per_nm=float(dx_d_lambda_nm),
            reciprocal_linear_dispersion_nm_per_mm=float(reciprocal_disp_nm_mm),
            theoretical_resolving_power=float(theoretical_R),
            diffraction_resolution_nm=float(diffraction_res_nm),
            instrumental_resolution_nm=float(instrumental_res_nm),
            pixel_limited_resolution_nm=pixel_res_nm,
            effective_resolution_nm=float(effective_res_nm),
            effective_resolving_power=float(effective_R),
            total_spectral_span_mm=0.0
        )

    @classmethod
    def analyze_system_dispersion(
        cls,
        system: OpticalSystem,
        wavelengths_um: List[float],
        num_rays_per_wavelength: int = 101,
        reference_wavelength_um: Optional[float] = None,
        pupil_radius_mm: float = 10.0,
        ccd_sensor: Optional[CCDSensor] = None,
        target_surface_index: int = -1,
        source_config: Optional[Any] = None
    ) -> GratingDispersionResult:
        """
        Full 3D sequential ray-trace spatial dispersion analysis across an OpticalSystem.
        Traces polychromatic ray bundles through LDE stack, computes 2D centroids,
        spatial separation (delta_x, delta_y), dispersion metrics, and populates CCD sensor.
        """
        if not wavelengths_um:
            wavelengths_um = [0.5876]

        sorted_wls = sorted(wavelengths_um)
        if reference_wavelength_um is None:
            reference_wavelength_um = sorted_wls[len(sorted_wls) // 2]

        # Find grating parameters from system stack if available
        grating_g = 600.0
        grating_order = 1
        surfaces = getattr(system, "surfaces", None) or (system.stack.surfaces if hasattr(system, "stack") else [])
        focal_length = 100.0
        for s in surfaces:
            if getattr(s, "surface_type", None) == 5: # SurfaceType.GRATING = 5
                grating_g = getattr(s, "grating_lines_per_mm", grating_g)
                grating_order = getattr(s, "grating_order", grating_order)
            elif getattr(s, "surface_type", None) == 1 and getattr(s, "radius", 0) > 0: # Focusing lens approx
                focal_length = getattr(s, "radius", focal_length) * 2.0 # rough paraxial approx
                
        if ccd_sensor:
            ccd_sensor.clear()

        n_rings = max(1, int(np.sqrt(num_rays_per_wavelength / np.pi)) + 1)
        
        cfg = source_config
        from optics_app.engine.source import SourceType
        
        def generate_ray_bundle(wl: float) -> List[Ray]:
            bundle = []
            if cfg is not None and getattr(cfg, 'source_type', None) == SourceType.FIBER:
                max_angle = np.radians(cfg.divergence_angle)
                spatial_rings = max(1, int(np.sqrt(n_rings)))
                angular_rings = max(1, int(np.sqrt(n_rings)))
                bundle.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                for sr in range(1, spatial_rings + 1):
                    s_norm = sr / spatial_rings
                    s_pts = max(4, int(4 * sr))
                    for sp in range(s_pts):
                        stheta = 2.0 * np.pi * sp / s_pts
                        px = s_norm * np.cos(stheta) * cfg.beam_radius
                        py = s_norm * np.sin(stheta) * cfg.beam_radius
                        bundle.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                        for ar in range(1, angular_rings + 1):
                            a_norm = ar / angular_rings
                            a_pts = max(4, int(4 * ar))
                            for ap in range(a_pts):
                                atheta = 2.0 * np.pi * ap / a_pts
                                dx = a_norm * np.sin(max_angle) * np.cos(atheta)
                                dy = a_norm * np.sin(max_angle) * np.sin(atheta)
                                dz = np.sqrt(1.0 - dx**2 - dy**2)
                                bundle.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wl))
            elif cfg is not None and getattr(cfg, 'source_type', None) == SourceType.POINT:
                max_angle = np.radians(cfg.divergence_angle)
                bundle.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                for ring in range(1, n_rings + 1):
                    r_norm = ring / n_rings
                    n_pts = max(6, int(6 * ring))
                    for p in range(n_pts):
                        theta = 2.0 * np.pi * p / n_pts
                        dx = r_norm * np.sin(max_angle) * np.cos(theta)
                        dy = r_norm * np.sin(max_angle) * np.sin(theta)
                        dz = np.sqrt(1.0 - dx**2 - dy**2)
                        bundle.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wl))
            else:
                bundle.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                for ring in range(1, n_rings + 1):
                    r_norm = ring / n_rings
                    n_pts = max(6, int(6 * ring))
                    for p in range(n_pts):
                        theta = 2.0 * np.pi * p / n_pts
                        rad = cfg.beam_radius if (cfg is not None and getattr(cfg, 'source_type', None) == SourceType.COLLIMATED) else pupil_radius_mm
                        px = r_norm * np.cos(theta) * rad
                        py = r_norm * np.sin(theta) * rad
                        bundle.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            return bundle

        spot_results_dict: Dict[float, Tuple[np.ndarray, np.ndarray, int, int, bool]] = {}

        for wl in sorted_wls:
            ray_list = generate_ray_bundle(wl)
            traced = system.trace_rays(ray_list)
            x_hits = []
            y_hits = []
            vign_cnt = 0
            for r in traced:
                if len(r.history) == 0:
                    vign_cnt += 1
                    continue
                idx = target_surface_index if target_surface_index != -1 else len(r.history) - 1
                if idx >= len(r.history) or (r.vignetted and idx >= len(r.history) - 1):
                    vign_cnt += 1
                else:
                    last_hit = r.history[idx]
                    x_hits.append(last_hit["x"])
                    y_hits.append(last_hit["y"])

            x_arr = np.array(x_hits, dtype=float)
            y_arr = np.array(y_hits, dtype=float)
            is_evan = (len(x_hits) == 0)
            spot_results_dict[wl] = (x_arr, y_arr, vign_cnt, len(ray_list), is_evan)

            if ccd_sensor is not None and len(x_arr) > 0:
                ccd_sensor.deposit_rays(x_arr, y_arr)

        # Get reference centroid
        ref_x, ref_y = 0.0, 0.0
        if reference_wavelength_um in spot_results_dict:
            rx_arr, ry_arr, _, _, _ = spot_results_dict[reference_wavelength_um]
            if len(rx_arr) > 0:
                ref_x, ref_y = float(np.mean(rx_arr)), float(np.mean(ry_arr))

        spot_list: List[WavelengthSpotResult] = []
        delta_x_list = []
        delta_y_list = []

        for wl in sorted_wls:
            x_arr, y_arr, vign_cnt, tot_cnt, is_evan = spot_results_dict[wl]
            if len(x_arr) > 0:
                cx, cy = float(np.mean(x_arr)), float(np.mean(y_arr))
                dx = cx - ref_x
                dy = cy - ref_y
                dr = float(np.sqrt(dx**2 + dy**2))
                spot_diag = SpotDiagram(x_arr, y_arr, wavelength_um=wl, f_number=4.0)
                rms_r = spot_diag.calculate_rms_radius()
                geo_r = spot_diag.calculate_geo_radius()
                airy_r = spot_diag.calculate_airy_disk_radius()
            else:
                cx, cy, dx, dy, dr = 0.0, 0.0, 0.0, 0.0, 0.0
                rms_r, geo_r, airy_r = 0.0, 0.0, 0.0

            delta_x_list.append(dx)
            delta_y_list.append(dy)

            grating_obj = DiffractionGrating(lines_per_mm=grating_g, order=grating_order)
            try:
                diff_angle = grating_obj.calculate_diffraction_angle(0.0, wl)
            except ValueError:
                diff_angle = None

            spot_list.append(WavelengthSpotResult(
                wavelength_um=float(wl),
                diffraction_angle_rad=diff_angle,
                centroid_x_mm=cx,
                centroid_y_mm=cy,
                delta_x_mm=dx,
                delta_y_mm=dy,
                delta_r_mm=dr,
                rms_spot_radius_mm=rms_r,
                geo_spot_radius_mm=geo_r,
                airy_radius_um=airy_r,
                vignetted_count=vign_cnt,
                total_rays=tot_cnt,
                is_evanescent=is_evan
            ))

        # Linear dispersion curve finite differences
        wls_array = np.array(sorted_wls, dtype=float)
        dx_array = np.array(delta_x_list, dtype=float)
        dy_array = np.array(delta_y_list, dtype=float)

        dx_span = float(np.max(dx_array) - np.min(dx_array)) if len(dx_array) > 0 else 0.0
        dy_span = float(np.max(dy_array) - np.min(dy_array)) if len(dy_array) > 0 else 0.0
        total_span = max(dx_span, dy_span)
        disp_array = dy_array if dy_span >= dx_span else dx_array

        lin_disp_curve = np.zeros_like(wls_array)
        if len(wls_array) > 1:
            lin_disp_curve = np.gradient(disp_array, wls_array)  # mm/um

        safe_lin_disp = np.where(np.abs(lin_disp_curve) > 1e-12, lin_disp_curve / 1000.0, 1.0)
        recip_disp_curve = np.where(np.abs(lin_disp_curve) > 1e-12, 1.0 / safe_lin_disp, 1e9)  # nm/mm
        avg_rms = float(np.mean([s.rms_spot_radius_mm for s in spot_list])) if spot_list else 0.0

        pix_pitch = ccd_sensor.pixel_pitch_x if ccd_sensor is not None else None

        metrics = cls.compute_dispersion_metrics(
            grating_lines_per_mm=grating_g,
            order=grating_order,
            diffraction_angle_rad=spot_list[len(spot_list)//2].diffraction_angle_rad or 0.0,
            focal_length_mm=focal_length,
            wavelength_um=reference_wavelength_um,
            beam_diameter_mm=2.0 * pupil_radius_mm,
            rms_spot_radius_mm=avg_rms,
            pixel_pitch_mm=pix_pitch
        )
        metrics.total_spectral_span_mm = total_span

        return GratingDispersionResult(
            metrics=metrics,
            spots=spot_list,
            wavelengths_um=wls_array,
            delta_x_array=dx_array,
            delta_y_array=dy_array,
            linear_dispersion_curve=lin_disp_curve,
            reciprocal_dispersion_curve=recip_disp_curve
        )

    @staticmethod
    def deposit_dispersion_to_ccd(
        analysis_result: GratingDispersionResult,
        ccd_sensor: CCDSensor,
        rays_per_spot: Dict[float, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Deposit multi-wavelength spot intersection arrays into CCD matrix.
        """
        for wl, (x_arr, y_arr) in rays_per_spot.items():
            ccd_sensor.deposit_rays(x_arr, y_arr)
