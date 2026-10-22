"""Geometric through-focus aberration analysis for a sequential optical system.

Focus is measured along the detector's local +Z axis.  The ray bundle is
traced once to an unlimited virtual detector and then intersected with nearby
parallel planes.  This does not modify the design or claim wavefront/PSF data.
"""

from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import SurfaceType
from optics_app.engine.system_config import SystemConfig
from optics_app.engine.tracer import OpticalSystem


@dataclass(frozen=True)
class FocusRay:
    wavelength_um: float
    x_mm: float
    y_mm: float
    slope_x: float
    slope_y: float
    weight: float = 1.0
    chief: bool = False


@dataclass
class FocusMetrics:
    shift_mm: float
    rms_radius_um: float
    rms_x_um: float
    rms_y_um: float
    geo_radius_um: float
    centroid_x_mm: float
    centroid_y_mm: float


@dataclass
class BestFocusResult:
    valid: bool
    reference: str
    detector_surface_index: int
    reference_wavelength_um: Optional[float] = None
    focus_shifts_mm: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rms_radial_um: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rms_x_um: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rms_y_um: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    per_wavelength_curves_um: Dict[float, np.ndarray] = field(default_factory=dict)
    per_wavelength_best_mm: Dict[float, float] = field(default_factory=dict)
    per_wavelength_rms_um: Dict[float, float] = field(default_factory=dict)
    best_shift_mm: Optional[float] = None
    x_focus_shift_mm: Optional[float] = None
    y_focus_shift_mm: Optional[float] = None
    circle_shift_mm: Optional[float] = None
    nominal: Optional[FocusMetrics] = None
    best: Optional[FocusMetrics] = None
    spots_nominal_mm: Dict[float, np.ndarray] = field(default_factory=dict)
    spots_best_mm: Dict[float, np.ndarray] = field(default_factory=dict)
    launched_by_wavelength: Dict[float, int] = field(default_factory=dict)
    traced_by_wavelength: Dict[float, int] = field(default_factory=dict)
    nominal_detector_hits_by_wavelength: Dict[float, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def longitudinal_color_mm(self) -> Optional[float]:
        values = list(self.per_wavelength_best_mm.values())
        return max(values) - min(values) if len(values) >= 2 else None

    @property
    def xy_astigmatic_separation_mm(self) -> Optional[float]:
        if self.x_focus_shift_mm is None or self.y_focus_shift_mm is None:
            return None
        return self.x_focus_shift_mm - self.y_focus_shift_mm


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _bundle_arrays(rays: Sequence[FocusRay], reference: str, primary_wavelength_um: Optional[float]):
    x = np.array([r.x_mm for r in rays], dtype=float)
    y = np.array([r.y_mm for r in rays], dtype=float)
    u = np.array([r.slope_x for r in rays], dtype=float)
    v = np.array([r.slope_y for r in rays], dtype=float)
    w = np.array([r.weight for r in rays], dtype=float)
    if reference == "centroid":
        xr, yr, ur, vr = (
            _weighted_mean(x, w), _weighted_mean(y, w),
            _weighted_mean(u, w), _weighted_mean(v, w),
        )
    else:
        chiefs = [r for r in rays if r.chief]
        if primary_wavelength_um is not None:
            primary_chiefs = [r for r in chiefs if abs(r.wavelength_um - primary_wavelength_um) < 1e-9]
            if primary_chiefs:
                chiefs = primary_chiefs
        chief = chiefs[0] if chiefs else rays[0]
        xr, yr, ur, vr = chief.x_mm, chief.y_mm, chief.slope_x, chief.slope_y
    return x, y, u, v, w, x - xr, y - yr, u - ur, v - vr


def _best_shift(a: np.ndarray, b: np.ndarray, ua: np.ndarray, ub: np.ndarray, weights: np.ndarray) -> Optional[float]:
    denominator = _weighted_mean(ua * ua + ub * ub, weights)
    if denominator <= 1e-18:
        return None
    numerator = _weighted_mean(a * ua + b * ub, weights)
    return -numerator / denominator


def _metrics(rays: Sequence[FocusRay], shift_mm: float, reference: str, primary_wavelength_um: Optional[float]) -> FocusMetrics:
    x, y, u, v, w, dx, dy, du, dv = _bundle_arrays(rays, reference, primary_wavelength_um)
    ex = dx + shift_mm * du
    ey = dy + shift_mm * dv
    radial_sq = ex * ex + ey * ey
    return FocusMetrics(
        shift_mm=float(shift_mm),
        rms_radius_um=1000.0 * float(np.sqrt(_weighted_mean(radial_sq, w))),
        rms_x_um=1000.0 * float(np.sqrt(_weighted_mean(ex * ex, w))),
        rms_y_um=1000.0 * float(np.sqrt(_weighted_mean(ey * ey, w))),
        geo_radius_um=1000.0 * float(np.sqrt(np.max(radial_sq))),
        centroid_x_mm=_weighted_mean(x + shift_mm * u, w),
        centroid_y_mm=_weighted_mean(y + shift_mm * v, w),
    )


def _equal_width_circle_shift(rays: Sequence[FocusRay], x_focus: Optional[float], y_focus: Optional[float]) -> Optional[float]:
    """Find an equal-RMS-width plane between the X and Y line foci, if one exists."""
    if x_focus is None or y_focus is None:
        return None
    _, _, _, _, w, dx, dy, du, dv = _bundle_arrays(rays, "centroid", None)
    a = _weighted_mean(du * du - dv * dv, w)
    b = 2.0 * _weighted_mean(dx * du - dy * dv, w)
    c = _weighted_mean(dx * dx - dy * dy, w)
    roots: List[float] = []
    if abs(a) <= 1e-15:
        if abs(b) > 1e-15:
            roots = [-c / b]
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0:
            roots = [(-b + np.sqrt(discriminant)) / (2.0 * a),
                     (-b - np.sqrt(discriminant)) / (2.0 * a)]
    lo, hi = sorted((x_focus, y_focus))
    roots = [float(root) for root in roots if np.isfinite(root) and lo - 1e-9 <= root <= hi + 1e-9]
    return min(roots, key=lambda shift: _metrics(rays, shift, "centroid", None).rms_radius_um) if roots else None


def analyze_focus_rays(
    rays: Sequence[FocusRay], *, reference: str = "centroid", primary_wavelength_um: Optional[float] = None,
    half_range_mm: float = 5.0, samples: int = 101, detector_surface_index: int = -1,
) -> BestFocusResult:
    """Analyze an already traced bundle without retracing or moving surfaces."""
    if reference not in ("centroid", "chief"):
        raise ValueError("reference must be 'centroid' or 'chief'")
    if not np.isfinite(half_range_mm) or half_range_mm <= 0 or samples < 3:
        raise ValueError("Focus range must be positive and samples must be at least 3")
    result = BestFocusResult(valid=False, reference=reference, detector_surface_index=detector_surface_index)
    clean = [r for r in rays if np.all(np.isfinite((r.x_mm, r.y_mm, r.slope_x, r.slope_y, r.weight))) and r.weight > 0]
    if len(clean) < 3:
        result.warnings.append("At least three valid detector-plane rays are required for a focus analysis.")
        return result

    x, y, u, v, w, dx, dy, du, dv = _bundle_arrays(clean, reference, primary_wavelength_um)
    best = _best_shift(dx, dy, du, dv, w)
    reference_band = ([r for r in clean if abs(r.wavelength_um - primary_wavelength_um) < 1e-9]
                      if primary_wavelength_um is not None else clean)
    if len(reference_band) < 3:
        reference_band = clean
    result.reference_wavelength_um = reference_band[0].wavelength_um if reference_band else None
    _, _, _, _, rw, rdx, rdy, rdu, rdv = _bundle_arrays(reference_band, reference, result.reference_wavelength_um)
    x_focus = _best_shift(rdx, np.zeros_like(rdy), rdu, np.zeros_like(rdv), rw)
    y_focus = _best_shift(np.zeros_like(rdx), rdy, np.zeros_like(rdu), rdv, rw)
    if best is None:
        result.warnings.append("The sampled bundle is effectively afocal; no finite least-confusion plane exists.")
        return result

    result.valid = True
    result.best_shift_mm = best
    result.x_focus_shift_mm = x_focus
    result.y_focus_shift_mm = y_focus
    result.circle_shift_mm = _equal_width_circle_shift(reference_band, x_focus, y_focus)
    result.nominal = _metrics(clean, 0.0, reference, primary_wavelength_um)
    result.best = _metrics(clean, best, reference, primary_wavelength_um)
    result.focus_shifts_mm = np.linspace(-half_range_mm, half_range_mm, samples)
    result.rms_radial_um = np.array([_metrics(clean, shift, reference, primary_wavelength_um).rms_radius_um for shift in result.focus_shifts_mm])
    result.rms_x_um = np.array([_metrics(clean, shift, reference, primary_wavelength_um).rms_x_um for shift in result.focus_shifts_mm])
    result.rms_y_um = np.array([_metrics(clean, shift, reference, primary_wavelength_um).rms_y_um for shift in result.focus_shifts_mm])
    if abs(best) > half_range_mm:
        result.warnings.append("Best focus lies outside the displayed focus range; increase the scan range.")

    for wl in sorted({r.wavelength_um for r in clean}):
        band = [r for r in clean if r.wavelength_um == wl]
        bx, by, bu, bv, bw, bdx, bdy, bdu, bdv = _bundle_arrays(band, reference, wl)
        band_best = _best_shift(bdx, bdy, bdu, bdv, bw)
        if band_best is not None:
            result.per_wavelength_best_mm[wl] = band_best
            result.per_wavelength_rms_um[wl] = _metrics(band, band_best, reference, wl).rms_radius_um
            if abs(band_best) > half_range_mm:
                result.warnings.append(f"{wl * 1000:.0f} nm best focus lies outside the displayed focus range.")
        result.per_wavelength_curves_um[wl] = np.array([
            _metrics(band, shift, reference, wl).rms_radius_um for shift in result.focus_shifts_mm
        ])
        result.spots_nominal_mm[wl] = np.column_stack((bx, by))
        result.spots_best_mm[wl] = np.column_stack((bx + best * bu, by + best * bv))

    if result.circle_shift_mm is None and x_focus is not None and y_focus is not None:
        result.warnings.append("No equal-RMS-width circle lies between the X and Y foci; the minimum-RMS plane is reported instead.")
    return result


def analyze_system_best_focus(
    system: OpticalSystem, *, system_config: Optional[Any] = None,
    wavelengths_um: Optional[Sequence[float]] = None, field_index: int = 0,
    pupil_samples: int = 11, include_source_extent: bool = False,
    reference: str = "centroid", half_range_mm: float = 5.0, samples: int = 101,
) -> BestFocusResult:
    """Trace a point/slit ray fan and evaluate nearby virtual detector planes."""
    source_stack = system.get_surface_stack()
    source_surfaces = list(system.surfaces)
    detector_idx = next((i for i in range(len(source_surfaces) - 1, -1, -1)
                         if source_surfaces[i].surface_type == SurfaceType.DETECTOR), -1)
    if detector_idx < 0:
        return BestFocusResult(False, reference, -1, warnings=["No detector surface is present."])

    if system_config is None:
        system_config = getattr(source_stack, "system_config", None)
    config = SystemConfig()
    if isinstance(system_config, dict):
        config.from_dict(system_config)
    elif system_config is not None and hasattr(system_config, "to_dict"):
        config.from_dict(system_config.to_dict())
    if not include_source_extent:
        config.slit_width_um = 0.0
    if config.fields:
        if field_index < 0 or field_index >= len(config.fields):
            raise IndexError("field_index is outside the configured field list")
        config.fields = [config.fields[field_index]]

    wavelengths = [float(w) for w in (wavelengths_um or config.wavelength_list_um)]
    if not wavelengths or any(not np.isfinite(w) or w <= 0 for w in wavelengths):
        raise ValueError("At least one positive wavelength is required")
    pupil_samples = min(51, max(5, int(pupil_samples) | 1))  # odd so the chief pupil point is sampled

    virtual_stack: SurfaceStack = deepcopy(source_stack)
    virtual_detector = virtual_stack.surfaces[detector_idx]
    virtual_detector.aperture_type = "Circular"
    virtual_detector.semi_diameter = 1e9
    virtual_detector.aperture_r_outer = 1e9
    virtual_system = OpticalSystem(virtual_stack)
    detector = source_surfaces[detector_idx]
    detector_rotation = np.asarray(detector.global_R, dtype=float)

    # A balanced two-dimensional pupil grid is needed here.  The standard
    # layout generator intentionally favors a tangential fan, which biases a
    # full 2D RMS focus estimate as sample density is increased.
    from optics_app.engine.pupil_solve import calculate_entrance_pupil, aim_ray_to_pupil, aim_collimated_ray
    from optics_app.engine.system_config import ApertureType, FieldType
    surfaces = virtual_system.surfaces
    d0 = float(surfaces[0].thickness)
    r1 = float(surfaces[1].semi_diameter) if len(surfaces) > 1 and surfaces[1].semi_diameter > 0 else 12.7
    is_collimated = (config.beam_type == "Collimated (Parallel Beam)"
                     or config.telecentric_object_space or d0 > 1e4)
    field = config.fields[0] if config.fields else None
    fx = float(field.x) if field else 0.0
    fy = float(field.y) if field else 0.0
    if config.field_type == FieldType.OBJECT_HEIGHT_MM:
        obj_x, obj_y = fx, fy
    elif config.field_type == FieldType.ANGLE_DEG and not is_collimated and d0 > 0:
        obj_x = -d0 * float(np.tan(np.radians(fx)))
        obj_y = -d0 * float(np.tan(np.radians(fy)))
    else:
        obj_x, obj_y = 0.0, 0.0
    slit_half = config.slit_width_um / 2000.0 if include_source_extent and config.slit_width_um > 0 else 0.0
    slit_points = (-slit_half, 0.0, slit_half) if slit_half > 0 else (0.0,)
    grid = np.linspace(-0.85, 0.85, pupil_samples)
    pupil_points = [(float(px), float(py)) for px in grid for py in grid
                    if px * px + py * py <= 0.85 * 0.85 + 1e-12]
    launched = []
    for wl in wavelengths:
        z_ep, r_ep, _ = calculate_entrance_pupil(
            surfaces, wavelength_um=wl,
            default_epd=(config.aperture_value if config.aperture_type == ApertureType.ENTRANCE_PUPIL_DIAMETER
                         else 2.0 * r1),
        )
        for slit_y in slit_points:
            for px, py in pupil_points:
                if is_collimated and config.field_type == FieldType.ANGLE_DEG:
                    launch = aim_collimated_ray(
                        surfaces=surfaces, field_angle_deg=(fx, fy), pupil_norm=(px, py),
                        wavelength_um=wl, z_ep_global=z_ep, r_ep=r_ep,
                    )
                else:
                    launch = aim_ray_to_pupil(
                        surfaces=surfaces, obj_point=(obj_x, obj_y + slit_y, 0.0),
                        pupil_norm=(px, py), wavelength_um=wl, z_ep_global=z_ep,
                        r_ep=r_ep, system=virtual_system,
                    )
                launched.append((launch, abs(px) < 1e-12 and abs(py) < 1e-12 and abs(slit_y) < 1e-12))
    planned_count = Counter(float(ray.wavelength) for ray, _ in launched)
    wavelength_weights = {float(entry.wavelength_um): float(entry.weight) for entry in config.wavelengths}
    samples_out: List[FocusRay] = []
    launched_count: Dict[float, int] = {w: 0 for w in wavelengths}
    traced_count: Dict[float, int] = {w: 0 for w in wavelengths}
    nominal_hits: Dict[float, int] = {w: 0 for w in wavelengths}
    for launch, is_chief in launched:
        wl = float(launch.wavelength)
        launched_count[wl] += 1
        traced = virtual_system.trace_ray(launch)
        hit = next((h for h in reversed(traced.history) if h["surface_index"] == detector_idx), None)
        if hit is None or traced.vignetted:
            continue
        direction = detector_rotation.T @ np.asarray(hit["dir"], dtype=float)
        if abs(direction[2]) < 1e-10:
            continue
        hx, hy = float(hit["x_local"]), float(hit["y_local"])
        traced_count[wl] += 1
        if detector.check_aperture(hx, hy):
            nominal_hits[wl] += 1
        samples_out.append(FocusRay(
            wavelength_um=wl, x_mm=hx, y_mm=hy,
            slope_x=float(direction[0] / direction[2]),
            slope_y=float(direction[1] / direction[2]),
            weight=wavelength_weights.get(wl, 1.0) / max(planned_count[wl], 1),
            chief=is_chief,
        ))

    primary = float(getattr(source_stack, "primary_wavelength", wavelengths[len(wavelengths) // 2]))
    result = analyze_focus_rays(
        samples_out, reference=reference, primary_wavelength_um=primary,
        half_range_mm=half_range_mm, samples=samples, detector_surface_index=detector_idx,
    )
    result.launched_by_wavelength = launched_count
    result.traced_by_wavelength = traced_count
    result.nominal_detector_hits_by_wavelength = nominal_hits
    for wl in wavelengths:
        if traced_count[wl] < launched_count[wl]:
            result.warnings.append(
                f"{wl * 1000:.0f} nm: only {traced_count[wl]}/{launched_count[wl]} rays reached the virtual detector; pupil clipping may bias the focus result."
            )
        if nominal_hits[wl] < traced_count[wl]:
            result.warnings.append(
                f"{wl * 1000:.0f} nm: {traced_count[wl] - nominal_hits[wl]} rays miss the physical detector aperture at its current plane."
            )
    if include_source_extent and config.slit_width_um > 0:
        result.warnings.append("Source/slit width is included, so the spot contains object-image blur as well as aberrations.")
    elif not include_source_extent and system_config is not None:
        original_slit = (system_config.get("aperture", {}).get("slit_width_um", 0.0)
                         if isinstance(system_config, dict)
                         else getattr(system_config, "slit_width_um", 0.0))
        if float(original_slit) > 0:
            result.warnings.append(f"The configured {float(original_slit):g} um slit is treated as a point for aberration isolation.")
    return result
