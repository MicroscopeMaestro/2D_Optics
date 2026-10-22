"""
Pre-Flight Diagnostic & Parameter Validation Engine for Optical Optimization.
Audits optical systems, solver configurations, variable parameters, physical bounds,
and ray transmission before running optimization to prevent numerical divergence,
unhandled exceptions, or unphysical geometries. Provides automated one-click repair (auto_fix).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from collections import Counter
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic findings."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DiagnosticItem:
    """Represents a specific diagnostic issue found during pre-flight check."""
    code: str
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING
    surface_index: Optional[int] = None
    parameter_name: Optional[str] = None
    suggested_fix: Optional[str] = None
    auto_fix_fn: Optional[Callable[[], None]] = None

    def __str__(self) -> str:
        loc = f" (Surface #{self.surface_index})" if self.surface_index is not None else ""
        return f"[{self.severity.name}]{loc} {self.message}"


@dataclass
class ValidationResult:
    """Structured report returned by the pre-flight validator."""
    is_valid: bool = True
    items: List[DiagnosticItem] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(item.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL) for item in self.items)

    @property
    def has_warnings(self) -> bool:
        return any(item.severity == DiagnosticSeverity.WARNING for item in self.items)

    @property
    def errors(self) -> List[str]:
        return [item.message for item in self.items if item.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)]

    @property
    def warnings(self) -> List[str]:
        return [item.message for item in self.items if item.severity == DiagnosticSeverity.WARNING]

    @property
    def auto_fix_available(self) -> bool:
        return any(item.auto_fix_fn is not None for item in self.items)

    def summary(self) -> str:
        """Human-readable multi-line diagnostic summary."""
        if not self.items:
            return "Pre-flight check passed: System configuration is 100% valid."
        lines = []
        if self.has_errors:
            lines.append(f"CRITICAL ERRORS ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  • {e}")
        if self.has_warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  • {w}")
        return "\n".join(lines)


class ModelSanityValidator:
    """Fast, non-mutating checks for suspicious optical model geometry and traces."""

    LARGE_REFRACTIVE_TILT_DEG = 20.0
    LENS_GROUP_TILT_MISMATCH_DEG = 5.0
    LOW_TRANSMISSION_FRACTION = 0.10

    @staticmethod
    def _is_coordinate_break(surface: Surface) -> bool:
        return bool(
            getattr(surface, "is_coord_break", False)
            or getattr(surface, "surface_type", None) == SurfaceType.COORDINATE_BREAK
        )

    @staticmethod
    def _is_reflective(surface: Surface) -> bool:
        if getattr(surface, "surface_type", None) == SurfaceType.GRATING:
            return True
        if hasattr(surface, "is_mirror") and surface.is_mirror():
            return True
        if hasattr(surface, "is_reflective"):
            return bool(surface.is_reflective())
        glass = str(getattr(surface, "glass", "") or "").upper()
        return glass in ("MIRROR", "MIRR", "REFL")

    @classmethod
    def inspect_geometry(cls, stack: SurfaceStack) -> ValidationResult:
        """Detect uncommon geometry that is often caused by an accidental UI edit."""
        items: List[DiagnosticItem] = []
        surfaces = list(getattr(stack, "surfaces", []) or [])

        from optics_app.engine.mechanisms import active_state_name, available_states, is_dmd
        calibration = getattr(stack, "spectrometer_calibration", None)
        has_physical_dump = bool(calibration.get("physical_beam_dump_modeled")) if isinstance(calibration, dict) else False
        for idx, surf in enumerate(surfaces):
            if is_dmd(surf) and available_states(surf) and active_state_name(surf) == "OFF":
                items.append(DiagnosticItem(
                    code="DMD_OFF_DETECTOR_DARK",
                    message=(
                        f"Surface #{idx} ('{surf.name}') is in the OFF state. Its micromirrors route light "
                        + ("toward the modeled beam dump" if has_physical_dump else "away from the detector arm; no physical beam dump is modeled")
                        + ", so no detector signal is expected."
                    ),
                    severity=DiagnosticSeverity.WARNING,
                    surface_index=idx,
                    parameter_name="mechanical_state",
                    suggested_fix="Switch the DMD to ON when simulating or measuring the detector path."
                ))

        for idx, surf in enumerate(surfaces):
            if cls._is_coordinate_break(surf) or cls._is_reflective(surf):
                continue
            if getattr(surf, "surface_type", None) not in (SurfaceType.STANDARD, SurfaceType.ASPHERIC):
                continue

            tilts = (
                float(getattr(surf, "tilt_x", 0.0)),
                float(getattr(surf, "tilt_y", 0.0)),
                float(getattr(surf, "tilt_z", 0.0)),
            )
            max_tilt = max(abs(v) for v in tilts)
            if max_tilt >= cls.LARGE_REFRACTIVE_TILT_DEG:
                axis_index = max(range(3), key=lambda i: abs(tilts[i]))
                axis = ("X", "Y", "Z")[axis_index]
                items.append(DiagnosticItem(
                    code="UNCOMMON_REFRACTIVE_SURFACE_TILT",
                    message=(
                        f"Surface #{idx} ('{surf.name}') is refractive but has a {tilts[axis_index]:.3g} deg "
                        f"Tilt {axis}. Large single-surface lens tilts commonly indicate that a mirror-fold preset was applied to a lens by mistake."
                    ),
                    severity=DiagnosticSeverity.WARNING,
                    surface_index=idx,
                    parameter_name=f"tilt_{axis.lower()}",
                    suggested_fix="Reset the surface tilt or rotate the complete lens element as a group."
                ))

        groups: Dict[str, List[Tuple[int, Surface]]] = {}
        for idx, surf in enumerate(surfaces):
            group_name = str(getattr(surf, "group_name", "") or "").strip()
            if group_name:
                groups.setdefault(group_name, []).append((idx, surf))

        for group_name, members in groups.items():
            if len(members) < 2:
                continue
            tilt_vectors = np.array([
                [float(getattr(s, "tilt_x", 0.0)), float(getattr(s, "tilt_y", 0.0)), float(getattr(s, "tilt_z", 0.0))]
                for _, s in members
            ], dtype=float)
            spread = np.ptp(tilt_vectors, axis=0)
            max_spread = float(np.max(np.abs(spread)))
            if max_spread >= cls.LENS_GROUP_TILT_MISMATCH_DEG:
                member_text = ", ".join(f"#{idx} '{surf.name}'" for idx, surf in members)
                items.append(DiagnosticItem(
                    code="LENS_GROUP_TILT_MISMATCH",
                    message=(
                        f"Lens group '{group_name}' has inconsistent surface tilts (maximum difference {max_spread:.3g} deg): "
                        f"{member_text}. A rigid lens element normally shares one orientation."
                    ),
                    severity=DiagnosticSeverity.WARNING,
                    surface_index=members[0][0],
                    parameter_name="tilt",
                    suggested_fix="Rotate the complete lens element as a group, or reset the unintended surface tilt."
                ))

        if isinstance(calibration, dict):
            layout_wavelength = calibration.get("layout_reference_wavelength_um")
            for idx, surf in enumerate(surfaces):
                if surf.surface_type == SurfaceType.GRATING and layout_wavelength is not None:
                    actual = getattr(surf, "grating_primary_wavelength_um", None)
                    if actual is None or abs(float(actual) - float(layout_wavelength)) > 1e-6:
                        items.append(DiagnosticItem(
                            code="GRATING_LAYOUT_REFERENCE_MISMATCH",
                            message=f"Grating #{idx} is folded at {actual} um, but this Raman layout is referenced to {layout_wavelength} um.",
                            surface_index=idx,
                            parameter_name="grating_primary_wavelength_um",
                            suggested_fix="Set the grating layout wavelength to the Raman-band reference and retrace the entire band.",
                        ))
                if not is_dmd(surf):
                    continue
                meta = getattr(surf, "catalog_metadata", {}) or {}
                active_size = meta.get("active_array_size_mm") if isinstance(meta, dict) else None
                if isinstance(active_size, (list, tuple)) and len(active_size) == 2:
                    # This project's long active-array axis is local Y.
                    axis = str(meta.get("long_axis_local", "y")).strip().lower()[:1]
                    physical_x = float(active_size[1] if axis == "y" else active_size[0])
                    physical_y = float(active_size[0] if axis == "y" else active_size[1])
                    if (float(surf.aperture_x_half) > physical_x / 2.0 + 1e-6
                            or float(surf.aperture_y_half) > physical_y / 2.0 + 1e-6):
                        items.append(DiagnosticItem(
                            code="DMD_ACTIVE_APERTURE_OVERSIZED",
                            message=f"DMD #{idx} clear aperture exceeds its catalog active array ({physical_x:.4g} x {physical_y:.4g} mm).",
                            surface_index=idx,
                            parameter_name="aperture_x_half/aperture_y_half",
                            suggested_fix="Use the active-array dimensions, not the package or mount dimensions.",
                        ))

            if not calibration.get("quantitative_throughput_calibrated", False):
                items.append(DiagnosticItem(
                    code="SPECTROMETER_THROUGHPUT_UNCALIBRATED",
                    message="Raman detector flux is uncalibrated: measured grating efficiency and DMD wavelength-dependent throughput are missing.",
                    suggested_fix="Load measured efficiency and throughput curves before interpreting ray intensity as detector power.",
                ))
            if not has_physical_dump:
                items.append(DiagnosticItem(
                    code="DMD_PHYSICAL_DUMP_UNMODELED",
                    message="DMD OFF rays miss the detector arm, but no physical beam dump or power rating is modeled.",
                    suggested_fix="Place and rate a physical dump on the rejected path before claiming it absorbs the beam.",
                ))

        return ValidationResult(is_valid=not any(i.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL) for i in items), items=items)

    @classmethod
    def inspect_paraxial_first_order(cls, stack: SurfaceStack, wavelength_um: float = 0.5876, config: Optional[Any] = None) -> Tuple[ValidationResult, Dict[str, Any]]:
        """
        Inspect first-order paraxial optical parameters: EFL, BFL, FFL, Back Pupil, Principal Planes, etc.
        Returns (validation_result, first_order_data_dict).
        """
        try:
            from optics_app.engine.first_order import calculate_first_order_data
        except ImportError:
            from v2.engine.first_order import calculate_first_order_data

        surfaces = getattr(stack, "surfaces", []) if stack else []
        data = calculate_first_order_data(surfaces, wavelength_um=wavelength_um, config=config)
        items: List[DiagnosticItem] = []

        effl = data.get("effl", float("inf"))
        bfl = data.get("bfl", float("inf"))
        is_afocal = data.get("is_afocal", False)
        f_num = data.get("f_number", float("inf"))
        expd = data.get("expd", 0.0)
        exp_pos = data.get("exp_pos", 0.0)
        p1 = data.get("p1", 0.0)
        p2 = data.get("p2", 0.0)

        # Check virtual focal point
        if not is_afocal and bfl < 0.0:
            items.append(DiagnosticItem(
                code="PARAXIAL_VIRTUAL_IMAGE",
                message=f"Back Focal Length is negative (BFL = {bfl:.2f} mm). Paraxial focus lies inside or in front of the rear element (virtual image).",
                severity=DiagnosticSeverity.WARNING,
                suggested_fix="Increase positive power or adjust lens spacings to move focal point behind rear surface."
            ))

        # Check extreme F/#
        if not is_afocal and f_num < 0.7:
            items.append(DiagnosticItem(
                code="PARAXIAL_EXTREME_FAST_FNUM",
                message=f"F-Number is extremely fast (F/{f_num:.2f} < 0.70). High aperture aberrations are expected.",
                severity=DiagnosticSeverity.INFO,
                suggested_fix="Reduce entrance pupil diameter or increase effective focal length."
            ))

        # Check buried exit pupil
        if not is_afocal and exp_pos < -data.get("lens_thickness", 10.0):
            items.append(DiagnosticItem(
                code="PARAXIAL_BURIED_BACK_PUPIL",
                message=f"Exit pupil / Back pupil is buried far inside optical element (ExP Pos = {exp_pos:.2f} mm).",
                severity=DiagnosticSeverity.INFO,
                suggested_fix="Adjust aperture stop position to bring exit pupil into accessible space if telecentricity is required."
            ))

        is_valid = not any(i.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL) for i in items)
        return ValidationResult(is_valid=is_valid, items=items), data

    @classmethod
    def inspect_trace(cls, stack: SurfaceStack, traced_rays: List[Ray]) -> ValidationResult:
        """Summarize whether rays reach the detector and identify the dominant blocker."""
        items: List[DiagnosticItem] = []
        surfaces = list(stack.get_active_sequence() if hasattr(stack, "get_active_sequence") else stack.surfaces)
        if not surfaces or not traced_rays:
            return ValidationResult(is_valid=True, items=[])

        detector_idx = next(
            (idx for idx, surf in enumerate(surfaces) if getattr(surf, "surface_type", None) == SurfaceType.DETECTOR),
            len(surfaces) - 1,
        )
        reached = 0
        blockers: Counter = Counter()
        missed_intersections: Counter = Counter()
        for ray in traced_rays:
            history = list(getattr(ray, "history", []) or [])
            if any(
                h.get("surface_index") == detector_idx
                and not bool(h.get("vignetted", False))
                and float(h.get("intensity", 0.0)) > 0.0
                for h in history
            ):
                reached += 1
                continue
            if not history:
                blockers[0] += 1
                continue
            last = history[-1]
            last_idx = int(last.get("surface_index", -1))
            if last.get("failure_reason") == "no_forward_intersection":
                missed_intersections[last_idx] += 1
            blocker_idx = last_idx if last.get("vignetted", False) else last_idx + 1
            blockers[max(0, min(blocker_idx, len(surfaces) - 1))] += 1

        total = len(traced_rays)
        transmission = reached / total
        if reached == total:
            return ValidationResult(is_valid=True, items=[])

        from optics_app.engine.mechanisms import active_state_name, available_states, is_dmd
        off_dmd = next(
            ((idx, surf) for idx, surf in enumerate(surfaces)
             if is_dmd(surf) and available_states(surf) and active_state_name(surf) == "OFF"),
            None,
        )
        if missed_intersections and off_dmd is None:
            miss_idx, miss_count = missed_intersections.most_common(1)[0]
            items.append(DiagnosticItem(
                code="NO_FORWARD_SURFACE_INTERSECTION",
                message=(f"{miss_count}/{total} rays have no forward intersection with Surface #{miss_idx} "
                         f"('{surfaces[miss_idx].name}'). Check the reflected direction, tilt and spacing; "
                         "the tracer will not move rays backward to reach a plane."),
                severity=DiagnosticSeverity.WARNING,
                surface_index=miss_idx,
                suggested_fix="Use Air Space for spacing-only rows and position physical surfaces on the outgoing optical path.",
            ))

        blocker_idx = blockers.most_common(1)[0][0] if blockers else None
        blocker_name = surfaces[blocker_idx].name if blocker_idx is not None else "Unknown"
        blocked = total - reached
        if reached == 0:
            if off_dmd is not None:
                dmd_idx, dmd = off_dmd
                items.append(DiagnosticItem(
                    code="DMD_OFF_NO_DETECTOR_SIGNAL_EXPECTED",
                    message=(
                        f"No rays reached the detector because Surface #{dmd_idx} ('{dmd.name}') is OFF "
                        "and steers the reflected beam away from the detector arm."
                    ),
                    severity=DiagnosticSeverity.WARNING,
                    surface_index=dmd_idx,
                    parameter_name="mechanical_state",
                    suggested_fix="Switch the DMD to ON to illuminate the detector arm."
                ))
                return ValidationResult(is_valid=True, items=items)
            items.append(DiagnosticItem(
                code="NO_RAYS_REACH_DETECTOR",
                message=(
                    f"No rays reached detector Surface #{detector_idx}; {blocked}/{total} rays were blocked. "
                    f"The dominant failure is Surface #{blocker_idx} ('{blocker_name}'). Check recent tilt, decenter, aperture, and spacing edits."
                ),
                severity=DiagnosticSeverity.ERROR,
                surface_index=blocker_idx,
                suggested_fix="Inspect the dominant blocking surface and the last upstream element for accidental geometry changes."
            ))
        elif transmission < cls.LOW_TRANSMISSION_FRACTION:
            items.append(DiagnosticItem(
                code="VERY_LOW_RAY_TRANSMISSION",
                message=(
                    f"Only {reached}/{total} rays ({100.0 * transmission:.1f}%) reached detector Surface #{detector_idx}. "
                    f"The dominant blocker is Surface #{blocker_idx} ('{blocker_name}')."
                ),
                severity=DiagnosticSeverity.WARNING,
                surface_index=blocker_idx,
                suggested_fix="Check apertures and recent tilt, decenter, or spacing edits near the blocking surface."
            ))

        return ValidationResult(is_valid=not any(i.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL) for i in items), items=items)


class OptimizationValidator:
    """
    Comprehensive Pre-Flight Integrity Auditor for Sequential Optical Systems and Optimizers.
    """

    @classmethod
    def validate_system(
        cls,
        stack: SurfaceStack,
        variable_indices: Optional[Union[dict, List[int]]] = None,
        wavelengths: Union[float, List[float], np.ndarray] = 0.5876,
        entrance_pupil_radius: float = 5.0,
        num_rays: int = 21,
        target: str = "Spot Size",
        target_spread: float = 12.0,
        target_surface_index: Optional[int] = None,
        target_spot_size: float = 0.0,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_iter: int = 100,
        opt_tilt: bool = False
    ) -> ValidationResult:
        """
        Execute comprehensive audit of system geometry, solver parameters, and ray tracing.
        Returns ValidationResult with detailed items and auto-fix closures.
        """
        items: List[DiagnosticItem] = []

        # Normalize variables
        if variable_indices is None:
            var_dict = {"thickness": [], "radius": []}
            for i, s in enumerate(stack.surfaces):
                if getattr(s, 'thickness_variable', False):
                    var_dict["thickness"].append(i)
                if getattr(s, 'radius_variable', False):
                    var_dict["radius"].append(i)
                if getattr(s, 'tilt_x_variable', False):
                    var_dict.setdefault("tilt_x", []).append(i)
                if getattr(s, 'tilt_y_variable', False):
                    var_dict.setdefault("tilt_y", []).append(i)
                if getattr(s, 'tilt_z_variable', False):
                    var_dict.setdefault("tilt_z", []).append(i)
                if getattr(s, 'decenter_x_variable', False):
                    var_dict.setdefault("decenter_x", []).append(i)
                if getattr(s, 'decenter_y_variable', False):
                    var_dict.setdefault("decenter_y", []).append(i)
        elif isinstance(variable_indices, dict):
            var_dict = {k: list(v) for k, v in variable_indices.items()}
        else:
            var_dict = {"thickness": list(variable_indices), "radius": []}

        # Normalize wavelengths
        if isinstance(wavelengths, (int, float)):
            wls = [float(wavelengths)]
        elif isinstance(wavelengths, np.ndarray):
            wls = [float(w) for w in wavelengths.tolist()]
        else:
            wls = [float(w) for w in wavelengths]

        # -------------------------------------------------------------
        # 1. Check Variable Degrees of Freedom
        # -------------------------------------------------------------
        n_total_vars = sum(len(v) for v in var_dict.values())
        if n_total_vars == 0 and not opt_tilt:
            # Recover from surface-level flags if present on stack
            for idx, surf in enumerate(stack.surfaces[:-1]):
                if getattr(surf, "thickness_variable", False) and idx not in var_dict.get("thickness", []):
                    var_dict.setdefault("thickness", []).append(idx)
                if getattr(surf, "radius_variable", False) and idx not in var_dict.get("radius", []):
                    var_dict.setdefault("radius", []).append(idx)
            n_total_vars = sum(len(v) for v in var_dict.values())

        if n_total_vars == 0 and not opt_tilt:
            def fix_zero_vars():
                # Apply Zemax default: Optical air gaps variable, glass & object fixed
                var_dict["thickness"] = []
                for idx, surf in enumerate(stack.surfaces[:-1]):
                    if idx == 0 and getattr(surf, "surface_type", None) == SurfaceType.OBJECT and len(stack.surfaces) > 2:
                        continue
                    glass_name = getattr(surf, "glass", "AIR") or "AIR"
                    if glass_name.upper() == "AIR":
                        var_dict["thickness"].append(idx)
                        setattr(surf, "thickness_variable", True)

            items.append(DiagnosticItem(
                code="NO_VARIABLES_SELECTED",
                message="No parameters are marked as variable. Optimization cannot improve merit function without degrees of freedom.",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix="Apply Zemax Defaults: Mark air gaps Variable (V) and solid lenses Fixed (F).",
                auto_fix_fn=fix_zero_vars
            ))

        # Check for out-of-bounds variable indices
        num_surfs = len(stack.surfaces)
        for attr, indices in var_dict.items():
            for idx in list(indices):
                if idx < 0 or idx >= num_surfs:
                    items.append(DiagnosticItem(
                        code="INVALID_VARIABLE_INDEX",
                        message=f"{attr.capitalize()} variable surface index #{idx} is out of range [0, {num_surfs - 1}].",
                        severity=DiagnosticSeverity.ERROR,
                        surface_index=idx
                    ))

        # Check if detector distance is fixed when optimizing for Focus
        if ("Focus" in target or "Spot" in target) and num_surfs > 1:
            det_idx = None
            for idx, s in enumerate(stack.surfaces):
                if s.surface_type == SurfaceType.DETECTOR:
                    det_idx = idx
                    break
            if det_idx is None:
                det_idx = num_surfs - 1
            if det_idx > 0:
                pre_det_idx = det_idx - 1
                pre_surf = stack.surfaces[pre_det_idx]
                thick_vars = var_dict.get("thickness", [])
                if pre_det_idx not in thick_vars and not getattr(pre_surf, 'thickness_variable', False):
                    def make_fix_float_bfl(s=pre_surf, p_idx=pre_det_idx):
                        def fix():
                            s.thickness_variable = True
                            if "thickness" not in var_dict:
                                var_dict["thickness"] = []
                            if p_idx not in var_dict["thickness"]:
                                var_dict["thickness"].append(p_idx)
                        return fix

                    items.append(DiagnosticItem(
                        code="FIXED_DETECTOR_DISTANCE",
                        message=f"Surface #{pre_det_idx} ('{pre_surf.name}') distance to detector is fixed ({pre_surf.thickness:.2f} mm). Optimizing focus spot without floating the back focal distance may prevent converging to a sharp focus on the detector.",
                        severity=DiagnosticSeverity.INFO,
                        surface_index=pre_det_idx,
                        parameter_name="thickness",
                        suggested_fix="Enable thickness variable (V) on the rear surface to allow the lens to focus on the detector.",
                        auto_fix_fn=make_fix_float_bfl()
                    ))

        # -------------------------------------------------------------
        # 2. Check Variable Bounds & Ranges
        # -------------------------------------------------------------
        if bounds:
            for b_idx, (b_min, b_max) in enumerate(bounds):
                if b_min >= b_max:
                    items.append(DiagnosticItem(
                        code="INVERTED_BOUNDS",
                        message=f"Variable #{b_idx} has inverted bounds: minimum ({b_min:.2f}) >= maximum ({b_max:.2f}).",
                        severity=DiagnosticSeverity.ERROR,
                        suggested_fix="Swap min and max bounds to ensure b_min < b_max."
                    ))

        # -------------------------------------------------------------
        # 3. Check Physical Thickness & Lens Geometry
        # -------------------------------------------------------------
        for idx, surf in enumerate(stack.surfaces):
            glass_name = getattr(surf, "glass", "AIR") or "AIR"
            is_glass = glass_name.upper() not in ("AIR", "MIRROR", "MIRR", "")
            
            # Check for negative center thicknesses
            if surf.thickness < 0 and idx < num_surfs - 1:
                def make_fix_thickness(s=surf, is_g=is_glass):
                    def fix():
                        s.thickness = 3.0 if is_g else 10.0
                    return fix

                items.append(DiagnosticItem(
                    code="NEGATIVE_THICKNESS",
                    message=f"Surface #{idx} ('{surf.name}') has negative thickness ({surf.thickness:.2f} mm).",
                    severity=DiagnosticSeverity.ERROR if is_glass else DiagnosticSeverity.WARNING,
                    surface_index=idx,
                    parameter_name="thickness",
                    suggested_fix=f"Set thickness to positive value (>= 1.0 mm for lenses, >= 0.0 mm for air).",
                    auto_fix_fn=make_fix_thickness()
                ))
            elif surf.thickness == 0 and is_glass and idx < num_surfs - 1:
                items.append(DiagnosticItem(
                    code="ZERO_LENS_THICKNESS",
                    message=f"Solid glass lens Surface #{idx} ('{surf.name}') has zero center thickness.",
                    severity=DiagnosticSeverity.WARNING,
                    surface_index=idx,
                    parameter_name="thickness",
                    suggested_fix="Set glass center thickness to at least 1.5 - 3.0 mm for manufacturability."
                ))

            # Check for near-zero radius of curvature singularities
            r = abs(getattr(surf, "radius", 0.0))
            if 0 < r < 0.1:
                items.append(DiagnosticItem(
                    code="NEAR_ZERO_RADIUS",
                    message=f"Surface #{idx} ('{surf.name}') has extremely small radius of curvature ({surf.radius:.4f} mm), which may cause ray-trace divergence.",
                    severity=DiagnosticSeverity.WARNING,
                    surface_index=idx,
                    parameter_name="radius",
                    suggested_fix="Set radius = 0.0 (Plano) or increase radius magnitude."
                ))

        # -------------------------------------------------------------
        # 4. Check Entrance Pupil & Sampling Parameters
        # -------------------------------------------------------------
        if entrance_pupil_radius <= 0:
            items.append(DiagnosticItem(
                code="INVALID_PUPIL_RADIUS",
                message=f"Entrance pupil semi-diameter must be strictly positive (got {entrance_pupil_radius:.2f} mm).",
                severity=DiagnosticSeverity.ERROR,
                parameter_name="entrance_pupil_radius",
                suggested_fix="Set entrance pupil radius >= 1.0 mm."
            ))

        if num_rays < 3:
            items.append(DiagnosticItem(
                code="INSUFFICIENT_RAY_SAMPLING",
                message=f"Ray count must be at least 3 rays to compute 2D/3D RMS spot radii (got {num_rays}).",
                severity=DiagnosticSeverity.WARNING,
                parameter_name="num_rays",
                suggested_fix="Increase ray count to 21 or 37 for accurate spot diagram optimization."
            ))

        if max_iter <= 0:
            items.append(DiagnosticItem(
                code="INVALID_MAX_ITERATIONS",
                message=f"Maximum iterations must be at least 1 (got {max_iter}).",
                severity=DiagnosticSeverity.ERROR,
                parameter_name="max_iter",
                suggested_fix="Set max_iter = 100."
            ))

        # -------------------------------------------------------------
        # 5. Check Target Plane & Evaluation Surface
        # -------------------------------------------------------------
        det_idx = num_surfs - 1
        for idx, s in enumerate(stack.surfaces):
            if s.surface_type == SurfaceType.DETECTOR:
                det_idx = idx
                break

        eval_idx = target_surface_index if (target_surface_index is not None and target_surface_index >= 0) else det_idx
        if eval_idx < 0 or eval_idx >= num_surfs:
            items.append(DiagnosticItem(
                code="INVALID_EVALUATION_SURFACE",
                message=f"Evaluation target surface index #{eval_idx} is out of bounds [0, {num_surfs - 1}].",
                severity=DiagnosticSeverity.ERROR,
                surface_index=eval_idx,
                suggested_fix=f"Set target surface index to {det_idx} (Detector Plane)."
            ))

        # -------------------------------------------------------------
        # 6. Check Target Metric & Spectral Prerequisites
        # -------------------------------------------------------------
        if target == "Spectrometer Spread":
            if len(wls) < 2:
                items.append(DiagnosticItem(
                    code="SPECTROMETER_NEEDS_MULTI_WAVELENGTH",
                    message=f"Spectrometer dispersion spread optimization requires at least 2 distinct wavelengths (currently {len(wls)}).",
                    severity=DiagnosticSeverity.ERROR,
                    suggested_fix="Add at least 2 distinct wavelengths in the Wavelengths editor."
                ))
            if target_spread <= 0:
                items.append(DiagnosticItem(
                    code="INVALID_TARGET_SPREAD",
                    message=f"Target spectrometer spread must be strictly positive (got {target_spread:.2f} mm).",
                    severity=DiagnosticSeverity.WARNING,
                    suggested_fix="Set target spread to match your detector active sensor length (e.g. 12.0 mm - 28.67 mm)."
                ))

        if "Target Spot" in target and target_spot_size < 0:
            items.append(DiagnosticItem(
                code="NEGATIVE_TARGET_SPOT",
                message=f"Target spot size cannot be negative (got {target_spot_size:.4f} mm).",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix="Set target spot size >= 0.0 mm."
            ))

        # -------------------------------------------------------------
        # 7. Check Optical Ray Tracing & Pre-Optimization Vignetting
        # -------------------------------------------------------------
        if num_surfs >= 2 and entrance_pupil_radius > 0:
            try:
                # Update positions
                stack._update_accumulated_positions()
                sys = OpticalSystem(stack)

                # Generate trial hexapolar rays
                trial_rays: List[Ray] = []
                primary_wl = wls[0] if wls else 0.5876
                z_dist = stack.surfaces[0].thickness if len(stack.surfaces) > 0 and stack.surfaces[0].thickness > 0 else 35.0
                
                # Chief ray
                trial_rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=primary_wl))
                # Marginal rays at 4 cardinal axes
                for angle in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
                    px = entrance_pupil_radius * np.cos(angle)
                    py = entrance_pupil_radius * np.sin(angle)
                    dir_v = np.array([px, py, z_dist], dtype=float)
                    dir_v /= np.linalg.norm(dir_v)
                    trial_rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=dir_v, wavelength=primary_wl))

                traced = sys.trace_rays(trial_rays)
                hits_on_eval = 0
                blocking_surface_idx = None
                
                for r in traced:
                    reached_eval = False
                    for h in r.history:
                        if h.get("surface_index") == eval_idx:
                            reached_eval = True
                            break
                    if reached_eval:
                        hits_on_eval += 1
                    else:
                        if len(r.history) > 0 and blocking_surface_idx is None:
                            last_h = r.history[-1]
                            last_reached = last_h.get("surface_index", 0)
                            if last_h.get("vignetted", False):
                                blocking_surface_idx = last_reached
                            else:
                                blocking_surface_idx = min(last_reached + 1, num_surfs - 1)

                if hits_on_eval == 0 and len(trial_rays) > 0:
                    blocking_name = stack.surfaces[blocking_surface_idx].name if (blocking_surface_idx is not None and 0 <= blocking_surface_idx < num_surfs) else "Unknown"
                    items.append(DiagnosticItem(
                        code="TOTAL_RAY_VIGNETTING",
                        message=f"All trial rays are blocked before reaching evaluation Surface #{eval_idx}. Obstruction occurred at Surface #{blocking_surface_idx} ('{blocking_name}').",
                        severity=DiagnosticSeverity.ERROR,
                        surface_index=blocking_surface_idx,
                        suggested_fix=f"Increase clear aperture / semi-diameter on Surface #{blocking_surface_idx} or check tilt/spacing."
                    ))
                elif hits_on_eval < len(trial_rays):
                    blocking_name = stack.surfaces[blocking_surface_idx].name if (blocking_surface_idx is not None and 0 <= blocking_surface_idx < num_surfs) else "Unknown"
                    items.append(DiagnosticItem(
                        code="PARTIAL_RAY_VIGNETTING",
                        message=f"Significant ray vignetting detected ({len(trial_rays) - hits_on_eval}/{len(trial_rays)} rays blocked before evaluation Surface #{eval_idx} at Surface #{blocking_surface_idx} '{blocking_name}').",
                        severity=DiagnosticSeverity.WARNING,
                        surface_index=blocking_surface_idx,
                        suggested_fix=f"Increase clear aperture / semi-diameter on Surface #{blocking_surface_idx} to prevent optical clipping."
                    ))
            except Exception as e:
                items.append(DiagnosticItem(
                    code="RAY_TRACE_EXCEPTION",
                    message=f"Pre-flight ray trace raised exception: {str(e)}",
                    severity=DiagnosticSeverity.ERROR
                ))

        is_valid = not any(item.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL) for item in items)
        return ValidationResult(is_valid=is_valid, items=items)

    @classmethod
    def auto_repair(
        cls,
        stack: SurfaceStack,
        variable_indices: Optional[Union[dict, List[int]]] = None,
        wavelengths: Union[float, List[float], np.ndarray] = 0.5876,
        target: str = "Spot Size"
    ) -> Tuple[SurfaceStack, Dict[str, List[int]], List[str]]:
        """
        Execute automated repair on optical stack and variable dictionary.
        Returns (repaired_stack, repaired_var_dict, applied_fixes_list).
        """
        repaired_stack = stack
        applied_fixes: List[str] = []

        # Normalize variables
        if variable_indices is None:
            var_dict = {"thickness": [], "radius": []}
        elif isinstance(variable_indices, dict):
            var_dict = {
                "thickness": list(variable_indices.get("thickness", [])),
                "radius": list(variable_indices.get("radius", []))
            }
        else:
            var_dict = {"thickness": list(variable_indices), "radius": []}

        # 1. Fix zero variables (first recover any surface-level variable flags)
        if not var_dict.get("thickness") and not var_dict.get("radius"):
            for idx, surf in enumerate(repaired_stack.surfaces[:-1]):
                if getattr(surf, "thickness_variable", False):
                    var_dict.setdefault("thickness", []).append(idx)
                if getattr(surf, "radius_variable", False):
                    var_dict.setdefault("radius", []).append(idx)

        if not var_dict.get("thickness") and not var_dict.get("radius"):
            var_dict["thickness"] = []
            for idx, surf in enumerate(repaired_stack.surfaces[:-1]):
                if idx == 0 and getattr(surf, "surface_type", None) == SurfaceType.OBJECT and len(repaired_stack.surfaces) > 2:
                    continue
                glass = getattr(surf, "glass", "AIR") or "AIR"
                if glass.upper() == "AIR":
                    var_dict["thickness"].append(idx)
                    setattr(surf, "thickness_variable", True)
            applied_fixes.append(f"Set {len(var_dict['thickness'])} air gap distances to Variable (V).")

        # 2. Fix unphysical lens thicknesses
        for idx, surf in enumerate(repaired_stack.surfaces[:-1]):
            glass = getattr(surf, "glass", "AIR") or "AIR"
            if glass.upper() not in ("AIR", "MIRROR", "MIRR", "") and surf.thickness <= 0:
                surf.thickness = 3.0
                applied_fixes.append(f"Reset non-positive glass lens Surface #{idx} thickness to 3.0 mm.")
            elif glass.upper() == "AIR" and surf.thickness < 0:
                surf.thickness = 10.0
                applied_fixes.append(f"Reset negative air gap Surface #{idx} thickness to 10.0 mm.")

        # 3. Allow rear surface to float if optimizing for focus and rear air gap is fixed
        if ("Focus" in target or "Spot" in target) and len(repaired_stack.surfaces) > 1:
            det_idx = len(repaired_stack.surfaces) - 1
            pre_det_idx = det_idx - 1
            if pre_det_idx not in var_dict.get("thickness", []):
                repaired_stack.surfaces[pre_det_idx].thickness_variable = True
                var_dict.setdefault("thickness", []).append(pre_det_idx)
                applied_fixes.append(f"Set rear surface #{pre_det_idx} to variable thickness to allow focusing on detector.")

        # 4. Update stack coordinates
        repaired_stack._update_accumulated_positions()

        return repaired_stack, var_dict, applied_fixes


__all__ = [
    "DiagnosticSeverity",
    "DiagnosticItem",
    "ValidationResult",
    "OptimizationValidator"
]
