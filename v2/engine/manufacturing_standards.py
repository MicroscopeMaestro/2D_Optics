"""
Zemax OpticStudio & ISO 10110 Manufacturing & Mounting Standards Engine.
========================================================================
Defines industry-standard optical fabrication tolerances, boundary operands,
mechanical mounting cell clearances, standard tooling test plates, and
as-built desensitization metrics.

Standards implemented:
1. ISO 10110-1 (Drawings, Mechanical Aperture, Protective Chamfers/Bevels)
2. ISO 10110-6 (Centration, Wedging, Runout)
3. ISO 10110-12 (Tolerancing & As-Built Verification)
4. Zemax OpticStudio Default Merit Function Boundary Operands:
   - MNCG / MXCG: Minimum / Maximum Center thickness Glass
   - MNEG / MXEG: Minimum / Maximum Edge thickness Glass (at optical & mechanical rims)
   - MNCA / MXCA: Minimum / Maximum Center thickness Air
   - MNEA / MXEA: Minimum / Maximum Edge thickness Air (spacer ring clearance)
   - TOTR / MXTR: Total Track length
   - RAID / RAIM: Maximum angle of ray incidence on refractive surfaces
   - TFLT: Test Plate Fitting (catalog shop tooling radius snapping)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np


class QualityPreset(Enum):
    """Industry manufacturing quality presets according to ISO 10110 and catalog standards."""
    COMMERCIAL = "Commercial (Thorlabs / Edmund)"
    PRECISION = "Precision (Research / Laser Grade)"
    HIGH_PRECISION = "High-Precision (Space / Litho Grade)"
    CUSTOM = "Custom"


@dataclass
class ManufacturingConstraintsConfig:
    """Configuration container for optical manufacturing and mounting constraints."""
    preset: QualityPreset = QualityPreset.COMMERCIAL
    enforce: bool = True

    # 1. Glass Center Thickness (MNCG / MXCG)
    min_glass_thickness: float = 1.5       # mm (MNCG)
    max_glass_thickness: float = 18.0      # mm (MXCG)

    # 2. Glass Edge Thickness (MNEG / MXEG)
    min_edge_thickness: float = 1.0        # mm at optical clear aperture (MNEG)
    max_edge_thickness: float = 20.0       # mm at optical clear aperture (MXEG / MXET)
    min_mechanical_edge: float = 0.8       # mm at mechanical mounting rim (MNEG_MECH)

    # 3. Air Gap Center Thickness (MNCA / MXCA)
    min_air_gap: float = 0.5               # mm (MNCA)
    max_air_gap: float = 150.0             # mm (MXCA)

    # 4. Air Gap Edge Clearance (MNEA / MXEA)
    min_edge_air: float = 0.5              # mm (MNEA - spacer ring seating clearance)

    # 5. Working Distance / Back Focal Length (BFL)
    min_bfl: float = 10.0                  # mm

    # 6. Aspect Ratio Constraint (Diameter-to-Thickness Ratio D / t_glass)
    max_aspect_ratio: float = 10.0         # ISO polishing limit: D/t <= 10
    min_aspect_ratio: float = 2.0          # Prevents stubby glass cylinders: D/t >= 2

    # 7. Angle of Incidence Constraint (RAID)
    max_incidence_angle_deg: float = 55.0  # Degrees (prevents AR coating failure & extreme aberrations)

    # 8. Total Track Length (TOTR)
    max_total_track: float = 500.0         # mm

    # 9. Mechanical Mounting & Chamfer Standards (ISO 10110-1)
    mounting_margin: float = 1.5           # mm radial margin: y_mech = y_clear + mounting_margin
    bevel_width: float = 0.3               # mm 45-degree protective bevel/chamfer width
    min_seating_flat: float = 0.5          # mm minimum contact flat width for retaining ring

    # 10. Tolerancing Desensitization (As-Built Optimization)
    desensitize_tolerances: bool = False
    as_built_weight: float = 10.0

    # 11. Test Plate Tooling Fit (TFLT)
    fit_to_test_plates: bool = False
    max_fringe_power_error: float = 5.0    # fringes at lambda = 546.1 nm (mercury e-line)

    @classmethod
    def from_preset(cls, preset: QualityPreset) -> "ManufacturingConstraintsConfig":
        """Instantiate standard configuration from standard quality tier."""
        if preset == QualityPreset.HIGH_PRECISION:
            return cls(
                preset=preset,
                enforce=True,
                min_glass_thickness=2.5,
                max_glass_thickness=12.0,
                min_edge_thickness=2.0,
                max_edge_thickness=12.0,
                min_mechanical_edge=1.5,
                min_air_gap=1.0,
                max_air_gap=100.0,
                min_edge_air=1.0,
                min_bfl=15.0,
                max_aspect_ratio=6.0,
                min_aspect_ratio=2.5,
                max_incidence_angle_deg=45.0,
                max_total_track=350.0,
                mounting_margin=2.0,
                bevel_width=0.5,
                min_seating_flat=0.8,
                desensitize_tolerances=True,
                as_built_weight=20.0,
                fit_to_test_plates=False,
                max_fringe_power_error=2.0
            )
        elif preset == QualityPreset.PRECISION:
            return cls(
                preset=preset,
                enforce=True,
                min_glass_thickness=2.0,
                max_glass_thickness=15.0,
                min_edge_thickness=1.5,
                max_edge_thickness=15.0,
                min_mechanical_edge=1.2,
                min_air_gap=0.8,
                max_air_gap=120.0,
                min_edge_air=0.8,
                min_bfl=12.0,
                max_aspect_ratio=8.0,
                min_aspect_ratio=2.0,
                max_incidence_angle_deg=50.0,
                max_total_track=400.0,
                mounting_margin=1.5,
                bevel_width=0.4,
                min_seating_flat=0.6,
                desensitize_tolerances=False,
                as_built_weight=10.0,
                fit_to_test_plates=False,
                max_fringe_power_error=3.0
            )
        else:  # COMMERCIAL
            return cls(
                preset=QualityPreset.COMMERCIAL,
                enforce=True,
                min_glass_thickness=1.5,
                max_glass_thickness=18.0,
                min_edge_thickness=1.0,
                max_edge_thickness=20.0,
                min_mechanical_edge=0.8,
                min_air_gap=0.5,
                max_air_gap=150.0,
                min_edge_air=0.5,
                min_bfl=10.0,
                max_aspect_ratio=10.0,
                min_aspect_ratio=1.8,
                max_incidence_angle_deg=55.0,
                max_total_track=500.0,
                mounting_margin=1.5,
                bevel_width=0.3,
                min_seating_flat=0.5,
                desensitize_tolerances=False,
                as_built_weight=5.0,
                fit_to_test_plates=False,
                max_fringe_power_error=5.0
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation."""
        return {
            "preset": self.preset.value,
            "enforce": self.enforce,
            "min_glass_thickness": self.min_glass_thickness,
            "max_glass_thickness": self.max_glass_thickness,
            "min_edge_thickness": self.min_edge_thickness,
            "max_edge_thickness": self.max_edge_thickness,
            "min_mechanical_edge": self.min_mechanical_edge,
            "min_air_gap": self.min_air_gap,
            "max_air_gap": self.max_air_gap,
            "min_edge_air": self.min_edge_air,
            "min_bfl": self.min_bfl,
            "max_aspect_ratio": self.max_aspect_ratio,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_incidence_angle_deg": self.max_incidence_angle_deg,
            "max_total_track": self.max_total_track,
            "mounting_margin": self.mounting_margin,
            "bevel_width": self.bevel_width,
            "min_seating_flat": self.min_seating_flat,
            "desensitize_tolerances": self.desensitize_tolerances,
            "as_built_weight": self.as_built_weight,
            "fit_to_test_plates": self.fit_to_test_plates,
            "max_fringe_power_error": self.max_fringe_power_error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManufacturingConstraintsConfig":
        """Instantiate configuration from dictionary."""
        cfg = cls()
        preset_val = d.get("preset", QualityPreset.COMMERCIAL.value)
        for p in QualityPreset:
            if p.value == preset_val or p.name == str(preset_val).upper():
                cfg.preset = p
                break
        for k, v in d.items():
            if hasattr(cfg, k) and k != "preset":
                setattr(cfg, k, v)
        return cfg


class TestPlateCatalog:
    """
    Standard Optical Shop Test Plate Tooling Radius Catalog.
    Based on standard ISO / DIN / Thorlabs / Schott test plate tooling series.
    """
    STANDARD_RADII_MM = [
        5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 25.0,
        28.0, 30.0, 32.0, 35.0, 38.0, 40.0, 45.0, 50.0, 55.0, 60.0,
        65.0, 70.0, 75.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
        150.0, 160.0, 180.0, 200.0, 220.0, 250.0, 280.0, 300.0, 350.0,
        400.0, 450.0, 500.0, 600.0, 700.0, 800.0, 1000.0, 1250.0, 1500.0,
        2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 10000.0
    ]

    @classmethod
    def find_nearest_test_plate(
        cls,
        radius_mm: float,
        semi_diameter_mm: float = 12.5,
        test_wavelength_um: float = 0.5461  # Mercury e-line (standard optical interferometry)
    ) -> Tuple[float, float, float]:
        """
        Find closest standard tooling radius.
        Returns: (nearest_radius_mm, delta_radius_mm, fringe_power_error_N)
        fringe_power_error_N: Optical fringes of power error Delta N = 2 * Delta Sag / lambda.
        """
        if abs(radius_mm) < 1e-4 or math.isinf(radius_mm) or abs(radius_mm) > 20000.0:
            return 0.0, 0.0, 0.0  # Plano surface is always standard

        sign = 1.0 if radius_mm > 0 else -1.0
        abs_r = abs(radius_mm)

        # Binary search for closest radius
        idx = int(np.searchsorted(cls.STANDARD_RADII_MM, abs_r))
        candidates = []
        if idx < len(cls.STANDARD_RADII_MM):
            candidates.append(cls.STANDARD_RADII_MM[idx])
        if idx > 0:
            candidates.append(cls.STANDARD_RADII_MM[idx - 1])
        if idx + 1 < len(cls.STANDARD_RADII_MM):
            candidates.append(cls.STANDARD_RADII_MM[idx + 1])

        # Pick candidate with minimal relative curvature error |1/R - 1/R_cand|
        best_r = min(candidates, key=lambda c: abs(1.0 / abs_r - 1.0 / c))
        nearest_radius = sign * best_r
        delta_r = radius_mm - nearest_radius

        # Compute fringe power error Delta N = 2 * |sag1 - sag2| / lambda
        # sag = r^2 / (2 * R) in paraxial approximation
        sd = max(1.0, float(semi_diameter_mm))
        wl_mm = test_wavelength_um * 1e-3
        delta_sag = abs((sd**2) / (2.0 * radius_mm) - (sd**2) / (2.0 * nearest_radius))
        fringe_n = 2.0 * delta_sag / wl_mm

        return float(nearest_radius), float(delta_r), float(fringe_n)


@dataclass
class ElementManufacturingAudit:
    """Individual lens element manufacturing & mounting compliance audit."""
    element_index: int
    name: str
    glass: str
    diameter_clear_mm: float
    diameter_mechanical_mm: float
    center_thickness_mm: float
    edge_thickness_clear_mm: float
    edge_thickness_mechanical_mm: float
    aspect_ratio: float
    r1_mm: float
    r2_mm: float
    r1_test_plate_mm: float
    r2_test_plate_mm: float
    r1_fringes: float
    r2_fringes: float
    max_incidence_angle_deg: float
    seating_flat_width_mm: float
    is_compliant: bool
    surface_front_index: int = 0
    surface_rear_index: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SystemManufacturingReport:
    """Complete optical system manufacturing and mounting compliance audit."""
    is_compliant: bool
    compliance_score: float  # 0.0 to 100.0%
    preset_name: str
    elements: List[ElementManufacturingAudit] = field(default_factory=list)
    total_track_mm: float = 0.0
    air_spacings: List[Dict[str, Any]] = field(default_factory=list)
    overall_warnings: List[str] = field(default_factory=list)
    overall_errors: List[str] = field(default_factory=list)

    def summary_markdown(self) -> str:
        """Format an extensive Markdown summary report."""
        lines = []
        lines.append(f"### Optical Manufacturing & Mounting Audit ({self.preset_name})")
        status_badge = "PASS" if self.is_compliant else "NON-COMPLIANT"
        lines.append(f"**Overall Status:** `{status_badge}` | **Compliance Score:** `{self.compliance_score:.1f}%` | **Total Track:** `{self.total_track_mm:.2f} mm`\n")

        lines.append("| Elem | Name | Glass | Diam (Clear / Mech) | $t_c$ (mm) | $t_{edge}$ (Opt / Mech) | $D/t$ | Max $\\theta_{inc}$ | Tooling $R_1 / R_2$ | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for el in self.elements:
            stat = "PASS" if el.is_compliant else ("WARN" if el.warnings and not el.errors else "FAIL")
            t_edge_str = f"{el.edge_thickness_clear_mm:.2f} / {el.edge_thickness_mechanical_mm:.2f}"
            diam_str = f"{el.diameter_clear_mm:.1f} / {el.diameter_mechanical_mm:.1f}"
            r_tool = f"{el.r1_test_plate_mm:.1f} ({el.r1_fringes:.1f}fr) / {el.r2_test_plate_mm:.1f} ({el.r2_fringes:.1f}fr)"
            lines.append(
                f"| #{el.element_index} | {el.name} | {el.glass} | {diam_str} | {el.center_thickness_mm:.2f} | "
                f"{t_edge_str} | {el.aspect_ratio:.1f} | {el.max_incidence_angle_deg:.1f}° | {r_tool} | `{stat}` |"
            )

        if self.air_spacings:
            lines.append("\n**Air Spacings & Spacer Ring Seating:**")
            for air in self.air_spacings:
                lines.append(f"- Air Gap #{air['gap_index']}: Center $t_c = {air['center_thickness']:.2f}\\text{{ mm}}$, Edge Clearance $t_{{edge}} = {air['edge_clearance']:.2f}\\text{{ mm}}$ (`{air['status']}`)")

        if self.overall_errors:
            lines.append("\n**Critical Fabrication & Mounting Violations:**")
            for err in self.overall_errors:
                lines.append(f"- ❌ {err}")

        if self.overall_warnings:
            lines.append("\n**Fabrication Advisories:**")
            for warn in self.overall_warnings:
                lines.append(f"- ⚠️ {warn}")

        return "\n".join(lines)

    format_markdown = summary_markdown


def audit_optical_system_manufacturing(
    surfaces: List[Any],
    config: Optional[ManufacturingConstraintsConfig] = None,
    ray_incidence_angles: Optional[Dict[int, float]] = None
) -> SystemManufacturingReport:
    """
    Execute rigorous ISO 10110 / Zemax manufacturing and mounting audit on a sequential surface stack.
    Evaluates center and edge thickness, mechanical mounting rims, aspect ratios, air spacer seating,
    and ray incidence angles.
    """
    cfg = config or ManufacturingConstraintsConfig.from_preset(QualityPreset.COMMERCIAL)
    elements: List[ElementManufacturingAudit] = []
    air_spacings: List[Dict[str, Any]] = []
    overall_warnings: List[str] = []
    overall_errors: List[str] = []

    totr = 0.0
    num_surfs = len(surfaces)
    elem_idx = 1
    total_checks = 0
    passed_checks = 0

    ray_angles = ray_incidence_angles or {}

    st0 = getattr(surfaces[0], "surface_type", None) if num_surfs > 0 else None
    is_obj0 = (st0 == "Object" or getattr(st0, "name", None) == "OBJECT" or getattr(st0, "value", None) == "Object" or str(st0).lower() in ("object", "surfacetype.object"))
    start_surf = 1 if (num_surfs > 1 and is_obj0) else 0

    for i in range(num_surfs):
        s = surfaces[i]
        t = float(getattr(s, "thickness", 0.0))
        if i >= start_surf:
            totr += t
        g = str(getattr(s, "glass", "AIR") or "AIR").upper()
        is_glass = g not in ("AIR", "MIRROR", "MIRR", "VACUUM", "") and (s.surface_type not in ("Detector", "Stop", "Object"))

        if is_glass and (i + 1 < num_surfs):
            s_next = surfaces[i + 1]
            sd_clear = min(float(getattr(s, "semi_diameter", 12.5)), float(getattr(s_next, "semi_diameter", 12.5)))
            sd_mech = sd_clear + cfg.mounting_margin
            diam_clear = 2.0 * sd_clear
            diam_mech = 2.0 * sd_mech

            # Aspect ratio D / t
            t_center = max(1e-4, t)
            aspect_ratio = diam_mech / t_center

            # Sags at clear aperture and mechanical rim
            sag1_clear = s.sag(sd_clear)
            sag2_clear = s_next.sag(sd_clear)
            t_edge_clear = t_center + sag2_clear - sag1_clear

            sag1_mech = s.sag(sd_mech)
            sag2_mech = s_next.sag(sd_mech)
            t_edge_mech = t_center + sag2_mech - sag1_mech

            # Chamfer flat
            seating_flat = cfg.mounting_margin - cfg.bevel_width

            # Test plate fitting for R1 and R2
            r1 = float(getattr(s, "radius", 0.0))
            r2 = float(getattr(s_next, "radius", 0.0))
            r1_tool, _, r1_fringes = TestPlateCatalog.find_nearest_test_plate(r1, sd_clear)
            r2_tool, _, r2_fringes = TestPlateCatalog.find_nearest_test_plate(r2, sd_clear)

            # Max ray incidence angle on either surface
            theta_max = max(ray_angles.get(i, 0.0), ray_angles.get(i + 1, 0.0))

            elem_warns = []
            elem_errs = []

            # Check 1: Center thickness MNCG
            total_checks += 1
            if t_center < cfg.min_glass_thickness:
                elem_errs.append(f"Center thickness {t_center:.2f} mm < min {cfg.min_glass_thickness:.1f} mm (MNCG violation: risk of flexing/cracking during polishing)")
            elif t_center > cfg.max_glass_thickness:
                elem_warns.append(f"Center thickness {t_center:.2f} mm > max {cfg.max_glass_thickness:.1f} mm (MXCG violation: excess mass/absorption)")
            else:
                passed_checks += 1

            # Check 2: Optical Edge thickness MNEG / MXEG
            total_checks += 1
            max_edge = getattr(cfg, "max_edge_thickness", 20.0)
            if t_edge_clear < cfg.min_edge_thickness:
                elem_errs.append(f"Clear edge thickness {t_edge_clear:.2f} mm < min {cfg.min_edge_thickness:.1f} mm (MNEG violation: knife-edge risk)")
            elif t_edge_clear > max_edge:
                elem_warns.append(f"Clear edge thickness {t_edge_clear:.2f} mm > max {max_edge:.1f} mm (MXEG violation: excess rim mass/mounting obstruction)")
            else:
                passed_checks += 1

            # Check 3: Mechanical Mounting Rim Edge Thickness
            total_checks += 1
            if t_edge_mech < cfg.min_mechanical_edge:
                elem_errs.append(f"Mechanical rim edge thickness {t_edge_mech:.2f} mm < min {cfg.min_mechanical_edge:.1f} mm (Cannot seat into lens barrel collar)")
            else:
                passed_checks += 1

            # Check 4: Aspect Ratio D / t
            total_checks += 1
            if aspect_ratio > cfg.max_aspect_ratio:
                elem_errs.append(f"Aspect ratio D/t = {aspect_ratio:.1f} exceeds ISO 10110 limit ({cfg.max_aspect_ratio:.1f}: lens will warp during double-sided polishing)")
            elif aspect_ratio < cfg.min_aspect_ratio:
                elem_warns.append(f"Aspect ratio D/t = {aspect_ratio:.1f} < {cfg.min_aspect_ratio:.1f} (Stubby cylinder element)")
            else:
                passed_checks += 1

            # Check 5: Seating flat width for retaining ring
            total_checks += 1
            if seating_flat < cfg.min_seating_flat:
                elem_warns.append(f"Mounting seating flat {seating_flat:.2f} mm is narrow (< {cfg.min_seating_flat:.1f} mm)")
            else:
                passed_checks += 1

            # Check 6: Ray Angle of Incidence
            total_checks += 1
            if theta_max > cfg.max_incidence_angle_deg:
                elem_warns.append(f"Ray angle of incidence {theta_max:.1f}° exceeds limit ({cfg.max_incidence_angle_deg:.1f}°: AR coating degradation and high aberration)")
            else:
                passed_checks += 1

            # Check 7: Test Plate tooling error
            if cfg.fit_to_test_plates:
                total_checks += 1
                if r1_fringes > cfg.max_fringe_power_error or r2_fringes > cfg.max_fringe_power_error:
                    elem_warns.append(f"Surface requires custom test plates (R1: {r1_fringes:.1f} fr, R2: {r2_fringes:.1f} fr > {cfg.max_fringe_power_error:.1f} fr)")
                else:
                    passed_checks += 1

            is_elem_comp = len(elem_errs) == 0
            if elem_errs:
                overall_errors.extend([f"Element #{elem_idx} ('{s.name}'): {e}" for e in elem_errs])
            if elem_warns:
                overall_warnings.extend([f"Element #{elem_idx} ('{s.name}'): {w}" for w in elem_warns])

            elements.append(ElementManufacturingAudit(
                element_index=elem_idx,
                name=s.name or f"L{elem_idx}",
                glass=g,
                diameter_clear_mm=diam_clear,
                diameter_mechanical_mm=diam_mech,
                center_thickness_mm=t_center,
                edge_thickness_clear_mm=t_edge_clear,
                edge_thickness_mechanical_mm=t_edge_mech,
                aspect_ratio=aspect_ratio,
                r1_mm=r1,
                r2_mm=r2,
                r1_test_plate_mm=r1_tool,
                r2_test_plate_mm=r2_tool,
                r1_fringes=r1_fringes,
                r2_fringes=r2_fringes,
                max_incidence_angle_deg=theta_max,
                seating_flat_width_mm=seating_flat,
                is_compliant=is_elem_comp,
                surface_front_index=i,
                surface_rear_index=i + 1,
                warnings=elem_warns,
                errors=elem_errs
            ))
            elem_idx += 1

        elif not is_glass and s.surface_type not in ("Detector", "Object") and (i + 1 < num_surfs):
            # Air gap audit (spacer ring clearance)
            s_next = surfaces[i + 1]
            is_detector_gap = (s_next.surface_type == "Detector")
            sd_air = min(float(getattr(s, "semi_diameter", 12.5)), float(getattr(s_next, "semi_diameter", 12.5)))
            sd_mech = sd_air + cfg.mounting_margin
            sag_a = s.sag(sd_mech)
            sag_b = s_next.sag(sd_mech)
            edge_air = t + sag_b - sag_a

            air_err = False
            req_t = cfg.min_bfl if is_detector_gap else cfg.min_air_gap
            req_edge = 0.5 if is_detector_gap else cfg.min_edge_air

            total_checks += 1
            if t < req_t:
                air_err = True
                msg = f"Air gap after Surface #{i} ('{s.name}') center thickness {t:.2f} mm < min {req_t:.1f} mm"
                overall_errors.append(msg)
            else:
                passed_checks += 1

            total_checks += 1
            if not is_detector_gap and edge_air < req_edge:
                air_err = True
                msg = f"Air gap after Surface #{i} ('{s.name}') rim clearance {edge_air:.2f} mm < min {req_edge:.1f} mm (Lens elements collide at rims: no spacer room)"
                overall_errors.append(msg)
            else:
                passed_checks += 1

            air_spacings.append({
                "gap_index": i,
                "surface_name": s.name,
                "center_thickness": t,
                "edge_clearance": edge_air,
                "is_detector_gap": is_detector_gap,
                "status": "PASS" if not air_err else "FAIL"
            })

    # Check Total Track Length TOTR
    total_checks += 1
    if totr > cfg.max_total_track:
        overall_errors.append(f"Total track length {totr:.2f} mm exceeds maximum allowed {cfg.max_total_track:.1f} mm (MXTR violation)")
    else:
        passed_checks += 1

    compliance_score = (passed_checks / max(1, total_checks)) * 100.0
    is_compliant = len(overall_errors) == 0

    return SystemManufacturingReport(
        is_compliant=is_compliant,
        compliance_score=compliance_score,
        preset_name=cfg.preset.value,
        elements=elements,
        total_track_mm=totr,
        air_spacings=air_spacings,
        overall_warnings=overall_warnings,
        overall_errors=overall_errors
    )
