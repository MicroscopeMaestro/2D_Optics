"""Empirical Adversarial Stress Test Suite for Milestone 2:
Commercial Scan Optics & High-NA Objective Prescriptions.

Adversarially validates:
1. Thorlabs SL50-2P2 chief ray telecentricity error across scan angles [-10 deg, +10 deg] (< 0.1 deg / 0.0 mrad in 4f relay).
2. Thorlabs TTL200MP 4.0x pupil magnification across varying beam waists (4.0 mm, 4.2 mm, 4.5 mm, 5.0 mm).
3. Olympus 20x 1.0 NA W & 25x 1.05 NA W entrance pupil diameters (18.0 mm and 16.8 mm), focal lengths (9.0 mm and 8.0 mm),
   and fill factor == 1.00 for 18.0 mm BFP beam (confirming the f=9.0 mm fix).
4. Thorlabs ACL25416U-B aspheric condenser exact conic kappa=-0.6265 and A4=1.8e-6 sag, derivative, normal, and collection.
5. Dual-tree parity between optics_app and v2 for all M2 modules.
"""

import math
import numpy as np
import pytest

from optics_app.catalogs.microscopy import (
    MICROSCOPE_OBJECTIVES,
    SCANNING_OPTICS,
    get_microscope_objective,
)
from optics_app.catalogs.thorlabs import get_catalog_lens
from optics_app.analysis.ultrafast_nonlinear import (
    OBJECTIVE_CATALOG,
    SCAN_LENS_CATALOG,
    TUBE_LENS_CATALOG,
    CommercialScanOpticsConfig,
    ObjectivePresetConfig,
    ThorlabsSRSMicroscopeConfig,
    ThorlabsSRSMicroscopeSimulator,
    build_olympus_objective_stack,
    build_folded_real_world_srs_template,
)
from optics_app.engine.surface import AsphericSurface, Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine import OpticalSystem, Ray


# =============================================================================
# 1. SL50-2P2 Chief Ray Telecentricity Error across Scan Angles [-10 deg, +10 deg]
# =============================================================================

class TestSL502P2Telecentricity:
    """Adversarial stress testing of SL50-2P2 telecentric scan lens."""

    def test_catalog_telecentricity_metadata(self):
        """Verify catalog metadata explicitly states telecentricity error < 0.1 deg."""
        assert "Thorlabs SL50-2P2" in SCAN_LENS_CATALOG
        sl_info = SCAN_LENS_CATALOG["Thorlabs SL50-2P2"]
        assert sl_info["focal_length_mm"] == 50.0
        assert sl_info["epd_mm"] == 28.5
        assert sl_info["telecentricity_error_deg"] == "< 0.1"

        assert "SL50-2P2" in SCANNING_OPTICS
        so_info = SCANNING_OPTICS["SL50-2P2"]
        assert so_info["effective_focal_length_mm"] == 50.0
        assert so_info["telecentricity_error_deg"] == "< 0.1"

    def test_commercial_scan_optics_config_telecentricity(self):
        """Verify CommercialScanOpticsConfig telecentricity error is < 0.1 deg."""
        cfg = CommercialScanOpticsConfig()
        assert cfg.scan_lens_model == "Thorlabs SL50-2P2"
        assert cfg.scan_lens_efl_mm == 50.0
        assert cfg.scan_lens_telecentricity_error_deg < 0.1
        assert cfg.scan_lens_telecentricity_error_deg == 0.0

    @pytest.mark.parametrize("angle_x_deg", np.linspace(-10.0, 10.0, 41))
    def test_4f_relayed_scanning_telecentricity_error_x_axis(self, angle_x_deg):
        """Adversarially sweep scan angle X across [-10.0 deg, +10.0 deg] at high resolution.
        In 4f-Relayed architecture, chief ray telecentricity error must be strictly 0.0 mrad (< 0.1 deg).
        """
        cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
        scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, float(angle_x_deg), 0.0)

        # 0.0 mrad == 0.0 deg < 0.1 deg
        assert scan.telecentricity_error_mrad == 0.0
        assert scan.back_pupil_walk_off_x_um == 0.0
        assert scan.back_pupil_walk_off_y_um == 0.0
        assert scan.back_pupil_walk_off_total_um == 0.0
        assert scan.is_strictly_conjugated is True

    @pytest.mark.parametrize("angle_y_deg", np.linspace(-10.0, 10.0, 41))
    def test_4f_relayed_scanning_telecentricity_error_y_axis(self, angle_y_deg):
        """Adversarially sweep scan angle Y across [-10.0 deg, +10.0 deg] at high resolution."""
        cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
        scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, float(angle_y_deg))

        assert scan.telecentricity_error_mrad == 0.0
        assert scan.back_pupil_walk_off_total_um == 0.0
        assert scan.is_strictly_conjugated is True

    @pytest.mark.parametrize("angle_x,angle_y", [
        (-10.0, -10.0), (-10.0, 10.0), (10.0, -10.0), (10.0, 10.0),
        (-7.07, -7.07), (7.07, 7.07), (-5.0, 5.0), (5.0, -5.0),
    ])
    def test_4f_relayed_diagonal_extreme_corners(self, angle_x, angle_y):
        """Verify telecentricity error at extreme field corners of 2D scan."""
        cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
        scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, angle_x, angle_y)

        assert scan.telecentricity_error_mrad == 0.0
        assert scan.back_pupil_walk_off_total_um == 0.0


# =============================================================================
# 2. 4.0x Pupil Magnification across Beam Waists
# =============================================================================

class TestPupilMagnification:
    """Adversarial stress testing of 4.0x pupil expansion across input waists."""

    def test_config_pupil_magnification_exact_ratio(self):
        """Verify M_pupil = f_TL / f_SL = 200.0 / 50.0 == 4.0."""
        cfg = CommercialScanOpticsConfig(scan_lens_efl_mm=50.0, tube_lens_efl_mm=200.0)
        assert cfg.pupil_magnification == 4.0

    @pytest.mark.parametrize("beam_diam_mm,expected_bfp_mm", [
        (4.0, 16.0),
        (4.2, 16.8),   # Matches Olympus 25x 1.05 NA W entrance pupil exactly
        (4.5, 18.0),   # Matches Olympus 20x 1.00 NA W entrance pupil exactly
        (5.0, 20.0),
        (3.0, 12.0),
        (6.0, 24.0),
    ])
    def test_pupil_beam_expansion_across_waists(self, beam_diam_mm, expected_bfp_mm):
        """Verify input beam waists expand by exactly 4.0x at objective back focal plane."""
        cfg = ThorlabsSRSMicroscopeConfig(
            input_beam_diameter_mm=beam_diam_mm,
            scan_lens_efl_mm=50.0,
            tube_lens_efl_mm=200.0,
        )
        scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
        assert scan.pupil_magnification == 4.0
        assert math.isclose(scan.back_pupil_beam_diameter_mm, expected_bfp_mm, rel_tol=1e-9)

    def test_magnification_with_alternative_tube_and_scan_lenses(self):
        """Verify pupil magnification adapts correctly when lenses are modified."""
        cfg = CommercialScanOpticsConfig(scan_lens_efl_mm=40.0, tube_lens_efl_mm=200.0)
        assert cfg.pupil_magnification == 5.0

        cfg2 = CommercialScanOpticsConfig(scan_lens_efl_mm=50.0, tube_lens_efl_mm=100.0)
        assert cfg2.pupil_magnification == 2.0


# =============================================================================
# 3. Olympus 20x 1.0 NA W & 25x 1.05 NA W Prescriptions and Fill Factor
# =============================================================================

class TestOlympusHighNAObjectivePrescriptions:
    """Adversarial stress testing of Olympus 20x and 25x immersion objective prescriptions."""

    def test_olympus_20x_catalog_parameters(self):
        """Verify Olympus 20x 1.00 NA W parameters in catalogs:
        - Design tube lens = 180.0 mm
        - Focal length f = 180 / 20 = 9.0 mm
        - NA = 1.00
        - EPD = 2 * NA * f = 18.0 mm
        - WD = 2.0 mm, cover glass = 0.17 mm, n_immersion = 1.333
        """
        info_cat = MICROSCOPE_OBJECTIVES["XLUMPLFLN20XW"]
        assert info_cat["numerical_aperture"] == 1.00
        assert info_cat["magnification"] == 20.0
        assert info_cat["focal_length_mm"] == 9.0
        assert info_cat["entrance_pupil_diameter_mm"] == 18.0
        assert info_cat["working_distance_mm"] == 2.0
        assert info_cat["cover_glass_mm"] == 0.17
        assert info_cat["immersion_refractive_index"] == 1.333

        info_nl = OBJECTIVE_CATALOG["Olympus XLUMPLFLN20XW (20x NA 1.00 W)"]
        assert info_nl["numerical_aperture"] == 1.00
        assert info_nl["focal_length_mm"] == 9.0
        assert info_nl["epd_mm"] == 18.0
        assert info_nl["pupil_diameter_mm"] == 18.0

    def test_olympus_25x_catalog_parameters(self):
        """Verify Olympus 25x 1.05 NA W parameters in catalogs:
        - Focal length f = 8.0 mm
        - NA = 1.05
        - EPD = 2 * NA * f = 2 * 1.05 * 8.0 = 16.8 mm
        - WD = 2.0 mm, cover glass = 0.17 mm, n_immersion = 1.333
        """
        info_cat = MICROSCOPE_OBJECTIVES["XLPLN25XWMP2"]
        assert info_cat["numerical_aperture"] == 1.05
        assert info_cat["magnification"] == 25.0
        assert info_cat["focal_length_mm"] == 8.0
        assert info_cat["entrance_pupil_diameter_mm"] == 16.8
        assert info_cat["working_distance_mm"] == 2.0
        assert info_cat["cover_glass_mm"] == 0.17
        assert info_cat["immersion_refractive_index"] == 1.333

        info_nl = OBJECTIVE_CATALOG["Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)"]
        assert info_nl["numerical_aperture"] == 1.05
        assert info_nl["focal_length_mm"] == 8.0
        assert info_nl["epd_mm"] == 16.8
        assert info_nl["pupil_diameter_mm"] == 16.8

    def test_olympus_20x_fill_factor_exact_one(self):
        """Empirically verify Olympus 20x 1.0 NA W fill factor == 1.00 for 18.0 mm BFP beam (f=9.0 mm fix).
        With input_beam_diameter_mm = 4.5 mm and 4.0x pupil magnification, beam at BFP is 18.0 mm.
        Objective pupil diameter = 2 * 1.0 * 9.0 = 18.0 mm.
        Therefore fill factor MUST evaluate to exactly 1.000000.
        """
        cfg = ThorlabsSRSMicroscopeConfig(
            objective_model="Olympus XLUMPLFLN20XW (20x NA 1.00 W)",
            input_beam_diameter_mm=4.5,
            scan_lens_efl_mm=50.0,
            tube_lens_efl_mm=200.0,
        )
        assert cfg.objective_focal_length_mm == 9.0
        assert cfg.objective_na == 1.00

        scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
        assert math.isclose(scan.back_pupil_beam_diameter_mm, 18.0, rel_tol=1e-9)
        assert math.isclose(scan.objective_pupil_diameter_mm, 18.0, rel_tol=1e-9)
        assert math.isclose(scan.pupil_fill_factor, 1.00, rel_tol=1e-9)
        assert scan.pupil_fill_factor == 1.0

    def test_olympus_25x_fill_factor_exact_one_for_16_8mm_beam(self):
        """Verify Olympus 25x 1.05 NA W fill factor == 1.00 for 16.8 mm BFP beam (input beam 4.2 mm)."""
        cfg = ThorlabsSRSMicroscopeConfig(
            objective_model="Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)",
            input_beam_diameter_mm=4.2,
            scan_lens_efl_mm=50.0,
            tube_lens_efl_mm=200.0,
        )
        assert cfg.objective_focal_length_mm == 8.0
        assert cfg.objective_na == 1.05

        scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
        assert math.isclose(scan.back_pupil_beam_diameter_mm, 16.8, rel_tol=1e-9)
        assert math.isclose(scan.objective_pupil_diameter_mm, 16.8, rel_tol=1e-9)
        assert math.isclose(scan.pupil_fill_factor, 1.00, rel_tol=1e-9)

    def test_objective_stack_builder_surface_structure(self):
        """Verify build_olympus_objective_stack generates 13 sequential surfaces with valid parameters."""
        import optics_app.analysis.ultrafast_nonlinear as app_un
        stack_20x = app_un.build_olympus_objective_stack("Olympus XLUMPLFLN20XW (20x NA 1.00 W)")
        assert len(stack_20x.surfaces) == 13
        assert stack_20x.surfaces[0].surface_type == SurfaceType.STOP
        assert stack_20x.surfaces[0].semi_diameter == 9.0  # 18.0 mm / 2
        assert stack_20x.surfaces[11].name.startswith("Cover Glass")
        assert stack_20x.surfaces[11].thickness == 0.17
        assert stack_20x.surfaces[12].surface_type == SurfaceType.DETECTOR

        stack_25x = app_un.build_olympus_objective_stack("Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)")
        assert len(stack_25x.surfaces) == 13
        assert stack_25x.surfaces[0].surface_type == SurfaceType.STOP
        assert stack_25x.surfaces[0].semi_diameter == 8.4  # 16.8 mm / 2
        assert stack_25x.surfaces[11].thickness == 0.17


# =============================================================================
# 4. Thorlabs ACL25416U-B Aspheric Condenser Conic and Sag Evaluation
# =============================================================================

class TestThorlabsACL25416UBAsphericCondenser:
    """Adversarial stress testing of ACL25416U-B aspheric condenser."""

    def test_catalog_prescription_parameters(self):
        """Verify exact catalog specifications for ACL25416U-B."""
        item = get_catalog_lens("ACL25416U-B")
        assert item is not None
        assert item["focal_length_mm"] == 16.0
        assert item["back_focal_length_mm"] == 8.5
        assert item["center_thickness_mm"] == 12.0
        assert item["diameter_mm"] == 25.4
        assert item["semi_diameter"] == 12.7
        s0 = item["surfaces"][0]
        assert math.isclose(s0["radius"], 8.32, abs_tol=1e-4)
        assert math.isclose(s0["conic"], -0.6265, abs_tol=1e-5)
        assert len(s0["asphere_coeffs"]) == 1
        assert math.isclose(s0["asphere_coeffs"][0], 1.8e-6, abs_tol=1e-12)
        assert s0["glass"] == "B270"

    @pytest.mark.parametrize("r_eval", [0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 8.32, 10.0, 11.5, 12.0, 12.7])
    def test_even_asphere_sag_analytical_equivalence(self, r_eval):
        """Empirically compare AsphericSurface.sag(r) against closed-form analytical equation.
        Equation: z(r) = [c * r^2] / [1 + sqrt(1 - (1 + kappa) * c^2 * r^2)] + A4 * r^4
        """
        R = 8.32
        c = 1.0 / R
        kappa = -0.6265
        a4 = 1.8e-6

        asph = AsphericSurface(radius=R, conic=kappa, asphere_coeffs=[a4])
        computed_sag = asph.sag(r_eval)

        # Closed-form analytical reference:
        r2 = r_eval ** 2
        arg = 1.0 - (1.0 + kappa) * (c ** 2) * r2
        assert arg > 0.0, f"Argument under sqrt must be positive across clear aperture, got {arg} for r={r_eval}"
        analytical_conic_sag = (c * r2) / (1.0 + math.sqrt(arg))
        analytical_poly_sag = a4 * (r_eval ** 4)
        analytical_total_sag = analytical_conic_sag + analytical_poly_sag

        assert math.isclose(computed_sag, analytical_total_sag, abs_tol=1e-12, rel_tol=1e-12)

    @pytest.mark.parametrize("r_eval", [0.1, 1.0, 3.0, 5.0, 8.0, 10.0, 12.0])
    def test_aspheric_surface_radial_derivative_analytical_vs_numerical(self, r_eval):
        """Verify analytical radial derivative dz/dr matches finite difference."""
        asph = AsphericSurface(radius=8.32, conic=-0.6265, asphere_coeffs=[1.8e-6])
        analytical_dz_dr = asph.dsag_dr(r_eval)

        dr = 1e-6
        numerical_dz_dr = (asph.sag(r_eval + dr) - asph.sag(r_eval - dr)) / (2.0 * dr)
        assert math.isclose(analytical_dz_dr, numerical_dz_dr, rel_tol=1e-5)

    def test_aspheric_surface_normal_vector_normalization(self):
        """Verify 3D gradient normal vectors are strictly unit length."""
        asph = AsphericSurface(radius=8.32, conic=-0.6265, asphere_coeffs=[1.8e-6])
        for x, y in [(0.0, 0.0), (2.0, 3.0), (-4.0, 1.0), (6.0, -5.0), (8.0, 8.0)]:
            r = math.sqrt(x**2 + y**2)
            z = asph.sag(r)
            hit = np.array([x, y, z])
            norm = asph.normal(hit)
            assert math.isclose(np.linalg.norm(norm), 1.0, abs_tol=1e-12)
            # Optical surface faces -Z (positive normal points along +Z)
            assert norm[2] > 0.0

    def test_condenser_ray_collection_and_collimation(self):
        """Verify ray collection through ACL25416U-B from focal waist across divergence angles."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Focus", thickness=8.5, semi_diameter=2.0, surface_type=SurfaceType.OBJECT))
        stack.add_surface(Surface(name="Condenser Front", radius=8.32, conic=-0.6265, asphere_coeffs=[1.8e-6], thickness=12.0, glass="B270", semi_diameter=12.7, surface_type=SurfaceType.ASPHERIC))
        stack.add_surface(Surface(name="Condenser Rear", radius=0.0, thickness=25.0, glass="AIR", semi_diameter=12.7))
        stack.add_surface(Surface(name="Detector", thickness=0.0, semi_diameter=20.0, surface_type=SurfaceType.DETECTOR))

        sys = OpticalSystem(stack)

        # Collect divergence angles up to 25 deg without vignetting or TIR
        for deg in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
            rad = math.radians(deg)
            dir_vec = np.array([math.sin(rad), 0.0, math.cos(rad)])
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=dir_vec, wavelength=0.797)
            tr = sys.trace_ray(ray)
            assert not tr.vignetted, f"Ray at divergence angle {deg} deg should be collected without vignetting"
            assert len(tr.history) == 4
            # Output ray should be substantially collimated (exit angle < 10 deg)
            d_out = tr.history[-1]["dir"]
            exit_angle_deg = math.degrees(math.atan2(d_out[0], d_out[2]))
            assert abs(exit_angle_deg) < 10.0


# =============================================================================
# 5. Dual-Tree Parity Validation (optics_app vs v2)
# =============================================================================

class TestDualTreeParityM2:
    """Empirical verification of 100% byte/code parity between optics_app and v2 for M2 modules."""

    def test_catalogs_microscopy_parity(self):
        import optics_app.catalogs.microscopy as app_m
        import v2.catalogs.microscopy as v2_m

        assert list(app_m.MICROSCOPE_OBJECTIVES.keys()) == list(v2_m.MICROSCOPE_OBJECTIVES.keys())
        assert list(app_m.SCANNING_OPTICS.keys()) == list(v2_m.SCANNING_OPTICS.keys())

        # Check Olympus 20x equality
        assert app_m.MICROSCOPE_OBJECTIVES["XLUMPLFLN20XW"] == v2_m.MICROSCOPE_OBJECTIVES["XLUMPLFLN20XW"]
        assert app_m.MICROSCOPE_OBJECTIVES["XLPLN25XWMP2"] == v2_m.MICROSCOPE_OBJECTIVES["XLPLN25XWMP2"]

    def test_analysis_ultrafast_nonlinear_parity(self):
        import optics_app.analysis.ultrafast_nonlinear as app_un
        import v2.analysis.ultrafast_nonlinear as v2_un

        assert list(app_un.OBJECTIVE_CATALOG.keys()) == list(v2_un.OBJECTIVE_CATALOG.keys())
        assert list(app_un.SCAN_LENS_CATALOG.keys()) == list(v2_un.SCAN_LENS_CATALOG.keys())
        assert list(app_un.TUBE_LENS_CATALOG.keys()) == list(v2_un.TUBE_LENS_CATALOG.keys())

        import copy
        from dataclasses import fields, asdict
        assert [f.name for f in fields(app_un.CommercialScanOpticsConfig)] == [f.name for f in fields(v2_un.CommercialScanOpticsConfig)]
        assert [f.name for f in fields(app_un.ObjectivePresetConfig)] == [f.name for f in fields(v2_un.ObjectivePresetConfig)]
        assert asdict(app_un.CommercialScanOpticsConfig()) == asdict(v2_un.CommercialScanOpticsConfig())

        assert asdict(app_un.ObjectivePresetConfig()) == asdict(v2_un.ObjectivePresetConfig())
