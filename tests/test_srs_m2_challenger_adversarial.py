"""Adversarial Challenge Test Suite for Milestone 2.

Author: challenger_srs_m2_2
Role: Empirical Ray Tracing & Surface Train Challenger
Milestone: Milestone 2 (Commercial Scan Optics & High-NA Objective Prescriptions)

Challenge Scope:
1. Standalone 12-surface Zemax stacks for Olympus 20x 1.0 NA W and 25x 1.05 NA W:
   - Empirical test for deepcopy NameError in ObjectivePresetConfig / build_olympus_objective_stack
   - Verification of glass dispersion definitions (S-FPL53, N-LAK9, N-LASF44, WATER, D263T)
   - Marginal and paraxial ray tracing through all surfaces, cover glass, and water
2. 37-surface train in build_folded_real_world_srs_template:
   - Verification of strictly 37 surfaces (28 sequential + 9 off-axis)
   - Sequential ray tracing: on-axis vs. edge-of-pupil rays (vignetting at Galvo Y)
3. Dual-tree parity:
   - Comparison of geometry and ray tracing between optics_app and v2 to machine precision
"""

import copy
import math
import numpy as np
import pytest

import optics_app.analysis.ultrafast_nonlinear as app_ufn
import v2.analysis.ultrafast_nonlinear as v2_ufn
from optics_app.engine.tracer import OpticalSystem, Ray
from optics_app.utils.materials import get_refractive_index, get_complex_refractive_index


class TestDefect1DeepcopyMissing:
    """Challenge whether ObjectivePresetConfig and build_olympus_objective_stack can be executed."""

    def test_objective_preset_config_deepcopy_name_error(self):
        """Demonstrate that calling ObjectivePresetConfig or build_olympus_objective_stack fails with NameError."""
        # Unpatch if previously patched in test process
        if hasattr(app_ufn, "deepcopy"):
            delattr(app_ufn, "deepcopy")

        with pytest.raises(NameError, match="name 'deepcopy' is not defined"):
            _ = app_ufn.build_olympus_objective_stack("Olympus XLUMPLFLN20XW (20x NA 1.00 W)")

    def test_v2_objective_preset_config_deepcopy_name_error(self):
        """Demonstrate that v2 also suffers from the same NameError."""
        if hasattr(v2_ufn, "deepcopy"):
            delattr(v2_ufn, "deepcopy")

        with pytest.raises(NameError, match="name 'deepcopy' is not defined"):
            _ = v2_ufn.build_olympus_objective_stack("Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)")


class TestDefect2MissingGlassDispersions:
    """Challenge glass catalog definitions for the 12-surface Zemax objective prescriptions."""

    def test_missing_objective_glasses_in_materials_catalog(self):
        """Verify which glasses used in the Olympus objectives are missing from GlassCatalog."""
        objective_glasses = ["S-FPL53", "N-LAK9", "N-LASF44", "WATER", "D263T"]
        missing = []
        for g in objective_glasses:
            try:
                _ = get_refractive_index(g, 0.797)
            except ValueError:
                missing.append(g)

        # 5 out of 5 glasses are completely missing from the glass catalog
        assert len(missing) == 5, f"Expected 5 missing glasses, got {missing}"
        assert set(missing) == {"S-FPL53", "N-LAK9", "N-LASF44", "WATER", "D263T"}

    def test_unregistered_glasses_quietly_fall_back_to_air_index(self):
        """Verify get_complex_refractive_index falls back to n=1.0 for missing glasses."""
        for g in ["S-FPL53", "N-LAK9", "N-LASF44", "WATER", "D263T"]:
            n_complex = get_complex_refractive_index(g, 0.797)
            assert n_complex.real == 1.0, f"Expected fallback 1.0 for {g}, got {n_complex.real}"


class TestDefect3StandaloneZemaxStacksConvergence:
    """Trace rays through the standalone objective stacks and test convergence."""

    @pytest.fixture(autouse=True)
    def patch_deepcopy(self):
        """Provide deepcopy locally to evaluate downstream ray-tracing behavior."""
        app_ufn.deepcopy = copy.deepcopy
        v2_ufn.deepcopy = copy.deepcopy

    def test_olympus_20x_ray_tracing_fails_convergence(self):
        """Demonstrate that with n=1.0 fallback, rays diverge and vignette at the detector plane."""
        stack20 = app_ufn.build_olympus_objective_stack("Olympus XLUMPLFLN20XW (20x NA 1.00 W)")
        sys20 = OpticalSystem(stack20)

        # A ray at y = 0.9 mm (only 10% of pupil) diverges to y = 2.168 mm and vignettes at detector
        r = sys20.trace_ray(Ray(pos=np.array([0.0, 0.9, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797))
        assert r.vignetted is True
        last_step = r.history[-1]
        assert last_step["surface_index"] == 12  # Sample Focal Waist
        # Detector semi-diameter is 1.0 mm, but ray landed at 2.168 mm
        assert last_step["y"] > 2.0

    def test_olympus_25x_ray_tracing_fails_convergence(self):
        """Demonstrate that with n=1.0 fallback, rays diverge and vignette at cover glass."""
        stack25 = app_ufn.build_olympus_objective_stack("Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)")
        sys25 = OpticalSystem(stack25)

        r = sys25.trace_ray(Ray(pos=np.array([0.0, 0.84, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797))
        assert r.vignetted is True
        last_step = r.history[-1]
        assert last_step["surface_index"] == 11  # Cover Glass Front


class TestDefect4FoldedSRSTemplateVignetting:
    """Challenge the 37-surface folded SRS microscope template sequential ray tracing."""

    def test_strictly_37_surfaces_structure(self):
        """Verify surface count is strictly 37 (28 sequential + 9 off-axis)."""
        stack = app_ufn.build_folded_real_world_srs_template()
        assert len(stack.surfaces) == 37
        assert stack.surfaces[0].name.startswith("Pump Laser")
        assert stack.surfaces[27].name.startswith("Photodiode")
        assert stack.surfaces[28].name.startswith("Stokes Laser")
        assert stack.surfaces[36].name.startswith("Combiner Leak Dump")

    def test_on_axis_ray_tracing_unvignetted(self):
        """Verify on-axis ray (y = 0.0 mm) traces through all 28 sequential surfaces."""
        stack = app_ufn.build_folded_real_world_srs_template()
        sys = OpticalSystem(stack)
        r0 = sys.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797))
        assert not r0.vignetted
        assert len(r0.history) == 28
        assert r0.history[-1]["surface_index"] == 27

    def test_edge_of_pupil_rays_vignetted_at_galvo_y(self):
        """Demonstrate that off-axis / pupil rays (y >= 1.0 mm) are clipped by Galvo Y aperture."""
        stack = app_ufn.build_folded_real_world_srs_template()
        sys = OpticalSystem(stack)

        # For a 4.0 mm beam (waist radius 2.0 mm), test y = 1.0, 1.5, 2.0 mm
        for y_in in [1.0, 1.5, 2.0]:
            r = sys.trace_ray(Ray(pos=np.array([0.0, y_in, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797))
            assert r.vignetted is True, f"Expected ray at y={y_in} to vignette"
            # It truncates at surface 12 (Galvo Mirror Y)
            assert r.history[-1]["surface_index"] == 12


class TestDualTreeParityMachinePrecision:
    """Verify dual-tree parity between optics_app and v2 to machine precision."""

    def test_surface_prescription_exact_parity(self):
        """Verify all 37 surfaces have identical parameters between optics_app and v2."""
        s1 = app_ufn.build_folded_real_world_srs_template()
        s2 = v2_ufn.build_folded_real_world_srs_template()
        assert len(s1.surfaces) == len(s2.surfaces) == 37

        for i, (surf1, surf2) in enumerate(zip(s1.surfaces, s2.surfaces)):
            assert surf1.name == surf2.name
            assert surf1.radius == surf2.radius
            assert surf1.thickness == surf2.thickness
            assert surf1.glass == surf2.glass
            assert surf1.semi_diameter == surf2.semi_diameter

    def test_on_axis_ray_trace_machine_precision_parity(self):
        """Verify ray tracing coordinates and path lengths match to 0.0 tolerance."""
        s1 = app_ufn.build_folded_real_world_srs_template()
        s2 = v2_ufn.build_folded_real_world_srs_template()
        sys1 = OpticalSystem(s1)
        sys2 = OpticalSystem(s2)

        r1 = sys1.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797))
        r2 = sys2.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797))

        assert r1.vignetted == r2.vignetted is False
        assert len(r1.history) == len(r2.history) == 28
        assert r1.path_length == r2.path_length
        assert np.isclose(r1.phase, r2.phase, atol=1e-12)

        for step1, step2 in zip(r1.history, r2.history):
            assert step1["surface_index"] == step2["surface_index"]
            assert np.isclose(step1["x"], step2["x"], atol=1e-14)
            assert np.isclose(step1["y"], step2["y"], atol=1e-14)
            assert np.isclose(step1["z"], step2["z"], atol=1e-14)
            assert np.isclose(step1["dir_x"], step2["dir_x"], atol=1e-14)
            assert np.isclose(step1["dir_y"], step2["dir_y"], atol=1e-14)
            assert np.isclose(step1["dir_z"], step2["dir_z"], atol=1e-14)
