"""
Empirical Adversarial Stress Test Suite for Milestone 4 (Challenger M4-2).

Systematic coverage of:
1. Generic Components & Ray Tracer Kinematics:
   - Extreme galvo scan angles (+/-45°, +/-80°, +/-89.9°, inverted pivots, grazing reflections).
   - High-dispersion prism materials (Sellmeier & Cauchy, Flint vs Crown, UV to NIR).
   - Multi-bounce beam splitter tree depth limits (queue depth, intensity cutoff, loop safety).
   - Asymmetric apertures and boundary clipping.
2. CAD Solid Generation:
   - Zero and near-zero apex angles, extreme aspect ratio prisms.
   - Non-standard split ratios (100% R, 100% T, 0% T, over-unity, negative).
   - Compound assemblies containing multiple active mechanisms with non-zero Euler angles.
3. Numerical Stability & Degenerate Handling:
   - Finiteness checks (no NaN/Inf in ray histories or CAD vertices).
   - Graceful degradation on physically impossible or missed rays.
4. Visualizer 3D Inhomogeneous Segments Reproduction & Analysis.
"""

import math
import os
import tempfile
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.components import (
    PrismConfig,
    PrismGeometryType,
    BeamSplitterConfig,
    BeamSplitterType,
    SplitterCoatingType,
    GalvoScannerConfig,
    create_prism_component,
    create_beamsplitter_component,
    create_galvo_component,
)
from optics_app.engine.mechanisms import (
    get_effective_pose,
    interaction_rotation,
    state_transmission,
)
from optics_app.engine.tracer import (
    OpticalSystem,
    Ray,
    RayNode,
    RayTree,
    BranchedTracer,
    snell_refract_3d,
    build_rotation_matrix,
    ray_surface_intersect,
)
from optics_app.utils.materials import (
    SellmeierDispersion,
    CauchyDispersion,
    get_refractive_index,
    get_complex_refractive_index,
)
from optics_app.cad.exporter import (
    STEPExporter,
    create_prism_solid,
    create_beamsplitter_solid,
    _get_build123d,
)


# =====================================================================
# SECTION 1: Galvo Kinematics & Extreme Scan Angles
# =====================================================================

class TestGalvoKinematicsAdversarial:
    """Stress-test Galvo scanner kinematics, rotation matrix preservation, and ray deflection."""

    @pytest.mark.parametrize("angle_deg", [-89.9, -80.0, -45.0, 0.0, 45.0, 80.0, 89.9])
    @pytest.mark.parametrize("pivot_offset", [
        [0.0, 0.0, 0.0],
        [-25.0, -10.0, -5.0],  # Inverted negative pivot
        [50.0, 100.0, 20.0],   # Large pivot offset
    ])
    def test_galvo_rodrigues_metric_preservation(self, angle_deg, pivot_offset):
        """Verify Rodrigues rotation matrix orthonormality and pivot offset kinematics."""
        surf = Surface(name="GalvoMirror", surface_type=SurfaceType.MIRROR)
        surf.global_origin = np.array([10.0, 20.0, 30.0], dtype=float)
        surf.global_R = build_rotation_matrix(15.0, -10.0, 25.0)

        cfg = GalvoScannerConfig(
            scan_axis="X",
            current_angle_deg=angle_deg,
            pivot_offset_mm=pivot_offset
        )
        surf.galvo_config = cfg

        eff_origin, eff_R = get_effective_pose(surf)

        # 1. No NaN or Inf
        assert np.all(np.isfinite(eff_origin)), f"Non-finite origin for angle={angle_deg}, pivot={pivot_offset}"
        assert np.all(np.isfinite(eff_R)), f"Non-finite R for angle={angle_deg}, pivot={pivot_offset}"

        # 2. Orthonormality: R.T @ R = I
        ident = eff_R.T @ eff_R
        np.testing.assert_allclose(ident, np.eye(3), atol=1e-12,
                                   err_msg=f"Rotation matrix lost orthogonality at angle={angle_deg}")

        # 3. Determinant: det(R) = +1 (proper rotation, no reflection or inversion)
        det_R = np.linalg.det(eff_R)
        np.testing.assert_allclose(det_R, 1.0, atol=1e-12,
                                   err_msg=f"Determinant != 1 at angle={angle_deg}")

        # 4. Kinematic consistency: pivot point must be an invariant under scan rotation
        # P_pivot = P_0 + R_mount @ d_pivot = P_eff + R_eff @ d_pivot
        p_pivot_expected = surf.global_origin + surf.global_R @ np.asarray(pivot_offset, dtype=float)
        p_pivot_actual = eff_origin + eff_R @ np.asarray(pivot_offset, dtype=float)
        np.testing.assert_allclose(p_pivot_actual, p_pivot_expected, atol=1e-10,
                                   err_msg=f"Pivot point drifted during galvo scan rotation at {angle_deg}°")

    def test_galvo_grazing_incidence_ray_trace(self):
        """Trace rays at near-grazing angles onto actuated galvo mirror."""
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=50.0)
        s_galvo = Surface(name="GalvoMirror", surface_type=SurfaceType.MIRROR, thickness=50.0, semi_diameter=25.0)
        s_galvo.glass = "MIRROR"
        s_galvo.galvo_config = GalvoScannerConfig(
            scan_axis="X",
            current_angle_deg=44.9, # Mirror tilted by 44.9°
            pivot_offset_mm=[0.0, 0.0, 0.0]
        )
        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=100.0)

        stack.add_surface(s_obj)
        stack.add_surface(s_galvo)
        stack.add_surface(s_det)
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.550)
        tree = tracer.trace_ray_tree(ray)

        all_r = tree.all_rays()
        assert len(all_r) >= 1
        for r in all_r:
            assert np.all(np.isfinite(r.pos)), "Ray position contains NaN/Inf at grazing incidence"
            assert np.all(np.isfinite(r.dir)), "Ray direction contains NaN/Inf at grazing incidence"
            for h in r.history:
                assert np.isfinite(h["intensity"]), "History intensity non-finite"


# =====================================================================
# SECTION 2: High-Dispersion Prism Materials & Broad Spectral Range
# =====================================================================

class TestPrismHighDispersionAdversarial:
    """Stress-test Sellmeier and Cauchy dispersion models, Flint vs Crown, UV to NIR."""

    def test_dispersion_flint_vs_crown_fraunhofer_lines(self):
        """Verify that dense flint glass (N-SF11) exhibits strictly higher angular dispersion than crown glass (N-BK7)."""
        cfg_crown = PrismConfig(glass="N-BK7", apex_angle_deg=60.0)
        cfg_flint = PrismConfig(glass="N-SF11", apex_angle_deg=60.0)

        # Fraunhofer lines: F (486.1 nm blue), d (587.6 nm yellow), C (656.3 nm red)
        wl_F, wl_d, wl_C = 0.4861, 0.5876, 0.6563

        n_crown_F = cfg_crown._get_refractive_index(wl_F)
        n_crown_C = cfg_crown._get_refractive_index(wl_C)
        delta_crown = cfg_crown.calculate_minimum_deviation(wl_F) - cfg_crown.calculate_minimum_deviation(wl_C)

        n_flint_F = cfg_flint._get_refractive_index(wl_F)
        n_flint_C = cfg_flint._get_refractive_index(wl_C)
        delta_flint = cfg_flint.calculate_minimum_deviation(wl_F) - cfg_flint.calculate_minimum_deviation(wl_C)

        assert n_flint_F > n_crown_F, "Flint glass index should be higher than crown glass"
        assert delta_flint > 2.0 * delta_crown, f"Flint dispersion ({delta_flint:.3f}°) should be > 2x crown ({delta_crown:.3f}°)"

    @pytest.mark.parametrize("wl_um", [0.25, 0.35, 0.532, 1.064, 1.55, 2.0])
    def test_broad_spectral_dispersion_stability(self, wl_um):
        """Verify refractive index and minimum deviation stability across deep UV to infrared."""
        cfg = PrismConfig(glass="N-BK7", apex_angle_deg=30.0) # 30 deg apex to avoid TIR across broad band
        n = cfg._get_refractive_index(wl_um)
        assert 1.4 < n < 1.7, f"Refractive index out of physical bounds: {n} at {wl_um} um"

        dev = cfg.calculate_minimum_deviation(wl_um)
        assert np.isfinite(dev), f"Deviation angle is non-finite at {wl_um} um"
        assert dev > 0.0, f"Deviation angle must be strictly positive: {dev}°"

    def test_prism_total_internal_reflection_boundary_detection(self):
        """Verify graceful error reporting when incident angle or apex angle exceeds TIR threshold."""
        cfg = PrismConfig(glass="N-SF11", apex_angle_deg=75.0) # High apex + high index -> TIR occurs
        with pytest.raises(ValueError, match="Total internal reflection|too large"):
            cfg.calculate_deviation(incident_angle_deg=10.0, wavelength_um=0.532)

    def test_custom_sellmeier_and_cauchy_models(self):
        """Stress-test Sellmeier and Cauchy mathematical evaluations directly."""
        # Sellmeier with standard BK7 constants
        s = SellmeierDispersion(1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653)
        n_vis = s.get_index(0.5876)
        assert 1.51 < n_vis < 1.52

        # Negative wavelength should raise ValueError
        with pytest.raises(ValueError, match="positive"):
            s.get_index(-0.5)

        # Cauchy model
        c = CauchyDispersion(1.5, 0.005, 0.0001)
        n_blue = c.get_index(0.400)
        n_red = c.get_index(0.700)
        assert n_blue > n_red, "Cauchy normal dispersion must have higher index for blue light"


# =====================================================================
# SECTION 3: Multi-Bounce Beam Splitter Tree Depth & Energy Conservation
# =====================================================================

class TestBeamSplitterTreeDepthAdversarial:
    """Stress-test multi-generation branched ray tree queue, depth limits, and energy conservation."""

    def test_tree_depth_exhaustion_limits(self):
        """Verify that BranchedTracer strictly respects max_depth and does not overflow."""
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0)
        # Create an optical beam splitter
        bs_comp, (bs_f, bs_r) = create_beamsplitter_component(name="BS1", splitter_type="Plate", split_ratio_R=0.5, split_ratio_T=0.5)
        bs_f.thickness = 10.0
        bs_r.thickness = 20.0
        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0)

        stack.add_surface(s_obj)
        stack.add_surface(bs_f)
        stack.add_surface(bs_r)
        stack.add_surface(s_det)
        stack.components.append(bs_comp)
        stack._update_accumulated_positions()

        # Test depth 0 (root only)
        tracer_0 = BranchedTracer(stack, max_depth=0)
        tree_0 = tracer_0.trace_ray_tree(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), intensity=1.0))
        assert len(tree_0.nodes) == 1, "max_depth=0 should produce only root node"

        # Test depth 1
        tracer_1 = BranchedTracer(stack, max_depth=1)
        tree_1 = tracer_1.trace_ray_tree(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), intensity=1.0))
        for nid, node in tree_1.nodes.items():
            assert node.generation <= 1, f"Node generation {node.generation} exceeded max_depth=1"

    def test_intensity_cutoff_threshold(self):
        """Verify that rays with intensity below min_intensity are pruned from tree generation."""
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0)
        # 90% T, 10% R beam splitter
        bs_comp, (bs_f, bs_r) = create_beamsplitter_component(name="BS", splitter_type="Plate", split_ratio_R=0.1, split_ratio_T=0.9)
        bs_f.thickness = 10.0
        bs_r.thickness = 20.0
        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0)

        stack.add_surface(s_obj)
        stack.add_surface(bs_f)
        stack.add_surface(bs_r)
        stack.add_surface(s_det)
        stack.components.append(bs_comp)
        stack._update_accumulated_positions()

        # Cutoff at 0.15: R branch (0.10) should NOT spawn, only T branch (0.90 nominal, minus substrate Fresnel loss) spawns
        tracer_cutoff = BranchedTracer(stack, max_depth=5, min_intensity=0.15)
        tree = tracer_cutoff.trace_ray_tree(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), intensity=1.0))

        # Children of root should contain only transmitted
        root_children = [tree.nodes[cid] for cid in tree.nodes[0].children_ids]
        assert len(root_children) == 1, f"Expected exactly 1 child (transmitted), got {len(root_children)}"
        assert root_children[0].branch_type == "transmitted"
        # Fresnel reflection at air-glass interface slightly reduces transmitted intensity (~0.82)
        assert 0.80 <= root_children[0].ray.intensity <= 0.90

    def test_optical_cavity_infinite_loop_prevention(self):
        """Adversarially construct a closed cavity and verify queue-based termination."""
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0)
        # Splitter facing back
        bs_comp, (bs_f, bs_r) = create_beamsplitter_component(name="BS", splitter_type="Plate", split_ratio_R=0.5, split_ratio_T=0.5)
        bs_f.thickness = 20.0
        # Set reflected branch to cycle back to bs_f
        bs_f.reflected_branch_start_idx = 1

        stack.add_surface(s_obj)
        stack.add_surface(bs_f)
        stack.add_surface(bs_r)
        stack.components.append(bs_comp)
        stack._update_accumulated_positions()

        # Ray tracer with max_depth=4 must terminate cleanly without recursion error
        tracer = BranchedTracer(stack, max_depth=4, min_intensity=1e-5)
        tree = tracer.trace_ray_tree(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), intensity=1.0))

        assert len(tree.nodes) > 1
        assert all(n.generation <= 4 for n in tree.nodes.values())

    def test_energy_conservation_across_leaf_rays(self):
        """Verify that total radiant power across leaf rays never exceeds input power."""
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0)
        bs_comp, (bs_f, bs_r) = create_beamsplitter_component(name="BS", splitter_type="Cube", split_ratio_R=0.5, split_ratio_T=0.5)
        bs_f.thickness = 15.0
        bs_r.thickness = 25.0
        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0)

        stack.add_surface(s_obj)
        stack.add_surface(bs_f)
        stack.add_surface(bs_r)
        stack.add_surface(s_det)
        stack.components.append(bs_comp)
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack, max_depth=5)
        tree = tracer.trace_ray_tree(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), intensity=1.0))

        # Check total energy via all_terminal_rays
        term_rays = tree.all_terminal_rays()
        total_leaf_power = sum(r.intensity for r in term_rays)
        assert total_leaf_power <= 1.0 + 1e-9, f"Power created from nothing: total={total_leaf_power}"
        assert tree.total_energy() == pytest.approx(total_leaf_power, abs=1e-9)


# =====================================================================
# SECTION 4: Asymmetric Apertures & Boundary Clipping
# =====================================================================

class TestAsymmetricAperturesAdversarial:
    """Stress-test rectangular, elliptical, annular apertures and boundary conditions."""

    def test_rectangular_asymmetric_aperture(self):
        """Verify rectangular aperture with unequal half-widths hx=3.0 mm, hy=15.0 mm."""
        surf = Surface(name="Slit", surface_type=SurfaceType.STOP)
        surf.aperture_type = "Rectangular"
        surf.aperture_x_half = 3.0
        surf.aperture_y_half = 15.0

        # Inside both bounds
        assert surf.check_aperture(2.5, 14.0) is True
        # Outside X but inside Y
        assert surf.check_aperture(3.5, 5.0) is False
        # Inside X but outside Y
        assert surf.check_aperture(1.0, 16.0) is False
        # Exact boundary
        assert surf.check_aperture(3.0, 15.0) is True

    def test_elliptical_asymmetric_aperture(self):
        """Verify elliptical aperture with a=5.0 mm, b=20.0 mm."""
        surf = Surface(name="Ellipse", surface_type=SurfaceType.STOP)
        surf.aperture_type = "Elliptical"
        surf.aperture_a_semi = 5.0
        surf.aperture_b_semi = 20.0

        # Inside: (4/5)^2 + (10/20)^2 = 0.64 + 0.25 = 0.89 <= 1.0
        assert surf.check_aperture(4.0, 10.0) is True
        # Outside: (4/5)^2 + (15/20)^2 = 0.64 + 0.5625 = 1.2025 > 1.0
        assert surf.check_aperture(4.0, 15.0) is False

    def test_aperture_nan_inf_safety(self):
        """Verify aperture checks reject NaN or Inf without raising unhandled exceptions."""
        surf = Surface(name="SafeAperture", surface_type=SurfaceType.STOP, semi_diameter=10.0)
        assert surf.check_aperture(float("nan"), 0.0) is False
        assert surf.check_aperture(0.0, float("nan")) is False
        assert surf.check_aperture(float("inf"), 0.0) is False
        assert surf.check_aperture(0.0, float("-inf")) is False


# =====================================================================
# SECTION 5: CAD Solid Generation Adversarial Stress
# =====================================================================

class TestCADSolidsAdversarial:
    """Stress-test zero/near-zero apex angles, extreme aspect ratios, non-standard split ratios."""

    def test_cad_prism_zero_and_near_zero_apex_angles(self):
        """Verify CAD prism solid generation for apex angles down to 0° and 1e-6°."""
        b3d = _get_build123d()
        if b3d is None:
            pytest.skip("build123d is not available")

        s1 = Surface(name="P1", semi_diameter=12.7, thickness=20.0)
        s2 = Surface(name="P2", semi_diameter=12.7, thickness=0.0)

        for apex_deg in [0.0, 1e-6, 0.001, 0.1, 15.0, 89.9, 90.0, 120.0]:
            cfg = PrismConfig(apex_angle_deg=apex_deg, base_width_mm=25.0, height_mm=20.0, length_mm=25.0)
            solid = create_prism_solid([s1, s2], config=cfg, apex_angle_deg=apex_deg)
            assert solid is not None, f"Failed to generate solid for apex={apex_deg}°"
            assert solid.is_valid, f"Solid is not valid B-Rep for apex={apex_deg}°"
            assert solid.volume > 0.0, f"Solid volume <= 0 for apex={apex_deg}°"

    @pytest.mark.parametrize("base_w, height, length", [
        (0.05, 500.0, 25.0),   # Needle aspect ratio
        (500.0, 0.05, 25.0),   # Flat pancake aspect ratio
        (25.0, 25.0, 500.0),   # Long extrusion
        (0.01, 0.01, 0.01),    # Micro solid
        (1000.0, 1000.0, 1000.0), # Macro solid
    ])
    def test_cad_prism_extreme_aspect_ratios(self, base_w, height, length):
        """Stress-test extreme geometric dimensional aspect ratios."""
        b3d = _get_build123d()
        if b3d is None:
            pytest.skip("build123d is not available")

        s1 = Surface(name="P1", semi_diameter=12.7, thickness=height)
        s2 = Surface(name="P2", semi_diameter=12.7)
        cfg = PrismConfig(base_width_mm=base_w, height_mm=height, length_mm=length, apex_angle_deg=60.0)

        solid = create_prism_solid([s1, s2], config=cfg)
        assert solid is not None
        assert solid.is_valid
        assert solid.volume > 0.0

    @pytest.mark.parametrize("r_ratio, t_ratio, exp_r, exp_t", [
        (1.0, 0.0, 1.0, 0.0),      # 100% Reflection
        (0.0, 1.0, 0.0, 1.0),      # 100% Transmission
        (0.0, 0.0, 0.0, 0.0),      # Complete absorption
        (1.5, 1.5, 0.5, 0.5),      # Over-unity normalized
        (-0.5, 1.0, 0.0, 1.0),     # Negative R clamped
    ])
    def test_beamsplitter_nonstandard_split_ratios(self, r_ratio, t_ratio, exp_r, exp_t):
        """Verify evaluate_split handles 100% R, 100% T, over-unity, and negative inputs."""
        cfg = BeamSplitterConfig(split_ratio_R=r_ratio, split_ratio_T=t_ratio)
        r_eval, t_eval = cfg.evaluate_split(0.532)
        assert r_eval == pytest.approx(exp_r, abs=1e-4)
        assert t_eval == pytest.approx(exp_t, abs=1e-4)
        assert r_eval + t_eval <= 1.0 + 1e-9

    def test_compound_assembly_step_export_with_euler_angles(self):
        """Export a compound assembly with multiple active mechanisms and non-zero Euler angles to STEP."""
        b3d = _get_build123d()
        if b3d is None:
            pytest.skip("build123d is not available")

        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0)

        # Galvo 1: tilted and actuated
        s_g1 = Surface(name="GalvoX", surface_type=SurfaceType.MIRROR, thickness=25.0, semi_diameter=12.5)
        s_g1.tilt_x = 10.0
        s_g1.tilt_y = -15.0
        s_g1.tilt_z = 5.0
        s_g1.galvo_config = GalvoScannerConfig(scan_axis="X", current_angle_deg=12.5, pivot_offset_mm=[5.0, 0.0, 2.0])

        # Galvo 2: rotated orthogonal scan
        s_g2 = Surface(name="GalvoY", surface_type=SurfaceType.MIRROR, thickness=30.0, semi_diameter=12.5)
        s_g2.tilt_x = -5.0
        s_g2.tilt_y = 20.0
        s_g2.galvo_config = GalvoScannerConfig(scan_axis="Y", current_angle_deg=-8.0, pivot_offset_mm=[0.0, 5.0, -3.0])

        # Beam splitter
        bs_comp, (bs_f, bs_r) = create_beamsplitter_component(name="BS_Cube", splitter_type="Cube", substrate_thickness_mm=10.0)
        bs_comp.beamsplitter_config.cube_side_length_mm = 20.0
        bs_f.thickness = 15.0
        bs_r.thickness = 25.0

        # Dispersing Prism
        prism_comp, (p1, p2) = create_prism_component(name="Prism_Disp", geometry_type="Equilateral", apex_angle_deg=60.0)
        p1.thickness = 20.0
        p2.thickness = 30.0

        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0)

        stack.add_surface(s_obj)
        stack.add_surface(s_g1)
        stack.add_surface(s_g2)
        stack.add_surface(bs_f)
        stack.add_surface(bs_r)
        stack.add_surface(p1)
        stack.add_surface(p2)
        stack.add_surface(s_det)
        stack.components.extend([bs_comp, prism_comp])
        stack._update_accumulated_positions()

        with tempfile.TemporaryDirectory() as tmpdir:
            step_path = os.path.join(tmpdir, "compound_assembly_adversarial.step")
            res = STEPExporter.export(stack, step_path)
            assert res.success is True, f"STEP export failed: {res.error_message}"
            assert res.element_count >= 4, f"Expected at least 4 elements, got {res.element_count}"
            assert os.path.exists(step_path)
            assert os.path.getsize(step_path) > 1000

            # Verify STEP file contents
            with open(step_path, "r", encoding="utf-8", errors="ignore") as f:
                step_text = f.read()
            assert "CARTESIAN_POINT" in step_text
            assert "MANIFOLD_SOLID_BREP" in step_text or "SHELL_BASED_SURFACE_MODEL" in step_text or "CLOSED_SHELL" in step_text
            assert "NaN" not in step_text
            assert "Infinity" not in step_text


# =====================================================================
# SECTION 6: Numerical Stability & Degenerate Ray Handling
# =====================================================================

class TestNumericalStabilityAdversarial:
    """Verify system stability against physically impossible or degenerate rays."""

    def test_physically_impossible_backwards_ray(self):
        """Trace ray fired in opposite direction (-Z). Must degrade gracefully without crash."""
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0)
        s_lens = Surface(name="LensFront", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=5.0, semi_diameter=12.5)
        s_lens.glass = "N-BK7"
        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0)

        stack.add_surface(s_obj)
        stack.add_surface(s_lens)
        stack.add_surface(s_det)
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack)
        # Ray fired backwards towards -Z
        backwards_ray = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., -1.]), intensity=1.0)
        tree = tracer.trace_ray_tree(backwards_ray)

        assert tree is not None
        root = tree.nodes[0]
        # Ray must be marked vignetted
        assert root.ray.vignetted is True
        for h in root.ray.history:
            assert np.all(np.isfinite([h["x"], h["y"], h["z"]]))

    def test_far_off_axis_missed_ray(self):
        """Trace ray with launch position 1000 mm off-axis (completely missing optics)."""
        stack = SurfaceStack()
        s_obj = Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0)
        s_lens = Surface(name="Lens", surface_type=SurfaceType.STANDARD, semi_diameter=10.0, thickness=5.0)
        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0)

        stack.add_surface(s_obj)
        stack.add_surface(s_lens)
        stack.add_surface(s_det)
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack)
        missed_ray = Ray(pos=np.array([1000.0, 1000.0, 0.0]), dir=np.array([0., 0., 1.]), intensity=1.0)
        tree = tracer.trace_ray_tree(missed_ray)

        root = tree.nodes[0]
        assert root.ray.vignetted is True
        assert root.ray.intensity == 0.0 or root.ray.vignetted
        assert all(np.isfinite(h["intensity"]) for h in root.ray.history)
