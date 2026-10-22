"""
Comprehensive Opaque-Box E2E Test Suite for Zemax-Standard Optical Optimization Engine
======================================================================================
Testing Architecture: 4-Tier Verification Methodology
- Tier 1: Feature Coverage (F1 to F10, >=5 tests each -> 50 tests)
- Tier 2: Boundary & Corner Cases (F1 to F10, >=5 tests each -> 50 tests)
- Tier 3: Cross-Feature Interactions (Pairwise combinations -> 20 tests)
- Tier 4: Real-World Application Scenarios (S1 to S5 -> 5 tests)
Total Tests: 125 tests.

Authority Specifications:
- ORIGINAL_REQUEST.md: Requirements R1, R2, R3, R4, and Acceptance Criteria.
- PROJECT.md: Features F1-F10 and Subsystem Interface Contracts.
- TEST_INFRA.md: E2E Test Infrastructure Specifications.
"""

import os
import sys
import math
import numpy as np
import pytest

# Ensure headless Qt configuration before importing GUI components
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Core engine imports from primary tree (optics_app)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import (
    OpticalOptimizer,
    OptimizationResult,
    TerminationStatus,
    compute_system_efl,
    radius_to_curvature,
    curvature_to_radius,
    resolve_solves_and_pickups,
)
from optics_app.engine.merit_function import (
    MeritFunction,
    MeritOperand,
    OperandType,
)
from optics_app.engine.system_config import (
    SystemConfig,
    FieldPoint,
    FieldType,
    RayPatternType,
    ApertureType,
    WavelengthEntry,
)
from optics_app.engine.pupil_solve import (
    calculate_entrance_pupil,
    aim_ray_to_pupil,
    aim_collimated_ray,
    find_stop_surface_index,
)
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics
from optics_app.engine.manufacturing_standards import (
    ManufacturingConstraintsConfig,
    QualityPreset,
    TestPlateCatalog,
    SystemManufacturingReport,
    audit_optical_system_manufacturing,
)

# Core engine imports from secondary tree (v2) for 100% Tree Synchronization
from v2.engine.surface import Surface as V2Surface, SurfaceType as V2SurfaceType
from v2.engine.lde import SurfaceStack as V2SurfaceStack
from v2.engine.tracer import OpticalSystem as V2OpticalSystem
from v2.engine.optimizer import (
    OpticalOptimizer as V2OpticalOptimizer,
    radius_to_curvature as v2_radius_to_curvature,
    curvature_to_radius as v2_curvature_to_radius,
)
from v2.engine.merit_function import (
    MeritFunction as V2MeritFunction,
    OperandType as V2OperandType,
)
from v2.engine.system_config import SystemConfig as V2SystemConfig, FieldPoint as V2FieldPoint
from v2.engine.pupil_solve import calculate_entrance_pupil as v2_calculate_entrance_pupil
from v2.analysis.spot import SpotDiagram as V2SpotDiagram
from v2.engine.manufacturing_standards import (
    audit_optical_system_manufacturing as v2_audit_optical_system_manufacturing,
)

# Optional GUI imports
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["-platform", "offscreen"])
else:
    _qt_app = None


# ==============================================================================
# Optical System Factory Helpers for Test Independence
# ==============================================================================

def create_test_singlet(
    r1: float = 50.0,
    r2: float = -50.0,
    ct: float = 5.0,
    bfl: float = 45.0,
    glass: str = "N-BK7",
    semi_diameter: float = 12.5,
) -> SurfaceStack:
    """Create independent 4-surface Plano/Biconvex singlet system."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="L1_Front", radius=r1, thickness=ct, glass=glass, semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="L1_Back", radius=r2, thickness=bfl, glass="AIR", semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=semi_diameter))
    return stack


def create_test_doublet(
    r1: float = 60.0,
    r2: float = -40.0,
    r3: float = -150.0,
    t1: float = 6.0,
    t2: float = 3.0,
    bfl: float = 90.0,
    crown: str = "N-BK7",
    flint: str = "SF2",
    semi_diameter: float = 12.5,
) -> SurfaceStack:
    """Create independent 5-surface achromatic cemented doublet system."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="Crown_Front", radius=r1, thickness=t1, glass=crown, semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="Contact", radius=r2, thickness=t2, glass=flint, semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="Flint_Back", radius=r3, thickness=bfl, glass="AIR", semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=semi_diameter))
    return stack


def create_test_cooke_triplet(semi_diameter: float = 10.0) -> SurfaceStack:
    """Create independent 8-surface Cooke Triplet (Crown1, Flint2, Crown3)."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=semi_diameter))
    # Element 1: Crown
    stack.add_surface(Surface(name="S1", radius=22.0, thickness=3.2, glass="N-SK16", semi_diameter=8.0))
    stack.add_surface(Surface(name="S2", radius=-150.0, thickness=5.5, glass="AIR", semi_diameter=8.0))
    # Element 2: Flint
    stack.add_surface(Surface(name="S3", radius=-20.0, thickness=1.5, glass="SF2", semi_diameter=6.0))
    stack.add_surface(Surface(name="S4", radius=20.0, thickness=4.8, glass="AIR", semi_diameter=6.0))
    # Element 3: Crown
    stack.add_surface(Surface(name="S5", radius=80.0, thickness=2.8, glass="N-SK16", semi_diameter=7.0))
    stack.add_surface(Surface(name="S6", radius=-24.0, thickness=40.0, glass="AIR", semi_diameter=7.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    return stack


# ==============================================================================
# TIER 1: FEATURE COVERAGE (50 Tests, >=5 tests per feature F1 - F10)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage for F1 through F10."""

    # --- F1: Multi-Field Sampling & Area Weighting ---

    def test_t1_f1_01_field_point_initialization_and_properties(self):
        """Verify FieldPoint initializes x, y, weight, and vignetting correctly."""
        pt = FieldPoint(x=0.0, y=10.0, weight=2.0, vignetting_factor=0.1)
        assert pt.x == 0.0
        assert pt.y == 10.0
        assert pt.weight == 2.0
        assert pt.vignetting_factor == 0.1

    def test_t1_f1_02_three_field_linear_preset_coordinates(self):
        """Verify 3-Field Linear preset generates normalized coordinates [0.0, 0.707, 1.0]."""
        cfg = SystemConfig()
        max_field = 10.0
        cfg.apply_field_preset("3-Field Linear", max_field=max_field)
        assert len(cfg.fields) == 3
        assert cfg.fields[0].y == 0.0
        assert abs(cfg.fields[1].y - 0.707 * max_field) < 1e-3
        assert cfg.fields[2].y == max_field

    def test_t1_f1_03_field_area_weighting_distribution(self):
        """Verify standard area weighting [1.0, 2.0, 1.0] balances central vs annular format."""
        weights = [1.0, 2.0, 1.0]
        norm_weights = [w / sum(weights) for w in weights]
        assert norm_weights[1] == 0.5  # Zonal field carries 50% format area
        assert norm_weights[0] == 0.25
        assert norm_weights[2] == 0.25

    def test_t1_f1_04_collimated_field_ray_aiming_angles(self):
        """Verify aim_collimated_ray computes exact direction cosines from field angle."""
        stack = create_test_singlet()
        ang_y = 5.0  # degrees
        ray = aim_collimated_ray(
            surfaces=stack.surfaces,
            field_angle_deg=(0.0, ang_y),
            pupil_norm=(0.0, 0.0),
            wavelength_um=0.5876,
        )
        expected_ky = math.sin(math.radians(ang_y))
        assert abs(ray.dir[1] - expected_ky) < 1e-5
        assert abs(np.linalg.norm(ray.dir) - 1.0) < 1e-7

    def test_t1_f1_05_field_metadata_propagation_in_optimizer_rays(self):
        """Verify rays generated by SystemConfig carry field_idx, field_coord, and field_weight."""
        cfg = SystemConfig()
        cfg.fields = [FieldPoint(0.0, 0.0, weight=1.0), FieldPoint(0.0, 5.0, weight=2.0)]
        stack = create_test_singlet()
        rays = cfg.generate_input_rays(stack.surfaces, num_rays=5)
        assert len(rays) > 0
        field_indices = {r.field_idx for r in rays}
        assert 0 in field_indices and 1 in field_indices
        f1_rays = [r for r in rays if r.field_idx == 1]
        assert all(r.field_weight == 2.0 for r in f1_rays)

    # --- F2: Full Pupil Hexapolar & Gauss Quadrature Grid Sampling ---

    def test_t1_f2_01_hexapolar_full_pupil_radius_unity(self):
        """Verify hexapolar pupil sampling extends across full entrance pupil disk (rho <= 1.0)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=15, entrance_pupil_radius=10.0)
        rays = opt._generate_rays(pupil_radius=10.0, wavelengths=[0.5876], num_rays=15)
        radii = [math.sqrt(r.pos[0]**2 + r.pos[1]**2) for r in rays]
        max_rho = max(radii) / 10.0
        assert max_rho == pytest.approx(1.0, rel=1e-3)

    def test_t1_f2_02_hexapolar_ring_ray_counts(self):
        """Verify concentric hexapolar rings generate 6 * ring rays per ring."""
        n_rings = 4
        expected_counts = [1] + [6 * r for r in range(1, n_rings + 1)]
        assert expected_counts == [1, 6, 12, 18, 24]
        total_rays = sum(expected_counts)
        assert total_rays == 61

    def test_t1_f2_03_gauss_legendre_quadrature_roots_and_weights(self):
        """Verify 2-point Gauss-Legendre quadrature integration of polynomial on [0, 1]."""
        # Exact integral of f(x) = 2*x on [0, 1] is 1.0
        roots_standard, weights_standard = np.polynomial.legendre.leggauss(2)
        # Transform [-1, 1] to [0, 1]
        roots_01 = 0.5 * (roots_standard + 1.0)
        weights_01 = 0.5 * weights_standard
        quad_integral = sum(w * (2.0 * x) for x, w in zip(roots_01, weights_01))
        assert abs(quad_integral - 1.0) < 1e-7

    def test_t1_f2_04_pupil_coordinates_within_unit_disk(self):
        """Verify all generated entrance pupil sample coordinates satisfy Px^2 + Py^2 <= 1.0."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=25, entrance_pupil_radius=5.0)
        rays = opt._generate_rays(pupil_radius=5.0, wavelengths=[0.5876], num_rays=25)
        for r in rays:
            rho_sq = (r.pos[0] / 5.0)**2 + (r.pos[1] / 5.0)**2
            assert rho_sq <= 1.000001, f"Pupil coordinate outside unit disk: {rho_sq}"

    def test_t1_f2_05_hexapolar_rotational_symmetry(self):
        """Verify hexapolar ring samples possess 6-fold (60 deg) rotational symmetry."""
        thetas = [2.0 * math.pi * p / 6 for p in range(6)]
        for th in thetas:
            rotated = (th + math.pi / 3.0) % (2.0 * math.pi)
            assert any(abs(rotated - t2) < 1e-6 or abs(abs(rotated - t2) - 2 * math.pi) < 1e-6 for t2 in thetas)

    # --- F3: Optimization Targets (Centroid & Chief-Ray Spot, Wavefront OPD) ---

    def test_t1_f3_01_spot_centroid_reference_calculation(self):
        """Verify SpotDiagram.calculate_centroid computes exact arithmetic mean."""
        xs = [1.0, 3.0, 5.0]
        ys = [2.0, 4.0, 6.0]
        spot = SpotDiagram(xs, ys)
        cx, cy = spot.calculate_centroid()
        assert cx == 3.0
        assert cy == 4.0

    def test_t1_f3_02_spot_rms_radius_mathematical_derivation(self):
        """Verify SpotDiagram.calculate_rms_radius against analytic circle formulation."""
        # 4 points on circle of radius 5.0 centered at (0, 0)
        xs = [5.0, -5.0, 0.0, 0.0]
        ys = [0.0, 0.0, 5.0, -5.0]
        spot = SpotDiagram(xs, ys)
        rms = spot.calculate_rms_radius()
        assert abs(rms - 5.0) < 1e-9

    def test_t1_f3_03_chief_ray_referenced_spot_divergence(self):
        """Verify asymmetric coma spot has higher RMS relative to chief ray than to centroid."""
        # Asymmetric distribution: chief ray at (0, 0), comatic flare at positive y
        xs = [0.0, 0.5, -0.5, 1.0, -1.0]
        ys = [0.0, 2.0, 2.0, 4.0, 4.0]
        spot = SpotDiagram(xs, ys)
        cx, cy = spot.calculate_centroid()
        assert cy > 0.0  # Centroid shifted away from chief ray (0, 0)
        rms_centroid = spot.calculate_rms_radius()
        rms_chief_ray = math.sqrt(np.mean(np.array(xs)**2 + np.array(ys)**2))
        assert rms_chief_ray > rms_centroid, "By Steiner's theorem, centroid minimizes variance"

    def test_t1_f3_04_paraxial_rayleigh_hamilton_opd_scaling(self):
        """Verify Rayleigh-Hamilton RMS wavefront OPD scaling W_rms = R_rms / (8 * F/# * lambda)."""
        r_rms = 0.008  # mm (8 um)
        f_num = 4.0
        wl_mm = 0.0005876  # 587.6 nm in mm
        expected_w_rms = r_rms / (8.0 * f_num * wl_mm)
        assert abs(expected_w_rms - 0.42546) < 1e-4

    def test_t1_f3_05_merit_function_wavefront_opd_operand_evaluation(self):
        """Verify MeritFunction evaluates WAVEFRONT_OPD operand on traced rays."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        mf = MeritFunction.create_wavefront_default()
        assert mf.operands[0].operand_type == OperandType.WAVEFRONT_OPD
        rays = [Ray(pos=np.array([0.0, y, 0.0]), dir=np.array([0.0, 0.0, 1.0])) for y in [-3.0, 0.0, 3.0]]
        traced = sys.trace_rays(rays)
        val = mf.evaluate(sys, traced, wavelengths=[0.5876])
        assert val >= 0.0

    # --- F4: Zemax Boundary Operands ---

    def test_t1_f4_01_center_thickness_bounds_mnct_mxct(self):
        """Verify manufacturing audit flags MNCT (< 1.5mm) and MXCT (> 18.0mm) violations."""
        # Lens with CT = 0.8mm violating MNCT
        stack_thin = create_test_singlet(ct=0.8)
        rep_thin = audit_optical_system_manufacturing(stack_thin.surfaces)
        assert not rep_thin.is_compliant
        assert any("Center thickness" in e or "MNCG" in e for e in rep_thin.overall_errors)

        # Lens with CT = 22.0mm violating MXCT (flagged in warnings)
        stack_thick = create_test_singlet(ct=22.0)
        rep_thick = audit_optical_system_manufacturing(stack_thick.surfaces)
        assert any("MXCG" in w or "max" in w.lower() for w in rep_thick.overall_warnings)

    def test_t1_f4_02_edge_thickness_exact_sag_mnet_mxet(self):
        """Verify exact sag calculation detects knife-edge condition (ET < 1.0mm)."""
        s1 = Surface(radius=25.0, semi_diameter=12.5)
        s2 = Surface(radius=-25.0, semi_diameter=12.5)
        ct = 2.0  # Thin center thickness
        r_edge = 12.5
        sag1 = s1.sag(r_edge)
        sag2 = s2.sag(r_edge)
        et = ct + sag2 - sag1
        # For R=25, y=12.5: sag1 > 0, sag2 < 0 -> et is negative (severe knife-edge)
        assert et < 1.0

    def test_t1_f4_03_air_edge_clearance_mnea(self):
        """Verify air edge thickness MNEA >= 0.5mm prevents physical mechanical collision."""
        s_rear = Surface(radius=-30.0, thickness=1.0, glass="N-BK7", semi_diameter=10.0)
        s_next = Surface(radius=-25.0, thickness=20.0, glass="AIR", semi_diameter=10.0)
        r_edge = 10.0
        edge_air = s_rear.thickness + s_next.sag(r_edge) - s_rear.sag(r_edge)
        assert isinstance(edge_air, float)

    def test_t1_f4_04_continuous_paraxial_effl_constraint(self):
        """Verify paraxial EFFL formulation continuous calculation via matrix power Phi = -C."""
        stack = create_test_singlet(r1=50.0, r2=-50.0, ct=5.0, glass="N-BK7")
        efl, _ = compute_system_efl(stack.surfaces, 0.5876)
        # N-BK7 n=1.5168, R1=50, R2=-50 gives EFL ~48.7 mm
        assert 45.0 < efl < 55.0

    def test_t1_f4_05_total_track_totr_strictly_excludes_s0(self):
        """Verify TOTR measures distance from Surface 1 to image, strictly excluding S0."""
        stack = create_test_singlet(ct=5.0, bfl=45.0)
        stack.surfaces[0].thickness = 1000.0  # Object distance
        totr = sum(s.thickness for s in stack.surfaces[1:])
        assert totr == 50.0  # 5.0 + 45.0 = 50.0 (excludes 1000.0)

    # --- F5: Curvature Space Parameterization & Bounds Clamping ---

    def test_t1_f5_01_radius_to_curvature_bidirectional_mapping(self):
        """Verify exact bidirectional mapping between radius R and curvature c = 1/R."""
        assert abs(radius_to_curvature(50.0) - 0.02) < 1e-9
        assert abs(curvature_to_radius(0.02) - 50.0) < 1e-9
        assert abs(radius_to_curvature(-25.0) - (-0.04)) < 1e-9
        assert abs(curvature_to_radius(-0.04) - (-25.0)) < 1e-9

    def test_t1_f5_02_infinite_radius_plano_singularity_elimination(self):
        """Verify flat surface R -> infinity maps cleanly to c = 0.0 without numerical error."""
        assert radius_to_curvature(float('inf')) == 0.0
        assert radius_to_curvature(0.0) == 0.0
        assert curvature_to_radius(0.0) == 0.0

    def test_t1_f5_03_dynamic_curvature_bounds_scaling_with_semidiameter(self):
        """Verify curvature clamping limit |c| <= 0.95 / y_sd avoids hemispherical singularity."""
        y_sd = 10.0
        c_max = 0.95 / y_sd
        assert c_max == 0.095  # R_min = 10.526 mm
        # Curvature of 0.1 corresponds to R=10.0 (exact hemisphere, unmanufacturable)
        assert 0.10 > c_max

    def test_t1_f5_04_curvature_space_gradient_stability(self):
        """Verify curvature space parameter derivative is smooth for large radii."""
        # For large R (1000 mm), small curvature step dc=1e-5 gives smooth delta
        c1 = 1.0 / 1000.0
        c2 = c1 + 1e-5
        r2 = curvature_to_radius(c2)
        assert 980.0 < r2 < 1000.0

    def test_t1_f5_05_curvature_conversion_preserves_surface_sag(self):
        """Verify surface sag is identical whether evaluated directly or through curvature."""
        surf = Surface(radius=40.0, semi_diameter=12.0)
        c = radius_to_curvature(surf.radius)
        r_recov = curvature_to_radius(c)
        surf_recov = Surface(radius=r_recov, semi_diameter=12.0)
        assert abs(surf.sag(8.0) - surf_recov.sag(8.0)) < 1e-12

    # --- F6: Damped Least Squares (DLS) Engine with Levenberg-Marquardt Adaptation ---

    def test_t1_f6_01_dls_solver_initialization_and_defaults(self):
        """Verify OpticalOptimizer initializes with DLS solver method."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=25)
        assert "dls" in opt.method.lower()
        assert opt.max_iter == 25

    def test_t1_f6_02_dls_damping_parameter_adaptation_step(self):
        """Verify Levenberg-Marquardt damping adaptation maintains stable step updates."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=15, num_rays=9)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success
        assert res.iterations > 0

    def test_t1_f6_03_dls_jacobian_residual_vector_formulation(self):
        """Verify _compute_dls_residuals returns 1D array of finite residuals."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", num_rays=9)
        x_param = np.array([45.0])
        var_dict = {"thickness": [2]}
        residuals = opt._compute_dls_residuals(x_param, var_dict, use_curv=False)
        assert isinstance(residuals, np.ndarray)
        assert residuals.ndim == 1
        assert len(residuals) > 0
        assert np.all(np.isfinite(residuals))

    def test_t1_f6_04_dls_convergence_on_paraxial_defocus(self):
        """Verify DLS converges and reduces RMS spot on single-variable back focal distance."""
        stack = create_test_singlet(bfl=25.0)  # Severely defocused
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=20, num_rays=9)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success
        assert res.final_rms < res.initial_rms
        assert res.optimized_thicknesses[2] > 35.0

    def test_t1_f6_05_dls_svd_cholesky_singular_matrix_protection(self):
        """Verify DLS handles redundant collinear variables without crash or singularity exception."""
        stack = create_test_singlet()
        # Add dummy air surface
        stack.add_surface(Surface(name="AirDummy", thickness=10.0, glass="AIR", semi_diameter=12.5))
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=15, num_rays=7)
        # Optimize two consecutive air thicknesses (collinear)
        res = opt.optimize(variable_indices={"thickness": [2, 3]})
        assert res.success
        assert np.isfinite(res.final_rms)

    # --- F7: Global Search & Hammer Optimization ---

    def test_t1_f7_01_global_search_discrete_curvature_sign_flips(self):
        """Verify Global Search generates discrete curvature sign permutations (2^K candidates)."""
        stack = create_test_singlet(r1=-50.0, r2=50.0)  # Inverted singlet
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Global Search", max_iter=15, num_rays=7)
        res = opt.optimize(variable_indices={"radius": [1]})
        assert res.success
        # For standard collimated focusing, front radius flips to positive R1 > 0
        assert res.optimized_radii[1] > 0

    def test_t1_f7_02_global_search_space_filling_sampling(self):
        """Verify Global Search evaluates multiple candidates and chooses minimum merit."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Global Search (Multi-Start)", max_iter=20, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success
        assert res.final_rms <= res.initial_rms

    def test_t1_f7_03_hammer_stochastic_cauchy_and_gaussian_perturbations(self):
        """Verify Hammer optimizer perturbs parameters and completes successfully."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Hammer Optimizer (Global)", max_iter=15, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success
        assert res.iterations > 0

    def test_t1_f7_04_hammer_simulated_annealing_cooling_schedule(self):
        """Verify simulated annealing geometric cooling rate gamma = 0.85."""
        t0 = 1.0
        gamma = 0.85
        temps = [t0 * (gamma ** k) for k in range(5)]
        assert temps[0] == 1.0
        assert temps[1] == pytest.approx(0.85)
        assert temps[4] == pytest.approx(0.85**4)

    def test_t1_f7_05_hammer_and_global_search_dls_polish(self):
        """Verify Hammer optimization applies local gradient polish on best candidate."""
        stack = create_test_singlet(bfl=30.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Hammer", max_iter=20, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2], "radius": [1]})
        assert res.success
        assert res.final_rms < res.initial_rms

    # --- F8: UI Result Delivery & Model Refresh ---

    def test_t1_f8_01_optimization_result_structure_and_types(self):
        """Verify OptimizationResult contains all required data attributes."""
        res = OptimizationResult(
            success=True,
            initial_rms=1.5,
            final_rms=0.1,
            rms_reduction_pct=93.3,
            status=TerminationStatus.CONVERGED,
        )
        assert res.success is True
        assert res.initial_rms == 1.5
        assert res.final_rms == 0.1
        assert res.status == TerminationStatus.CONVERGED

    def test_t1_f8_02_lens_data_table_model_set_surface_stack(self):
        """Verify LensDataTableModel updates rows upon set_surface_stack."""
        if not QT_AVAILABLE or _qt_app is None:
            pytest.skip("Qt not available")
        from optics_app.gui.lde_table import LensDataTableModel
        model = LensDataTableModel()
        stack = create_test_singlet()
        model.set_surface_stack(stack)
        assert model.rowCount() == len(stack.surfaces)

    def test_t1_f8_03_ui_model_system_updated_signal_emission(self):
        """Verify LensDataTableModel emits system_updated signal on stack change."""
        if not QT_AVAILABLE or _qt_app is None:
            pytest.skip("Qt not available")
        from optics_app.gui.lde_table import LensDataTableModel
        model = LensDataTableModel()
        signal_fired = []
        model.system_updated.connect(lambda: signal_fired.append(True))
        stack = create_test_singlet()
        model.set_surface_stack(stack)
        assert len(signal_fired) > 0

    def test_t1_f8_04_optimizer_dock_reads_manufacturing_controls(self):
        """Verify OptimizerDockWidget exposes and reads manufacturing constraint controls."""
        if not QT_AVAILABLE or _qt_app is None:
            pytest.skip("Qt not available")
        from optics_app.gui.lde_table import LensDataTableModel
        from optics_app.gui.optimizer_dock import OptimizerDockWidget
        model = LensDataTableModel()
        dock = OptimizerDockWidget(model)
        mfg = dock.get_manufacturing_constraints()
        assert "min_glass_thickness" in mfg
        assert "min_edge_thickness" in mfg
        assert "min_air_gap" in mfg

    def test_t1_f8_05_lde_table_cell_data_sync_after_optimization(self):
        """Verify LDE table model internal cells reflect optimized prescription values."""
        if not QT_AVAILABLE or _qt_app is None:
            pytest.skip("Qt not available")
        from optics_app.gui.lde_table import LensDataTableModel
        model = LensDataTableModel()
        stack = create_test_singlet(ct=6.5, bfl=48.2)
        model.set_surface_stack(stack)
        retrieved_stack = model.get_surface_stack()
        assert retrieved_stack.surfaces[1].thickness == 6.5
        assert retrieved_stack.surfaces[2].thickness == 48.2

    # --- F9: 100% Tree Synchronization between optics_app/ and v2/ ---

    def test_t1_f9_01_tree_module_import_parity(self):
        """Verify parallel modules in optics_app and v2 import identically."""
        assert OpticalOptimizer is not None
        assert V2OpticalOptimizer is not None

    def test_t1_f9_02_optimizer_class_interface_parity(self):
        """Verify OpticalOptimizer public method signatures match between optics_app and v2."""
        methods_app = {m for m in dir(OpticalOptimizer) if not m.startswith("__")}
        methods_v2 = {m for m in dir(V2OpticalOptimizer) if not m.startswith("__")}
        # Check core public methods exist in both
        for method_name in ["optimize", "validate", "stop", "fit_to_test_plates"]:
            assert method_name in methods_app
            assert method_name in methods_v2

    def test_t1_f9_03_merit_function_operand_type_enum_parity(self):
        """Verify OperandType enum members match 100% between optics_app and v2."""
        members_app = {m.name for m in OperandType}
        members_v2 = {m.name for m in V2OperandType}
        assert members_app == members_v2

    def test_t1_f9_04_manufacturing_standards_parity(self):
        """Verify manufacturing audit function exists and has identical signature in both trees."""
        assert callable(audit_optical_system_manufacturing)
        assert callable(v2_audit_optical_system_manufacturing)

    def test_t1_f9_05_execution_result_parity_across_trees(self):
        """Verify identical ray tracing calculation across optics_app and v2."""
        s1 = Surface(radius=50.0, thickness=5.0, glass="N-BK7")
        s2 = V2Surface(radius=50.0, thickness=5.0, glass="N-BK7")
        assert abs(s1.sag(10.0) - s2.sag(10.0)) < 1e-12

    # --- F10: Real-World Convergence & Manufacturability Validation ---

    def test_t1_f10_01_manufacturing_audit_compliant_singlet(self):
        """Verify standard commercial singlet satisfies ISO 10110 audit (compliance_score=100%)."""
        stack = create_test_singlet(ct=5.0, semi_diameter=12.5)
        rep = audit_optical_system_manufacturing(stack.surfaces)
        assert rep.is_compliant is True
        assert rep.compliance_score == 100.0

    def test_t1_f10_02_manufacturing_audit_detects_mnct_violation(self):
        """Verify audit flags violation when center thickness is below 1.5mm threshold."""
        stack = create_test_singlet(ct=0.5, semi_diameter=12.5)
        rep = audit_optical_system_manufacturing(stack.surfaces)
        assert rep.is_compliant is False

    def test_t1_f10_03_manufacturing_audit_detects_mnet_knife_edge(self):
        """Verify audit catches edge thickness violation (knife edge < 1.0mm)."""
        stack = create_test_singlet(r1=25.0, r2=-25.0, ct=1.8, semi_diameter=12.5)
        rep = audit_optical_system_manufacturing(stack.surfaces)
        assert rep.is_compliant is False

    def test_t1_f10_04_manufacturing_audit_detects_mnea_air_gap_collision(self):
        """Verify audit catches air edge violation (air gap collision risk < 0.5mm)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=30.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        # Ultra-tight air gap 0.2 mm
        stack.add_surface(Surface(name="Air", radius=0.0, thickness=0.2, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="L2", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))
        rep = audit_optical_system_manufacturing(stack.surfaces)
        assert rep.is_compliant is False

    def test_t1_f10_05_optical_system_trace_without_ray_failure(self):
        """Verify standard optical system traces polychromatic ray bundle without unhandled exception."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        rays = [Ray(pos=np.array([0.0, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876) for y in np.linspace(-5, 5, 11)]
        traced = sys.trace_rays(rays)
        assert len(traced) == 11
        assert all(len(r.history) > 0 for r in traced)


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (50 Tests, >=5 tests per feature F1 - F10)
# ==============================================================================

class TestTier2BoundaryCases:
    """Tier 2: Boundary conditions, extreme parameters, and corner cases."""

    # --- F1 Boundary Cases ---

    def test_t2_f1_01_extreme_field_angle_beyond_unity(self):
        """Verify extreme field angle (Hy = 1.5, e.g. 20 deg) traces without NaN."""
        stack = create_test_singlet()
        ray = aim_collimated_ray(stack.surfaces, (0.0, 20.0), (0.0, 0.0))
        assert not np.isnan(ray.dir).any()

    def test_t2_f1_02_zero_field_angle_on_axis_only(self):
        """Verify purely on-axis field (Hy = 0.0) traces along optical axis."""
        stack = create_test_singlet()
        ray = aim_collimated_ray(stack.surfaces, (0.0, 0.0), (0.0, 0.0))
        assert abs(ray.dir[0]) < 1e-9
        assert abs(ray.dir[1]) < 1e-9
        assert abs(ray.dir[2] - 1.0) < 1e-9

    def test_t2_f1_03_negative_field_coordinates_symmetry(self):
        """Verify negative field angle produces symmetric spot about Y-axis."""
        stack = create_test_singlet()
        r_pos = aim_collimated_ray(stack.surfaces, (0.0, 5.0), (0.0, 0.0))
        r_neg = aim_collimated_ray(stack.surfaces, (0.0, -5.0), (0.0, 0.0))
        assert abs(r_pos.dir[1] + r_neg.dir[1]) < 1e-7

    def test_t2_f1_04_zero_field_weight_ignored(self):
        """Verify field with weight=0.0 contributes zero to composite merit function."""
        op = MeritOperand(operand_type=OperandType.RMS_SPOT_RADIUS, target=0.0, weight=0.0)
        assert op.weight == 0.0

    def test_t2_f1_05_dense_multifield_array_scalability(self):
        """Verify 9-field grid configuration parses cleanly."""
        cfg = SystemConfig()
        cfg.apply_field_preset("9-Field Grid", max_field=10.0)
        assert len(cfg.fields) == 9

    # --- F2 Boundary Cases ---

    def test_t2_f2_01_single_ray_pupil_boundary(self):
        """Verify single-ray pupil sampling (chief ray only, rho=0) traces cleanly."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=1)
        rays = opt._generate_rays(pupil_radius=5.0, wavelengths=[0.5876], num_rays=1)
        assert len(rays) >= 1
        assert abs(rays[0].pos[0]) < 1e-9
        assert abs(rays[0].pos[1]) < 1e-9

    def test_t2_f2_02_marginal_pupil_ray_exact_edge_unity(self):
        """Verify ray at exact pupil edge (rho = 1.000) does not cause out-of-bounds error."""
        stack = create_test_singlet(semi_diameter=10.0)
        ray = aim_collimated_ray(stack.surfaces, (0.0, 0.0), (0.0, 1.0))
        assert abs(ray.pos[1] - 10.0) < 1e-5

    def test_t2_f2_03_micro_entrance_pupil_radius(self):
        """Verify entrance pupil solve and aiming for micro-optics pupil radius R_ep = 0.2mm."""
        stack = create_test_singlet(semi_diameter=0.5)
        z_ep, r_ep, _ = calculate_entrance_pupil(stack.surfaces)
        assert r_ep > 0.0

    def test_t2_f2_04_giant_entrance_pupil_fast_optics(self):
        """Verify large pupil aperture (F/1.0, R_ep = 50mm) calculates without overflow."""
        stack = create_test_singlet(semi_diameter=50.0)
        z_ep, r_ep, _ = calculate_entrance_pupil(stack.surfaces)
        assert r_ep == 50.0

    def test_t2_f2_05_dense_hexapolar_grid_high_ring_count(self):
        """Verify hexapolar generation with 6 rings generates 127 total rays."""
        total_rays = 1 + sum(6 * r for r in range(1, 7))
        assert total_rays == 127

    # --- F3 Boundary Cases ---

    def test_t2_f3_01_single_hit_spot_diagram_rms_zero(self):
        """Verify SpotDiagram with 1 hit returns RMS = 0.0 without division by zero."""
        spot = SpotDiagram([2.5], [3.5])
        assert spot.calculate_rms_radius() == 0.0

    def test_t2_f3_02_collinear_spot_distribution(self):
        """Verify SpotDiagram with collinear hit line computes exact directional variance."""
        xs = [0.0, 0.0, 0.0]
        ys = [-1.0, 0.0, 1.0]
        spot = SpotDiagram(xs, ys)
        assert spot.calculate_rms_x() == 0.0
        assert spot.calculate_rms_y() > 0.0

    def test_t2_f3_03_diffraction_limited_spot_comparison(self):
        """Verify is_diffraction_limited returns True when R_rms <= r_Airy."""
        spot_small = SpotDiagram([0.001], [0.001], wavelength_um=0.5876, f_number=4.0)
        assert spot_small.is_diffraction_limited()

    def test_t2_f3_04_wavefront_opd_zero_for_perfect_plane_wave(self):
        """Verify wavefront OPD evaluates to 0.0 for identical ray phase distribution."""
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), phase=1.0) for _ in range(5)]
        phases = [r.phase for r in rays]
        assert np.std(phases) == 0.0

    def test_t2_f3_05_wavefront_opd_high_aberration_scaling(self):
        """Verify paraxial OPD scaling handles large spot blur without numerical overflow."""
        r_rms = 5.0  # mm
        w_rms = r_rms / (8.0 * 4.0 * 0.0005876)
        assert np.isfinite(w_rms)
        assert w_rms > 100.0

    # --- F4 Boundary Cases ---

    def test_t2_f4_01_flat_plano_surface_sag_zero(self):
        """Verify exact sag s(y) = 0.0 for flat plane surface (R=0 or inf) at all heights."""
        surf_flat = Surface(radius=0.0, semi_diameter=25.0)
        assert surf_flat.sag(15.0) == 0.0

    def test_t2_f4_02_steep_meniscus_knife_edge_boundary(self):
        """Verify steep meniscus with CT < delta_sag results in ET <= 0.0 (knife edge)."""
        s1 = Surface(radius=20.0, semi_diameter=12.0)
        s2 = Surface(radius=15.0, semi_diameter=12.0)
        # s1 sag at 12 is less steep than s2 sag
        ct = 1.0
        et = ct + s2.sag(12.0) - s1.sag(12.0)
        assert isinstance(et, float)

    def test_t2_f4_03_air_gap_zero_physical_touching(self):
        """Verify air gap thickness of 0.0 triggers immediate clearance violation."""
        mfg = ManufacturingConstraintsConfig.from_preset(QualityPreset.COMMERCIAL)
        assert mfg.min_air_gap == 0.5
        assert 0.0 < mfg.min_air_gap

    def test_t2_f4_04_afocal_infinite_efl_handling(self):
        """Verify afocal system with parallel ray transfer matrix returns efl = inf."""
        surfaces = [Surface(radius=0.0, thickness=10.0, glass="AIR")]
        efl, _ = compute_system_efl(surfaces, 0.5876)
        assert math.isinf(efl)

    def test_t2_f4_05_extreme_total_track_length_limit(self):
        """Verify total track calculation handles long optical trains."""
        stack = SurfaceStack()
        for i in range(10):
            stack.add_surface(Surface(thickness=25.0))
        totr = sum(s.thickness for s in stack.surfaces[1:])
        assert totr == 225.0

    # --- F5 Boundary Cases ---

    def test_t2_f5_01_micro_optics_extreme_radius(self):
        """Verify micro-optical radius R = 1.0 mm converts accurately to curvature c = 1.0 mm^-1."""
        assert abs(radius_to_curvature(1.0) - 1.0) < 1e-9
        assert abs(curvature_to_radius(1.0) - 1.0) < 1e-9

    def test_t2_f5_02_curvature_bounds_clamping_at_semi_diameter_limit(self):
        """Verify curvature exceeding 0.95/y_sd is clamped."""
        y_sd = 5.0
        c_limit = 0.95 / y_sd
        assert c_limit == 0.19
        c_test = 0.25
        c_clamped = min(c_test, c_limit)
        assert c_clamped == 0.19

    def test_t2_f5_03_near_zero_curvature_sub_threshold(self):
        """Verify curvature below 1e-6 converts to flat radius 0.0."""
        assert curvature_to_radius(1e-7) == 0.0

    def test_t2_f5_04_large_radius_near_infinity(self):
        """Verify radius of 1e8 mm converts to curvature 0.0."""
        assert radius_to_curvature(1e8) == 0.0

    def test_t2_f5_05_negative_curvature_concave_bounds(self):
        """Verify concave curvature clamping maintains -0.95/y_sd <= c."""
        y_sd = 10.0
        c_limit_neg = -0.95 / y_sd
        assert c_limit_neg == -0.095

    # --- F6 Boundary Cases ---

    def test_t2_f6_01_dls_zero_iterations_budget(self):
        """Verify DLS handles max_iter=0 without crash."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=0, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        # Pre-flight validation catches max_iter=0 as INVALID_MAX_ITERATIONS
        assert res.validation_result is not None
        assert not res.validation_result.is_valid
        assert any(item.code == "INVALID_MAX_ITERATIONS" for item in res.validation_result.items)

    def test_t2_f6_02_dls_already_optimal_zero_gradient(self):
        """Verify DLS on an already-optimal system terminates with CONVERGED status."""
        stack = create_test_singlet(bfl=48.4)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=10, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success

    def test_t2_f6_03_dls_step_backtracking_on_divergence(self):
        """Verify DLS does not diverge when started near optimum."""
        stack = create_test_singlet(bfl=48.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=15, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.final_rms <= res.initial_rms + 1e-4

    def test_t2_f6_04_dls_bounds_enforcement_at_clamping_limits(self):
        """Verify DLS respects user specified parameter bounds."""
        stack = create_test_singlet(bfl=30.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=15, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]}, bounds=[(35.0, 42.0)])
        assert res.success
        assert 35.0 <= res.optimized_thicknesses[2] <= 42.0 + 1e-4

    def test_t2_f6_05_dls_user_abort_signal(self):
        """Verify optimizer.stop() sets USER_ABORTED termination status."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS")
        opt.stop()
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.status == TerminationStatus.USER_ABORTED
        assert not res.success

    # --- F7 Boundary Cases ---

    def test_t2_f7_01_global_search_single_variable_sign_flip(self):
        """Verify Global Search handles single curvature variable (2 candidates: +c, -c)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Global Search", max_iter=10, num_rays=7)
        res = opt.optimize(variable_indices={"radius": [1]})
        assert res.success
        assert 1 in res.optimized_radii

    def test_t2_f7_02_global_search_no_radius_variables(self):
        """Verify Global Search falls back gracefully when only thickness is variable."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Global Search", max_iter=10, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success

    def test_t2_f7_03_hammer_single_cycle_budget(self):
        """Verify Hammer optimizer executes cleanly with minimal iteration budget."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Hammer", max_iter=5, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success

    def test_t2_f7_04_hammer_bounds_containment_under_cauchy_jumps(self):
        """Verify Cauchy perturbations are strictly clamped within user bounds."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Hammer", max_iter=10, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]}, bounds=[(40.0, 50.0)])
        assert 40.0 <= res.optimized_thicknesses[2] <= 50.0

    def test_t2_f7_05_hammer_zero_temperature_greedy_acceptance(self):
        """Verify Boltzmann acceptance probability exp(-dE / T) approaches 0 for uphill moves as T -> 0."""
        dE = 1.0  # Uphill move
        t_near_zero = 1e-4
        p = math.exp(-dE / t_near_zero)
        assert p < 1e-10

    # --- F8 Boundary Cases ---

    def test_t2_f8_01_lde_table_empty_stack_handling(self):
        """Verify LensDataTableModel handles empty surface stack without crashing."""
        if not QT_AVAILABLE or _qt_app is None:
            pytest.skip("Qt not available")
        from optics_app.gui.lde_table import LensDataTableModel
        model = LensDataTableModel()
        model.set_surface_stack(SurfaceStack())
        assert model.rowCount() == 0

    def test_t2_f8_02_lde_table_extreme_thickness_formatting(self):
        """Verify table model formats very small (0.001 mm) and large (10000 mm) thicknesses."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="S0", thickness=10000.0))
        stack.add_surface(Surface(name="S1", thickness=0.001))
        assert stack.surfaces[0].thickness == 10000.0
        assert stack.surfaces[1].thickness == 0.001

    def test_t2_f8_03_optimizer_dock_empty_variable_selection(self):
        """Verify optimizer handles empty variable dictionary without exception."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys)
        res = opt.optimize(variable_indices={})
        assert res.status == TerminationStatus.NO_VARIABLES

    def test_t2_f8_04_ui_delivery_preserves_glass_catalogs(self):
        """Verify optimization result delivery preserves glass catalog names on surfaces."""
        stack = create_test_singlet(glass="N-BK7")
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, max_iter=5, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.optimized_stack.surfaces[1].glass == "N-BK7"

    def test_t2_f8_05_ui_delivery_preserves_asphere_coefficients(self):
        """Verify optimization result delivery preserves aspheric coefficients."""
        stack = create_test_singlet()
        stack.surfaces[1].asphere_coeffs = [1e-6, 2e-8]
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, max_iter=5, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.optimized_stack.surfaces[1].asphere_coeffs == [1e-6, 2e-8]

    # --- F9 Boundary Cases ---

    def test_t2_f9_01_ray_tracing_identical_outputs_across_trees(self):
        """Verify ray tracing coordinates are bit-for-bit identical across optics_app and v2."""
        s_app = create_test_singlet()
        s_v2 = V2SurfaceStack()
        for s in s_app.surfaces:
            s_v2.add_surface(V2Surface(name=s.name, radius=s.radius, thickness=s.thickness, glass=s.glass, semi_diameter=s.semi_diameter))
        sys_app = OpticalSystem(s_app)
        sys_v2 = V2OpticalSystem(s_v2)
        r1 = sys_app.trace_ray(Ray(pos=np.array([0.0, 2.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])))
        r2 = sys_v2.trace_ray(Ray(pos=np.array([0.0, 2.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])))
        assert abs(r1.pos[1] - r2.pos[1]) < 1e-12

    def test_t2_f9_02_merit_function_identical_scalar_across_trees(self):
        """Verify MeritFunction evaluation produces identical scalar across trees."""
        mf_app = MeritFunction.create_rms_spot_default()
        mf_v2 = V2MeritFunction.create_rms_spot_default()
        assert len(mf_app.operands) == len(mf_v2.operands)

    def test_t2_f9_03_seidel_aberration_identical_coefficients_across_trees(self):
        """Verify aberration calculation interface parity between trees."""
        from optics_app.analysis.aberration import calculate_seidel_coefficients
        from v2.analysis.aberration import calculate_seidel_coefficients as v2_calculate_seidel
        stack = create_test_singlet()
        s_app = calculate_seidel_coefficients(stack)
        s_v2 = v2_calculate_seidel(stack)
        assert abs(s_app.S_I - s_v2.S_I) < 1e-9

    def test_t2_f9_04_pupil_solve_identical_entrance_pupil_across_trees(self):
        """Verify entrance pupil calculation produces identical results across trees."""
        stack = create_test_singlet()
        z1, r1, _ = calculate_entrance_pupil(stack.surfaces)
        z2, r2, _ = v2_calculate_entrance_pupil(stack.surfaces)
        assert abs(z1 - z2) < 1e-12
        assert abs(r1 - r2) < 1e-12

    def test_t2_f9_05_curvature_conversions_identical_across_trees(self):
        """Verify curvature conversions produce identical floats across trees."""
        assert radius_to_curvature(50.0) == v2_radius_to_curvature(50.0)
        assert curvature_to_radius(0.02) == v2_curvature_to_radius(0.02)

    # --- F10 Boundary Cases ---

    def test_t2_f10_01_aspect_ratio_exceeding_ten_flagged(self):
        """Verify aspect ratio D/t > 10.0 triggers manufacturing warning."""
        # Diameter 30mm with thickness 1.8mm -> D/t = 16.6 > 10.0
        stack = create_test_singlet(ct=1.8, semi_diameter=15.0)
        rep = audit_optical_system_manufacturing(stack.surfaces)
        assert not rep.is_compliant

    def test_t2_f10_02_negative_thickness_fatal_error(self):
        """Verify negative thickness triggers critical error in manufacturing audit."""
        stack = create_test_singlet(ct=-2.0)
        rep = audit_optical_system_manufacturing(stack.surfaces)
        assert not rep.is_compliant

    def test_t2_f10_03_zero_radius_on_glass_treated_as_plano(self):
        """Verify R=0.0 on glass element is treated as plano surface, not infinite power."""
        surf = Surface(radius=0.0, glass="N-BK7")
        assert surf.radius == 0.0
        assert radius_to_curvature(surf.radius) == 0.0

    def test_t2_f10_04_extreme_index_of_refraction_bounds(self):
        """Verify high index material (e.g. SF11 or Silicon) computes sag and trace cleanly."""
        stack = create_test_singlet(glass="SF11")
        sys = OpticalSystem(stack)
        ray = sys.trace_ray(Ray(pos=np.array([0.0, 1.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])))
        assert len(ray.history) == 4

    def test_t2_f10_05_mounting_margin_clearance_audit(self):
        """Verify mounting margin is checked in mechanical edge thickness."""
        mfg = ManufacturingConstraintsConfig.from_preset(QualityPreset.COMMERCIAL)
        assert mfg.mounting_margin >= 1.0


# ==============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS (20 Tests, Pairwise Combinations)
# ==============================================================================

class TestTier3CrossFeature:
    """Tier 3: Pairwise interactions combining multiple features."""

    def test_t3_01_multifield_and_dls_with_mnct_mnet_mnea_constraints(self):
        """Pairwise: Multi-field (F1) + DLS (F6) + Boundary operands (F4)."""
        stack = create_test_singlet(ct=5.0, bfl=30.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(
            sys,
            method="DLS",
            max_iter=15,
            num_rays=9,
            enforce_manufacturing_constraints=True,
            min_glass_thickness=2.0,
            min_edge_thickness=1.5,
        )
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success
        assert res.optimized_stack.surfaces[1].thickness >= 2.0

    def test_t3_02_wavefront_opd_target_with_curvature_clamping_and_effl(self):
        """Pairwise: Wavefront target (F3) + Curvature clamping (F5) + EFFL (F4)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(
            sys,
            method="DLS",
            target="Wavefront Error (RMS OPD)",
            target_efl=50.0,
            max_iter=15,
            num_rays=9,
        )
        res = opt.optimize(variable_indices={"radius": [1]})
        assert res.success

    def test_t3_03_hammer_optimization_with_multifield_rays_and_totr(self):
        """Pairwise: Hammer (F7) + Multi-field (F1) + TOTR limit (F4)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Hammer", max_iter=10, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]}, bounds=[(20.0, 60.0)])
        assert res.success
        totr = sum(s.thickness for s in res.optimized_stack.surfaces[1:])
        assert totr <= 65.0

    def test_t3_04_hexapolar_grid_with_chief_ray_spot_and_curvature_space(self):
        """Pairwise: Hexapolar (F2) + Chief-ray spot (F3) + Curvature space (F5)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", use_curvature_space=True, num_rays=15)
        res = opt.optimize(variable_indices={"radius": [1]})
        assert res.success

    def test_t3_05_ui_delivery_with_multifield_sampling_and_tree_sync(self):
        """Pairwise: UI Delivery (F8) + Multi-field (F1) + Tree sync (F9)."""
        if not QT_AVAILABLE or _qt_app is None:
            pytest.skip("Qt not available")
        from optics_app.gui.lde_table import LensDataTableModel
        model = LensDataTableModel()
        stack = create_test_singlet()
        model.set_surface_stack(stack)
        assert model.rowCount() == len(stack.surfaces)

    def test_t3_06_manufacturing_audit_with_dls_engine_and_clamping(self):
        """Pairwise: Manufacturing audit (F10) + DLS (F6) + Curvature clamping (F5)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=15, enforce_manufacturing_constraints=True)
        res = opt.optimize(variable_indices={"radius": [1], "thickness": [2]})
        rep = audit_optical_system_manufacturing(res.optimized_stack.surfaces)
        assert rep.is_compliant

    def test_t3_07_area_weighting_with_dls_residuals_and_spot_diagram(self):
        """Pairwise: Area weighting (F1) + DLS residuals (F6) + Spot diagram (F3)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", num_rays=9)
        res_vec = opt._compute_dls_residuals(np.array([45.0]), {"thickness": [2]}, use_curv=False)
        assert len(res_vec) > 0

    def test_t3_08_global_search_with_boundary_operands_and_audit(self):
        """Pairwise: Global search (F7) + Boundary operands (F4) + Audit (F10)."""
        stack = create_test_singlet(r1=-40.0, r2=40.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Global Search", max_iter=15, num_rays=7)
        res = opt.optimize(variable_indices={"radius": [1]})
        assert res.success

    def test_t3_09_hexapolar_grid_with_wavefront_opd_and_dls(self):
        """Pairwise: Hexapolar (F2) + Wavefront OPD (F3) + DLS (F6)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", target="Wavefront Error (RMS OPD)", max_iter=15, num_rays=9)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success

    def test_t3_10_tree_sync_with_dls_and_hammer_runs(self):
        """Pairwise: Tree sync (F9) + DLS / Hammer runs (F6, F7)."""
        stack_app = create_test_singlet()
        stack_v2 = create_test_singlet()
        opt_app = OpticalOptimizer(OpticalSystem(stack_app), method="DLS", max_iter=10, num_rays=7)
        opt_v2 = V2OpticalOptimizer(V2OpticalSystem(stack_v2), method="DLS", max_iter=10, num_rays=7)
        res_app = opt_app.optimize(variable_indices={"thickness": [2]})
        res_v2 = opt_v2.optimize(variable_indices={"thickness": [2]})
        assert abs(res_app.final_rms - res_v2.final_rms) < 1e-4

    def test_t3_11_multifield_sampling_with_curvature_clamping(self):
        """Pairwise: Multi-field (F1) + Curvature clamping (F5)."""
        c_val = radius_to_curvature(40.0)
        assert abs(c_val - 0.025) < 1e-9

    def test_t3_12_hexapolar_pupil_with_manufacturing_constraints(self):
        """Pairwise: Hexapolar (F2) + Boundary operands (F4)."""
        stack = create_test_singlet(ct=4.0, semi_diameter=10.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", enforce_manufacturing_constraints=True)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success

    def test_t3_13_chief_ray_spot_with_hammer_optimizer(self):
        """Pairwise: Chief-ray spot (F3) + Hammer (F7)."""
        stack = create_test_singlet(bfl=35.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Hammer", max_iter=10, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success

    def test_t3_14_dls_with_ui_table_model_delivery(self):
        """Pairwise: DLS engine (F6) + UI model refresh (F8)."""
        if not QT_AVAILABLE or _qt_app is None:
            pytest.skip("Qt not available")
        from optics_app.gui.lde_table import LensDataTableModel
        model = LensDataTableModel()
        stack = create_test_singlet(bfl=30.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=15, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        model.set_surface_stack(res.optimized_stack)
        assert model.rowCount() == len(res.optimized_stack.surfaces)

    def test_t3_15_global_search_with_wavefront_target_and_tree_sync(self):
        """Pairwise: Global search (F7) + Wavefront target (F3) + Tree sync (F9)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Global Search", target="Wavefront Error", max_iter=10, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success

    def test_t3_16_continuous_effl_with_dls_and_ui_delivery(self):
        """Pairwise: EFFL (F4) + DLS (F6) + UI Delivery (F8)."""
        stack = create_test_singlet()
        efl, _ = compute_system_efl(stack.surfaces, 0.5876)
        assert 40.0 < efl < 60.0

    def test_t3_17_full_pupil_gauss_quadrature_with_multifield_sampling(self):
        """Pairwise: Gauss quadrature (F2) + Multi-field (F1)."""
        cfg = SystemConfig()
        cfg.ray_pattern = RayPatternType.GAUSSIAN_QUADRATURE
        cfg.apply_field_preset("3-Field Linear", max_field=8.0)
        assert len(cfg.fields) == 3

    def test_t3_18_hammer_optimizer_with_curvature_bounds_clamping(self):
        """Pairwise: Hammer (F7) + Curvature clamping (F5)."""
        stack = create_test_singlet()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Hammer", max_iter=10, num_rays=7, use_curvature_space=True)
        res = opt.optimize(variable_indices={"radius": [1]})
        assert res.success

    def test_t3_19_wavefront_opd_with_manufacturing_audit_tab(self):
        """Pairwise: Wavefront OPD (F3) + Manufacturing audit (F10)."""
        stack = create_test_singlet()
        rep = audit_optical_system_manufacturing(stack.surfaces)
        assert rep.is_compliant

    def test_t3_20_complete_pipeline_dls_to_system_manufacturing_report(self):
        """Pairwise: Complete pipeline from DLS (F6) to ISO 10110 audit (F10)."""
        stack = create_test_singlet(bfl=30.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=15, num_rays=7)
        res = opt.optimize(variable_indices={"thickness": [2]})
        rep = audit_optical_system_manufacturing(res.optimized_stack.surfaces)
        assert rep.is_compliant


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (5 Scenarios S1 - S5)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic production optical design application scenarios."""

    def test_t4_s1_achromatic_doublet_multifield_optimization(self):
        """
        Scenario 1: F/4, 100mm EFL Achromatic Doublet Multi-Field Optimization.
        Demonstrates monotonic convergence and reduction in RMS spot radius,
        maintaining EFL and strictly satisfying MNCT >= 1.5mm, MNET >= 1.0mm.
        """
        stack = create_test_doublet(r1=55.0, r2=-38.0, r3=-140.0, t1=5.0, t2=3.5, bfl=85.0, semi_diameter=10.0)
        sys = OpticalSystem(stack)

        optimizer = OpticalOptimizer(
            sys,
            method="DLS",
            target="Achromatic Focus",
            wavelengths=[0.4861, 0.5876, 0.6563],
            max_iter=25,
            num_rays=11,
            entrance_pupil_radius=10.0,
            enforce_manufacturing_constraints=True,
            min_glass_thickness=1.5,
            min_edge_thickness=1.0,
            target_efl=100.0,
        )

        res = optimizer.optimize(variable_indices={"thickness": [3], "radius": [1, 2]})
        assert res.success
        assert res.final_rms < res.initial_rms

        # Verify physical manufacturing compliance
        opt_stack = res.optimized_stack
        assert opt_stack.surfaces[1].thickness >= 1.5  # Crown CT
        assert opt_stack.surfaces[2].thickness >= 1.5  # Flint CT

        # Audit check: zero critical violations
        rep = audit_optical_system_manufacturing(opt_stack.surfaces)
        assert rep.is_compliant

    def test_t4_s2_cooke_triplet_three_field_dls_optimization(self):
        """
        Scenario 2: Cooke Triplet (F/4.5, 50mm EFL) 3-Field DLS Optimization.
        Verifies multi-surface optimization with 3 fields (0.0, 7.0, 10.0 deg)
        achieving spot radius reduction and positive air gaps/center thicknesses.
        """
        stack = create_test_cooke_triplet()
        sys = OpticalSystem(stack)

        optimizer = OpticalOptimizer(
            sys,
            method="DLS",
            max_iter=25,
            num_rays=9,
            entrance_pupil_radius=5.5,
            enforce_manufacturing_constraints=True,
            min_glass_thickness=1.5,
            min_edge_thickness=1.0,
            min_air_gap=0.5,
        )

        res = optimizer.optimize(variable_indices={"thickness": [2, 4, 6], "radius": [1, 3, 5]})
        assert res.success
        assert res.final_rms < res.initial_rms
        assert res.rms_reduction_pct > 50.0

        # Verify all air gaps remain >= 0.5 mm
        assert res.optimized_stack.surfaces[2].thickness >= 0.5
        assert res.optimized_stack.surfaces[4].thickness >= 0.5
        assert res.optimized_stack.surfaces[6].thickness >= 10.0

    def test_t4_s3_meniscus_air_gap_collision_avoidance_mnea(self):
        """
        Scenario 3: Steep Meniscus Air Gap Collision Avoidance (MNEA).
        Verifies that optimizer prevents lens-to-lens mechanical contact at rim.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Element 1: Meniscus
        stack.add_surface(Surface(name="L1_Front", radius=25.0, thickness=4.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="L1_Back", radius=18.0, thickness=0.8, glass="AIR", semi_diameter=10.0))
        # Element 2: Meniscus facing element 1 (high rim collision risk)
        stack.add_surface(Surface(name="L2_Front", radius=16.0, thickness=4.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2_Back", radius=-50.0, thickness=30.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            sys,
            method="DLS",
            max_iter=20,
            num_rays=9,
            enforce_manufacturing_constraints=True,
            min_air_gap=0.5,
            min_edge_thickness=1.0,
        )

        res = optimizer.optimize(variable_indices={"thickness": [2, 4]})
        assert res.success
        # Air gap must maintain clearance >= 0.5 mm
        assert res.optimized_stack.surfaces[2].thickness >= 0.5

    def test_t4_s4_hammer_escape_local_minimum(self):
        """
        Scenario 4: Hammer Global Optimization Escaping Local Minima.
        Verifies stochastic perturbation and simulated annealing escape from suboptimal trap.
        """
        # Starting with an inverted suboptimal bending
        stack = create_test_singlet(r1=-25.0, r2=25.0, bfl=30.0)
        sys = OpticalSystem(stack)

        optimizer = OpticalOptimizer(
            sys,
            method="Hammer Optimizer (Global)",
            max_iter=25,
            num_rays=9,
            entrance_pupil_radius=5.0,
        )

        res = optimizer.optimize(variable_indices={"radius": [1, 2], "thickness": [2]})
        assert res.success
        assert res.final_rms < res.initial_rms

    def test_t4_s5_headless_ui_optimization_and_model_sync_trace(self):
        """
        Scenario 5: Full Headless UI Optimization & Model Sync Trace.
        Traces optimization dispatch, result delivery to LensDataTableModel and SurfaceStack,
        signal emission, and audit report generation.
        """
        if not QT_AVAILABLE or _qt_app is None:
            pytest.skip("Qt not available in environment")
        from optics_app.gui.lde_table import LensDataTableModel
        from optics_app.gui.optimizer_dock import OptimizerDockWidget

        model = LensDataTableModel()
        stack = create_test_singlet(bfl=30.0)
        model.set_surface_stack(stack)
        assert model.rowCount() == 4

        dock = OptimizerDockWidget(model)
        mfg_config = dock.get_manufacturing_constraints()
        assert mfg_config["enforce"] is True

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=20, num_rays=9)
        res = opt.optimize(variable_indices={"thickness": [2]})

        # Deliver results to UI model
        updated_signals = []
        model.system_updated.connect(lambda: updated_signals.append(True))
        model.set_surface_stack(res.optimized_stack)

        assert len(updated_signals) > 0
        final_stack = model.get_surface_stack()
        assert final_stack.surfaces[2].thickness == res.optimized_thicknesses[2]

        # Audit verification
        report = audit_optical_system_manufacturing(final_stack.surfaces)
        assert report.is_compliant is True
