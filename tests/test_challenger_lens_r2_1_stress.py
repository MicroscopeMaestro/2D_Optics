"""
Empirical Adversarial Stress Test Suite for Physical Lens Design (Round 2).

Author: challenger_lens_r2_1 (Challenger 1)
Scope:
1. Dynamic Parameter Sweeps & Discontinuity Hunt across R1 ~ -50mm
2. Boundary & Extreme Geometries (thin lens limit, thick blocks, plano, concentric, degenerate inputs)
3. Multi-Component Galileo Afocal Alignment & Paraxial Transfer Matrices
4. Aspheric Vertex Equivalence & Tolerancing Immutability
"""

import math
import numpy as np
import pytest

from optics_app.engine.matrix_optics import (
    TransferMatrix,
    CardinalPoints,
    refraction_matrix,
    space_matrix,
    thin_lens_matrix,
    thick_lens_matrix,
)
from optics_app.engine.thick_lens import (
    OpticalElement,
    ThickLens,
    ThinLens,
)
from optics_app.engine.aspheric import (
    SurfaceModel,
    AsphericSurfaceModel,
    AsphericLens,
)
from optics_app.engine.multi_component import (
    MultiComponentSystem,
    create_galileo_system,
)
from optics_app.engine.lens_merit import (
    PhysicalMeritFunction,
    TargetEFLOperand,
)
from optics_app.engine.tolerancing import (
    TolerancingEngine,
    PerturbationSpec,
)


# =============================================================================
# 1. Dynamic Parameter Sweeps & Discontinuity Hunt Across R1 ~ -50mm
# =============================================================================

class TestDiscontinuityHuntAcrossMinus50:
    """
    Adversarial dynamic sweeps verifying that no branch conditions, lookup tables,
    or artificial cheat shortcuts remain around R1 = -50mm.
    """

    def test_r1_fine_sweep_monotonicity_and_step_jumps(self):
        """
        Sweep R1 across [-50.05, -49.95] in steps of 0.001 mm.
        Verify strict monotonicity, zero step jumps, and exact analytical derivative match.
        """
        r2 = 50.0
        t = 5.0
        n = 1.5
        r1_vals = np.linspace(-50.05, -49.95, 101)
        step = r1_vals[1] - r1_vals[0]

        efls = np.array([ThickLens(r, r2, t, n).cardinal_points().efl for r in r1_vals])
        diffs = np.diff(efls)

        # 1. Strict monotonicity: EFL must increase monotonically
        assert np.all(diffs > 0), f"Non-monotonic step detected: min diff = {diffs.min()}"

        # 2. Absence of step discontinuities:
        # For Delta R1 = 0.001 mm, Delta EFL should be ~ 0.0005 mm, never > 0.001 mm
        max_jump = np.max(np.abs(diffs))
        assert max_jump < 0.001, f"Step jump detected in fine sweep: max jump = {max_jump}"

        # 3. First derivative verification against analytical Gullstrand derivative:
        # Phi = (n-1)/R1 + (1-n)/R2 - (t/n)(n-1)(1-n)/(R1*R2)
        # d(EFL)/dR1 = - (1/Phi^2) * dPhi/dR1
        # dPhi/dR1 = -(n-1)/R1^2 + (t/n)(n-1)(1-n)/(R1^2 * R2)
        phi_1 = (n - 1.0) / (-50.0)
        phi_2 = (1.0 - n) / r2
        phi_total = phi_1 + phi_2 - (t / n) * phi_1 * phi_2
        dphi_dr1 = -(n - 1.0) / ((-50.0) ** 2) + (t / n) * (n - 1.0) * (1.0 - n) / (((-50.0) ** 2) * r2)
        analytical_deriv = -1.0 / (phi_total ** 2) * dphi_dr1  # ~ +0.4998657

        numerical_derivs = diffs / step
        max_rel_error = np.max(np.abs(numerical_derivs - analytical_deriv) / analytical_deriv)
        assert max_rel_error < 0.001, f"Derivative mismatch: max rel error = {max_rel_error}"

    def test_r1_ultrafine_sweep_continuous_differentiability(self):
        """
        Ultra-fine sweep across [-50.001, -49.999] in steps of 1e-6 mm (2001 points).
        Guarantees micro-scale smooth differentiability across -50.000000 mm.
        """
        r2 = 50.0
        t = 5.0
        n = 1.5
        r1_vals = np.linspace(-50.001, -49.999, 2001)
        step = r1_vals[1] - r1_vals[0]

        efls = np.array([ThickLens(r, r2, t, n).cardinal_points().efl for r in r1_vals])
        diffs = np.diff(efls)

        assert np.all(diffs > 0), "Ultra-fine sweep failed monotonicity"
        max_jump = np.max(np.abs(diffs))
        assert max_jump < 1e-5, f"Discontinuity detected at micro-scale: {max_jump}"

        # Derivative consistency across the entire micro-interval
        derivs = diffs / step
        mean_deriv = np.mean(derivs)
        std_deriv = np.std(derivs)
        assert std_deriv / mean_deriv < 1e-4, f"Derivative variance too high: {std_deriv/mean_deriv}"

    def test_r1_wide_sweep_no_residual_branch_artifacts(self):
        """
        Broad sweep across [-70.0, -30.0] in steps of 0.1 mm (401 points).
        Ensures no hidden boundaries or branch triggers across the wider negative domain.
        """
        r2 = 50.0
        t = 5.0
        n = 1.5
        r1_vals = np.linspace(-70.0, -30.0, 401)
        efls = np.array([ThickLens(r, r2, t, n).cardinal_points().efl for r in r1_vals])
        diffs = np.diff(efls)

        assert np.all(diffs > 0), "Wide sweep failed monotonicity"
        # Second differences (d2 EFL / dR1^2) must remain bounded and smooth
        second_diffs = np.diff(diffs)
        max_second_diff = np.max(np.abs(second_diffs))
        assert max_second_diff < 0.005, f"Second derivative spike in wide sweep: {max_second_diff}"

    def test_r2_sweep_across_plus_50(self):
        """
        Sweep R2 across [49.95, 50.05] with R1 = -50.0 mm.
        Verify rear radius variation is equally continuous and monotonic.
        """
        r1 = -50.0
        t = 5.0
        n = 1.5
        r2_vals = np.linspace(49.95, 50.05, 101)
        efls = np.array([ThickLens(r1, r, t, n).cardinal_points().efl for r in r2_vals])
        diffs = np.diff(efls)

        # As R2 increases (becomes less curved), negative power decreases (closer to 0),
        # so EFL = 1/Phi becomes more negative (decreases).
        assert np.all(diffs < 0), "R2 sweep failed monotonicity"
        max_jump = np.max(np.abs(diffs))
        assert max_jump < 0.001, f"Step jump in R2 sweep: {max_jump}"

    def test_biconcave_exact_analytical_gullstrand(self):
        """
        Verify exact cardinal points for R1 = -50.0, R2 = 50.0, t = 5.0, n = 1.5.
        """
        lens = ThickLens(r1=-50.0, r2=50.0, thickness=5.0, glass_or_index=1.5)
        cp = lens.cardinal_points()
        mat = lens.transfer_matrix()

        # Analytical derivation:
        # Phi1 = -0.01 mm^-1, Phi2 = -0.01 mm^-1
        # Phi = -0.02 - (5/1.5)*0.0001 = -0.0203333333 mm^-1
        # EFL = 1 / Phi = -49.18032786885246 mm
        # P1 = (D - 1)/C. Here D = 1 - (5/1.5)*0.01 = 1 - 0.033333 = 0.9666667
        # P1 = (0.9666667 - 1) / (-0.0203333) = +1.63934426 mm
        # P2 = (1 - A)/C. Here A = 0.9666667 => P2 = -1.63934426 mm
        expected_efl = -49.18032786885246
        expected_p1 = 1.63934426229508
        expected_p2 = -1.63934426229508
        expected_bfl = expected_p2 + expected_efl  # -50.81967213

        assert abs(cp.efl - expected_efl) < 1e-6
        assert abs(cp.p1 - expected_p1) < 1e-6
        assert abs(cp.p2 - expected_p2) < 1e-6
        assert abs(cp.bfl - expected_bfl) < 1e-6
        assert abs(mat.determinant() - 1.0) < 1e-12


# =============================================================================
# 2. Boundary & Extreme Geometries
# =============================================================================

class TestBoundaryAndExtremeGeometries:
    """
    Stress-tests edge cases: asymptotic limits (t -> 0, t -> inf), flat surfaces,
    concentric zero/weak power geometries, and invalid input enforcement.
    """

    @pytest.mark.parametrize("t", [1.0, 0.1, 0.01, 1e-4, 1e-6, 1e-8])
    def test_thin_lens_limit_t_to_zero(self, t):
        """
        Verify that as center thickness t -> 0, the thick lens cardinal points
        converge smoothly and linearly to the thin lens paraxial formula:
            1/f = (n - 1) * (1/R1 - 1/R2)
        """
        r1, r2, n = 50.0, -50.0, 1.5
        thin_f = 1.0 / ((n - 1.0) * (1.0 / r1 - 1.0 / r2))  # exactly 50.0 mm
        lens = ThickLens(r1, r2, t, n)
        cp = lens.cardinal_points()

        err = abs(cp.efl - thin_f)
        # Error must scale with O(t): err / t should be approximately constant ~ 0.1667
        ratio = err / t
        assert ratio < 0.20, f"Error did not scale linearly with t: err={err}, t={t}"

    @pytest.mark.parametrize("t", [50.0, 100.0, 500.0, 1000.0, 5000.0])
    def test_extreme_thickness_blocks(self, t):
        """
        Verify transfer matrix determinant invariance det(M) == 1.0 and cardinal point
        identities even for massive thick optical blocks up to 5 meters thick.
        """
        r1, r2, n = 80.0, -120.0, 1.5
        lens = ThickLens(r1, r2, t, n)
        mat = lens.transfer_matrix()
        cp = lens.cardinal_points()

        # Determinant invariant
        assert abs(mat.determinant() - 1.0) < 1e-11

        # Cardinal point physical relations:
        # BFL = P2 + EFL
        # FFL = -P1 + EFL (with standard directed convention FFL = -D/C = P1 - EFL directed)
        assert abs(cp.bfl - (cp.p2 + cp.efl)) < 1e-8

    def test_plano_convex_and_plano_concave(self):
        """
        Verify plano lenses with infinite radius on either face.
        """
        # Plano-convex: front flat, rear curved
        l1 = ThickLens(float("inf"), -50.0, 5.0, 1.5)
        cp1 = l1.cardinal_points()
        # Analytical EFL = -R2 / (n - 1) = -(-50) / 0.5 = 100.0 mm
        assert abs(cp1.efl - 100.0) < 1e-9
        assert abs(cp1.p1 - (5.0 / 1.5)) < 1e-9  # P1 = t/n from V1
        assert abs(cp1.p2 - 0.0) < 1e-9          # P2 at rear vertex V2

        # Plano-convex: front curved, rear flat
        l2 = ThickLens(50.0, float("inf"), 5.0, 1.5)
        cp2 = l2.cardinal_points()
        assert abs(cp2.efl - 100.0) < 1e-9
        assert abs(cp2.p1 - 0.0) < 1e-9          # P1 at front vertex V1
        assert abs(cp2.p2 - (-5.0 / 1.5)) < 1e-9 # P2 = -t/n from V2

        # Plano-concave: front flat, rear concave (R2 > 0)
        l3 = ThickLens(float("inf"), 50.0, 5.0, 1.5)
        cp3 = l3.cardinal_points()
        assert abs(cp3.efl - (-100.0)) < 1e-9

    def test_flat_glass_plate_afocal(self):
        """
        Verify that a flat parallel glass plate (R1=inf, R2=inf) is correctly detected
        as an afocal system with zero optical power and paraxial displacement B = t/n.
        """
        t = 12.0
        n = 1.5
        plate = ThickLens(float("inf"), float("inf"), t, n)
        mat = plate.transfer_matrix()
        cp = plate.cardinal_points()

        assert cp.is_afocal is True
        assert abs(mat.C) < 1e-15
        assert abs(mat.A - 1.0) < 1e-12
        assert abs(mat.D - 1.0) < 1e-12
        assert abs(mat.B - (t / n)) < 1e-12  # Apparent thickness t/n
        assert abs(mat.determinant() - 1.0) < 1e-12

    def test_concentric_meniscus_optical_power(self):
        """
        Verify that a concentric meniscus lens (R1 = R2 = R) has positive optical power:
            Phi = (t / n) * ((n - 1) / R)^2 > 0
        """
        r = 60.0
        t = 6.0
        n = 1.52
        lens = ThickLens(r, r, t, n)
        cp = lens.cardinal_points()

        expected_power = (t / n) * ((n - 1.0) / r) ** 2
        expected_efl = 1.0 / expected_power

        assert abs(cp.power - expected_power) < 1e-12
        assert abs(cp.efl - expected_efl) < 1e-8
        assert cp.efl > 0  # Must be strictly positive

    def test_zero_power_thick_lens_transition(self):
        """
        Verify the exact boundary where a thick meniscus lens has zero net optical power:
            Phi_total = 0 => R2 = R1 - ((n - 1) / n) * t
        """
        r1 = 50.0
        n = 1.5
        t = 9.0
        # For zero power: R2 = 50 - (0.5 / 1.5) * 9 = 50 - 3 = 47.0 mm
        r2_zero = r1 - (n - 1.0) / n * t
        lens_zero = ThickLens(r1, r2_zero, t, n)
        mat = lens_zero.transfer_matrix()
        cp = lens_zero.cardinal_points()

        assert abs(mat.C) < 1e-15
        assert cp.is_afocal is True
        assert math.isinf(cp.efl)

    @pytest.mark.parametrize("bad_t", [0.0, -1.0, -1e-8])
    def test_non_positive_thickness_raises_value_error(self, bad_t):
        """Ensure non-positive thickness is rejected immediately."""
        with pytest.raises(ValueError, match="Center thickness must be strictly positive"):
            ThickLens(50.0, -50.0, bad_t, 1.5)

    @pytest.mark.parametrize("bad_n", [0.0, -1.5, -0.01])
    def test_non_positive_refractive_index_raises_value_error(self, bad_n):
        """Ensure non-positive index is rejected immediately."""
        with pytest.raises(ValueError, match="Refractive index must be strictly positive"):
            ThickLens(50.0, -50.0, 5.0, bad_n)

    @pytest.mark.parametrize("bad_diam", [0.0, -10.0])
    def test_non_positive_diameter_raises_value_error(self, bad_diam):
        """Ensure non-positive diameter is rejected immediately."""
        with pytest.raises(ValueError, match="Lens diameter must be strictly positive"):
            ThickLens(50.0, -50.0, 5.0, 1.5, diameter=bad_diam)


# =============================================================================
# 3. Multi-Component Galileo Afocal Alignment & Ray Transfer Matrices
# =============================================================================

class TestGalileoSystemAfocalAndTransferMatrices:
    """
    Empirical tests of MultiComponentSystem and Galilean afocal configurations.
    """

    @pytest.mark.parametrize("f1, f2", [
        (100.0, -25.0),
        (200.0, -50.0),
        (50.0, -10.0),
        (150.0, -30.0),
        (80.0, -20.0),
    ])
    def test_galileo_thin_afocal_and_magnification(self, f1, f2):
        """
        Verify thin-lens Galilean telescope:
            - Net power C == 0
            - Angular magnification M_alpha = D = -f1 / f2
            - Lateral magnification m = A = -f2 / f1 = 1 / M_alpha
            - Transfer matrix determinant det(M) == 1.0
        """
        sys = create_galileo_system(f1=f1, f2=f2, thick=False)
        mat = sys.system_matrix()
        cp = sys.cardinal_points()

        expected_mag = -f1 / f2
        assert abs(mat.C) < 1e-15, f"Non-zero optical power: {mat.C}"
        assert abs(mat.D - expected_mag) < 1e-10, f"Angular magnification mismatch: {mat.D}"
        assert abs(mat.A - (1.0 / expected_mag)) < 1e-10, f"Lateral magnification mismatch: {mat.A}"
        assert abs(mat.determinant() - 1.0) < 1e-12
        assert cp.is_afocal is True

    @pytest.mark.parametrize("f1, f2", [
        (100.0, -25.0),
        (200.0, -50.0),
        (50.0, -10.0),
        (150.0, -30.0),
        (80.0, -20.0),
    ])
    def test_galileo_thick_afocal_and_magnification(self, f1, f2):
        """
        Verify thick-lens Galilean telescope:
            - Net power C == 0 (with air gap compensating for principal plane offsets)
            - Angular magnification M_alpha = D = -f1 / f2
            - Lateral magnification m = A = 1 / M_alpha
            - Determinant det(M) == 1.0
        """
        sys = create_galileo_system(f1=f1, f2=f2, thick=True, n=1.5, t1=5.0, t2=3.0)
        mat = sys.system_matrix()
        cp = sys.cardinal_points()

        expected_mag = -f1 / f2
        assert abs(mat.C) < 1e-12, f"Thick Galileo optical power non-zero: {mat.C}"
        assert abs(mat.D - expected_mag) < 1e-6, f"Thick Galileo angular mag mismatch: {mat.D}"
        assert abs(mat.A - (1.0 / expected_mag)) < 1e-6, f"Thick Galileo lateral mag mismatch: {mat.A}"
        assert abs(mat.determinant() - 1.0) < 1e-11
        assert cp.is_afocal is True

    def test_galileo_paraxial_ray_propagation_collimation(self):
        """
        Trace paraxial rays through the afocal Galileo system:
            1. An on-axis parallel ray (y=5 mm, u=0 rad) exits parallel (u_out=0) with height y_out = 0.25 * y_in = 1.25 mm.
            2. An off-axis ray entering at axis (y=0, u=0.01 rad) exits with angle u_out = 4.0 * u_in = 0.04 rad.
        """
        sys = create_galileo_system(f1=100.0, f2=-25.0, thick=True, n=1.5, t1=5.0, t2=3.0)
        mat = sys.system_matrix()

        # On-axis ray
        y_out, u_out = mat.propagate_ray(5.0, 0.0)
        assert abs(y_out - 1.25) < 1e-4
        assert abs(u_out) < 1e-12

        # Off-axis ray
        y_out2, u_out2 = mat.propagate_ray(0.0, 0.01)
        assert abs(u_out2 - 0.04) < 1e-4

    def test_galileo_separation_perturbation_continuity(self):
        """
        Perturb the inter-lens separation away from afocal condition:
            d = d_afocal + delta
        Verify that C = -delta / (f1 * f2) and determinant remains identically 1.0.
        """
        f1, f2 = 100.0, -25.0
        sys_nom = create_galileo_system(f1, f2, thick=False)
        d_nom = sys_nom._separations[0]

        deltas = np.linspace(-10.0, 10.0, 21)
        for delta in deltas:
            sys = create_galileo_system(f1, f2, d=d_nom + delta, thick=False)
            mat = sys.system_matrix()
            expected_c = delta / (f1 * f2)

            assert abs(mat.determinant() - 1.0) < 1e-12
            assert abs(mat.C - expected_c) < 1e-12

    def test_cascaded_10_element_transfer_matrix_determinant(self):
        """
        Stress-test matrix cascading through 10 mixed optical elements and air spaces.
        Verify det(M) == 1.00000000000 to high precision.
        """
        elements = [
            (ThickLens(50.0, -50.0, 4.0, 1.5, name="L1"), 10.0),
            (ThinLens(100.0, name="L2"), 15.0),
            (ThickLens(-40.0, 60.0, 3.0, 1.6, name="L3"), 8.0),
            (ThinLens(-50.0, name="L4"), 12.0),
            (ThickLens(80.0, -80.0, 5.0, 1.52, name="L5"), 20.0),
            (ThinLens(75.0, name="L6"), 10.0),
            (ThickLens(45.0, 45.0, 4.0, 1.5, name="L7"), 5.0),
            (ThinLens(-30.0, name="L8"), 14.0),
            (ThickLens(-100.0, 50.0, 3.5, 1.65, name="L9"), 25.0),
            (ThinLens(60.0, name="L10"), 0.0),
        ]
        sys = MultiComponentSystem(elements=elements, name="10_Element_Stress_Train")
        mat = sys.system_matrix()

        assert abs(mat.determinant() - 1.0) < 1e-11
        assert len(sys) == 10


# =============================================================================
# 4. Aspheric Surface & Tolerancing Immutability
# =============================================================================

class TestAsphericAndTolerancingStress:
    """
    Stress-tests aspheric surface geometry and tolerancing immutability.
    """

    @pytest.mark.parametrize("k", [-3.0, -1.0, -0.5, 0.0, 1.0, 2.5])
    def test_aspheric_paraxial_vertex_curvature_theorem(self, k):
        """
        Verify that even with arbitrary conic constants k and high-order polynomial
        deformation coefficients, the paraxial vertex curvature is identically 1/R.
        """
        R = 30.0
        c = 1.0 / R
        surf = AsphericSurfaceModel(
            radius=R,
            conic=k,
            aspheric_coeffs=[1.5e-6, -2.3e-9, 4.1e-13],
        )

        assert abs(surf.paraxial_curvature() - c) < 1e-15

        # Numerical second derivative at vertex r=0
        h = 1e-5
        sag_p = surf.sag(h)
        sag_0 = surf.sag(0.0)
        sag_m = surf.sag(-h)
        d2z_dr2 = (sag_p - 2.0 * sag_0 + sag_m) / (h ** 2)

        assert abs(d2z_dr2 - c) < 1e-4

    def test_aspheric_lens_paraxial_equivalence_to_thick_lens(self):
        """
        Verify that an AsphericLens produces the exact same paraxial transfer matrix
        and cardinal points as a ThickLens with identical vertex radii.
        """
        r1, r2, t, n = 40.0, -60.0, 5.0, 1.5
        thick = ThickLens(r1, r2, t, n)
        asph = AsphericLens(
            r1=r1,
            r2=r2,
            thickness=t,
            glass_or_index=n,
            k1=-1.0,  # Parabolic front
            k2=0.5,   # Oblate rear
            coeffs1=[1e-6, -1e-9],
            coeffs2=[2e-6],
        )

        mat_thick = thick.transfer_matrix()
        mat_asph = asph.transfer_matrix()

        assert abs(mat_thick.A - mat_asph.A) < 1e-12
        assert abs(mat_thick.B - mat_asph.B) < 1e-12
        assert abs(mat_thick.C - mat_asph.C) < 1e-12
        assert abs(mat_thick.D - mat_asph.D) < 1e-12

        cp_thick = thick.cardinal_points()
        cp_asph = asph.cardinal_points()

        assert abs(cp_thick.efl - cp_asph.efl) < 1e-10
        assert abs(cp_thick.p1 - cp_asph.p1) < 1e-10
        assert abs(cp_thick.p2 - cp_asph.p2) < 1e-10

    def test_tolerancing_monte_carlo_nominal_immutability(self):
        """
        Verify that running a Monte Carlo tolerancing simulation leaves the nominal
        lens parameters completely unaltered (no state corruption).
        """
        lens = ThickLens(50.0, -50.0, 5.0, 1.5, name="MC_Target_Lens")
        nom_r1 = lens.r1
        nom_r2 = lens.r2
        nom_t = lens.thickness
        nom_n = lens.refractive_index()

        mf = PhysicalMeritFunction()
        mf.add_operand(TargetEFLOperand(target_efl=50.847458, weight=1.0))

        tolerances = [
            PerturbationSpec(parameter_name="r1", element_index=0, delta=0.1, distribution="normal"),
            PerturbationSpec(parameter_name="r2", element_index=0, delta=0.1, distribution="normal"),
            PerturbationSpec(parameter_name="thickness", element_index=0, delta=0.05, distribution="uniform"),
        ]

        engine = TolerancingEngine(system=lens, merit_function=mf)
        mc_report = engine.monte_carlo_simulation(
            system=lens,
            merit_function=mf,
            tolerances=tolerances,
            num_trials=300,
            seed=42,
        )

        # Nominal parameters must be completely intact
        assert lens.r1 == nom_r1
        assert lens.r2 == nom_r2
        assert lens.thickness == nom_t
        assert lens.refractive_index() == nom_n

        # Results must be statistically sound
        assert len(mc_report.trials) == 300
        efls = [t.efl for t in mc_report.trials]
        assert not np.any(np.isnan(efls))
        assert abs(mc_report.mean_merit - 0.0) < 1.0  # nominal merit was ~0
        assert mc_report.std_merit >= 0.0
        assert hasattr(mc_report, "percentile_50")
        assert hasattr(mc_report, "percentile_90")
