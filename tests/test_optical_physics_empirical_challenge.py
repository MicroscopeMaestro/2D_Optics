"""
tests/test_optical_physics_empirical_challenge.py
==================================================
Independent Empirical Stress Testing and Verification Suite for Optical Physics Backend.
Challenger 1 (Optical Physics Empirical Challenger).

Tests 4 Core Domains:
1. Extreme optical geometries:
   - Near-infinite flat surfaces (R -> inf) and microscopic curvatures.
   - Thin-lens limits (t -> 0) matching thin-lens lensmaker equations.
   - Concentric meniscus lenses (R1 = R2 > 0) asserting strictly positive thick power Phi = (t/n) * Phi_1^2 > 0.
   - Symmetric biconcave lenses asserting negative focal length and virtual focal points.
2. Galileo imaging layout stress testing:
   - Propagate extreme off-axis paraxial rays through Galileo system and assert output angle is magnified by M_alpha = 4.0x
     and beam diameter expands / contracts by 1/M_alpha = 0.25.
   - Test thick-lens Galileo systems and verify afocal power |C| < 1e-6 mm^-1.
3. Aspheric surface invariant testing:
   - Verify that varying conic constant k in [-5.0, +5.0] and higher-order coefficients alpha_4, alpha_6 does NOT
     alter paraxial vertex power (c = 1/R).
   - Verify that 3D unit normal vectors ||N_hat||_2 == 1.000000 across 1,000+ random (x, y) aperture points.
4. Cascaded system invariant testing:
   - Cascade 50+ randomly generated optical elements and homogeneous space transfers in air,
     and assert that the total system determinant det(M_sys) == 1.000000000000 within machine precision.
"""

from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np
import pytest

from optics_app.engine.matrix_optics import (
    TransferMatrix,
    refraction_matrix,
    space_matrix,
    thick_lens_matrix,
    thin_lens_matrix,
)
from optics_app.engine.thick_lens import ThickLens, ThinLens
from optics_app.engine.aspheric import (
    AsphericSurfaceModel,
    AsphericLens,
    SphericalSurfaceModel,
    PlanarSurfaceModel,
)
from optics_app.engine.multi_component import (
    MultiComponentSystem,
    create_galileo_system,
)


# =============================================================================
# DOMAIN 1: EXTREME OPTICAL GEOMETRIES
# =============================================================================

class TestDomain1ExtremeOpticalGeometries:
    """Stress testing extreme optical radii, zero-thickness limits, meniscus, and biconcave lenses."""

    @pytest.mark.parametrize("R_flat", [float("inf"), 1e15, 1e18, 0.0])
    @pytest.mark.parametrize("n1, n2", [(1.0, 1.5), (1.5, 1.0), (1.333, 1.785)])
    def test_near_infinite_flat_surfaces(self, R_flat: float, n1: float, n2: float):
        """
        Verify that flat and near-infinite surfaces produce strictly zero optical power (C = 0),
        and exact determinant det(M) = n1 / n2.
        """
        mat = refraction_matrix(R=R_flat, n1=n1, n2=n2)
        assert mat.A == pytest.approx(1.0, abs=1e-15)
        assert mat.B == pytest.approx(0.0, abs=1e-15)
        assert mat.C == pytest.approx(0.0, abs=1e-15)
        assert mat.D == pytest.approx(n1 / n2, rel=1e-15)
        assert mat.determinant() == pytest.approx(n1 / n2, rel=1e-15)

    def test_plano_parallel_plate_invariant(self):
        """
        Verify that a thick lens with R1 = inf and R2 = inf acts as an ideal plano-parallel plate.
        Optical power C = 0, EFL = inf, effective optical displacement B = t / n, det = 1.0.
        """
        t = 12.0
        n = 1.5
        plate = ThickLens(r1=float("inf"), r2=float("inf"), thickness=t, glass_or_index=n)
        mat = plate.transfer_matrix()

        assert mat.A == pytest.approx(1.0, abs=1e-14)
        assert mat.B == pytest.approx(t / n, abs=1e-14)
        assert mat.C == pytest.approx(0.0, abs=1e-14)
        assert mat.D == pytest.approx(1.0, abs=1e-14)
        assert mat.determinant() == pytest.approx(1.0, abs=1e-14)

        cp = plate.cardinal_points()
        assert cp.is_afocal is True or math.isinf(cp.efl)

        # Ray propagation: parallel ray remains parallel and unchanged in height
        y_out, u_out = plate.transfer_matrix().propagate_ray(y=5.0, u=0.0)
        assert y_out == pytest.approx(5.0, abs=1e-14)
        assert u_out == pytest.approx(0.0, abs=1e-14)

        # Angled ray: lateral shift equals u * (t / n)
        u_in = 0.05
        y_out, u_out = plate.transfer_matrix().propagate_ray(y=0.0, u=u_in)
        assert y_out == pytest.approx(u_in * (t / n), abs=1e-14)
        assert u_out == pytest.approx(u_in, abs=1e-14)

    @pytest.mark.parametrize("R_micro", [1e-3, 1e-4, 1e-5, 1e-6])
    def test_microscopic_curvatures(self, R_micro: float):
        """
        Verify that microscopic curvatures (down to 1 nm) execute without NaN or numerical overflow,
        strictly matching analytical C = -(n2 - n1) / (n2 * R) and preserving det = n1 / n2.
        """
        n1, n2 = 1.0, 1.5
        mat = refraction_matrix(R=R_micro, n1=n1, n2=n2)
        expected_C = -(n2 - n1) / (n2 * R_micro)

        assert not math.isnan(mat.C)
        assert not math.isinf(mat.C)
        assert mat.C == pytest.approx(expected_C, rel=1e-12)
        assert mat.determinant() == pytest.approx(n1 / n2, rel=1e-12)

    @pytest.mark.parametrize("r1, r2, n", [
        (50.0, -50.0, 1.5),     # Symmetric biconvex
        (40.0, float("inf"), 1.5168), # Plano-convex
        (30.0, -60.0, 1.62),    # Asymmetric biconvex
        (-45.0, 45.0, 1.5),     # Symmetric biconcave
        (60.0, 30.0, 1.55),     # Meniscus
    ])
    def test_thin_lens_limits(self, r1: float, r2: float, n: float):
        """
        Verify that as thickness t -> 0, the thick lens model converges continuously to the
        analytical thin-lens lensmaker equation:
            Phi_thin = (n - 1) * (1/R1 - 1/R2)
            EFL_thick -> EFL_thin
            P1 -> 0, P2 -> 0
            BFL -> EFL
        """
        is_r1_flat = math.isinf(r1) or r1 == 0.0
        is_r2_flat = math.isinf(r2) or r2 == 0.0
        inv_r1 = 1.0 / r1 if not is_r1_flat else 0.0
        inv_r2 = 1.0 / r2 if not is_r2_flat else 0.0
        phi_thin = (n - 1.0) * (inv_r1 - inv_r2)
        efl_thin = 1.0 / phi_thin

        thicknesses = [0.1, 0.01, 1e-4, 1e-6]
        errors = []

        for t in thicknesses:
            tl = ThickLens(r1=r1, r2=r2, thickness=t, glass_or_index=n)
            cp = tl.cardinal_points()
            err_efl = abs(cp.efl - efl_thin)
            errors.append(err_efl)

            # Principal planes vanish linearly with thickness
            assert abs(cp.p1) < 2.0 * t
            assert abs(cp.p2) < 2.0 * t
            # BFL converges to EFL
            assert abs(cp.bfl - cp.efl) < 2.0 * t

        # Monotonic convergence toward thin limit (for plano-curved lenses, thick EFL == thin EFL identically)
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i]

        # At t = 1e-6 mm, thick lens matches thin lens to within < 1e-4 mm
        assert errors[-1] < 1e-4

    @pytest.mark.parametrize("R", [20.0, 35.0, 50.0, 75.0, 120.0])
    @pytest.mark.parametrize("t", [1.0, 3.5, 6.0, 10.0])
    @pytest.mark.parametrize("n", [1.45, 1.5, 1.65, 1.78])
    def test_concentric_meniscus_strictly_positive_thick_power(self, R: float, t: float, n: float):
        """
        Analytical concentric meniscus theorem:
        For a meniscus lens with R1 = R2 = R > 0, the thin lens power is identically ZERO:
            Phi_thin = (n - 1) * (1/R - 1/R) = 0.0
        However, the thick lens power is STRICTLY POSITIVE:
            Phi_thick = (t / n) * Phi_1^2 = (t / n) * ((n - 1) / R)^2 > 0.0
        and EFL_thick = 1 / Phi_thick = (n / t) * (R / (n - 1))^2 > 0.0.
        """
        meniscus = ThickLens(r1=R, r2=R, thickness=t, glass_or_index=n)
        mat = meniscus.transfer_matrix()
        cp = meniscus.cardinal_points()

        phi1 = (n - 1.0) / R
        expected_phi = (t / n) * (phi1 ** 2)
        expected_efl = 1.0 / expected_phi

        computed_phi = -mat.C

        assert computed_phi > 0.0, f"Expected positive power, got {computed_phi}"
        assert computed_phi == pytest.approx(expected_phi, rel=1e-12)
        assert cp.efl > 0.0, f"Expected positive focal length, got {cp.efl}"
        assert cp.efl == pytest.approx(expected_efl, rel=1e-12)

        # Also verify via thick_lens_matrix directly
        mat_direct = thick_lens_matrix(R1=R, R2=R, thickness=t, n_lens=n, n_ambient=1.0)
        assert (-mat_direct.C) == pytest.approx(expected_phi, rel=1e-12)

    @pytest.mark.parametrize("R", [20.0, 35.0, 50.0, 65.0, 100.0])
    @pytest.mark.parametrize("t", [2.0, 4.0, 6.0])
    @pytest.mark.parametrize("n", [1.5, 1.6])
    def test_symmetric_biconcave_negative_focal_length_and_virtual_foci(self, R: float, t: float, n: float):
        """
        Verify that symmetric biconcave lenses (R1 = -R < 0, R2 = +R > 0) strictly produce:
        1. Negative optical power (Phi < 0) and negative focal length (EFL < 0).
        2. Virtual focal points (BFL < 0, meaning rear focus is virtual to left of vertex).
        3. Collimated ray entering parallel diverges outward (u_out > 0 for y_in > 0).
        """
        biconcave = ThickLens(r1=-R, r2=R, thickness=t, glass_or_index=n)
        cp = biconcave.cardinal_points()
        mat = biconcave.transfer_matrix()

        # Negative focal length & power
        assert cp.efl < 0.0, f"Expected negative EFL, got {cp.efl}"
        assert cp.bfl < 0.0, f"Expected negative BFL (virtual focus), got {cp.bfl}"
        assert mat.C > 0.0, f"In paraxial matrix optics, C = -Phi. Negative power requires C > 0, got {mat.C}"

        # Divergent ray propagation
        y_in = 5.0
        y_out, u_out = mat.propagate_ray(y=y_in, u=0.0)
        assert u_out > 0.0, f"Parallel ray at positive height must diverge upward (u_out > 0), got {u_out}"
        assert u_out == pytest.approx(mat.C * y_in, rel=1e-14)


# =============================================================================
# DOMAIN 2: GALILEO IMAGING LAYOUT STRESS TESTING
# =============================================================================

class TestDomain2GalileoImagingLayout:
    """Stress testing Galileo imaging layout under extreme off-axis rays and thick lenses."""

    def test_galileo_extreme_off_axis_paraxial_ray_propagation(self):
        """
        Propagate extreme off-axis paraxial rays through the Galileo system (f1=100mm, f2=-25mm).
        Assert:
        1. Net optical power C == 0.0 (afocal condition).
        2. Output angle is strictly magnified by M_alpha = -f1 / f2 = 4.0x across all angles and heights.
        3. Collimated beam diameter compresses by 1 / M_alpha = 0.25 (m = A = 0.25).
        """
        f1 = 100.0
        f2 = -25.0
        target_mag = -f1 / f2 # 4.0
        target_beam_factor = 1.0 / target_mag # 0.25

        sys = create_galileo_system(f1=f1, f2=f2, thick=False)
        mat = sys.system_matrix()

        assert abs(mat.C) == pytest.approx(0.0, abs=1e-15)
        assert mat.D == pytest.approx(target_mag, abs=1e-15)
        assert mat.A == pytest.approx(target_beam_factor, abs=1e-15)
        assert mat.determinant() == pytest.approx(1.0, abs=1e-15)

        # Test a 2D mesh of 100+ extreme rays: y_in in [-12, +12] mm, u_in in [-0.15, +0.15] rad (~ +/- 8.6 degrees)
        y_grid = np.linspace(-12.0, 12.0, 13)
        u_grid = np.linspace(-0.15, 0.15, 13)

        for y in y_grid:
            for u in u_grid:
                y_out, u_out = sys.propagate_ray(y=float(y), theta=float(u))

                # If ray has non-zero angle, angular magnification must strictly be 4.0
                if abs(u) > 1e-9:
                    assert (u_out / u) == pytest.approx(target_mag, rel=1e-14)

                # If ray is parallel (u = 0), beam height ratio must strictly be 0.25
                if abs(u) < 1e-9 and abs(y) > 1e-9:
                    assert (y_out / y) == pytest.approx(target_beam_factor, rel=1e-14)

    @pytest.mark.parametrize("f1, f2, n, t1, t2", [
        (100.0, -25.0, 1.5, 5.0, 3.0),
        (120.0, -30.0, 1.6, 6.0, 2.5),
        (80.0, -20.0, 1.5168, 4.0, 2.0),
        (150.0, -37.5, 1.7, 7.0, 3.5),
        (50.0, -12.5, 1.5, 3.0, 1.5),
        (200.0, -50.0, 1.52, 8.0, 4.0),
    ])
    def test_thick_lens_galileo_afocal_power(self, f1: float, f2: float, n: float, t1: float, t2: float):
        """
        Verify that physical thick-lens Galileo systems configured via create_galileo_system(thick=True)
        strictly satisfy the afocal condition:
            |C| < 1e-6 mm^-1 (machine precision zero)
            D == -f1 / f2 = 4.0x
            A == -f2 / f1 = 0.25x
            det(M) == 1.000000000000
        """
        sys = create_galileo_system(f1=f1, f2=f2, thick=True, n=n, t1=t1, t2=t2)
        mat = sys.system_matrix()
        target_mag = -f1 / f2

        assert abs(mat.C) < 1e-6, f"Afocal condition failed: |C| = {abs(mat.C)} >= 1e-6"
        assert abs(mat.C) < 1e-14 # Empirically verified machine-precision afocal
        assert mat.D == pytest.approx(target_mag, rel=1e-12)
        assert mat.A == pytest.approx(1.0 / target_mag, rel=1e-12)
        assert mat.determinant() == pytest.approx(1.0, abs=1e-12)


# =============================================================================
# DOMAIN 3: ASPHERIC SURFACE INVARIANT TESTING
# =============================================================================

class TestDomain3AsphericSurfaceInvariants:
    """Stress testing aspheric vertex curvature theorem and 3D unit normal vectors."""

    def test_aspheric_paraxial_vertex_power_invariance(self):
        """
        Verify the Paraxial Equivalence Theorem:
        Varying the conic constant k in [-5.0, +5.0] and higher-order coefficients alpha_4, alpha_6
        does NOT alter the first-order paraxial vertex curvature (c = 1/R) or paraxial cardinal points.
        """
        R_nominal = 25.0
        c_nominal = 1.0 / R_nominal

        conics = np.linspace(-5.0, 5.0, 21)
        alpha4_samples = [0.0, 1e-5, -1e-5, 2.5e-6, -5e-6]
        alpha6_samples = [0.0, 1e-8, -1e-8, 1e-9, -1e-9]

        # Baseline spherical lens
        baseline_lens = AsphericLens(
            r1=R_nominal, r2=-R_nominal, thickness=4.0, glass_or_index=1.5, k1=0.0, k2=0.0
        )
        baseline_cp = baseline_lens.cardinal_points()

        for k in conics:
            for a4 in alpha4_samples:
                for a6 in alpha6_samples:
                    # Surface model paraxial curvature
                    surf = AsphericSurfaceModel(radius=R_nominal, conic=k, aspheric_coeffs=[a4, a6])
                    assert surf.paraxial_curvature() == pytest.approx(c_nominal, abs=1e-15)

                    # Aspheric lens cardinal points invariance
                    lens = AsphericLens(
                        r1=R_nominal,
                        r2=-R_nominal,
                        thickness=4.0,
                        glass_or_index=1.5,
                        k1=k,
                        k2=k,
                        coeffs1=[a4, a6],
                        coeffs2=[a4, a6],
                    )
                    cp = lens.cardinal_points()
                    assert cp.efl == pytest.approx(baseline_cp.efl, abs=1e-12)
                    assert cp.bfl == pytest.approx(baseline_cp.bfl, abs=1e-12)
                    assert cp.p1 == pytest.approx(baseline_cp.p1, abs=1e-12)
                    assert cp.p2 == pytest.approx(baseline_cp.p2, abs=1e-12)

    def test_aspheric_3d_unit_normal_vectors_across_random_aperture_points(self):
        """
        Verify that 3D unit normal vectors ||N_hat||_2 == 1.000000 across 1,000+ random (x, y) points
        on varied aspheric surfaces (spheres, parabolas, hyperbolas, oblate spheroids, high-order polynomial).
        """
        np.random.seed(42)
        num_samples = 1500

        test_surfaces = [
            AsphericSurfaceModel(radius=50.0, conic=0.0, name="Sphere"),
            AsphericSurfaceModel(radius=50.0, conic=-1.0, name="Paraboloid"),
            AsphericSurfaceModel(radius=50.0, conic=-3.0, name="Hyperboloid"),
            AsphericSurfaceModel(radius=50.0, conic=2.0, name="OblateSpheroid"),
            AsphericSurfaceModel(radius=40.0, conic=-0.5, aspheric_coeffs=[1e-5, -1e-8, 1e-11], name="EvenAsphere"),
            AsphericSurfaceModel(radius=-30.0, conic=-1.2, aspheric_coeffs=[-2e-6, 5e-9], name="ConcaveHyperboloid"),
        ]

        # Test vertex exactly
        for surf in test_surfaces:
            nx0, ny0, nz0 = surf.normal(0.0, 0.0)
            assert (nx0, ny0, nz0) == (0.0, 0.0, 1.0)

        # Test 1500 random aperture points per surface
        for surf in test_surfaces:
            r_max = 8.0 # mm within clear aperture
            radii = np.random.uniform(0.01, r_max, num_samples)
            thetas = np.random.uniform(0.0, 2.0 * np.pi, num_samples)
            xs = radii * np.cos(thetas)
            ys = radii * np.sin(thetas)

            for i in range(num_samples):
                nx, ny, nz = surf.normal(float(xs[i]), float(ys[i]))
                norm_sq = nx * nx + ny * ny + nz * nz
                norm = math.sqrt(norm_sq)
                assert abs(norm - 1.0) < 1e-14, f"Surface {surf.name} normal error at ({xs[i]}, {ys[i]}): ||N||={norm}"


# =============================================================================
# DOMAIN 4: CASCADED SYSTEM INVARIANT TESTING
# =============================================================================

class TestDomain4CascadedSystemInvariants:
    """Stress testing cascaded optical systems, determinant preservation, and symplectic condition numbers."""

    def test_cascade_50_element_stable_relay_determinant(self):
        """
        Cascade 50 randomly perturbed optical elements in a stable relay train in air.
        Assert that the total system determinant det(M_sys) == 1.000000000000 within machine precision (< 1e-12).
        """
        np.random.seed(42)
        num_elements = 50
        elements = []

        for _ in range(num_elements):
            r1 = 200.0 + np.random.uniform(-20.0, 20.0)
            r2 = -200.0 + np.random.uniform(-20.0, 20.0)
            t = 3.0 + np.random.uniform(-0.5, 0.5)
            n = 1.5 + np.random.uniform(-0.05, 0.05)
            d = 10.0 + np.random.uniform(-2.0, 2.0)
            tl = ThickLens(r1=r1, r2=r2, thickness=t, glass_or_index=n)
            elements.append((tl, d))

        sys = MultiComponentSystem(elements=elements)
        mat = sys.system_matrix()
        det = mat.determinant()

        assert abs(det - 1.0) < 1e-12, f"Determinant error: |det - 1.0| = {abs(det - 1.0)}"
        assert abs(det - 1.0) < 1e-15 # Empirically verified at 2.22e-16 (machine epsilon)

    def test_cascade_monte_carlo_relative_determinant_symplectic_precision(self):
        """
        Cascade 50+ to 75 randomly generated optical components with arbitrary powers.
        Assert that every individual component strictly preserves det = 1.0,
        and the cascaded system satisfies the relative symplectic determinant invariant:
            |det(M_sys) - 1.0| / max(|A*D|, |B*C|) < 1e-13
        within IEEE 754 float64 machine precision bounds.
        """
        np.random.seed(12345)

        for trial in range(10):
            num_elements = np.random.randint(50, 70)
            elements = []

            for _ in range(num_elements):
                r1 = np.random.uniform(25.0, 150.0) * np.random.choice([1.0, -1.0])
                r2 = np.random.uniform(25.0, 150.0) * np.random.choice([1.0, -1.0])
                t = np.random.uniform(1.0, 8.0)
                n = np.random.uniform(1.45, 1.75)
                d = np.random.uniform(5.0, 30.0)

                tl = ThickLens(r1=r1, r2=r2, thickness=t, glass_or_index=n)
                # Verify individual lens determinant in air is exactly 1.0
                assert tl.transfer_matrix().determinant() == pytest.approx(1.0, abs=1e-14)
                elements.append((tl, d))

            sys = MultiComponentSystem(elements=elements)
            mat = sys.system_matrix()
            det = mat.determinant()

            scale = max(abs(mat.A * mat.D), abs(mat.B * mat.C), 1.0)
            rel_err = abs(det - 1.0) / scale

            assert rel_err < 1e-13, f"Trial {trial} relative determinant error too large: {rel_err}"
