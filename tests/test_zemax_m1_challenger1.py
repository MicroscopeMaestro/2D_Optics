"""
Empirical Adversarial Stress Test Suite for Milestone 1 - Challenger 1
Author: Challenger 1 (critic, specialist)

Objectives:
1. Multi-field ray generation and aiming under extreme field angles (+/- 45 deg, +/- 80 deg),
   zero/negative object distance, and telecentric systems.
2. Full-pupil Hexapolar and Gauss-Legendre Quadrature patterns across 100+ random configurations.
   Verify no crashes, valid direction cosines, and weight normalization.
3. Empirical audit of paraxial entrance pupil calculation and refraction fidelity.
"""

import math
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.system_config import (
    SystemConfig, FieldType, FieldPoint, RayPatternType, ApertureType
)
from optics_app.engine.pupil_solve import (
    calculate_entrance_pupil,
    aim_collimated_ray,
    aim_ray_to_pupil,
    generate_pupil_grid,
    find_stop_surface_index,
)
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics
from optics_app.engine.matrix_optics import refraction_matrix, space_matrix
from optics_app.utils.materials import get_refractive_index


def _build_standard_test_system(d0: float = 50.0):
    """Build a standard test singlet with stop downstream."""
    stack = SurfaceStack()
    s0 = Surface(name="Object", thickness=d0, glass="AIR")
    s1 = Surface(name="L1 Front", radius=30.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0)
    s2 = Surface(name="L1 Rear", radius=-40.0, thickness=10.0, glass="AIR", semi_diameter=15.0)
    s3 = Surface(name="Stop", radius=np.inf, thickness=30.0, glass="AIR", semi_diameter=5.0)
    s3.is_stop = True
    s3.surface_type = SurfaceType.STOP
    s4 = Surface(name="Detector", radius=np.inf, thickness=0.0, glass="AIR", semi_diameter=15.0)
    s4.surface_type = SurfaceType.DETECTOR
    stack.surfaces.extend([s0, s1, s2, s3, s4])
    stack._update_accumulated_positions()
    return stack


# ==============================================================================
# Challenge 1: Extreme Field Angles (+/- 45 deg, +/- 80 deg) & Diagonal Fields
# ==============================================================================

class TestExtremeFieldAnglesAiming:
    """Stress test multi-field ray generation and aiming under extreme field angles."""

    @pytest.mark.parametrize("ang_x,ang_y", [
        (0.0, 0.0),
        (45.0, 0.0),
        (-45.0, 0.0),
        (0.0, 45.0),
        (0.0, -45.0),
        (45.0, 45.0),
        (-45.0, -45.0),
        (45.0, -45.0),
        (80.0, 0.0),
        (-80.0, 0.0),
        (0.0, 80.0),
        (0.0, -80.0),
        (80.0, 80.0),
        (-80.0, -80.0),
        (89.0, 0.0),
        (0.0, 89.0),
    ])
    def test_collimated_extreme_angles_direction_cosines_and_fidelity(self, ang_x, ang_y):
        stack = _build_standard_test_system()
        z_ep, r_ep, _ = calculate_entrance_pupil(stack.surfaces)

        # Test at various normalized pupil coordinates
        pupil_coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-0.707, 0.707)]
        for px, py in pupil_coords:
            ray = aim_collimated_ray(
                surfaces=stack.surfaces,
                field_angle_deg=(ang_x, ang_y),
                pupil_norm=(px, py),
                wavelength_um=0.5876,
                z_ep_global=z_ep,
                r_ep=r_ep,
            )

            # 1. No NaNs or Infs
            assert not np.isnan(ray.pos).any(), f"NaN in pos for ({ang_x}, {ang_y})"
            assert not np.isinf(ray.pos).any(), f"Inf in pos for ({ang_x}, {ang_y})"
            assert not np.isnan(ray.dir).any(), f"NaN in dir for ({ang_x}, {ang_y})"
            assert not np.isinf(ray.dir).any(), f"Inf in dir for ({ang_x}, {ang_y})"

            # 2. Strict direction cosine normalization: kx^2 + ky^2 + kz^2 == 1.0
            norm = float(np.linalg.norm(ray.dir))
            assert abs(norm - 1.0) < 1e-11, f"Direction cosine norm != 1.0: {norm}"

            # 3. Ray propagation fidelity: does ray pass through entrance pupil target?
            # When propagated from Z=0 to Z=z_ep along ray.dir:
            if abs(ray.dir[2]) > 1e-5:
                t = (z_ep - ray.pos[2]) / ray.dir[2]
                hit_pupil = ray.pos + t * ray.dir
                expected_pupil = np.array([px * r_ep, py * r_ep, z_ep])
                diff = float(np.linalg.norm(hit_pupil - expected_pupil))
                assert diff < 1e-8, f"Pupil intercept mismatch for ({ang_x}, {ang_y}) at (px,py)=({px},{py}): diff={diff}"

    def test_optimizer_generate_rays_extreme_field_angles(self):
        """Verify OpticalOptimizer._generate_rays behaves robustly with extreme field angles."""
        stack = _build_standard_test_system()
        sys = OpticalSystem(stack)
        config = SystemConfig()
        config.field_type = FieldType.ANGLE_DEG
        config.fields = [
            FieldPoint(x=0.0, y=0.0, weight=1.0),
            FieldPoint(x=0.0, y=45.0, weight=2.0),
            FieldPoint(x=0.0, y=-45.0, weight=1.0),
            FieldPoint(x=0.0, y=80.0, weight=1.0),
            FieldPoint(x=0.0, y=-80.0, weight=1.0),
        ]
        config.beam_type = "Collimated (Parallel Beam)"
        opt = OpticalOptimizer(sys, system_config=config, num_rays=25, entrance_pupil_radius=5.0)
        rays = opt._generate_rays(pupil_radius=5.0, wavelengths=[0.5876], num_rays=25)

        assert len(rays) > 0
        field_indices = {r.field_idx for r in rays}
        assert field_indices == {0, 1, 2, 3, 4}

        for r in rays:
            norm = float(np.linalg.norm(r.dir))
            assert abs(norm - 1.0) < 1e-9
            assert not np.isnan(r.pos).any()
            assert not np.isnan(r.dir).any()


# ==============================================================================
# Challenge 2: Zero and Negative Object Distance (d0 = 0, d0 < 0)
# ==============================================================================

class TestZeroAndNegativeObjectDistance:
    """Stress test ray aiming and generation when object distance is zero or negative."""

    @pytest.mark.parametrize("d0", [0.0, -5.0, -25.0, -100.0])
    def test_paraxial_pupil_solve_zero_and_negative_d0(self, d0):
        stack = _build_standard_test_system(d0=d0)
        z_ep, r_ep, stop_idx = calculate_entrance_pupil(stack.surfaces)

        assert stop_idx == 3
        assert r_ep > 0.0
        assert not math.isnan(z_ep)
        assert not math.isnan(r_ep)

        # Collimated ray aiming
        r_coll = aim_collimated_ray(stack.surfaces, (0.0, 30.0), (0.5, 0.5), z_ep_global=z_ep, r_ep=r_ep)
        assert not np.isnan(r_coll.pos).any()
        assert not np.isnan(r_coll.dir).any()
        assert abs(np.linalg.norm(r_coll.dir) - 1.0) < 1e-11

        # Finite conjugate ray aiming
        r_finite = aim_ray_to_pupil(stack.surfaces, (0.0, 5.0, 0.0), (0.5, 0.5), z_ep_global=z_ep, r_ep=r_ep)
        assert not np.isnan(r_finite.pos).any()
        assert not np.isnan(r_finite.dir).any()
        assert abs(np.linalg.norm(r_finite.dir) - 1.0) < 1e-11

    @pytest.mark.parametrize("d0", [0.0, -10.0])
    def test_optimizer_ray_generation_with_zero_and_negative_d0(self, d0):
        stack = _build_standard_test_system(d0=d0)
        sys = OpticalSystem(stack)
        config = SystemConfig()
        config.field_type = FieldType.OBJECT_HEIGHT_MM
        config.fields = [
            FieldPoint(x=0.0, y=0.0, weight=1.0),
            FieldPoint(x=0.0, y=5.0, weight=2.0),
        ]
        config.beam_type = "Divergent (Point Source)"
        opt = OpticalOptimizer(sys, system_config=config, num_rays=15, entrance_pupil_radius=5.0)
        rays = opt._generate_rays(pupil_radius=5.0, wavelengths=[0.5876], num_rays=15)

        assert len(rays) > 0
        for r in rays:
            assert abs(np.linalg.norm(r.dir) - 1.0) < 1e-9
            assert not np.isnan(r.pos).any()
            assert not np.isnan(r.dir).any()


# ==============================================================================
# Challenge 3: Telecentric Systems & Negative Entrance Pupil (Z_ep < 0)
# ==============================================================================

class TestTelecentricAndNegativeEntrancePupil:
    """Stress test systems where the entrance pupil is negative or telecentric."""

    def test_strongly_negative_entrance_pupil_aiming(self):
        """Construct a system where Z_ep_global is strongly negative."""
        surfaces = [
            Surface(name="Object", thickness=10.0),
            Surface(name="S1", radius=-25.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0),
            Surface(name="S2", radius=25.0, thickness=2.0, glass="AIR", semi_diameter=15.0),
            Surface(name="Stop", radius=np.inf, thickness=20.0, glass="AIR", semi_diameter=4.0),
            Surface(name="Image", thickness=0.0)
        ]
        surfaces[3].is_stop = True
        surfaces[3].surface_type = SurfaceType.STOP

        z_ep, r_ep, stop_idx = calculate_entrance_pupil(surfaces)

        # Force a negative entrance pupil position test
        test_z_ep = -50.0
        test_r_ep = 5.0

        for ang_y in [0.0, 30.0, -30.0, 60.0]:
            ray = aim_collimated_ray(
                surfaces=surfaces,
                field_angle_deg=(0.0, ang_y),
                pupil_norm=(0.0, 1.0),
                z_ep_global=test_z_ep,
                r_ep=test_r_ep,
            )

            assert not np.isnan(ray.pos).any()
            assert not np.isnan(ray.dir).any()
            assert abs(np.linalg.norm(ray.dir) - 1.0) < 1e-11

            # Intercept check at z = test_z_ep
            t = (test_z_ep - ray.pos[2]) / ray.dir[2]
            hit = ray.pos + t * ray.dir
            assert abs(hit[1] - test_r_ep) < 1e-8

    def test_object_telecentric_system_aperture_clamping(self):
        """Object-space telecentric system with stop near front focal point."""
        surfaces = [
            Surface(name="Object", thickness=50.0),
            Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0),
            Surface(name="S2", radius=np.inf, thickness=96.7, glass="AIR", semi_diameter=20.0),
            Surface(name="Stop", radius=np.inf, thickness=20.0, glass="AIR", semi_diameter=5.0),
            Surface(name="Image", thickness=0.0)
        ]
        surfaces[3].is_stop = True
        surfaces[3].surface_type = SurfaceType.STOP

        z_ep, r_ep, stop_idx = calculate_entrance_pupil(surfaces)
        # Should remain numerically bounded and finite
        assert not math.isnan(z_ep)
        assert not math.isnan(r_ep)
        assert not math.isinf(z_ep)
        assert not math.isinf(r_ep)
        assert 0.1 <= r_ep <= 500.0


# ==============================================================================
# Challenge 4: Full-Pupil Sampling across 100+ Randomized Configurations
# ==============================================================================

class TestFullPupilSamplingRandomConfigurations:
    """Stress test Hexapolar and Gauss-Legendre Quadrature across 100+ random configurations."""

    def test_100_random_optical_configurations(self):
        rng = np.random.RandomState(42)
        n_configs = 100
        glasses = ["N-BK7", "SF11", "F2", "AIR"]

        for trial in range(n_configs):
            # 1. Generate randomized surface stack (2 to 5 optical surfaces)
            n_surfs = rng.randint(2, 5)
            surfaces = [Surface(name="Object", thickness=rng.uniform(0.0, 100.0))]

            for s_i in range(n_surfs):
                rad = rng.choice([rng.uniform(-150.0, 150.0), np.inf])
                if abs(rad) < 5.0 and not np.isinf(rad):
                    rad = 5.0 * np.sign(rad)
                thick = rng.uniform(1.0, 20.0)
                glass = rng.choice(glasses)
                semi_d = rng.uniform(5.0, 25.0)
                surfaces.append(Surface(
                    name=f"S{s_i+1}",
                    radius=float(rad),
                    thickness=float(thick),
                    glass=glass,
                    semi_diameter=float(semi_d)
                ))

            stop_pos = rng.randint(1, len(surfaces))
            surfaces[stop_pos].is_stop = True
            surfaces[stop_pos].surface_type = SurfaceType.STOP
            surfaces.append(Surface(name="Detector", thickness=0.0, surface_type=SurfaceType.DETECTOR))

            stack = SurfaceStack()
            stack.surfaces.extend(surfaces)
            stack._update_accumulated_positions()
            sys = OpticalSystem(stack)

            # 2. Randomize configuration
            config = SystemConfig()
            pattern = rng.choice([RayPatternType.HEXAPOLAR, RayPatternType.GAUSSIAN_QUADRATURE])
            config.ray_pattern = pattern
            config.pupil_rings = int(rng.randint(1, 6))
            config.pupil_arms = int(rng.choice([4, 6, 8, 12]))
            config.field_type = rng.choice([FieldType.ANGLE_DEG, FieldType.OBJECT_HEIGHT_MM])

            max_field = 30.0 if config.field_type == FieldType.ANGLE_DEG else 15.0
            config.fields = [
                FieldPoint(x=0.0, y=0.0, weight=1.0),
                FieldPoint(x=0.0, y=rng.uniform(0.1, max_field), weight=2.0),
                FieldPoint(x=rng.uniform(-max_field, max_field), y=rng.uniform(-max_field, max_field), weight=1.0),
            ]
            config.beam_type = rng.choice(["Collimated (Parallel Beam)", "Divergent (Point Source)"])

            # 3. Test generate_pupil_grid directly
            p_pat_str = pattern.value if hasattr(pattern, "value") else str(pattern)
            grid_pts = generate_pupil_grid(
                pattern=p_pat_str,
                rings=config.pupil_rings,
                arms=config.pupil_arms,
                include_chief_ray=True,
                linear_arms=True,
                max_radius=1.0
            )

            # Validate grid points
            assert len(grid_pts) > 0, f"Trial {trial}: Empty pupil grid"
            w_sum = sum(w for _, _, w in grid_pts)
            assert abs(w_sum - 1.0) < 1e-5, f"Trial {trial}: Weight sum = {w_sum} != 1.0"

            for px, py, w in grid_pts:
                r2 = px * px + py * py
                assert r2 <= 1.0 + 1e-6, f"Trial {trial}: Pupil point ({px}, {py}) outside unit disk"
                assert w >= -1e-12, f"Trial {trial}: Negative weight {w}"

            # 4. Test OpticalOptimizer._generate_rays
            opt = OpticalOptimizer(sys, system_config=config, num_rays=len(grid_pts), entrance_pupil_radius=5.0)
            rays = opt._generate_rays(pupil_radius=5.0, wavelengths=[0.5876], num_rays=len(grid_pts))
            assert len(rays) > 0, f"Trial {trial}: No rays generated"

            for r in rays:
                norm = float(np.linalg.norm(r.dir))
                assert abs(norm - 1.0) < 1e-9, f"Trial {trial}: Non-unit dir norm {norm}"
                assert not np.isnan(r.pos).any(), f"Trial {trial}: NaN in ray pos"
                assert not np.isnan(r.dir).any(), f"Trial {trial}: NaN in ray dir"
                assert hasattr(r, "pupil_weight"), f"Trial {trial}: Missing pupil_weight"
                assert hasattr(r, "field_idx"), f"Trial {trial}: Missing field_idx"


# ==============================================================================
# Challenge 5: Empirical Bug Audit of calculate_entrance_pupil
# ==============================================================================

class TestCalculateEntrancePupilRefractionBug:
    """
    Adversarial verification of the entrance pupil refraction bug:
    In pupil_solve.py line 109-111, hasattr(cur_surf, "n") evaluates to False
    because Surface has no 'n' method. Thus n_in=1.0 and n_out=1.0 always,
    causing calculate_entrance_pupil to completely bypass optical refraction.
    """

    def test_refraction_matrix_n_attribute_defect(self):
        """Demonstrate that Surface instances lack an 'n' method, causing pupil_solve to fail to refract."""
        surf = Surface(name="L1", glass="N-BK7")
        assert not hasattr(surf, "n"), (
            "Surface does not have an 'n' method; pupil_solve.py lines 109-111 "
            "rely on hasattr(cur_surf, 'n') which silently falls back to 1.0!"
        )

    def test_entrance_pupil_refraction_neglect(self):
        """
        Verify that calculate_entrance_pupil fails to compute the true paraxial
        entrance pupil magnification and position for an aperture stop behind a lens.
        """
        # S0: Object
        s0 = Surface(name="Object", thickness=50.0)
        # S1: Plano-convex front face R1=25.0, d=5.0, N-BK7
        s1 = Surface(name="L1 Front", radius=25.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5)
        # S2: Flat exit face R2=inf, d=10.0, AIR
        s2 = Surface(name="L1 Rear", radius=np.inf, thickness=10.0, glass="AIR", semi_diameter=12.5)
        # S3: Stop, semi_diameter=4.0
        s3 = Surface(name="Stop", radius=np.inf, thickness=15.0, glass="AIR", semi_diameter=4.0)
        s3.is_stop = True
        s3.surface_type = SurfaceType.STOP

        z_ep, r_ep, stop_idx = calculate_entrance_pupil([s0, s1, s2, s3], wavelength_um=0.5876)

        # Theoretical Paraxial ABCD computation:
        # Refraction at S1: AIR (1.0) -> N-BK7 (~1.5168), R1=25.0
        # Space across L1: d=5.0, n=1.5168
        # Refraction at S2: N-BK7 -> AIR, R2=inf
        # Space across air: d=10.0, n=1.0
        n_glass = get_refractive_index("N-BK7", 0.5876)
        M_r1 = refraction_matrix(R=25.0, n1=1.0, n2=n_glass)
        M_s1 = space_matrix(d=5.0, n=n_glass)
        M_r2 = refraction_matrix(R=float("inf"), n1=n_glass, n2=1.0)
        M_s2 = space_matrix(d=10.0, n=1.0)
        M_true = M_s2 @ M_r2 @ M_s1 @ M_r1

        expected_z_ep_from_s1 = M_true.B / M_true.A  # ~18.3364 mm
        expected_z_ep_global = 50.0 + expected_z_ep_from_s1  # ~68.3364 mm
        expected_r_ep = 4.0 / abs(M_true.A)  # ~5.5162 mm

        # VERIFICATION OF REMEDIATION:
        # calculate_entrance_pupil now correctly refracts through S1 and propagates across L1
        # matching true paraxial ABCD values: Z_ep ~ 68.3364 mm and R_ep ~ 5.5162 mm
        assert abs(z_ep - expected_z_ep_global) < 1e-3, (
            f"calculate_entrance_pupil Z_ep {z_ep:.4f} != expected {expected_z_ep_global:.4f}"
        )
        assert abs(r_ep - expected_r_ep) < 1e-3, (
            f"calculate_entrance_pupil R_ep {r_ep:.4f} != expected {expected_r_ep:.4f}"
        )
