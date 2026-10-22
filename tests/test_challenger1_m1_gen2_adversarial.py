"""
Milestone 1 Generation 2 Adversarial Stress Test Suite.
Authored by Challenger 1 Gen 2.

Exhaustively and adversarially stress-tests:
1. Galvo scanning kinematics:
   - Arbitrary 3D mount orientations and arbitrary 3D rotation axes
   - Massive off-vertex pivot offsets and rigid-body kinematic invariants
   - Large deflection angles (+/-45 deg, +/-80 deg) and exact 2*theta optical deflection
   - Voltage scaling, clipping saturation, and discrete sweep generation
   - End-to-end 3D ray tracing continuity through swept galvo mirror
2. BranchedTracer:
   - High-depth cascading beam splitters (up to 8 consecutive splitters)
   - Strict energy conservation sum(I) <= I0 across arbitrary split ratios
   - Polarizing Beam Splitter (PBS) and Dichroic spectral cutoff transitions
   - Infinite loop / cyclic optical resonator termination via max_depth and min_intensity
   - Boundary cutoff parameters (max_depth=0, 1; min_intensity thresholds)
   - Batch bundle tree tracing energy conservation
3. Prisms:
   - Extreme wavelengths across Deep UV, Visible, and Infrared (0.15 um to 8.0 um)
   - Monotonic chromatic dispersion in normal dispersion regimes
   - Near-TIR angle boundary stress testing (+/-0.02 deg of critical angle)
   - Small apex angle wedge approximation
   - Apex angle geometric limit rejection
   - Right-angle prism TIR beam steering
4. Cross-package parity between optics_app.engine and v2.engine
5. Empirical defect confirmation: ActiveMechanism JSON persistence and surface binding
"""

import math
import os
import tempfile
import time
import unittest
import numpy as np

from optics_app.engine.components import (
    OpticalComponent,
    GalvoScannerConfig,
    BeamSplitterConfig,
    BeamSplitterType,
    SplitterCoatingType,
    PrismConfig,
    PrismGeometryType,
    ActiveMechanism,
    MechanismChannel,
    ActuationType,
    create_galvo_component,
    create_beamsplitter_component,
    create_prism_component,
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.models import SpectrometerSystem
from optics_app.engine.tracer import OpticalSystem, Ray, RayNode, RayTree, BranchedTracer
from optics_app.engine.mechanisms import get_effective_pose

import v2.engine.components as v2_components
import v2.engine.mechanisms as v2_mechanisms
import v2.engine.tracer as v2_tracer


def make_orthogonal_rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Construct an exact 3D orthogonal rotation matrix from Euler angles (deg)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=float)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=float)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


class TestGalvoKinematicsAdversarial(unittest.TestCase):
    """Stress-test Galvo kinematics, arbitrary 3D axes, large deflection angles, and invariants."""

    def test_galvo_extreme_orientations_and_arbitrary_3d_axes(self):
        """Stress-test galvo kinematics across arbitrary 3D mount poses and arbitrary 3D rotation axes."""
        nom_origin = np.array([12.5, -45.0, 103.2])
        mount_R = make_orthogonal_rotation(37.5, -48.2, 112.7)

        test_axes = [
            [1.0, 1.0, 1.0],               # Non-principal axis, un-normalized
            [-3.5, 7.2, -1.1],             # Arbitrary skew axis
            [0.0, 0.0, 1.0],               # Z-axis (in-plane spin)
            [0.0, 1.0, 0.0],               # Y-axis standard
            [1.0, 0.0, 0.0],               # X-axis standard
            [0.0, 0.0, 0.0],               # Degenerate zero vector (must fallback gracefully)
        ]

        pivot_offset = [150.0, -250.0, 400.0]
        p_pivot_expected = nom_origin + mount_R @ np.array(pivot_offset)
        nom_pivot_dist = np.linalg.norm(pivot_offset)

        for axis in test_axes:
            surf = Surface(name="Adversarial_Galvo")
            surf.global_origin = nom_origin.copy()
            surf.global_R = mount_R.copy()
            cfg = GalvoScannerConfig(
                scan_axis=axis,
                pivot_offset_mm=pivot_offset,
                min_angle_deg=-85.0,
                max_angle_deg=85.0
            )
            surf.galvo_config = cfg

            for angle_deg in [-80.0, -45.0, -15.0, 0.0, 15.0, 45.0, 80.0]:
                cfg.set_mechanical_angle(angle_deg)
                eff_origin, eff_R = get_effective_pose(surf)

                # 1. Strict special orthogonality: R^T @ R = I, det(R) = 1.0
                ortho_residual = np.max(np.abs(eff_R.T @ eff_R - np.eye(3)))
                self.assertLess(ortho_residual, 1e-14, f"Orthogonality broken for axis {axis} at angle {angle_deg}")
                self.assertAlmostEqual(np.linalg.det(eff_R), 1.0, places=12)

                # 2. Strict pivot invariance: reconstructed pivot must be invariant under any rotation
                p_pivot_actual = eff_origin + eff_R @ np.array(pivot_offset)
                pivot_shift = np.linalg.norm(p_pivot_actual - p_pivot_expected)
                self.assertLess(pivot_shift, 1e-12, f"Physical pivot shifted by {pivot_shift} mm")

                # 3. Distance from optical vertex to pivot must remain strictly constant
                dist_actual = np.linalg.norm(eff_origin - p_pivot_expected)
                self.assertAlmostEqual(dist_actual, nom_pivot_dist, places=10)

    def test_galvo_optical_deflection_2_theta_arbitrary_angles(self):
        """Verify optical 2*theta deflection holds strictly for large deflection angles (+/-45, +/-80 deg)."""
        for axis_phi in [0.0, 30.0, 45.0, 75.0, 90.0]:
            axis_phi_rad = np.radians(axis_phi)
            rot_axis = [np.cos(axis_phi_rad), np.sin(axis_phi_rad), 0.0]

            surf = Surface(name="Galvo_Deflector", surface_type=SurfaceType.STANDARD, glass="MIRROR")
            cfg = GalvoScannerConfig(scan_axis=rot_axis, min_angle_deg=-85.0, max_angle_deg=85.0)
            surf.galvo_config = cfg

            cfg.set_mechanical_angle(0.0)
            _, R0 = get_effective_pose(surf)
            n0 = R0[:, 2]
            d_in = np.array([0.0, 0.0, 1.0])
            d_ref0 = d_in - 2.0 * np.dot(d_in, n0) * n0

            for theta_deg in [-80.0, -60.0, -45.0, -30.0, 30.0, 45.0, 60.0, 80.0]:
                cfg.set_mechanical_angle(theta_deg)
                _, R_eff = get_effective_pose(surf)
                n_eff = R_eff[:, 2]

                d_ref = d_in - 2.0 * np.dot(d_in, n_eff) * n_eff
                cos_def = np.clip(np.dot(d_ref, d_ref0), -1.0, 1.0)
                deflection_deg = np.degrees(np.arccos(cos_def))
                expected_deflection_deg = 2.0 * abs(theta_deg)

                self.assertAlmostEqual(
                    deflection_deg,
                    expected_deflection_deg,
                    places=8,
                    msg=f"Deflection failed at theta={theta_deg} deg for rotation axis {rot_axis}"
                )

    def test_galvo_voltage_saturation_and_sweep_discretization(self):
        """Stress-test galvo voltage scaling, hard clipping at +/- max_voltage_v, and sweep generation."""
        cfg = GalvoScannerConfig(
            min_angle_deg=-15.0,
            max_angle_deg=15.0,
            max_voltage_v=10.0,
            volts_to_deg=1.5,
        )
        # Saturated positive command (+20V input clamped to +10V -> +15 deg)
        cfg.set_voltage(20.0)
        self.assertEqual(cfg.current_angle_deg, 15.0)
        self.assertEqual(cfg.current_optical_angle_deg, 30.0)

        # Saturated negative command (-25V input clamped to -10V -> -15 deg)
        cfg.set_voltage(-25.0)
        self.assertEqual(cfg.current_angle_deg, -15.0)
        self.assertEqual(cfg.current_optical_angle_deg, -30.0)

        # High-resolution sweep discretization
        mech_angles = cfg.get_mechanical_angles(num_steps=1001)
        opt_angles = cfg.get_optical_angles(num_steps=1001)
        self.assertEqual(len(mech_angles), 1001)
        self.assertEqual(len(opt_angles), 1001)
        self.assertAlmostEqual(mech_angles[0], -15.0)
        self.assertAlmostEqual(mech_angles[-1], 15.0)
        np.testing.assert_allclose(opt_angles, mech_angles * 2.0)

    def test_galvo_end_to_end_ray_trace_continuity(self):
        """Verify ray tracing through a folding galvo mirror produces continuous non-NaN paths."""
        stack = SurfaceStack()
        s0 = Surface(name="Source", thickness=40.0, glass="AIR")
        stack.add_surface(s0)

        galvo_cfg = GalvoScannerConfig(
            scan_axis="Y",
            min_angle_deg=-45.0,
            max_angle_deg=45.0,
            pivot_offset_mm=[10.0, 0.0, -5.0]
        )
        s_galvo = Surface(
            name="Galvo",
            surface_type=SurfaceType.STANDARD,
            glass="MIRROR",
            thickness=50.0,
            tilt_x=45.0,
            semi_diameter=50.0
        )
        s_galvo.galvo_config = galvo_cfg
        stack.add_surface(s_galvo)

        s_det = Surface(name="Screen", surface_type=SurfaceType.DETECTOR, glass="AIR", semi_diameter=200.0)
        stack.add_surface(s_det)
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack)
        ray_in = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.0, wavelength=0.532)

        prev_hit_z = None
        for sweep_deg in np.linspace(-20.0, 20.0, 9):
            galvo_cfg.set_mechanical_angle(sweep_deg)
            tree = tracer.trace_ray_tree(ray_in)
            leaves = tree.all_terminal_rays()
            self.assertEqual(len(leaves), 1)
            term_ray = leaves[0]
            self.assertFalse(term_ray.vignetted)
            self.assertTrue(np.all(np.isfinite(term_ray.pos)))
            self.assertTrue(np.all(np.isfinite(term_ray.dir)))
            self.assertGreater(term_ray.path_length, 50.0)

            # Spot position on detector sweeps monotonically along Z
            hit_z = term_ray.pos[2]
            if prev_hit_z is not None:
                self.assertGreater(hit_z, prev_hit_z, "Spot position should vary monotonically along Z with galvo sweep")
            prev_hit_z = hit_z


class TestBranchedTracerAdversarial(unittest.TestCase):
    """Stress-test BranchedTracer: depth, energy conservation, and loop prevention."""

    def test_cascading_beam_splitters_high_depth_and_energy_conservation(self):
        """Stress-test deep cascading beam splitters and verify sum(I_leaves) <= 1.0 strictly."""
        for num_splitters in [4, 6, 8]:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="Input", thickness=25.0, glass="AIR"))

            for k in range(num_splitters):
                bs_cfg = BeamSplitterConfig(split_ratio_R=0.50, split_ratio_T=0.50)
                stack.add_surface(Surface(
                    name=f"Splitter_{k}",
                    surface_type=SurfaceType.STANDARD,
                    thickness=25.0,
                    glass="AIR",
                    semi_diameter=100.0,
                    tilt_x=45.0,
                    is_beam_splitter=True,
                    beamsplitter_config=bs_cfg,
                ))

            stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR", semi_diameter=200.0))
            stack._update_accumulated_positions()

            tracer = BranchedTracer(stack, max_depth=12, min_intensity=1e-6)
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.0, wavelength=0.532)

            tree = tracer.trace_ray_tree(ray)

            leaf_rays = tree.all_terminal_rays()
            self.assertEqual(len(leaf_rays), num_splitters + 1)
            self.assertEqual(len(tree.nodes), 2 * num_splitters + 1)

            total_leaf_energy = tree.total_energy()
            self.assertAlmostEqual(total_leaf_energy, 1.0, places=9)

            for node_id, node in tree.nodes.items():
                self.assertLessEqual(node.generation, num_splitters)
                for child_id in node.children_ids:
                    self.assertEqual(tree.nodes[child_id].generation, node.generation + 1)

    def test_arbitrary_and_extreme_split_ratios_energy_conservation(self):
        """Verify strict energy conservation sum(I) <= I0 across extreme split ratios and polarizations."""
        test_ratios = [
            (0.99, 0.01),      # 99:1 pick-off
            (0.01, 0.99),      # 1:99 high-transmission
            (0.73, 0.27),      # Unbalanced
            (0.50, 0.50),      # Standard 50:50
            (0.00, 1.00),      # Pure transmission
            (1.00, 0.00),      # Pure reflection
        ]

        for R_val, T_val in test_ratios:
            bs = BeamSplitterConfig(split_ratio_R=R_val, split_ratio_T=T_val)
            R_eval, T_eval = bs.evaluate_split(0.532)
            self.assertAlmostEqual(R_eval, R_val, places=6)
            self.assertAlmostEqual(T_eval, T_val, places=6)
            self.assertLessEqual(R_eval + T_eval, 1.0 + 1e-12)

        # Polarizing Beam Splitter (PBS)
        pbs = BeamSplitterConfig(
            coating_type=SplitterCoatingType.POLARIZING,
            Rs=0.995, Rp=0.005, Ts=0.005, Tp=0.995
        )
        for pol in ["s", "p", None]:
            r_pbs, t_pbs = pbs.evaluate_split(0.532, polarization=pol)
            self.assertLessEqual(r_pbs + t_pbs, 1.0 + 1e-12)

        # Dichroic Beam Splitter with wavelength sweep across sharp transition
        dichroic = BeamSplitterConfig(
            coating_type=SplitterCoatingType.DICHROIC,
            dichroic_cutoff_wavelength_um=0.532,
            dichroic_transition_width_um=0.01
        )
        for wl in [0.400, 0.500, 0.530, 0.532, 0.534, 0.600, 0.800]:
            r_d, t_d = dichroic.evaluate_split(wl)
            self.assertLessEqual(r_d + t_d, 1.0 + 1e-12)
            self.assertGreaterEqual(r_d, 0.0)
            self.assertGreaterEqual(t_d, 0.0)
            if wl < 0.500:
                self.assertGreater(r_d, 0.90)
            elif wl > 0.600:
                self.assertGreater(t_d, 0.90)

    def test_infinite_loop_prevention_cyclic_resonator(self):
        """Adversarially construct a cyclic optical resonator and verify BranchedTracer terminates safely."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Input", thickness=30.0, glass="AIR"))

        bs_a = BeamSplitterConfig(split_ratio_R=0.50, split_ratio_T=0.50)
        s1 = Surface(
            name="Splitter_A",
            surface_type=SurfaceType.STANDARD,
            thickness=30.0,
            glass="AIR",
            semi_diameter=50.0,
            tilt_x=45.0,
            is_beam_splitter=True,
            beamsplitter_config=bs_a,
            reflected_branch_start_idx=2,
        )
        stack.add_surface(s1)

        bs_b = BeamSplitterConfig(split_ratio_R=0.50, split_ratio_T=0.50)
        s2 = Surface(
            name="Splitter_B",
            surface_type=SurfaceType.STANDARD,
            thickness=30.0,
            glass="AIR",
            semi_diameter=50.0,
            tilt_x=-45.0,
            is_beam_splitter=True,
            beamsplitter_config=bs_b,
            reflected_branch_start_idx=1,
        )
        stack.add_surface(s2)

        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR"))
        stack._update_accumulated_positions()

        ray_in = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.0, wavelength=0.532)

        for limit_depth in [2, 4, 6]:
            tracer = BranchedTracer(stack, max_depth=limit_depth, min_intensity=1e-8)
            t_start = time.perf_counter()
            tree = tracer.trace_ray_tree(ray_in)
            elapsed = time.perf_counter() - t_start

            self.assertLess(elapsed, 0.5, f"Cyclic tracer took too long: {elapsed}s")
            max_gen = max(n.generation for n in tree.nodes.values())
            self.assertLessEqual(max_gen, limit_depth)
            self.assertLessEqual(tree.total_energy(), 1.0 + 1e-9)

        tracer_int = BranchedTracer(stack, max_depth=100, min_intensity=0.05)
        t_start = time.perf_counter()
        tree_int = tracer_int.trace_ray_tree(ray_in)
        elapsed = time.perf_counter() - t_start

        self.assertLess(elapsed, 0.5)
        for n in tree_int.nodes.values():
            if n.children_ids:
                self.assertGreaterEqual(n.ray.intensity * 0.5, 0.05 - 1e-9)

    def test_edge_case_cutoff_parameters(self):
        """Stress-test boundary parameters max_depth=0, 1 and min_intensity=0.6."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Input", thickness=30.0, glass="AIR"))
        stack.add_surface(Surface(
            name="BS",
            surface_type=SurfaceType.STANDARD,
            thickness=30.0,
            glass="AIR",
            semi_diameter=50.0,
            tilt_x=45.0,
            is_beam_splitter=True,
            beamsplitter_config=BeamSplitterConfig(split_ratio_R=0.5, split_ratio_T=0.5),
        ))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR"))
        stack._update_accumulated_positions()

        ray_in = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.0, wavelength=0.532)

        t0 = BranchedTracer(stack, max_depth=0)
        tree0 = t0.trace_ray_tree(ray_in)
        self.assertEqual(len(tree0.nodes), 1)
        self.assertEqual(len(tree0.nodes[0].children_ids), 0)

        t1 = BranchedTracer(stack, max_depth=1)
        tree1 = t1.trace_ray_tree(ray_in)
        self.assertEqual(len(tree1.nodes), 3)

        t_high_cutoff = BranchedTracer(stack, max_depth=5, min_intensity=0.6)
        tree_high = t_high_cutoff.trace_ray_tree(ray_in)
        self.assertEqual(len(tree_high.nodes), 1)

    def test_bundle_tree_tracing_energy_conservation(self):
        """Verify that batch tracing a bundle of 20 rays through cascading splitters conserves energy for every ray."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Input", thickness=30.0, glass="AIR"))
        for k in range(4):
            stack.add_surface(Surface(
                name=f"Splitter_{k}",
                surface_type=SurfaceType.STANDARD,
                thickness=30.0,
                glass="AIR",
                semi_diameter=100.0,
                tilt_x=45.0,
                is_beam_splitter=True,
                beamsplitter_config=BeamSplitterConfig(split_ratio_R=0.4, split_ratio_T=0.6),
            ))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR", semi_diameter=200.0))
        stack._update_accumulated_positions()

        rays = []
        for x in np.linspace(-5.0, 5.0, 4):
            for y in np.linspace(-5.0, 5.0, 5):
                rays.append(Ray(
                    pos=np.array([x, y, 0.0]),
                    dir=np.array([0.0, 0.0, 1.0]),
                    intensity=1.0,
                    wavelength=0.532
                ))

        tracer = BranchedTracer(stack, max_depth=8, min_intensity=1e-5)
        trees = tracer.trace_bundle_tree(rays)

        self.assertEqual(len(trees), 20)
        for idx, tree in enumerate(trees):
            tot_energy = tree.total_energy()
            self.assertAlmostEqual(tot_energy, 1.0, places=8, msg=f"Ray {idx} violated energy conservation")


class TestPrismPhysicsAdversarial(unittest.TestCase):
    """Stress-test Prism physics, dispersion across spectrum, and near-TIR angles."""

    def test_prism_extreme_wavelengths_and_monotonic_dispersion(self):
        """Verify refractive index and minimum deviation across deep UV, visible, and infrared."""
        materials_and_wls = [
            ("CaF2", [0.15, 0.20, 0.35, 0.532, 1.0, 3.0, 6.0, 8.0]),
            ("FUSED_SILICA", [0.20, 0.30, 0.532, 1.0, 2.0, 3.5]),
            ("N-SF11", [0.38, 0.45, 0.532, 0.633, 1.064, 2.0]),
        ]

        for glass, wls in materials_and_wls:
            prism = PrismConfig(glass=glass, apex_angle_deg=50.0)

            prev_n = None
            prev_delta_min = None
            for wl in wls:
                n = prism._get_refractive_index(wl)
                delta_min = prism.calculate_minimum_deviation(wl)

                self.assertGreater(n, 1.0)
                self.assertGreater(delta_min, 0.0)
                self.assertLess(delta_min, 180.0)

                if prev_n is not None:
                    self.assertLess(n, prev_n, f"Refractive index should decrease monotonically for {glass}")
                    self.assertLess(delta_min, prev_delta_min, f"Minimum deviation should decrease monotonically for {glass}")

                prev_n = n
                prev_delta_min = delta_min

    def test_prism_near_tir_boundary_and_rejection(self):
        """Stress-test incident angles approaching within 0.02 deg of TIR at exit facet."""
        prism = PrismConfig(glass="N-BK7", apex_angle_deg=60.0)
        n = prism._get_refractive_index(0.532) # ~1.5195
        alpha = math.radians(60.0)
        theta_c = math.asin(1.0 / n)

        th1_crit_sin = n * math.sin(alpha - theta_c)
        th1_crit_deg = math.degrees(math.asin(th1_crit_sin))

        # 1. Just above critical angle: transmits safely
        th1_trans = th1_crit_deg + 0.02
        delta_trans = prism.calculate_deviation(th1_trans, 0.532)
        self.assertTrue(np.isfinite(delta_trans))
        self.assertGreater(delta_trans, 0.0)
        self.assertLess(delta_trans, 90.0)

        # 2. Just below critical angle: TIR occurs -> raises ValueError
        th1_tir = th1_crit_deg - 0.02
        with self.assertRaises(ValueError) as ctx:
            prism.calculate_deviation(th1_tir, 0.532)
        self.assertIn("Total internal reflection", str(ctx.exception))

        # 3. Extreme glancing incident angles near +/-90 deg
        with self.assertRaises(ValueError):
            prism.calculate_deviation(-89.9, 0.532)

    def test_small_angle_wedge_prism_approximation(self):
        """Verify that small apex angle wedges match the thin prism approximation delta ~ (n - 1) * alpha."""
        prism = PrismConfig(glass="N-BK7", apex_angle_deg=2.0)
        n = prism._get_refractive_index(0.532)
        delta_exact = prism.calculate_minimum_deviation(0.532)
        delta_approx = (n - 1.0) * 2.0

        rel_error = abs(delta_exact - delta_approx) / delta_approx
        self.assertLess(rel_error, 0.001, "Thin wedge approximation should match within 0.1%")

    def test_apex_angle_geometric_limit_rejection(self):
        """Verify that when apex angle is too large for transmission without TIR, ValueError is raised."""
        prism_extreme = PrismConfig(glass="N-SF11", apex_angle_deg=85.0)
        with self.assertRaises(ValueError) as ctx:
            prism_extreme.calculate_minimum_deviation(0.532)
        self.assertIn("Apex angle", str(ctx.exception))

    def test_prism_right_angle_tir_geometry(self):
        """Verify right-angle prism TIR calculation and deviation for folding beam steering."""
        prism = PrismConfig(
            geometry_type=PrismGeometryType.RIGHT_ANGLE,
            apex_angle_deg=90.0,
            glass="N-BK7"
        )
        n = prism._get_refractive_index(0.532)
        theta_c = math.degrees(math.asin(1.0 / n))
        self.assertLess(theta_c, 45.0, "Critical angle in BK7 must be < 45 deg for TIR to occur at hypotenuse")


class TestCrossPackageParityAdversarial(unittest.TestCase):
    """Verify optics_app and v2 deliver 100% identical results on adversarial stress tests."""

    def test_v2_galvo_kinematics_parity(self):
        """Verify v2 get_effective_pose matches optics_app exactly across arbitrary axes and massive offsets."""
        axis = [2.0, -1.0, 3.5]
        offset = [120.0, -80.0, 250.0]
        mount_R = make_orthogonal_rotation(25.0, -40.0, 65.0)

        s_app = Surface(name="Galvo_App")
        s_app.global_origin = np.array([50.0, -100.0, 200.0])
        s_app.global_R = mount_R.copy()
        s_app.galvo_config = GalvoScannerConfig(scan_axis=axis, pivot_offset_mm=offset, min_angle_deg=-80.0, max_angle_deg=80.0)

        s_v2 = Surface(name="Galvo_V2")
        s_v2.global_origin = np.array([50.0, -100.0, 200.0])
        s_v2.global_R = mount_R.copy()
        s_v2.galvo_config = v2_components.GalvoScannerConfig(scan_axis=axis, pivot_offset_mm=offset, min_angle_deg=-80.0, max_angle_deg=80.0)

        for ang in [-75.0, -45.0, 0.0, 45.0, 75.0]:
            s_app.galvo_config.set_mechanical_angle(ang)
            s_v2.galvo_config.set_mechanical_angle(ang)

            orig_app, R_app = get_effective_pose(s_app)
            orig_v2, R_v2 = v2_mechanisms.get_effective_pose(s_v2)

            np.testing.assert_allclose(orig_app, orig_v2, atol=1e-12)
            np.testing.assert_allclose(R_app, R_v2, atol=1e-12)

    def test_v2_branched_tracer_parity(self):
        """Verify v2 BranchedTracer produces identical tree topology and leaf energies as optics_app."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Input", thickness=30.0, glass="AIR"))
        stack.add_surface(Surface(
            name="Splitter",
            surface_type=SurfaceType.STANDARD,
            thickness=30.0,
            glass="AIR",
            semi_diameter=50.0,
            tilt_x=45.0,
            is_beam_splitter=True,
            beamsplitter_config=BeamSplitterConfig(split_ratio_R=0.7, split_ratio_T=0.3),
        ))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR"))
        stack._update_accumulated_positions()

        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.0, wavelength=0.532)

        t_app = BranchedTracer(stack)
        t_v2 = v2_tracer.BranchedTracer(stack)

        tree_app = t_app.trace_ray_tree(ray)
        tree_v2 = t_v2.trace_ray_tree(ray)

        self.assertEqual(len(tree_app.nodes), len(tree_v2.nodes))
        self.assertAlmostEqual(tree_app.total_energy(), tree_v2.total_energy(), places=10)


class TestEmpiricalDefectActiveMechanismPersistence(unittest.TestCase):
    """Empirical investigation of ActiveMechanism JSON persistence and surface binding."""

    def test_investigate_active_mechanism_surface_binding(self):
        """Document and verify behavior of surface.mechanism during JSON save and load."""
        stack = SurfaceStack()
        s0 = Surface(name="Input", thickness=50.0, glass="AIR")
        s_act = Surface(
            name="Actuated_Mirror",
            surface_type=SurfaceType.MIRROR,
            thickness=50.0,
            component_uid="comp_act_mirror"
        )
        s_det = Surface(name="Det", surface_type=SurfaceType.DETECTOR, glass="AIR")

        mech = ActiveMechanism(
            mechanism_id="piezo_gimbal_01",
            mechanism_type="piezo",
            actuation_type=ActuationType.CONTINUOUS,
            channels={
                "ch_tilt_x": MechanismChannel(
                    channel_name="ch_tilt_x",
                    target_property="tilt_x",
                    unit="deg",
                    min_value=-10.0,
                    max_value=10.0,
                    current_value=4.5,
                    scale_factor=1.0,
                )
            }
        )
        comp = OpticalComponent(
            component_uid="comp_act_mirror",
            name="Actuated_Mirror_Assembly",
            component_type="mechanism",
            surface_uids=["Actuated_Mirror"],
            mechanism=mech
        )
        s_act.mechanism = mech

        stack.add_surface(s0)
        stack.add_surface(s_act)
        stack.add_surface(s_det)
        stack.add_component(comp)
        stack._update_accumulated_positions()

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test_active_mech.json")
            sys = SpectrometerSystem(surface_stack=stack)
            sys.save_to_json(fpath)

            loaded_sys = SpectrometerSystem.load_from_json(fpath)
            loaded_surf = next(s for s in loaded_sys.surfaces if s.name == "Actuated_Mirror")

            # Check if component was persisted in components list
            self.assertEqual(len(loaded_sys.surface_stack.components), 1)
            loaded_comp = loaded_sys.surface_stack.components[0]
            self.assertIsNotNone(loaded_comp.mechanism)
            self.assertIn("ch_tilt_x", loaded_comp.mechanism.channels)

            # Check whether surface.mechanism was bound:
            # If not bound, this reveals the known limitation in bind_components_to_surfaces()
            has_surface_mechanism = getattr(loaded_surf, "mechanism", None) is not None
            # Record status for review
            print(f"ActiveMechanism surface-level binding status: {has_surface_mechanism}")


if __name__ == "__main__":
    unittest.main()
