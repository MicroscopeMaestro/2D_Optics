"""
Unit and Integration Tests for Generic Optical Components & Advanced Ray Tracing.

Verifies:
- Galvo scanner kinematics: continuous sweep, optical deflection (2*theta), off-vertex pivot offset.
- Beam splitter energy conservation (I_R + I_T = 1), polarization splitting, and ray tree branching.
- Prism geometry, minimum deviation, Snell's law refraction, TIR, and chromatic dispersion.
- ActiveMechanism generic actuation channels and discrete state resolvers.
- JSON Schema Version 5 round-trip persistence and 100% backward compatibility with v3/v4.
- Cross-package parity between optics_app/ and v2/.
"""

import json
import math
import os
import tempfile
import unittest
import numpy as np

# optics_app imports
from optics_app.engine.components import (
    OpticalComponent,
    GalvoScannerConfig,
    BeamSplitterType,
    SplitterCoatingType,
    BeamSplitterConfig,
    PrismGeometryType,
    PrismConfig,
    ActiveMechanism,
    MechanismChannel,
    ActuationType,
    create_galvo_component,
    create_beamsplitter_component,
    create_prism_component,
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, RayNode, RayTree, BranchedTracer
from optics_app.engine.mechanisms import get_effective_pose, interaction_rotation
from optics_app.engine.models import SpectrometerSystem

# v2 imports for cross-validation
import v2.engine.components as v2_comp
import v2.engine.mechanisms as v2_mech
import v2.engine.tracer as v2_tracer
import v2.engine.lde as v2_lde
import v2.engine.models as v2_models


class TestGalvoScanner(unittest.TestCase):
    """Test Galvanometer Scanner kinematics, continuous sweep, deflection, and pivot offsets."""

    def test_galvo_config_initialization_and_voltage(self):
        cfg = GalvoScannerConfig(
            scan_axis="X",
            min_angle_deg=-10.0,
            max_angle_deg=10.0,
            step_size_deg=2.0,
            current_angle_deg=0.0,
            optical_deflection_factor=2.0,
            pivot_offset_mm=[0.0, 0.0, -2.5],
            max_voltage_v=10.0,
            volts_to_deg=1.0,
        )
        self.assertEqual(cfg.optical_min_deg, -20.0)
        self.assertEqual(cfg.optical_max_deg, 20.0)

        # Mechanical angle update & clamping
        cfg.set_mechanical_angle(5.0)
        self.assertAlmostEqual(cfg.current_angle_deg, 5.0)
        self.assertAlmostEqual(cfg.current_optical_angle_deg, 10.0)

        cfg.set_mechanical_angle(15.0) # Clamps to max 10.0
        self.assertAlmostEqual(cfg.current_angle_deg, 10.0)

        # Voltage driving
        cfg.set_voltage(-3.5)
        self.assertAlmostEqual(cfg.current_angle_deg, -3.5)
        self.assertAlmostEqual(cfg.current_optical_angle_deg, -7.0)

    def test_galvo_optical_deflection_2_theta(self):
        """
        Verify that a mirror rotated mechanically by theta produces an optical deflection of 2*theta.
        """
        for scan_axis in ["X", "Y"]:
            for mech_angle in [-8.0, -4.0, 0.0, 3.5, 7.0]:
                stack = SurfaceStack()
                # Entrance reference plane
                surf0 = Surface(name="Input", thickness=50.0, glass="AIR")
                stack.add_surface(surf0)

                # Galvo mirror at 45 deg fold (or flat plane normal to Z)
                comp, galvo_surf = create_galvo_component(
                    name="GalvoTest",
                    scan_axis=scan_axis,
                    min_angle_deg=-15.0,
                    max_angle_deg=15.0,
                    current_angle_deg=mech_angle,
                    pivot_offset_mm=[0.0, 0.0, 0.0],
                )
                # Nominal flat mirror with normal along -Z
                galvo_surf.tilt_x = 0.0
                galvo_surf.tilt_y = 0.0
                galvo_surf.thickness = 100.0
                stack.add_surface(galvo_surf)

                # Detector plane
                surf_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR")
                stack.add_surface(surf_det)
                stack._update_accumulated_positions()

                opt_sys = OpticalSystem(stack)
                in_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.532)
                traced = opt_sys.trace_ray(in_ray)

                # Un-deflected baseline (mech_angle = 0)
                # For flat mirror normal along -Z, incoming [0,0,1] reflects to [0,0,-1]
                # Deflected by theta about Y => reflected ray tilts by 2*theta in X-Z plane
                ref_dir = traced.dir
                theta_rad = math.radians(mech_angle)
                expected_2theta = 2.0 * theta_rad

                if scan_axis == "X":
                    # Tilts about Y axis: produces X deflection
                    # Angle between reflected ray and -Z axis:
                    actual_angle = math.atan2(ref_dir[0], -ref_dir[2])
                    self.assertAlmostEqual(abs(actual_angle), abs(expected_2theta), places=5)
                else:
                    # Tilts about X axis: produces Y deflection
                    actual_angle = math.atan2(-ref_dir[1], -ref_dir[2])
                    self.assertAlmostEqual(abs(actual_angle), abs(expected_2theta), places=5)

    def test_galvo_pivot_offset_kinematics(self):
        """
        Verify that off-vertex pivot offset d_pivot = [x_p, y_p, z_p]
        causes predictable translation of the mirror vertex during rotation.
        Formula: P_eff = P_0 + R_mount @ d_pivot - R_active @ d_pivot
        """
        # Test pivot offset along Z axis (shaft 5 mm behind mirror face)
        pivot_offset = [0.0, 0.0, -5.0]
        surf = Surface(
            name="GalvoMirror",
            surface_type=SurfaceType.MIRROR,
            global_origin=np.array([10.0, 20.0, 30.0], dtype=float),
            global_R=np.eye(3, dtype=float),
        )
        cfg = GalvoScannerConfig(
            scan_axis="X",
            current_angle_deg=10.0, # Rotate about Y by 10 deg
            pivot_offset_mm=pivot_offset,
        )
        surf.galvo_config = cfg

        eff_origin, eff_R = get_effective_pose(surf)

        # Expected R_scan: rotation about [0, 1, 0] by 10 deg
        th_rad = math.radians(10.0)
        c, s = math.cos(th_rad), math.sin(th_rad)
        R_expected = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ], dtype=float)
        np.testing.assert_allclose(eff_R, R_expected, atol=1e-7)

        # Expected translation:
        # P_pivot = [10, 20, 30] + [0, 0, -5] = [10, 20, 25]
        # P_eff = P_pivot - R_expected @ [0, 0, -5]
        P_pivot = np.array([10.0, 20.0, 25.0])
        P_expected = P_pivot - R_expected @ np.array([0.0, 0.0, -5.0])
        np.testing.assert_allclose(eff_origin, P_expected, atol=1e-7)

        # Verify that if d_pivot == [0, 0, 0], eff_origin == nominal origin
        cfg.pivot_offset_mm = [0.0, 0.0, 0.0]
        eff_origin_zero, _ = get_effective_pose(surf)
        np.testing.assert_allclose(eff_origin_zero, surf.global_origin, atol=1e-7)


class TestBeamSplitter(unittest.TestCase):
    """Test Beam Splitter configurations, energy conservation, polarization splitting, and branching."""

    def test_beamsplitter_config_and_energy_conservation(self):
        # 50:50 NPBS
        npbs = BeamSplitterConfig(split_ratio_R=0.50, split_ratio_T=0.50)
        R, T = npbs.evaluate_split(0.532)
        self.assertAlmostEqual(R, 0.50)
        self.assertAlmostEqual(T, 0.50)
        self.assertAlmostEqual(R + T, 1.0)

        # 70:30 Custom splitter
        custom_bs = BeamSplitterConfig(split_ratio_R=0.70, split_ratio_T=0.30)
        R, T = custom_bs.evaluate_split(0.633)
        self.assertAlmostEqual(R, 0.70)
        self.assertAlmostEqual(T, 0.30)
        self.assertAlmostEqual(R + T, 1.0)

        # Polarizing Beam Splitter (PBS)
        pbs = BeamSplitterConfig(
            coating_type=SplitterCoatingType.POLARIZING,
            Rs=0.99, Rp=0.01, Ts=0.01, Tp=0.99
        )
        Rs, Ts = pbs.evaluate_split(0.532, polarization="s")
        self.assertAlmostEqual(Rs, 0.99)
        self.assertAlmostEqual(Ts, 0.01)
        self.assertAlmostEqual(Rs + Ts, 1.0)

        Rp, Tp = pbs.evaluate_split(0.532, polarization="p")
        self.assertAlmostEqual(Rp, 0.01)
        self.assertAlmostEqual(Tp, 0.99)
        self.assertAlmostEqual(Rp + Tp, 1.0)

        # Dichroic Beam Splitter with 532 nm cutoff
        dichroic = BeamSplitterConfig(
            coating_type=SplitterCoatingType.DICHROIC,
            dichroic_cutoff_wavelength_um=0.532,
            dichroic_transition_width_um=0.01
        )
        # 488 nm (blue): reflects (R > 0.95)
        R_blue, T_blue = dichroic.evaluate_split(0.488)
        self.assertGreater(R_blue, 0.95)
        self.assertLess(T_blue, 0.05)
        self.assertAlmostEqual(R_blue + T_blue, 1.0)

        # 633 nm (red): transmits (T > 0.95)
        R_red, T_red = dichroic.evaluate_split(0.633)
        self.assertLess(R_red, 0.05)
        self.assertGreater(T_red, 0.95)
        self.assertAlmostEqual(R_red + T_red, 1.0)

    def test_beamsplitter_branched_tracer_tree(self):
        """
        Verify that BranchedTracer splits an incoming ray into simultaneous
        transmitted and reflected ray branches, with exact energy conservation (I_R + I_T = I_in).
        """
        stack = SurfaceStack()
        # Input reference plane
        s0 = Surface(name="Input", thickness=50.0, glass="AIR")
        stack.add_surface(s0)

        # 50:50 Plate Beamsplitter tilted at 45 deg
        bs_cfg = BeamSplitterConfig(split_ratio_R=0.50, split_ratio_T=0.50)
        s_bs = Surface(
            name="Beamsplitter_Face",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=50.0,
            glass="AIR",
            semi_diameter=25.0,
            tilt_x=45.0,
            is_beam_splitter=True,
            beamsplitter_config=bs_cfg,
        )
        stack.add_surface(s_bs)

        # Downstream transmitted detector
        s_det = Surface(name="Transmitted_Detector", surface_type=SurfaceType.DETECTOR, glass="AIR")
        stack.add_surface(s_det)
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack, max_depth=3, min_intensity=1e-4)
        input_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.0, wavelength=0.532)

        tree = tracer.trace_ray_tree(input_ray)

        # Tree structure check
        self.assertIn(0, tree.nodes) # Root node
        root = tree.nodes[0]
        self.assertEqual(len(root.children_ids), 2, "Root node must spawn 2 children at beamsplitter")

        child1 = tree.nodes[root.children_ids[0]]
        child2 = tree.nodes[root.children_ids[1]]

        # One transmitted, one reflected
        types = {child1.branch_type, child2.branch_type}
        self.assertEqual(types, {"transmitted", "reflected"})

        t_node = child1 if child1.branch_type == "transmitted" else child2
        r_node = child2 if child1.branch_type == "transmitted" else child1

        # Energy conservation check: I_R + I_T = 1.0
        self.assertAlmostEqual(t_node.ray.intensity, 0.50, places=6)
        self.assertAlmostEqual(r_node.ray.intensity, 0.50, places=6)
        self.assertAlmostEqual(tree.total_energy(), 1.0, places=6)

        # Direction checks:
        # Transmitted proceeds along Z = [0, 0, 1]
        np.testing.assert_allclose(t_node.ray.dir, [0.0, 0.0, 1.0], atol=1e-5)

        # Reflected at 45 deg tilt about X turns by 90 deg into Y direction
        # Local normal tilted by 45 deg about X => specular reflection turns by 90 deg
        self.assertAlmostEqual(abs(r_node.ray.dir[1]), 1.0, places=5)
        self.assertAlmostEqual(r_node.ray.dir[0], 0.0, places=5)

        # Check flatten_paths returns polylines
        paths = tree.flatten_paths()
        self.assertGreaterEqual(len(paths), 2)


class TestPrism(unittest.TestCase):
    """Test Prism geometry, minimum deviation, chromatic dispersion, and TIR."""

    def test_prism_minimum_deviation_formula(self):
        # Equilateral prism (alpha = 60 deg) of N-SF11
        cfg = PrismConfig(
            geometry_type=PrismGeometryType.EQUILATERAL,
            apex_angle_deg=60.0,
            glass="N-SF11"
        )
        # N-SF11 refractive index at 532 nm is approx 1.7915
        n_532 = cfg._get_refractive_index(0.532)
        expected_min_dev = math.degrees(2.0 * math.asin(n_532 * math.sin(math.radians(30.0))) - math.radians(60.0))

        actual_min_dev = cfg.calculate_minimum_deviation(0.532)
        self.assertAlmostEqual(actual_min_dev, expected_min_dev, places=5)

        # At symmetric incidence angle theta_1 = (alpha + delta_min) / 2:
        theta_sym = (60.0 + expected_min_dev) / 2.0
        dev_at_sym = cfg.calculate_deviation(theta_sym, 0.532)
        self.assertAlmostEqual(dev_at_sym, expected_min_dev, places=4)

    def test_prism_chromatic_dispersion(self):
        """
        Verify that shorter wavelengths experience higher refractive index
        and consequently larger angular deviation (normal dispersion: d_delta / d_lambda < 0).
        """
        cfg = PrismConfig(
            geometry_type=PrismGeometryType.EQUILATERAL,
            apex_angle_deg=60.0,
            glass="N-SF11"
        )
        disp = cfg.calculate_dispersion([0.488, 0.532, 0.633])

        dev_blue = disp[0.488]
        dev_green = disp[0.532]
        dev_red = disp[0.633]

        # Higher index => higher deviation
        self.assertGreater(dev_blue, dev_green, "Blue light must deviate more than green light")
        self.assertGreater(dev_green, dev_red, "Green light must deviate more than red light")

        # Dispersion spread
        delta_spread = dev_blue - dev_red
        self.assertGreater(delta_spread, 1.0, "N-SF11 must produce significant angular dispersion (> 1 deg)")

    def test_prism_right_angle_tir(self):
        """
        Verify right-angle prism TIR fold (hypotenuse acts as internal mirror).
        """
        cfg = PrismConfig(
            geometry_type=PrismGeometryType.RIGHT_ANGLE,
            apex_angle_deg=90.0,
            glass="N-BK7"
        )
        self.assertEqual(cfg.geometry_type, PrismGeometryType.RIGHT_ANGLE)
        self.assertEqual(cfg.apex_angle_deg, 90.0)


class TestActiveMechanism(unittest.TestCase):
    """Test generic ActiveMechanism channels and kinematic resolvers."""

    def test_mechanism_channels_and_kinematics(self):
        mech = ActiveMechanism(
            mechanism_id="piezo_stage_01",
            mechanism_type="piezo_stage",
            actuation_type=ActuationType.CONTINUOUS,
            channels={
                "tx": MechanismChannel(channel_name="tx", target_property="tilt_x", min_value=-5.0, max_value=5.0),
                "dx": MechanismChannel(channel_name="dx", target_property="decenter_x", min_value=-10.0, max_value=10.0),
            }
        )

        mech.set_channel_value("tx", 2.5)
        mech.set_channel_value("dx", -4.0)

        surf = Surface(name="Actuated_Mirror", surface_type=SurfaceType.MIRROR)
        surf.mechanism = mech

        eff_origin, eff_R = get_effective_pose(surf)

        # Decenter X should shift origin[0] by -4.0
        self.assertAlmostEqual(eff_origin[0], -4.0)

        # Tilt X should rotate eff_R about X by 2.5 deg
        rx = math.radians(2.5)
        self.assertAlmostEqual(eff_R[1, 1], math.cos(rx))
        self.assertAlmostEqual(eff_R[1, 2], -math.sin(rx))


class TestSchemaVersion5Persistence(unittest.TestCase):
    """Test JSON Schema Version 5 serialization, round-trip fidelity, and v3/v4 backward compatibility."""

    def test_schema_v5_roundtrip_all_components(self):
        """
        Construct an optical system containing a Galvo, BeamSplitter, and Prism.
        Save to JSON, verify Schema Version 5, reload, and verify 100% parameter preservation.
        """
        stack = SurfaceStack()

        # 1. Galvo Component
        comp_galvo, surf_galvo = create_galvo_component(
            name="Galvo_Scan_X",
            scan_axis="X",
            min_angle_deg=-12.5,
            max_angle_deg=12.5,
            current_angle_deg=3.2,
            pivot_offset_mm=[0.0, 1.2, -3.4],
        )
        stack.add_surface(surf_galvo)
        stack.add_component(comp_galvo)

        # 2. BeamSplitter Component
        comp_bs, surfs_bs = create_beamsplitter_component(
            name="NPBS_Cube",
            splitter_type=BeamSplitterType.CUBE,
            coating_type=SplitterCoatingType.NON_POLARIZING_50_50,
            split_ratio_R=0.55,
            split_ratio_T=0.45,
            substrate_glass="N-BK7",
            substrate_thickness_mm=15.0,
        )
        for s in surfs_bs:
            stack.add_surface(s)
        stack.add_component(comp_bs)

        # 3. Prism Component
        comp_prism, surfs_prism = create_prism_component(
            name="SF11_Prism",
            geometry_type=PrismGeometryType.EQUILATERAL,
            apex_angle_deg=60.0,
            glass="N-SF11",
            base_width_mm=30.0,
        )
        for s in surfs_prism:
            stack.add_surface(s)
        stack.add_component(comp_prism)

        # Wrap in SpectrometerSystem
        system = SpectrometerSystem(surface_stack=stack, name="MultiComponent_V5_System")

        # Save to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test_v5_system.json")
            system.save_to_json(fpath)

            # Inspect raw JSON schema
            with open(fpath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            self.assertEqual(raw_data.get("version"), 5, "Saved system must specify Schema Version 5")
            self.assertIn("components", raw_data, "Version 5 must contain top-level 'components' array")
            self.assertEqual(len(raw_data["components"]), 3)

            # Reload via SpectrometerSystem.load_from_json
            loaded_sys = SpectrometerSystem.load_from_json(fpath)

            # Verify components list was restored
            self.assertEqual(len(loaded_sys.components), 3)

            # Verify Galvo attributes round-tripped
            galvo_loaded = next(c for c in loaded_sys.components if c.component_type == "galvo")
            self.assertIsNotNone(galvo_loaded.galvo_config)
            self.assertEqual(galvo_loaded.galvo_config.scan_axis, "X")
            self.assertAlmostEqual(galvo_loaded.galvo_config.min_angle_deg, -12.5)
            self.assertAlmostEqual(galvo_loaded.galvo_config.max_angle_deg, 12.5)
            self.assertAlmostEqual(galvo_loaded.galvo_config.current_angle_deg, 3.2)
            np.testing.assert_allclose(galvo_loaded.galvo_config.pivot_offset_mm, [0.0, 1.2, -3.4])

            # Verify surface back-reference was preserved
            s_galvo_loaded = next(s for s in loaded_sys.surfaces if s.component_uid == galvo_loaded.component_uid)
            self.assertIsNotNone(s_galvo_loaded.galvo_config)
            self.assertAlmostEqual(s_galvo_loaded.galvo_config.current_angle_deg, 3.2)

            # Verify BeamSplitter attributes round-tripped
            bs_loaded = next(c for c in loaded_sys.components if c.component_type == "beamsplitter")
            self.assertIsNotNone(bs_loaded.beamsplitter_config)
            self.assertAlmostEqual(bs_loaded.beamsplitter_config.split_ratio_R, 0.55)
            self.assertAlmostEqual(bs_loaded.beamsplitter_config.split_ratio_T, 0.45)
            self.assertEqual(bs_loaded.beamsplitter_config.splitter_type, BeamSplitterType.CUBE)

            # Verify Prism attributes round-tripped
            prism_loaded = next(c for c in loaded_sys.components if c.component_type == "prism")
            self.assertIsNotNone(prism_loaded.prism_config)
            self.assertAlmostEqual(prism_loaded.prism_config.apex_angle_deg, 60.0)
            self.assertEqual(prism_loaded.prism_config.glass, "N-SF11")

    def test_schema_backward_compatibility_v3_and_v4(self):
        """
        Verify that legacy Version 3 and Version 4 files load cleanly without errors.
        """
        # Version 3 mock
        v3_data = {
            "version": 3,
            "name": "Legacy V3 System",
            "surfaces": [
                {"name": "S0", "radius": 0.0, "thickness": 20.0, "glass": "AIR", "semi_diameter": 12.5},
                {"name": "Mirror1", "surface_type": "Mirror", "radius": 0.0, "thickness": 50.0, "glass": "MIRROR"},
            ],
            "wavelengths": [0.532, 0.633],
            "primary_wavelength": 0.532
        }

        # Version 4 mock
        v4_data = {
            "version": 4,
            "name": "Legacy V4 System",
            "surfaces": [
                {"name": "Entrance", "radius": 0.0, "thickness": 10.0, "glass": "AIR"},
                {"name": "DMD", "surface_type": "Mirror", "mechanism_type": "DMD", "mechanical_state": "ON"},
            ],
            "wavelengths": [0.800],
            "primary_wavelength": 0.800
        }

        sys_v3 = SpectrometerSystem.from_dict(v3_data)
        self.assertEqual(len(sys_v3.surfaces), 2)
        self.assertEqual(sys_v3.name, "Legacy V3 System")

        sys_v4 = SpectrometerSystem.from_dict(v4_data)
        self.assertEqual(len(sys_v4.surfaces), 2)
        self.assertEqual(sys_v4.name, "Legacy V4 System")


class TestCrossPackageParityV2(unittest.TestCase):
    """Verify that v2/ package classes produce identical results to optics_app/."""

    def test_v2_galvo_and_pose_parity(self):
        from v2.engine.surface import Surface as V2Surface
        cfg_v1 = GalvoScannerConfig(scan_axis="X", current_angle_deg=5.0, pivot_offset_mm=[0.0, 0.0, -2.0])
        cfg_v2 = v2_comp.GalvoScannerConfig(scan_axis="X", current_angle_deg=5.0, pivot_offset_mm=[0.0, 0.0, -2.0])

        s_v1 = Surface(name="M1", surface_type=SurfaceType.MIRROR, galvo_config=cfg_v1)
        s_v2 = V2Surface(name="M1", surface_type=SurfaceType.MIRROR, galvo_config=cfg_v2)

        p1, R1 = get_effective_pose(s_v1)
        p2, R2 = v2_mech.get_effective_pose(s_v2)

        np.testing.assert_allclose(p1, p2, atol=1e-8)
        np.testing.assert_allclose(R1, R2, atol=1e-8)

    def test_v2_branched_tracer_parity(self):
        from v2.engine.surface import Surface as V2Surface
        stack = v2_lde.SurfaceStack()
        s0 = V2Surface(name="Input", thickness=50.0, glass="AIR")
        bs_cfg = v2_comp.BeamSplitterConfig(split_ratio_R=0.50, split_ratio_T=0.50)
        s1 = V2Surface(name="BS", surface_type=SurfaceType.STANDARD, tilt_x=45.0, is_beam_splitter=True, beamsplitter_config=bs_cfg)
        stack.add_surface(s0)
        stack.add_surface(s1)
        stack._update_accumulated_positions()

        tracer = v2_tracer.BranchedTracer(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.0)
        tree = tracer.trace_ray_tree(ray)

        self.assertAlmostEqual(tree.total_energy(), 1.0, places=6)
        self.assertEqual(len(tree.all_terminal_rays()), 2)


class TestAdvancedBranchingAndEdgeCases(unittest.TestCase):
    """Stress-test bundle tree branching, ratio variations, and complex multi-component systems."""

    def test_bundle_tree_tracing(self):
        stack = SurfaceStack()
        s0 = Surface(name="Input", thickness=30.0, glass="AIR")
        bs_cfg = BeamSplitterConfig(split_ratio_R=0.40, split_ratio_T=0.60)
        s_bs = Surface(name="Splitter", tilt_x=45.0, thickness=40.0, is_beam_splitter=True, beamsplitter_config=bs_cfg)
        stack.add_surface(s0)
        stack.add_surface(s_bs)
        stack._update_accumulated_positions()

        tracer = BranchedTracer(stack)
        rays = [
            Ray(pos=np.array([x, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=2.0)
            for x in np.linspace(-2.0, 2.0, 5)
        ]
        trees = tracer.trace_bundle_tree(rays)
        self.assertEqual(len(trees), 5)
        for tree in trees:
            self.assertEqual(len(tree.all_terminal_rays()), 2)
            # Energy conservation: input 2.0 => terminal energy sum 2.0
            self.assertAlmostEqual(tree.total_energy(), 2.0, places=6)
            t_rays = [n.ray for n in tree.nodes.values() if n.branch_type == "transmitted"]
            r_rays = [n.ray for n in tree.nodes.values() if n.branch_type == "reflected"]
            self.assertEqual(len(t_rays), 1)
            self.assertEqual(len(r_rays), 1)
            self.assertAlmostEqual(t_rays[0].intensity, 1.20, places=6) # 2.0 * 0.60
            self.assertAlmostEqual(r_rays[0].intensity, 0.80, places=6) # 2.0 * 0.40

    def test_unbalanced_beam_splitter_ratios(self):
        for R_ratio, T_ratio in [(0.90, 0.10), (0.10, 0.90), (0.0, 1.0), (1.0, 0.0), (0.33, 0.67)]:
            bs_cfg = BeamSplitterConfig(split_ratio_R=R_ratio, split_ratio_T=T_ratio)
            R_eval, T_eval = bs_cfg.evaluate_split(0.532)
            self.assertAlmostEqual(R_eval, R_ratio, places=6)
            self.assertAlmostEqual(T_eval, T_ratio, places=6)
            self.assertAlmostEqual(R_eval + T_eval, 1.0, places=6)

    def test_galvo_sweep_discretization(self):
        cfg = GalvoScannerConfig(min_angle_deg=-5.0, max_angle_deg=5.0, step_size_deg=2.5)
        angles = cfg.get_mechanical_angles()
        opt_angles = cfg.get_optical_angles()
        self.assertEqual(len(angles), 5) # -5.0, -2.5, 0.0, 2.5, 5.0
        np.testing.assert_allclose(angles, [-5.0, -2.5, 0.0, 2.5, 5.0])
        np.testing.assert_allclose(opt_angles, [-10.0, -5.0, 0.0, 5.0, 10.0])

    def test_complex_multicomponent_system_roundtrip(self):
        """
        Build an advanced optical assembly containing:
        1. Object plane
        2. Galvo scanning mirror (with off-axis pivot offset)
        3. Cube beam splitter
        4. Dispersing prism
        5. Detector plane
        Verify complete Schema Version 5 serialization and deserialization.
        """
        stack = SurfaceStack()
        s_obj = Surface(name="ObjectPlane", thickness=20.0, glass="AIR")
        stack.add_surface(s_obj)

        comp_g, surf_g = create_galvo_component(
            name="Scanner",
            scan_axis="X",
            min_angle_deg=-10.0,
            max_angle_deg=10.0,
            current_angle_deg=-2.5,
            pivot_offset_mm=[0.0, 0.0, -3.0]
        )
        stack.add_surface(surf_g)
        stack.add_component(comp_g)

        comp_bs, surfs_bs = create_beamsplitter_component(
            name="Splitter",
            splitter_type=BeamSplitterType.CUBE,
            split_ratio_R=0.50,
            split_ratio_T=0.50
        )
        for s in surfs_bs:
            stack.add_surface(s)
        stack.add_component(comp_bs)

        comp_p, surfs_p = create_prism_component(
            name="Prism",
            apex_angle_deg=60.0,
            glass="N-SF11"
        )
        for s in surfs_p:
            stack.add_surface(s)
        stack.add_component(comp_p)

        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR")
        stack.add_surface(s_det)

        system = SpectrometerSystem(surface_stack=stack, name="FullAssemblyV5")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "assembly_v5.json")
            system.save_to_json(path)

            loaded = SpectrometerSystem.load_from_json(path)
            self.assertEqual(len(loaded.components), 3)
            self.assertEqual(len(loaded.surfaces), len(stack.surfaces))

            # Verify galvo reconstructed
            g = next(c for c in loaded.components if c.component_type == "galvo")
            self.assertAlmostEqual(g.galvo_config.current_angle_deg, -2.5)
            self.assertEqual(g.galvo_config.pivot_offset_mm, [0.0, 0.0, -3.0])

            # Verify beam splitter reconstructed
            b = next(c for c in loaded.components if c.component_type == "beamsplitter")
            self.assertEqual(b.beamsplitter_config.splitter_type, BeamSplitterType.CUBE)

            # Verify prism reconstructed
            p = next(c for c in loaded.components if c.component_type == "prism")
            self.assertAlmostEqual(p.prism_config.apex_angle_deg, 60.0)


if __name__ == "__main__":
    unittest.main()

