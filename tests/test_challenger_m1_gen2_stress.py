"""
Adversarial Stress Test Suite for Milestone 1 (Generation 2).
Authored by Challenger 2 Gen 2.

Covers:
1. Cross-package parity: 1,000 rays traced through optics_app.engine vs v2.engine (delta == 0.0).
2. Non-sequential branched tracer parity: 1,000 rays through optics_app vs v2.
3. Schema Version 5 JSON persistence across randomized systems with Galvos, Splitters, Prisms,
   and multi-channel Mechanisms.
4. Investigation of ActiveMechanism persistence and post-reload surface binding.
"""

import json
import os
import tempfile
import unittest
import numpy as np

# optics_app imports
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
from optics_app.engine.tracer import OpticalSystem, BranchedTracer
from optics_app.engine.ray import Ray
from optics_app.engine.mechanisms import get_effective_pose

# v2 imports
from v2.engine.surface import Surface as V2Surface, SurfaceType as V2SurfaceType
from v2.engine.lde import SurfaceStack as V2SurfaceStack
from v2.engine.tracer import OpticalSystem as V2OpticalSystem, BranchedTracer as V2BranchedTracer
from v2.engine.ray import Ray as V2Ray
from v2.engine.components import (
    GalvoScannerConfig as V2GalvoScannerConfig,
    BeamSplitterConfig as V2BeamSplitterConfig,
    PrismConfig as V2PrismConfig,
    OpticalComponent as V2OpticalComponent,
    ActiveMechanism as V2ActiveMechanism,
    MechanismChannel as V2MechanismChannel,
)


class TestCrossPackageParity1000Rays(unittest.TestCase):
    """Stress-test numerical parity between optics_app.engine and v2.engine across 1,000 rays."""

    def test_sequential_tracer_1000_rays_parity(self):
        """Trace 1,000 randomized rays through a multi-element system with galvo mirror and verify delta == 0.0."""
        # Setup optics_app system
        s1 = SurfaceStack()
        s1.add_surface(Surface(name="Input", thickness=20.0, glass="AIR"))
        s1.add_surface(Surface(name="Asphere", radius=45.0, conic=-0.75, thickness=7.5, glass="N-BK7", semi_diameter=20.0))
        s1.add_surface(Surface(name="Asphere_Back", radius=-55.0, thickness=35.0, glass="AIR", semi_diameter=20.0))
        g_cfg1 = GalvoScannerConfig(scan_axis="X", current_angle_deg=3.5, pivot_offset_mm=[0.25, -0.5, 1.2])
        s1.add_surface(Surface(name="Galvo", surface_type=SurfaceType.MIRROR, tilt_x=30.0, tilt_y=5.0, thickness=45.0, galvo_config=g_cfg1))
        s1.add_surface(Surface(name="FoldMirror", surface_type=SurfaceType.MIRROR, tilt_y=45.0, thickness=50.0))
        s1.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, glass="AIR"))
        s1._update_accumulated_positions()
        opt1 = OpticalSystem(s1)

        # Setup v2 system
        s2 = V2SurfaceStack()
        s2.add_surface(V2Surface(name="Input", thickness=20.0, glass="AIR"))
        s2.add_surface(V2Surface(name="Asphere", radius=45.0, conic=-0.75, thickness=7.5, glass="N-BK7", semi_diameter=20.0))
        s2.add_surface(V2Surface(name="Asphere_Back", radius=-55.0, thickness=35.0, glass="AIR", semi_diameter=20.0))
        g_cfg2 = V2GalvoScannerConfig(scan_axis="X", current_angle_deg=3.5, pivot_offset_mm=[0.25, -0.5, 1.2])
        s2.add_surface(V2Surface(name="Galvo", surface_type=V2SurfaceType.MIRROR, tilt_x=30.0, tilt_y=5.0, thickness=45.0, galvo_config=g_cfg2))
        s2.add_surface(V2Surface(name="FoldMirror", surface_type=V2SurfaceType.MIRROR, tilt_y=45.0, thickness=50.0))
        s2.add_surface(V2Surface(name="Det", surface_type=V2SurfaceType.DETECTOR, glass="AIR"))
        s2._update_accumulated_positions()
        opt2 = V2OpticalSystem(s2)

        rng = np.random.default_rng(seed=20261008)
        max_delta_pos = 0.0
        max_delta_dir = 0.0

        for _ in range(1000):
            x = rng.uniform(-3.5, 3.5)
            y = rng.uniform(-3.5, 3.5)
            wl = float(rng.choice([0.488, 0.532, 0.633, 0.780, 1.064]))

            ray1 = Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            ray2 = V2Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)

            out1 = opt1.trace_ray(ray1)
            out2 = opt2.trace_ray(ray2)

            self.assertEqual(out1.vignetted, out2.vignetted)
            d_pos = float(np.max(np.abs(out1.pos - out2.pos)))
            d_dir = float(np.max(np.abs(out1.dir - out2.dir)))
            if d_pos > max_delta_pos:
                max_delta_pos = d_pos
            if d_dir > max_delta_dir:
                max_delta_dir = d_dir

        self.assertEqual(max_delta_pos, 0.0, f"Position delta exceeded 0.0: {max_delta_pos}")
        self.assertEqual(max_delta_dir, 0.0, f"Direction delta exceeded 0.0: {max_delta_dir}")

    def test_branched_tracer_1000_rays_parity(self):
        """Trace 1,000 rays through BranchedTracer in both packages and verify zero delta."""
        s1 = SurfaceStack()
        s1.add_surface(Surface(name="Input", thickness=30.0, glass="AIR"))
        bs1 = BeamSplitterConfig(split_ratio_R=0.35, split_ratio_T=0.65)
        s1.add_surface(Surface(name="BS", tilt_x=45.0, thickness=40.0, is_beam_splitter=True, beamsplitter_config=bs1))
        s1.add_surface(Surface(name="Det_T", surface_type=SurfaceType.DETECTOR, glass="AIR"))
        s1._update_accumulated_positions()
        tracer1 = BranchedTracer(s1)

        s2 = V2SurfaceStack()
        s2.add_surface(V2Surface(name="Input", thickness=30.0, glass="AIR"))
        bs2 = V2BeamSplitterConfig(split_ratio_R=0.35, split_ratio_T=0.65)
        s2.add_surface(V2Surface(name="BS", tilt_x=45.0, thickness=40.0, is_beam_splitter=True, beamsplitter_config=bs2))
        s2.add_surface(V2Surface(name="Det_T", surface_type=V2SurfaceType.DETECTOR, glass="AIR"))
        s2._update_accumulated_positions()
        tracer2 = V2BranchedTracer(s2)

        rng = np.random.default_rng(seed=424242)
        rays1 = [
            Ray(pos=np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.5)
            for _ in range(1000)
        ]
        rng = np.random.default_rng(seed=424242)
        rays2 = [
            V2Ray(pos=np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.5)
            for _ in range(1000)
        ]

        trees1 = tracer1.trace_bundle_tree(rays1)
        trees2 = tracer2.trace_bundle_tree(rays2)

        for idx, (t1, t2) in enumerate(zip(trees1, trees2)):
            self.assertEqual(len(t1.nodes), len(t2.nodes))
            self.assertEqual(len(t1.all_terminal_rays()), len(t2.all_terminal_rays()))
            self.assertEqual(t1.total_energy(), t2.total_energy())


class TestSchemaVersion5PersistenceStress(unittest.TestCase):
    """Stress-test Schema Version 5 JSON persistence across randomized systems."""

    def test_schema_v5_galvo_splitter_prism_roundtrip_equivalence(self):
        """Randomized system with Galvo, Splitter, and Prism configs: verify exact attribute restoration."""
        rng = np.random.default_rng(seed=1337)

        # 1. Randomized Galvo
        galvo_cfg = GalvoScannerConfig(
            scan_axis="X",
            min_angle_deg=float(rng.uniform(-15.0, -5.0)),
            max_angle_deg=float(rng.uniform(5.0, 15.0)),
            step_size_deg=float(rng.uniform(0.2, 1.0)),
            current_angle_deg=float(rng.uniform(-4.0, 4.0)),
            optical_deflection_factor=2.0,
            pivot_offset_mm=[float(rng.uniform(-3.0, 3.0)) for _ in range(3)],
            mirror_thickness_mm=float(rng.uniform(1.5, 4.0)),
            clear_aperture_xy_mm=(float(rng.uniform(10.0, 20.0)), float(rng.uniform(15.0, 25.0))),
            max_voltage_v=10.0,
            volts_to_deg=float(rng.uniform(0.5, 2.0)),
            surface_uid="Surf_Galvo_Rand",
        )
        comp_galvo = OpticalComponent(
            component_uid="comp_galvo_rand",
            name="RandomGalvo",
            component_type="galvo",
            surface_uids=["Surf_Galvo_Rand"],
            galvo_config=galvo_cfg,
        )

        # 2. Randomized BeamSplitter
        r_ratio = float(rng.uniform(0.2, 0.8))
        bs_cfg = BeamSplitterConfig(
            splitter_type=BeamSplitterType.CUBE,
            coating_type=SplitterCoatingType.POLARIZING,
            split_ratio_R=r_ratio,
            split_ratio_T=1.0 - r_ratio,
            substrate_glass="N-SF11",
            substrate_thickness_mm=float(rng.uniform(5.0, 20.0)),
            clear_aperture_mm=float(rng.uniform(15.0, 30.0)),
            dichroic_cutoff_wavelength_um=float(rng.uniform(0.5, 0.8)),
            dichroic_transition_width_um=float(rng.uniform(0.01, 0.05)),
            cube_side_length_mm=float(rng.uniform(15.0, 35.0)),
            orientation_angle_deg=45.0,
            is_polarizing=True,
            Rs=0.90, Rp=0.10, Ts=0.10, Tp=0.90,
        )
        comp_bs = OpticalComponent(
            component_uid="comp_bs_rand",
            name="RandomBS",
            component_type="beamsplitter",
            surface_uids=["Surf_BS_Rand"],
            beamsplitter_config=bs_cfg,
        )

        # 3. Randomized Prism
        prism_cfg = PrismConfig(
            geometry_type=PrismGeometryType.EQUILATERAL,
            apex_angle_deg=float(rng.uniform(55.0, 65.0)),
            base_width_mm=float(rng.uniform(20.0, 40.0)),
            height_mm=float(rng.uniform(20.0, 40.0)),
            length_mm=float(rng.uniform(20.0, 40.0)),
            glass="N-BK7",
            orientation_angle_deg=float(rng.uniform(-15.0, 15.0)),
        )
        comp_prism = OpticalComponent(
            component_uid="comp_prism_rand",
            name="RandomPrism",
            component_type="prism",
            surface_uids=["Surf_Prism_Rand"],
            prism_config=prism_cfg,
        )

        stack = SurfaceStack()
        s0 = Surface(name="Input", thickness=25.0, glass="AIR")
        s_g = Surface(name="Surf_Galvo_Rand", surface_type=SurfaceType.MIRROR, component_uid="comp_galvo_rand", thickness=35.0, galvo_config=galvo_cfg)
        s_b = Surface(name="Surf_BS_Rand", surface_type=SurfaceType.STANDARD, component_uid="comp_bs_rand", thickness=30.0, is_beam_splitter=True, beamsplitter_config=bs_cfg, reflected_branch_start_idx=4)
        s_p = Surface(name="Surf_Prism_Rand", surface_type=SurfaceType.STANDARD, component_uid="comp_prism_rand", thickness=45.0, prism_config=prism_cfg)
        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR")

        for s in [s0, s_g, s_b, s_p, s_det]:
            stack.add_surface(s)
        for c in [comp_galvo, comp_bs, comp_prism]:
            stack.add_component(c)

        sys = SpectrometerSystem(surface_stack=stack, name="StressPersistenceSystem")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stress_persistence.json")
            sys.save_to_json(path)
            loaded = SpectrometerSystem.load_from_json(path)

            self.assertEqual(len(loaded.components), 3)
            self.assertEqual(len(loaded.surfaces), 5)

            # Check Galvo equivalence
            c_g = next(c for c in loaded.components if c.component_type == "galvo")
            self.assertAlmostEqual(c_g.galvo_config.current_angle_deg, galvo_cfg.current_angle_deg, places=7)
            self.assertAlmostEqual(c_g.galvo_config.min_angle_deg, galvo_cfg.min_angle_deg, places=7)
            self.assertAlmostEqual(c_g.galvo_config.max_angle_deg, galvo_cfg.max_angle_deg, places=7)
            np.testing.assert_allclose(c_g.galvo_config.pivot_offset_mm, galvo_cfg.pivot_offset_mm, atol=1e-7)

            # Check Splitter equivalence
            c_b = next(c for c in loaded.components if c.component_type == "beamsplitter")
            self.assertAlmostEqual(c_b.beamsplitter_config.split_ratio_R, bs_cfg.split_ratio_R, places=7)
            self.assertAlmostEqual(c_b.beamsplitter_config.split_ratio_T, bs_cfg.split_ratio_T, places=7)
            self.assertAlmostEqual(c_b.beamsplitter_config.cube_side_length_mm, bs_cfg.cube_side_length_mm, places=7)

            # Check Prism equivalence
            c_p = next(c for c in loaded.components if c.component_type == "prism")
            self.assertAlmostEqual(c_p.prism_config.apex_angle_deg, prism_cfg.apex_angle_deg, places=7)
            self.assertEqual(c_p.prism_config.glass, prism_cfg.glass)


class TestActiveMechanismAdversarialFailure(unittest.TestCase):
    """
    Adversarially probe ActiveMechanism persistence and surface binding.
    Demonstrates the defect where surface.mechanism is NOT bound upon JSON loading.
    """

    def test_active_mechanism_surface_binding_after_json_load(self):
        """
        Verify that an ActiveMechanism bound to a surface preserves its surface-level
        binding and kinematics after JSON round-trip.
        """
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

        # Compute pre-save ray trace
        opt_pre = OpticalSystem(stack)
        ray_pre = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        out_pre = opt_pre.trace_ray(ray_pre)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test_active_mech.json")
            sys = SpectrometerSystem(surface_stack=stack)
            sys.save_to_json(fpath)

            loaded_sys = SpectrometerSystem.load_from_json(fpath)
            loaded_surf = next(s for s in loaded_sys.surfaces if s.name == "Actuated_Mirror")

            # Check whether surface.mechanism was restored
            # DEFECT: SurfaceStack.bind_components_to_surfaces() only binds galvo, bs, prism!
            # It fails to bind mechanism!
            self.assertIsNotNone(
                getattr(loaded_surf, "mechanism", None),
                "DEFECT FOUND: loaded surface.mechanism is None! ActiveMechanism was not bound to the surface after loading."
            )

            # Compute post-save ray trace
            opt_post = OpticalSystem(loaded_sys.surface_stack)
            ray_post = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
            out_post = opt_post.trace_ray(ray_post)

            # Verify ray direction and hit position parity
            dir_diff = np.linalg.norm(out_pre.dir - out_post.dir)
            pos_diff = np.linalg.norm(out_pre.pos - out_post.pos)

            self.assertAlmostEqual(dir_diff, 0.0, places=6, msg=f"Ray direction deviated after JSON reload: {dir_diff}")
            self.assertAlmostEqual(pos_diff, 0.0, places=6, msg=f"Ray position deviated after JSON reload: {pos_diff}")

    def test_extract_components_dict_with_unbound_mechanism_surface(self):
        """
        Verify that extract_components_dict() properly synthesizes an ActiveMechanism component
        from a surface when stack.components is empty.
        """
        stack = SurfaceStack()
        s = Surface(name="Motorized_Mirror", surface_type=SurfaceType.MIRROR, component_uid="mech_surf_0")
        mech = ActiveMechanism(
            mechanism_id="rot_stage",
            mechanism_type="rotation_stage",
            channels={"rot": MechanismChannel(channel_name="rot", target_property="tilt_z", current_value=15.0)}
        )
        s.mechanism = mech
        stack.add_surface(s)

        # When stack.components is empty, extract_components_dict() should extract all surface mechanisms
        extracted = stack.extract_components_dict()
        has_mech = any(c.get("mechanism") is not None or c.get("component_type") == "mechanism" for c in extracted)

        self.assertTrue(
            has_mech,
            "DEFECT FOUND: extract_components_dict() ignores surface.mechanism when stack.components is empty."
        )


if __name__ == "__main__":
    unittest.main()
