"""
Milestone 1 Generation 3 Gate Verification Test Suite.
Authored by Challenger 1 Gen 3.

Empirical verification of ActiveMechanism remediation:
1. Ray tracing deflection consistency after JSON save and reload across multi-axis tilts and decenters.
2. Deflection consistency for standalone surface mechanisms (unregistered in stack.components).
3. Post-reload dynamic actuation sweeps: verify that modifying mechanism channels on reloaded systems
   yields identical ray trajectories as the reference pre-save system.
4. Multi-mechanism BranchedTracer tree consistency across beam splitters and actuated mirrors.
5. Cross-engine parity between optics_app.engine and v2.engine on reloaded systems.
"""

import os
import tempfile
import unittest
import numpy as np

# optics_app imports
from optics_app.engine.components import (
    OpticalComponent,
    ActiveMechanism,
    MechanismChannel,
    ActuationType,
    GalvoScannerConfig,
    BeamSplitterConfig,
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.models import SpectrometerSystem
from optics_app.engine.tracer import OpticalSystem, BranchedTracer, Ray
from optics_app.engine.mechanisms import get_effective_pose

# v2 imports
import v2.engine.components as v2_components
import v2.engine.surface as v2_surface
import v2.engine.lde as v2_lde
import v2.engine.models as v2_models
import v2.engine.tracer as v2_tracer


class TestActiveMechanismDeflectionConsistency(unittest.TestCase):
    """Rigorous empirical validation of ray tracing deflection consistency after JSON round-trip."""

    def test_active_mechanism_full_pose_deflection_consistency_roundtrip(self):
        """
        Verify that an optical system with an actuated mirror (3D tilts + decenters)
        exhibits 0.0 ray deflection deviation after JSON save and reload across 100 rays.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=25.0, glass="AIR"))
        stack.add_surface(Surface(name="Singlet_F", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="Singlet_B", radius=-100.0, thickness=40.0, glass="AIR", semi_diameter=15.0))

        # Actuated Mirror with multiple active channels (tilt_x, tilt_y, decenter_x, decenter_y)
        mech = ActiveMechanism(
            mechanism_id="piezo_tip_tilt_01",
            mechanism_type="tip_tilt_platform",
            actuation_type=ActuationType.CONTINUOUS,
            channels={
                "pitch": MechanismChannel(
                    channel_name="pitch",
                    target_property="tilt_x",
                    unit="deg",
                    min_value=-5.0,
                    max_value=5.0,
                    current_value=3.25,
                    scale_factor=1.0,
                ),
                "yaw": MechanismChannel(
                    channel_name="yaw",
                    target_property="tilt_y",
                    unit="deg",
                    min_value=-5.0,
                    max_value=5.0,
                    current_value=-1.75,
                    scale_factor=1.0,
                ),
                "shift_x": MechanismChannel(
                    channel_name="shift_x",
                    target_property="decenter_x",
                    unit="mm",
                    min_value=-2.0,
                    max_value=2.0,
                    current_value=0.85,
                    scale_factor=1.0,
                ),
                "shift_y": MechanismChannel(
                    channel_name="shift_y",
                    target_property="decenter_y",
                    unit="mm",
                    min_value=-2.0,
                    max_value=2.0,
                    current_value=-0.45,
                    scale_factor=1.0,
                ),
            }
        )
        comp = OpticalComponent(
            component_uid="comp_act_tip_tilt",
            name="TipTiltStage",
            component_type="ActiveMechanism",
            surface_uids=["Actuated_Mirror"],
            mechanism=mech,
        )

        s_act = Surface(
            name="Actuated_Mirror",
            surface_type=SurfaceType.MIRROR,
            thickness=50.0,
            tilt_x=45.0,
            semi_diameter=25.0,
            component_uid="comp_act_tip_tilt",
        )
        s_act.mechanism = mech

        s_fold = Surface(
            name="Fold_Mirror",
            surface_type=SurfaceType.MIRROR,
            thickness=60.0,
            tilt_y=-30.0,
            semi_diameter=30.0,
        )
        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR", semi_diameter=100.0)

        stack.add_surface(s_act)
        stack.add_surface(s_fold)
        stack.add_surface(s_det)
        stack.add_component(comp)
        stack._update_accumulated_positions()

        sys_orig = SpectrometerSystem(surface_stack=stack, name="ActiveTipTiltSys")
        opt_pre = OpticalSystem(stack)

        # Generate 100 randomized rays across pupil and spectrum
        rng = np.random.default_rng(seed=20261009)
        test_rays = []
        for _ in range(100):
            rx = float(rng.uniform(-4.0, 4.0))
            ry = float(rng.uniform(-4.0, 4.0))
            dx = float(rng.uniform(-0.02, 0.02))
            dy = float(rng.uniform(-0.02, 0.02))
            dz = np.sqrt(max(0.0, 1.0 - dx*dx - dy*dy))
            wl = float(rng.choice([0.488, 0.532, 0.6328, 0.780, 0.850]))
            test_rays.append(Ray(pos=np.array([rx, ry, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wl, intensity=1.0))

        # Pre-save ray trace results
        pre_results = [opt_pre.trace_ray(r) for r in test_rays]

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "active_mechanism_system.json")
            sys_orig.save_to_json(json_path)

            loaded_sys = SpectrometerSystem.load_from_json(json_path)
            opt_post = OpticalSystem(loaded_sys.surface_stack)

            # Post-reload ray trace results
            post_results = [opt_post.trace_ray(r) for r in test_rays]

            max_pos_err = 0.0
            max_dir_err = 0.0
            max_opl_err = 0.0

            for idx, (r_pre, r_post) in enumerate(zip(pre_results, post_results)):
                self.assertEqual(r_pre.vignetted, r_post.vignetted, f"Ray {idx} vignetting state mismatch")
                if not r_pre.vignetted:
                    pos_err = float(np.linalg.norm(r_pre.pos - r_post.pos))
                    dir_err = float(np.linalg.norm(r_pre.dir - r_post.dir))
                    opl_err = float(abs(r_pre.path_length - r_post.path_length))

                    max_pos_err = max(max_pos_err, pos_err)
                    max_dir_err = max(max_dir_err, dir_err)
                    max_opl_err = max(max_opl_err, opl_err)

            self.assertLessEqual(max_pos_err, 1e-12, f"Position error exceeded tolerance: {max_pos_err} mm")
            self.assertLessEqual(max_dir_err, 1e-12, f"Direction error exceeded tolerance: {max_dir_err}")
            self.assertLessEqual(max_opl_err, 1e-12, f"Path length error exceeded tolerance: {max_opl_err} mm")

    def test_standalone_surface_mechanism_deflection_consistency(self):
        """
        Verify that a standalone surface.mechanism (without explicit stack.components entry)
        is synthesized during serialization, restored on reload, and maintains 0.0 deflection deviation.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Input", thickness=30.0, glass="AIR"))

        s_act = Surface(
            name="Standalone_Motorized_Mirror",
            surface_type=SurfaceType.MIRROR,
            thickness=50.0,
            tilt_x=45.0,
            semi_diameter=25.0,
        )
        mech = ActiveMechanism(
            mechanism_id="motor_rot_z",
            mechanism_type="rotator",
            channels={
                "rot": MechanismChannel(
                    channel_name="rot",
                    target_property="tilt_z",
                    unit="deg",
                    current_value=12.5,
                )
            }
        )
        s_act.mechanism = mech
        stack.add_surface(s_act)
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR", semi_diameter=100.0))
        stack._update_accumulated_positions()

        # Notice stack.components was NOT populated manually
        self.assertEqual(len(stack.components), 0)

        sys_orig = SpectrometerSystem(surface_stack=stack)
        opt_pre = OpticalSystem(stack)
        ray_in = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.532)
        out_pre = opt_pre.trace_ray(ray_in)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "standalone_mech.json")
            sys_orig.save_to_json(fpath)

            loaded_sys = SpectrometerSystem.load_from_json(fpath)
            loaded_surf = next(s for s in loaded_sys.surfaces if s.name == "Standalone_Motorized_Mirror")

            # Must have mechanism bound
            self.assertIsNotNone(getattr(loaded_surf, "mechanism", None))
            self.assertEqual(loaded_surf.component_type, "ActiveMechanism")

            opt_post = OpticalSystem(loaded_sys.surface_stack)
            out_post = opt_post.trace_ray(ray_in)

            pos_diff = np.linalg.norm(out_pre.pos - out_post.pos)
            dir_diff = np.linalg.norm(out_pre.dir - out_post.dir)

            self.assertEqual(pos_diff, 0.0)
            self.assertEqual(dir_diff, 0.0)

    def test_post_reload_dynamic_channel_actuation_sweep(self):
        """
        Verify that changing mechanism channel values on a reloaded system produces
        exact identical ray positions and directions as doing so on the original system.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Input", thickness=40.0, glass="AIR"))

        mech = ActiveMechanism(
            mechanism_id="scanner_x",
            mechanism_type="stepper",
            channels={
                "tilt": MechanismChannel(
                    channel_name="tilt",
                    target_property="tilt_x",
                    current_value=0.0,
                    min_value=-15.0,
                    max_value=15.0,
                )
            }
        )
        s_act = Surface(
            name="Scannable_Mirror",
            surface_type=SurfaceType.MIRROR,
            thickness=60.0,
            tilt_x=30.0,
            semi_diameter=30.0,
            component_uid="comp_scan",
        )
        s_act.mechanism = mech

        stack.add_surface(s_act)
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, glass="AIR", semi_diameter=150.0))
        stack.add_component(OpticalComponent(
            component_uid="comp_scan",
            name="ScanComp",
            component_type="ActiveMechanism",
            surface_uids=["Scannable_Mirror"],
            mechanism=mech
        ))
        stack._update_accumulated_positions()

        sys_orig = SpectrometerSystem(surface_stack=stack)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "sweep_test.json")
            sys_orig.save_to_json(fpath)
            loaded_sys = SpectrometerSystem.load_from_json(fpath)

            orig_act_surf = next(s for s in sys_orig.surfaces if s.name == "Scannable_Mirror")
            loaded_act_surf = next(s for s in loaded_sys.surfaces if s.name == "Scannable_Mirror")

            ray = Ray(pos=np.array([0.0, 1.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))

            for test_val in [-12.0, -5.5, 0.0, 4.25, 11.8]:
                # Update channel on both systems
                orig_act_surf.mechanism.channels["tilt"].current_value = test_val
                loaded_act_surf.mechanism.channels["tilt"].current_value = test_val

                sys_orig.surface_stack._update_accumulated_positions()
                loaded_sys.surface_stack._update_accumulated_positions()

                out_orig = OpticalSystem(sys_orig.surface_stack).trace_ray(ray)
                out_loaded = OpticalSystem(loaded_sys.surface_stack).trace_ray(ray)

                pos_delta = np.linalg.norm(out_orig.pos - out_loaded.pos)
                dir_delta = np.linalg.norm(out_orig.dir - out_loaded.dir)

                self.assertLessEqual(pos_delta, 1e-12, f"Pos delta failed at angle {test_val}: {pos_delta}")
                self.assertLessEqual(dir_delta, 1e-12, f"Dir delta failed at angle {test_val}: {dir_delta}")

    def test_multi_mechanism_branched_tracer_consistency(self):
        """
        Verify that BranchedTracer tree topology and leaf ray trajectories remain
        strictly identical after JSON reload in a branched optical system with ActiveMechanisms.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Input", thickness=35.0, glass="AIR"))

        # BeamSplitter at surface 1
        bs_cfg = BeamSplitterConfig(split_ratio_R=0.45, split_ratio_T=0.55)
        stack.add_surface(Surface(
            name="BeamSplitter_01",
            surface_type=SurfaceType.STANDARD,
            thickness=40.0,
            glass="AIR",
            semi_diameter=50.0,
            tilt_x=45.0,
            is_beam_splitter=True,
            beamsplitter_config=bs_cfg,
        ))

        # Actuated Mirror on transmitted arm
        mech_t = ActiveMechanism(
            mechanism_id="piezo_trans",
            mechanism_type="piezo",
            channels={
                "tx": MechanismChannel(channel_name="tx", target_property="tilt_x", current_value=2.5),
                "ty": MechanismChannel(channel_name="ty", target_property="tilt_y", current_value=-1.2),
            }
        )
        s_act_t = Surface(
            name="Transmitted_Actuated_Mirror",
            surface_type=SurfaceType.MIRROR,
            thickness=50.0,
            tilt_x=30.0,
            semi_diameter=30.0,
            component_uid="comp_piezo_trans",
        )
        s_act_t.mechanism = mech_t
        stack.add_surface(s_act_t)
        stack.add_component(OpticalComponent(
            component_uid="comp_piezo_trans",
            name="PiezoTrans",
            component_type="ActiveMechanism",
            surface_uids=["Transmitted_Actuated_Mirror"],
            mechanism=mech_t,
        ))

        stack.add_surface(Surface(name="Detector_Main", surface_type=SurfaceType.DETECTOR, glass="AIR", semi_diameter=120.0))
        stack._update_accumulated_positions()

        sys_orig = SpectrometerSystem(surface_stack=stack)
        tracer_orig = BranchedTracer(stack)

        ray_in = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), intensity=1.0, wavelength=0.532)
        tree_orig = tracer_orig.trace_ray_tree(ray_in)
        leaves_orig = tree_orig.all_terminal_rays()

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "branched_mech.json")
            sys_orig.save_to_json(fpath)

            loaded_sys = SpectrometerSystem.load_from_json(fpath)
            tracer_loaded = BranchedTracer(loaded_sys.surface_stack)
            tree_loaded = tracer_loaded.trace_ray_tree(ray_in)
            leaves_loaded = tree_loaded.all_terminal_rays()

            self.assertEqual(len(tree_orig.nodes), len(tree_loaded.nodes))
            self.assertEqual(len(leaves_orig), len(leaves_loaded))
            self.assertAlmostEqual(tree_orig.total_energy(), tree_loaded.total_energy(), places=12)

            for l_orig, l_loaded in zip(leaves_orig, leaves_loaded):
                self.assertAlmostEqual(l_orig.intensity, l_loaded.intensity, places=12)
                pos_diff = np.linalg.norm(l_orig.pos - l_loaded.pos)
                dir_diff = np.linalg.norm(l_orig.dir - l_loaded.dir)
                self.assertLessEqual(pos_diff, 1e-12)
                self.assertLessEqual(dir_diff, 1e-12)

    def test_v2_active_mechanism_deflection_consistency_parity(self):
        """
        Verify cross-package parity: optics_app and v2 loaded systems produce 0.0 deflection divergence.
        """
        # Create systems in both packages
        s_app = Surface(name="Mirror_App", surface_type=SurfaceType.MIRROR, thickness=50.0, tilt_x=25.0)
        mech_app = ActiveMechanism(
            mechanism_id="v2_parity_mech",
            channels={"tx": MechanismChannel(channel_name="tx", target_property="tilt_x", current_value=4.0)}
        )
        s_app.mechanism = mech_app

        stack_app = SurfaceStack()
        stack_app.add_surface(Surface(name="Input", thickness=30.0, glass="AIR"))
        stack_app.add_surface(s_app)
        stack_app.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, glass="AIR"))
        stack_app._update_accumulated_positions()
        sys_app = SpectrometerSystem(surface_stack=stack_app)

        s_v2 = v2_surface.Surface(name="Mirror_V2", surface_type=v2_surface.SurfaceType.MIRROR, thickness=50.0, tilt_x=25.0)
        mech_v2 = v2_components.ActiveMechanism(
            mechanism_id="v2_parity_mech",
            channels={"tx": v2_components.MechanismChannel(channel_name="tx", target_property="tilt_x", current_value=4.0)}
        )
        s_v2.mechanism = mech_v2

        stack_v2 = v2_lde.SurfaceStack()
        stack_v2.add_surface(v2_surface.Surface(name="Input", thickness=30.0, glass="AIR"))
        stack_v2.add_surface(s_v2)
        stack_v2.add_surface(v2_surface.Surface(name="Det", surface_type=v2_surface.SurfaceType.DETECTOR, glass="AIR"))
        stack_v2._update_accumulated_positions()
        sys_v2 = v2_models.SpectrometerSystem(surface_stack=stack_v2)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_app = os.path.join(tmpdir, "sys_app.json")
            fpath_v2 = os.path.join(tmpdir, "sys_v2.json")

            sys_app.save_to_json(fpath_app)
            sys_v2.save_to_json(fpath_v2)

            loaded_app = SpectrometerSystem.load_from_json(fpath_app)
            loaded_v2 = v2_models.SpectrometerSystem.load_from_json(fpath_v2)

            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
            ray_v2 = v2_tracer.Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))

            out_app = OpticalSystem(loaded_app.surface_stack).trace_ray(ray)
            out_v2 = v2_tracer.OpticalSystem(loaded_v2.surface_stack).trace_ray(ray_v2)

            pos_diff = np.linalg.norm(out_app.pos - out_v2.pos)
            dir_diff = np.linalg.norm(out_app.dir - out_v2.dir)

            self.assertEqual(pos_diff, 0.0)
            self.assertEqual(dir_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
