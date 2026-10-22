"""
Adversarial Stress Test Suite for Milestone 3:
Diagnostics Vignetting, Paraxial Ray Tracing & First-Order Analysis Under Extreme Optical Configurations.

Challenger 2 Verification Matrix:
1. Intermediate tiny aperture clipping: zero off-by-one errors across intermediate surfaces.
2. Wide field angles (Hy = 1.0, 1.5) and severe off-axis vignetting.
3. Paraxial ray tracing edge cases in compute_paraxial_marginal_and_principal_rays():
   - Afocal systems (power Phi ~ 0)
   - Flat plane-parallel plates
   - Telecentric systems (image-space, object-space, bi-telecentric)
   - Front stop vs internal stop vs rear stop vs detector stop
   - Finite conjugate (S0 = 100 mm, 10 mm, 1 mm, near-singularity) vs infinite conjugate
   - All-reflective mirrors
   - Numerical stability (no ZeroDivisionError, no NaN, no infinite loops)
"""

import math
import pytest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.diagnostics import OptimizationValidator, DiagnosticSeverity
from optics_app.engine.first_order import (
    compute_paraxial_marginal_and_principal_rays,
    calculate_first_order_data,
    format_zemax_first_order_report
)
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.engine.system_config import SystemConfig, FieldPoint
from optics_app.analysis.spot import SpotDiagram


# =====================================================================
# Target 1: Intermediate Aperture Clipping & Exact Blocking Surface
# =====================================================================

class TestIntermediateApertureClipping:
    """Stress tests verifying that validate_system() accurately identifies the exact blocking surface."""

    def _build_multi_surface_stack(self, block_surface_index: int, block_semi_diameter: float = 0.5) -> SurfaceStack:
        """
        Builds a 7-surface optical stack:
        0: Obj
        1: Lens 1 front
        2: Lens 1 rear
        3: Aperture Stop
        4: Lens 2 front
        5: Lens 2 rear
        6: Detector
        """
        stack = SurfaceStack()
        stack.add_surface(Surface('Obj', thickness=25.0, semi_diameter=30.0))
        stack.add_surface(Surface('L1_f', radius=60.0, thickness=6.0, glass='N-BK7',
                                  semi_diameter=block_semi_diameter if block_surface_index == 1 else 20.0))
        stack.add_surface(Surface('L1_r', radius=-120.0, thickness=12.0, glass='AIR',
                                  semi_diameter=block_semi_diameter if block_surface_index == 2 else 20.0))
        s3 = Surface('Stop', thickness=10.0, glass='AIR',
                     semi_diameter=block_semi_diameter if block_surface_index == 3 else 15.0)
        s3.surface_type = SurfaceType.STOP
        stack.add_surface(s3)
        stack.add_surface(Surface('L2_f', radius=40.0, thickness=5.0, glass='N-SF11',
                                  semi_diameter=block_semi_diameter if block_surface_index == 4 else 20.0))
        stack.add_surface(Surface('L2_r', radius=-80.0, thickness=25.0, glass='AIR',
                                  semi_diameter=block_semi_diameter if block_surface_index == 5 else 20.0))
        det = Surface('Detector', thickness=0.0, semi_diameter=30.0)
        det.surface_type = SurfaceType.DETECTOR
        stack.add_surface(det)
        return stack

    @pytest.mark.parametrize("block_idx", [1, 2, 3, 4, 5])
    def test_intermediate_aperture_clipping_exact_surface_detection(self, block_idx: int):
        """Verify that validate_system() detects the exact blocking surface with zero off-by-one errors."""
        stack = self._build_multi_surface_stack(block_surface_index=block_idx, block_semi_diameter=0.5)
        res = OptimizationValidator.validate_system(stack, entrance_pupil_radius=5.0)

        vig_items = [it for it in res.items if "VIGNETTING" in it.code]
        assert len(vig_items) > 0, f"Expected vignetting warning for tiny aperture on Surface #{block_idx}"
        
        detected_surface = vig_items[0].surface_index
        assert detected_surface == block_idx, (
            f"Off-by-one error detected! Intentionally choked Surface #{block_idx}, "
            f"but validate_system() reported Surface #{detected_surface}"
        )
        assert vig_items[0].severity in (DiagnosticSeverity.WARNING, DiagnosticSeverity.ERROR)
        assert f"Surface #{block_idx}" in vig_items[0].message

    def test_total_vs_partial_vignetting_codes(self):
        """
        Verify distinction between TOTAL_RAY_VIGNETTING (ERROR) and PARTIAL_RAY_VIGNETTING (WARNING).
        """
        # Partial vignetting: semi_diameter=0.5 mm clips marginal rays, chief ray passes
        stack_partial = self._build_multi_surface_stack(block_surface_index=3, block_semi_diameter=0.5)
        res_partial = OptimizationValidator.validate_system(stack_partial, entrance_pupil_radius=5.0)
        partial_items = [it for it in res_partial.items if it.code == "PARTIAL_RAY_VIGNETTING"]
        assert len(partial_items) == 1
        assert partial_items[0].surface_index == 3
        assert partial_items[0].severity == DiagnosticSeverity.WARNING

        # Total vignetting: All trial rays are obstructed because Surface 1 is shifted completely out of beam
        stack_total = SurfaceStack()
        stack_total.add_surface(Surface('Obj', thickness=20.0, semi_diameter=30.0))
        stack_total.add_surface(Surface('Shifted_Lens', radius=60.0, thickness=10.0, glass='N-BK7',
                                        semi_diameter=1.0, decenter_x=50.0))
        det = Surface('Det', thickness=0.0, semi_diameter=30.0)
        det.surface_type = SurfaceType.DETECTOR
        stack_total.add_surface(det)
        stack_total._update_accumulated_positions()

        res_total = OptimizationValidator.validate_system(stack_total, entrance_pupil_radius=5.0)
        total_items = [it for it in res_total.items if it.code == "TOTAL_RAY_VIGNETTING"]
        assert len(total_items) == 1
        assert total_items[0].surface_index == 1
        assert total_items[0].severity == DiagnosticSeverity.ERROR
        assert res_total.is_valid is False

    def test_multiple_vignetting_surfaces_earliest_detected(self):
        """
        When multiple surfaces have restricted apertures, the ray is clipped at the earliest surface encountered.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface('Obj', thickness=20.0, semi_diameter=30.0))
        stack.add_surface(Surface('S1', radius=50.0, thickness=5.0, glass='N-BK7', semi_diameter=1.0))  # chokes at S1
        stack.add_surface(Surface('S2', radius=-50.0, thickness=10.0, glass='AIR', semi_diameter=0.5)) # chokes even more at S2
        det = Surface('Det', thickness=0.0, semi_diameter=30.0)
        det.surface_type = SurfaceType.DETECTOR
        stack.add_surface(det)

        res = OptimizationValidator.validate_system(stack, entrance_pupil_radius=5.0)
        vig_items = [it for it in res.items if "VIGNETTING" in it.code]
        assert len(vig_items) > 0
        # The rays hit Surface 1 first with r = 5.0 > semi_diameter 1.0, so Surface 1 must be flagged
        assert vig_items[0].surface_index == 1


# =====================================================================
# Target 2: Wide Field Angles (Hy = 1.0, 1.5) & Off-Axis Clipping
# =====================================================================

class TestWideFieldAngleAdversarial:
    """Stress tests verifying paraxial and real ray tracing under wide field angles."""

    def test_paraxial_wide_field_angles_hy_progression(self):
        """
        Test paraxial ray propagation and vignetting factors as field angle increases
        from on-axis (Hy=0.0) through nominal edge (Hy=1.0) and extreme off-axis (Hy=1.5).
        """
        s0 = Surface('Obj', thickness=0.0)
        s1 = Surface('L1_f', radius=40.0, thickness=6.0, glass='N-BK7', semi_diameter=12.0)
        s2 = Surface('L1_r', radius=-60.0, thickness=15.0, glass='AIR', semi_diameter=12.0)
        s3 = Surface('Stop', thickness=15.0, semi_diameter=6.0)
        s3.surface_type = SurfaceType.STOP
        s4 = Surface('L2_f', radius=35.0, thickness=5.0, glass='N-SF11', semi_diameter=12.0)
        s5 = Surface('L2_r', radius=-50.0, thickness=35.0, glass='AIR', semi_diameter=12.0)
        det = Surface('Det', thickness=0.0, semi_diameter=20.0)
        det.surface_type = SurfaceType.DETECTOR
        surfaces = [s0, s1, s2, s3, s4, s5, det]

        nominal_field = 14.0  # degrees (Hy = 1.0)
        fields = {
            "Hy_0_0": 0.0,
            "Hy_0_707": 0.707 * nominal_field,
            "Hy_1_0": 1.0 * nominal_field,
            "Hy_1_5": 1.5 * nominal_field,
            "Hy_2_0": 2.0 * nominal_field,
            "Hy_extreme_80deg": 80.0
        }

        prev_vig = -1e-9
        for name, ang in fields.items():
            res = compute_paraxial_marginal_and_principal_rays(surfaces, field_angle_deg=ang)
            vig = res["system_vignetting_factor"]
            
            assert 0.0 <= vig <= 1.0, f"Vignetting factor for {name} ({ang} deg) out of [0, 1]: {vig}"
            assert not math.isnan(vig), f"Vignetting factor is NaN for {name}"
            
            # Trajectory sanity
            assert len(res["surfaces_data"]) == len(surfaces)
            for sd in res["surfaces_data"]:
                for key in ["y_marginal", "u_marginal", "y_chief", "u_chief", "beam_radius", "vignetting_factor"]:
                    val = sd[key]
                    assert not math.isnan(val), f"NaN encountered in {key} at surface #{sd['surface_index']}"
                    assert not math.isinf(val), f"Inf encountered in {key} at surface #{sd['surface_index']}"

            # Envelope upper must be strictly >= envelope lower
            up = res["envelope_upper"]["y"]
            dn = res["envelope_lower"]["y"]
            for u_val, d_val in zip(up, dn):
                assert u_val >= d_val - 1e-9, f"Envelope inversion: upper {u_val} < lower {d_val}"

            # Vignetting should be monotonically non-decreasing with field angle
            assert vig >= prev_vig - 1e-6, f"Vignetting decreased with field: prev={prev_vig}, curr={vig}"
            prev_vig = vig

    def test_optimizer_multi_field_ray_generation_wide_angles(self):
        """
        Verify that OpticalOptimizer generates, traces, and evaluates multi-field rays
        for wide field angles (Hy = 1.0, 1.5) without raising exceptions.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface('Obj', thickness=0.0))
        stack.add_surface(Surface('L1_f', radius=35.0, thickness=5.0, glass='N-BK7', semi_diameter=12.0))
        stack.add_surface(Surface('L1_r', radius=-45.0, thickness=8.0, glass='AIR', semi_diameter=12.0))
        s_stop = Surface('Stop', thickness=8.0, semi_diameter=6.0)
        s_stop.surface_type = SurfaceType.STOP
        stack.add_surface(s_stop)
        stack.add_surface(Surface('L2_f', radius=28.0, thickness=4.0, glass='N-SF11', semi_diameter=12.0))
        stack.add_surface(Surface('L2_r', radius=-55.0, thickness=30.0, glass='AIR', semi_diameter=12.0))
        det = Surface('Det', thickness=0.0, semi_diameter=20.0)
        det.surface_type = SurfaceType.DETECTOR
        stack.add_surface(det)
        stack._update_accumulated_positions()

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, wavelengths=[0.5876], entrance_pupil_radius=5.0)
        # 3 fields: on-axis, nominal 20 deg (Hy=1.0), extreme 30 deg (Hy=1.5)
        opt.fields = [(0.0, 0.0, 1.0), (0.0, 20.0, 2.0), (0.0, 30.0, 1.0)]

        rays = opt._generate_rays(pupil_radius=5.0, wavelengths=[0.5876], num_rays=31, eval_stack=stack)
        assert len(rays) > 0

        traced = sys.trace_rays(rays)
        assert len(traced) == len(rays)

        # Confirm off-axis rays at Hy=1.5 (30 deg) clip while on-axis rays pass
        field_0_vig = sum(1 for r in traced if getattr(r, 'field_idx', 0) == 0 and r.vignetted)
        field_2_vig = sum(1 for r in traced if getattr(r, 'field_idx', 0) == 2 and r.vignetted)
        assert field_0_vig == 0, "On-axis rays should not be vignetted in this standard geometry"
        assert field_2_vig > 0, "Wide angle Hy=1.5 (30 deg) rays should suffer partial vignetting"

        # Spot diagram analysis per field
        for f_idx in [0, 1, 2]:
            field_rays = [r for r in traced if getattr(r, 'field_idx', 0) == f_idx]
            spot_hits_x = []
            spot_hits_y = []
            c_coord = None
            for r in field_rays:
                for h in r.history:
                    if h.get("surface_index") == len(stack.surfaces) - 1:
                        spot_hits_x.append(h["x"])
                        spot_hits_y.append(h["y"])
                        if getattr(r, "is_chief", False) and c_coord is None:
                            c_coord = (float(h["x"]), float(h["y"]))
                        break

            spot = SpotDiagram(spot_hits_x, spot_hits_y, reference="centroid", chief_ray_coord=c_coord)
            rms_c = spot.calculate_rms_radius(reference="centroid")
            rms_cr = spot.calculate_rms_radius(reference="chief_ray")
            assert not math.isnan(rms_c)
            assert not math.isnan(rms_cr)
            assert rms_c >= 0.0
            assert rms_cr >= 0.0

        # Merit evaluation survives wide-field clipping
        pure_merit, penalty = opt._evaluate_merit_and_penalty(
            np.array([5.0]), {"thickness": [1]}
        )
        assert not math.isnan(pure_merit)
        assert not math.isinf(pure_merit)
        assert pure_merit > 0.0


# =====================================================================
# Target 3: Paraxial Ray Tracing Edge Cases
# =====================================================================

class TestParaxialRayTracingEdgeCases:
    """Stress tests covering extreme paraxial configurations without ZeroDivisionError or NaN."""

    def test_afocal_system_near_zero_power(self):
        """
        Verify paraxial ray tracing and first-order calculation on an afocal beam expander
        where optical power Phi = -C = 0.
        """
        s0 = Surface('Obj', thickness=0.0)
        # Thin positive lens: f1 = 50 mm
        s1 = Surface('L1_f', radius=25.84, thickness=0.0, glass='N-BK7')
        s2 = Surface('L1_r', radius=0.0, thickness=150.0, glass='AIR')
        s3 = Surface('L2_f', radius=51.68, thickness=0.0, glass='N-BK7')
        s4 = Surface('L2_r', radius=0.0, thickness=50.0, glass='AIR')
        det = Surface('Det', thickness=0.0)
        det.surface_type = SurfaceType.DETECTOR
        surfaces = [s0, s1, s2, s3, s4, det]

        fo = calculate_first_order_data(surfaces)
        p = compute_paraxial_marginal_and_principal_rays(surfaces, field_angle_deg=1.0)

        # Check no NaN
        for k, v in fo.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"NaN in first_order_data['{k}']"
        for sd in p["surfaces_data"]:
            for k, v in sd.items():
                if isinstance(v, float):
                    assert not math.isnan(v), f"NaN in surface #{sd['surface_index']} item '{k}'"

    def test_flat_plane_parallel_plates(self):
        """
        Verify paraxial trace through flat plane-parallel plates (R = 0, Phi = 0).
        """
        s0 = Surface('Obj', thickness=25.0)
        s1 = Surface('Window1_f', radius=0.0, thickness=12.0, glass='N-BK7', semi_diameter=25.0)
        s2 = Surface('Window1_r', radius=0.0, thickness=20.0, glass='AIR', semi_diameter=25.0)
        s3 = Surface('Window2_f', radius=0.0, thickness=15.0, glass='FSILICA', semi_diameter=25.0)
        s4 = Surface('Window2_r', radius=0.0, thickness=30.0, glass='AIR', semi_diameter=25.0)
        det = Surface('Det', thickness=0.0, semi_diameter=25.0)
        det.surface_type = SurfaceType.DETECTOR
        surfaces = [s0, s1, s2, s3, s4, det]

        fo = calculate_first_order_data(surfaces)
        assert fo["is_afocal"] is True
        assert math.isinf(fo["effl"])
        assert fo["matrix"].C == 0.0

        p = compute_paraxial_marginal_and_principal_rays(surfaces, field_angle_deg=3.0)
        assert p["is_vignetted"] is False
        for sd in p["surfaces_data"]:
            assert not math.isnan(sd["y_marginal"])
            assert not math.isnan(sd["y_chief"])

    def test_telecentric_system_image_space(self):
        """
        Image-space telecentric: Stop placed at front focal point of imaging lens (t_stop = f).
        Exit pupil is at infinity.
        """
        s0 = Surface('Obj', thickness=50.0)
        s_stop = Surface('Stop', thickness=96.75, semi_diameter=10.0)
        s_stop.surface_type = SurfaceType.STOP
        s1 = Surface('Lens_f', radius=50.0, thickness=5.0, glass='N-BK7', semi_diameter=20.0)
        s2 = Surface('Lens_r', radius=0.0, thickness=96.75, glass='AIR', semi_diameter=20.0)
        det = Surface('Det', thickness=0.0, semi_diameter=20.0)
        det.surface_type = SurfaceType.DETECTOR
        surfaces = [s0, s_stop, s1, s2, det]

        fo = calculate_first_order_data(surfaces)
        assert any("Telecentric" in d for d in fo["paraxial_diagnostics"])
        assert not math.isnan(fo["exp_pos"])
        assert not math.isnan(fo["expd"])

        p = compute_paraxial_marginal_and_principal_rays(surfaces, field_angle_deg=2.0)
        assert not math.isnan(p["system_vignetting_factor"])

    def test_telecentric_system_object_space(self):
        """
        Object-space telecentric: Stop placed at rear focal point of front lens.
        Entrance pupil is at infinity.
        """
        s0 = Surface('Obj', thickness=50.0)
        s1 = Surface('Lens_f', radius=50.0, thickness=5.0, glass='N-BK7', semi_diameter=20.0)
        s2 = Surface('Lens_r', radius=0.0, thickness=96.75, glass='AIR', semi_diameter=20.0)
        s_stop = Surface('Stop', thickness=50.0, semi_diameter=10.0)
        s_stop.surface_type = SurfaceType.STOP
        det = Surface('Det', thickness=0.0, semi_diameter=20.0)
        det.surface_type = SurfaceType.DETECTOR
        surfaces = [s0, s1, s2, s_stop, det]

        fo = calculate_first_order_data(surfaces)
        assert not math.isnan(fo["enp_pos"])
        assert not math.isnan(fo["epd"])

        p = compute_paraxial_marginal_and_principal_rays(surfaces, field_angle_deg=2.0)
        assert not math.isnan(p["system_vignetting_factor"])

    def test_bi_telecentric_system(self):
        """
        Bi-telecentric 4f relay: Stop at shared focal plane between identical lenses.
        """
        s0 = Surface('Obj', thickness=96.75)
        s1 = Surface('L1_f', radius=50.0, thickness=5.0, glass='N-BK7', semi_diameter=20.0)
        s2 = Surface('L1_r', radius=0.0, thickness=96.75, glass='AIR', semi_diameter=20.0)
        s_stop = Surface('Stop', thickness=96.75, semi_diameter=10.0)
        s_stop.surface_type = SurfaceType.STOP
        s3 = Surface('L2_f', radius=0.0, thickness=5.0, glass='N-BK7', semi_diameter=20.0)
        s4 = Surface('L2_r', radius=-50.0, thickness=96.75, glass='AIR', semi_diameter=20.0)
        det = Surface('Det', thickness=0.0, semi_diameter=20.0)
        det.surface_type = SurfaceType.DETECTOR
        surfaces = [s0, s1, s2, s_stop, s3, s4, det]

        fo = calculate_first_order_data(surfaces)
        p = compute_paraxial_marginal_and_principal_rays(surfaces, field_angle_deg=1.5)
        assert not math.isnan(fo["effl"])
        assert not math.isnan(p["system_vignetting_factor"])

    @pytest.mark.parametrize("stop_location", ["front", "internal", "rear", "detector"])
    def test_stop_position_variations(self, stop_location: str):
        """
        Verify that compute_paraxial_marginal_and_principal_rays() and calculate_first_order_data()
        correctly handle stop at Surface 1 (front), intermediate, rear, and image plane.
        """
        s0 = Surface('Obj', thickness=40.0)
        s1 = Surface('L1_f', radius=50.0, thickness=5.0, glass='N-BK7', semi_diameter=15.0)
        s2 = Surface('L1_r', radius=-50.0, thickness=10.0, glass='AIR', semi_diameter=15.0)
        s3 = Surface('L2_f', radius=40.0, thickness=5.0, glass='N-SF11', semi_diameter=15.0)
        s4 = Surface('L2_r', radius=-80.0, thickness=30.0, glass='AIR', semi_diameter=15.0)
        det = Surface('Det', thickness=0.0, semi_diameter=15.0)
        det.surface_type = SurfaceType.DETECTOR

        if stop_location == "front":
            s1.surface_type = SurfaceType.STOP
        elif stop_location == "internal":
            s2.surface_type = SurfaceType.STOP
        elif stop_location == "rear":
            s4.surface_type = SurfaceType.STOP
        elif stop_location == "detector":
            det.surface_type = SurfaceType.STOP

        surfs = [s0, s1, s2, s3, s4, det]
        fo = calculate_first_order_data(surfs)
        p = compute_paraxial_marginal_and_principal_rays(surfs, field_angle_deg=2.0)

        assert not math.isnan(fo["enp_pos"])
        assert not math.isnan(fo["exp_pos"])
        assert not math.isnan(p["system_vignetting_factor"])
        assert p["stop_index"] >= 0

    @pytest.mark.parametrize("s0_val", [1e9, 1000.0, 100.0, 10.0, 1.0, 0.001, 0.0, -15.0])
    def test_finite_and_infinite_conjugate_distances(self, s0_val: float):
        """
        Verify robustness across finite conjugates (1000 to 0.001 mm), zero, negative,
        and infinite conjugates without ZeroDivisionError or NaN.
        """
        s0 = Surface('Obj', thickness=s0_val)
        s0.surface_type = SurfaceType.OBJECT
        s1 = Surface('L1_f', radius=50.0, thickness=5.0, glass='N-BK7', semi_diameter=15.0)
        s2 = Surface('L1_r', radius=-50.0, thickness=10.0, glass='AIR', semi_diameter=15.0)
        det = Surface('Det', thickness=0.0, semi_diameter=15.0)
        det.surface_type = SurfaceType.DETECTOR
        surfs = [s0, s1, s2, det]

        fo = calculate_first_order_data(surfs)
        p = compute_paraxial_marginal_and_principal_rays(surfs, field_angle_deg=2.0)

        assert not math.isnan(fo["effl"])
        assert not math.isnan(p["system_vignetting_factor"])
        assert 0.0 <= p["system_vignetting_factor"] <= 1.0

    def test_finite_conjugate_near_singularity_protection(self):
        """
        Test the extreme edge case where s0 + z_enp == 0.
        Verifies that denominator clamping (abs(s0 + z_enp) > 1e-6) prevents ZeroDivisionError.
        """
        s0 = Surface('Obj', thickness=10.0)
        s0.surface_type = SurfaceType.OBJECT
        s1 = Surface('S1', thickness=10.0, semi_diameter=10.0)
        # S2 is stop so z_enp = 10.0
        s2 = Surface('Stop', thickness=10.0, semi_diameter=10.0)
        s2.surface_type = SurfaceType.STOP
        s3 = Surface('Det', thickness=0.0, semi_diameter=10.0)
        s3.surface_type = SurfaceType.DETECTOR
        surfs = [s0, s1, s2, s3]

        # Artificially set s0 = -10.0 so s0 + z_enp = -10.0 + 10.0 = 0.0
        s0.thickness = -10.0
        res = compute_paraxial_marginal_and_principal_rays(surfs, field_angle_deg=1.0)
        assert res is not None
        assert not math.isnan(res["system_vignetting_factor"])

    def test_all_reflective_mirror_sequential_trace(self):
        """
        Verify sequential paraxial ray trace through a 2-mirror reflective system.
        Note that when Surface 0 is SurfaceType.OBJECT, the returned surfaces_data
        corresponds to optical surfaces 1 through end (len(surfs) - 1).
        """
        s0 = Surface('Obj', thickness=50.0)
        s0.surface_type = SurfaceType.OBJECT
        s1 = Surface('Primary_Mirror', radius=-120.0, thickness=-40.0, glass='MIRROR', semi_diameter=25.0)
        s1.surface_type = SurfaceType.MIRROR
        s2 = Surface('Secondary_Mirror', radius=-60.0, thickness=35.0, glass='MIRROR', semi_diameter=15.0)
        s2.surface_type = SurfaceType.MIRROR
        det = Surface('Det', thickness=0.0, semi_diameter=15.0)
        det.surface_type = SurfaceType.DETECTOR
        surfs = [s0, s1, s2, det]

        p = compute_paraxial_marginal_and_principal_rays(surfs, field_angle_deg=0.5)
        assert p is not None
        assert not math.isnan(p["system_vignetting_factor"])
        # Optical surfaces: Primary_Mirror, Secondary_Mirror, Det
        assert len(p["surfaces_data"]) == len(surfs) - 1

    def test_minimal_boundary_surface_counts(self):
        """
        Verify graceful return on empty, single-surface, or degenerate stacks.
        """
        res_empty = compute_paraxial_marginal_and_principal_rays([])
        assert res_empty["surfaces_data"] == []
        assert res_empty["system_vignetting_factor"] == 0.0

        res_single = compute_paraxial_marginal_and_principal_rays([Surface('Only')])
        assert res_single["surfaces_data"] == []
        assert res_single["system_vignetting_factor"] == 0.0

        fo_empty = calculate_first_order_data([])
        assert fo_empty["is_afocal"] is True
        assert math.isinf(fo_empty["effl"])

    @pytest.mark.parametrize("epd_val", [0.001, 0.1, 10.0, 100.0, 1000.0])
    def test_extreme_pupil_diameters(self, epd_val: float):
        """
        Verify that compute_paraxial_marginal_and_principal_rays() handles tiny and massive EPD values.
        """
        s0 = Surface('Obj', thickness=50.0)
        s0.surface_type = SurfaceType.OBJECT
        s1 = Surface('L1_f', radius=50.0, thickness=5.0, glass='N-BK7', semi_diameter=15.0)
        s2 = Surface('L1_r', radius=-50.0, thickness=20.0, glass='AIR', semi_diameter=15.0)
        det = Surface('Det', thickness=0.0, semi_diameter=15.0)
        det.surface_type = SurfaceType.DETECTOR
        surfs = [s0, s1, s2, det]

        p = compute_paraxial_marginal_and_principal_rays(surfs, epd=epd_val, field_angle_deg=1.0)
        assert not math.isnan(p["system_vignetting_factor"])
        assert 0.0 <= p["system_vignetting_factor"] <= 1.0
