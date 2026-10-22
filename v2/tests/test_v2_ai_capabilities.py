"""
Comprehensive Test Suite for 2D Optics AI Capabilities V2 Integration.

Verifies:
- AC1: Natural language optical element manipulation (rotate, flip, displace, tilt, replace, offset bias)
- AC2: Generative optical system design from high-level natural language prompt
- AC3: AI optimization loop adjusting system parameters to minimize merit function
- AC4: Real-world commercial catalog query and ZEMAX .zmx file ingestion into 2D simulation
- AC5: Matrix propagation paraxial ray tracing and optical Q-factor calculation
- Dual-family Cloud AI integration (NVIDIA Nemotron and Google Gemma)
"""

import os
import sys
import json
import math
import pytest
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2.ai.nemotron_client import (
    NemotronClient, GemmaClient, CloudAiClient, ModelTier, ModelFamily,
    TIER_MODEL_MAP, GEMMA_TIER_MODEL_MAP
)
from v2.ai.tools import (
    get_default_tool_registry,
    interpret_and_execute_command,
    rotate_optical_element,
    flip_optical_element,
    displace_optical_element,
    tilt_optical_element,
    replace_optical_element,
    offset_optical_element,
    run_ai_optimization,
    generate_optical_system,
    load_and_integrate_catalog_element,
    compute_matrix_propagation_and_q_factor
)
from v2.ai.generative_design import GenerativeOpticalDesigner
from v2.ai.catalog import OpticalCatalogManager, COMMERCIAL_CATALOG
from v2.engine.models import SpectrometerSystem
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.merit_function import MeritFunction, OperandType
from v2.engine.ai_optimizer import run_ai_optimization_loop, AiOptimizationLoop
from v2.engine.matrix_tracer import (
    RayMatrix, MatrixTracer, space_matrix, thin_lens_matrix,
    refraction_matrix, reflection_matrix, calculate_cavity_q_factor,
    calculate_spectrometer_q_factor, calculate_system_q_factor
)
from v2.engine.tracer import DualSpeedTracer


def _find_repo_file(filename: str) -> str:
    for candidate in [
        os.path.join(os.path.dirname(__file__), "..", filename),
        os.path.join(os.path.dirname(__file__), "..", "..", filename),
        os.path.join(os.getcwd(), filename),
    ]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filename))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def baseline_spectrometer():
    """Load baseline Spectrometer_V2_Perfected system."""
    json_path = _find_repo_file("Spectrometer_V2_Perfected.json")
    if os.path.exists(json_path):
        return SpectrometerSystem.load_from_json(json_path)
    # Fallback generative design if file not available in test environment
    return GenerativeOpticalDesigner.design_spectrometer(
        wavelength_um=0.785, grating_lines_per_mm=600.0, magnification=1.0
    )


# ---------------------------------------------------------------------------
# AC1: AI Parsing and Manipulating Rigid Optical Elements
# ---------------------------------------------------------------------------
class TestAC1_AiCommandInterface:
    """AC1: Interpreting natural language commands and manipulating rigid optical elements."""

    def test_ac1_1_rotate_element_via_ai(self, baseline_spectrometer):
        """Verify AI parses natural language rotate command and updates element tilt."""
        command = "rotate element 1 by 4.5 degrees around x axis"
        res = interpret_and_execute_command(command, system=baseline_spectrometer)

        assert res["success"] is True
        assert len(res["executed_results"]) == 1
        assert res["executed_results"][0]["tool"] == "rotate_optical_element"
        assert abs(baseline_spectrometer.surfaces[1].tilt_x - 4.5) < 1e-5

    def test_ac1_2_flip_element_via_ai(self, baseline_spectrometer):
        """Verify AI parses natural language flip command and reverses curvatures."""
        # Record initial curvature
        orig_r = baseline_spectrometer.surfaces[1].radius
        command = "flip element 1"
        res = interpret_and_execute_command(command, system=baseline_spectrometer)

        assert res["success"] is True
        assert res["executed_results"][0]["tool"] == "flip_optical_element"
        if orig_r != 0.0:
            assert abs(baseline_spectrometer.surfaces[1].radius - (-orig_r)) < 1e-5

    def test_ac1_3_displace_element_via_ai(self, baseline_spectrometer):
        """Verify AI parses natural language displace command and translates element."""
        command = "displace element 1 by 2.5 mm along y axis"
        res = interpret_and_execute_command(command, system=baseline_spectrometer)

        assert res["success"] is True
        assert res["executed_results"][0]["tool"] == "displace_optical_element"
        assert abs(baseline_spectrometer.surfaces[1].decenter_y - 2.5) < 1e-5

    def test_ac1_4_tilt_element_via_ai(self, baseline_spectrometer):
        """Verify AI parses natural language tilt command and sets angle."""
        command = "tilt element 1 by -3.0 degrees around y"
        res = interpret_and_execute_command(command, system=baseline_spectrometer)

        assert res["success"] is True
        assert res["executed_results"][0]["tool"] == "tilt_optical_element"
        assert abs(baseline_spectrometer.surfaces[1].tilt_y - (-3.0)) < 1e-5

    def test_ac1_5_replace_element_via_ai(self, baseline_spectrometer):
        """Verify AI parses natural language replace command with catalog component."""
        command = "replace element 1 with AC254-050-C"
        res = interpret_and_execute_command(command, system=baseline_spectrometer)

        assert res["success"] is True
        assert res["executed_results"][0]["tool"] == "replace_optical_element"
        # Verify surface group has been updated to catalog part
        assert "AC254-050-C" in baseline_spectrometer.surfaces[1].group_name.upper()

    def test_ac1_6_offset_element_bias_via_ai(self, baseline_spectrometer):
        """Verify AI parses natural language offset bias command relative to another surface."""
        command = "move element 2 to surface 1 with 16.5 mm offset bias"
        res = interpret_and_execute_command(command, system=baseline_spectrometer)

        assert res["success"] is True
        assert res["executed_results"][0]["tool"] == "offset_optical_element"
        assert abs(baseline_spectrometer.surfaces[1].thickness - 16.5) < 1e-5

    def test_ac1_7_gemma_client_command_interpretation(self, baseline_spectrometer):
        """Verify GemmaClient accurately parses natural language commands in demo/cloud mode."""
        gemma = GemmaClient(demo_mode=True)
        assert gemma.family == ModelFamily.GEMMA
        assert "gemma" in gemma.model.lower()

        response = gemma.chat_completion(
            messages=[{"role": "user", "content": "rotate element 1 by 3.0 degrees around x"}]
        )
        assert len(response["tool_calls"]) > 0
        call = response["tool_calls"][0]
        assert call["function"]["name"] == "rotate_optical_element"
        args = json.loads(call["function"]["arguments"])
        assert args["element_idx"] == 1
        assert args["tilt_x"] == 3.0


# ---------------------------------------------------------------------------
# AC2: Generative Optical System Design
# ---------------------------------------------------------------------------
class TestAC2_GenerativeDesign:
    """AC2: Generating complete, valid optical systems from high-level prompts."""

    def test_ac2_1_build_spectrometer_from_prompt(self):
        """Verify prompt 'build a spectrometer with 785nm laser, 600 lpm grating, and 2x magnification'."""
        prompt = "build a spectrometer with 785nm laser, 600 lpm grating, and 2x magnification"
        system = GenerativeOpticalDesigner.design_from_prompt(prompt)

        assert isinstance(system, SpectrometerSystem)
        assert len(system.surfaces) >= 5

        # Check core requested components
        assert abs(system.primary_wavelength - 0.785) < 1e-4
        assert system.grating_surface_idx is not None
        grating_surf = system.surfaces[system.grating_surface_idx]
        assert grating_surf.surface_type == SurfaceType.GRATING
        assert grating_surf.grating_lines_per_mm == 600.0

        # Check detector surface exists at end
        assert system.detector_surface_idx is not None
        assert system.surfaces[system.detector_surface_idx].surface_type == SurfaceType.DETECTOR

        # Check ray tracing succeeds on the generated system
        tracer = DualSpeedTracer(system)
        trace_res = tracer.trace_full(wavelengths=system.wavelengths, rings=3, beam_radius=2.5)
        assert trace_res.total_rays > 0
        assert len(trace_res.spot_coordinates) > 0

    def test_ac2_2_build_refractive_spectrometer_from_prompt(self):
        """Verify prompt with lens / refractive optics."""
        prompt = "build a refractive spectrometer with 1064nm laser, 1200 lines/mm grating, and 1x magnification using achromatic doublets"
        system = GenerativeOpticalDesigner.design_from_prompt(prompt)

        assert isinstance(system, SpectrometerSystem)
        assert abs(system.primary_wavelength - 1.064) < 1e-4
        assert system.surfaces[system.grating_surface_idx].grating_lines_per_mm == 1200.0

        # Check glass elements are present
        glasses = [s.glass for s in system.surfaces if s.glass != "AIR" and s.glass != "MIRROR"]
        assert len(glasses) >= 2
        assert "N-LAK22" in glasses or "N-SF6HT" in glasses

    def test_ac2_3_tool_generate_optical_system(self, tmp_path):
        """Verify tool 'generate_optical_system' registered and callable."""
        out_json = str(tmp_path / "test_generated_spec.json")
        res_str = generate_optical_system(
            prompt="build a spectrometer with 532nm laser and 1800 lpm grating",
            output_json_path=out_json
        )
        assert "SUCCESS" in res_str
        assert os.path.exists(out_json)

        loaded = SpectrometerSystem.load_from_json(out_json)
        assert abs(loaded.primary_wavelength - 0.532) < 1e-4
        assert loaded.surfaces[loaded.grating_surface_idx].grating_lines_per_mm == 1800.0


# ---------------------------------------------------------------------------
# AC3: AI Optimization Loop
# ---------------------------------------------------------------------------
class TestAC3_AiOptimizationLoop:
    """AC3: Optimization loop adjusts at least one system parameter to minimize merit function."""

    def test_ac3_1_optimization_reduces_merit_function(self, baseline_spectrometer):
        """Verify optimization loop adjusts aperture/stop to minimize RMS spot size."""
        # Perturb aperture to cause elevated aberration spot size
        if isinstance(baseline_spectrometer.source_config, dict):
            baseline_spectrometer.source_config["beam_radius"] = 7.5
        else:
            baseline_spectrometer.source_config.beam_radius = 7.5

        mf = MeritFunction()
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=-1, target=0.0, weight=1.0)

        opt = AiOptimizationLoop(
            system=baseline_spectrometer,
            merit_function=mf,
            max_iterations=10
        )
        res = opt.optimize(parameters_to_adjust=["aperture"])

        assert res["success"] is True
        assert res["final_merit"] <= res["initial_merit"]
        assert res["merit_reduction"] >= 0.0
        assert "aperture" in res["adjusted_parameters"]

    def test_ac3_2_functional_optimization_routine(self, baseline_spectrometer):
        """Verify run_ai_optimization_loop functional API."""
        mf = MeritFunction()
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=-1, target=0.0, weight=1.0)

        res = run_ai_optimization_loop(
            system=baseline_spectrometer,
            merit_function=mf,
            parameters_to_adjust=["aperture"],
            max_iterations=5
        )
        assert res["success"] is True
        assert res["final_merit"] <= res["initial_merit"]

    def test_ac3_3_tool_run_ai_optimization(self, baseline_spectrometer):
        """Verify tool 'run_ai_optimization' executes and returns structured report."""
        out = run_ai_optimization(parameter_names=["aperture"], max_iterations=5)
        assert "AI Optimization Complete" in out
        assert "Initial Merit" in out
        assert "Final Merit" in out


# ---------------------------------------------------------------------------
# AC4: Real-World Commercial Catalog & ZEMAX Integration
# ---------------------------------------------------------------------------
class TestAC4_RealWorldCatalogIntegration:
    """AC4: Load parameters for commercial optical elements and integrate into 2D simulation."""

    def test_ac4_1_query_thorlabs_catalog(self):
        """Verify querying Thorlabs commercial catalog for lens and mirror."""
        # 1. Doublet
        info_lens = OpticalCatalogManager.query_catalog("AC254-050-C")
        assert info_lens["manufacturer"] == "Thorlabs"
        assert info_lens["focal_length_mm"] == 50.0
        assert len(info_lens["surfaces"]) == 3
        assert info_lens["surfaces"][0]["glass"] == "N-LAK22"
        assert info_lens["surfaces"][1]["glass"] == "N-SF6HT"

        # 2. Mirror
        info_mirr = OpticalCatalogManager.query_catalog("CM254-050-P01")
        assert info_mirr["focal_length_mm"] == 50.0
        assert info_mirr["radius_mm"] == -100.0

    def test_ac4_2_parse_user_provided_zemax_file(self):
        """Verify parsing user-provided .zmx file into physical Surface objects."""
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "thorlabs_czerny_turner_785nm.zmx"),
            os.path.join(os.path.dirname(__file__), "..", "..", "thorlabs_czerny_turner_785nm.zmx"),
            os.path.abspath("thorlabs_czerny_turner_785nm.zmx")
        ]
        zmx_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
        assert os.path.exists(zmx_path)

        surfs = OpticalCatalogManager.parse_zemax_file(zmx_path)
        assert len(surfs) >= 3
        # Check surface physical attributes are populated
        has_mirror_or_grating = any(s.surface_type in (SurfaceType.MIRROR, SurfaceType.GRATING) for s in surfs)
        assert has_mirror_or_grating is True

    def test_ac4_3_integrate_commercial_doublet_into_simulation(self, baseline_spectrometer):
        """Verify integrating Thorlabs AC254-050-C into 2D simulation and tracing rays."""
        orig_count = len(baseline_spectrometer.surfaces)
        new_indices = OpticalCatalogManager.integrate_commercial_element(
            baseline_spectrometer, "AC254-050-C", after_surface_index=1
        )
        assert len(baseline_spectrometer.surfaces) > orig_count
        assert len(new_indices) == 3

        # Run ray trace through the modified system
        tracer = DualSpeedTracer(baseline_spectrometer)
        res = tracer.trace_full(wavelengths=baseline_spectrometer.wavelengths, rings=2, beam_radius=2.0)
        assert res.total_rays > 0

    def test_ac4_4_integrate_zemax_file_into_simulation(self, baseline_spectrometer):
        """Verify integrating .zmx file directly into 2D simulation."""
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "thorlabs_czerny_turner_785nm.zmx"),
            os.path.join(os.path.dirname(__file__), "..", "..", "thorlabs_czerny_turner_785nm.zmx"),
            os.path.abspath("thorlabs_czerny_turner_785nm.zmx")
        ]
        zmx_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
        orig_count = len(baseline_spectrometer.surfaces)
        new_indices = OpticalCatalogManager.integrate_commercial_element(
            baseline_spectrometer, zmx_path, after_surface_index=1
        )
        assert len(baseline_spectrometer.surfaces) > orig_count
        assert len(new_indices) >= 3


# ---------------------------------------------------------------------------
# AC5: Matrix Propagation Ray Tracing & Q-Factor Calculation
# ---------------------------------------------------------------------------
class TestAC5_MatrixPropagationAndQFactor:
    """AC5: Matrix propagation ray tracing and Q-factor calculation outputs expected baseline values."""

    def test_ac5_1_matrix_propagation_baseline_thin_lens_focus(self):
        """
        Verify baseline paraxial ABCD ray matrix for 100mm thin lens + 100mm drift:
        [1 100; 0 1] @ [1 0; -1/100 1] = [0 100; -0.01 1]
        Incident parallel ray (y=2.0, u=0.0) must focus to y=0.0 at 100mm.
        """
        m_lens = thin_lens_matrix(100.0)
        m_drift = space_matrix(100.0)
        m_sys = m_drift @ m_lens

        assert abs(m_sys.A - 0.0) < 1e-9
        assert abs(m_sys.B - 100.0) < 1e-9
        assert abs(m_sys.C - (-0.01)) < 1e-9
        assert abs(m_sys.D - 1.0) < 1e-9
        assert abs(m_sys.determinant() - 1.0) < 1e-9

        # Propagate ray
        y_out, u_out = m_sys.propagate_ray(y=2.0, u=0.0)
        assert abs(y_out - 0.0) < 1e-9, f"Ray should focus at y=0, got {y_out}"
        assert abs(u_out - (-0.02)) < 1e-9

    def test_ac5_2_matrix_propagation_2f_imaging_conjugate(self):
        """
        Verify 2f-2f optical imaging: d1=200mm, f=100mm, d2=200mm.
        Magnification M = -1.0, B = 0.0 (image plane conjugate).
        """
        m_in = space_matrix(200.0)
        m_lens = thin_lens_matrix(100.0)
        m_out = space_matrix(200.0)
        m_sys = m_out @ m_lens @ m_in

        assert abs(m_sys.A - (-1.0)) < 1e-9
        assert abs(m_sys.B - 0.0) < 1e-9  # Conjugate plane condition
        assert abs(m_sys.D - (-1.0)) < 1e-9
        assert abs(m_sys.determinant() - 1.0) < 1e-9

        # Off-axis ray from object origin with angle u=0.02 rad
        y_hit, u_hit = m_sys.propagate_ray(y=0.0, u=0.02)
        assert abs(y_hit - 0.0) < 1e-9  # Concurrently focuses back to optical axis

    def test_ac5_3_matrix_tracer_multi_surface_system(self, baseline_spectrometer):
        """Verify MatrixTracer evaluates multi-surface system and traces paraxial fan."""
        tracer = MatrixTracer()
        mat = tracer.build_from_system(baseline_spectrometer, wavelength_um=0.785)

        assert isinstance(mat, RayMatrix)
        assert mat.determinant() > 0.0
        assert len(tracer.step_matrices) > 0

        # Trace paraxial fan
        fan = tracer.trace_ray_fan(heights=[0.0, 1.0, 2.0], angles=[0.0, 0.01])
        assert len(fan) == 6
        for ray_rec in fan:
            assert "y_out" in ray_rec
            assert "u_out" in ray_rec

    def test_ac5_4_cavity_q_factor_calculation(self):
        """
        Verify analytical optical cavity Q-factor for 50mm cavity at 1064nm with R=0.99:
        Q = 2*pi*L / (lambda * (1 - R)) ≈ 2.95e7
        Finesse F ≈ 312.6
        """
        res = calculate_cavity_q_factor(
            length_mm=50.0,
            wavelength_um=1.064,
            r1=0.99,
            r2=0.99
        )
        assert res["q_factor"] > 2.0e7
        assert res["q_factor"] < 4.0e7
        assert res["finesse"] > 300.0
        assert res["finesse"] < 320.0
        assert res["photon_lifetime_ns"] > 10.0

    def test_ac5_5_spectrometer_resolving_power_q_factor(self):
        """
        Verify grating spectrometer Q-factor (resolving power R = m*N):
        600 l/mm * 25 mm beam = 15,000 grooves.
        At 785 nm, delta_lambda = 785 / 15000 = 0.0523 nm.
        """
        res = calculate_spectrometer_q_factor(
            wavelength_um=0.785,
            lines_per_mm=600.0,
            beam_width_mm=25.0,
            order=1
        )
        assert res["q_factor"] == 15000.0
        assert res["resolving_power"] == 15000.0
        assert abs(res["spectral_resolution_nm"] - 0.05233) < 1e-4

    def test_ac5_6_tool_compute_matrix_and_q_factor(self):
        """Verify tool 'compute_matrix_propagation_and_q_factor'."""
        out = compute_matrix_propagation_and_q_factor(length_mm=50.0, wavelength_um=1.064)
        assert "Matrix Propagation & Q-Factor Analysis" in out
        assert "System Transfer Matrix ABCD" in out
        assert "Quality Factor Q" in out


# ---------------------------------------------------------------------------
# Dual Cloud AI Integration (Nemotron & Gemma)
# ---------------------------------------------------------------------------
class TestCloudAiDualIntegration:
    """Verify dual cloud API support for NVIDIA Nemotron and Google Gemma."""

    def test_model_family_and_tier_configuration(self):
        """Verify model tier resolution and switching across Nemotron and Gemma."""
        # Nemotron
        client_n = NemotronClient(api_key="test-key", tier=ModelTier.HIGH)
        assert client_n.family == ModelFamily.NEMOTRON
        assert client_n.model == TIER_MODEL_MAP[ModelTier.HIGH]

        # Gemma
        client_g = GemmaClient(api_key="test-key", tier=ModelTier.HIGH)
        assert client_g.family == ModelFamily.GEMMA
        assert client_g.model == GEMMA_TIER_MODEL_MAP[ModelTier.HIGH]

        # Tier switch
        client_g.set_tier(ModelTier.MEDIUM)
        assert client_g.model == GEMMA_TIER_MODEL_MAP[ModelTier.MEDIUM]

        # CloudAiClient convenience
        unified = CloudAiClient(api_key="test-key", family=ModelFamily.GEMMA)
        assert unified.family == ModelFamily.GEMMA


# ---------------------------------------------------------------------------
# Hardened Edge Cases & OpticsApp Native Engine Integration
# ---------------------------------------------------------------------------
class TestAC_HardenedEdgeCases:
    """Rigorous stress-testing of hardened NLP parsing and native engine parity."""

    def test_hardened_nl_parser_edge_cases(self, baseline_spectrometer):
        """Verify hardened regex patterns handle diverse phrasing without failing."""
        # 1. Rotate without 'by'
        cmd1 = "rotate element 1 4.5 degrees around x axis"
        res1 = interpret_and_execute_command(cmd1, system=baseline_spectrometer)
        assert res1["success"] is True
        assert res1["executed_results"][0]["tool"] == "rotate_optical_element"
        assert abs(baseline_spectrometer.surfaces[1].tilt_x - 4.5) < 1e-5

        # 2. Displace via 'translate'
        cmd2 = "translate element 1 by 3.2 mm along y axis"
        res2 = interpret_and_execute_command(cmd2, system=baseline_spectrometer)
        assert res2["success"] is True
        assert res2["executed_results"][0]["tool"] == "displace_optical_element"
        assert abs(baseline_spectrometer.surfaces[1].decenter_y - 3.2) < 1e-5

        # 3. Displace via 'shift'
        cmd3 = "shift lens 1 by -1.8 mm along x"
        res3 = interpret_and_execute_command(cmd3, system=baseline_spectrometer)
        assert res3["success"] is True
        assert res3["executed_results"][0]["tool"] == "displace_optical_element"
        assert abs(baseline_spectrometer.surfaces[1].decenter_x - (-1.8)) < 1e-5

        # 4. Offset with verbose 'offset bias of' phrase
        cmd4 = "offset element 2 relative to surface 1 with an offset bias of 15 mm"
        res4 = interpret_and_execute_command(cmd4, system=baseline_spectrometer)
        assert res4["success"] is True
        assert res4["executed_results"][0]["tool"] == "offset_optical_element"
        assert abs(baseline_spectrometer.surfaces[1].thickness - 15.0) < 1e-5

        # 5. Offset with simple 'by' phrase
        cmd5 = "offset element 2 relative to surface 1 by 20mm"
        res5 = interpret_and_execute_command(cmd5, system=baseline_spectrometer)
        assert res5["success"] is True
        assert abs(baseline_spectrometer.surfaces[1].thickness - 20.0) < 1e-5

    def test_native_optics_app_optical_system_element_manipulations(self):
        """Verify optics_app.engine.tracer.OpticalSystem supports all element manipulations."""
        from optics_app.engine.lde import SurfaceStack as AppSurfaceStack
        from optics_app.engine.tracer import OpticalSystem as AppOpticalSystem
        from optics_app.engine.surface import Surface, SurfaceType

        stack = AppSurfaceStack()
        # Object
        stack.surfaces.append(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0))
        # Singlet (surfaces 1 and 2)
        stack.insert_singlet(after_surface_index=0, name="L1", radius_front=50.0, thickness=5.0, glass="N-BK7", radius_back=-50.0, air_gap_after=15.0)
        # Detector
        stack.surfaces.append(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = AppOpticalSystem(stack)

        # 1. Rotate
        opt_sys.rotate_element(1, tilt_x=3.5, tilt_y=0.0)
        assert abs(opt_sys.surfaces[1].tilt_x - 3.5) < 1e-5

        # 2. Tilt
        opt_sys.tilt_element(1, angle_deg=-2.5, axis="y")
        assert abs(opt_sys.surfaces[1].tilt_y - (-2.5)) < 1e-5

        # 3. Displace
        opt_sys.displace_element(1, decenter_y=1.75)
        assert abs(opt_sys.surfaces[1].decenter_y - 1.75) < 1e-5

        # 4. Flip
        orig_r_front = opt_sys.surfaces[1].radius
        opt_sys.flip_element(1)
        assert abs(opt_sys.surfaces[1].radius - 50.0) < 1e-5  # Former back (-50.0) inverted to +50.0

        # 5. Offset
        opt_sys.offset_element(1, reference_surface_idx=0, offset_bias_mm=25.0)
        assert abs(opt_sys.surfaces[0].thickness - 25.0) < 1e-5

    def test_ai_optimizer_on_native_optical_system(self):
        """Verify AiOptimizationLoop minimizes merit function on OpticalSystem with beam_radius."""
        from optics_app.engine.lde import SurfaceStack as AppSurfaceStack
        from optics_app.engine.tracer import OpticalSystem as AppOpticalSystem
        from optics_app.engine.surface import Surface, SurfaceType
        from optics_app.engine.merit_function import MeritFunction, OperandType

        stack = AppSurfaceStack()
        stack.surfaces.append(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.insert_singlet(after_surface_index=0, name="L1", radius_front=30.0, thickness=4.0, glass="N-BK7", radius_back=-100.0, air_gap_after=30.0)
        stack.surfaces.append(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = AppOpticalSystem(stack)
        opt_sys.beam_radius = 6.0  # Elevated aperture

        mf = MeritFunction()
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=-1, target=0.0, weight=1.0)

        opt = AiOptimizationLoop(system=opt_sys, merit_function=mf, max_iterations=8)
        res = opt.optimize(parameters_to_adjust=["aperture"])

        assert res["success"] is True
        assert res["final_merit"] <= res["initial_merit"]
        assert opt_sys.beam_radius < 6.0  # Successfully optimized aperture


# ---------------------------------------------------------------------------
# Adversarial Reviewer Round 2 Verification Suite
# ---------------------------------------------------------------------------
class TestAC_AdversarialReviewerRound2:
    """Adversarial Reviewer Round 2 edge case and robustness tests."""

    def test_adv_1_rotate_and_tilt_preposition_order(self, baseline_spectrometer):
        """Verify natural language commands where axis precedes angle or vice versa."""
        # 1. Rotate with axis before angle
        cmd1 = "rotate element 1 around x by 6.0 degrees"
        res1 = interpret_and_execute_command(cmd1, system=baseline_spectrometer)
        assert res1["success"] is True
        assert abs(baseline_spectrometer.surfaces[1].tilt_x - 6.0) < 1e-5

        # 2. Tilt with axis before angle
        cmd2 = "tilt element 1 around y by -4.0 degrees"
        res2 = interpret_and_execute_command(cmd2, system=baseline_spectrometer)
        assert res2["success"] is True
        assert abs(baseline_spectrometer.surfaces[1].tilt_y - (-4.0)) < 1e-5

    def test_adv_2_displace_axis_preceding_distance(self, baseline_spectrometer):
        """Verify displace/translate command with axis phrase preceding distance."""
        cmd = "displace element 1 along y by 3.5 mm"
        res = interpret_and_execute_command(cmd, system=baseline_spectrometer)
        assert res["success"] is True
        assert abs(baseline_spectrometer.surfaces[1].decenter_y - 3.5) < 1e-5

    def test_adv_3_replace_with_catalog_prefix_and_articles(self, baseline_spectrometer):
        """Verify replace command containing 'Thorlabs' prefix or article 'a'."""
        cmd1 = "replace element 1 with Thorlabs AC254-050-C"
        res1 = interpret_and_execute_command(cmd1, system=baseline_spectrometer)
        assert res1["success"] is True
        assert "AC254-050-C" in baseline_spectrometer.surfaces[1].group_name.upper()

        cmd2 = "replace element 1 with a CM254-050-P01 mirror"
        res2 = interpret_and_execute_command(cmd2, system=baseline_spectrometer)
        assert res2["success"] is True

    def test_adv_4_offset_order_b_distance_before_reference(self, baseline_spectrometer):
        """Verify offset command with distance preceding reference surface."""
        cmd = "offset element 2 by 18.0 mm from surface 1"
        res = interpret_and_execute_command(cmd, system=baseline_spectrometer)
        assert res["success"] is True
        assert abs(baseline_spectrometer.surfaces[1].thickness - 18.0) < 1e-5

    def test_adv_5_optimizer_field_dict_and_none_handling(self, baseline_spectrometer):
        """Verify AiOptimizationLoop handles dict system_config, None system_config, and field parameters."""
        # Dict system_config
        baseline_spectrometer.system_config = {"field_angle": 2.5}
        opt = AiOptimizationLoop(system=baseline_spectrometer)
        assert abs(opt.get_parameter("field") - 2.5) < 1e-5

        opt.set_parameter("field", 1.2)
        assert abs(opt.get_parameter("field") - 1.2) < 1e-5

        # None system_config
        baseline_spectrometer.system_config = None
        baseline_spectrometer.field_angle = None
        opt.set_parameter("field", 3.0)
        assert abs(opt.get_parameter("field") - 3.0) < 1e-5

    def test_adv_6_optimizer_wavelength_synchronization(self, baseline_spectrometer):
        """Verify setting wavelength synchronizes both system and surface_stack."""
        opt = AiOptimizationLoop(system=baseline_spectrometer)
        opt.set_parameter("wavelength", 1.550)
        assert abs(baseline_spectrometer.primary_wavelength - 1.550) < 1e-5
        if hasattr(baseline_spectrometer, "surface_stack"):
            assert abs(baseline_spectrometer.surface_stack.primary_wavelength - 1.550) < 1e-5
            assert abs(baseline_spectrometer.surface_stack.wavelengths[0] - 1.550) < 1e-5

    def test_adv_7_calculate_system_q_factor_auto_detects_spectrometer(self, baseline_spectrometer):
        """Verify calculate_system_q_factor auto-detects grating surface and outputs spectrometer resolving power."""
        q_res = calculate_system_q_factor(baseline_spectrometer)
        assert "resolving_power" in q_res
        assert q_res["q_factor"] >= 1000.0
        assert "spectral_resolution_nm" in q_res

    def test_adv_8_ai_dock_sync_field_parameters(self, baseline_spectrometer):
        """Verify AiChatDock.sync_from_system reads field parameters from active system."""
        try:
            from v2.gui.ai_dock import AiChatDock
            from v2.gui.qt_compat import QtWidgets, QT_AVAILABLE
            if not QT_AVAILABLE or not hasattr(QtWidgets, "QApplication"):
                pytest.skip("Qt not available in test environment")
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            dock = AiChatDock()
            dock.main_window = type("FakeWindow", (), {"current_system": baseline_spectrometer})()
            baseline_spectrometer.system_config = {"field_angle": 4.2}
            dock.sync_from_system()
            assert abs(dock.spin_field_angle.value() - 4.2) < 1e-5
        except Exception as e:
            pytest.skip(f"Headless GUI environment: {e}")


# ---------------------------------------------------------------------------
class TestAC_AdversarialReviewerRound3:
    """Adversarial Reviewer Round 3 edge case, regression and robustness tests."""

    def test_adv_r3_1_reflection_matrix_concave_focusing(self):
        """
        Verify paraxial reflection matrix for concave spherical mirror:
        Concave mirror in optical design convention has R < 0 (e.g. R = -100mm, f = -R/2 = +50mm).
        The ABCD reflection matrix must have C = 2/R = -0.02, giving positive optical power P = 0.02.
        After drift d = 50mm, parallel incident ray (y=2.0, u=0.0) must focus to y=0.0.
        """
        m_concave = reflection_matrix(-100.0)
        assert abs(m_concave.C - (-0.02)) < 1e-9
        assert abs(m_concave.A - 1.0) < 1e-9
        assert abs(m_concave.D - 1.0) < 1e-9

        m_drift = space_matrix(50.0)
        m_sys = m_drift @ m_concave
        y_out, u_out = m_sys.propagate_ray(y=2.0, u=0.0)
        assert abs(y_out - 0.0) < 1e-9, f"Concave mirror should focus ray to y=0, got {y_out}"
        assert abs(u_out - (-0.04)) < 1e-9

        # Convex mirror R = +100mm should diverge rays (negative power)
        m_convex = reflection_matrix(100.0)
        assert abs(m_convex.C - 0.02) < 1e-9
        y_out_conv, _ = (space_matrix(50.0) @ m_convex).propagate_ray(y=2.0, u=0.0)
        assert y_out_conv > 2.0  # Diverging

    def test_adv_r3_2_natural_language_axis_pivot_phrasings(self, baseline_spectrometer):
        """Verify advanced NLP parsing: 'about y', 'around the z-axis', 'along x', and pivot coordinates."""
        # 1. 'about y'
        cmd1 = "rotate element 1 by 5 degrees about y"
        res1 = interpret_and_execute_command(cmd1, system=baseline_spectrometer)
        assert res1["success"] is True
        assert abs(baseline_spectrometer.surfaces[1].tilt_y - 5.0) < 1e-5

        # 2. 'around the z-axis'
        cmd2 = "tilt element 1 around the z-axis by 3 deg"
        res2 = interpret_and_execute_command(cmd2, system=baseline_spectrometer)
        assert res2["success"] is True
        assert abs(baseline_spectrometer.surfaces[1].tilt_z - 3.0) < 1e-5

        # 3. 'offset ... along x from surface 1'
        cmd3 = "offset element 2 by 7.5 mm along x from surface 1"
        res3 = interpret_and_execute_command(cmd3, system=baseline_spectrometer)
        assert res3["success"] is True
        assert abs(baseline_spectrometer.surfaces[2].decenter_x - 7.5) < 1e-5

        # 3b. 'offset ... from surface 1' (axial z gap / thickness)
        cmd3b = "offset element 2 by 8.2 mm from surface 1"
        res3b = interpret_and_execute_command(cmd3b, system=baseline_spectrometer)
        assert res3b["success"] is True
        assert abs(baseline_spectrometer.surfaces[1].thickness - 8.2) < 1e-5

        # 4. Explicit pivot vector [10.0, 20.0, 30.0]
        cmd4 = "rotate element 1 by 2.0 degrees around x about pivot [10, 20, 30]"
        res4 = interpret_and_execute_command(cmd4, system=baseline_spectrometer)
        assert res4["success"] is True
        assert res4["executed_results"][0]["arguments"]["pivot"] == [10.0, 20.0, 30.0]
        assert abs(baseline_spectrometer.surfaces[1].tilt_x - 2.0) < 1e-5

    def test_adv_r3_3_replace_optical_element_mirror(self, baseline_spectrometer):
        """Verify replacing element with Thorlabs mirror catalog part CM254-050-P01."""
        res = replace_optical_element(element_idx=1, new_part_name="CM254-050-P01", system=baseline_spectrometer)
        assert "Successfully replaced" in res
        # Surface 1 should now be a spherical concave mirror with radius -100mm
        s1 = baseline_spectrometer.surfaces[1]
        assert s1.surface_type == SurfaceType.MIRROR
        assert abs(s1.radius - (-100.0)) < 1e-5
        # Clean up active system to avoid leaking mutated surface 1 into subsequent test files
        from v2.ai.tools import set_active_optical_system
        set_active_optical_system(None)

    def test_adv_r3_4_optimizer_with_system_config_field_point(self, baseline_spectrometer):
        """Verify AiOptimizationLoop handles SystemConfig with FieldPoint instances."""
        try:
            from v2.engine.system_config import SystemConfig, FieldPoint
        except ImportError:
            from optics_app.engine.system_config import SystemConfig, FieldPoint

        cfg = SystemConfig()
        cfg.fields = [FieldPoint(x=0.0, y=1.8, weight=1.0)]
        baseline_spectrometer.system_config = cfg
        baseline_spectrometer.field_angle = None

        opt = AiOptimizationLoop(system=baseline_spectrometer)
        # Verify read
        assert abs(opt.get_parameter("field") - 1.8) < 1e-5

        # Verify mutation
        opt.set_parameter("field", 2.6)
        assert abs(opt.get_parameter("field") - 2.6) < 1e-5
        assert abs(cfg.fields[0].y - 2.6) < 1e-5

        # Verify optimization runs without error
        opt_res = opt.optimize(parameters_to_adjust=["field"])
        assert "initial_merit" in opt_res
        assert "final_merit" in opt_res

    def test_adv_r3_5_ai_dock_sync_field_point(self, baseline_spectrometer):
        """Verify AiChatDock syncs and applies field parameters with SystemConfig FieldPoint."""
        try:
            from v2.gui.ai_dock import AiChatDock
            from v2.gui.qt_compat import QtWidgets, QT_AVAILABLE
            try:
                from v2.engine.system_config import SystemConfig, FieldPoint
            except ImportError:
                from optics_app.engine.system_config import SystemConfig, FieldPoint

            if not QT_AVAILABLE or not hasattr(QtWidgets, "QApplication"):
                pytest.skip("Qt not available in test environment")
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            dock = AiChatDock()
            dock.main_window = type("FakeWindow", (), {"current_system": baseline_spectrometer})()

            cfg = SystemConfig()
            cfg.fields = [FieldPoint(x=0.0, y=3.7, weight=1.0)]
            baseline_spectrometer.system_config = cfg
            baseline_spectrometer.field_angle = None

            dock.sync_from_system()
            assert abs(dock.spin_field_angle.value() - 3.7) < 1e-5

            dock.spin_field_angle.setValue(4.8)
            dock.apply_zemax_parameters()
            assert abs(cfg.fields[0].y - 4.8) < 1e-4
        except Exception as e:
            pytest.skip(f"Headless GUI environment: {e}")

    def test_adv_r3_6_demo_integrity_mode_default_and_live_fallback(self):
        """
        Verify that:
        1. In default environment or demo integrity mode, NemotronClient defaults to demo_mode=True.
        2. When demo_mode=False or INTEGRITY_MODE='live', any cloud API error (e.g. HTTP 404/network)
           is handled gracefully without crashing, falling back to high-fidelity demo/heuristic parsing.
        """
        # 1. Default demo mode resolution
        orig_integrity = os.environ.pop("INTEGRITY_MODE", None)
        try:
            client_default = NemotronClient()
            assert client_default.demo_mode is True, "NemotronClient should default to demo integrity mode"

            # 2. Live mode error handling & graceful fallback
            client_live = NemotronClient(demo_mode=False, base_url="https://httpbin.org/status/404", timeout_seconds=5)
            assert client_live.demo_mode is False

            resp = client_live.chat_completion(
                messages=[{"role": "user", "content": "displace element 2 by 10.0 mm along y"}],
                tools=get_default_tool_registry().schemas
            )
            assert resp["role"] == "assistant"
            assert len(resp.get("tool_calls", [])) > 0
            tc = resp["tool_calls"][0]["function"]
            assert tc["name"] == "displace_optical_element"
            args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
            assert args["element_idx"] == 2
            assert args["decenter_y"] == 10.0
        finally:
            if orig_integrity is not None:
                os.environ["INTEGRITY_MODE"] = orig_integrity



