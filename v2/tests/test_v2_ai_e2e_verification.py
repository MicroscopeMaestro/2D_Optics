"""
Pytest integration for V2 AI Capabilities E2E Verification.

Runs the comprehensive End-to-End verification against Nemotron/Gemma APIs,
Zemax file I/O, rigid component manipulation, and merit function optimization loop.
"""

import pytest
import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import verify_v2_ai_capabilities as v2_verify


class TestV2AiCapabilitiesE2E:
    """End-to-End verification test suite for V2 AI capabilities."""

    def test_e2e_r1_live_api_querying(self):
        """Verify R1: Live API querying and authentication with Nemotron and Gemma."""
        res = v2_verify.test_r1_live_api_querying_and_authentication()
        assert "nemotron" in res
        assert "gemma" in res
        assert res["nemotron"]["tool_call"]["name"] == "displace_optical_element"
        assert res["nemotron"]["arguments"]["element_idx"] == 2
        assert res["nemotron"]["arguments"]["decenter_y"] == 10.0
        assert res["gemma"]["tool_call"]["name"] == "run_ai_optimization"

    def test_e2e_r2_zemax_file_io_lossless(self):
        """Verify R2: AI triggered Zemax export and import preserving exact surface count."""
        res = v2_verify.test_r2_zemax_file_io_verification()
        assert res["roundtrip_lossless"] is True
        assert res["initial_surface_count"] == res["reimported_surface_count"]

    def test_e2e_r3_component_manipulation(self):
        """Verify R3: AI triggered rigid optical component movement and replacement."""
        res = v2_verify.test_r3_component_manipulation_verification()
        assert abs(res["final_decenter_y"] - 10.0) < 1e-5
        assert abs(res["replaced_element_radius"] - (-100.0)) < 1e-5
        assert abs(res["replaced_mirror_radius"] - (-150.0)) < 1e-5
        assert abs(res["rotated_tilt_x"] - 4.5) < 1e-5
        assert abs(res["mirror_tilt_y"] - (-5.0)) < 1e-5

    def test_e2e_r4_ai_optimization_loop(self):
        """Verify R4: AI optimization loop minimizes merit function below initial value."""
        res = v2_verify.test_r4_ai_optimization_loop_verification()
        assert res["final_merit"] < res["initial_merit"]
        assert res["merit_reduction"] > 0.0

    def test_e2e_full_suite_execution(self):
        """Verify Acceptance Criteria 1: Full verification suite executes to completion."""
        success = v2_verify.run_all_verifications()
        assert success is True
