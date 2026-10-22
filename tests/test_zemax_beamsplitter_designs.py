"""
Test Zemax Beamsplitter Reference Architectures, Presets, and AI Tools.
"""

import os
import json
import math
import pytest
import numpy as np

from optics_app.engine.zemax_beamsplitter_designer import (
    build_zemax_michelson_interferometer,
    build_zemax_mach_zehnder_interferometer,
    build_zemax_pbs_optical_isolator,
    build_zemax_beam_sampler,
    export_beamsplitter_design_to_zmx,
)
from v2.engine.zemax_beamsplitter_designer import (
    build_zemax_michelson_interferometer as build_michelson_v2,
    build_zemax_mach_zehnder_interferometer as build_mach_zehnder_v2,
    build_zemax_pbs_optical_isolator as build_pbs_isolator_v2,
    build_zemax_beam_sampler as build_beam_sampler_v2,
)
from optics_app.engine.lde import SurfaceStack
from optics_app.ai.tools import get_default_tool_registry
from optics_app.ai.nemotron_client import parse_optical_command_heuristics


def test_zemax_michelson_interferometer_physics_and_zmx():
    """Verify Michelson Interferometer design calculations and Zemax export."""
    # Test zero displacement (constructive interference)
    m0 = build_zemax_michelson_interferometer(arm_length_mm=100.0, d_delta_um=0.0, wavelength_um=0.6328)
    assert m0.opd_um == 0.0
    assert m0.phase_difference_deg == 0.0
    assert m0.fringe_visibility == 1.0
    assert m0.interference_intensity > 0.90
    assert len(m0.system.surfaces) >= 10
    assert "VERS" in m0.zmx_text
    assert "COORDBRK" in m0.zmx_text
    assert "GLAS N-BK7" in m0.zmx_text

    # Test half-wave OPD: delta_d = lambda / 4 = 0.1582 um -> OPD = 2 * delta_d = lambda / 2 = 0.3164 um -> phase diff = 180 deg (destructive null)
    m_half = build_zemax_michelson_interferometer(arm_length_mm=100.0, d_delta_um=0.1582, wavelength_um=0.6328)
    assert abs(m_half.opd_um - 0.3164) < 1e-4
    assert abs(m_half.phase_difference_deg - 180.0) < 0.1
    assert abs(m_half.interference_intensity) < 1e-4  # Complete destructive interference null!
    assert m_half.fringe_visibility == 1.0


def test_zemax_mach_zehnder_interferometer_physics_and_zmx():
    """Verify Mach-Zehnder complementary dual-port interference and energy conservation."""
    # phi = 0: Port 1 = 1.0, Port 2 = 0.0
    mz0 = build_zemax_mach_zehnder_interferometer(phase_shift_deg=0.0)
    assert abs(mz0.port1_intensity - 1.0) < 1e-6
    assert abs(mz0.port2_intensity - 0.0) < 1e-6
    assert abs(mz0.total_transmitted_intensity - 1.0) < 1e-6

    # phi = 90 deg: Port 1 = 0.5, Port 2 = 0.5
    mz90 = build_zemax_mach_zehnder_interferometer(phase_shift_deg=90.0)
    assert abs(mz90.port1_intensity - 0.5) < 1e-6
    assert abs(mz90.port2_intensity - 0.5) < 1e-6
    assert abs(mz90.total_transmitted_intensity - 1.0) < 1e-6

    # phi = 180 deg: Port 1 = 0.0, Port 2 = 1.0
    mz180 = build_zemax_mach_zehnder_interferometer(phase_shift_deg=180.0)
    assert abs(mz180.port1_intensity - 0.0) < 1e-6
    assert abs(mz180.port2_intensity - 1.0) < 1e-6
    assert abs(mz180.total_transmitted_intensity - 1.0) < 1e-6


def test_zemax_pbs_optical_isolator_physics_and_zmx():
    """Verify PBS Optical Isolator polarization conversion and high isolation rejection."""
    iso = build_zemax_pbs_optical_isolator(pbs_part="PBS251", wavelength_um=0.6328)
    assert iso.forward_transmitted_power >= 0.89  # Tp >= 89%
    assert iso.return_polarization == "Converted S-Polarization"
    assert iso.dump_rejected_power >= 0.85  # Dump receives majority of back-reflection
    assert iso.laser_leakage_power <= 0.01  # Tiny laser back-reflection
    assert iso.optical_isolation_db >= 20.0  # High optical isolation
    assert "VERS" in iso.zmx_text
    assert "MIRROR" in iso.zmx_text


def test_zemax_beam_sampler_pickoff():
    """Verify calibrated laser beam sampler power split and insertion loss."""
    samp = build_zemax_beam_sampler(bs_part="BS019", input_power_mw=100.0)
    assert abs(samp.transmitted_power_mw + samp.sampled_power_mw - 100.0) < 1e-6
    assert samp.pick_off_ratio_percent > 0.0
    assert samp.insertion_loss_db > 0.0


def test_shipped_json_presets_loading():
    """Verify all 4 shipped beamsplitter JSON presets in optics_app/ and v2/ load cleanly."""
    preset_names = [
        "thorlabs_michelson_interferometer.json",
        "thorlabs_mach_zehnder_interferometer.json",
        "thorlabs_pbs_optical_isolator.json",
        "thorlabs_beam_sampler.json",
    ]

    for base_dir in ["optics_app/catalogs/presets", "v2/catalogs/presets"]:
        for name in preset_names:
            p = os.path.join(base_dir, name)
            assert os.path.exists(p), f"Preset file {p} does not exist"
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            stack = SurfaceStack.from_dict(data)
            assert len(stack.surfaces) >= 5, f"Too few surfaces in preset {p}"
            assert any(s.is_beam_splitter for s in stack.surfaces), f"No beamsplitter surface in {p}"


def test_ai_tools_registry_beamsplitter_executions():
    """Verify registered AI tools execute correctly and return informative strings."""
    reg = get_default_tool_registry()

    # 1. analyze_beamsplitter_polarization
    res_pol = reg.execute("analyze_beamsplitter_polarization", {
        "part_number": "PBS251",
        "input_polarization": "P",
        "wavelength_um": 0.6328
    })
    assert "Polarizing Beam Splitter" in res_pol
    assert "TRANSMITTED BRANCH" in res_pol
    assert "Energy Conservation" in res_pol

    # 2. create_zemax_interferometer_design
    res_michelson = reg.execute("create_zemax_interferometer_design", {
        "design_type": "michelson",
        "d_delta_um": 0.1582
    })
    assert "Michelson Interferometer Design Report" in res_michelson
    assert "OPD" in res_michelson

    # 3. insert_thorlabs_beamsplitter on a fresh stack
    stack = SurfaceStack()
    stack.create_clean_system()
    from optics_app.ai.tools import set_active_optical_system
    set_active_optical_system(stack)

    res_ins = reg.execute("insert_thorlabs_beamsplitter", {
        "part_number": "BS016",
        "surface_index": 1
    })
    assert "Successfully inserted" in res_ins
    assert "BS016" in res_ins


def test_nemotron_command_heuristics_parsing():
    """Verify natural language heuristics parse beamsplitter and interferometer commands."""
    res_m = parse_optical_command_heuristics("build a michelson interferometer with 0.15 um displacement")
    assert res_m is not None
    assert res_m["name"] == "create_zemax_interferometer_design"
    assert res_m["arguments"]["design_type"] == "michelson"
    assert abs(res_m["arguments"]["d_delta_um"] - 0.15) < 1e-4

    res_mz = parse_optical_command_heuristics("design a mach-zehnder interferometer with 45 deg phase shift")
    assert res_mz is not None
    assert res_mz["name"] == "create_zemax_interferometer_design"
    assert res_mz["arguments"]["design_type"] == "mach_zehnder"
    assert abs(res_mz["arguments"]["phase_shift_deg"] - 45.0) < 1e-4

    res_pol = parse_optical_command_heuristics("analyze polarization split for PBS251 with s-pol")
    assert res_pol is not None
    assert res_pol["name"] == "analyze_beamsplitter_polarization"
    assert res_pol["arguments"]["part_number"] == "PBS251"
    assert res_pol["arguments"]["input_polarization"] == "S"


def test_parity_optics_app_and_v2_zemax_designers():
    """Verify exact parity between optics_app and v2 zemax beamsplitter designer implementations."""
    m1 = build_zemax_michelson_interferometer(d_delta_um=0.10)
    m2 = build_michelson_v2(d_delta_um=0.10)
    assert abs(m1.opd_um - m2.opd_um) < 1e-6
    assert abs(m1.interference_intensity - m2.interference_intensity) < 1e-6

    mz1 = build_zemax_mach_zehnder_interferometer(phase_shift_deg=60.0)
    mz2 = build_mach_zehnder_v2(phase_shift_deg=60.0)
    assert abs(mz1.port1_intensity - mz2.port1_intensity) < 1e-6
    assert abs(mz1.port2_intensity - mz2.port2_intensity) < 1e-6

    iso1 = build_zemax_pbs_optical_isolator()
    iso2 = build_pbs_isolator_v2()
    assert abs(iso1.optical_isolation_db - iso2.optical_isolation_db) < 1e-6

    samp1 = build_zemax_beam_sampler()
    samp2 = build_beam_sampler_v2()
    assert abs(samp1.transmitted_power_mw - samp2.transmitted_power_mw) < 1e-6
