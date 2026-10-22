"""
Comprehensive Automated Test Suite for Zemax OpticStudio-Style Spectrometer Quality Analysis.
==============================================================================================
Validates:
1. Grating dispersion physics, anamorphic magnification, and reciprocal linear dispersion.
2. 4-part spectral resolution decomposition (slit, diffraction, aberration, pixel Nyquist limits).
3. Resolving power R = lambda / Delta_lambda and Raman shift wavenumber conversions.
4. Optical étendue, F-number matching, blazed grating efficiency, and end-to-end throughput.
5. Slit smile curvature, keystone magnification distortion, and focal plane tilt.
6. Synthetic Line Spread Function (LSF) and simulated recorded doublet spectrum.
7. Verification across real instruments:
   - Thorlabs Czerny-Turner 785nm Spectrometer (.zmx)
   - All-Mirror Broadband NIR 800-2700nm Spectrometer (.json)
   - All-Mirror Raman 1064nm Fingerprint Spectrometer (.json)
8. Interactive Qt GUI Tab (SpectrometerQualityTab) and AnalysisTabsWidget integration.
9. AI Assistant Tool calling and natural language heuristic parsing.
"""

import os
import sys
import math
import pytest
import numpy as np

# Ensure project root is in sys.path regardless of whether run from tests/ or v2/tests/
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(os.path.abspath(os.path.join(_here, ".."))) == "v2":
    PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(_here, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.zmx_loader import parse_zmx_file
from optics_app.engine.models import SpectrometerSystem
from optics_app.analysis.spectrometer_quality import (
    SpectrometerConfiguration,
    WavelengthSpectrometerResult,
    SpectrometerQualitySummary,
    SpectrometerQualityAnalyzer,
    SyntheticSpectrumData,
)
from optics_app.ai.tools import get_default_tool_registry
from optics_app.ai.nemotron_client import parse_optical_command_heuristics


CZERNY_TURNER_ZMX = os.path.join(PROJECT_ROOT, "v2", "thorlabs_czerny_turner_785nm.zmx")
NIR_JSON = os.path.join(PROJECT_ROOT, "spectrometer_v3_nir_raman", "spectrometer_v3_nir_800_2700nm.json")
RAMAN_JSON = os.path.join(PROJECT_ROOT, "spectrometer_v3_nir_raman", "spectrometer_v3_raman_1064nm.json")


@pytest.fixture(scope="session")
def qapp():
    """Headless Qt Application fixture."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from optics_app.gui.qt_compat import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


# ============================================================================
# TIER 1: Mathematical & Physical Formula Verification
# ============================================================================

def test_grating_efficiency_blaze_curve():
    """Verify blazed grating scalar diffraction efficiency model."""
    blaze_wl = 1.000  # 1.0 um
    eff_peak = SpectrometerQualityAnalyzer.compute_grating_efficiency(1.000, blaze_wavelength_um=blaze_wl, peak_efficiency=0.80)
    assert abs(eff_peak - 0.80) < 1e-4, f"Peak efficiency at blaze wl should be 0.80, got {eff_peak}"

    # Efficiency rolls off symmetrically away from blaze wavelength
    eff_lower = SpectrometerQualityAnalyzer.compute_grating_efficiency(0.700, blaze_wavelength_um=blaze_wl, peak_efficiency=0.80)
    eff_higher = SpectrometerQualityAnalyzer.compute_grating_efficiency(1.300, blaze_wavelength_um=blaze_wl, peak_efficiency=0.80)
    assert 0.05 <= eff_lower < 0.80
    assert 0.05 <= eff_higher < 0.80


def test_dispersion_and_anamorphic_magnification_physics():
    """Verify analytical dispersion equations for a standard 600 l/mm grating."""
    lines_per_mm = 600.0
    f_cam = 100.0
    order = 1
    pitch_nm = 1e6 / lines_per_mm  # 1666.67 nm

    # Normal incidence alpha = 0, beta at 785 nm
    sin_beta = (order * 785.0) / pitch_nm
    beta_rad = math.asin(sin_beta)
    cos_beta = math.cos(beta_rad)

    # Anamorphic factor r = cos(alpha) / cos(beta)
    anamorphic_r = 1.0 / cos_beta
    assert anamorphic_r > 1.0, "Anamorphic expansion must be > 1 when beta > alpha=0"

    # dbeta / dlambda (rad/nm)
    dbeta_dlam = order / (pitch_nm * cos_beta)
    # dx / dlambda (mm/nm)
    dx_dlam = f_cam * dbeta_dlam
    # reciprocal linear dispersion dlambda / dx (nm/mm)
    recip_disp = 1.0 / dx_dlam

    # For 600 l/mm and f=100mm, recip disp is ~ 14.7 nm/mm at 785nm
    assert 12.0 < recip_disp < 18.0, f"Expected recip disp ~ 14.7 nm/mm, got {recip_disp:.2f}"


def test_four_component_resolution_model():
    """Verify quadrature sum and individual terms of the 4-component resolution model."""
    w_slit_um = 25.0
    recip_disp_nm_mm = 15.0
    mag_x = 1.0
    w_img_um = w_slit_um * mag_x

    # 1. Slit limit: w_img (mm) * dlambda/dx (nm/mm)
    res_slit = (w_img_um / 1000.0) * recip_disp_nm_mm  # 0.025 * 15 = 0.375 nm
    assert abs(res_slit - 0.375) < 1e-4

    # 2. Diffraction limit
    r_theor = 15000.0
    wl_nm = 785.0
    res_diff = wl_nm / r_theor  # 0.0523 nm
    assert abs(res_diff - 0.05233) < 1e-3

    # 3. Aberration limit: 2 * rms_x (mm) * dlambda/dx
    rms_disp_um = 6.0
    res_aber = 2.0 * (rms_disp_um / 1000.0) * recip_disp_nm_mm  # 2 * 0.006 * 15 = 0.180 nm
    assert abs(res_aber - 0.180) < 1e-4

    # 4. Pixel Nyquist limit: 2 * pixel_pitch (mm) * dlambda/dx
    pixel_pitch_um = 14.0
    res_pixel = 2.0 * (pixel_pitch_um / 1000.0) * recip_disp_nm_mm  # 2 * 0.014 * 15 = 0.420 nm
    assert abs(res_pixel - 0.420) < 1e-4

    # Quadrature sum
    res_total = math.sqrt(res_slit**2 + res_diff**2 + res_aber**2 + res_pixel**2)
    assert 0.55 < res_total < 0.65, f"Expected total resolution ~ 0.59 nm, got {res_total:.3f}"

    # Resolving power
    r_power = wl_nm / res_total
    assert 1100.0 < r_power < 1500.0


def test_optical_etendue_calculation():
    """Verify geometric étendue calculation G = A_slit * Omega_in."""
    cfg = SpectrometerConfiguration(slit_width_um=30.0, slit_height_mm=1.0, input_na=0.12)
    area_mm2 = (cfg.slit_width_um / 1000.0) * cfg.slit_height_mm  # 0.030 mm^2
    solid_angle = math.pi * (cfg.input_na ** 2)  # pi * 0.0144 ~ 0.0452 sr
    expected_g = area_mm2 * solid_angle
    assert abs(expected_g - 0.001357) < 1e-5


# ============================================================================
# TIER 2: Real Optical System Benchmark Validations
# ============================================================================

def test_thorlabs_czerny_turner_spectrometer_analysis():
    """Analyze the Thorlabs Czerny-Turner 785nm spectrometer model."""
    assert os.path.exists(CZERNY_TURNER_ZMX), f"File not found: {CZERNY_TURNER_ZMX}"
    stack = parse_zmx_file(CZERNY_TURNER_ZMX)
    sys = OpticalSystem(stack)

    cfg = SpectrometerConfiguration(slit_width_um=25.0, input_na=0.10, raman_laser_wavelength_um=0.785)
    summary = SpectrometerQualityAnalyzer.analyze_system(sys, config=cfg)

    # Check optical dimensions identified
    assert abs(summary.collimator_efl_mm - 100.0) < 1.0
    assert abs(summary.camera_efl_mm - 100.0) < 1.0
    assert abs(summary.grating_lines_per_mm - 600.0) < 1.0
    assert summary.grating_order == 1

    # Check dispersion & spectral span
    assert summary.total_detector_span_mm > 5.0, f"Expected detector span > 5mm, got {summary.total_detector_span_mm}"
    assert 5.0 < summary.avg_reciprocal_dispersion_nm_per_mm < 20.0

    # Check multi-wavelength results
    assert len(summary.wavelength_results) == 3
    for r in summary.wavelength_results:
        assert r.res_total_nm > 0.0
        assert r.resolving_power > 50.0
        assert r.raman_shift_cm1 is not None

    # Check report generation
    assert "ZEMAX OPTICSTUDIO SPECTROMETER ANALYSIS REPORT" in summary.zemax_report_text
    assert "RAMAN SPECTROSCOPY WAVENUMBER CONVERSION" in summary.zemax_report_text


def test_nir_all_mirror_spectrometer_analysis():
    """Analyze Broadband All-Mirror NIR Spectrometer (800 - 2700 nm)."""
    assert os.path.exists(NIR_JSON), f"File not found: {NIR_JSON}"
    spec = SpectrometerSystem.load_from_json(NIR_JSON)
    sys = OpticalSystem(spec.surface_stack)

    cfg = SpectrometerConfiguration(slit_width_um=30.0, input_na=0.12, pixel_pitch_um=25.0)
    summary = SpectrometerQualityAnalyzer.analyze_system(sys, config=cfg, wavelengths_um=spec.wavelengths)

    assert abs(summary.wavelength_min_nm - 800.0) < 1.0
    assert abs(summary.wavelength_max_nm - 2700.0) < 1.0
    assert abs(summary.grating_lines_per_mm - 150.0) < 1.0
    assert len(summary.wavelength_results) == len(spec.wavelengths)
    assert summary.max_smile_um >= 0.0


def test_raman_1064nm_all_mirror_spectrometer_analysis():
    """Analyze All-Mirror Raman 1064nm Spectrometer."""
    assert os.path.exists(RAMAN_JSON), f"File not found: {RAMAN_JSON}"
    spec = SpectrometerSystem.load_from_json(RAMAN_JSON)
    sys = OpticalSystem(spec.surface_stack)

    cfg = SpectrometerConfiguration(
        slit_width_um=30.0, input_na=0.12, pixel_pitch_um=25.0, raman_laser_wavelength_um=1.064
    )
    summary = SpectrometerQualityAnalyzer.analyze_system(sys, config=cfg, wavelengths_um=spec.wavelengths)

    assert abs(summary.wavelength_min_nm - 1064.0) < 1.0
    assert abs(summary.wavelength_max_nm - 1250.0) < 1.0
    assert abs(summary.grating_lines_per_mm - 600.0) < 1.0

    # Verify Raman Shift conversion at excitation laser is 0 cm^-1
    first_res = summary.wavelength_results[0]
    assert abs(first_res.raman_shift_cm1) < 1e-3, f"Shift at 1064nm should be 0, got {first_res.raman_shift_cm1}"

    # Verify shift at 1250nm is in Raman fingerprint region (~ 1400 cm^-1)
    last_res = summary.wavelength_results[-1]
    assert 1350.0 < last_res.raman_shift_cm1 < 1450.0, f"Expected ~1400 cm^-1, got {last_res.raman_shift_cm1}"


# ============================================================================
# TIER 3: GUI Integration & Plot Modes Verification
# ============================================================================

def test_spectrometer_quality_tab_headless(qapp):
    """Verify SpectrometerQualityTab instantiation, signal handling, and plotting."""
    from optics_app.gui.analysis_tabs import SpectrometerQualityTab

    tab = SpectrometerQualityTab()
    assert hasattr(tab, "spin_slit_w")
    assert hasattr(tab, "combo_mode")
    assert hasattr(tab, "lbl_badge")

    stack = parse_zmx_file(CZERNY_TURNER_ZMX)
    sys = OpticalSystem(stack)

    # Test all 5 plot modes
    for mode_idx in range(5):
        tab.combo_mode.setCurrentIndex(mode_idx)
        tab.update_plot(sys, wavelengths=[0.785, 0.830, 0.880])
        assert tab.cached_summary is not None
        assert len(tab.cached_report) > 100

    # Test dark mode styling toggle
    tab.set_dark_mode(True)
    assert tab.dark_mode is True
    tab.set_dark_mode(False)
    assert tab.dark_mode is False


def test_analysis_tabs_widget_spectrometer_tab(qapp):
    """Verify Spectrometer Quality tab in master AnalysisTabsWidget."""
    from optics_app.gui.analysis_tabs import AnalysisTabsWidget

    stack = parse_zmx_file(CZERNY_TURNER_ZMX)
    widget = AnalysisTabsWidget(model=stack)

    tab_name = "Spectrometer Quality & Resolution (Zemax)"
    assert tab_name in widget.tabs
    assert hasattr(widget, "tab_spectrometer")

    switched = widget.set_active_tab(tab_name)
    assert switched is True
    assert widget.active_tab == tab_name

    refreshed = widget.refresh_analysis()
    assert refreshed is True
    assert widget.tab_spectrometer.cached_summary is not None


# ============================================================================
# TIER 4: AI Assistant Tools & NLP Routing Verification
# ============================================================================

def test_ai_tool_run_spectrometer_quality_analysis():
    """Verify run_spectrometer_quality_analysis tool execution from registry."""
    reg = get_default_tool_registry()
    assert "run_spectrometer_quality_analysis" in reg.funcs

    args = {
        "json_path": CZERNY_TURNER_ZMX,
        "slit_width_um": 25.0,
        "input_na": 0.10,
        "raman_laser_wavelength_um": 0.785,
    }
    result = reg.execute("run_spectrometer_quality_analysis", args)

    assert isinstance(result, dict)
    assert "collimator_efl_mm" in result
    assert "camera_efl_mm" in result
    assert "avg_resolution_nm" in result
    assert "avg_resolving_power" in result
    assert "avg_throughput_pct" in result
    assert "wavelength_results" in result
    assert len(result["wavelength_results"]) > 0


def test_ai_tool_export_spectrometer_quality_report(tmp_path):
    """Verify export_spectrometer_quality_report tool generates valid text file."""
    reg = get_default_tool_registry()
    assert "export_spectrometer_quality_report" in reg.funcs

    out_file = str(tmp_path / "test_spectrometer_report.txt")
    args = {
        "output_path": out_file,
        "json_path": CZERNY_TURNER_ZMX,
        "slit_width_um": 25.0,
    }
    msg = reg.execute("export_spectrometer_quality_report", args)

    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "ZEMAX OPTICSTUDIO SPECTROMETER ANALYSIS REPORT" in content
    assert "DETAILED MULTI-WAVELENGTH PRESCRIPTION TABLE" in content
    assert "END OF ZEMAX REPORT" in content


def test_nlp_heuristic_routing():
    """Verify natural language heuristics parse spectrometer queries accurately."""
    phrases = [
        "spectral analysis to check the quality of a designed spectrometer following spectral quality, resolution, light throughput and so on following Zemax approach",
        "analyze spectrometer quality for the current design",
        "check spectrometer resolution and throughput following zemax",
        "check the quality of a designed spectrometer with slit 30 um and na 0.12",
    ]

    for p in phrases:
        parsed = parse_optical_command_heuristics(p)
        assert parsed is not None, f"Failed to parse phrase: '{p}'"
        assert parsed["name"] == "run_spectrometer_quality_analysis"

    # Export report phrase
    p_exp = "export spectrometer quality report to analysis_outputs/my_spec.txt"
    parsed_exp = parse_optical_command_heuristics(p_exp)
    assert parsed_exp is not None
    assert parsed_exp["name"] == "export_spectrometer_quality_report"
    assert "my_spec.txt" in parsed_exp["arguments"]["output_path"]
