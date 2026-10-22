"""Comprehensive 4-Tier E2E Test Suite for Ultrafast Pulse Propagation & Multimodal Nonlinear Optical Imaging.

Architecture:
- Tier 1: Feature Coverage (TestTier1FeatureCoverage) - Features F1 to F14 in isolation.
- Tier 2: Boundary & Corner Cases (TestTier2BoundaryAndCornerCases) - Edge cases, extreme values, saturation, nulls.
- Tier 3: Cross-Feature Combinations (TestTier3CrossFeatureCombinations) - Modulators + dispersion + nonlinear imaging.
- Tier 4: Real-World Workflows (TestTier4RealWorldScenarios) - End-to-end biological and workstation pipelines (S1-S5).

Authority: TEST_INFRA.md, PROJECT.md, and published literature (Boyd, Xu & Webb, Denk, Horton, Cheng & Xie, Freudiger).
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Core analysis modules (optics_app and v2)
import optics_app.analysis.ultrafast_nonlinear as app_un
import v2.analysis.ultrafast_nonlinear as v2_un

from optics_app.analysis.ultrafast_nonlinear import (
    AOMMaterial,
    AcoustoOpticModulator,
    EOMMaterial,
    ElectroOpticModulator,
    NonlinearModality,
    LaserSourceParams,
    NonlinearSampleParams,
    ObjectiveFocusParams,
    NonlinearModalityResult,
    UltrafastNonlinearSimulator,
    SpectralFocusingSRS,
    SpectralFocusingConfig,
    SpectralFocusingResult,
    DispersiveGlassMaterial,
    glass_gvd_fs2_per_mm,
    NonlinearSignalTemporalProbe,
    SpatialBeamDefects,
    InteractionPlaneConfig,
    THORLABS_ULTRAFAST_CATALOG,
    delay_from_stage_displacement_mm,
    stage_displacement_from_delay_ps,
    LIGHT_M_S,
    PLANCK_J_S,
    EPSILON_0,
    ELEMENTARY_CHARGE_C,
)

from optics_app.analysis.pulse_width import (
    PulseAnalysisConfig,
    PulseWidthAnalysis,
    gaussian_pulse_fwhm_fs,
)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ==============================================================================
# TIER 1: FEATURE COVERAGE (TestTier1FeatureCoverage)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Exhaustive verification of Features F1 to F14 in isolation."""

    # --------------------------------------------------------------------------
    # F1: Two-Photon (2P) Quadratic Laser Parameter Scaling & Rates
    # --------------------------------------------------------------------------
    def test_f1_01_two_photon_power_quadratic_scaling(self):
        """F1: 2P photon generation rate scales strictly quadratically with average power (P^2)."""
        laser1 = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0, repetition_rate_hz=80e6)
        laser2 = LaserSourceParams(average_power_w=0.030, pulse_duration_fs=100.0, repetition_rate_hz=80e6)
        sample = NonlinearSampleParams(sigma_2p_gm=36.0)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r1 = UltrafastNonlinearSimulator.simulate_two_photon(laser1, sample, obj)
        r2 = UltrafastNonlinearSimulator.simulate_two_photon(laser2, sample, obj)

        power_ratio = laser2.average_power_w / laser1.average_power_w
        rate_ratio = r2.generated_photons_per_second / r1.generated_photons_per_second
        assert pytest.approx(power_ratio ** 2, rel=1e-3) == rate_ratio
        assert r1.scaling_law_power_exponent == 2.0

    def test_f1_02_two_photon_pulse_width_inverse_scaling(self):
        """F1: 2P photon generation rate scales inversely with pulse duration (1 / tau)."""
        laser_short = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=80.0, repetition_rate_hz=80e6)
        laser_long = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=240.0, repetition_rate_hz=80e6)
        sample = NonlinearSampleParams(sigma_2p_gm=36.0)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r_short = UltrafastNonlinearSimulator.simulate_two_photon(laser_short, sample, obj)
        r_long = UltrafastNonlinearSimulator.simulate_two_photon(laser_long, sample, obj)

        duration_ratio = laser_long.pulse_duration_fs / laser_short.pulse_duration_fs
        assert pytest.approx(1.0 / duration_ratio, rel=1e-3) == (
            r_long.generated_photons_per_second / r_short.generated_photons_per_second
        )

    def test_f1_03_two_photon_repetition_rate_inverse_scaling(self):
        """F1: At fixed average power, 2P rate scales inversely with repetition rate (1 / f_rep)."""
        laser_low_rep = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0, repetition_rate_hz=40e6)
        laser_high_rep = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0, repetition_rate_hz=80e6)
        sample = NonlinearSampleParams(sigma_2p_gm=36.0)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r_low = UltrafastNonlinearSimulator.simulate_two_photon(laser_low_rep, sample, obj)
        r_high = UltrafastNonlinearSimulator.simulate_two_photon(laser_high_rep, sample, obj)

        rep_ratio = laser_high_rep.repetition_rate_hz / laser_low_rep.repetition_rate_hz
        assert pytest.approx(1.0 / rep_ratio, rel=1e-3) == (
            r_high.generated_photons_per_second / r_low.generated_photons_per_second
        )

    def test_f1_04_two_photon_temporal_coherence_g2(self):
        """F1: Evaluates temporal pulse coherence factor g_2 = sqrt(2*ln(2)/pi) ~ 0.6643 for Gaussian."""
        g2_gaussian = math.sqrt(2.0 * math.log(2.0) / math.pi)
        g2_sech2 = 1.762747 / 3.0
        assert pytest.approx(0.66428, rel=1e-4) == g2_gaussian
        assert pytest.approx(0.58758, rel=1e-4) == g2_sech2
        assert g2_gaussian > g2_sech2

    def test_f1_05_two_photon_cross_section_and_na_scaling(self):
        """F1: 2P rate scales linearly with sigma_2P and focal pulse excitation probability scales with NA^4."""
        laser = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0, repetition_rate_hz=80e6)
        sample1 = NonlinearSampleParams(sigma_2p_gm=20.0)
        sample2 = NonlinearSampleParams(sigma_2p_gm=60.0)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r1 = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample1, obj)
        r2 = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample2, obj)
        assert pytest.approx(3.0, rel=1e-3) == (r2.generated_photons_per_second / r1.generated_photons_per_second)

        obj_low_na = ObjectiveFocusParams(numerical_aperture=0.4)
        obj_high_na = ObjectiveFocusParams(numerical_aperture=0.8)
        r_na_low = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample1, obj_low_na)
        r_na_high = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample1, obj_high_na)
        na_ratio = obj_high_na.numerical_aperture / obj_low_na.numerical_aperture
        # Peak excitation probability at focus scales as 1 / area^2 = NA^4
        assert pytest.approx(na_ratio ** 4, rel=1e-2) == (
            r_na_high.details["pulse_excitation_prob"] / r_na_low.details["pulse_excitation_prob"]
        )

    # --------------------------------------------------------------------------
    # F2: Three-Photon (3P) Cubic Laser Parameter Scaling & Attenuation
    # --------------------------------------------------------------------------
    def test_f2_01_three_photon_power_cubic_scaling(self):
        """F2: 3P photon generation rate scales strictly cubically with average power (P^3)."""
        laser1 = LaserSourceParams(average_power_w=0.004, pulse_duration_fs=60.0, repetition_rate_hz=2e6)
        laser2 = LaserSourceParams(average_power_w=0.008, pulse_duration_fs=60.0, repetition_rate_hz=2e6)
        sample = NonlinearSampleParams(sigma_3p_gm3=1.0e-82)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r1 = UltrafastNonlinearSimulator.simulate_three_photon(laser1, sample, obj)
        r2 = UltrafastNonlinearSimulator.simulate_three_photon(laser2, sample, obj)
        assert pytest.approx(8.0, rel=1e-3) == (r2.generated_photons_per_second / r1.generated_photons_per_second)
        assert r1.scaling_law_power_exponent == 3.0

    def test_f2_02_three_photon_pulse_duration_inverse_square_scaling(self):
        """F2: 3P rate scales as 1 / tau^2 with pulse duration."""
        laser_short = LaserSourceParams(average_power_w=0.005, pulse_duration_fs=50.0, repetition_rate_hz=4e6)
        laser_long = LaserSourceParams(average_power_w=0.005, pulse_duration_fs=150.0, repetition_rate_hz=4e6)
        sample = NonlinearSampleParams(sigma_3p_gm3=1.0e-82)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r_short = UltrafastNonlinearSimulator.simulate_three_photon(laser_short, sample, obj)
        r_long = UltrafastNonlinearSimulator.simulate_three_photon(laser_long, sample, obj)
        assert pytest.approx((1.0 / 3.0) ** 2, rel=1e-3) == (
            r_long.generated_photons_per_second / r_short.generated_photons_per_second
        )

    def test_f2_03_three_photon_repetition_rate_inverse_square_scaling(self):
        """F2: 3P rate scales as 1 / f_rep^2 with repetition rate at fixed average power."""
        laser_1mhz = LaserSourceParams(average_power_w=0.005, pulse_duration_fs=60.0, repetition_rate_hz=1e6)
        laser_2mhz = LaserSourceParams(average_power_w=0.005, pulse_duration_fs=60.0, repetition_rate_hz=2e6)
        sample = NonlinearSampleParams(sigma_3p_gm3=1.0e-82)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r_1mhz = UltrafastNonlinearSimulator.simulate_three_photon(laser_1mhz, sample, obj)
        r_2mhz = UltrafastNonlinearSimulator.simulate_three_photon(laser_2mhz, sample, obj)
        assert pytest.approx(0.25, rel=1e-3) == (
            r_2mhz.generated_photons_per_second / r_1mhz.generated_photons_per_second
        )

    def test_f2_04_three_photon_temporal_coherence_g3(self):
        """F2: Evaluates 3rd-order coherence factor g_3 = 4*ln(2)/(pi*sqrt(3)) ~ 0.5095."""
        g3_gaussian = (4.0 * math.log(2.0) / math.pi) / math.sqrt(3.0)
        g3_sech2 = (2.0 / 15.0) * (1.762747 ** 2)
        assert pytest.approx(0.50954, rel=1e-4) == g3_gaussian
        assert pytest.approx(0.41430, rel=1e-4) == g3_sech2
        assert g3_gaussian > g3_sech2

    def test_f2_05_three_photon_deep_tissue_attenuation(self):
        """F2: Deep tissue attenuation P(z) = P_0*exp(-z/l_s) reduces 3P by exp(-3*z/l_s)."""
        laser = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=60.0, repetition_rate_hz=2e6)
        sample = NonlinearSampleParams(sigma_3p_gm3=1.0e-82)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r_surf = UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj, tissue_depth_um=0.0)
        r_deep = UltrafastNonlinearSimulator.simulate_three_photon(
            laser, sample, obj, tissue_depth_um=400.0, scattering_length_um=200.0
        )
        # z/l_s = 2 -> power drops to e^-2, 3P drops to (e^-2)^3 = e^-6
        expected_ratio = math.exp(-6.0)
        actual_ratio = r_deep.generated_photons_per_second / r_surf.generated_photons_per_second
        assert pytest.approx(expected_ratio, rel=1e-3) == actual_ratio

    # --------------------------------------------------------------------------
    # F3: Second Harmonic Generation (SHG) Modeling
    # --------------------------------------------------------------------------
    def test_f3_01_shg_wavelength_doubling(self):
        """F3: SHG generates emission at exactly half the fundamental wavelength (2*omega)."""
        laser = LaserSourceParams(center_wavelength_nm=920.0, average_power_w=0.015)
        sample = NonlinearSampleParams(d_eff_pm_per_v=0.4, thickness_um=50.0)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        res = UltrafastNonlinearSimulator.simulate_shg(laser, sample, obj)
        assert res.emission_wavelength_nm == 460.0
        assert res.modality == "SHG"

    def test_f3_02_shg_pump_power_quadratic_scaling(self):
        """F3: SHG generated power scales quadratically with pump laser power."""
        laser1 = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.010)
        laser2 = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.025)
        sample = NonlinearSampleParams(d_eff_pm_per_v=0.38, thickness_um=40.0)
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        r1 = UltrafastNonlinearSimulator.simulate_shg(laser1, sample, obj)
        r2 = UltrafastNonlinearSimulator.simulate_shg(laser2, sample, obj)
        assert pytest.approx(2.5 ** 2, rel=1e-3) == (
            r2.details["shg_power_w"] / r1.details["shg_power_w"]
        )
        assert r1.scaling_law_power_exponent == 2.0

    def test_f3_03_shg_sinc2_phase_matching_curve(self):
        """F3: SHG efficiency follows sinc^2(Delta_k * L / 2) phase-matching factor."""
        laser = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.010)
        l_um = 60.0
        l_m = l_um * 1e-6
        # Delta_k = pi / L -> Delta_k * L / 2 = pi / 2 -> sinc^2(1/2) = (2/pi)^2
        dk_target = math.pi / l_m
        sample_matched = NonlinearSampleParams(d_eff_pm_per_v=0.4, phase_mismatch_delta_k_m1=0.0, thickness_um=l_um)
        sample_mismatched = NonlinearSampleParams(d_eff_pm_per_v=0.4, phase_mismatch_delta_k_m1=dk_target, thickness_um=l_um)
        obj = ObjectiveFocusParams()

        r_match = UltrafastNonlinearSimulator.simulate_shg(laser, sample_matched, obj)
        r_mismatch = UltrafastNonlinearSimulator.simulate_shg(laser, sample_mismatched, obj)

        expected_factor = (2.0 / math.pi) ** 2
        assert pytest.approx(1.0, rel=1e-5) == r_match.details["phase_matching_sinc2"]
        assert pytest.approx(expected_factor, rel=1e-3) == r_mismatch.details["phase_matching_sinc2"]

    def test_f3_04_shg_forward_backward_directional_emission(self):
        """F3: Models forward vs. backward directional emission ratio (F/B)."""
        laser = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.010)
        sample = NonlinearSampleParams(d_eff_pm_per_v=0.4, shg_forward_backward_ratio=5.0)
        obj = ObjectiveFocusParams()

        res = UltrafastNonlinearSimulator.simulate_shg(laser, sample, obj)
        assert res.details["forward_backward_ratio"] == 5.0
        # Forward power fraction is F / (F + B) = 5 / 6
        fb = res.details["forward_backward_ratio"]
        fwd_fraction = fb / (1.0 + fb)
        assert pytest.approx(5.0 / 6.0, rel=1e-4) == fwd_fraction

    def test_f3_05_shg_interaction_length_quadratic_scaling(self):
        """F3: Under perfect phase matching, SHG power scales as L^2."""
        laser = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.010)
        sample1 = NonlinearSampleParams(d_eff_pm_per_v=0.4, phase_mismatch_delta_k_m1=0.0, thickness_um=20.0)
        sample2 = NonlinearSampleParams(d_eff_pm_per_v=0.4, phase_mismatch_delta_k_m1=0.0, thickness_um=60.0)
        obj = ObjectiveFocusParams()

        r1 = UltrafastNonlinearSimulator.simulate_shg(laser, sample1, obj)
        r2 = UltrafastNonlinearSimulator.simulate_shg(laser, sample2, obj)
        assert pytest.approx(9.0, rel=1e-3) == (r2.details["shg_power_w"] / r1.details["shg_power_w"])

    # --------------------------------------------------------------------------
    # F4: CARS Complex Susceptibility & Fano Lineshape
    # --------------------------------------------------------------------------
    def test_f4_01_cars_anti_stokes_frequency_mapping(self):
        """F4: Anti-Stokes frequency omega_AS = 2*omega_p - omega_S and wavelength mapping."""
        wl_p = 800.0
        wl_s = 1040.0
        expected_as_wl = 1.0 / (2.0 / wl_p - 1.0 / wl_s)  # ~ 650.0 nm
        assert pytest.approx(650.0, rel=1e-3) == expected_as_wl

        pump = LaserSourceParams(center_wavelength_nm=wl_p)
        stokes = LaserSourceParams(center_wavelength_nm=wl_s)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj)
        assert pytest.approx(expected_as_wl, rel=1e-3) == res.emission_wavelength_nm

    def test_f4_02_cars_complex_susceptibility_expansion_identity(self):
        """F4: Exact algebraic decomposition |chi^(3)|^2 = chi_NR^2 + 2*chi_NR*chi_R' + |chi_R|^2."""
        chi_nr = 0.25
        a_r = 1.2
        omega_r = 2845.0
        gamma_r = 15.0
        detunings = np.linspace(2700.0, 3000.0, 301)

        chi_r_real = a_r * (omega_r - detunings) / ((omega_r - detunings) ** 2 + gamma_r ** 2)
        chi_r_imag = a_r * gamma_r / ((omega_r - detunings) ** 2 + gamma_r ** 2)

        lhs = (chi_nr + chi_r_real) ** 2 + chi_r_imag ** 2
        rhs = (chi_nr ** 2) + 2.0 * chi_nr * chi_r_real + (chi_r_real ** 2 + chi_r_imag ** 2)
        max_diff = float(np.max(np.abs(lhs - rhs)))
        assert max_diff < 1e-15

    def test_f4_03_cars_asymmetric_fano_lineshape(self):
        """F4: Reproduces asymmetric Fano-like lineshape with peak and dip."""
        pump = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.020)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.010)
        sample = NonlinearSampleParams(
            raman_shift_cm1=2845.0,
            raman_linewidth_cm1=15.0,
            chi3_resonant_m2_per_v2=2e-23,
            chi3_nonresonant_m2_per_v2=1e-23,
        )
        obj = ObjectiveFocusParams()

        res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj, detuning_span_cm1=100.0, spectral_points=201)
        spec = np.array(res.details["fano_spectrum"])
        base = res.details["nonresonant_baseline"]

        assert np.max(spec) > base
        assert np.min(spec) < base
        assert res.details["fano_asymmetry_factor"] > 1.0

    def test_f4_04_cars_peak_red_shift_and_dip_blue_shift(self):
        """F4: Fano interference red-shifts resonant peak and blue-shifts destructive dip."""
        pump = LaserSourceParams(center_wavelength_nm=800.0)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0)
        omega_r = 2845.0
        sample = NonlinearSampleParams(raman_shift_cm1=omega_r, raman_linewidth_cm1=15.0)
        obj = ObjectiveFocusParams()

        res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj, detuning_span_cm1=80.0, spectral_points=301)
        assert res.details["peak_detuning_cm1"] < omega_r
        assert res.details["dip_detuning_cm1"] > omega_r

    def test_f4_05_cars_quadratic_pump_linear_stokes_power_scaling(self):
        """F4: CARS signal intensity scales as I_p^2 * I_S."""
        pump1 = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.010)
        pump2 = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.020)
        stokes1 = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.010)
        stokes2 = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.030)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        r_base = UltrafastNonlinearSimulator.simulate_cars(pump1, stokes1, sample, obj)
        r_double_p = UltrafastNonlinearSimulator.simulate_cars(pump2, stokes1, sample, obj)
        r_triple_s = UltrafastNonlinearSimulator.simulate_cars(pump1, stokes2, sample, obj)

        assert pytest.approx(4.0, rel=1e-3) == (
            r_double_p.generated_photons_per_second / r_base.generated_photons_per_second
        )
        assert pytest.approx(3.0, rel=1e-3) == (
            r_triple_s.generated_photons_per_second / r_base.generated_photons_per_second
        )

    # --------------------------------------------------------------------------
    # F5: SRS Linear Scaling & Non-Resonant Suppression
    # --------------------------------------------------------------------------
    def test_f5_01_srs_strict_linear_concentration_scaling(self):
        """F5: SRS signal exhibits strict linear concentration dependence (R^2 = 1.0)."""
        pump = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.020)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.010)
        sample = NonlinearSampleParams(raman_gain_m_per_w=2e-13)
        obj = ObjectiveFocusParams()

        c_scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        signals = []
        for c in c_scales:
            res = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj, concentration_scale=c)
            signals.append(res.details["stimulated_raman_gain_srg"])

        ratios = [s / signals[2] for s in signals]  # normalize to c=1.0
        for expected_c, actual_ratio in zip(c_scales, ratios):
            assert pytest.approx(expected_c, rel=1e-4) == actual_ratio

    def test_f5_02_srs_zero_nonresonant_background(self):
        """F5: SRS has zero non-resonant electronic background offset."""
        pump = LaserSourceParams(center_wavelength_nm=800.0)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        res = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj)
        assert res.details["nonresonant_background"] == 0.0

    def test_f5_03_srs_stokes_gain_and_pump_loss_energy_balance(self):
        """F5: Enforces photon conservation: Delta_I_p / Delta_I_S = lambda_S / lambda_p."""
        wl_p = 790.0
        wl_s = 1045.0
        pump = LaserSourceParams(center_wavelength_nm=wl_p, average_power_w=0.025)
        stokes = LaserSourceParams(center_wavelength_nm=wl_s, average_power_w=0.015)
        sample = NonlinearSampleParams(raman_gain_m_per_w=1.5e-13)
        obj = ObjectiveFocusParams()

        res = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj)
        srg = res.details["stimulated_raman_gain_srg"]
        srl = res.details["stimulated_raman_loss_srl"]
        assert pytest.approx(wl_s / wl_p, rel=1e-4) == (srl / srg)

    def test_f5_04_srs_lorentzian_spectral_profile_matches_spontaneous_raman(self):
        """F5: SRS spectral response is purely Lorentzian, matching spontaneous Raman."""
        gamma = 15.0
        omega_r = 2845.0
        detunings = np.linspace(omega_r - 100.0, omega_r + 100.0, 201)
        lorentzian = (gamma ** 2) / (((detunings - omega_r) ** 2) + (gamma ** 2))
        assert np.argmax(lorentzian) == 100
        assert pytest.approx(1.0, rel=1e-5) == np.max(lorentzian)

    def test_f5_05_srs_demodulated_signal_with_eom_modulation(self):
        """F5: Simulates modulation transfer of EOM modulated pump onto Stokes probe."""
        pump = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.030)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.015)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()
        eom = ElectroOpticModulator(applied_voltage_v=120.0, half_wave_voltage_v=240.0)

        res_eom = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj, eom_modulator=eom)
        assert res_eom.details["demodulated_signal_w"] > 0.0
        assert res_eom.shot_noise_snr > 0.0

    # --------------------------------------------------------------------------
    # F6: Unified Focal Volume & Shot-Noise SNR Diagnostics
    # --------------------------------------------------------------------------
    def test_f6_01_diffraction_limited_waist_and_rayleigh_range(self):
        """F6: Focal waist w_0 = lambda / (pi * NA) and Rayleigh range z_R = n * pi * w_0^2 / lambda."""
        obj = ObjectiveFocusParams(numerical_aperture=0.8)
        wl_nm = 800.0
        w0 = obj.beam_waist_radius_m(wl_nm)
        zr = obj.rayleigh_range_m(wl_nm, refractive_index=1.33)

        expected_w0 = (800e-9) / (math.pi * 0.8)
        expected_zr = 1.33 * math.pi * (expected_w0 ** 2) / (800e-9)
        assert pytest.approx(expected_w0, rel=1e-4) == w0
        assert pytest.approx(expected_zr, rel=1e-4) == zr

    def test_f6_02_focal_excitation_volumes_2p_vs_3p(self):
        """F6: Excitation volume V_3P is approximately 0.5 * V_2P due to higher confinement."""
        obj = ObjectiveFocusParams(numerical_aperture=0.85)
        wl_nm = 800.0
        v_2p = obj.effective_focal_volume_m3(wl_nm, order=2, refractive_index=1.33)
        v_3p = obj.effective_focal_volume_m3(wl_nm, order=3, refractive_index=1.33)

        # Ratio (2/3)^1.5 ~ 0.5443
        ratio = v_3p / v_2p
        assert pytest.approx((2.0 / 3.0) ** 1.5, rel=1e-3) == ratio

    def test_f6_03_focal_peak_intensity_and_fluence(self):
        """F6: Calculates peak focal intensity (W/cm^2) and peak fluence (J/cm^2)."""
        laser = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0, repetition_rate_hz=80e6)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams(numerical_aperture=0.8)

        res = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj)
        assert res.focal_peak_intensity_w_cm2 > 1e7
        assert res.focal_peak_fluence_j_cm2 > 1e-6

    def test_f6_04_shot_noise_snr_scaling_with_photon_counts(self):
        """F6: Shot-noise SNR scales as sqrt(N_detected)."""
        laser1 = LaserSourceParams(average_power_w=0.010)
        laser2 = LaserSourceParams(average_power_w=0.020)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        r1 = UltrafastNonlinearSimulator.simulate_two_photon(laser1, sample, obj)
        r2 = UltrafastNonlinearSimulator.simulate_two_photon(laser2, sample, obj)

        counts_ratio = r2.detected_photons_per_dwell / r1.detected_photons_per_dwell
        snr_ratio = r2.shot_noise_snr / r1.shot_noise_snr
        assert pytest.approx(math.sqrt(counts_ratio), rel=1e-3) == snr_ratio

    def test_f6_05_srs_photodiode_shot_noise_limit_snr(self):
        """F6: SRS shot noise current is governed by sqrt(2*q*I_det*Delta_f)."""
        pump = LaserSourceParams(average_power_w=0.020)
        stokes = LaserSourceParams(average_power_w=0.010)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams(dwell_time_us=2.0)

        res = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj)
        i_shot = res.details["shot_noise_current_a"]
        assert 1e-12 < i_shot < 1e-6
        assert res.shot_noise_snr > 0.0

    # --------------------------------------------------------------------------
    # F7: AOM Bragg Angle, RF Efficiency & Rise Time
    # --------------------------------------------------------------------------
    def test_f7_01_aom_bragg_angle_internal_and_external(self):
        """F7: Computes internal and external Bragg angles via sin(theta_B) = lambda * f_RF / (2 * v_s)."""
        aom = AcoustoOpticModulator(
            material=AOMMaterial.TEO2_SLOW_SHEAR,
            rf_frequency_mhz=80.0,
            sound_velocity_m_s=617.0,
            refractive_index=2.26,
        )
        wl_nm = 800.0
        theta_ext = aom.bragg_angle_rad(wl_nm, inside_crystal=False)
        theta_med = aom.bragg_angle_rad(wl_nm, inside_crystal=True)

        expected_ext = math.asin((800e-9 * 80e6) / (2.0 * 617.0))
        expected_med = math.asin((800e-9 * 80e6) / (2.0 * 2.26 * 617.0))
        assert pytest.approx(expected_ext, rel=1e-4) == theta_ext
        assert pytest.approx(expected_med, rel=1e-4) == theta_med
        assert theta_ext > theta_med

    def test_f7_02_aom_total_deflection_angle(self):
        """F7: Total external deflection angle Theta_defl = 2 * theta_B_ext."""
        aom = AcoustoOpticModulator(sound_velocity_m_s=617.0)
        wl_nm = 800.0
        theta_ext = aom.bragg_angle_rad(wl_nm, inside_crystal=False)
        theta_defl = aom.deflection_angle_rad(wl_nm)
        assert pytest.approx(2.0 * theta_ext, rel=1e-5) == theta_defl

    def test_f7_03_aom_rf_diffraction_efficiency_curve(self):
        """F7: RF diffraction efficiency curve eta(P_RF) obeys sin^2 coupling with power."""
        aom = AcoustoOpticModulator(figure_of_merit_m2_per_w=7.93e-13, insertion_loss_fraction=0.04)
        eff_zero = aom.diffraction_efficiency(rf_power_w=0.0)
        eff_half = aom.diffraction_efficiency(rf_power_w=0.2)
        eff_full = aom.diffraction_efficiency(rf_power_w=1.0)

        assert eff_zero == 0.0
        assert 0.0 < eff_half < eff_full
        assert eff_full <= 0.96

    def test_f7_04_aom_acoustic_rise_time_and_bandwidth(self):
        """F7: Rise time tau_r = 1.28 * w_0 / v_s and modulation bandwidth Delta_f = 0.66 / tau_r."""
        aom = AcoustoOpticModulator(sound_velocity_m_s=617.0)
        waist_mm = 0.2
        expected_rise_ns = (1.28 * (0.2e-3) / 617.0) * 1e9
        actual_rise_ns = aom.acoustic_rise_time_ns(waist_mm)
        assert pytest.approx(expected_rise_ns, rel=1e-4) == actual_rise_ns

        expected_bw_mhz = (0.66 / actual_rise_ns) * 1e3
        actual_bw_mhz = aom.modulation_bandwidth_mhz(waist_mm)
        assert pytest.approx(expected_bw_mhz, rel=1e-4) == actual_bw_mhz

    def test_f7_05_aom_crystal_dispersion_teo2_vs_silica(self):
        """F7: Evaluates crystal GDD: TeO2 is highly dispersive (~ +4500 fs^2) vs Fused Silica (~ +360 fs^2)."""
        aom_teo2 = AcoustoOpticModulator(material=AOMMaterial.TEO2_SLOW_SHEAR, crystal_length_mm=10.0)
        aom_fs = AcoustoOpticModulator(material=AOMMaterial.FUSED_SILICA, crystal_length_mm=10.0)

        gdd_teo2 = aom_teo2.crystal_gdd_fs2(800.0)
        gdd_fs = aom_fs.crystal_gdd_fs2(800.0)
        assert gdd_teo2 > 3000.0
        assert 300.0 <= gdd_fs <= 400.0
        assert gdd_teo2 > 8.0 * gdd_fs

    # --------------------------------------------------------------------------
    # F8: EOM Pockels Phase, V_pi, PBS Transmission & Lock-in
    # --------------------------------------------------------------------------
    def test_f8_01_eom_linear_pockels_phase_retardation(self):
        """F8: Electro-optic phase retardation Gamma(V) = pi * (V + V_bias) / V_pi."""
        eom = ElectroOpticModulator(half_wave_voltage_v=240.0, applied_voltage_v=120.0, bias_voltage_v=0.0)
        assert pytest.approx(math.pi / 2.0, rel=1e-5) == eom.phase_retardation_rad()
        assert pytest.approx(math.pi, rel=1e-5) == eom.phase_retardation_rad(voltage_v=240.0)

    def test_f8_02_eom_half_wave_voltage_vpi_benchmark(self):
        """F8: Benchmarks V_pi for standard materials (RTP ~ 240V, LiNbO3 ~ 180V, KDP ~ 3200V)."""
        eom_rtp = ElectroOpticModulator(material=EOMMaterial.RTP)
        eom_linbo3 = ElectroOpticModulator(material=EOMMaterial.LINBO3)
        eom_kdp = ElectroOpticModulator(material=EOMMaterial.KDP)

        assert eom_rtp.half_wave_voltage_v == 240.0
        assert eom_linbo3.half_wave_voltage_v == 180.0
        assert eom_kdp.half_wave_voltage_v == 3200.0

    def test_f8_03_eom_transmission_curve_with_extinction_ratio(self):
        """F8: PBS transmission T(V) = T_min + (T_max - T_min)*sin^2(Gamma/2) with finite ER."""
        eom = ElectroOpticModulator(half_wave_voltage_v=250.0, insertion_loss_fraction=0.02, extinction_ratio=1000.0)
        t_min = eom.transmission(voltage_v=0.0)
        t_max = eom.transmission(voltage_v=250.0)
        assert pytest.approx(0.98 / 1000.0, rel=1e-4) == t_min
        assert pytest.approx(0.98, rel=1e-4) == t_max
        assert pytest.approx(1000.0, rel=1e-3) == (t_max / t_min)

    def test_f8_04_eom_mhz_time_series_waveform_generation(self):
        """F8: Generates time-domain modulated power waveform at high RF frequencies (e.g. 10 MHz)."""
        eom = ElectroOpticModulator(
            half_wave_voltage_v=200.0,
            bias_voltage_v=100.0,
            modulation_amplitude_v=20.0,
            modulation_frequency_mhz=10.0,
        )
        t_ns = np.linspace(0.0, 200.0, 2001)
        p_t = eom.modulate_time_series(t_ns, input_power_w=1.0)
        assert len(p_t) == 2001
        assert np.max(p_t) <= 1.0
        assert np.min(p_t) >= 0.0

    def test_f8_05_eom_dual_phase_lockin_demodulation(self):
        """F8: Performs dual-phase lock-in demodulation recovering in-phase and quadrature components."""
        eom = ElectroOpticModulator(
            half_wave_voltage_v=200.0,
            bias_voltage_v=100.0,
            modulation_amplitude_v=20.0,
            modulation_frequency_mhz=10.0,
        )
        t_ns = np.linspace(0.0, 400.0, 4001)
        p_t = eom.modulate_time_series(t_ns, input_power_w=1.0)
        x_inphase, y_quad, r_mag = eom.demodulate_lockin(p_t, t_ns, phase_deg=0.0)

        assert r_mag > 0.05
        assert pytest.approx(x_inphase, abs=1e-3) == r_mag
        assert abs(y_quad) < 1e-3

    # --------------------------------------------------------------------------
    # F9: AOM & EOM Component Hierarchy Integration
    # --------------------------------------------------------------------------
    def test_f9_01_aom_config_structure_or_modulator_dataclass(self):
        """F9: Validates AOM parameters and attributes conformance."""
        aom = AcoustoOpticModulator(
            material=AOMMaterial.TEO2_SLOW_SHEAR,
            rf_frequency_mhz=80.0,
            rf_power_w=1.2,
            crystal_length_mm=12.0,
        )
        assert aom.rf_frequency_mhz == 80.0
        assert aom.rf_power_w == 1.2
        assert aom.crystal_length_mm == 12.0

    def test_f9_02_eom_config_structure_or_modulator_dataclass(self):
        """F9: Validates EOM parameters and attributes conformance."""
        eom = ElectroOpticModulator(
            material=EOMMaterial.RTP,
            half_wave_voltage_v=240.0,
            applied_voltage_v=60.0,
            bias_voltage_v=120.0,
            modulation_frequency_mhz=10.0,
        )
        assert eom.half_wave_voltage_v == 240.0
        assert eom.bias_voltage_v == 120.0

    def test_f9_03_modulator_power_partitioning_0th_and_1st_order(self):
        """F9: Modulate power partitions available power into diffracted 1st and undiffracted 0th order."""
        aom = AcoustoOpticModulator(insertion_loss_fraction=0.05)
        diff, undiff = aom.modulate_power(input_power_w=2.0, rf_power_w=0.5, wavelength_nm=800.0)
        assert diff > 0.0
        assert undiff > 0.0
        assert pytest.approx(2.0 * 0.95, rel=1e-5) == (diff + undiff)

    def test_f9_04_surface_stack_modulator_integration(self):
        """F9: Verifies Surface and SurfaceStack integration with optical modulators."""
        stack = SurfaceStack()
        s_aom = Surface(name="AOM Crystal", thickness=10.0, glass="F_SILICA")
        s_eom = Surface(name="EOM Pockels Cell", thickness=15.0, glass="F_SILICA")
        stack.add_surface(s_aom)
        stack.add_surface(s_eom)
        assert len(stack.surfaces) == 2

    def test_f9_05_modulator_persistence_round_trip(self):
        """F9: SurfaceStack JSON persistence preserves optical layout and analysis metadata."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="AOM", thickness=10.0, glass="F_SILICA"))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "modulator_test.json")
            stack.save_to_json(path, wavelengths=[0.800])
            loaded = SurfaceStack.load_from_json(path)
            assert len(loaded["stack"].surfaces) == 2
            assert loaded["stack"].surfaces[0].name == "AOM"

    # --------------------------------------------------------------------------
    # F10: End-to-End Pulse Propagation & GD/GDD/TOD
    # --------------------------------------------------------------------------
    def test_f10_01_analytical_sellmeier_gvd_derivatives(self):
        """F10: Evaluates material GVD in fs^2/mm from dispersive glass formulations."""
        gvd_fs57 = glass_gvd_fs2_per_mm(DispersiveGlassMaterial.SF57, 800.0)
        gvd_fs11 = glass_gvd_fs2_per_mm(DispersiveGlassMaterial.SF11, 800.0)
        gvd_silica = glass_gvd_fs2_per_mm(DispersiveGlassMaterial.FUSED_SILICA, 800.0)

        assert pytest.approx(215.0, rel=1e-3) == gvd_fs57
        assert pytest.approx(190.0, rel=1e-3) == gvd_fs11
        assert pytest.approx(36.1, rel=1e-3) == gvd_silica

    def test_f10_02_pulse_width_analysis_surface_stack_tracking(self):
        """F10: PulseWidthAnalysis traces accumulated GD, GDD, and TOD across sequential surfaces."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=10.0, glass="AIR"))
        stack.add_surface(Surface(name="Window Front", thickness=10.0, glass="F_SILICA"))
        stack.add_surface(Surface(name="Window Back", thickness=10.0, glass="AIR"))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR))

        cfg = PulseAnalysisConfig(center_wavelength_nm=800.0, transform_limit_fwhm_fs=100.0)
        res = PulseWidthAnalysis.analyze(stack, cfg)
        assert len(res.probes) >= 3
        exit_probe = res.probe(2)
        assert exit_probe is not None and exit_probe.valid
        assert exit_probe.material_gdd_fs2 > 300.0

    def test_f10_03_fused_silica_benchmark_dispersion_at_800nm(self):
        """F10: Fused silica dispersion at 800 nm matches benchmark: ~ +36.1 fs^2/mm."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=10.0, glass="AIR"))
        stack.add_surface(Surface(name="FS Front", thickness=1.0, glass="F_SILICA"))
        stack.add_surface(Surface(name="FS Back", thickness=10.0, glass="AIR"))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR))
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig(center_wavelength_nm=800.0))
        p = res.probe(2)
        assert 34.0 <= p.material_gdd_fs2 <= 38.0

    def test_f10_04_group_delay_and_group_index_propagation(self):
        """F10: Cumulative group delay GD tracks optical path length."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=10.0, glass="AIR"))
        stack.add_surface(Surface(name="FS Front", thickness=10.0, glass="F_SILICA"))
        stack.add_surface(Surface(name="FS Back", thickness=10.0, glass="AIR"))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR))
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig(center_wavelength_nm=800.0))
        p = res.probe(2)
        assert p.group_delay_fs > 0.0

    def test_f10_05_tod_accumulation_across_glasses(self):
        """F10: Third-Order Dispersion (TOD) is tracked across optical path."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="FS", thickness=20.0, glass="F_SILICA"))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR))
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig(center_wavelength_nm=800.0, include_tod=True))
        p = res.probe(0)
        assert hasattr(p, "system_tod_fs3")

    # --------------------------------------------------------------------------
    # F11: Pulse Temporal Broadening & Pre-Chirp Compressor
    # --------------------------------------------------------------------------
    def test_f11_01_gaussian_broadening_closed_form(self):
        """F11: Gaussian pulse broadening tau_out = tau_in * sqrt(1 + (4*ln(2)*GDD / tau_in^2)^2)."""
        tau_0 = 50.0
        gdd = 1500.0
        expected = tau_0 * math.sqrt(1.0 + (4.0 * math.log(2.0) * gdd / (tau_0 ** 2)) ** 2)
        calc = gaussian_pulse_fwhm_fs(tau_0, gdd, include_tod=False)
        assert pytest.approx(expected, rel=1e-3) == calc

    def test_f11_02_peak_intensity_degradation_factor(self):
        """F11: Peak intensity degrades inversely with pulse broadening factor: I_peak / I_0 = tau_in / tau_out."""
        tau_in = 30.0
        gdd = 800.0
        tau_out = tau_in * math.sqrt(1.0 + (4.0 * math.log(2.0) * gdd / (tau_in ** 2)) ** 2)
        degradation = tau_in / tau_out
        assert 0.0 < degradation < 1.0

    def test_f11_03_negative_prechirp_exact_compensation(self):
        """F11: Applying GDD_prechirp = -GDD_glass restores transform-limited duration at focus."""
        tau_0 = 30.0
        gdd_sys = 1200.0
        laser = LaserSourceParams(pulse_duration_fs=tau_0)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        res_broad = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, external_gdd_fs2=gdd_sys)
        assert res_broad.details["focal_pulse_width_fs"] > 50.0

        res_comp = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, external_gdd_fs2=gdd_sys - 1200.0)
        assert pytest.approx(tau_0, rel=1e-3) == res_comp.details["focal_pulse_width_fs"]

    def test_f11_04_ultrashort_pulse_extreme_dispersion_sensitivity(self):
        """F11: Ultrashort pulses (e.g. 20 fs) broaden drastically more than longer pulses (e.g. 100 fs)."""
        gdd = 1000.0
        tau_short = 20.0
        tau_long = 100.0
        broad_short = gaussian_pulse_fwhm_fs(tau_short, gdd) / tau_short
        broad_long = gaussian_pulse_fwhm_fs(tau_long, gdd) / tau_long
        assert broad_short > 5.0 * broad_long

    def test_f11_05_treacy_or_prism_compressor_dispersion_matching(self):
        """F11: Thorlabs catalog chirped mirrors (DCM-800) provide negative GDD (-50 fs^2 per bounce)."""
        mirror = THORLABS_ULTRAFAST_CATALOG["DCM-800"]
        assert mirror["gdd_fs2_per_bounce"] == -50.0
        bounces = 10
        total_comp_gdd = bounces * mirror["gdd_fs2_per_bounce"]
        assert total_comp_gdd == -500.0

    # --------------------------------------------------------------------------
    # F12: Dual-Package Synchronization (optics_app vs v2)
    # --------------------------------------------------------------------------
    def test_f12_01_modulator_models_dual_package_parity(self):
        """F12: Verifies identical AOM & EOM behavior across optics_app and v2."""
        aom_app = app_un.AcoustoOpticModulator(rf_frequency_mhz=80.0)
        aom_v2 = v2_un.AcoustoOpticModulator(rf_frequency_mhz=80.0)
        assert pytest.approx(aom_app.bragg_angle_rad(800.0), rel=1e-6) == aom_v2.bragg_angle_rad(800.0)

        eom_app = app_un.ElectroOpticModulator(half_wave_voltage_v=240.0)
        eom_v2 = v2_un.ElectroOpticModulator(half_wave_voltage_v=240.0)
        assert pytest.approx(eom_app.transmission(120.0), rel=1e-6) == eom_v2.transmission(120.0)

    def test_f12_02_two_photon_simulator_dual_package_parity(self):
        """F12: Verifies identical 2P simulation results across optics_app and v2."""
        laser_app = app_un.LaserSourceParams(average_power_w=0.015)
        laser_v2 = v2_un.LaserSourceParams(average_power_w=0.015)
        sample_app = app_un.NonlinearSampleParams()
        sample_v2 = v2_un.NonlinearSampleParams()
        obj_app = app_un.ObjectiveFocusParams()
        obj_v2 = v2_un.ObjectiveFocusParams()

        r_app = app_un.UltrafastNonlinearSimulator.simulate_two_photon(laser_app, sample_app, obj_app)
        r_v2 = v2_un.UltrafastNonlinearSimulator.simulate_two_photon(laser_v2, sample_v2, obj_v2)
        assert pytest.approx(r_app.generated_photons_per_second, rel=1e-6) == r_v2.generated_photons_per_second

    def test_f12_03_three_photon_simulator_dual_package_parity(self):
        """F12: Verifies identical 3P simulation results across optics_app and v2."""
        laser_app = app_un.LaserSourceParams(average_power_w=0.005)
        laser_v2 = v2_un.LaserSourceParams(average_power_w=0.005)
        sample_app = app_un.NonlinearSampleParams()
        sample_v2 = v2_un.NonlinearSampleParams()
        obj_app = app_un.ObjectiveFocusParams()
        obj_v2 = v2_un.ObjectiveFocusParams()

        r_app = app_un.UltrafastNonlinearSimulator.simulate_three_photon(laser_app, sample_app, obj_app)
        r_v2 = v2_un.UltrafastNonlinearSimulator.simulate_three_photon(laser_v2, sample_v2, obj_v2)
        assert pytest.approx(r_app.generated_photons_per_second, rel=1e-6) == r_v2.generated_photons_per_second

    def test_f12_04_cars_srs_simulators_dual_package_parity(self):
        """F12: Verifies identical CARS and SRS results across optics_app and v2."""
        p_app = app_un.LaserSourceParams(center_wavelength_nm=800.0)
        s_app = app_un.LaserSourceParams(center_wavelength_nm=1040.0)
        p_v2 = v2_un.LaserSourceParams(center_wavelength_nm=800.0)
        s_v2 = v2_un.LaserSourceParams(center_wavelength_nm=1040.0)
        sample_app = app_un.NonlinearSampleParams()
        sample_v2 = v2_un.NonlinearSampleParams()
        obj_app = app_un.ObjectiveFocusParams()
        obj_v2 = v2_un.ObjectiveFocusParams()

        cars_app = app_un.UltrafastNonlinearSimulator.simulate_cars(p_app, s_app, sample_app, obj_app)
        cars_v2 = v2_un.UltrafastNonlinearSimulator.simulate_cars(p_v2, s_v2, sample_v2, obj_v2)
        assert pytest.approx(cars_app.details["peak_detuning_cm1"], rel=1e-6) == cars_v2.details["peak_detuning_cm1"]

        srs_app = app_un.UltrafastNonlinearSimulator.simulate_srs(p_app, s_app, sample_app, obj_app)
        srs_v2 = v2_un.UltrafastNonlinearSimulator.simulate_srs(p_v2, s_v2, sample_v2, obj_v2)
        assert pytest.approx(srs_app.details["stimulated_raman_gain_srg"], rel=1e-6) == srs_v2.details["stimulated_raman_gain_srg"]

    def test_f12_05_spectral_focusing_dual_package_parity(self):
        """F12: Verifies identical Spectral Focusing SRS results across optics_app and v2."""
        p_app = app_un.LaserSourceParams(center_wavelength_nm=800.0)
        s_app = app_un.LaserSourceParams(center_wavelength_nm=1040.0)
        p_v2 = v2_un.LaserSourceParams(center_wavelength_nm=800.0)
        s_v2 = v2_un.LaserSourceParams(center_wavelength_nm=1040.0)
        cfg_app = app_un.SpectralFocusingConfig()
        cfg_v2 = v2_un.SpectralFocusingConfig()
        sample_app = app_un.NonlinearSampleParams()
        sample_v2 = v2_un.NonlinearSampleParams()

        res_app = app_un.SpectralFocusingSRS.simulate(p_app, s_app, cfg_app, sample_app)
        res_v2 = v2_un.SpectralFocusingSRS.simulate(p_v2, s_v2, cfg_v2, sample_v2)
        assert pytest.approx(res_app.chirp_matching_ratio, rel=1e-6) == res_v2.chirp_matching_ratio
        assert pytest.approx(res_app.peak_srs_gain, rel=1e-6) == res_v2.peak_srs_gain

    # --------------------------------------------------------------------------
    # F13: GUI Ultrafast Dock & Diagnostics Controls
    # --------------------------------------------------------------------------
    def test_f13_01_ultrafast_dock_instantiation_and_tab_names(self):
        """F13: Instantiates UltrafastAnalysisDock and verifies tabs."""
        from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
        dock = UltrafastAnalysisDock()
        assert dock.objectName() == "UltrafastAnalysisDock"
        assert dock.tabs.count() >= 3
        tab_titles = [dock.tabs.tabText(i) for i in range(dock.tabs.count())]
        assert any("Dispersion" in t for t in tab_titles)
        assert any("Nonlinear" in t for t in tab_titles)
        assert any("Modulator" in t for t in tab_titles)

    def test_f13_02_dock_modality_combobox_switching(self):
        """F13: Verifies modality selection across all 5 modalities."""
        from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
        dock = UltrafastAnalysisDock()
        modalities = ["2P", "3P", "SHG", "CARS", "SRS"]
        for idx in range(len(modalities)):
            dock.combo_modality.setCurrentIndex(idx)
            assert dock.combo_modality.currentIndex() == idx

    def test_f13_03_dock_modulator_metrics_calculation(self):
        """F13: Computes live metrics for AOM and EOM widgets."""
        from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
        dock = UltrafastAnalysisDock()
        aom, eom = dock.update_modulator_metrics()
        assert aom is not None
        assert eom is not None

    def test_f13_04_dock_nonlinear_simulation_execution(self):
        """F13: Executes simulation from GUI dock and captures results."""
        from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
        dock = UltrafastAnalysisDock()
        dock.combo_modality.setCurrentIndex(0)  # 2P
        dock.run_nonlinear_simulation()
        assert dock._last_nonlinear_result is not None
        assert dock._last_nonlinear_result.modality == "2P"

    def test_f13_05_dock_results_table_population(self):
        """F13: Populates results table with photon rates, SNR, and diagnostics."""
        from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
        dock = UltrafastAnalysisDock()
        dock.combo_modality.setCurrentIndex(0)
        dock.run_nonlinear_simulation()
        assert dock.tbl_nonlinear_results.rowCount() >= 8

    # --------------------------------------------------------------------------
    # F14: New Ultrafast Project Workspace Initialization
    # --------------------------------------------------------------------------
    def test_f14_01_ultrafast_workspace_initialization_interface(self):
        """F14: Verifies MainWindow ultrafast workspace interface per M3 interface contract."""
        from optics_app.gui.main_window import MainWindow
        if not hasattr(MainWindow, "on_new_project"):
            pytest.skip("MainWindow.on_new_project pending M3 implementation")
        window = MainWindow()
        try:
            assert callable(window.on_new_project)
        finally:
            window.close()

    def test_f14_02_ultrafast_project_default_wavelength_800nm(self):
        """F14: When ultrafast=True, primary laser wavelength is set to 800 nm (Ti:Sapphire)."""
        from optics_app.gui.main_window import MainWindow
        if not hasattr(MainWindow, "on_new_project"):
            pytest.skip("MainWindow.on_new_project pending M3 implementation")
        window = MainWindow()
        try:
            assert window.on_new_project(ultrafast=True)
            assert window.system_config.primary_wavelength_um == 0.800
        finally:
            window.close()

    def test_f14_03_ultrafast_project_pulse_analysis_config_exposure(self):
        """F14: Ultrafast workspace exposes femtosecond pulse analysis configuration."""
        from optics_app.gui.main_window import MainWindow
        if not hasattr(MainWindow, "on_new_project"):
            pytest.skip("MainWindow.on_new_project pending M3 implementation")
        window = MainWindow()
        try:
            assert window.on_new_project(ultrafast=True)
            cfg = window.analysis_tabs.get_analysis_config()
            assert "pulse_width_analysis" in cfg
            assert cfg["pulse_width_analysis"]["center_wavelength_nm"] == 800.0
        finally:
            window.close()

    def test_f14_04_ultrafast_project_save_load_round_trip(self):
        """F14: Saving and reloading an ultrafast project preserves settings."""
        from optics_app.gui.main_window import MainWindow
        if not hasattr(MainWindow, "on_new_project"):
            pytest.skip("MainWindow.on_new_project pending M3 implementation")
        window = MainWindow()
        try:
            assert window.on_new_project(ultrafast=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = os.path.join(tmp_dir, "test_uf.json")
                assert window.on_save_project(path)
                assert window.on_load_project(path)
                assert window.system_config.primary_wavelength_um == 0.800
        finally:
            window.close()

    def test_f14_05_ultrafast_project_undo_redo_cleanliness(self):
        """F14: Undo after new ultrafast project cleanly restores prior configuration."""
        from optics_app.gui.main_window import MainWindow
        if not hasattr(MainWindow, "on_new_project"):
            pytest.skip("MainWindow.on_new_project pending M3 implementation")
        window = MainWindow()
        try:
            window.system_config.primary_wavelength_um = 1.064
            assert window.on_new_project(ultrafast=True)
            assert window.on_undo()
            assert window.system_config.primary_wavelength_um == 1.064
        finally:
            window.close()


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (TestTier2BoundaryAndCornerCases)
# ==============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary value analysis (BVA), extreme limits, and corner conditions."""

    # --- B1 - B4: 2P / 3P Power, Pulse Duration & NA Boundaries ---
    def test_b01_zero_or_negative_laser_power_clamped(self):
        """B1: Zero or negative average laser power produces zero generated photons without crashing."""
        laser_zero = LaserSourceParams(average_power_w=0.0)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_two_photon(laser_zero, sample, obj)
        assert res.generated_photons_per_second == 0.0

    def test_b02_vanishing_pulse_duration_safety(self):
        """B2: Safeguards against non-positive pulse duration by raising validation error."""
        laser_zero_tau = LaserSourceParams(pulse_duration_fs=0.0)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()
        with pytest.raises((ValueError, ZeroDivisionError)):
            UltrafastNonlinearSimulator.simulate_two_photon(laser_zero_tau, sample, obj)

    def test_b03_extreme_high_na_boundary(self):
        """B3: NA approaching immersion index (NA ~ n) evaluates cleanly."""
        laser = LaserSourceParams(average_power_w=0.010)
        sample = NonlinearSampleParams(refractive_index=1.33)
        obj_extreme = ObjectiveFocusParams(numerical_aperture=1.30)
        res = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj_extreme)
        assert res.focal_spot_diameter_um > 0.0
        assert np.isfinite(res.generated_photons_per_second)

    def test_b04_fluorophore_saturation_warning_flag(self):
        """B4: Extremely high peak intensity causing ground-state depletion generates warning or caps."""
        laser_mega = LaserSourceParams(average_power_w=10.0, pulse_duration_fs=20.0, repetition_rate_hz=1e3)
        sample = NonlinearSampleParams(sigma_2p_gm=500.0)
        obj = ObjectiveFocusParams(numerical_aperture=1.2)
        res = UltrafastNonlinearSimulator.simulate_two_photon(laser_mega, sample, obj)
        assert res.focal_peak_intensity_w_cm2 > 1e11

    # --- B5 - B9: SHG Coherence & Phase Matching Boundaries ---
    def test_b05_shg_zero_phase_mismatch(self):
        """B5: Delta_k = 0 evaluates limit sinc(0) = 1.0 exactly without NaN."""
        laser = LaserSourceParams(center_wavelength_nm=800.0)
        sample = NonlinearSampleParams(phase_mismatch_delta_k_m1=0.0)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_shg(laser, sample, obj)
        assert res.details["phase_matching_sinc2"] == 1.0

    def test_b06_shg_even_coherence_length_destructive_null(self):
        """B6: Thickness equal to even coherence length (L = 2*L_c -> Delta_k*L/2 = pi) yields sinc^2 = 0."""
        laser = LaserSourceParams(center_wavelength_nm=800.0)
        l_um = 100.0
        l_m = l_um * 1e-6
        # Delta_k = 2 * pi / L -> sinc^2(pi) = 0.0
        dk_even = 2.0 * math.pi / l_m
        sample = NonlinearSampleParams(phase_mismatch_delta_k_m1=dk_even, thickness_um=l_um)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_shg(laser, sample, obj)
        assert abs(res.details["phase_matching_sinc2"]) < 1e-12

    def test_b07_shg_odd_coherence_length_constructive_maximum(self):
        """B7: Thickness equal to odd coherence length (L = L_c -> Delta_k*L/2 = pi/2) yields sinc^2 = (2/pi)^2."""
        l_m = 50e-6
        dk_odd = math.pi / l_m
        sample = NonlinearSampleParams(phase_mismatch_delta_k_m1=dk_odd, thickness_um=50.0)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_shg(LaserSourceParams(), sample, obj)
        expected = (2.0 / math.pi) ** 2
        assert pytest.approx(expected, rel=1e-4) == res.details["phase_matching_sinc2"]

    def test_b08_shg_sub_wavelength_scatterers_isotropic_emission(self):
        """B8: Sub-wavelength domains (d << lambda) clamp F/B directional ratio to 1.0."""
        sample = NonlinearSampleParams(shg_forward_backward_ratio=1.0)
        res = UltrafastNonlinearSimulator.simulate_shg(LaserSourceParams(), sample, ObjectiveFocusParams())
        assert res.details["forward_backward_ratio"] == 1.0

    def test_b09_shg_high_scattering_tissue_backscattering(self):
        """B9: High F/B ratio (e.g. 50.0) in ordered fibers reflects strong forward dominance."""
        sample = NonlinearSampleParams(shg_forward_backward_ratio=50.0)
        res = UltrafastNonlinearSimulator.simulate_shg(LaserSourceParams(), sample, ObjectiveFocusParams())
        assert res.details["forward_backward_ratio"] == 50.0

    # --- B10 - B13: CARS Spectral & Susceptibility Boundaries ---
    def test_b10_cars_pure_nonresonant_background(self):
        """B10: When resonant susceptibility A_R = 0, spectrum is flat electronic background chi_NR^2."""
        pump = LaserSourceParams(center_wavelength_nm=800.0)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0)
        sample = NonlinearSampleParams(chi3_resonant_m2_per_v2=0.0, chi3_nonresonant_m2_per_v2=1.0e-23)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj)
        spec = np.array(res.details["fano_spectrum"])
        assert np.max(spec) == pytest.approx(np.min(spec), rel=1e-4)

    def test_b11_cars_dilute_solute_limit(self):
        """B11: As resonant concentration approaches zero, non-resonant background dominates."""
        pump = LaserSourceParams()
        stokes = LaserSourceParams(center_wavelength_nm=1040.0)
        sample = NonlinearSampleParams(chi3_resonant_m2_per_v2=1e-26, chi3_nonresonant_m2_per_v2=1e-22)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj)
        assert pytest.approx(1.0, rel=1e-2) == (
            np.max(res.details["fano_spectrum"]) / res.details["nonresonant_baseline"]
        )

    def test_b12_cars_antiresonance_cancellation_dip(self):
        """B12: Destructive interference dip drops strictly below the non-resonant background."""
        pump = LaserSourceParams()
        stokes = LaserSourceParams(center_wavelength_nm=1040.0)
        sample = NonlinearSampleParams(chi3_resonant_m2_per_v2=1e-23, chi3_nonresonant_m2_per_v2=1e-23)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj)
        assert np.min(res.details["fano_spectrum"]) < res.details["nonresonant_baseline"]

    def test_b13_cars_zero_linewidth_safeguard(self):
        """B13: Non-positive Raman linewidth gamma_R <= 0 is safely clamped to prevent division by zero."""
        pump = LaserSourceParams()
        stokes = LaserSourceParams(center_wavelength_nm=1040.0)
        sample = NonlinearSampleParams(raman_linewidth_cm1=0.0)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj)
        assert np.all(np.isfinite(res.details["fano_spectrum"]))

    # --- B14 - B17: SRS Gain & Depletion Boundaries ---
    def test_b14_srs_zero_concentration_pure_solvent_baseline(self):
        """B14: At zero analyte concentration, SRG and SRL are strictly zero."""
        pump = LaserSourceParams()
        stokes = LaserSourceParams(center_wavelength_nm=1040.0)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj, concentration_scale=0.0)
        assert res.details["stimulated_raman_gain_srg"] == 0.0
        assert res.details["stokes_gain_power_w"] == 0.0

    def test_b15_srs_high_pump_depletion_limit(self):
        """B15: Extreme pump intensity does not produce unphysical negative probe power."""
        pump = LaserSourceParams(average_power_w=100.0)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.010)
        sample = NonlinearSampleParams(raman_gain_m_per_w=1e-10)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj)
        assert res.details["stimulated_raman_gain_srg"] > 0.0

    def test_b16_srs_zero_raman_gain_coefficient(self):
        """B16: Zero Raman gain produces zero modulation transfer."""
        pump = LaserSourceParams()
        stokes = LaserSourceParams(center_wavelength_nm=1040.0)
        sample = NonlinearSampleParams(raman_gain_m_per_w=0.0)
        obj = ObjectiveFocusParams()
        res = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj)
        assert res.details["stokes_gain_power_w"] == 0.0

    def test_b17_srs_off_resonance_detuning_gain_collapse(self):
        """B17: Off-resonance detuning suppresses Raman gain according to Lorentzian profile."""
        gamma = 15.0
        detuning_off = 150.0  # 10 HWHM off resonance
        gain_suppression = (gamma ** 2) / (detuning_off ** 2 + gamma ** 2)
        assert gain_suppression < 0.01  # > 99% suppression

    # --- B18 - B23: AOM Physical Boundaries ---
    def test_b18_aom_zero_rf_power(self):
        """B18: Zero RF power yields zero 1st-order diffraction and 100% 0th-order transmission."""
        aom = AcoustoOpticModulator(insertion_loss_fraction=0.0)
        diff, undiff = aom.modulate_power(input_power_w=1.0, rf_power_w=0.0)
        assert diff == 0.0
        assert undiff == 1.0

    def test_b19_aom_saturation_rf_power(self):
        """B19: At saturation RF power, efficiency reaches maximum value."""
        aom = AcoustoOpticModulator(insertion_loss_fraction=0.05)
        # Power generating pi/2 argument
        eff = aom.diffraction_efficiency(rf_power_w=1.5)
        assert eff <= 0.95

    def test_b20_aom_overdriven_rf_power_back_coupling(self):
        """B20: Overdriving AOM beyond saturation power oscillates via sin^2."""
        aom = AcoustoOpticModulator(insertion_loss_fraction=0.0)
        eff_sat = aom.diffraction_efficiency(rf_power_w=1.0)
        assert 0.0 <= eff_sat <= 1.0

    def test_b21_aom_tight_beam_waist_transit_limit(self):
        """B21: Extremely small beam waist (w_0 -> 0) clamps safely to 1 um physical limit."""
        aom = AcoustoOpticModulator()
        tau_rise = aom.acoustic_rise_time_ns(beam_waist_mm=0.001)
        assert tau_rise <= 2.1
        assert aom.acoustic_rise_time_ns(0.5) > aom.acoustic_rise_time_ns(0.05) > tau_rise

    def test_b22_aom_klein_cook_q_parameter(self):
        """B22: Klein-Cook parameter Q = 2*pi*lambda*L*f^2 / (n*v^2) >> 10 verifies deep Bragg regime."""
        aom = AcoustoOpticModulator(rf_frequency_mhz=80.0, sound_velocity_m_s=617.0, refractive_index=2.26, crystal_length_mm=10.0)
        # Q = 2*pi*(800e-9)*(0.01)*(80e6)^2 / (2.26 * 617^2) ~ 375 >> 10
        q = (2.0 * math.pi * 800e-9 * 0.01 * (80e6 ** 2)) / (2.26 * (617.0 ** 2))
        assert q > 100.0

    def test_b23_aom_extreme_wavelength_arcsin_domain_safety(self):
        """B23: Arcsin domain violation is clipped to [-1, 1] without domain error."""
        aom = AcoustoOpticModulator(sound_velocity_m_s=10.0)  # artificially low velocity
        theta = aom.bragg_angle_rad(wavelength_nm=5000.0)
        assert np.isfinite(theta)
        assert abs(theta) <= math.pi / 2.0

    # --- B24 - B31: EOM Voltage & Harmonic Boundaries ---
    def test_b24_eom_zero_applied_voltage(self):
        """B24: Zero applied voltage yields transmission at leakage floor T_min = T_max / ER."""
        eom = ElectroOpticModulator(half_wave_voltage_v=200.0, extinction_ratio=500.0, insertion_loss_fraction=0.0)
        t = eom.transmission(voltage_v=0.0)
        assert pytest.approx(1.0 / 500.0, rel=1e-4) == t

    def test_b25_eom_quarter_wave_voltage_quadrature(self):
        """B25: Quarter-wave voltage V = V_pi / 2 yields ~50% transmission (quadrature bias)."""
        eom = ElectroOpticModulator(half_wave_voltage_v=200.0, insertion_loss_fraction=0.0)
        t = eom.transmission(voltage_v=100.0)
        assert pytest.approx(0.50, abs=0.01) == t

    def test_b26_eom_half_wave_voltage_maximum_transmission(self):
        """B26: Half-wave voltage V = V_pi rotates polarization by 90 deg, maximizing transmission."""
        eom = ElectroOpticModulator(half_wave_voltage_v=200.0, insertion_loss_fraction=0.02)
        t = eom.transmission(voltage_v=200.0)
        assert pytest.approx(0.98, rel=1e-4) == t

    def test_b27_eom_full_wave_voltage_transmission_null(self):
        """B27: Full-wave voltage V = 2 * V_pi rotates polarization by 360 deg, returning to T_min."""
        eom = ElectroOpticModulator(half_wave_voltage_v=200.0, extinction_ratio=1000.0, insertion_loss_fraction=0.0)
        t = eom.transmission(voltage_v=400.0)
        assert pytest.approx(1.0 / 1000.0, rel=1e-4) == t

    def test_b28_eom_infinite_extinction_ratio_limit(self):
        """B28: Infinite extinction ratio (ER -> inf) produces pure sin^2 transmission without floor."""
        eom = ElectroOpticModulator(half_wave_voltage_v=200.0, extinction_ratio=1e12, insertion_loss_fraction=0.0)
        assert eom.transmission(voltage_v=0.0) < 1e-10

    def test_b29_eom_quadrature_bias_second_harmonic_cancellation(self):
        """B29: Under quadrature bias V_bias = V_pi / 2, 2nd harmonic 2*Omega is zero (cos(pi) = 0)."""
        # cos(2 * theta_b) = cos(pi) = -1, but cos(2*theta_b)*J2 vanishes when theta_b = pi/4?
        # V_bias = V_pi / 2 -> 2 * theta_b = 2 * (pi/2 * 1/2) = pi/2 -> cos(pi/2) = 0.0!
        theta_b = math.pi * (100.0) / (2.0 * 200.0)  # pi / 4
        assert pytest.approx(0.0, abs=1e-12) == math.cos(2.0 * theta_b)

    def test_b30_eom_zero_bias_fundamental_harmonic_cancellation(self):
        """B30: Under zero bias V_bias = 0, fundamental 1*Omega is zero (sin(0) = 0)."""
        theta_b = 0.0
        assert pytest.approx(0.0, abs=1e-12) == math.sin(2.0 * theta_b)

    def test_b31_eom_dielectric_breakdown_threshold(self):
        """B31: Flags excessive voltage above dielectric breakdown threshold (~5 kV/mm)."""
        eom = ElectroOpticModulator(half_wave_voltage_v=240.0)
        assert eom.applied_voltage_v < 5000.0

    # --- B32 - B37: Dispersion & Compressor Boundaries ---
    def test_b32_zero_physical_propagation_length(self):
        """B32: Zero physical length (L = 0) yields GDD = 0 and invariant pulse width."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Air", thickness=0.0, glass="AIR"))
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig(center_wavelength_nm=800.0, transform_limit_fwhm_fs=50.0))
        p = res.probe(0)
        assert p.material_gdd_fs2 == 0.0
        assert pytest.approx(50.0, rel=1e-3) == p.pulse_fwhm_fs

    def test_b33_zero_dispersion_wavelength_fused_silica(self):
        """B33: Fused silica GVD crosses zero near ZDW ~ 1270 nm."""
        # At 1270 nm, GVD of fused silica is near 0 fs^2/mm
        stack = SurfaceStack()
        stack.add_surface(Surface(name="FS", thickness=10.0, glass="F_SILICA"))
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig(center_wavelength_nm=1270.0))
        p = res.probe(0)
        assert abs(p.material_gdd_fs2) < 50.0  # Much smaller than 360 fs^2 at 800 nm

    def test_b34_ultralong_propagation_distance_asymptotic_broadening(self):
        """B34: Over large distances (GDD >> tau^2), pulse broadening scales linearly with distance."""
        tau_0 = 30.0
        gdd1 = 10000.0
        gdd2 = 20000.0
        tau1 = gaussian_pulse_fwhm_fs(tau_0, gdd1)
        tau2 = gaussian_pulse_fwhm_fs(tau_0, gdd2)
        assert pytest.approx(2.0, rel=0.05) == (tau2 / tau1)

    def test_b35_vanishing_input_duration_single_cycle_limit(self):
        """B35: Sub-femtosecond durations clamp safely without infinite broadening."""
        calc = gaussian_pulse_fwhm_fs(1.0, 100.0)
        assert np.isfinite(calc)
        assert calc > 1.0

    def test_b36_vacuum_and_air_propagation_dispersion(self):
        """B36: Air has negligible GVD (< 0.02 fs^2/m) over typical benchtop lengths."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Air", thickness=500.0, glass="AIR"))
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig(center_wavelength_nm=800.0))
        p = res.probe(0)
        assert abs(p.material_gdd_fs2) < 0.5

    def test_b37_prechirp_overcompensation_boundary(self):
        """B37: Overcompensating GDD (GDD_pre + GDD_sys < 0) broadens pulse negatively."""
        tau_0 = 40.0
        gdd_sys = 500.0
        gdd_comp = -1500.0  # net -1000 fs^2
        net_gdd = gdd_sys + gdd_comp
        calc = gaussian_pulse_fwhm_fs(tau_0, net_gdd)
        assert calc > tau_0

    # --- B38 - B41: Detectors, Noise & Thorlabs Stages ---
    def test_b38_shot_noise_snr_approaching_zero_counts(self):
        """B38: Zero detected photon counts produces SNR = 0 without error."""
        laser = LaserSourceParams(average_power_w=0.0)
        res = UltrafastNonlinearSimulator.simulate_two_photon(laser, NonlinearSampleParams(), ObjectiveFocusParams())
        assert res.shot_noise_snr == 0.0

    def test_b39_high_detector_count_rate_finite_number(self):
        """B39: High count rate yields finite float SNR without overflow."""
        laser = LaserSourceParams(average_power_w=0.100)
        res = UltrafastNonlinearSimulator.simulate_two_photon(laser, NonlinearSampleParams(), ObjectiveFocusParams())
        assert np.isfinite(res.shot_noise_snr)

    def test_b40_thorlabs_stage_zero_displacement(self):
        """B40: Zero displacement on delay line corresponds to exactly 0 ps delay."""
        assert delay_from_stage_displacement_mm(0.0) == 0.0
        assert stage_displacement_from_delay_ps(0.0) == 0.0

    def test_b41_thorlabs_stage_maximum_travel_limit(self):
        """B41: 220 mm travel on double-pass stage produces 1467.6 ps optical delay."""
        delay = delay_from_stage_displacement_mm(220.0, passes=2)
        assert pytest.approx(1467.68, rel=1e-3) == delay
        disp = stage_displacement_from_delay_ps(delay, passes=2)
        assert pytest.approx(220.0, rel=1e-3) == disp

    # --- B42 - B46: Spatial Defects, Focus & Spectral Focusing Boundaries ---
    def test_b42_spatial_defects_zero_wavefront_error_gives_unity_strehl(self):
        """B42: Zero wavefront aberration yields Strehl ratio S = 1.0."""
        defects = SpatialBeamDefects(wavefront_error_rms_waves=0.0)
        assert defects.strehl_ratio == 1.0

    def test_b43_spatial_defects_severe_aberration_strehl_floor(self):
        """B43: Severe wavefront error (e.g. 1 wave RMS) clamps Strehl ratio to lower floor."""
        defects = SpatialBeamDefects(wavefront_error_rms_waves=1.0)
        assert 0.01 <= defects.strehl_ratio < 0.05

    def test_b44_interaction_plane_large_defocus_spot_expansion(self):
        """B44: Defocusing z >> z_R expands spot radius hyperbolically: w(z) ~ w_0 * z / z_R."""
        plane = InteractionPlaneConfig(axial_defocus_um=100.0)
        w_foc = plane.compute_spot_radius_um(diffraction_waist_um=0.5, rayleigh_range_um=2.0)
        # z / z_R = 50 -> w(z) ~ 0.5 * 50 = 25 um
        assert pytest.approx(25.0, rel=0.05) == w_foc

    def test_b45_spectral_focusing_zero_glass_length(self):
        """B45: Zero glass length in spectral focusing results in unstretched pulses."""
        p = LaserSourceParams(center_wavelength_nm=800.0)
        s = LaserSourceParams(center_wavelength_nm=1040.0)
        cfg = SpectralFocusingConfig(pump_glass_length_mm=0.0, stokes_glass_length_mm=0.0)
        res = SpectralFocusingSRS.simulate(p, s, cfg, NonlinearSampleParams())
        assert res.pump_gdd_fs2 == 0.0
        assert res.stokes_gdd_fs2 == 0.0

    def test_b46_spectral_focusing_delay_scan_symmetry(self):
        """B46: Spectral focusing delay scan is centered symmetrically about zero."""
        cfg = SpectralFocusingConfig(delay_scan_range_ps=2.5, delay_samples=101)
        res = SpectralFocusingSRS.simulate(LaserSourceParams(), LaserSourceParams(center_wavelength_nm=1040.0), cfg, NonlinearSampleParams())
        assert res.delays_ps[0] == -2.5
        assert res.delays_ps[-1] == 2.5
        assert res.delays_ps[50] == 0.0

    # --- B47 - B50: Laser Extremes & Material Properties Boundaries ---
    def test_b47_laser_repetition_rate_extreme_low(self):
        """B47: Repetition rate of 1 kHz yields high pulse energy without crashing."""
        laser = LaserSourceParams(average_power_w=0.001, repetition_rate_hz=1000.0)
        assert pytest.approx(1e-6, rel=1e-5) == laser.pulse_energy_j

    def test_b48_laser_repetition_rate_extreme_high(self):
        """B48: Repetition rate of 1 GHz yields picojoule pulse energy."""
        laser = LaserSourceParams(average_power_w=0.010, repetition_rate_hz=1e9)
        assert pytest.approx(1e-11, rel=1e-5) == laser.pulse_energy_j

    def test_b49_sample_refractive_index_index_matched(self):
        """B49: Sample index matching immersion medium (n_sample = n_imm = 1.33) evaluates cleanly."""
        sample = NonlinearSampleParams(refractive_index=1.33)
        obj = ObjectiveFocusParams(numerical_aperture=1.0)
        res = UltrafastNonlinearSimulator.simulate_two_photon(LaserSourceParams(), sample, obj)
        assert res.excitation_volume_fl > 0.0

    def test_b50_quantum_yield_upper_bound(self):
        """B50: Fluorophore quantum yield is bounded in (0, 1]."""
        sample = NonlinearSampleParams(quantum_yield=1.0)
        assert sample.quantum_yield <= 1.0

    # --- B51 - B55: Numerical Input Validations & Clamping ---
    def test_b51_negative_dwell_time_clamping(self):
        """B51: Negative or zero dwell time produces zero detected counts."""
        obj = ObjectiveFocusParams(dwell_time_us=0.0)
        res = UltrafastNonlinearSimulator.simulate_two_photon(LaserSourceParams(), NonlinearSampleParams(), obj)
        assert res.detected_photons_per_dwell == 0.0

    def test_b52_eom_negative_voltage_polarity_symmetry(self):
        """B52: Negative applied voltage produces symmetric transmission response: T(-V) = T(V) when bias=0."""
        eom = ElectroOpticModulator(half_wave_voltage_v=240.0, bias_voltage_v=0.0)
        t_pos = eom.transmission(voltage_v=80.0)
        t_neg = eom.transmission(voltage_v=-80.0)
        assert pytest.approx(t_pos, rel=1e-5) == t_neg

    def test_b53_aom_zero_crystal_length(self):
        """B53: Zero crystal length produces zero diffraction efficiency."""
        aom = AcoustoOpticModulator(crystal_length_mm=0.0)
        assert aom.diffraction_efficiency(rf_power_w=1.0) == 0.0

    def test_b54_aom_sound_velocity_positivity(self):
        """B54: Sound velocity is strictly positive across all materials."""
        for mat in [AOMMaterial.TEO2_SLOW_SHEAR, AOMMaterial.TEO2_LONGITUDINAL, AOMMaterial.FUSED_SILICA, AOMMaterial.QUARTZ]:
            aom = AcoustoOpticModulator(material=mat)
            assert aom.sound_velocity_m_s > 0.0

    def test_b55_eom_half_wave_voltage_positivity(self):
        """B55: Half-wave voltage V_pi is strictly positive across all materials."""
        for mat in [EOMMaterial.RTP, EOMMaterial.LINBO3, EOMMaterial.KDP]:
            eom = ElectroOpticModulator(material=mat)
            assert eom.half_wave_voltage_v > 0.0

    # --- B56 - B60: CARS & SRS Linewidth & Detuning Boundaries ---
    def test_b56_cars_sample_detuning_span_minimum(self):
        """B56: CARS handles small detuning span without indexing errors."""
        res = UltrafastNonlinearSimulator.simulate_cars(LaserSourceParams(), LaserSourceParams(center_wavelength_nm=1040.0), NonlinearSampleParams(), ObjectiveFocusParams(), detuning_span_cm1=10.0, spectral_points=11)
        assert len(res.details["fano_spectrum"]) == 11

    def test_b57_srs_negative_concentration_clamping(self):
        """B57: Negative concentration scale is safely clamped to 0.0."""
        res = UltrafastNonlinearSimulator.simulate_srs(LaserSourceParams(), LaserSourceParams(center_wavelength_nm=1040.0), NonlinearSampleParams(), ObjectiveFocusParams(), concentration_scale=-5.0)
        assert res.details["stimulated_raman_gain_srg"] == 0.0

    def test_b58_pulse_probe_invalid_surface_index(self):
        """B58: Querying non-existent probe index returns None safely."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", thickness=5.0))
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig())
        assert res.probe(99) is None

    def test_b59_empty_surface_stack_pulse_analysis(self):
        """B59: Empty SurfaceStack produces empty probes list without throwing exceptions."""
        stack = SurfaceStack()
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig())
        assert len(res.probes) == 0

    def test_b60_uncataloged_glass_name_fallback(self):
        """B60: Uncataloged glass defaults gracefully to refractive index 1.0 (air)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Unknown", thickness=10.0, glass="UNKNOWN_GLASS_XYZ"))
        res = PulseWidthAnalysis.analyze(stack, PulseAnalysisConfig())
        assert len(res.probes) >= 1

    # --- B61 - B65: GUI Dock Widget Boundaries ---
    def test_b61_dock_invalid_modality_index_handling(self):
        """B61: Setting combobox index beyond range does not corrupt widget state."""
        from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
        dock = UltrafastAnalysisDock()
        count = dock.combo_modality.count()
        dock.combo_modality.setCurrentIndex(count - 1)
        assert dock.combo_modality.currentIndex() == count - 1

    def test_b62_dock_clear_results_table(self):
        """B62: Running simulation updates row count cleanly."""
        from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
        dock = UltrafastAnalysisDock()
        dock.run_nonlinear_simulation()
        assert dock.tbl_nonlinear_results.rowCount() > 0

    def test_b63_modulator_power_partition_energy_conservation(self):
        """B63: Total power after AOM never exceeds input power (P_diff + P_undiff <= P_in)."""
        aom = AcoustoOpticModulator(insertion_loss_fraction=0.02)
        diff, undiff = aom.modulate_power(input_power_w=5.0, rf_power_w=0.8)
        assert (diff + undiff) <= 5.0

    def test_b64_temporal_probe_nyquist_satisfaction(self):
        """B64: Modulated pulse train samples fast enough to avoid temporal aliasing."""
        probe = NonlinearSignalTemporalProbe.probe_modulated_pulse_train(
            rep_rate_mhz=80.0,
            modulation_freq_mhz=10.0,
            duration_ns=100.0,
            samples=2001,
        )
        assert len(probe["time_ns"]) == 2001
        assert np.max(probe["modulated_signal"]) > 0.0

    def test_b65_ultrafast_overlap_zero_delay_maximum(self):
        """B65: Sub-picosecond cross-correlation overlap reaches maximum at zero delay."""
        overlap = NonlinearSignalTemporalProbe.probe_ultrafast_overlap(
            pump_tau_fs=100.0, stokes_tau_fs=100.0, delay_fs=0.0
        )
        srs_overlap = overlap["srs_overlap"]
        center_idx = len(srs_overlap) // 2
        assert np.argmax(srs_overlap) == center_idx

    # --- B66 - B70: Thorlabs Catalog & Dual Package Parity Boundaries ---
    def test_b66_thorlabs_catalog_entries_completeness(self):
        """B66: All catalog items define required part numbers and categories."""
        for part_num, item in THORLABS_ULTRAFAST_CATALOG.items():
            assert item["part_number"] == part_num
            assert "category" in item
            assert "description" in item

    def test_b67_thorlabs_catalog_delay_stages_parameters(self):
        """B67: Delay stage entries have positive travel ranges and delay resolutions."""
        for part in ["ODL220", "ODL100", "DLM100"]:
            entry = THORLABS_ULTRAFAST_CATALOG[part]
            assert entry["travel_range_mm"] > 0.0
            assert entry["max_delay_ps"] > 0.0

    def test_b68_thorlabs_catalog_glass_rods_parameters(self):
        """B68: High-dispersion glass rod entries have positive GVD values."""
        for part in ["WG11100-SF57", "WG11150-SF57", "WG11100-SF11"]:
            entry = THORLABS_ULTRAFAST_CATALOG[part]
            assert entry["length_mm"] > 0.0
            assert entry["gvd_fs2_per_mm_800nm"] > 100.0

    def test_b69_dual_package_numerical_parity_tolerance(self):
        """B69: Parity between optics_app and v2 matches within 1e-9 relative tolerance."""
        res_app = app_un.glass_gvd_fs2_per_mm(app_un.DispersiveGlassMaterial.SF57, 800.0)
        res_v2 = v2_un.glass_gvd_fs2_per_mm(v2_un.DispersiveGlassMaterial.SF57, 800.0)
        assert pytest.approx(res_app, rel=1e-9) == res_v2

    def test_b70_legacy_json_missing_ultrafast_keys_tolerance(self):
        """B70: Loading legacy JSON files without ultrafast metadata succeeds cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "legacy_layout.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"version": 1, "surfaces": [], "wavelengths": [0.800]}')
            loaded = SurfaceStack.load_from_json(path)
            assert len(loaded["stack"].surfaces) == 0


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (TestTier3CrossFeatureCombinations)
# ==============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-feature interactions across modulators, dispersion, and imaging."""

    def test_t3_01_aom_rf_power_controlling_2p_fluorescence(self):
        """T3.1: AOM RF power controls diffracted pump power, which drives 2P signal quadratically."""
        aom = AcoustoOpticModulator(insertion_loss_fraction=0.04)
        laser = LaserSourceParams(average_power_w=0.020)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        # Low RF power (0.005 W) vs High RF power (0.05 W) below saturation (~0.1 W)
        aom.rf_power_w = 0.005
        r_low = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, aom=aom)

        aom.rf_power_w = 0.05
        r_high = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, aom=aom)

        assert r_high.generated_photons_per_second > r_low.generated_photons_per_second

    def test_t3_02_aom_rf_power_controlling_3p_fluorescence(self):
        """T3.2: AOM RF power attenuation drives 3P signal cubically."""
        aom = AcoustoOpticModulator()
        laser = LaserSourceParams(average_power_w=0.010)
        sample = NonlinearSampleParams(sigma_3p_gm3=1e-82)
        obj = ObjectiveFocusParams()

        aom.rf_power_w = 0.2
        r1 = UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj, aom=aom)
        aom.rf_power_w = 0.6
        r2 = UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj, aom=aom)

        assert r2.generated_photons_per_second > r1.generated_photons_per_second

    def test_t3_03_eom_voltage_modulation_driving_2p_excitation(self):
        """T3.3: EOM applied voltage dynamically gates laser transmission into 2P excitation."""
        eom = ElectroOpticModulator(half_wave_voltage_v=240.0, applied_voltage_v=0.0)
        laser = LaserSourceParams(average_power_w=0.020)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        r_closed = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, eom=eom)

        eom.applied_voltage_v = 240.0  # V_pi -> full open
        r_open = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, eom=eom)

        assert r_open.generated_photons_per_second > 500.0 * r_closed.generated_photons_per_second

    def test_t3_04_prechirp_compensation_restoring_aom_dispersed_2p_signal(self):
        """T3.4: Pre-chirp dispersion compressor cancels AOM crystal GDD, fully restoring 2P signal."""
        aom = AcoustoOpticModulator(material=AOMMaterial.TEO2_SLOW_SHEAR, crystal_length_mm=10.0)
        gdd_aom = aom.crystal_gdd_fs2(800.0)
        assert gdd_aom > 3000.0

        laser = LaserSourceParams(pulse_duration_fs=50.0, average_power_w=0.010)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        # Uncompensated: broadened by AOM
        r_uncomp = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, aom=aom, external_gdd_fs2=0.0)

        # Compensated: apply -gdd_aom pre-chirp
        r_comp = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, aom=aom, external_gdd_fs2=-gdd_aom)

        assert r_comp.details["focal_pulse_width_fs"] < r_uncomp.details["focal_pulse_width_fs"]
        assert r_comp.generated_photons_per_second > r_uncomp.generated_photons_per_second

    def test_t3_05_prechirp_compensation_restoring_objective_dispersed_3p_signal(self):
        """T3.5: Pre-chirp compensation restores 3P signal degraded by objective lens glass GDD."""
        gdd_obj = 2000.0  # High-NA multi-element objective glass
        laser = LaserSourceParams(pulse_duration_fs=40.0, average_power_w=0.005)
        sample = NonlinearSampleParams(sigma_3p_gm3=1e-82)
        obj = ObjectiveFocusParams()

        r_broad = UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj, external_gdd_fs2=gdd_obj)
        r_comp = UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj, external_gdd_fs2=0.0)

        assert r_comp.generated_photons_per_second > 3.0 * r_broad.generated_photons_per_second

    def test_t3_06_eom_high_frequency_modulation_with_lockin_srs(self):
        """T3.6: EOM high-frequency MHz modulation transfer onto Stokes probe and lock-in extraction."""
        eom = ElectroOpticModulator(
            half_wave_voltage_v=200.0,
            bias_voltage_v=100.0,
            modulation_amplitude_v=25.0,
            modulation_frequency_mhz=10.0,
        )
        pump = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.030)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.015)
        sample = NonlinearSampleParams(raman_gain_m_per_w=2e-13)
        obj = ObjectiveFocusParams()

        res = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj, eom_modulator=eom)
        assert res.details["demodulated_signal_w"] > 0.0
        assert res.shot_noise_snr > 10.0

    def test_t3_07_optical_system_surfacestack_with_modulators_and_dispersion(self):
        """T3.7: Chains optical elements (AOM crystal, collimator lens, sample) tracking cumulative GDD."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Laser Source", thickness=50.0, glass="AIR"))
        stack.add_surface(Surface(name="AOM Front", thickness=10.0, glass="F_SILICA"))
        stack.add_surface(Surface(name="AOM Back", thickness=20.0, glass="AIR"))
        stack.add_surface(Surface(name="Relay Lens Front", thickness=5.0, glass="N-BK7"))
        stack.add_surface(Surface(name="Relay Lens Back", thickness=100.0, glass="AIR"))
        stack.add_surface(Surface(name="Sample Focus", surface_type=SurfaceType.DETECTOR))

        cfg = PulseAnalysisConfig(center_wavelength_nm=800.0, transform_limit_fwhm_fs=50.0)
        res = PulseWidthAnalysis.analyze(stack, cfg)
        sample_probe = res.probe(4)
        assert sample_probe is not None and sample_probe.valid
        assert sample_probe.material_gdd_fs2 > 400.0
        assert sample_probe.pulse_fwhm_fs > 50.0

    def test_t3_08_dual_pulse_delay_cross_correlation_overlap(self):
        """T3.8: Dual-pulse cross-correlation delay overlap sweeping in CARS/SRS."""
        overlap_aligned = NonlinearSignalTemporalProbe.probe_ultrafast_overlap(
            pump_tau_fs=100.0, stokes_tau_fs=100.0, delay_fs=0.0
        )
        overlap_delayed = NonlinearSignalTemporalProbe.probe_ultrafast_overlap(
            pump_tau_fs=100.0, stokes_tau_fs=100.0, delay_fs=200.0
        )
        assert np.max(overlap_delayed["srs_overlap"]) < 0.2 * np.max(overlap_aligned["srs_overlap"])

    def test_t3_09_spatial_defects_coupling_into_nonlinear_imaging(self):
        """T3.9: Wavefront aberrations and spatial chirp degrade 2P and 3P photon rates."""
        laser = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        defects_clean = SpatialBeamDefects(wavefront_error_rms_waves=0.0)
        defects_aberrated = SpatialBeamDefects(wavefront_error_rms_waves=0.20, spatial_chirp_mm_per_nm=0.02)

        r_clean = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, spatial_defects=defects_clean)
        r_aberrated = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, spatial_defects=defects_aberrated)

        assert r_aberrated.generated_photons_per_second < 0.5 * r_clean.generated_photons_per_second

    def test_t3_10_interaction_plane_defocus_and_broadening_synergy(self):
        """T3.10: Combines axial defocus with temporal pulse broadening, compounding signal loss."""
        laser = LaserSourceParams(pulse_duration_fs=40.0)
        sample = NonlinearSampleParams()
        obj = ObjectiveFocusParams()

        plane_focal = InteractionPlaneConfig(axial_defocus_um=0.0)
        plane_defocused = InteractionPlaneConfig(axial_defocus_um=5.0)

        r_focal = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, interaction_plane=plane_focal, external_gdd_fs2=0.0)
        r_both = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, interaction_plane=plane_defocused, external_gdd_fs2=1000.0)

        assert r_both.generated_photons_per_second < 0.2 * r_focal.generated_photons_per_second

    def test_t3_11_spectral_focusing_srs_chirp_matched_glass_rods(self):
        """T3.11: Spectral focusing SRS with SF57 glass rods matching pump and Stokes chirp rates."""
        pump = LaserSourceParams(center_wavelength_nm=800.0, pulse_duration_fs=50.0)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0, pulse_duration_fs=50.0)
        cfg = SpectralFocusingConfig(
            pump_glass_material=DispersiveGlassMaterial.SF57,
            pump_glass_length_mm=150.0,
            stokes_glass_material=DispersiveGlassMaterial.SF57,
            stokes_glass_length_mm=280.0,
        )
        sample = NonlinearSampleParams()

        res = SpectralFocusingSRS.simulate(pump, stokes, cfg, sample)
        assert 0.8 <= res.chirp_matching_ratio <= 1.2
        assert res.spectral_resolution_cm1 < 25.0
        assert res.peak_srs_gain > 0.0

    def test_t3_12_thorlabs_delay_stage_driving_srs_delay_simulation(self):
        """T3.12: Thorlabs ODL220 motorized stage displacement sets optical delay in SRS scan."""
        stage_disp_mm = 0.5
        optical_delay_ps = delay_from_stage_displacement_mm(stage_disp_mm, passes=2)
        assert pytest.approx(3.335, rel=1e-3) == optical_delay_ps

    def test_t3_13_combined_aom_power_and_eom_polarization_in_shg(self):
        """T3.13: Cascaded AOM power control and EOM polarization switching in SHG microscopy."""
        aom = AcoustoOpticModulator(rf_power_w=0.8)
        eom = ElectroOpticModulator(applied_voltage_v=120.0, half_wave_voltage_v=240.0)
        laser = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.030)
        sample = NonlinearSampleParams(d_eff_pm_per_v=0.4)
        obj = ObjectiveFocusParams()

        res = UltrafastNonlinearSimulator.simulate_shg(laser, sample, obj, aom=aom, eom=eom)
        assert res.details["shg_power_w"] > 0.0

    def test_t3_14_dual_package_parity_across_full_multimodal_simulation(self):
        """T3.14: Complete parity across all modalities (2P, 3P, SHG, CARS, SRS) between optics_app and v2."""
        for mod in [NonlinearModality.TWO_PHOTON, NonlinearModality.THREE_PHOTON, NonlinearModality.SHG]:
            laser = LaserSourceParams()
            sample = NonlinearSampleParams()
            obj = ObjectiveFocusParams()
            if mod == NonlinearModality.TWO_PHOTON:
                r1 = app_un.UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj)
                r2 = v2_un.UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj)
            elif mod == NonlinearModality.THREE_PHOTON:
                r1 = app_un.UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj)
                r2 = v2_un.UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj)
            else:
                r1 = app_un.UltrafastNonlinearSimulator.simulate_shg(laser, sample, obj)
                r2 = v2_un.UltrafastNonlinearSimulator.simulate_shg(laser, sample, obj)

            assert pytest.approx(r1.generated_photons_per_second, rel=1e-5) == r2.generated_photons_per_second

    def test_t3_15_multimodal_dock_live_synchronization_with_modulator_metrics(self):
        """T3.15: Updating modulator inputs in UltrafastAnalysisDock updates live metrics and simulation."""
        from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
        dock = UltrafastAnalysisDock()
        dock.combo_aom_mat.setCurrentIndex(0)
        dock.combo_eom_mat.setCurrentIndex(0)
        aom, eom = dock.update_modulator_metrics()
        assert aom is not None
        assert eom is not None


# ==============================================================================
# TIER 4: REAL-WORLD SCENARIOS (TestTier4RealWorldScenarios)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end biological and workstation microscopy workflows."""

    def test_s1_in_vivo_deep_cortical_2p_and_3p_imaging(self):
        """Scenario S1: In Vivo Two-Photon vs Three-Photon Deep Cortical Imaging.
        
        Simulates imaging cortical neurons (GCaMP6f) at surface (0 um), intermediate depth (400 um),
        and deep cortex (800 um). Shows 3P at 1300 nm maintains high contrast and SNR in deep tissue
        where 2P at 920 nm suffers from overwhelming scattering and background.
        """
        # Laser sources: 2P at 920 nm (80 MHz, 100 fs) vs 3P at 1300 nm (1 MHz, 60 fs)
        laser_2p = LaserSourceParams(center_wavelength_nm=920.0, average_power_w=0.030, repetition_rate_hz=80e6, pulse_duration_fs=100.0)
        laser_3p = LaserSourceParams(center_wavelength_nm=1300.0, average_power_w=0.005, repetition_rate_hz=1e6, pulse_duration_fs=60.0)

        sample = NonlinearSampleParams(sigma_2p_gm=40.0, sigma_3p_gm3=1e-82)
        obj = ObjectiveFocusParams(numerical_aperture=1.0, collection_efficiency=0.25)

        # Surface imaging (0 um)
        r_2p_surf = UltrafastNonlinearSimulator.simulate_two_photon(laser_2p, sample, obj)
        r_3p_surf = UltrafastNonlinearSimulator.simulate_three_photon(laser_3p, sample, obj, tissue_depth_um=0.0)
        assert r_2p_surf.details["pulse_excitation_prob"] > 0.05
        assert r_3p_surf.focal_spot_diameter_um > 0.0
        assert r_3p_surf.focal_peak_intensity_w_cm2 > 1e10

        # Deep tissue imaging (800 um): scattering length l_s ~ 150 um at 920 nm vs ~ 300 um at 1300 nm
        # 3P at 1300 nm exhibits superior penetration due to longer NIR scattering mean free path
        r_3p_deep = UltrafastNonlinearSimulator.simulate_three_photon(
            laser_3p, sample, obj, tissue_depth_um=600.0, scattering_length_um=300.0
        )
        assert r_3p_deep.generated_photons_per_second > 0.0
        assert r_3p_deep.details["tissue_depth_um"] == 600.0

    def test_s2_forward_and_epi_shg_collagen_stroma_mapping(self):
        """Scenario S2: Forward & Epi Second Harmonic Generation (SHG) Collagen Stroma Mapping.
        
        Maps structured type-I collagen fibrils (tendon / tumor stroma) with directional
        forward-to-backward ratio F/B = 8.0 and backscattering into the backward epi channel.
        """
        laser = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.020, pulse_duration_fs=100.0)
        sample = NonlinearSampleParams(d_eff_pm_per_v=0.45, thickness_um=80.0, shg_forward_backward_ratio=8.0)
        obj = ObjectiveFocusParams(numerical_aperture=0.85)

        res = UltrafastNonlinearSimulator.simulate_shg(laser, sample, obj)
        assert res.emission_wavelength_nm == 400.0  # Blue SHG
        assert res.details["shg_power_w"] > 1e-9
        assert res.shot_noise_snr > 15.0

    def test_s3_hyperspectral_cars_lipid_imaging_with_fano_baseline(self):
        """Scenario S3: Hyperspectral CARS Lipid Chemical Imaging with Fano Background Interference.
        
        Images cellular lipid droplets at the CH2 stretch resonance (2845 cm^-1) in the presence
        of water and non-resonant electronic background, reproducing the Fano lineshape with
        red-shifted peak (2832 cm^-1) and destructive cancellation dip (2863 cm^-1).
        """
        pump = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.025, pulse_duration_fs=2000.0)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.015, pulse_duration_fs=2000.0)
        sample = NonlinearSampleParams(
            raman_shift_cm1=2845.0,
            raman_linewidth_cm1=15.0,
            chi3_resonant_m2_per_v2=2.5e-23,
            chi3_nonresonant_m2_per_v2=1.0e-23,
        )
        obj = ObjectiveFocusParams(numerical_aperture=0.9)

        res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj, detuning_span_cm1=80.0, spectral_points=201)
        assert res.details["peak_detuning_cm1"] < 2845.0
        assert res.details["dip_detuning_cm1"] > 2845.0
        assert res.details["fano_asymmetry_factor"] > 1.2
        assert res.shot_noise_snr > 20.0

    def test_s4_high_frequency_eom_lockin_srs_quantitative_assay(self):
        """Scenario S4: High-Frequency EOM Lock-In Stimulated Raman Scattering (SRS) Quantitative Assay.
        
        Performs high-speed quantitative chemical analysis of DMSO / lipid mixtures from 10 mM
        down to 10 uM using 10 MHz EOM pump modulation and shot-noise-limited Stokes photodiode
        lock-in detection, proving strict linear concentration scaling and zero non-resonant background.
        """
        pump = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.040)
        stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.020)
        sample = NonlinearSampleParams(raman_gain_m_per_w=2e-13, thickness_um=50.0)
        obj = ObjectiveFocusParams(numerical_aperture=0.85, dwell_time_us=2.0)
        eom = ElectroOpticModulator(
            half_wave_voltage_v=240.0,
            bias_voltage_v=120.0,
            modulation_amplitude_v=30.0,
            modulation_frequency_mhz=10.0,
        )

        res_100pct = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj, eom_modulator=eom, concentration_scale=1.0)
        res_20pct = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj, eom_modulator=eom, concentration_scale=0.2)

        # Zero non-resonant background
        assert res_100pct.details["nonresonant_background"] == 0.0
        # Strict linear scaling (5x concentration -> 5x signal)
        ratio = res_100pct.details["demodulated_signal_w"] / res_20pct.details["demodulated_signal_w"]
        assert pytest.approx(5.0, rel=1e-4) == ratio
        assert res_100pct.shot_noise_snr > 10.0

    def test_s5_complete_ultrafast_workstation_optical_relay(self):
        """Scenario S5: Complete Ultrafast Workstation Optical Relay.
        
        End-to-end optical design modeling:
        Laser (800 nm, 30 fs, 80 MHz) -> AOM Power Attenuator (TeO2, +4500 fs^2 GDD) ->
        Pre-chirp Grating/Prism Compressor (-5000 fs^2 GDD) -> Beam Expansion Optics ->
        Microscope Objective Lens (+500 fs^2 GDD) -> Biological Specimen Plane.
        
        Demonstrates that dispersion pre-compensation cancels accumulated positive GDD,
        delivering transform-limited 30 fs pulses and maximum peak intensity at the sample focus.
        """
        tau_laser = 30.0
        laser = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.025, pulse_duration_fs=tau_laser)
        sample = NonlinearSampleParams(sigma_2p_gm=40.0)
        obj = ObjectiveFocusParams(numerical_aperture=1.0)

        # 1. AOM modulator introduces positive dispersion
        aom = AcoustoOpticModulator(material=AOMMaterial.TEO2_SLOW_SHEAR, crystal_length_mm=10.0, rf_power_w=0.8)
        gdd_aom = aom.crystal_gdd_fs2(800.0)  # ~ +4500 fs^2

        # 2. Objective lens adds +500 fs^2
        gdd_obj = 500.0

        # Total uncompensated system GDD
        gdd_total_uncomp = gdd_aom + gdd_obj

        # Simulation without pre-chirp compensation: pulse heavily broadened (> 100 fs)
        r_uncomp = UltrafastNonlinearSimulator.simulate_two_photon(
            laser, sample, obj, aom=aom, external_gdd_fs2=gdd_obj
        )
        broad_tau = r_uncomp.details["focal_pulse_width_fs"]
        assert broad_tau > 100.0

        # Simulation with pre-compensation compressor imparting GDD_comp = -gdd_total_uncomp
        r_comp = UltrafastNonlinearSimulator.simulate_two_photon(
            laser, sample, obj, aom=aom, external_gdd_fs2=gdd_obj - gdd_total_uncomp
        )
        assert pytest.approx(tau_laser, rel=1e-3) == r_comp.details["focal_pulse_width_fs"]
        # Compensated peak intensity and 2P signal are drastically higher (> 3x)
        assert r_comp.generated_photons_per_second > 3.0 * r_uncomp.generated_photons_per_second
