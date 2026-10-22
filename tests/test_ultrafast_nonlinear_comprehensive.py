"""Comprehensive Physics & Integration Tests for Ultrafast Optics & Multimodal Nonlinear Imaging.

Tests:
1. Acousto-Optic Modulator (AOM): Bragg angle, RF diffraction efficiency, acoustic rise time, crystal GDD.
2. Electro-Optic Modulator (EOM / Pockels cell): retardation, V_pi, transmission, extinction ratio, lock-in demodulation.
3. Two-Photon Fluorescence (2PEF / 2P): exact quadratic P^2 / (tau * f_rep) scaling, focal waist, saturation check.
4. Three-Photon Fluorescence (3PEF / 3P): exact cubic P^3 / (tau^2 * f_rep^2) scaling, deep tissue attenuation.
5. Second Harmonic Generation (SHG): quadratic power scaling, sinc^2 phase matching, forward-backward ratio.
6. Coherent Anti-Stokes Raman Scattering (CARS): asymmetric Fano lineshape from resonant + nonresonant interference.
7. Stimulated Raman Scattering (SRS): strict linear concentration scaling, zero nonresonant background, SRG & SRL energy balance.
8. Dispersion management & Pre-chirp compensation.
9. GUI Integration: UltrafastAnalysisDock tabs, modulators, and simulation invocation.
"""

import math
import numpy as np
import pytest

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
    DispersiveGlassMaterial,
    glass_gvd_fs2_per_mm,
    SpectralFocusingConfig,
    SpectralFocusingResult,
    SpectralFocusingSRS,
    NonlinearSignalTemporalProbe,
    SpatialBeamDefects,
    InteractionPlaneConfig,
    THORLABS_ULTRAFAST_CATALOG,
    delay_from_stage_displacement_mm,
    stage_displacement_from_delay_ps,
)
from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ==============================================================================
# 1. ACOUSTO-OPTIC MODULATOR (AOM) TESTS
# ==============================================================================

def test_aom_bragg_angle_and_deflection():
    """Verify Bragg condition: sin(theta_B) = lambda * f_RF / (2 * v_s)."""
    aom = AcoustoOpticModulator(
        material=AOMMaterial.TEO2_SLOW_SHEAR,
        rf_frequency_mhz=80.0,
        sound_velocity_m_s=617.0,
    )
    wl_nm = 800.0
    wl_m = 800.0 * 1e-9
    expected_sin = (wl_m * 80e6) / (2.0 * 617.0)
    expected_theta_b = math.asin(expected_sin)

    calc_theta_b = aom.bragg_angle_rad(wl_nm, inside_crystal=False)
    assert pytest.approx(expected_theta_b, rel=1e-5) == calc_theta_b

    # Total external deflection angle between 0th and 1st order
    calc_defl = aom.deflection_angle_rad(wl_nm)
    assert pytest.approx(2.0 * expected_theta_b, rel=1e-5) == calc_defl


def test_aom_rf_diffraction_efficiency_and_saturation():
    """Verify analytical sin^2 efficiency curve with RF power and insertion loss."""
    aom = AcoustoOpticModulator(
        material=AOMMaterial.TEO2_SLOW_SHEAR,
        rf_power_w=0.0,
        insertion_loss_fraction=0.05,
    )
    # Zero RF power -> 0% diffraction
    assert aom.diffraction_efficiency(rf_power_w=0.0) == 0.0

    # Low power linearity: eta proportional to P_rf
    eff_low1 = aom.diffraction_efficiency(rf_power_w=0.001)
    eff_low2 = aom.diffraction_efficiency(rf_power_w=0.002)
    assert pytest.approx(2.0, rel=0.05) == (eff_low2 / eff_low1)

    # Upper bound respect insertion loss (max <= 95%)
    eff_high = aom.diffraction_efficiency(rf_power_w=1.5)
    assert eff_high <= 0.95


def test_aom_acoustic_rise_time_and_bandwidth():
    """Verify acoustic rise time tau_r = 1.28 * w_0 / v_s and Delta_f = 0.66 / tau_r."""
    aom = AcoustoOpticModulator(
        material=AOMMaterial.TEO2_SLOW_SHEAR,
        sound_velocity_m_s=617.0,
    )
    waist_mm = 0.5  # 500 um
    w0_m = 0.5e-3
    expected_rise_ns = (1.28 * w0_m / 617.0) * 1e9
    calc_rise_ns = aom.acoustic_rise_time_ns(waist_mm)
    assert pytest.approx(expected_rise_ns, rel=1e-4) == calc_rise_ns

    expected_bw_mhz = (0.66 / calc_rise_ns) * 1e3
    calc_bw_mhz = aom.modulation_bandwidth_mhz(waist_mm)
    assert pytest.approx(expected_bw_mhz, rel=1e-4) == calc_bw_mhz


def test_aom_crystal_dispersion():
    """Verify positive group delay dispersion for TeO2 vs Fused Silica."""
    aom_teo2 = AcoustoOpticModulator(material=AOMMaterial.TEO2_SLOW_SHEAR, crystal_length_mm=10.0)
    gdd_teo2 = aom_teo2.crystal_gdd_fs2(800.0)
    assert gdd_teo2 > 3500.0  # TeO2 has strong positive dispersion (~ +4500 fs² for 10 mm)

    aom_fs = AcoustoOpticModulator(material=AOMMaterial.FUSED_SILICA, crystal_length_mm=10.0)
    gdd_fs = aom_fs.crystal_gdd_fs2(800.0)
    assert 340.0 <= gdd_fs <= 380.0  # Fused silica: ~36.1 fs²/mm * 10 mm = 361 fs²


# ==============================================================================
# 2. ELECTRO-OPTIC MODULATOR (EOM / POCKELS CELL) TESTS
# ==============================================================================

def test_eom_phase_retardation_and_vpi():
    """Verify linear retardation Gamma = pi * (V + V_bias) / V_pi."""
    eom = ElectroOpticModulator(
        material=EOMMaterial.RTP,
        half_wave_voltage_v=240.0,
        applied_voltage_v=120.0,
        bias_voltage_v=0.0,
    )
    # Quarter-wave voltage -> pi/2 retardation
    assert pytest.approx(math.pi / 2.0, rel=1e-5) == eom.phase_retardation_rad()

    # Half-wave voltage -> pi retardation
    assert pytest.approx(math.pi, rel=1e-5) == eom.phase_retardation_rad(voltage_v=240.0)


def test_eom_transmission_and_extinction_ratio():
    """Verify T(V) = T_min + (T_max - T_min) * sin^2(Gamma / 2) with finite extinction."""
    eom = ElectroOpticModulator(
        material=EOMMaterial.RTP,
        half_wave_voltage_v=250.0,
        insertion_loss_fraction=0.02,
        extinction_ratio=1000.0,
        bias_voltage_v=0.0,
    )
    # V = 0: minimum transmission T_min = 0.98 / 1000 = 9.8e-4
    t_zero = eom.transmission(voltage_v=0.0)
    assert pytest.approx(0.98 / 1000.0, rel=1e-4) == t_zero

    # V = V_pi (250 V): maximum transmission T_max = 1 - 0.02 = 0.98
    t_vpi = eom.transmission(voltage_v=250.0)
    assert pytest.approx(0.98, rel=1e-4) == t_vpi

    # Extinction ratio check
    assert pytest.approx(1000.0, rel=1e-3) == (t_vpi / t_zero)


def test_eom_mhz_lockin_demodulation():
    """Verify high-frequency MHz modulation and lock-in recovery of signal."""
    eom = ElectroOpticModulator(
        material=EOMMaterial.RTP,
        half_wave_voltage_v=200.0,
        bias_voltage_v=100.0,        # Biased at quadrature (pi/2)
        modulation_amplitude_v=20.0,  # 10% modulation depth
        modulation_frequency_mhz=10.0, # 10 MHz
    )
    t_ns = np.linspace(0.0, 500.0, 5001)  # 500 ns = 5 full cycles
    input_pwr = 1.0  # 1 Watt
    p_t = eom.modulate_time_series(t_ns, input_pwr)

    # Demodulate lock-in
    x_inphase, y_quad, r_mag = eom.demodulate_lockin(p_t, t_ns, phase_deg=0.0)
    assert r_mag > 0.05
    assert pytest.approx(x_inphase, abs=1e-3) == r_mag
    assert abs(y_quad) < 1e-3  # Zero quadrature component for in-phase reference


# ==============================================================================
# 3. TWO-PHOTON FLUORESCENCE (2P) TESTS
# ==============================================================================

def test_two_photon_exact_quadratic_power_scaling():
    """Verify F_2P proportional to P_avg^2."""
    laser_base = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0, repetition_rate_hz=80e6)
    sample = NonlinearSampleParams(sigma_2p_gm=36.0)
    obj = ObjectiveFocusParams(numerical_aperture=0.8)

    r_base = UltrafastNonlinearSimulator.simulate_two_photon(laser_base, sample, obj)

    laser_double = LaserSourceParams(average_power_w=0.020, pulse_duration_fs=100.0, repetition_rate_hz=80e6)
    r_double = UltrafastNonlinearSimulator.simulate_two_photon(laser_double, sample, obj)

    ratio = r_double.generated_photons_per_second / r_base.generated_photons_per_second
    assert pytest.approx(4.0, rel=1e-3) == ratio
    assert r_base.scaling_law_power_exponent == 2.0


def test_two_photon_pulse_width_and_rep_rate_scaling():
    """Verify F_2P proportional to 1 / (tau * f_rep)."""
    laser_base = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0, repetition_rate_hz=80e6)
    sample = NonlinearSampleParams(sigma_2p_gm=36.0)
    obj = ObjectiveFocusParams(numerical_aperture=0.8)
    r_base = UltrafastNonlinearSimulator.simulate_two_photon(laser_base, sample, obj)

    # Doubled pulse duration -> 0.5x signal
    laser_broad = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=200.0, repetition_rate_hz=80e6)
    r_broad = UltrafastNonlinearSimulator.simulate_two_photon(laser_broad, sample, obj)
    assert pytest.approx(0.5, rel=1e-3) == (r_broad.generated_photons_per_second / r_base.generated_photons_per_second)

    # Doubled repetition rate at fixed average power -> 0.5x signal
    laser_fast = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=100.0, repetition_rate_hz=160e6)
    r_fast = UltrafastNonlinearSimulator.simulate_two_photon(laser_fast, sample, obj)
    assert pytest.approx(0.5, rel=1e-3) == (r_fast.generated_photons_per_second / r_base.generated_photons_per_second)


# ==============================================================================
# 4. THREE-PHOTON FLUORESCENCE (3P) TESTS
# ==============================================================================

def test_three_photon_exact_cubic_power_scaling():
    """Verify F_3P proportional to P_avg^3."""
    laser_base = LaserSourceParams(average_power_w=0.005, pulse_duration_fs=80.0, repetition_rate_hz=4e6)
    sample = NonlinearSampleParams(sigma_3p_gm3=1.0e-82)
    obj = ObjectiveFocusParams(numerical_aperture=0.8)

    r_base = UltrafastNonlinearSimulator.simulate_three_photon(laser_base, sample, obj)

    laser_double = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=80.0, repetition_rate_hz=4e6)
    r_double = UltrafastNonlinearSimulator.simulate_three_photon(laser_double, sample, obj)

    ratio = r_double.generated_photons_per_second / r_base.generated_photons_per_second
    assert pytest.approx(8.0, rel=1e-3) == ratio
    assert r_base.scaling_law_power_exponent == 3.0


def test_three_photon_inverse_square_pulse_and_rep_rate_scaling():
    """Verify F_3P proportional to 1 / (tau^2 * f_rep^2)."""
    laser_base = LaserSourceParams(average_power_w=0.005, pulse_duration_fs=80.0, repetition_rate_hz=4e6)
    sample = NonlinearSampleParams(sigma_3p_gm3=1.0e-82)
    obj = ObjectiveFocusParams(numerical_aperture=0.8)
    r_base = UltrafastNonlinearSimulator.simulate_three_photon(laser_base, sample, obj)

    # Doubled pulse duration -> 1 / 2^2 = 0.25x signal
    laser_broad = LaserSourceParams(average_power_w=0.005, pulse_duration_fs=160.0, repetition_rate_hz=4e6)
    r_broad = UltrafastNonlinearSimulator.simulate_three_photon(laser_broad, sample, obj)
    assert pytest.approx(0.25, rel=1e-3) == (r_broad.generated_photons_per_second / r_base.generated_photons_per_second)

    # Doubled rep rate -> 1 / 2^2 = 0.25x signal
    laser_fast = LaserSourceParams(average_power_w=0.005, pulse_duration_fs=80.0, repetition_rate_hz=8e6)
    r_fast = UltrafastNonlinearSimulator.simulate_three_photon(laser_fast, sample, obj)
    assert pytest.approx(0.25, rel=1e-3) == (r_fast.generated_photons_per_second / r_base.generated_photons_per_second)


def test_three_photon_deep_tissue_attenuation():
    """Verify tissue scattering loss P(z) = P_0 * exp(-z / l_s) reduces 3P by exp(-3*z / l_s)."""
    laser = LaserSourceParams(average_power_w=0.010, pulse_duration_fs=80.0, repetition_rate_hz=4e6)
    sample = NonlinearSampleParams(sigma_3p_gm3=1.0e-82)
    obj = ObjectiveFocusParams(numerical_aperture=0.8)

    r_surface = UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj, tissue_depth_um=0.0)
    r_deep = UltrafastNonlinearSimulator.simulate_three_photon(laser, sample, obj, tissue_depth_um=300.0, scattering_length_um=300.0)

    # At z = l_s, power drops to 1/e, so 3P signal drops to (1/e)^3 = exp(-3)
    expected_ratio = math.exp(-3.0)
    actual_ratio = r_deep.generated_photons_per_second / r_surface.generated_photons_per_second
    assert pytest.approx(expected_ratio, rel=1e-3) == actual_ratio


# ==============================================================================
# 5. SECOND HARMONIC GENERATION (SHG) TESTS
# ==============================================================================

def test_shg_quadratic_power_and_sinc2_phase_matching():
    """Verify P_SHG proportional to P_pump^2 * sinc^2(Delta_k * L / 2)."""
    laser_base = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.010)
    sample_matched = NonlinearSampleParams(d_eff_pm_per_v=0.4, phase_mismatch_delta_k_m1=0.0, thickness_um=50.0)
    obj = ObjectiveFocusParams(numerical_aperture=0.8)

    r_base = UltrafastNonlinearSimulator.simulate_shg(laser_base, sample_matched, obj)
    assert r_base.emission_wavelength_nm == 400.0  # Frequency doubled

    # Doubled pump power -> 4x SHG
    laser_double = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.020)
    r_double = UltrafastNonlinearSimulator.simulate_shg(laser_double, sample_matched, obj)
    assert pytest.approx(4.0, rel=1e-3) == (r_double.generated_photons_per_second / r_base.generated_photons_per_second)

    # Phase mismatched (Delta_k = pi / L -> sinc(1/2)^2 = (2/pi)^2 ~ 0.405)
    l_m = 50.0 * 1e-6
    dk_test = math.pi / l_m
    sample_mismatched = NonlinearSampleParams(d_eff_pm_per_v=0.4, phase_mismatch_delta_k_m1=dk_test, thickness_um=50.0)
    r_mismatch = UltrafastNonlinearSimulator.simulate_shg(laser_base, sample_mismatched, obj)

    expected_sinc2 = (2.0 / math.pi) ** 2
    assert pytest.approx(expected_sinc2, rel=1e-3) == r_mismatch.details["phase_matching_sinc2"]
    assert pytest.approx(expected_sinc2, rel=1e-3) == (r_mismatch.generated_photons_per_second / r_base.generated_photons_per_second)


# ==============================================================================
# 6. COHERENT ANTI-STOKES RAMAN SCATTERING (CARS) TESTS
# ==============================================================================

def test_cars_asymmetric_fano_lineshape():
    """Verify CARS asymmetric Fano lineshape with peak on red side and dip below nonresonant background."""
    pump = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.020)
    stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.010)
    sample = NonlinearSampleParams(
        raman_shift_cm1=2845.0,
        raman_linewidth_cm1=15.0,
        chi3_resonant_m2_per_v2=2.0e-23,
        chi3_nonresonant_m2_per_v2=1.0e-23,  # Finite non-resonant electronic background
    )
    obj = ObjectiveFocusParams(numerical_aperture=0.8)

    res = UltrafastNonlinearSimulator.simulate_cars(pump, stokes, sample, obj, detuning_span_cm1=100.0, spectral_points=201)

    spectrum = np.array(res.details["fano_spectrum"])
    detunings = np.array(res.details["detunings_cm1"])
    nonres_base = res.details["nonresonant_baseline"]

    # Fano interference creates peak above baseline and dip below baseline
    max_val = np.max(spectrum)
    min_val = np.min(spectrum)
    assert max_val > nonres_base
    assert min_val < nonres_base

    # Resonant peak occurs on the red side (Delta_omega < Omega_R)
    omega_r = sample.raman_shift_cm1
    peak_detuning = res.details["peak_detuning_cm1"]
    dip_detuning = res.details["dip_detuning_cm1"]
    assert peak_detuning < omega_r
    assert dip_detuning > omega_r
    assert res.details["fano_asymmetry_factor"] > 1.0


# ==============================================================================
# 7. STIMULATED RAMAN SCATTERING (SRS) TESTS
# ==============================================================================

def test_srs_linear_concentration_and_zero_background():
    """Verify SRS is strictly linear in concentration and has zero non-resonant background."""
    pump = LaserSourceParams(center_wavelength_nm=800.0, average_power_w=0.020)
    stokes = LaserSourceParams(center_wavelength_nm=1040.0, average_power_w=0.010)
    sample = NonlinearSampleParams(raman_gain_m_per_w=2e-13, thickness_um=50.0)
    obj = ObjectiveFocusParams(numerical_aperture=0.8)

    res1 = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj, concentration_scale=1.0)
    res2 = UltrafastNonlinearSimulator.simulate_srs(pump, stokes, sample, obj, concentration_scale=2.5)

    # Strictly zero non-resonant background
    assert res1.details["nonresonant_background"] == 0.0

    # Linear concentration scaling: 2.5x concentration -> 2.5x SRG and signal
    ratio = res2.details["stimulated_raman_gain_srg"] / res1.details["stimulated_raman_gain_srg"]
    assert pytest.approx(2.5, rel=1e-4) == ratio

    # Energy balance between Stokes Gain (SRG) and Pump Loss (SRL)
    # Delta_I_p / Delta_I_S = lambda_S / lambda_p = 1040 / 800 = 1.30
    srg = res1.details["stimulated_raman_gain_srg"]
    srl = res1.details["stimulated_raman_loss_srl"]
    assert pytest.approx(1040.0 / 800.0, rel=1e-4) == (srl / srg)


# ==============================================================================
# 8. DISPERSION MANAGEMENT & PRE-CHIRP COMPENSATION TESTS
# ==============================================================================

def test_gdd_pulse_broadening_and_prechirp_compensation():
    """Verify pulse broadening equation and exact pre-chirp cancellation."""
    tau_0 = 30.0  # 30 fs transform limit
    gdd_glass = 1000.0  # +1000 fs²
    # Closed-form: tau(GDD) = tau_0 * sqrt(1 + (4 ln 2 * GDD / tau_0^2)^2)
    expected_broadened = tau_0 * math.sqrt(1.0 + (4.0 * math.log(2.0) * gdd_glass / (tau_0 ** 2)) ** 2)

    laser = LaserSourceParams(pulse_duration_fs=tau_0)
    sample = NonlinearSampleParams()
    obj = ObjectiveFocusParams()

    # Broadened case
    res_broad = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, external_gdd_fs2=gdd_glass)
    assert pytest.approx(expected_broadened, rel=1e-3) == res_broad.details["focal_pulse_width_fs"]

    # Pre-chirp compensated case: apply -1000 fs²
    res_comp = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, external_gdd_fs2=gdd_glass - 1000.0)
    assert pytest.approx(tau_0, rel=1e-3) == res_comp.details["focal_pulse_width_fs"]


# ==============================================================================
# 9. SPECTRAL FOCUSING SRS TESTS
# ==============================================================================

def test_spectral_focusing_srs_chirp_matching_and_pulse_stretching():
    """Verify broadband pulse stretching through SF57 glass rods to picosecond duration."""
    pump = LaserSourceParams(center_wavelength_nm=800.0, pulse_duration_fs=100.0)
    stokes = LaserSourceParams(center_wavelength_nm=1040.0, pulse_duration_fs=100.0)

    # 150 mm SF57 for pump, 280 mm SF57 for Stokes
    cfg = SpectralFocusingConfig(
        pump_glass_material=DispersiveGlassMaterial.SF57,
        pump_glass_length_mm=150.0,
        stokes_glass_material=DispersiveGlassMaterial.SF57,
        stokes_glass_length_mm=280.0,
    )
    sample = NonlinearSampleParams()

    res = SpectralFocusingSRS.simulate(pump, stokes, cfg, sample)
    # Both pulses stretched from 100 fs to near-picosecond duration (> 0.8 ps)
    assert res.chirped_pump_fwhm_ps > 0.8
    assert res.chirped_stokes_fwhm_ps > 0.8
    # GDD values positive and in tens of thousands of fs²
    assert res.pump_gdd_fs2 > 20000.0
    assert res.stokes_gdd_fs2 > 20000.0
    # Chirp matching ratio close to 1.0 (within 30%)
    assert 0.7 <= res.chirp_matching_ratio <= 1.4
    # High spectral resolution (5 to 35 cm^-1)
    assert 5.0 <= res.spectral_resolution_cm1 <= 35.0
    # Reconstructed spectrum has distinct non-zero peaks
    assert res.peak_srs_gain > 0.0


def test_spectral_focusing_srs_delay_to_frequency_scan():
    """Verify linear frequency difference tuning Delta_omega(tau_d) with delay scan."""
    pump = LaserSourceParams(center_wavelength_nm=800.0, pulse_duration_fs=100.0)
    stokes = LaserSourceParams(center_wavelength_nm=1040.0, pulse_duration_fs=100.0)

    cfg = SpectralFocusingConfig(
        delay_scan_range_ps=2.0,
        delay_samples=51,
        sample_raman_peaks=[(2845.0, 15.0, 1.0)],
    )
    sample = NonlinearSampleParams()

    res = SpectralFocusingSRS.simulate(pump, stokes, cfg, sample)
    delays = res.delays_ps
    freqs = res.instantaneous_frequency_difference_cm1

    # Frequency difference should vary monotonically across delay
    diffs = np.diff(freqs)
    assert np.all(diffs > 0) or np.all(diffs < 0)

    # Maximum gain occurs near the configured Raman peak (2845 cm^-1)
    peak_idx = int(np.argmax(res.reconstructed_raman_spectrum))
    peak_freq = freqs[peak_idx]
    assert pytest.approx(2845.0, abs=60.0) == peak_freq


# ==============================================================================
# 10. TEMPORAL SIGNAL PROBE TESTS
# ==============================================================================

def test_temporal_signal_probe_ultrafast_overlap():
    """Verify sub-picosecond cross-correlation envelope and nonlinear generation."""
    res = NonlinearSignalTemporalProbe.probe_ultrafast_overlap(
        pump_tau_fs=100.0,
        stokes_tau_fs=100.0,
        delay_fs=0.0,
        window_fs=400.0,
    )
    t = res["time_fs"]
    zero_idx = np.argmin(np.abs(t))

    # At zero delay, envelopes and overlaps reach unity peak
    assert pytest.approx(1.0, rel=1e-3) == res["pump_envelope"][zero_idx]
    assert pytest.approx(1.0, rel=1e-3) == res["stokes_envelope"][zero_idx]
    assert pytest.approx(1.0, rel=1e-3) == res["2p_overlap"][zero_idx]
    assert pytest.approx(1.0, rel=1e-3) == res["3p_overlap"][zero_idx]
    assert pytest.approx(1.0, rel=1e-3) == res["srs_overlap"][zero_idx]

    # At large delay (e.g. 200 fs), overlap drops significantly
    far_idx = np.argmin(np.abs(t - 200.0))
    assert res["2p_overlap"][far_idx] < 0.1
    assert res["3p_overlap"][far_idx] < 0.05


def test_temporal_signal_probe_modulated_pulse_train():
    """Verify periodic pulse train with RF modulation envelope for lock-in detection."""
    res = NonlinearSignalTemporalProbe.probe_modulated_pulse_train(
        rep_rate_mhz=80.0,
        modulation_freq_mhz=10.0,
        duration_ns=100.0,
    )
    t = res["time_ns"]
    envelope = res["modulation_envelope"]
    signal = res["modulated_signal"]

    # 10 MHz modulation over 100 ns corresponds to exactly 1 complete RF cycle
    assert len(t) > 500
    assert np.min(envelope) < 0.5
    assert np.max(envelope) > 0.9
    # Periodic pulses detected
    assert np.max(signal) > 0.0


# ==============================================================================
# 11. SPATIAL DEFECTS & BEAM DISTORTIONS TESTS
# ==============================================================================

def test_spatial_beam_defects_strehl_and_pulse_front_tilt():
    """Verify Marechal Strehl ratio, pulse front tilt, and intensity degradation."""
    defects = SpatialBeamDefects(
        spatial_chirp_mm_per_nm=0.02,
        angular_dispersion_mrad_per_nm=0.5,
        wavefront_error_rms_waves=0.05,  # lambda / 20 wavefront aberration
        beam_ellipticity=1.2,
    )
    # S = exp(-(2 * pi * 0.05)^2) ~ 0.906
    expected_strehl = math.exp(-((2.0 * math.pi * 0.05) ** 2))
    assert pytest.approx(expected_strehl, rel=1e-2) == defects.strehl_ratio

    # Pulse front tilt gamma = lambda * d_theta / d_lambda
    # 800 nm * 0.5 mrad/nm = 800e-9 m * 5.0e5 rad/m = 0.4 rad
    calc_pft = defects.pulse_front_tilt_rad(wavelength_nm=800.0)
    assert pytest.approx(0.4, rel=1e-3) == calc_pft

    # Pulse duration broadening due to PFT
    tau_eff = defects.effective_pulse_duration_fs(transform_limit_fs=100.0, beam_waist_um=1.0, wavelength_nm=800.0)
    assert tau_eff >= 100.0

    # Spot area enlargement due to spatial chirp and ellipticity
    area_eff = defects.effective_spot_area_um2(beam_waist_um=1.0, bandwidth_nm=10.0)
    ideal_area = math.pi * (1.0 ** 2)
    assert area_eff > ideal_area

    # Combined degradation factor in [0, 1]
    deg = defects.intensity_degradation_factor(beam_waist_um=1.0, transform_limit_fs=100.0, wavelength_nm=800.0)
    assert 0.0 < deg < 1.0


def test_spatial_defects_impact_on_multiphoton_imaging():
    """Verify that spatial defects reduce 2P and 3P peak intensities and detected counts."""
    laser = LaserSourceParams(center_wavelength_nm=800.0, pulse_duration_fs=100.0)
    sample = NonlinearSampleParams()
    obj = ObjectiveFocusParams()

    res_ideal = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj)

    defects = SpatialBeamDefects(
        wavefront_error_rms_waves=0.10,  # significant aberration
        spatial_chirp_mm_per_nm=0.01,
    )
    res_aberrated = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, spatial_defects=defects)

    assert res_aberrated.focal_peak_intensity_w_cm2 < res_ideal.focal_peak_intensity_w_cm2
    assert res_aberrated.detected_photons_per_dwell < res_ideal.detected_photons_per_dwell
    assert res_aberrated.details["strehl_ratio"] < 1.0


# ==============================================================================
# 12. INTERACTION PLANE & DEFOCUS TESTS
# ==============================================================================

def test_interaction_plane_defocus_spot_expansion_and_signal_drop():
    """Verify beam expansion w(z) = w0 * sqrt(1 + (z / zR)^2) and signal drop with defocus."""
    laser = LaserSourceParams(center_wavelength_nm=800.0)
    sample = NonlinearSampleParams()
    obj = ObjectiveFocusParams(numerical_aperture=0.8)

    w0 = obj.beam_waist_radius_m(800.0)
    zr = obj.rayleigh_range_m(800.0, sample.refractive_index)

    # Defocus by 1 Rayleigh range
    plane_foc = InteractionPlaneConfig(axial_defocus_um=0.0)
    plane_defoc = InteractionPlaneConfig(axial_defocus_um=zr * 1e6)

    # Spot size at 1 z_R should be sqrt(2) * w0
    r_defoc = plane_defoc.compute_spot_radius_um(w0 * 1e6, zr * 1e6)
    assert pytest.approx(w0 * 1e6 * math.sqrt(2.0), rel=1e-3) == r_defoc

    res_foc = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, interaction_plane=plane_foc)
    res_defoc = UltrafastNonlinearSimulator.simulate_two_photon(laser, sample, obj, interaction_plane=plane_defoc)

    # Intensity at z = z_R drops by factor of 2 (w^2)
    assert pytest.approx(0.5, rel=0.05) == (res_defoc.focal_peak_intensity_w_cm2 / res_foc.focal_peak_intensity_w_cm2)
    # 2P total count rate drops with defocus
    assert res_defoc.detected_photons_per_dwell < res_foc.detected_photons_per_dwell


# ==============================================================================
# 13. THORLABS CATALOG & DELAY CONVERSION TESTS
# ==============================================================================

def test_thorlabs_delay_stage_conversion():
    """Verify stage displacement (mm) <-> pulse delay (ps) for double-pass ODL."""
    # 1 mm displacement in double-pass (passes = 2) = 2 * 1e-3 / c ~ 6.671 ps
    delay_ps = delay_from_stage_displacement_mm(1.0, passes=2)
    assert pytest.approx(6.671, rel=1e-3) == delay_ps

    # Invert
    disp_mm = stage_displacement_from_delay_ps(delay_ps, passes=2)
    assert pytest.approx(1.0, rel=1e-3) == disp_mm

    # Check catalog entries
    assert "ODL220" in THORLABS_ULTRAFAST_CATALOG
    assert "ODL100" in THORLABS_ULTRAFAST_CATALOG
    assert "WG11100-SF57" in THORLABS_ULTRAFAST_CATALOG
    assert THORLABS_ULTRAFAST_CATALOG["ODL220"]["travel_range_mm"] == 220.0


# ==============================================================================
# 14. GUI INTEGRATION TESTS (ALL 5 TABS & CONTROLS)
# ==============================================================================

def test_ultrafast_analysis_dock_initialization_and_tabs():
    """Verify UltrafastAnalysisDock can be instantiated with all 6 tabs and widgets."""
    dock = UltrafastAnalysisDock()
    assert dock.objectName() == "UltrafastAnalysisDock"
    assert dock.tabs.count() == 6

    # Switch modalities
    for idx in range(5):
        dock.combo_modality.setCurrentIndex(idx)
        assert dock.combo_modality.currentIndex() == idx

    # Trigger modulator calculations
    aom, eom = dock.update_modulator_metrics()
    assert aom is not None
    assert eom is not None

    # Run nonlinear simulation with interaction plane & defects
    dock.combo_modality.setCurrentIndex(0)  # 2P
    dock.spin_defocus_um.setValue(5.0)
    dock.cb_enable_defects.setChecked(True)
    dock.spin_wavefront_rms.setValue(0.05)
    dock.run_nonlinear_simulation()
    assert dock._last_nonlinear_result is not None
    assert dock._last_nonlinear_result.modality == "2P"
    assert dock.tbl_nonlinear_results.rowCount() > 8

    # Run Spectral Focusing SRS through GUI
    dock.run_spectral_focusing_simulation()
    assert dock._last_spectral_focusing_result is not None
    assert dock.tbl_sf_spectrum.rowCount() > 10

    # Run Temporal Probes through GUI
    dock.run_overlap_probe()
    assert dock._last_probe_result is not None

    dock.run_pulse_train_probe()
    assert dock._last_probe_result is not None

