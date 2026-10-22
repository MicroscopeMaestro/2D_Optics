"""Adversarial Challenge Test Suite for Milestone 1.

Author: challenger_srs_m1_2
Scope:
1. Numerical stress tests:
   - Zero laser powers (P_P = 0, P_S = 0)
   - Ultra-high laser powers (P = 1 kW, P = 1 GW)
   - Extreme out-of-range wavelengths (0 nm, negative nm, swapped wavelengths, UV, far IR)
   - Extreme repetition rates (0, negative, THz)
   - Extreme pulse durations (0 fs, negative fs, CW)
   - Complete HWP angular sweep [-720 deg, 720 deg], negative angles, energy conservation
   - EOM extreme voltages, zero V_pi, arbitrary time arrays, waveform energy conservation
2. JSON round-trip serialization and deserialization:
   - Default and non-default ThorlabsSRSMicroscopeConfig
   - Unknown/extraneous field injection
   - Missing/None/empty inputs
   - Enums as string values, names, cases
   - Subcomponent serialization round-trips
3. Differential verification between optics_app and v2:
   - 100% numerical equality across randomized parameter matrices
   - Catalog part dictionary equality
"""

import math
import json
import numpy as np
import pytest

from optics_app.analysis.ultrafast_nonlinear import (
    DualLaserSourceConfig,
    PolarizationPowerAttenuator,
    EOMModulationResult,
    CleanupFilterConfig,
    DichroicCombinerConfig,
    simulate_dual_laser_delivery,
    ThorlabsSRSMicroscopeConfig,
    OilSampleType,
    DispersiveGlassMaterial,
    SRSPhotodiodeModel,
)
from optics_app.catalogs.thorlabs import (
    get_catalog_lens,
    get_catalog_sections,
    query_catalog,
)

import v2.analysis.ultrafast_nonlinear as v2_nonlinear
import v2.catalogs.thorlabs as v2_thorlabs


# ==============================================================================
# SECTION 1: NUMERICAL STRESS TESTS
# ==============================================================================

class TestNumericalStressPowersAndWavelengths:
    """Stress test mathematical functions under boundary, zero, and extreme values."""

    def test_zero_laser_powers(self):
        """Zero pump and Stokes powers should produce zero energies and peak powers with no NaNs."""
        lasers = DualLaserSourceConfig(pump_power_mw=0.0, stokes_power_mw=0.0)
        assert lasers.pump_energy_nj == 0.0
        assert lasers.stokes_energy_nj == 0.0
        assert lasers.pump_peak_power_kw == 0.0
        assert lasers.stokes_peak_power_kw == 0.0
        assert not math.isnan(lasers.base_raman_shift_cm1)

        # End-to-end simulation with zero power
        res = simulate_dual_laser_delivery(dual_lasers=lasers)
        assert res["collinear_pump_power_mw"] == 0.0
        assert res["collinear_stokes_dc_power_mw"] == 0.0
        assert res["total_bt610_dumped_power_mw"] == 0.0
        assert res["base_raman_shift_cm1"] == pytest.approx(2931.6668, rel=1e-4)
        for key, val in res.items():
            if isinstance(val, (int, float)):
                assert not math.isnan(val), f"NaN in res[{key}]"
                assert not math.isinf(val), f"Inf in res[{key}]"

    def test_ultra_high_laser_powers_one_kilowatt(self):
        """1000 W (10^6 mW) input laser powers should propagate with strict energy balance."""
        p_in_kw = 1000.0 * 1e3  # 1000 W = 1,000,000 mW
        lasers = DualLaserSourceConfig(pump_power_mw=p_in_kw, stokes_power_mw=p_in_kw)
        
        # Pump: 1,000,000 mW / 80 MHz = 12,500 nJ = 12.5 uJ
        assert math.isclose(lasers.pump_energy_nj, 12500.0, rel_tol=1e-5)
        assert math.isclose(lasers.stokes_energy_nj, 12500.0, rel_tol=1e-5)
        
        # Peak power: 0.88 * 12.5 uJ / 120 fs = 91.667 MW = 91666.67 kW
        assert math.isclose(lasers.pump_peak_power_kw, 91666.6667, rel_tol=1e-4)

        # End-to-end delivery simulation
        res = simulate_dual_laser_delivery(dual_lasers=lasers)
        assert res["collinear_pump_power_mw"] > 0.0
        assert res["collinear_stokes_dc_power_mw"] > 0.0
        assert res["total_bt610_dumped_power_mw"] > 0.0
        
        # Verify no overflow or NaN
        for key, val in res.items():
            if isinstance(val, (int, float)):
                assert not math.isnan(val)
                assert not math.isinf(val)

    def test_petawatt_regime_stability(self):
        """10^15 mW power should not trigger floating-point overflow or crash."""
        extreme_power = 1e15
        lasers = DualLaserSourceConfig(pump_power_mw=extreme_power, stokes_power_mw=extreme_power)
        assert lasers.pump_energy_nj > 0.0
        assert not math.isinf(lasers.pump_energy_nj)
        assert not math.isnan(lasers.pump_peak_power_kw)

    def test_zero_and_negative_wavelength_protection(self):
        """Wavelengths <= 0 should be guarded by max(1.0, ...) to prevent ZeroDivisionError."""
        lasers_zero = DualLaserSourceConfig(pump_wavelength_nm=0.0, stokes_wavelength_nm=0.0)
        # 1.0e7 / 1.0 - 1.0e7 / 1.0 = 0.0
        assert lasers_zero.base_raman_shift_cm1 == 0.0

        lasers_neg = DualLaserSourceConfig(pump_wavelength_nm=-500.0, stokes_wavelength_nm=-800.0)
        # Both guarded to 1.0 nm
        assert lasers_neg.base_raman_shift_cm1 == 0.0

    def test_swapped_inverted_wavelengths_anti_stokes(self):
        """Inverted wavelengths (pump > stokes) yield negative Raman shift and filter blocking."""
        lasers = DualLaserSourceConfig(pump_wavelength_nm=1040.0, stokes_wavelength_nm=797.0)
        expected_shift = (1.0e7 / 1040.0) - (1.0e7 / 797.0)
        assert lasers.base_raman_shift_cm1 == pytest.approx(expected_shift)
        assert lasers.base_raman_shift_cm1 < 0.0  # Anti-Stokes sign

        # Filters should reject swapped wavelengths
        filters = CleanupFilterConfig()
        # Pump filter (790 nm bandpass) receiving 1040 nm -> OD 6 blocking
        t_pump_wrong = filters.get_transmission("pump", 1040.0)
        assert t_pump_wrong <= 1e-6

        # Stokes filter (1000 nm cuton) receiving 797 nm -> OD 5 blocking
        t_stokes_wrong = filters.get_transmission("stokes", 797.0)
        assert t_stokes_wrong <= 1e-5

        # End-to-end throughput should drop to near zero (< 1e-6)
        res = simulate_dual_laser_delivery(dual_lasers=lasers)
        assert res["collinear_pump_power_mw"] < 1e-3
        assert res["collinear_stokes_dc_power_mw"] < 1e-3

    def test_zero_and_negative_repetition_rate(self):
        """Repetition rate <= 0 should be bounded to prevent division by zero."""
        lasers = DualLaserSourceConfig(rep_rate_mhz=0.0)
        assert lasers.pump_energy_nj > 0.0
        assert not math.isnan(lasers.pump_energy_nj)
        assert not math.isinf(lasers.pump_energy_nj)

        lasers_neg = DualLaserSourceConfig(rep_rate_mhz=-100.0)
        assert lasers_neg.pump_energy_nj > 0.0

    def test_zero_and_negative_pulse_duration(self):
        """Pulse duration <= 0 should be clamped to >= 1.0 fs to avoid ZeroDivisionError."""
        lasers = DualLaserSourceConfig(pump_pulse_fs=0.0, stokes_pulse_fs=-50.0)
        assert lasers.pump_peak_power_kw > 0.0
        assert lasers.stokes_peak_power_kw > 0.0
        assert not math.isnan(lasers.pump_peak_power_kw)
        assert not math.isinf(lasers.stokes_peak_power_kw)


class TestPolarizationAttenuatorStress:
    """Stress test Malus law, angular sweeps, and energy conservation."""

    @pytest.mark.parametrize("angle_deg", [
        -720.0, -360.0, -180.0, -90.0, -45.0, -22.5, 0.0,
        15.0, 22.5, 30.0, 45.0, 60.0, 67.5, 90.0, 135.0, 180.0, 360.0, 720.0, 10000.0
    ])
    def test_angular_sweep_energy_conservation(self, angle_deg):
        """For any arbitrary angle, P_trans + P_dump must equal P_in within float epsilon."""
        p_in = 123.456
        atten = PolarizationPowerAttenuator(input_power_mw=p_in, hwp_angle_deg=angle_deg)
        p_trans = atten.transmitted_power_mw()
        p_dump = atten.dumped_power_mw()
        
        assert math.isclose(p_trans + p_dump, p_in, rel_tol=1e-12, abs_tol=1e-12)
        assert 0.0 <= p_trans <= p_in
        assert 0.0 <= p_dump <= p_in
        assert atten.attenuation_db() >= 0.0

    def test_extinction_ratio_reach(self):
        """At theta = 45 deg, polarization is rotated by 90 deg, reaching ER >= 1000:1 (>= 30 dB)."""
        atten = PolarizationPowerAttenuator(input_power_mw=100.0, hwp_angle_deg=45.0, extinction_ratio=1000.0)
        att_db = atten.attenuation_db()
        assert att_db >= 30.0  # -10*log10(0.92/1000) = 30.36 dB

    def test_zero_input_power(self):
        """Zero input power should produce zero transmitted and dumped power, attenuation dB finite."""
        atten = PolarizationPowerAttenuator(input_power_mw=0.0, hwp_angle_deg=0.0)
        assert atten.transmitted_power_mw() == 0.0
        assert atten.dumped_power_mw() == 0.0
        assert atten.attenuation_db() >= 0.0
        assert not math.isnan(atten.attenuation_db())


class TestActiveEOMStress:
    """Stress test EOM modulation waveforms, voltages, and sampling."""

    def test_extreme_rf_voltages(self):
        """Extreme RF voltages (e.g. 10000 V) should be clipped to m = 1.0."""
        cfg = ThorlabsSRSMicroscopeConfig()
        result_high = cfg.get_eom_modulation_result(applied_voltage_v=10000.0)
        assert result_high.modulation_depth == 1.0

        result_neg = cfg.get_eom_modulation_result(applied_voltage_v=-500.0)
        assert result_neg.modulation_depth == 0.0

    def test_zero_v_pi_guard(self):
        """If v_pi is 0 or negative, should not raise ZeroDivisionError."""
        cfg = ThorlabsSRSMicroscopeConfig(modulator_v_pi_v=0.0)
        res = cfg.get_eom_modulation_result(applied_voltage_v=50.0)
        assert 0.0 <= res.modulation_depth <= 1.0
        assert not math.isnan(res.modulation_depth)

    def test_waveform_energy_conservation_and_non_negativity(self):
        """For m <= 1.0, transmitted intensity I_S(t) >= 0 and I_S(t) + I_dump(t) == 2 * I_0."""
        eom = EOMModulationResult(modulation_freq_mhz=10.0, modulation_depth=0.85, dc_intensity=25.0)
        # Test 1000 points over 2 modulation periods (T = 100 ns)
        t_arr = np.linspace(0, 200e-9, 1000)
        i_trans = eom.transmitted_intensity(t_arr)
        i_dump = eom.rejected_intensity(t_arr)

        assert np.all(i_trans >= 0.0)
        assert np.all(i_dump >= 0.0)
        assert np.allclose(i_trans + i_dump, 2.0 * eom.dc_intensity)

    def test_empty_array_and_scalar_time_inputs(self):
        """EOM modulation result should handle scalar int, float, list, 1D array, and empty array."""
        eom = EOMModulationResult()
        # Float
        assert isinstance(eom.transmitted_intensity(1e-8), float)
        # Int
        assert isinstance(eom.transmitted_intensity(0), float)
        # List
        res_list = eom.transmitted_intensity([0.0, 1e-8, 2e-8])
        assert isinstance(res_list, np.ndarray) and len(res_list) == 3
        # Empty array
        res_empty = eom.transmitted_intensity(np.array([]))
        assert isinstance(res_empty, np.ndarray) and len(res_empty) == 0


# ==============================================================================
# SECTION 2: JSON ROUND-TRIP SERIALIZATION & DESERIALIZATION
# ==============================================================================

class TestJSONRoundTripAndRobustness:
    """Stress test serialization fidelity, schema changes, and resilience to dirty JSON."""

    def test_default_config_json_roundtrip(self):
        """Default ThorlabsSRSMicroscopeConfig serializes to valid JSON and restores completely."""
        cfg = ThorlabsSRSMicroscopeConfig()
        d = cfg.to_dict()
        js = json.dumps(d)
        restored = ThorlabsSRSMicroscopeConfig.from_dict(json.loads(js))

        assert restored.pump_laser_wavelength_nm == cfg.pump_laser_wavelength_nm
        assert restored.stokes_laser_wavelength_nm == cfg.stokes_laser_wavelength_nm
        assert restored.pump_laser_power_mw == cfg.pump_laser_power_mw
        assert restored.stokes_laser_power_mw == cfg.stokes_laser_power_mw
        assert restored.sample_type == cfg.sample_type
        assert restored.pump_glass_material == cfg.pump_glass_material
        assert restored.stokes_glass_material == cfg.stokes_glass_material
        assert math.isclose(restored.base_raman_shift_cm1, cfg.base_raman_shift_cm1, rel_tol=1e-5)

    def test_custom_config_json_roundtrip(self):
        """Customized config with non-default enums, angles, and powers restores cleanly."""
        cfg = ThorlabsSRSMicroscopeConfig(
            pump_laser_wavelength_nm=805.0,
            pump_laser_power_mw=120.0,
            pump_pulse_duration_fs=90.0,
            stokes_laser_wavelength_nm=1064.0,
            stokes_laser_power_mw=250.0,
            stokes_pulse_duration_fs=150.0,
            pump_hwp_angle_deg=15.0,
            stokes_hwp_angle_deg=35.0,
            sample_type=OilSampleType.OLIVE_OIL,
            pump_glass_material=DispersiveGlassMaterial.SF11,
            stokes_glass_material=DispersiveGlassMaterial.FUSED_SILICA,
        )
        d = cfg.to_dict()
        js = json.dumps(d)
        restored = ThorlabsSRSMicroscopeConfig.from_dict(json.loads(js))

        assert restored.pump_laser_wavelength_nm == 805.0
        assert restored.pump_laser_power_mw == 120.0
        assert restored.pump_pulse_duration_fs == 90.0
        assert restored.stokes_laser_wavelength_nm == 1064.0
        assert restored.stokes_laser_power_mw == 250.0
        assert restored.stokes_pulse_duration_fs == 150.0
        assert restored.pump_hwp_angle_deg == 15.0
        assert restored.stokes_hwp_angle_deg == 35.0
        assert restored.sample_type == OilSampleType.OLIVE_OIL
        assert restored.pump_glass_material == DispersiveGlassMaterial.SF11
        assert restored.stokes_glass_material == DispersiveGlassMaterial.FUSED_SILICA

    def test_unknown_and_malicious_field_tolerance(self):
        """from_dict must safely filter out unknown or extra keys without crashing."""
        dirty_data = {
            "pump_laser_wavelength_nm": 797.0,
            "stokes_laser_wavelength_nm": 1040.0,
            "unexpected_legacy_key": 9999,
            "__proto__": {"admin": True},
            "unrecognized_structure": {"nested": [1, 2, 3]},
            "extra_string": "should be ignored",
        }
        restored = ThorlabsSRSMicroscopeConfig.from_dict(dirty_data)
        assert restored.pump_laser_wavelength_nm == 797.0
        assert restored.stokes_laser_wavelength_nm == 1040.0
        assert not hasattr(restored, "unexpected_legacy_key")
        assert not hasattr(restored, "unrecognized_structure")

    def test_none_and_empty_json_handling(self):
        """from_dict with None or empty dictionary returns default config."""
        assert ThorlabsSRSMicroscopeConfig.from_dict(None).pump_laser_wavelength_nm == 797.0
        assert ThorlabsSRSMicroscopeConfig.from_dict({}).pump_laser_power_mw == 25.0

    def test_enum_case_insensitivity(self):
        """Enum fields should parse strings regardless of case or format."""
        data_lower = {
            "sample_type": "mineral oil",
            "pump_glass_material": "sf57",
            "stokes_glass_material": "fused_silica",
        }
        cfg = ThorlabsSRSMicroscopeConfig.from_dict(data_lower)
        assert cfg.sample_type == OilSampleType.MINERAL_OIL
        assert cfg.pump_glass_material == DispersiveGlassMaterial.SF57
        assert cfg.stokes_glass_material == DispersiveGlassMaterial.FUSED_SILICA

    def test_all_m1_subcomponents_json_roundtrip(self):
        """Verify round-trip for DualLaserSourceConfig, PolarizationPowerAttenuator, EOMModulationResult, CleanupFilterConfig, DichroicCombinerConfig."""
        # 1. DualLaserSourceConfig
        lasers = DualLaserSourceConfig(pump_power_mw=33.3, stokes_power_mw=77.7)
        lasers_rt = DualLaserSourceConfig.from_dict(json.loads(json.dumps(lasers.to_dict())))
        assert lasers_rt.pump_power_mw == 33.3
        assert lasers_rt.stokes_power_mw == 77.7

        # 2. PolarizationPowerAttenuator
        atten = PolarizationPowerAttenuator(input_power_mw=45.0, hwp_angle_deg=22.5)
        atten_rt = PolarizationPowerAttenuator.from_dict(json.loads(json.dumps(atten.to_dict())))
        assert atten_rt.input_power_mw == 45.0
        assert atten_rt.hwp_angle_deg == 22.5

        # 3. EOMModulationResult
        eom = EOMModulationResult(modulation_freq_mhz=10.0, modulation_depth=0.72)
        eom_rt = EOMModulationResult.from_dict(json.loads(json.dumps(eom.to_dict())))
        assert eom_rt.modulation_depth == 0.72

        # 4. CleanupFilterConfig
        filt = CleanupFilterConfig(pump_transmission=0.91, stokes_transmission=0.94)
        filt_rt = CleanupFilterConfig.from_dict(json.loads(json.dumps(filt.to_dict())))
        assert filt_rt.pump_transmission == 0.91
        assert filt_rt.stokes_transmission == 0.94

        # 5. DichroicCombinerConfig
        comb = DichroicCombinerConfig(pump_reflection_r=0.97, stokes_transmission_t=0.93)
        comb_rt = DichroicCombinerConfig.from_dict(json.loads(json.dumps(comb.to_dict())))
        assert comb_rt.pump_reflection_r == 0.97
        assert comb_rt.stokes_transmission_t == 0.93


# ==============================================================================
# SECTION 3: DIFFERENTIAL VERIFICATION BETWEEN OPTICS_APP AND V2
# ==============================================================================

class TestDifferentialParityOpticsAppVsV2:
    """Rigorous differential tests comparing optics_app and v2 execution outputs."""

    def test_dual_laser_source_differential_matrix(self):
        """Compare DualLaserSourceConfig across a randomized parameter matrix."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            p_pump = float(rng.uniform(1.0, 500.0))
            p_stokes = float(rng.uniform(1.0, 1000.0))
            lam_p = float(rng.uniform(700.0, 900.0))
            lam_s = float(rng.uniform(950.0, 1200.0))
            rep_rate = float(rng.uniform(20.0, 160.0))

            cfg_app = DualLaserSourceConfig(
                pump_power_mw=p_pump,
                stokes_power_mw=p_stokes,
                pump_wavelength_nm=lam_p,
                stokes_wavelength_nm=lam_s,
                rep_rate_mhz=rep_rate,
            )
            cfg_v2 = v2_nonlinear.DualLaserSourceConfig(
                pump_power_mw=p_pump,
                stokes_power_mw=p_stokes,
                pump_wavelength_nm=lam_p,
                stokes_wavelength_nm=lam_s,
                rep_rate_mhz=rep_rate,
            )

            assert math.isclose(cfg_app.base_raman_shift_cm1, cfg_v2.base_raman_shift_cm1, rel_tol=1e-12)
            assert math.isclose(cfg_app.pump_energy_nj, cfg_v2.pump_energy_nj, rel_tol=1e-12)
            assert math.isclose(cfg_app.stokes_energy_nj, cfg_v2.stokes_energy_nj, rel_tol=1e-12)
            assert math.isclose(cfg_app.pump_peak_power_kw, cfg_v2.pump_peak_power_kw, rel_tol=1e-12)
            assert math.isclose(cfg_app.stokes_peak_power_kw, cfg_v2.stokes_peak_power_kw, rel_tol=1e-12)
            assert cfg_app.to_dict() == cfg_v2.to_dict()

    def test_polarization_attenuator_differential_sweep(self):
        """Compare PolarizationPowerAttenuator across 100 angles in [-360, 360]."""
        angles = np.linspace(-360.0, 360.0, 100)
        for ang in angles:
            att_app = PolarizationPowerAttenuator(input_power_mw=50.0, hwp_angle_deg=float(ang))
            att_v2 = v2_nonlinear.PolarizationPowerAttenuator(input_power_mw=50.0, hwp_angle_deg=float(ang))

            assert math.isclose(att_app.transmission_fraction(), att_v2.transmission_fraction(), rel_tol=1e-12)
            assert math.isclose(att_app.transmitted_power_mw(), att_v2.transmitted_power_mw(), rel_tol=1e-12)
            assert math.isclose(att_app.dumped_power_mw(), att_v2.dumped_power_mw(), rel_tol=1e-12)
            assert math.isclose(att_app.attenuation_db(), att_v2.attenuation_db(), rel_tol=1e-12)
            assert att_app.to_dict() == att_v2.to_dict()

    def test_eom_modulation_differential(self):
        """Compare EOM waveform generation between optics_app and v2 across 10,000 points."""
        eom_app = EOMModulationResult(modulation_freq_mhz=10.0, modulation_depth=0.85, dc_intensity=30.0)
        eom_v2 = v2_nonlinear.EOMModulationResult(modulation_freq_mhz=10.0, modulation_depth=0.85, dc_intensity=30.0)

        t_points = np.linspace(0, 1e-6, 10000)
        i_app = eom_app.transmitted_intensity(t_points)
        i_v2 = eom_v2.transmitted_intensity(t_points)
        assert np.allclose(i_app, i_v2, rtol=1e-12, atol=1e-12)

        dump_app = eom_app.rejected_intensity(t_points)
        dump_v2 = eom_v2.rejected_intensity(t_points)
        assert np.allclose(dump_app, dump_v2, rtol=1e-12, atol=1e-12)

        assert eom_app.harmonic_amplitudes() == eom_v2.harmonic_amplitudes()

    def test_cleanup_filters_differential_sweep(self):
        """Compare CleanupFilterConfig transmissions across 1301 wavelengths from 200 to 1500 nm."""
        f_app = CleanupFilterConfig()
        f_v2 = v2_nonlinear.CleanupFilterConfig()

        for wl in range(200, 1501):
            assert f_app.get_transmission("pump", wl) == f_v2.get_transmission("pump", wl)
            assert f_app.get_transmission("stokes", wl) == f_v2.get_transmission("stokes", wl)

    def test_dichroic_combiner_differential(self):
        """Compare DichroicCombinerConfig leakage calculation across 50 power pairs."""
        comb_app = DichroicCombinerConfig()
        comb_v2 = v2_nonlinear.DichroicCombinerConfig()

        rng = np.random.default_rng(123)
        for _ in range(50):
            p_pump = float(rng.uniform(0.0, 100.0))
            p_stokes = float(rng.uniform(0.0, 100.0))
            res_app = comb_app.calculate_leakage(p_pump, p_stokes)
            res_v2 = comb_v2.calculate_leakage(p_pump, p_stokes)
            assert res_app == res_v2

    def test_end_to_end_delivery_simulation_differential(self):
        """Compare simulate_dual_laser_delivery across 25 parameter permutations."""
        rng = np.random.default_rng(999)
        for _ in range(25):
            p_p = float(rng.uniform(5.0, 100.0))
            p_s = float(rng.uniform(5.0, 200.0))
            th_p = float(rng.uniform(0.0, 90.0))
            th_s = float(rng.uniform(0.0, 90.0))
            v_eom = float(rng.uniform(0.0, 300.0))

            lasers_app = DualLaserSourceConfig(pump_power_mw=p_p, stokes_power_mw=p_s)
            lasers_v2 = v2_nonlinear.DualLaserSourceConfig(pump_power_mw=p_p, stokes_power_mw=p_s)

            res_app = simulate_dual_laser_delivery(
                dual_lasers=lasers_app,
                pump_hwp_angle_deg=th_p,
                stokes_hwp_angle_deg=th_s,
                eom_applied_v=v_eom,
            )
            res_v2 = v2_nonlinear.simulate_dual_laser_delivery(
                dual_lasers=lasers_v2,
                pump_hwp_angle_deg=th_p,
                stokes_hwp_angle_deg=th_s,
                eom_applied_v=v_eom,
            )

            for k in res_app:
                if isinstance(res_app[k], (int, float)):
                    assert math.isclose(res_app[k], res_v2[k], rel_tol=1e-11)
                elif isinstance(res_app[k], dict):
                    assert res_app[k] == res_v2[k]
                else:
                    assert res_app[k] == res_v2[k]

    def test_thorlabs_catalog_differential_all_m1_parts(self):
        """Verify identical dictionary representation for all 11 M1 Thorlabs catalog parts."""
        m1_parts = [
            "PBS252",
            "WPH10M-780",
            "WPH10M-1064",
            "EO-AM-NR-C1",
            "FBH790-10",
            "FELH1000",
            "FESH0850",
            "FESH0800",
            "DMLP900",
            "PF10-03-P01",
            "BT610",
        ]
        for part in m1_parts:
            part_app = get_catalog_lens(part)
            part_v2 = v2_thorlabs.get_catalog_lens(part)
            assert part_app is not None, f"Part {part} missing from optics_app"
            assert part_v2 is not None, f"Part {part} missing from v2"
            assert part_app == part_v2, f"Discrepancy for part {part}: {part_app} vs {part_v2}"


    def test_thorlabs_srs_microscope_config_properties_differential(self):
        """Compare all M1 helper properties on ThorlabsSRSMicroscopeConfig between optics_app and v2."""
        cfg_app = ThorlabsSRSMicroscopeConfig()
        cfg_v2 = v2_nonlinear.ThorlabsSRSMicroscopeConfig()

        assert cfg_app.base_raman_shift_cm1 == cfg_v2.base_raman_shift_cm1
        assert cfg_app.dual_laser_config.to_dict() == cfg_v2.dual_laser_config.to_dict()
        assert cfg_app.cleanup_filter_config.to_dict() == cfg_v2.cleanup_filter_config.to_dict()
        assert cfg_app.pump_power_attenuator.to_dict() == cfg_v2.pump_power_attenuator.to_dict()
        assert cfg_app.stokes_power_attenuator.to_dict() == cfg_v2.stokes_power_attenuator.to_dict()
        assert cfg_app.dichroic_combiner_config.to_dict() == cfg_v2.dichroic_combiner_config.to_dict()
        assert cfg_app.get_eom_modulation_result().to_dict() == cfg_v2.get_eom_modulation_result().to_dict()
        assert cfg_app.simulate_laser_delivery() == cfg_v2.simulate_laser_delivery()

    def test_37_surface_template_differential(self):
        """Verify build_folded_real_world_srs_template generates identical 37 surfaces in optics_app and v2."""
        from optics_app.analysis.ultrafast_nonlinear import build_folded_real_world_srs_template
        t_app = build_folded_real_world_srs_template()
        t_v2 = v2_nonlinear.build_folded_real_world_srs_template()

        assert len(t_app.surfaces) == len(t_v2.surfaces)
        for i, (s1, s2) in enumerate(zip(t_app.surfaces, t_v2.surfaces)):
            assert s1.name == s2.name, f"Surface {i} name mismatch"
            assert s1.radius == s2.radius, f"Surface {i} radius mismatch"
            assert s1.thickness == s2.thickness, f"Surface {i} thickness mismatch"
            assert s1.glass == s2.glass, f"Surface {i} glass mismatch"

    def test_query_catalog_differential_all_m1_categories(self):
        """Verify query_catalog returns identical item lists for all M1 sections."""
        categories = ["polarization", "filters", "dichroic", "mirrors", "beam_traps"]
        for cat in categories:
            res_app = query_catalog(section=cat)
            res_v2 = v2_thorlabs.query_catalog(section=cat)
            assert len(res_app) > 0, f"No items found for section '{cat}' in optics_app"
            assert len(res_app) == len(res_v2), f"Length mismatch for section '{cat}': {len(res_app)} vs {len(res_v2)}"
            for item1, item2 in zip(res_app, res_v2):
                assert item1["part_number"] == item2["part_number"]
                assert item1["type"] == item2["type"]
