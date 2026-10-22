"""Adversarial stress-test suite for Milestone 1: Dual-Laser Delivery & EOM Modulation.

Challenger: challenger_srs_m1_1
Target: Milestone 1 components in optics_app and v2
Scope:
1. Extreme HWP angles (negative, >360 deg, Malus periodicity, zero-angle override bug).
2. 10.0 MHz sinusoidal integration over multiple periods (1 to 1000 periods, quadrature,
   overmodulation, time-averaged rejected power vs dump_absorbed_power_mw).
3. Filter blocking ODs (in-band transmission, out-of-band rejection, edge transitions, extreme wavelengths).
4. Combiner and end-to-end delivery energy balance (conservation laws, unaccounted optical power).
"""

import math
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
    ElectroOpticModulator,
    EOMMaterial,
)
import v2.analysis.ultrafast_nonlinear as v2_nonlinear


# ==============================================================================
# 1. EXTREME HWP ANGLES & ATTENUATION
# ==============================================================================

class TestExtremeHWPAngles:
    """Stress-test half-wave plate rotation angles and polarization attenuation."""

    @pytest.mark.parametrize("angle_deg", [
        0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 360.0, 720.0, 1000.0, 1e5, 1e6,
        -45.0, -90.0, -180.0, -360.0, -720.0, -1e5, -1e6,
        22.5, 67.5, 112.5, -22.5, -67.5,
    ])
    def test_hwp_angles_energy_conservation(self, angle_deg):
        """Verify exact energy conservation P_trans + P_dump == P_in for all angles."""
        atten = PolarizationPowerAttenuator(input_power_mw=37.5, t_max=0.92, extinction_ratio=1000.0)
        p_trans = atten.transmitted_power_mw(angle_deg)
        p_dump = atten.dumped_power_mw(angle_deg)
        assert math.isclose(p_trans + p_dump, 37.5, rel_tol=1e-5)
        assert p_trans >= 0.0
        assert p_dump >= 0.0

    @pytest.mark.parametrize("angle_deg", [
        15.0, 32.7, 58.3, 120.0, 250.0, -17.3, -83.5, 945.2,
    ])
    def test_malus_law_periodicity(self, angle_deg):
        """Malus' law through HWP and PBS has fundamental angular period of 90 degrees."""
        atten = PolarizationPowerAttenuator(input_power_mw=50.0)
        t_base = atten.transmission_fraction(angle_deg)
        t_plus_90 = atten.transmission_fraction(angle_deg + 90.0)
        t_plus_180 = atten.transmission_fraction(angle_deg + 180.0)
        t_minus_90 = atten.transmission_fraction(angle_deg - 90.0)
        assert math.isclose(t_base, t_plus_90, abs_tol=1e-5)
        assert math.isclose(t_base, t_plus_180, abs_tol=1e-5)
        assert math.isclose(t_base, t_minus_90, abs_tol=1e-5)

    def test_attenuation_db_bounds_and_monotinicity(self):
        """Verify attenuation_db remains finite and bounded across angles."""
        atten = PolarizationPowerAttenuator(t_max=0.92, extinction_ratio=1000.0)
        min_att = atten.attenuation_db(0.0)
        max_att = atten.attenuation_db(45.0)
        assert math.isclose(min_att, -10.0 * math.log10(0.92), rel_tol=1e-4)
        assert max_att >= 30.0
        # Intermediate angle 22.5 deg must be between min and max
        mid_att = atten.attenuation_db(22.5)
        assert min_att < mid_att < max_att

    def test_zero_angle_override_bug_in_simulate_dual_laser_delivery(self):
        """Verify calling simulate_dual_laser_delivery with pump_hwp_angle_deg=0.0
        when config has non-zero angle correctly applies 0.0."""
        cfg = ThorlabsSRSMicroscopeConfig(pump_hwp_angle_deg=45.0, stokes_hwp_angle_deg=45.0)
        # Caller explicitly requests 0.0 deg (full transmission)
        res = simulate_dual_laser_delivery(cfg, pump_hwp_angle_deg=0.0, stokes_hwp_angle_deg=0.0)
        applied_pump_angle = res["pump_attenuator"]["hwp_angle_deg"]
        applied_stokes_angle = res["stokes_attenuator"]["hwp_angle_deg"]
        
        assert applied_pump_angle == 0.0
        assert applied_stokes_angle == 0.0
        assert res["collinear_pump_power_mw"] > 18.0
        assert res["collinear_stokes_dc_power_mw"] > 18.0


# ==============================================================================
# 2. 10.0 MHz SINUSOIDAL INTEGRATION OVER MULTIPLE PERIODS
# ==============================================================================

class TestSinusoidalIntegration:
    """Stress-test 10.0 MHz EOM intensity modulation integration over multiple periods."""

    @pytest.mark.parametrize("n_periods", [1, 2, 5, 10, 50, 100, 1000])
    def test_multi_period_intensity_integration(self, n_periods):
        """Integrating I_S(t) = I_{S,0} [1 + m cos(Omega t + phi)] over any integer
        number of periods must strictly equal N * T * I_{S,0}.
        """
        f_mod = 10.0e6  # 10 MHz
        period = 1.0 / f_mod  # 100 ns
        total_time = n_periods * period
        dc_intensity = 24.5
        m_depth = 0.85
        phi = 0.785398  # pi/4

        res = EOMModulationResult(
            modulation_freq_mhz=10.0,
            modulation_depth=m_depth,
            dc_intensity=dc_intensity,
            phase_rad=phi,
        )

        # High-resolution numerical integration (midpoint rule over 200 samples per period)
        samples_per_period = 200
        n_samples = n_periods * samples_per_period
        t_arr = np.linspace(0.0, total_time, n_samples, endpoint=False) + (period / (2.0 * samples_per_period))
        i_trans = res.transmitted_intensity(t_arr)

        numerical_mean = float(np.mean(i_trans))
        assert math.isclose(numerical_mean, dc_intensity, rel_tol=1e-5), (
            f"Mean intensity over {n_periods} periods deviated from DC intensity: {numerical_mean} vs {dc_intensity}"
        )

    def test_discrete_laser_pulse_sampling_at_80mhz(self):
        """In SRS, the 10.0 MHz modulated Stokes beam is sampled by an 80.0 MHz pulsed train.
        There are exactly 8 pulses per modulation period.
        Verify that averaging over 8 pulses yields exact DC intensity.
        """
        res = EOMModulationResult(
            modulation_freq_mhz=10.0,
            modulation_depth=0.90,
            dc_intensity=30.0,
            phase_rad=0.35,
        )
        # 8 pulses at t = k * 12.5 ns
        pulse_times = np.arange(8) * 12.5e-9
        i_pulses = res.transmitted_intensity(pulse_times)
        assert math.isclose(float(np.mean(i_pulses)), 30.0, rel_tol=1e-6)

    def test_eom_overmodulation_negative_intensity_defect(self):
        """Verify intensity remains non-negative under overmodulation (m > 1.0)."""
        res = EOMModulationResult(
            modulation_freq_mhz=10.0,
            modulation_depth=1.2,  # Overmodulated
            dc_intensity=20.0,
        )
        t_arr = np.linspace(0.0, 1.0e-7, 101)
        i_trans = res.transmitted_intensity(t_arr)
        min_intensity = np.min(i_trans)
        assert min_intensity >= 0.0, "Intensity must be non-negative under overmodulation"

    def test_eom_dump_absorbed_power_contradiction(self):
        """Verify exact agreement between rejected waveform mean and dump_absorbed_power_mw."""
        cfg = ThorlabsSRSMicroscopeConfig()
        eom = cfg.get_eom_modulation_result()

        period = 1.0 / (eom.modulation_freq_mhz * 1e6)
        t_arr = np.linspace(0.0, 10.0 * period, 1000, endpoint=False)
        actual_rejected_mean = float(np.mean(eom.rejected_intensity(t_arr)))

        assert math.isclose(actual_rejected_mean, 27.6, rel_tol=1e-3)
        assert math.isclose(eom.dump_absorbed_power_mw, 27.6, rel_tol=1e-3)
        discrepancy = abs(actual_rejected_mean - eom.dump_absorbed_power_mw)
        assert discrepancy < 0.05, f"Discrepancy {discrepancy} mW exceeds 0.05 mW"


# ==============================================================================
# 3. FILTER BLOCKING ODs & TRANSITIONS
# ==============================================================================

class TestFilterBlockingODs:
    """Stress-test clean-up filter optical densities and edge band transmissions."""

    def test_pump_filter_blocking_od_levels(self):
        """FBH790-10 must exhibit OD > 6 outside passband."""
        flt = CleanupFilterConfig()
        # Out-of-band test at Stokes wavelength 1040 nm
        t_out = flt.get_transmission("pump", 1040.0)
        assert t_out <= 1.0e-6
        # Equivalent OD
        od_out = -math.log10(t_out)
        assert od_out >= 6.0

    def test_stokes_filter_blocking_od_levels(self):
        """FELH1000 must exhibit OD > 5 below cut-on (1000 nm)."""
        flt = CleanupFilterConfig()
        # Out-of-band test at Pump wavelength 797 nm
        t_out = flt.get_transmission("stokes", 797.0)
        assert t_out <= 1.0e-5
        # Equivalent OD
        od_out = -math.log10(t_out)
        assert od_out >= 5.0

    def test_pump_filter_passband_edge_sensitivity(self):
        """FBH790-10 has CWL = 790 nm and FWHM = 10 nm.
        Passband is defined as abs(lambda - 790) <= 8.0 nm ([782, 798] nm).
        Verify behavior at 797 nm (passband) vs 799 nm (abrupt OD 6 rejection).
        """
        flt = CleanupFilterConfig()
        assert flt.is_pump_transmission_valid(797.0)
        assert flt.get_transmission("pump", 797.0) == pytest.approx(0.92)

        # At 799.0 nm (just 2 nm higher), transmission drops 6 orders of magnitude
        assert not flt.is_pump_transmission_valid(799.0)
        assert flt.get_transmission("pump", 799.0) <= 1.0e-6

    def test_filter_extreme_and_unphysical_wavelengths(self):
        """Verify behavior on extreme/negative wavelengths."""
        flt = CleanupFilterConfig()
        # Negative wavelength should be blocked by pump filter
        assert flt.get_transmission("pump", -790.0) <= 1.0e-6
        # Extreme short wavelength (X-ray / UV 10 nm)
        assert flt.get_transmission("pump", 10.0) <= 1.0e-6
        assert flt.get_transmission("stokes", 10.0) <= 1.0e-5
        # Extreme long wavelength (Far-IR 100,000 nm)
        # Note: Stokes longpass will treat any lambda >= 1000 nm as transparent
        assert flt.get_transmission("stokes", 100000.0) == pytest.approx(0.92)

    def test_unknown_beam_name_handling(self):
        flt = CleanupFilterConfig()
        assert flt.get_transmission("unknown_beam", 800.0) == 0.0


# ==============================================================================
# 4. COMBINER AND DELIVERY ENERGY BALANCE
# ==============================================================================

class TestEnergyBalanceAndConservation:
    """Stress-test energy balance across dichroic combiner and delivery subsystem."""

    def test_dichroic_combiner_isolated_energy_conservation(self):
        """In isolated DichroicCombinerConfig, combined + leakage must equal input exactly."""
        combiner = DichroicCombinerConfig(pump_reflection_r=0.98, stokes_transmission_t=0.92)
        pump_in = 25.0
        stokes_in = 60.0
        leak = combiner.calculate_leakage(pump_in, stokes_in)

        # Pump energy conservation
        assert math.isclose(leak["combined_pump_mw"] + leak["pump_leakage_mw"], pump_in, rel_tol=1e-6)
        # Stokes energy conservation
        assert math.isclose(leak["combined_stokes_mw"] + leak["stokes_leakage_mw"], stokes_in, rel_tol=1e-6)
        # Total energy conservation
        assert math.isclose(
            leak["combined_pump_mw"] + leak["combined_stokes_mw"] + leak["total_dump_power_mw"],
            pump_in + stokes_in,
            rel_tol=1e-6,
        )

    def test_full_delivery_simulation_energy_leak_defect(self):
        """Verify full optical energy conservation across the dual-laser delivery subsystem."""
        cfg = ThorlabsSRSMicroscopeConfig()
        sim = simulate_dual_laser_delivery(cfg)

        p_in_total = cfg.pump_laser_power_mw + cfg.stokes_laser_power_mw  # 85.0 mW
        p_collinear_total = sim["collinear_pump_power_mw"] + sim["collinear_stokes_dc_power_mw"]
        p_dumped_total = sim["total_bt610_dumped_power_mw"]
        p_loss_total = sim["total_insertion_loss_mw"]
        p_accounted = p_collinear_total + p_dumped_total + p_loss_total
        p_missing = abs(p_in_total - p_accounted)

        assert p_in_total == pytest.approx(85.0)
        assert math.isclose(p_accounted, 85.0, abs_tol=1e-4)
        assert p_missing < 1e-4, f"Missing power is {p_missing} mW, expected < 1e-4 mW"
        assert sim["total_bt610_dumped_power_mw"] > 30.0  # BT610 dumps absorb ~32.45 mW
        assert sim["unaccounted_power_mw"] < 1e-4
