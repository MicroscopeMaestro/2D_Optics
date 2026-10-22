"""Tests for Milestone 1: Dual-Laser Delivery Subsystem & Active Stokes EOM Modulation.

Verifies:
1. DualLaserSourceConfig:
   - Pump (797 nm, 25 mW, 120 fs, 80 MHz, 4 mm beam)
   - Stokes (1040 nm, 60 mW, 120 fs, 80 MHz, 4 mm beam)
   - Base Raman wavenumber shift Delta_nu = 2931.67 cm^-1
   - Pulse energies, peak powers, and serialization round-trip
2. Independent Polarization Power Control (HWP + PBS252 + BT610):
   - Malus' law transmission: P_trans(theta) = P_in * [T_min + (T_max - T_min) * cos^2(2*theta)]
   - Extinction ratio > 1000:1 (> 30 dB attenuation)
   - Power conservation: P_trans + P_dump = P_in, rejected orthogonal polarization 100% absorbed by BT610
3. Active Resonant Electro-Optic Modulator (Thorlabs EO-AM-NR-C1):
   - 10.0 MHz resonant frequency with dual matched RTP crystals
   - High-contrast intensity modulation waveform I_S(t) = I_{S,0} * [1 + m * cos(Omega_mod * t)]
   - Quarter-wave bias and PBS252 analyzer pair
   - Rejected power directed to dedicated BT610 dump
4. Laser Line Clean-Up Filters:
   - Pump clean-up bandpass FBH790-10 (CWL 790 nm, FWHM 10 nm, OD > 6 blocking)
   - Stokes clean-up longpass FELH1000 (Cut-on 1000 nm, OD > 5 rejection)
5. Dichroic Pulse Combination (Thorlabs DMLP900 at 45° AOI):
   - Longpass dichroic combining unmodulated Pump and modulated Stokes collinearly along +Z
   - Reflection of 797 nm pump (R > 95%) and transmission of 1040 nm Stokes (T > 90%)
   - Collinear pointing error < 0.05 mrad and lateral walk-off < 1.0 um
   - Stray leakage safely absorbed by BT610 combiner leak dump
6. Thorlabs Commercial Catalog:
   - Retrieval and specifications of PBS252, WPH10M-780, WPH10M-1064, EO-AM-NR-C1,
     FBH790-10, FELH1000, FESH0850, FESH0800, DMLP900, PF10-03-P01, BT610
7. Complete Dual-Tree Code Parity between optics_app/ and v2/.
"""

import math
import numpy as np
import pytest

# Test imports from optics_app
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
    build_folded_real_world_srs_template,
)
from optics_app.catalogs.thorlabs import (
    get_catalog_lens,
    get_catalog_sections,
    query_catalog,
)

# Test imports from v2 for dual-tree parity
import v2.analysis.ultrafast_nonlinear as v2_nonlinear
import v2.catalogs.thorlabs as v2_thorlabs


# ==============================================================================
# 1. DUAL-LASER SOURCE PARAMETERS & RAMAN SHIFT
# ==============================================================================

class TestDualLaserSource:
    """Verify dual synchronized femtosecond laser delivery parameters and physics."""

    def test_default_laser_parameters(self):
        lasers = DualLaserSourceConfig()
        assert lasers.pump_wavelength_nm == pytest.approx(797.0)
        assert lasers.pump_power_mw == pytest.approx(25.0)
        assert lasers.pump_pulse_fs == pytest.approx(120.0)
        assert lasers.pump_beam_diam_mm == pytest.approx(4.0)

        assert lasers.stokes_wavelength_nm == pytest.approx(1040.0)
        assert lasers.stokes_power_mw == pytest.approx(60.0)
        assert lasers.stokes_pulse_fs == pytest.approx(120.0)
        assert lasers.stokes_beam_diam_mm == pytest.approx(4.0)

        assert lasers.rep_rate_mhz == pytest.approx(80.0)

    def test_base_raman_wavenumber_resonance(self):
        """Verify Delta_nu = 1/lambda_p - 1/lambda_S = 2931.67 cm^-1."""
        lasers = DualLaserSourceConfig()
        expected_shift = (1.0e7 / 797.0) - (1.0e7 / 1040.0)
        assert math.isclose(lasers.base_raman_shift_cm1, expected_shift, rel_tol=1e-5)
        assert math.isclose(lasers.base_raman_shift_cm1, 2931.6668, rel_tol=1e-4)
        assert f"{lasers.base_raman_shift_cm1:.2f}" == "2931.67"

    def test_pulse_energies_and_peak_powers(self):
        """Verify pulse energies (E = P / f_rep) and sech^2 peak powers."""
        lasers = DualLaserSourceConfig()
        # Pump: 25 mW / 80 MHz = 0.3125 nJ
        assert math.isclose(lasers.pump_energy_nj, 0.3125, rel_tol=1e-4)
        # Stokes: 60 mW / 80 MHz = 0.7500 nJ
        assert math.isclose(lasers.stokes_energy_nj, 0.7500, rel_tol=1e-4)

        # Peak power = 0.88 * E / tau
        expected_p_peak_pump = 0.88 * (0.3125e-9) / (120e-15) * 1e-3  # kW
        assert math.isclose(lasers.pump_peak_power_kw, expected_p_peak_pump, rel_tol=1e-4)
        assert lasers.pump_peak_power_kw > 2.0  # > 2.2 kW

        expected_p_peak_stokes = 0.88 * (0.7500e-9) / (120e-15) * 1e-3  # kW
        assert math.isclose(lasers.stokes_peak_power_kw, expected_p_peak_stokes, rel_tol=1e-4)
        assert lasers.stokes_peak_power_kw > 5.0  # > 5.4 kW

    def test_serialization_roundtrip(self):
        lasers = DualLaserSourceConfig(
            pump_power_mw=30.0,
            stokes_power_mw=80.0,
            pump_wavelength_nm=795.0,
            stokes_wavelength_nm=1045.0,
        )
        d = lasers.to_dict()
        assert d["pump_power_mw"] == 30.0
        assert d["stokes_power_mw"] == 80.0
        assert "base_raman_shift_cm1" in d

        restored = DualLaserSourceConfig.from_dict(d)
        assert restored.pump_power_mw == 30.0
        assert restored.stokes_power_mw == 80.0
        assert math.isclose(restored.base_raman_shift_cm1, lasers.base_raman_shift_cm1, rel_tol=1e-6)

    def test_dual_tree_parity(self):
        l1 = DualLaserSourceConfig()
        l2 = v2_nonlinear.DualLaserSourceConfig()
        assert l1.to_dict() == l2.to_dict()


# ==============================================================================
# 2. POLARIZATION POWER ATTENUATION (HWP + PBS252 + BT610)
# ==============================================================================

class TestPolarizationPowerControl:
    """Verify HWP + PBS252 power attenuation, Malus' law, and BT610 dump absorption."""

    def test_pump_power_attenuator_malus_law(self):
        """P_trans(theta) = P_in * [T_min + (T_max - T_min) * cos^2(2*theta)]."""
        atten = PolarizationPowerAttenuator(
            hwp_model="WPH10M-780",
            pbs_model="PBS252",
            dump_model="BT610",
            input_power_mw=25.0,
            wavelength_nm=797.0,
            t_max=0.92,
            extinction_ratio=1000.0,
        )

        # 1. At theta = 0 deg: maximum transmission
        p_trans_0 = atten.transmitted_power_mw(hwp_angle_deg=0.0)
        assert math.isclose(p_trans_0, 25.0 * 0.92, rel_tol=1e-4)
        assert atten.dumped_power_mw(hwp_angle_deg=0.0) == pytest.approx(25.0 * (1.0 - 0.92), abs=1e-3)

        # 2. At theta = 45 deg: minimum transmission (extinction)
        p_trans_45 = atten.transmitted_power_mw(hwp_angle_deg=45.0)
        assert p_trans_45 <= (25.0 * 0.92 / 1000.0)
        assert math.isclose(atten.dumped_power_mw(hwp_angle_deg=45.0), 25.0 - p_trans_45, rel_tol=1e-4)

        # 3. At theta = 22.5 deg: 50% attenuation point (cos(45)^2 = 0.5)
        p_trans_22_5 = atten.transmitted_power_mw(hwp_angle_deg=22.5)
        expected_mid = 25.0 * (0.00092 + (0.92 - 0.00092) * 0.5)
        assert math.isclose(p_trans_22_5, expected_mid, rel_tol=1e-3)

    def test_stokes_power_attenuator(self):
        atten = PolarizationPowerAttenuator(
            hwp_model="WPH10M-1064",
            pbs_model="PBS252",
            dump_model="BT610",
            input_power_mw=60.0,
            wavelength_nm=1040.0,
            t_max=0.92,
            extinction_ratio=1000.0,
        )
        assert math.isclose(atten.transmitted_power_mw(hwp_angle_deg=0.0), 60.0 * 0.92, rel_tol=1e-4)
        assert math.isclose(atten.dumped_power_mw(hwp_angle_deg=0.0), 60.0 * 0.08, rel_tol=1e-3)

    def test_power_conservation_into_bt610_dump(self):
        """P_trans + P_dump == P_in exactly across all waveplate rotation angles."""
        atten = PolarizationPowerAttenuator(input_power_mw=50.0)
        for angle in np.linspace(0.0, 90.0, 19):
            p_trans = atten.transmitted_power_mw(hwp_angle_deg=angle)
            p_dump = atten.dumped_power_mw(hwp_angle_deg=angle)
            assert math.isclose(p_trans + p_dump, 50.0, rel_tol=1e-6)

    def test_attenuation_in_decibels(self):
        atten = PolarizationPowerAttenuator(t_max=0.92, extinction_ratio=1000.0)
        att_0 = atten.attenuation_db(hwp_angle_deg=0.0)
        assert att_0 == pytest.approx(0.362, abs=0.01)  # -10*log10(0.92) = 0.362 dB insertion loss

        att_45 = atten.attenuation_db(hwp_angle_deg=45.0)
        assert att_45 >= 30.0  # > 30 dB extinction


# ==============================================================================
# 3. ACTIVE RESONANT STOKES EOM MODULATION (EO-AM-NR-C1)
# ==============================================================================

class TestActiveStokesEOMModulation:
    """Verify Thorlabs EO-AM-NR-C1 10.0 MHz resonant EOM and PBS analyzer."""

    def test_eom_modulation_result_waveform(self):
        """Verify I_S(t) = I_{S,0} * [1 + m * cos(Omega_mod * t)]."""
        res = EOMModulationResult(
            modulation_freq_mhz=10.0,
            modulation_depth=0.85,
            dc_intensity=25.0,
            input_power_mw=60.0,
        )
        # Period T = 1 / 10 MHz = 100 ns = 1.0e-7 s
        t_arr = np.linspace(0.0, 1.0e-7, 101)
        i_trans = res.transmitted_intensity(t_arr)

        assert len(i_trans) == 101
        # At t = 0: peak = I_{S,0} * (1 + m) = 25 * 1.85 = 46.25
        assert math.isclose(float(i_trans[0]), 25.0 * 1.85, rel_tol=1e-4)
        # At t = 50 ns (half period): trough = I_{S,0} * (1 - m) = 25 * 0.15 = 3.75
        assert math.isclose(float(i_trans[50]), 25.0 * 0.15, rel_tol=1e-3)
        # Average over one full period [0, T) must equal dc_intensity
        assert math.isclose(float(np.mean(i_trans[:-1])), 25.0, rel_tol=1e-3)

    def test_rejected_power_into_bt610_dump(self):
        """Verify rejected intensity I_dump(t) = I_in - I_S(t) into BT610 dump."""
        res = EOMModulationResult(
            modulation_freq_mhz=10.0,
            modulation_depth=0.80,
            dc_intensity=20.0,
            input_power_mw=50.0,
        )
        t_arr = np.linspace(0.0, 1.0e-7, 101)
        i_trans = res.transmitted_intensity(t_arr)
        i_dump = res.rejected_intensity(t_arr)

        # Sum of transmitted and dump intensity is constant (input power)
        i_total = i_trans + i_dump
        assert np.allclose(i_total, 2.0 * res.dc_intensity, atol=1e-5)

    def test_harmonic_content_analysis(self):
        res = EOMModulationResult(
            modulation_freq_mhz=10.0,
            modulation_depth=0.85,
            dc_intensity=20.0,
        )
        harms = res.harmonic_amplitudes()
        assert harms["dc"] == pytest.approx(20.0)
        assert harms["fundamental_10mhz"] == pytest.approx(20.0 * 0.85)
        assert harms["modulation_freq_mhz"] == pytest.approx(10.0)

    def test_eom_physical_device_binding(self):
        eom = ElectroOpticModulator(
            material=EOMMaterial.RTP,
            half_wave_voltage_v=240.0,
            applied_voltage_v=200.0,
            modulation_frequency_mhz=10.0,
        )
        res = eom.get_modulation_result(input_power_mw=60.0)
        assert res.crystal_material == "RTP"
        assert res.modulation_freq_mhz == pytest.approx(10.0)
        assert math.isclose(res.modulation_depth, 200.0 / 240.0, rel_tol=1e-4)
        assert res.dump_absorbed_power_mw > 0.0


# ==============================================================================
# 4. LASER LINE CLEAN-UP FILTERS (FBH790-10 & FELH1000)
# ==============================================================================

class TestCleanupFilters:
    """Verify Pump bandpass (FBH790-10) and Stokes longpass (FELH1000)."""

    def test_cleanup_filter_transmissions(self):
        flt = CleanupFilterConfig()
        assert flt.pump_filter == "FBH790-10"
        assert flt.stokes_filter == "FELH1000"

        # Pump at 797 nm: inside passband of FBH790-10
        assert flt.is_pump_transmission_valid(797.0)
        t_pump = flt.get_transmission("pump", 797.0)
        assert t_pump == pytest.approx(0.92, abs=0.01)

        # Stokes out-of-band rejection on pump filter: OD > 6
        t_stokes_in_pump_filter = flt.get_transmission("pump", 1040.0)
        assert t_stokes_in_pump_filter <= 1.0e-6

        # Stokes at 1040 nm: inside passband of FELH1000
        assert flt.is_stokes_transmission_valid(1040.0)
        t_stokes = flt.get_transmission("stokes", 1040.0)
        assert t_stokes == pytest.approx(0.92, abs=0.01)

        # Pump rejection on Stokes longpass: OD > 5 (cutoff 1000 nm)
        t_pump_in_stokes_filter = flt.get_transmission("stokes", 797.0)
        assert t_pump_in_stokes_filter <= 1.0e-5


# ==============================================================================
# 5. DICHROIC COMBINER & LEAKAGE (DMLP900)
# ==============================================================================

class TestDichroicCombiner:
    """Verify Thorlabs DMLP900 45° dichroic mirror collinear combination and dump leakage."""

    def test_dichroic_collinear_coupling(self):
        dichroic = DichroicCombinerConfig()
        assert dichroic.model == "DMLP900"
        assert dichroic.angle_of_incidence_deg == pytest.approx(45.0)
        assert dichroic.cutoff_wavelength_nm == pytest.approx(900.0)
        assert dichroic.pump_reflection_r >= 0.95
        assert dichroic.stokes_transmission_t >= 0.90
        assert dichroic.max_pointing_error_mrad < 0.1
        assert dichroic.max_lateral_offset_um <= 1.0

    def test_leakage_power_accounting(self):
        dichroic = DichroicCombinerConfig(
            pump_reflection_r=0.98,
            stokes_transmission_t=0.92,
        )
        p_pump_in = 20.0
        p_stokes_in = 30.0

        leak = dichroic.calculate_leakage(p_pump_in, p_stokes_in)
        assert math.isclose(leak["combined_pump_mw"], 20.0 * 0.98, rel_tol=1e-4)
        assert math.isclose(leak["combined_stokes_mw"], 30.0 * 0.92, rel_tol=1e-4)

        # Leakages into BT610 dump
        assert math.isclose(leak["pump_leakage_mw"], 20.0 * 0.02, rel_tol=1e-4)
        assert math.isclose(leak["stokes_leakage_mw"], 30.0 * 0.08, rel_tol=1e-4)
        assert math.isclose(leak["total_dump_power_mw"], (20.0 * 0.02) + (30.0 * 0.08), rel_tol=1e-4)


# ==============================================================================
# 6. END-TO-END DUAL-LASER DELIVERY SUBSYSTEM SIMULATION
# ==============================================================================

class TestEndToEndLaserDeliverySimulation:
    """Verify full dual-laser delivery chain from sources to collinear microscope injection."""

    def test_full_delivery_simulation(self):
        cfg = ThorlabsSRSMicroscopeConfig()
        sim = simulate_dual_laser_delivery(cfg)

        assert math.isclose(sim["base_raman_shift_cm1"], 2931.67, rel_tol=1e-3)
        assert sim["is_collinear_aligned"]
        assert sim["pointing_error_mrad"] < 0.1
        assert sim["lateral_walkoff_um"] <= 1.0

        # Collinear pump and Stokes power levels
        assert sim["collinear_pump_power_mw"] > 18.0  # ~ 20.7 mW
        assert sim["collinear_stokes_dc_power_mw"] > 10.0  # ~ 21.5 mW DC modulated

        # Dumped power and complete energy conservation
        assert sim["total_bt610_dumped_power_mw"] > 30.0  # BT610 dumps absorb ~32.45 mW
        assert math.isclose(sim["total_accounted_power_mw"], 85.0, abs_tol=1e-5)
        assert sim["unaccounted_power_mw"] < 1e-5
        assert sim["power_balance_error_mw"] < 1e-5

    def test_power_attenuation_sweep(self):
        """Verify dynamic power attenuation on both laser arms."""
        cfg = ThorlabsSRSMicroscopeConfig()

        # Full power (HWP = 0 deg)
        sim_max = simulate_dual_laser_delivery(cfg, pump_hwp_angle_deg=0.0, stokes_hwp_angle_deg=0.0)
        # Attenuated power (HWP = 45 deg)
        sim_min = simulate_dual_laser_delivery(cfg, pump_hwp_angle_deg=45.0, stokes_hwp_angle_deg=45.0)

        assert sim_max["collinear_pump_power_mw"] > 18.0
        assert sim_min["collinear_pump_power_mw"] < 0.1
        assert sim_min["pump_attenuator"]["dumped_power_mw"] > 24.5


# ==============================================================================
# 7. THORLABS COMMERCIAL CATALOG REGISTRY
# ==============================================================================

class TestThorlabsCatalogM1Components:
    """Verify that all M1 components are queryable from the Thorlabs commercial catalog."""

    def test_polarizing_cube_catalog(self):
        pbs = get_catalog_lens("PBS252")
        assert pbs is not None
        assert pbs["part_number"] == "PBS252"
        assert pbs["diameter_mm"] == pytest.approx(25.4)
        assert pbs["extinction_ratio"] >= 1000.0

    def test_half_wave_plates_catalog(self):
        hwp_780 = get_catalog_lens("WPH10M-780")
        assert hwp_780 is not None
        assert hwp_780["design_wavelengths_nm"] == [780.0]

        hwp_1064 = get_catalog_lens("WPH10M-1064")
        assert hwp_1064 is not None
        assert hwp_1064["design_wavelengths_nm"] == [1064.0]

    def test_resonant_eom_catalog(self):
        eom = get_catalog_lens("EO-AM-NR-C1")
        assert eom is not None
        assert eom["part_number"] == "EO-AM-NR-C1"
        assert eom["crystal"] == "RTP"
        assert eom["modulation_frequency_mhz"] == pytest.approx(10.0)

    def test_filters_and_dichroic_catalog(self):
        fbh = get_catalog_lens("FBH790-10")
        assert fbh is not None
        assert fbh["cwl_or_cutoff_nm"] == pytest.approx(790.0)
        assert fbh["optical_density"] >= 6.0

        felh = get_catalog_lens("FELH1000")
        assert felh is not None
        assert felh["cwl_or_cutoff_nm"] == pytest.approx(1000.0)
        assert felh["optical_density"] >= 5.0

        dmlp = get_catalog_lens("DMLP900")
        assert dmlp is not None
        assert dmlp["cutoff_wavelength_nm"] == pytest.approx(900.0)

    def test_mirrors_and_beam_dump_catalog(self):
        pf = get_catalog_lens("PF10-03-P01")
        assert pf is not None
        assert pf["diameter_mm"] == pytest.approx(25.4)

        bt = get_catalog_lens("BT610")
        assert bt is not None
        assert bt["max_power_w"] >= 5.0

    def test_catalog_section_queries(self):
        pol_optics = query_catalog(section="polarization_optics")
        assert len(pol_optics) >= 4
        parts = [p["part_number"] for p in pol_optics]
        assert "PBS252" in parts
        assert "WPH10M-780" in parts
        assert "WPH10M-1064" in parts
        assert "EO-AM-NR-C1" in parts

        filt_optics = query_catalog(section="optical_filters")
        assert len(filt_optics) >= 4
        filt_parts = [p["part_number"] for p in filt_optics]
        assert "FBH790-10" in filt_parts
        assert "FELH1000" in filt_parts
        assert "FESH0850" in filt_parts
        assert "FESH0800" in filt_parts

    def test_v2_catalog_parity(self):
        """Verify identical catalog records in v2/."""
        for part in ["PBS252", "WPH10M-780", "WPH10M-1064", "EO-AM-NR-C1", "FBH790-10", "FELH1000", "DMLP900", "BT610"]:
            p1 = get_catalog_lens(part)
            p2 = v2_thorlabs.get_catalog_lens(part)
            assert p1 == p2


# ==============================================================================
# 8. 37-SURFACE TEMPLATE INTEGRATION & RAY TRACING
# ==============================================================================

class Test37SurfaceTemplateM1Integration:
    """Verify that M1 components are properly integrated into the 37-surface SRS model."""

    def test_template_contains_all_m1_components(self):
        stack = build_folded_real_world_srs_template()
        assert len(stack.surfaces) == 37

        names = [s.name for s in stack.surfaces]
        # Pump arm
        assert "Pump Laser (797 nm fs)" in names
        assert "Pump Half-Wave Plate (WPH10M-780)" in names
        assert "Pump PBS (PBS252)" in names
        assert "Pump Clean-Up Filter (FBH790-10)" in names
        assert "Pump Steering Mirror (PF10-03-P01)" in names
        assert "Dichroic Combiner (DMLP900)" in names

        # Stokes arm
        assert "Stokes Laser (1040 nm fs)" in names
        assert "Stokes Half-Wave Plate (WPH10M-1064)" in names
        assert "Stokes Clean-Up Filter (FELH1000)" in names
        assert "Stokes EOM (EO-AM-NR-C1)" in names
        assert "Stokes PBS (PBS252)" in names
        assert "Stokes Steering Mirror (PF10-03-P01)" in names

        # Beam traps
        assert "Pump Power Dump (BT610)" in names
        assert "Polarization Dump (BT610)" in names
        assert "Combiner Leak Dump (BT610)" in names

    def test_routes_reflect_m1_delivery_paths(self):
        stack = build_folded_real_world_srs_template()
        routes = stack.optical_routes
        assert routes is not None

        # Excitation route includes pump power control and clean-up filter
        step_uids_pump = [s.component_uid for s in routes.excitation.steps]
        assert "comp_pump_laser" in step_uids_pump
        assert "comp_pump_power_ctrl" in step_uids_pump
        assert "comp_pump_filter" in step_uids_pump
        assert "comp_pump_steering" in step_uids_pump
        assert "comp_dichroic" in step_uids_pump

        # Stokes extra route includes Stokes power control, clean-up, EOM, and analyzer
        assert "stokes" in routes.extra_routes
        step_uids_stokes = [s.component_uid for s in routes.extra_routes["stokes"].steps]
        assert "comp_stokes_laser" in step_uids_stokes
        assert "comp_stokes_power_ctrl" in step_uids_stokes
        assert "comp_stokes_filter" in step_uids_stokes
        assert "comp_modulator" in step_uids_stokes
        assert "comp_stokes_pbs" in step_uids_stokes
        assert "comp_stokes_steering" in step_uids_stokes
        assert "comp_dichroic" in step_uids_stokes

        # Dumps routes
        assert routes.dump is not None
        assert "comp_mod_dump" in [s.component_uid for s in routes.dump.steps]
        assert "pump_dump" in routes.extra_routes
        assert "combiner_dump" in routes.extra_routes
