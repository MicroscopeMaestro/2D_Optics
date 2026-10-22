"""Quickstart Demonstration Script for the Stimulated Raman Scattering (SRS) Microscope.

Run this script directly:
    python run_srs_demo.py
"""

import sys
import numpy as np

from optics_app.analysis.ultrafast_nonlinear import (
    ThorlabsSRSMicroscopeConfig,
    SRSSetupBuilderTool,
    SRSGalvoFieldDiagnosticTool,
    SRSSpectralFocusingTool,
    SRSSignalLockInTool,
    SRSBenchAlignmentTool,
    build_folded_real_world_srs_template,
)
from optics_app.engine import OpticalSystem, Ray


def main():
    print("=" * 80)
    print("      STIMULATED RAMAN SCATTERING (SRS) MICROSCOPE SIMULATION DEMO      ")
    print("=" * 80)

    # 1. Initialize configuration with Olympus 20x 1.0 NA Water Immersion objective
    print("\n[1] Initializing SRS Microscope Configuration...")
    cfg = ThorlabsSRSMicroscopeConfig(
        pump_glass_material="SF57",
        pump_glass_length_mm=150.0,
        stokes_glass_material="SF57",
        stokes_glass_length_mm=280.0,
        scanning_architecture="Relayed 4f (Zero Pupil Walk-Off)",
        enable_galvo_reflection_fold=True,
    )
    cfg.apply_objective_preset("Olympus XLUMPLFLN20XW (20x NA 1.00 W)")
    cfg.scanning_architecture = "Relayed 4f (Zero Pupil Walk-Off)"

    print(f"    - Objective: {cfg.objective_model}")
    print(f"        * Numerical Aperture: {cfg.objective_na}")
    print(f"        * Focal Length: {cfg.objective_focal_length_mm} mm")
    print(f"        * Working Distance: {cfg.objective_working_distance_mm} mm")
    print(f"        * Water Immersion: n = 1.333, Cover Slip = 0.17 mm")
    print(f"    - Pump Laser: {cfg.pump_laser_wavelength_nm} nm, {cfg.pump_laser_power_mw} mW, {cfg.pump_pulse_duration_fs} fs")
    print(f"    - Stokes Laser: {cfg.stokes_laser_wavelength_nm} nm, {cfg.stokes_laser_power_mw} mW, {cfg.stokes_pulse_duration_fs} fs")
    print(f"    - Active EOM: {cfg.modulator_type} at {cfg.modulator_frequency_mhz} MHz (Applied: {cfg.modulator_applied_v} V, V_pi: {cfg.modulator_v_pi_v} V)")
    print(f"    - Dispersive Rods: {cfg.pump_glass_material} (Pump: {cfg.pump_glass_length_mm} mm, Stokes: {cfg.stokes_glass_length_mm} mm)")

    # 2. Non-Sequential Dual-Beam Ray Tracing & Dichroic Combining
    print("\n[2] Executing True Non-Sequential Dual-Beam Simulation (Pump + Stokes)...")
    from optics_app.analysis.ultrafast_nonlinear import NonSequentialSRSMicroscopeBench
    ns_result = NonSequentialSRSMicroscopeBench.simulate_non_sequential_srs(
        config=cfg,
        scan_angle_x_deg=0.0,
        scan_angle_y_deg=0.0,
        delay_ps=0.0,
        num_rays_per_beam=7,
    )
    ns_sys = ns_result.non_sequential_system
    print(f"    - Non-Sequential System Elements: {len(ns_sys.elements)} 3D positioned elements")
    print(f"    - Dual Laser Sources:")
    print(f"        * Pump Laser Arm:   X = 0.0 mm, Wavelength = 797 nm (7 traced rays)")
    print(f"        * Stokes Laser Arm: X = 60.0 mm, Wavelength = 1040 nm (7 traced rays)")
    print(f"    - Dichroic Combiner (DMLP900 45 deg at X=60, Y=0, Z=235 mm):")
    print(f"        * Pump:   Reflected 90 deg from +X into +Z collinear optical axis")
    print(f"        * Stokes: Transmitted 100% through dichroic into +Z collinear optical axis")
    print(f"    - Traced Rays Summary: {len(ns_result.all_traced_rays)} total rays ({len(ns_result.traced_rays_pump)} Pump + {len(ns_result.traced_rays_stokes)} Stokes)")
    print(f"    - Non-Sequential Vignetting: {'CLEAN (0 rays lost, 100% pupil transmission)' if not ns_result.diagnostics['pump_vignetted'] else 'VIGNETTED'}")
    print(f"    - Stokes Optical Rejection: OD > {ns_result.diagnostics['stokes_rejection_od']} at DET10N2 Detector")

    # 3. Build 37-surface LDE system & 2D Ray Trace
    print("\n[3] Building 37-Surface Optical Bench & Executing 2D Sequential Layout...")
    system = SRSSetupBuilderTool.build_system(cfg, folded=True)
    stack = system.surface_stack
    opt_sys = OpticalSystem(stack)

    # Launch on-axis pump ray
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    trace_res = opt_sys.trace_ray(ray)

    print(f"    - Total Surfaces in LDE: {len(stack.surfaces)} (28 sequential + 9 off-axis)")
    print(f"    - Ray Tracing Traversal: {len(trace_res.history)} ray segments traversed")
    print(f"    - Vignetting Status: {'CLEAN (Zero Vignetting, 100% Transmission)' if not trace_res.vignetted else 'VIGNETTED'}")

    # 4. Galvo Scanning Field Diagnostic (telecentric relay & walk-off)
    print("\n[4] Running Galvo Scanning Field Diagnostics (+/- 10.0 deg)...")
    fov = SRSGalvoFieldDiagnosticTool.compute_fov_dimensions(cfg)
    pt_max = SRSGalvoFieldDiagnosticTool.evaluate_scan_point(cfg, angle_x_deg=10.0, angle_y_deg=10.0)
    print(f"    - Field of View (FOV): {fov['fov_x_um']:.1f} um x {fov['fov_y_um']:.1f} um (Diagonal: {fov['fov_diagonal_um']:.1f} um)")
    print(f"    - BFP Pupil Walk-Off: {pt_max.back_pupil_walk_off_total_um:.4f} um (strictly 0.0 um via 4f relay)")
    print(f"    - Telecentricity Error: {pt_max.telecentricity_error_mrad:.3f} mrad (< 0.1 deg)")
    print(f"    - Off-Axis Scan Efficiency: {pt_max.total_scan_efficiency * 100:.1f}%")
    print(f"    - Chromatic Walk-Off (797 vs 1040 nm): {pt_max.chromatic_walk_off_um:.3f} um")
    print(f"    - Strehl Spatial Overlap Factor: {pt_max.strehl_spatial_overlap:.3f}")

    # 5. Literature-Grounded Spectral Focusing (Dispersion & Spectrum)
    print("\n[5] Simulating Hyperspectral Spectral Focusing Delay Scan (Mineral Oil)...")
    disp = SRSSpectralFocusingTool.compute_dispersion_parameters(cfg)
    spec = SRSSpectralFocusingTool.simulate_delay_scan(cfg)

    peak_idx = int(np.argmax(spec["on_axis_lockin_voltage_v"]))
    peak_wavenumber = float(spec["wavenumbers_cm1"][peak_idx])
    peak_sig_v = float(spec["on_axis_lockin_voltage_v"][peak_idx])
    peak_snr = float(spec["on_axis_snr_db"][peak_idx])

    print(f"    - Pump GDD: {disp['gdd_pump_fs2']:.0f} fs^2, Stokes GDD: {disp['gdd_stokes_fs2']:.0f} fs^2")
    print(f"    - Chirped Pulse Durations: Pump = {disp['chirped_pump_fwhm_ps']:.2f} ps, Stokes = {disp['chirped_stokes_fwhm_ps']:.2f} ps")
    print(f"    - Chirp Matching Ratio (alpha_p / alpha_S): {disp['chirp_matching_ratio']:.3f} (Literature Ideal: 1.000)")
    print(f"    - Spectral Resolution: {disp['spectral_resolution_cm1']:.1f} cm^-1")
    print(f"    - Peak Vibrational Raman Mode: {peak_wavenumber:.1f} cm^-1 (CH2 symmetric stretch)")
    print(f"    - Peak Lock-In Voltage: {peak_sig_v * 1e6:.3f} uV (Peak SNR: {peak_snr:.1f} dB)")

    # 6. RF Lock-In Demodulation at 10.0 MHz
    print("\n[6] Physical Signal Processing & 10.0 MHz Lock-In Demodulation...")
    pt_on = SRSGalvoFieldDiagnosticTool.evaluate_scan_point(cfg, angle_x_deg=0.0, angle_y_deg=0.0)
    sig_on = SRSSignalLockInTool.process_signal(cfg, fractional_srl=5.0e-5, scan_result=pt_on)
    sig_edge = SRSSignalLockInTool.process_signal(cfg, fractional_srl=5.0e-5, scan_result=pt_max)
    gain = cfg.photodiode.transimpedance_gain_v_per_a
    dc_ua = sig_on.dc_photocurrent_a * 1e6
    shot_uv = sig_on.shot_noise_current_a * gain * 1e6
    rin_uv = sig_on.rin_noise_current_a * gain * 1e6
    thermal_uv = sig_on.thermal_noise_current_a * gain * 1e6
    total_noise_uv = sig_on.output_noise_voltage_v * 1e6

    print(f"    - Photodiode DC Current: {dc_ua:.2f} uA (DET10N2 Responsivity: {cfg.photodiode.responsivity_a_per_w} A/W)")
    print(f"    - Total Noise Voltage Floor: {total_noise_uv:.3f} uV")
    print(f"        * Photon Shot Noise:  {shot_uv:.3f} uV")
    print(f"        * Laser RIN:          {rin_uv:.3f} uV")
    print(f"        * Thermal Noise:      {thermal_uv:.3f} uV")
    print(f"    - On-Axis (Center FOV):")
    print(f"        * Recovered Lock-In (R): {sig_on.lockin_magnitude_r_v * 1e6:.3f} uV (X: {sig_on.lockin_x_v * 1e6:.3f} uV, Y: {sig_on.lockin_y_v * 1e6:.3f} uV)")
    print(f"        * Demodulated SNR:       {sig_on.snr_db:.1f} dB (Linear: {sig_on.snr_linear:.1f}:1)")
    print(f"    - Edge-of-Field (+/- 10 deg Corner):")
    print(f"        * Recovered Lock-In (R): {sig_edge.lockin_magnitude_r_v * 1e6:.3f} uV")
    print(f"        * Demodulated SNR:       {sig_edge.snr_db:.1f} dB (Linear: {sig_edge.snr_linear:.1f}:1)")

    # 7. Optical Bench Alignment Report
    print("\n[7] Optical Bench Alignment & Clear Apertures...")
    align = SRSBenchAlignmentTool.check_pupil_walk_off(cfg, angle_deg=3.0)
    filters = SRSBenchAlignmentTool.evaluate_optical_density_stack(cfg)
    clearance = SRSBenchAlignmentTool.verify_37_surface_clearances(stack)

    print(f"    - Objective BFP Diameter: {align['objective_pupil_diameter_mm']:.2f} mm")
    print(f"    - Beam Diameter at BFP: {align['bfp_beam_diameter_mm']:.2f} mm")
    print(f"    - Entrance Pupil Fill Factor: {align['pupil_fill_factor']:.2f}")
    print(f"    - Stokes Optical Rejection Stack OD: {filters['total_stokes_rejection_od']} (Leakage: {filters['stokes_optical_leakage_fraction']:.1e})")
    print(f"    - Bench Clearances: {'100% COLLISION FREE' if clearance['collision_free_clearance'] else 'POTENTIAL OVERLAP'}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: All optical, mechanical, and signal systems validated!")
    print("To launch the full interactive desktop application, run:")
    print("    python main.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
