"""
Adversarial Stress Test Suite for M6 Physics, Polarization, and Gaussian Beam Engine.
Tests edge cases, extreme angles, complex metals, TIR, Jones calculus, waveplates, and q-parameter propagation.
"""
import sys
import os
import math
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.engine.polarization import (
    compute_fresnel_coefficients, compute_sp_basis, rotate_jones_vector, FresnelResult
)
from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, snell_refract_3d
from optics_app.utils.materials import get_complex_refractive_index, get_refractive_index, METALS
from optics_app.engine.gaussian import GaussianBeam, trace_system_gaussian_beam


def run_all_stress_tests():
    results = []

    def record(category, test_name, status, detail, data=None):
        results.append({
            "category": category,
            "test_name": test_name,
            "status": status,  # "PASS", "FAIL", "WARN"
            "detail": detail,
            "data": data or {}
        })
        tag = f"[{status}]"
        print(f"{tag:<7} [{category}] {test_name}: {detail}")

    print("======================================================================")
    print("      M6 ADVERSARIAL STRESS TEST HARNESS — PHYSICS & POLARIZATION     ")
    print("======================================================================")

    # -------------------------------------------------------------------------
    # Category 1: Incidence Angle Extremes (Normal theta_i=0 and Grazing theta_i -> 90)
    # -------------------------------------------------------------------------
    cat = "Incidence Extremes"
    
    # 1.1 Normal incidence
    f_norm = compute_fresnel_coefficients(1.0, 1.5, cos_theta1=1.0)
    r_s_norm = abs(f_norm.r_s)
    r_p_norm = abs(f_norm.r_p)
    t_s_norm = abs(f_norm.t_s)
    t_p_norm = abs(f_norm.t_p)
    R_norm = r_s_norm**2
    T_norm = 1.5 * t_s_norm**2
    
    if abs(r_s_norm - 0.2) < 1e-5 and abs(r_p_norm - 0.2) < 1e-5 and abs(R_norm + T_norm - 1.0) < 1e-5:
        record(cat, "Normal Incidence (0 deg)", "PASS", f"R={R_norm:.6f}, T={T_norm:.6f}, |r_s|=|r_p|=0.2")
    else:
        record(cat, "Normal Incidence (0 deg)", "FAIL", f"R={R_norm:.6f}, T={T_norm:.6f}, |r_s|={r_s_norm}, |r_p|={r_p_norm}")

    # 1.2 Grazing incidence theta_i = 89.9999 deg
    cos_grazing = np.cos(np.radians(89.9999))
    f_graz = compute_fresnel_coefficients(1.0, 1.5, cos_theta1=cos_grazing)
    R_s_graz = abs(f_graz.r_s)**2
    R_p_graz = abs(f_graz.r_p)**2
    
    if abs(R_s_graz - 1.0) < 1e-3 and abs(R_p_graz - 1.0) < 1e-3:
        record(cat, "Grazing Incidence (89.9999 deg)", "PASS", f"R_s={R_s_graz:.6f}, R_p={R_p_graz:.6f}")
    else:
        record(cat, "Grazing Incidence (89.9999 deg)", "FAIL", f"R_s={R_s_graz:.6f}, R_p={R_p_graz:.6f}")

    # 1.3 Exact theta_i = 90 deg (cos_theta1 = 0.0)
    try:
        f_90 = compute_fresnel_coefficients(1.0, 1.5, cos_theta1=0.0)
        record(cat, "Exact 90 deg Incidence (cos=0)", "PASS", f"|r_s|={abs(f_90.r_s):.6f}, |r_p|={abs(f_90.r_p):.6f}")
    except Exception as e:
        record(cat, "Exact 90 deg Incidence (cos=0)", "FAIL", f"Exception raised: {e}")

    # 1.4 Matched indices n1 = n2 = 1.5 at cos=0 (singularity test)
    try:
        f_sing = compute_fresnel_coefficients(1.5, 1.5, cos_theta1=0.0)
        if not (math.isnan(f_sing.r_s.real) or math.isnan(f_sing.r_p.real)):
            record(cat, "Matched Index Grazing Singularity (n1=n2, cos=0)", "PASS", f"r_s={f_sing.r_s}, r_p={f_sing.r_p}")
        else:
            record(cat, "Matched Index Grazing Singularity (n1=n2, cos=0)", "FAIL", f"Returned NaN: r_s={f_sing.r_s}")
    except Exception as e:
        record(cat, "Matched Index Grazing Singularity (n1=n2, cos=0)", "FAIL", f"Exception raised: {e}")

    # -------------------------------------------------------------------------
    # Category 2: Complex Metal Refractive Indices (Au, Al, Ag, Cu)
    # -------------------------------------------------------------------------
    cat = "Complex Metals"
    metal_expected = {
        "AU": (0.18 + 3.0j, 0.930718),
        "AL": (1.097 + 6.79j, 0.916892),
        "AG": (0.05 + 4.27j, 0.989656),
        "CU": (0.25 + 3.4j, 0.930062),
    }

    for name, (expected_n, expected_R) in metal_expected.items():
        c_n = get_complex_refractive_index(name)
        f_metal = compute_fresnel_coefficients(1.0, c_n, cos_theta1=1.0)
        R_calc = abs(f_metal.r_s)**2
        diff = abs(R_calc - expected_R)
        if abs(c_n - expected_n) < 1e-4 and diff < 1e-3:
            record(cat, f"Metal {name} Normal Reflectance", "PASS", f"n={c_n}, R_calc={R_calc:.6f} (exp {expected_R:.6f})")
        else:
            record(cat, f"Metal {name} Normal Reflectance", "FAIL", f"n={c_n} (exp {expected_n}), R_calc={R_calc:.6f} (exp {expected_R:.6f})")

    # Oblique metal reflection (Au at 45 deg)
    c_au = get_complex_refractive_index("GOLD")
    f_au_45 = compute_fresnel_coefficients(1.0, c_au, cos_theta1=np.cos(np.radians(45)))
    R_s_au45 = abs(f_au_45.r_s)**2
    R_p_au45 = abs(f_au_45.r_p)**2
    if R_s_au45 > R_p_au45 and R_s_au45 > 0.9 and R_p_au45 > 0.8:
        record(cat, "Gold Oblique 45 deg Reflection", "PASS", f"R_s={R_s_au45:.6f}, R_p={R_p_au45:.6f} (R_s > R_p verified)")
    else:
        record(cat, "Gold Oblique 45 deg Reflection", "FAIL", f"R_s={R_s_au45:.6f}, R_p={R_p_au45:.6f}")

    # -------------------------------------------------------------------------
    # Category 3: Total Internal Reflection (TIR) & Fresnel Rhomb
    # -------------------------------------------------------------------------
    cat = "TIR & Fresnel Rhomb"

    # 3.1 Critical angle check (n1=1.5 -> n2=1.0)
    sin_critical = 1.0 / 1.5
    theta_critical_deg = np.degrees(np.arcsin(sin_critical))

    f_below = compute_fresnel_coefficients(1.5, 1.0, np.cos(np.radians(41.0)))
    f_above = compute_fresnel_coefficients(1.5, 1.0, np.cos(np.radians(43.0)))

    if not f_below.tir and f_above.tir and abs(abs(f_above.r_s) - 1.0) < 1e-6 and abs(abs(f_above.r_p) - 1.0) < 1e-6:
        record(cat, "TIR Critical Angle Threshold", "PASS", f"Critical={theta_critical_deg:.2f} deg, 41deg tir={f_below.tir}, 43deg tir={f_above.tir}")
    else:
        record(cat, "TIR Critical Angle Threshold", "FAIL", f"41deg tir={f_below.tir}, 43deg tir={f_above.tir}, |r_s|={abs(f_above.r_s)}")

    # 3.2 Fresnel Rhomb phase retardance (angle = 53.3 deg for glass n=1.5)
    f_rhomb = compute_fresnel_coefficients(1.5, 1.0, np.cos(np.radians(53.3)))
    delta_s = np.angle(f_rhomb.r_s)
    delta_p = np.angle(f_rhomb.r_p)
    raw_diff = np.degrees(delta_p - delta_s)
    retardance = abs((raw_diff + 180.0) % 360.0 - 180.0)
    
    if abs(retardance - 45.0) < 1.0:
        record(cat, "Fresnel Rhomb Single Reflection (45 deg retardance)", "PASS", f"Single TIR Retardance = {retardance:.2f} deg (expected ~45 deg)")
    else:
        record(cat, "Fresnel Rhomb Single Reflection (45 deg retardance)", "FAIL", f"Single TIR Retardance = {retardance:.2f} deg")

    # 3.3 Two TIR reflections in series (Fresnel Rhomb total = 90 deg QWP)
    J_in = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rhomb_mat = np.array([
        [f_rhomb.r_s**2, 0.0],
        [0.0, f_rhomb.r_p**2]
    ], dtype=complex)
    J_out_rhomb = rhomb_mat @ J_in
    abs_s = abs(J_out_rhomb[0])
    abs_p = abs(J_out_rhomb[1])
    phase_diff = np.degrees(np.angle(J_out_rhomb[1]) - np.angle(J_out_rhomb[0]))
    phase_diff_wrapped = abs((phase_diff + 180.0) % 360.0 - 180.0)

    if abs(abs_s - abs_p) < 1e-5 and abs(phase_diff_wrapped - 90.0) < 2.0:
        record(cat, "Fresnel Rhomb 2-TIR System (Circular Polarization)", "PASS", f"|Es|={abs_s:.4f}, |Ep|={abs_p:.4f}, Phase Diff={phase_diff_wrapped:.2f} deg")
    else:
        record(cat, "Fresnel Rhomb 2-TIR System (Circular Polarization)", "FAIL", f"|Es|={abs_s:.4f}, |Ep|={abs_p:.4f}, Phase Diff={phase_diff_wrapped:.2f} deg")

    # -------------------------------------------------------------------------
    # Category 4: Jones Calculus, Polarizers, Waveplates, Malus Law & 3-Pol Paradox
    # -------------------------------------------------------------------------
    cat = "Jones Calculus & Polarization"

    def make_linear_polarizer(angle_deg):
        th = np.radians(angle_deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)

    def make_waveplate(retardance_deg, fast_axis_deg):
        th = np.radians(fast_axis_deg)
        ret = np.radians(retardance_deg)
        c, s = np.cos(th), np.sin(th)
        W0 = np.array([[np.exp(-1j * ret / 2.0), 0], [0, np.exp(1j * ret / 2.0)]], dtype=complex)
        R = np.array([[c, -s], [s, c]], dtype=complex)
        return R @ W0 @ R.T

    # 4.1 Malus' Law
    J_horiz = np.array([1.0+0j, 0.0+0j])
    angles = [0, 30, 45, 60, 90]
    malus_passed = True
    for a in angles:
        pol = make_linear_polarizer(a)
        J_out = pol @ J_horiz
        I_out = float(np.sum(np.abs(J_out)**2))
        I_expected = np.cos(np.radians(a))**2
        if abs(I_out - I_expected) > 1e-6:
            malus_passed = False

    if malus_passed:
        record(cat, "Malus' Law across Angles (0, 30, 45, 60, 90 deg)", "PASS", "Transmitted intensity matches cos^2(theta) exactly")
    else:
        record(cat, "Malus' Law across Angles (0, 30, 45, 60, 90 deg)", "FAIL", "Discrepancy in transmitted intensity")

    # 4.2 3-Polarizer Paradox
    pol0 = make_linear_polarizer(0)
    pol45 = make_linear_polarizer(45)
    pol90 = make_linear_polarizer(90)

    J_2pol = pol90 @ (pol0 @ J_horiz)
    I_2pol = float(np.sum(np.abs(J_2pol)**2))

    J_3pol = pol90 @ (pol45 @ (pol0 @ J_horiz))
    I_3pol = float(np.sum(np.abs(J_3pol)**2))

    if abs(I_2pol - 0.0) < 1e-6 and abs(I_3pol - 0.25) < 1e-6:
        record(cat, "3-Polarizer Paradox", "PASS", f"2-pol I={I_2pol:.6f}, 3-pol I={I_3pol:.6f} (25% transmission restored)")
    else:
        record(cat, "3-Polarizer Paradox", "FAIL", f"2-pol I={I_2pol:.6f}, 3-pol I={I_3pol:.6f}")

    # 4.3 Quarter-Wave Plate (QWP)
    qwp45 = make_waveplate(90.0, 45.0)
    J_qwp_out = qwp45 @ J_horiz
    abs_qwp_s = abs(J_qwp_out[0])
    abs_qwp_p = abs(J_qwp_out[1])
    diff_qwp_phase = np.degrees(np.angle(J_qwp_out[1]) - np.angle(J_qwp_out[0]))
    diff_qwp_phase_wrapped = abs((diff_qwp_phase + 180.0) % 360.0 - 180.0)

    if abs(abs_qwp_s - 1/np.sqrt(2)) < 1e-5 and abs(abs_qwp_p - 1/np.sqrt(2)) < 1e-5 and abs(diff_qwp_phase_wrapped - 90.0) < 1e-5:
        record(cat, "Quarter-Wave Plate (QWP @ 45 deg -> Circular)", "PASS", f"|Es|={abs_qwp_s:.4f}, |Ep|={abs_qwp_p:.4f}, phase diff={diff_qwp_phase_wrapped:.2f} deg")
    else:
        record(cat, "Quarter-Wave Plate (QWP @ 45 deg -> Circular)", "FAIL", f"|Es|={abs_qwp_s:.4f}, |Ep|={abs_qwp_p:.4f}, phase diff={diff_qwp_phase_wrapped:.2f} deg")

    # 4.4 Half-Wave Plate (HWP)
    hwp225 = make_waveplate(180.0, 22.5)
    J_hwp_out = hwp225 @ J_horiz
    ratio_hwp = J_hwp_out[1] / J_hwp_out[0]
    if abs(abs(ratio_hwp) - 1.0) < 1e-5 and abs(np.angle(ratio_hwp)) < 1e-4:
        record(cat, "Half-Wave Plate (HWP @ 22.5 deg -> Rotates 45 deg)", "PASS", f"Output ratio Ep/Es = {ratio_hwp:.4f} (Linear @ 45 deg)")
    else:
        record(cat, "Half-Wave Plate (HWP @ 22.5 deg -> Rotates 45 deg)", "FAIL", f"Output ratio Ep/Es = {ratio_hwp:.4f}")

    # 4.5 Full OpticalSystem Ray Trace with Surface Jones Matrix
    stack_pol = SurfaceStack()
    s0 = Surface(name="Source", thickness=10.0)
    s1 = Surface(name="Polarizer 0 deg", thickness=10.0, jones_matrix=pol0)
    s2 = Surface(name="Polarizer 45 deg", thickness=10.0, jones_matrix=pol45)
    s3 = Surface(name="Polarizer 90 deg", thickness=10.0, jones_matrix=pol90)
    s4 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR)
    stack_pol.surfaces = [s0, s1, s2, s3, s4]

    opt_pol = OpticalSystem(stack_pol)
    ray_in = Ray(pos=[0, 0, 0], dir=[0, 0, 1], jones_vector=[1.0, 0.0], intensity=1.0)
    ray_out = opt_pol.trace_ray(ray_in)

    if abs(ray_out.intensity - 0.25) < 1e-5 and not ray_out.vignetted:
        record(cat, "OpticalSystem Trace with 3-Polarizer Jones Surfaces", "PASS", f"Traced Intensity = {ray_out.intensity:.6f} (expected 0.25)")
    else:
        record(cat, "OpticalSystem Trace with 3-Polarizer Jones Surfaces", "FAIL", f"Traced Intensity = {ray_out.intensity:.6f}, vignetted={ray_out.vignetted}")

    # -------------------------------------------------------------------------
    # Category 5: Gaussian Beam q-Parameter Propagation
    # -------------------------------------------------------------------------
    cat = "Gaussian Beam Physics"

    w0 = 0.05
    wl_um = 0.6328
    beam = GaussianBeam(w0=w0, wavelength=wl_um, n=1.0, z=0.0)
    zR_calc = beam.z_R
    zR_exp = (np.pi * 1.0 * (w0**2)) / (wl_um * 1e-3)

    if abs(zR_calc - zR_exp) < 1e-6:
        record(cat, "Rayleigh Range Formula zR", "PASS", f"zR = {zR_calc:.4f} mm (waist w0={w0*1000:.1f} um)")
    else:
        record(cat, "Rayleigh Range Formula zR", "FAIL", f"zR_calc = {zR_calc:.4f}, zR_exp = {zR_exp:.4f}")

    distances = [0.5 * zR_calc, zR_calc, 2.0 * zR_calc, 10.0 * zR_calc]
    g_free_passed = True
    for d in distances:
        b_test = GaussianBeam(w0=w0, wavelength=wl_um, n=1.0, z=0.0)
        b_test.propagate_distance(d)
        w_act = b_test.spot_radius
        w_exp = w0 * np.sqrt(1.0 + (d / zR_calc)**2)
        R_act = b_test.radius_of_curvature
        R_exp = d * (1.0 + (zR_calc / d)**2)
        if abs(w_act - w_exp) > 1e-5 or abs(R_act - R_exp) > 1e-4:
            g_free_passed = False

    if g_free_passed:
        record(cat, "Free-Space q-Parameter Expansion (w(z) & R(z))", "PASS", f"Spot size w(zR)={w0 * np.sqrt(2):.6f} mm, R(zR)={2*zR_calc:.4f} mm")
    else:
        record(cat, "Free-Space q-Parameter Expansion (w(z) & R(z))", "FAIL", "Discrepancy in w(z) or R(z)")

    f_lens = 50.0
    b_lens = GaussianBeam(w0=w0, wavelength=wl_um, n=1.0, z=0.0)
    b_lens.propagate_distance(f_lens)
    b_lens.pass_thin_lens(f_lens)
    w0_prime_exp = (wl_um * 1e-3 * f_lens) / (np.pi * w0)
    b_lens.propagate_distance(f_lens)
    w_at_focus = b_lens.spot_radius
    R_at_focus = b_lens.radius_of_curvature

    if abs(w_at_focus - w0_prime_exp) < 1e-5 and abs(R_at_focus) > 1e6:
        record(cat, "Gaussian Beam Thin Lens Focusing (f=50mm)", "PASS", f"Focused waist w0' = {w_at_focus*1000:.2f} um (expected {w0_prime_exp*1000:.2f} um), R -> inf")
    else:
        record(cat, "Gaussian Beam Thin Lens Focusing (f=50mm)", "FAIL", f"w_focus={w_at_focus*1000:.2f} um, R_focus={R_at_focus}")

    b_mirr = GaussianBeam(w0=w0, wavelength=wl_um, n=1.0, z=0.0)
    b_mirr.propagate_distance(100.0)
    w_before_mirr = b_mirr.spot_radius
    b_mirr.reflect_mirror(radius=-100.0)
    w_after_mirr = b_mirr.spot_radius

    if abs(w_before_mirr - w_after_mirr) < 1e-6:
        record(cat, "Curved Mirror Reflection (Spot Continuity)", "PASS", f"Spot size preserved across mirror interface: w={w_after_mirr*1000:.2f} um")
    else:
        record(cat, "Curved Mirror Reflection (Spot Continuity)", "FAIL", f"w_before={w_before_mirr}, w_after={w_after_mirr}")

    stack_gb = SurfaceStack()
    g0 = Surface(name="Source", thickness=20.0)
    g1 = Surface(name="Lens Front", radius=50.0, thickness=5.0, glass="BK7")
    g2 = Surface(name="Lens Back", radius=-50.0, thickness=30.0, glass="AIR")
    g3 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR)
    stack_gb.surfaces = [g0, g1, g2, g3]

    gb_trace_res = trace_system_gaussian_beam(stack_gb, w0=w0, wavelength=wl_um, source_z=0.0)
    
    if len(gb_trace_res.nodes) == 4 and gb_trace_res.final_w > 0:
        record(cat, "Multi-Surface Gaussian Beam Trace System", "PASS", f"Traced 4 nodes, final spot size w = {gb_trace_res.final_w*1000:.2f} um")
    else:
        record(cat, "Multi-Surface Gaussian Beam Trace System", "FAIL", f"Nodes = {len(gb_trace_res.nodes)}, final_w = {gb_trace_res.final_w}")

    # -------------------------------------------------------------------------
    # Category 6: Edge-case / Boundary Robustness
    # -------------------------------------------------------------------------
    cat = "Robustness Edge Cases"

    try:
        b_tiny = GaussianBeam(w0=1e-6, wavelength=wl_um)
        b_tiny.propagate_distance(1.0)
        w_tiny = b_tiny.spot_radius
        record(cat, "Extreme Small Gaussian Waist (w0=1 nm)", "PASS", f"w(1mm) = {w_tiny:.4f} mm")
    except Exception as e:
        record(cat, "Extreme Small Gaussian Waist (w0=1 nm)", "FAIL", f"Exception: {e}")

    try:
        b_huge = GaussianBeam(w0=1000.0, wavelength=wl_um)
        b_huge.propagate_distance(1000.0)
        w_huge = b_huge.spot_radius
        record(cat, "Extreme Large Gaussian Waist (w0=1 m)", "PASS", f"w(1m) = {w_huge:.4f} mm")
    except Exception as e:
        record(cat, "Extreme Large Gaussian Waist (w0=1 m)", "FAIL", f"Exception: {e}")

    print("\n======================================================================")
    passed_cnt = sum(1 for r in results if r["status"] == "PASS")
    total_cnt = len(results)
    print(f" Stress Test Results: {passed_cnt}/{total_cnt} tests passed.")
    print("======================================================================")

    return passed_cnt == total_cnt, results


if __name__ == "__main__":
    success, res = run_all_stress_tests()
    sys.exit(0 if success else 1)
