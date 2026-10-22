"""
Standalone empirical validation and benchmark runner for M2 OpticalOptimizer.
Executes all challenge scenarios and outputs structured empirical metrics.
"""

import sys
import time
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.analysis.spot import SpotDiagram
from optics_app.utils.materials import GlassCatalog, get_refractive_index


def add_var_attrs(surf: Surface, var: bool = True, t_min: float = 0.0, t_max: float = 1000.0) -> Surface:
    setattr(surf, "thickness_variable", var)
    setattr(surf, "thickness_min", t_min)
    setattr(surf, "thickness_max", t_max)
    return surf


def run_benchmark():
    print("=" * 70)
    print("M2 OPTICAL OPTIMIZER ENGINE — EMPIRICAL CHALLENGE BENCHMARK")
    print("=" * 70)

    results = []

    # ------------------------------------------------------------------
    # Challenge 1: 3D Fold Mirror & Coordinate Frame Tracking
    # ------------------------------------------------------------------
    print("\n--- Challenge 1: 3D Geometry & Fold Mirrors ---")

    # 1.1 45-deg Fold Mirror Newtonian Focus
    stack1 = SurfaceStack()
    stack1.add_surface(Surface(name="OBJ", thickness=1000.0))
    stack1.add_surface(Surface(name="LensFront", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
    stack1.add_surface(Surface(name="LensBack", radius=0.0, thickness=40.0, glass="AIR", semi_diameter=20.0))
    s_fold = Surface(name="FoldMirror", radius=0.0, thickness=20.0, glass="MIRROR", tilt_x=45.0, semi_diameter=25.0)
    add_var_attrs(s_fold, var=True, t_min=10.0, t_max=120.0)
    stack1.add_surface(s_fold)
    stack1.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

    t0 = time.perf_counter()
    opt1 = OpticalOptimizer(OpticalSystem(stack1), method="Nelder-Mead", max_iter=80, num_rays=25, entrance_pupil_radius=8.0)
    res1 = opt1.optimize(variable_indices=[3], bounds=[(10.0, 120.0)])
    dt1 = time.perf_counter() - t0

    det_pos1 = res1.optimized_stack.surfaces[4].global_origin
    print(f"1.1 Fold Mirror Focus: Initial RMS={res1.initial_rms:.4f}mm -> Final RMS={res1.final_rms:.4f}mm (-{res1.rms_reduction_pct:.2f}%) in {dt1*1000:.1f}ms", flush=True)
    print(f"    Optimized Thickness={res1.optimized_thicknesses[3]:.3f}mm, Det Global Origin=[{det_pos1[0]:.2f}, {det_pos1[1]:.2f}, {det_pos1[2]:.2f}]", flush=True)
    assert res1.success and res1.rms_reduction_pct >= 70.0, "1.1 Failed"
    results.append(("1.1 45-deg Fold Mirror", "PASS", f"RMS: {res1.initial_rms:.4f} -> {res1.final_rms:.4f} (-{res1.rms_reduction_pct:.1f}%)"))

    # 1.2 Compound 3D Tilts & Decenters
    stack2 = SurfaceStack()
    stack2.add_surface(Surface(name="OBJ", thickness=500.0))
    stack2.add_surface(Surface(name="TiltedLensFront", radius=60.0, thickness=6.0, glass="N-BK7", 
                               decenter_x=2.0, decenter_y=-1.5, tilt_x=5.0, tilt_y=-3.0, semi_diameter=20.0))
    s2_back = Surface(name="TiltedLensBack", radius=-100.0, thickness=20.0, glass="AIR", semi_diameter=20.0)
    add_var_attrs(s2_back, var=True, t_min=10.0, t_max=180.0)
    stack2.add_surface(s2_back)
    s2_fold = Surface(name="CompoundMirror", radius=0.0, thickness=30.0, glass="MIRROR", 
                      tilt_x=30.0, tilt_y=20.0, tilt_z=10.0, semi_diameter=25.0)
    add_var_attrs(s2_fold, var=True, t_min=10.0, t_max=150.0)
    stack2.add_surface(s2_fold)
    stack2.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=40.0))

    t0 = time.perf_counter()
    opt2 = OpticalOptimizer(OpticalSystem(stack2), method="Nelder-Mead", max_iter=60, num_rays=20, entrance_pupil_radius=6.0)
    res2 = opt2.optimize(variable_indices=[2, 3], bounds=[(10.0, 180.0), (10.0, 150.0)])
    dt2 = time.perf_counter() - t0
    print(f"1.2 Compound 3D Tilts/Decenters: Initial RMS={res2.initial_rms:.4f}mm -> Final RMS={res2.final_rms:.4f}mm (-{res2.rms_reduction_pct:.2f}%) in {dt2*1000:.1f}ms", flush=True)
    assert res2.success and res2.rms_reduction_pct >= 40.0, "1.2 Failed"
    results.append(("1.2 Compound 3D Tilts/Decenters", "PASS", f"RMS: {res2.initial_rms:.4f} -> {res2.final_rms:.4f} (-{res2.rms_reduction_pct:.1f}%)"))

    # 1.3 Dual Fold Mirror (Periscope)
    stack3 = SurfaceStack()
    stack3.add_surface(Surface(name="OBJ", thickness=1000.0))
    stack3.add_surface(Surface(name="Lens", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
    s3_d1 = Surface(name="Air1", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=20.0)
    add_var_attrs(s3_d1, var=True, t_min=10.0, t_max=50.0)
    stack3.add_surface(s3_d1)
    s3_m1 = Surface(name="Mirror1", radius=0.0, thickness=30.0, glass="MIRROR", tilt_y=45.0, semi_diameter=20.0)
    add_var_attrs(s3_m1, var=True, t_min=10.0, t_max=50.0)
    stack3.add_surface(s3_m1)
    s3_m2 = Surface(name="Mirror2", radius=0.0, thickness=20.0, glass="MIRROR", tilt_y=-45.0, semi_diameter=20.0)
    add_var_attrs(s3_m2, var=True, t_min=10.0, t_max=80.0)
    stack3.add_surface(s3_m2)
    stack3.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=30.0))

    t0 = time.perf_counter()
    opt3 = OpticalOptimizer(OpticalSystem(stack3), method="Nelder-Mead", max_iter=70, num_rays=20, entrance_pupil_radius=8.0)
    res3 = opt3.optimize(variable_indices=[2, 3, 4], bounds=[(10.0, 50.0), (10.0, 50.0), (10.0, 80.0)])
    dt3 = time.perf_counter() - t0
    total_opt_path = res3.optimized_thicknesses[2] + res3.optimized_thicknesses[3] + res3.optimized_thicknesses[4]
    print(f"1.3 Dual Fold Periscope: Initial RMS={res3.initial_rms:.4f}mm -> Final RMS={res3.final_rms:.4f}mm (-{res3.rms_reduction_pct:.2f}%) in {dt3*1000:.1f}ms", flush=True)
    print(f"    Total Track Length = {total_opt_path:.3f}mm (Target ~96.7mm)", flush=True)
    assert res3.success and res3.rms_reduction_pct >= 50.0, "1.3 Failed"
    results.append(("1.3 Dual Fold Periscope", "PASS", f"RMS: {res3.initial_rms:.4f} -> {res3.final_rms:.4f} (-{res3.rms_reduction_pct:.1f}%)"))

    # ------------------------------------------------------------------
    # Challenge 2: Chromatic Optimization Across Dispersion Curves
    # ------------------------------------------------------------------
    print("\n--- Challenge 2: Chromatic Optimization & Dispersion Curves ---")

    # 2.1 Multi-wavelength Achromat (F, d, C lines)
    stack4 = SurfaceStack()
    stack4.add_surface(Surface(name="OBJ", thickness=2000.0))
    stack4.add_surface(Surface(name="Crown1", radius=62.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0))
    s4_gap = Surface(name="AirGap", radius=-43.0, thickness=0.5, glass="AIR", semi_diameter=15.0)
    add_var_attrs(s4_gap, var=True, t_min=0.1, t_max=5.0)
    stack4.add_surface(s4_gap)
    stack4.add_surface(Surface(name="Flint1", radius=-43.0, thickness=3.5, glass="N-SF11", semi_diameter=15.0))
    s4_bfl = Surface(name="Flint2", radius=-140.0, thickness=40.0, glass="AIR", semi_diameter=15.0)
    add_var_attrs(s4_bfl, var=True, t_min=30.0, t_max=150.0)
    stack4.add_surface(s4_bfl)
    stack4.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

    wl_fdc = [0.48613, 0.58756, 0.65627]
    t0 = time.perf_counter()
    opt4 = OpticalOptimizer(OpticalSystem(stack4), wavelengths=wl_fdc, method="Nelder-Mead", max_iter=70, num_rays=25, entrance_pupil_radius=10.0)
    res4 = opt4.optimize(variable_indices=[2, 4], bounds=[(0.1, 5.0), (30.0, 150.0)])
    dt4 = time.perf_counter() - t0
    print(f"2.1 Achromatic Doublet (F, d, C): Initial RMS={res4.initial_rms:.4f}mm -> Final RMS={res4.final_rms:.4f}mm (-{res4.rms_reduction_pct:.2f}%) in {dt4*1000:.1f}ms", flush=True)
    print(f"    Optimized AirGap={res4.optimized_thicknesses[2]:.3f}mm, BFL={res4.optimized_thicknesses[4]:.3f}mm", flush=True)
    assert res4.success and res4.rms_reduction_pct >= 50.0, "2.1 Failed"
    results.append(("2.1 Achromatic Doublet (F, d, C)", "PASS", f"RMS: {res4.initial_rms:.4f} -> {res4.final_rms:.4f} (-{res4.rms_reduction_pct:.1f}%)"))

    # 2.2 Broadband Multi-Glass Catalog
    glasses = ["FUSED_SILICA", "F2", "N-SK16", "N-SF6"]
    for g in glasses:
        st = SurfaceStack()
        st.add_surface(Surface(name="OBJ", thickness=500.0))
        st.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass=g, semi_diameter=12.0))
        sb = Surface(name="L2", radius=-80.0, thickness=20.0, glass="AIR", semi_diameter=12.0)
        add_var_attrs(sb, var=True, t_min=10.0, t_max=120.0)
        st.add_surface(sb)
        st.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        opt_g = OpticalOptimizer(OpticalSystem(st), wavelengths=[0.500, 0.600], method="Nelder-Mead", max_iter=40, num_rays=15, entrance_pupil_radius=6.0)
        res_g = opt_g.optimize(variable_indices=[2])
        print(f"2.2 Glass {g:12s}: Initial RMS={res_g.initial_rms:.4f}mm -> Final RMS={res_g.final_rms:.4f}mm (-{res_g.rms_reduction_pct:.2f}%)", flush=True)
        assert res_g.success and res_g.rms_reduction_pct >= 40.0, f"2.2 Failed on {g}"
    results.append(("2.2 Multi-Glass Catalog (4 Glasses)", "PASS", "Fused Silica, F2, N-SK16, N-SF6 converged"))

    # ------------------------------------------------------------------
    # Challenge 3: Degenerate Cases & Robustness
    # ------------------------------------------------------------------
    print("\n--- Challenge 3: Degenerate Optical Cases & Robustness ---", flush=True)

    # 3.1 100% Vignetted Rays (Annular central obscuration blocking all rays)
    stack_vig = SurfaceStack()
    stack_vig.add_surface(Surface(name="OBJ", thickness=100.0))
    s_vig1 = Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=25.0,
                     aperture_type="Annular", aperture_r_inner=20.0, aperture_r_outer=25.0)
    stack_vig.add_surface(s_vig1)
    stack_vig.add_surface(Surface(name="L2", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=25.0))
    stack_vig.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
    opt_vig = OpticalOptimizer(OpticalSystem(stack_vig), entrance_pupil_radius=10.0)
    merit_vig = opt_vig.merit_function(np.array([50.0]), [2])
    print(f"3.1 100% Vignetted Penalty = {merit_vig:.1f} (Expected 100000.0)", flush=True)
    assert merit_vig == 1e5, "3.1 Failed"
    results.append(("3.1 100% Vignetted Rays", "PASS", f"Penalty={merit_vig:.0f}"))

    # 3.2 High Vignetting Partial Penalty (60 of 61 rays blocked)
    stack_pvig = SurfaceStack()
    stack_pvig.add_surface(Surface(name="OBJ", thickness=100.0))
    stack_pvig.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=0.001))
    stack_pvig.add_surface(Surface(name="L2", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=0.001))
    stack_pvig.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=0.001))
    opt_pvig = OpticalOptimizer(OpticalSystem(stack_pvig), entrance_pupil_radius=10.0, penalty_weight=1000.0)
    merit_pvig = opt_pvig.merit_function(np.array([50.0]), [2])
    print(f"3.2 High Vignetting Penalty = {merit_pvig:.1f} (Expected 983.6)", flush=True)
    assert abs(merit_pvig - 983.6065) < 0.1, "3.2 Failed"
    results.append(("3.2 High Vignetting Partial Penalty", "PASS", f"Penalty={merit_pvig:.1f}"))

    # 3.3 Single Ray / Needle Beam
    stack_sr = SurfaceStack()
    stack_sr.add_surface(Surface(name="OBJ", thickness=100.0))
    stack_sr.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    s_sr2 = Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0)
    add_var_attrs(s_sr2, var=True, t_min=10.0, t_max=150.0)
    stack_sr.add_surface(s_sr2)
    stack_sr.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
    opt_sr = OpticalOptimizer(OpticalSystem(stack_sr), num_rays=1, entrance_pupil_radius=5.0)
    merit_sr = opt_sr.merit_function(np.array([30.0]), [2])
    print(f"3.3 Single Ray Sampling Merit = {merit_sr:.6f} (No ZeroDivisionError)", flush=True)
    assert not np.isnan(merit_sr) and not np.isinf(merit_sr) and merit_sr > 0.0, "3.3 Failed"
    results.append(("3.3 Single Ray Sampling", "PASS", f"Merit={merit_sr:.4f} without ZeroDivisionError"))

    # 3.4 Extreme Pupil Radii (0.0, 1e-12, 5000.0, -10.0)
    pupils = [0.0, 1e-12, 5000.0, -10.0]
    for p in pupils:
        opt_p = OpticalOptimizer(OpticalSystem(stack_sr), entrance_pupil_radius=p, num_rays=15)
        m_p = opt_p.merit_function(np.array([40.0]), [2])
        print(f"3.4 Pupil Radius {p:10.2e} -> Merit = {m_p:.4f}", flush=True)
        assert not np.isnan(m_p) and not np.isinf(m_p), f"3.4 Failed for pupil={p}"
    results.append(("3.4 Extreme Pupil Radii", "PASS", "Handled 0.0, 1e-12, 5000.0, -10.0 gracefully"))

    # 3.5 Negative Parameter Barrier Penalty
    neg_merit = opt_sr.merit_function(np.array([-25.0]), [2])
    print(f"3.5 Negative Parameter Barrier = {neg_merit:.1f} (>1e5)", flush=True)
    assert neg_merit >= 1e5, "3.5 Failed"
    results.append(("3.5 Negative Parameter Barrier", "PASS", f"Penalty={neg_merit:.0f} >= 1e5"))

    # 3.6 Parameter Bounds Clamping
    opt_clamp = OpticalOptimizer(OpticalSystem(stack_sr), method="Nelder-Mead", max_iter=50, num_rays=20)
    res_clamp = opt_clamp.optimize(variable_indices=[2], bounds=[(10.0, 35.0)])
    print(f"3.6 Clamped Optimization: Result={res_clamp.optimized_thicknesses[2]:.4f}mm (Bounded in [10.0, 35.0])", flush=True)
    assert 10.0 <= res_clamp.optimized_thicknesses[2] <= 35.0 + 1e-5, "3.6 Failed"
    results.append(("3.6 Parameter Bounds Clamping", "PASS", f"Clamped to {res_clamp.optimized_thicknesses[2]:.2f}mm"))

    # 3.7 Multi-Solver Validation
    for solver in ["Nelder-Mead", "L-BFGS-B", "Powell", "SLSQP"]:
        opt_s = OpticalOptimizer(OpticalSystem(stack_sr), method=solver, max_iter=30, num_rays=15, entrance_pupil_radius=8.0)
        res_s = opt_s.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
        print(f"3.7 Solver {solver:12s}: Success={res_s.success}, Final={res_s.optimized_thicknesses[2]:.3f}mm in {res_s.iterations} iters", flush=True)
        assert res_s.success, f"3.7 Failed on solver {solver}"
    results.append(("3.7 Multi-Solver Validation", "PASS", "Nelder-Mead, L-BFGS-B, Powell, SLSQP all passed"))

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EMIRICAL CHALLENGE RESULTS SUMMARY")
    print("=" * 70)
    for name, status, details in results:
        print(f"[{status}] {name:38s} | {details}")
    print("=" * 70)
    print("ALL EMPIRICAL CHALLENGES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
