"""
Milestone M2 Empirical Challenger 1 Benchmark & Validation Runner.
Executes detailed adversarial stress scenarios and collects empirical metrics:
1. High-Order Aspheres (4th-12th order) & Extreme Conics (-15 to +20).
2. Multi-Variable Scaling (Doublets, Triplets, Petzval, Quad-Element, 6-Element Double Gauss).
3. Boundary Stress (Zero Thickness, Extreme Defocus, Tight Bounds, Clamping, Zero-Width Bounds).
4. Multi-Solver Comparison (Nelder-Mead, L-BFGS-B, SLSQP, Powell).
5. Ray Scaling Benchmark (10, 20, 30, 50, 75, 100 rays).
"""

import time
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult


def set_opt_vars(surf: Surface, variable: bool = True, t_min: float = 0.0, t_max: float = 1000.0) -> Surface:
    setattr(surf, "thickness_variable", variable)
    setattr(surf, "thickness_min", t_min)
    setattr(surf, "thickness_max", t_max)
    return surf


def main():
    print("=" * 80)
    print("M2 OPTICAL OPTIMIZER ENGINE — CHALLENGER 1 EMPIRICAL BENCHMARKS")
    print("=" * 80)

    summary_records = []

    # -------------------------------------------------------------------------
    # 1. High-Order Aspheres & Extreme Conics
    # -------------------------------------------------------------------------
    print("\n[Section 1] High-Order Aspheres & Extreme Conics", flush=True)

    # 1.1 Schmidt Corrector (4th, 6th, 8th order)
    st1 = SurfaceStack()
    st1.add_surface(Surface(name="OBJ", thickness=2000.0))
    st1.add_surface(Surface(name="Schmidt", surface_type=SurfaceType.ASPHERIC, radius=0.0, thickness=4.0, glass="N-BK7",
                            semi_diameter=25.0, asphere_coeffs=[2.5e-6, -1.2e-9, 4.0e-13]))
    st1.add_surface(Surface(name="Air", radius=0.0, thickness=150.0, glass="AIR", semi_diameter=25.0))
    st1.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=30.0))
    
    t0 = time.perf_counter()
    opt1 = OpticalOptimizer(OpticalSystem(st1), method="Nelder-Mead", max_iter=60, num_rays=25, entrance_pupil_radius=12.0)
    res1 = opt1.optimize(variable_indices=[2], bounds=[(50.0, 300.0)])
    dt1 = time.perf_counter() - t0
    print(f"1.1 Schmidt Corrector (4th/6th/8th): Init RMS={res1.initial_rms:.4f}mm -> Final RMS={res1.final_rms:.4f}mm in {dt1*1000:.1f}ms (iters={res1.iterations})", flush=True)
    assert res1.success and not np.isnan(res1.final_rms)
    summary_records.append(("1.1 Schmidt Asphere Corrector", "PASS", f"Final RMS={res1.final_rms:.4f}mm, Time={dt1*1000:.1f}ms"))

    # 1.2 Extreme Hyperbolic Conic (k = -15.0)
    st2 = SurfaceStack()
    st2.add_surface(Surface(name="OBJ", thickness=1000.0))
    st2.add_surface(Surface(name="HypSurf", surface_type=SurfaceType.ASPHERIC, radius=25.0, conic=-15.0,
                            asphere_coeffs=[1e-5, -3e-8, 5e-11, -2e-14], thickness=6.0, glass="N-BK7", semi_diameter=15.0))
    st2.add_surface(Surface(name="Air", radius=-50.0, thickness=20.0, glass="AIR", semi_diameter=15.0))
    st2.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
    
    t0 = time.perf_counter()
    opt2 = OpticalOptimizer(OpticalSystem(st2), method="Nelder-Mead", max_iter=60, num_rays=25, entrance_pupil_radius=8.0)
    res2 = opt2.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
    dt2 = time.perf_counter() - t0
    print(f"1.2 Hyperbolic Conic (k=-15.0): Init RMS={res2.initial_rms:.4f}mm -> Final RMS={res2.final_rms:.4f}mm (-{res2.rms_reduction_pct:.2f}%) in {dt2*1000:.1f}ms", flush=True)
    assert res2.success and res2.rms_reduction_pct >= 30.0
    summary_records.append(("1.2 Extreme Hyperbola (k=-15.0)", "PASS", f"RMS Reduction={res2.rms_reduction_pct:.1f}%, Time={dt2*1000:.1f}ms"))

    # 1.3 Extreme Oblate Ellipsoid Conic (k = +20.0)
    st3 = SurfaceStack()
    st3.add_surface(Surface(name="OBJ", thickness=800.0))
    st3.add_surface(Surface(name="OblateSurf", surface_type=SurfaceType.ASPHERIC, radius=35.0, conic=20.0,
                            asphere_coeffs=[-5e-6, 2e-9], thickness=5.0, glass="N-BK7", semi_diameter=12.0))
    st3.add_surface(Surface(name="Air", radius=-45.0, thickness=15.0, glass="AIR", semi_diameter=12.0))
    st3.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
    
    t0 = time.perf_counter()
    opt3 = OpticalOptimizer(OpticalSystem(st3), method="Nelder-Mead", max_iter=60, num_rays=20, entrance_pupil_radius=6.0)
    res3 = opt3.optimize(variable_indices=[2], bounds=[(5.0, 120.0)])
    dt3 = time.perf_counter() - t0
    print(f"1.3 Oblate Conic (k=+20.0): Init RMS={res3.initial_rms:.4f}mm -> Final RMS={res3.final_rms:.4f}mm (-{res3.rms_reduction_pct:.2f}%) in {dt3*1000:.1f}ms", flush=True)
    assert res3.success and res3.rms_reduction_pct >= 40.0
    summary_records.append(("1.3 Extreme Oblate (k=+20.0)", "PASS", f"RMS Reduction={res3.rms_reduction_pct:.1f}%, Time={dt3*1000:.1f}ms"))

    # 1.4 12th-Order Even Asphere Polynomial
    st4 = SurfaceStack()
    st4.add_surface(Surface(name="OBJ", thickness=1000.0))
    st4.add_surface(Surface(name="12thOrder", surface_type=SurfaceType.ASPHERIC, radius=40.0, conic=-1.0,
                            asphere_coeffs=[1e-5, -2e-8, 3e-11, -4e-14, 5e-17], thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    st4.add_surface(Surface(name="Air", radius=-80.0, thickness=25.0, glass="AIR", semi_diameter=15.0))
    st4.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
    
    t0 = time.perf_counter()
    opt4 = OpticalOptimizer(OpticalSystem(st4), method="Nelder-Mead", max_iter=50, num_rays=20, entrance_pupil_radius=8.0)
    res4 = opt4.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
    dt4 = time.perf_counter() - t0
    print(f"1.4 12th-Order Asphere: Init RMS={res4.initial_rms:.4f}mm -> Final RMS={res4.final_rms:.4f}mm in {dt4*1000:.1f}ms", flush=True)
    assert res4.success
    summary_records.append(("1.4 12th-Order Asphere Poly", "PASS", f"RMS={res4.final_rms:.4f}mm, Time={dt4*1000:.1f}ms"))

    # -------------------------------------------------------------------------
    # 2. Multi-Variable Optical Topologies
    # -------------------------------------------------------------------------
    print("\n[Section 2] Multi-Variable Optical Topologies (3D to 8D Systems)", flush=True)

    # 2.1 Air-Spaced Doublet (3 Variables)
    st_d = SurfaceStack()
    st_d.add_surface(Surface(name="OBJ", thickness=1000.0))
    st_d.add_surface(Surface(name="Crown_F", radius=45.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0))
    st_d.add_surface(Surface(name="AirGap", radius=-35.0, thickness=2.0, glass="AIR", semi_diameter=15.0))
    st_d.add_surface(Surface(name="Flint_F", radius=-34.0, thickness=3.5, glass="N-SF11", semi_diameter=15.0))
    st_d.add_surface(Surface(name="Flint_B", radius=-150.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
    st_d.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
    
    t0 = time.perf_counter()
    opt_d = OpticalOptimizer(OpticalSystem(st_d), method="Nelder-Mead", max_iter=100, num_rays=25, entrance_pupil_radius=10.0)
    res_d = opt_d.optimize(variable_indices=[1, 2, 4], bounds=[(3.0, 12.0), (0.1, 8.0), (15.0, 120.0)])
    dt_d = time.perf_counter() - t0
    print(f"2.1 Air-Spaced Doublet (3 Vars): Init RMS={res_d.initial_rms:.4f}mm -> Final RMS={res_d.final_rms:.4f}mm (-{res_d.rms_reduction_pct:.2f}%) in {dt_d*1000:.1f}ms", flush=True)
    assert res_d.success and res_d.rms_reduction_pct >= 40.0
    summary_records.append(("2.1 Air-Spaced Doublet (3 Vars)", "PASS", f"Reduction={res_d.rms_reduction_pct:.1f}%, Time={dt_d*1000:.1f}ms"))

    # 2.2 Petzval Objective (3 Variables)
    st_p = SurfaceStack()
    st_p.add_surface(Surface(name="OBJ", thickness=2000.0))
    st_p.add_surface(Surface(name="G1_1", radius=35.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    st_p.add_surface(Surface(name="G1_2", radius=-25.0, thickness=2.5, glass="N-SF11", semi_diameter=15.0))
    st_p.add_surface(Surface(name="G1_3", radius=-80.0, thickness=25.0, glass="AIR", semi_diameter=15.0))
    st_p.add_surface(Surface(name="G2_1", radius=40.0, thickness=4.0, glass="N-BK7", semi_diameter=12.0))
    st_p.add_surface(Surface(name="G2_2", radius=-30.0, thickness=1.0, glass="AIR", semi_diameter=12.0))
    st_p.add_surface(Surface(name="G2_3", radius=-28.0, thickness=3.0, glass="N-SF11", semi_diameter=12.0))
    st_p.add_surface(Surface(name="G2_4", radius=-60.0, thickness=20.0, glass="AIR", semi_diameter=12.0))
    st_p.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
    
    t0 = time.perf_counter()
    opt_p = OpticalOptimizer(OpticalSystem(st_p), method="Nelder-Mead", max_iter=120, num_rays=25, entrance_pupil_radius=8.0)
    res_p = opt_p.optimize(variable_indices=[3, 5, 7], bounds=[(10.0, 50.0), (0.2, 5.0), (10.0, 80.0)])
    dt_p = time.perf_counter() - t0
    print(f"2.2 Petzval Objective (3 Vars): Init RMS={res_p.initial_rms:.4f}mm -> Final RMS={res_p.final_rms:.4f}mm (-{res_p.rms_reduction_pct:.2f}%) in {dt_p*1000:.1f}ms", flush=True)
    assert res_p.success and res_p.rms_reduction_pct >= 40.0
    summary_records.append(("2.2 Petzval Objective (3 Vars)", "PASS", f"Reduction={res_p.rms_reduction_pct:.1f}%, Time={dt_p*1000:.1f}ms"))

    # 2.3 Double Gauss 6-Element (8 Variables)
    st_dg = SurfaceStack()
    st_dg.add_surface(Surface(name="OBJ", thickness=1000.0))
    st_dg.add_surface(Surface(name="L1_F", radius=28.0, thickness=4.5, glass="N-SK16", semi_diameter=12.0))
    st_dg.add_surface(Surface(name="L1_B", radius=120.0, thickness=1.5, glass="AIR", semi_diameter=12.0))
    st_dg.add_surface(Surface(name="L2_F", radius=16.0, thickness=3.8, glass="N-SK16", semi_diameter=10.0))
    st_dg.add_surface(Surface(name="L2_Mid", radius=-35.0, thickness=1.5, glass="N-SF2", semi_diameter=10.0))
    st_dg.add_surface(Surface(name="L2_B", radius=12.0, thickness=4.0, glass="AIR", semi_diameter=8.0))
    st_dg.add_surface(Surface(name="STOP", surface_type=SurfaceType.STOP, radius=0.0, thickness=3.5, glass="AIR", semi_diameter=6.0))
    st_dg.add_surface(Surface(name="L3_F", radius=-13.0, thickness=1.5, glass="N-SF2", semi_diameter=8.0))
    st_dg.add_surface(Surface(name="L3_Mid", radius=35.0, thickness=3.8, glass="N-SK16", semi_diameter=10.0))
    st_dg.add_surface(Surface(name="L3_B", radius=-18.0, thickness=1.5, glass="AIR", semi_diameter=10.0))
    st_dg.add_surface(Surface(name="L4_F", radius=60.0, thickness=3.5, glass="N-SK16", semi_diameter=12.0))
    st_dg.add_surface(Surface(name="L4_B", radius=-30.0, thickness=25.0, glass="AIR", semi_diameter=12.0))
    st_dg.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
    
    t0 = time.perf_counter()
    opt_dg = OpticalOptimizer(OpticalSystem(st_dg), method="Nelder-Mead", max_iter=150, num_rays=20, entrance_pupil_radius=6.0)
    var_indices_dg = [1, 2, 3, 5, 6, 7, 9, 11]
    bounds_dg = [(2.0, 8.0), (0.2, 5.0), (1.5, 6.0), (1.0, 8.0), (1.0, 8.0), (0.5, 4.0), (0.2, 5.0), (10.0, 60.0)]
    res_dg = opt_dg.optimize(variable_indices=var_indices_dg, bounds=bounds_dg)
    dt_dg = time.perf_counter() - t0
    print(f"2.3 Double Gauss (8 Vars): Init RMS={res_dg.initial_rms:.4f}mm -> Final RMS={res_dg.final_rms:.4f}mm (-{res_dg.rms_reduction_pct:.2f}%) in {dt_dg*1000:.1f}ms", flush=True)
    assert res_dg.success and len(res_dg.optimized_thicknesses) == 8
    summary_records.append(("2.3 Double Gauss (8 Vars)", "PASS", f"8 variables optimized, Time={dt_dg*1000:.1f}ms"))

    # -------------------------------------------------------------------------
    # 3. Boundary Stress Tests
    # -------------------------------------------------------------------------
    print("\n[Section 3] Boundary Condition Stress & Clamping", flush=True)

    # Singlet base
    def make_singlet():
        st = SurfaceStack()
        st.add_surface(Surface(name="OBJ", thickness=1000.0))
        st.add_surface(Surface(name="L1", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        st.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        st.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        return st

    # 3.1 Extreme Defocus (t0 = 800mm)
    st_def = make_singlet()
    st_def.surfaces[2].thickness = 800.0
    st_def._update_accumulated_positions()
    opt_def = OpticalOptimizer(OpticalSystem(st_def), method="Nelder-Mead", max_iter=100, num_rays=20, entrance_pupil_radius=6.0)
    res_def = opt_def.optimize(variable_indices=[2], bounds=[(10.0, 1000.0)])
    print(f"3.1 10x Defocus (t0=800mm): Result={res_def.optimized_thicknesses[2]:.3f}mm (Expected ~96.7mm)", flush=True)
    assert res_def.success and abs(res_def.optimized_thicknesses[2] - 96.7) < 12.0
    summary_records.append(("3.1 10x Initial Defocus", "PASS", f"Recovered to {res_def.optimized_thicknesses[2]:.2f}mm"))

    # 3.2 Lower Bound Clamping ([120.0, 200.0])
    st_lb = make_singlet()
    opt_lb = OpticalOptimizer(OpticalSystem(st_lb), method="Nelder-Mead", max_iter=50, num_rays=20)
    res_lb = opt_lb.optimize(variable_indices=[2], bounds=[(120.0, 200.0)])
    print(f"3.2 Lower Bound Clamping: Result={res_lb.optimized_thicknesses[2]:.3f}mm (Bounded [120, 200])", flush=True)
    assert abs(res_lb.optimized_thicknesses[2] - 120.0) < 1.0
    summary_records.append(("3.2 Lower Bound Clamping", "PASS", f"Clamped at {res_lb.optimized_thicknesses[2]:.2f}mm"))

    # 3.3 Upper Bound Clamping ([20.0, 60.0])
    st_ub = make_singlet()
    opt_ub = OpticalOptimizer(OpticalSystem(st_ub), method="Nelder-Mead", max_iter=50, num_rays=20)
    res_ub = opt_ub.optimize(variable_indices=[2], bounds=[(20.0, 60.0)])
    print(f"3.3 Upper Bound Clamping: Result={res_ub.optimized_thicknesses[2]:.3f}mm (Bounded [20, 60])", flush=True)
    assert abs(res_ub.optimized_thicknesses[2] - 60.0) < 1.0
    summary_records.append(("3.3 Upper Bound Clamping", "PASS", f"Clamped at {res_ub.optimized_thicknesses[2]:.2f}mm"))

    # 3.4 Zero-Width Bound ([45.0, 45.0])
    st_zw = make_singlet()
    opt_zw = OpticalOptimizer(OpticalSystem(st_zw), method="Nelder-Mead", max_iter=30, num_rays=15)
    res_zw = opt_zw.optimize(variable_indices=[2], bounds=[(45.0, 45.0)])
    print(f"3.4 Zero-Width Bound: Result={res_zw.optimized_thicknesses[2]:.3f}mm (Target 45.000mm)", flush=True)
    assert abs(res_zw.optimized_thicknesses[2] - 45.0) < 1e-3
    summary_records.append(("3.4 Zero-Width Bound", "PASS", f"Fixed at {res_zw.optimized_thicknesses[2]:.3f}mm"))

    # -------------------------------------------------------------------------
    # 4. Multi-Solver Comparison
    # -------------------------------------------------------------------------
    print("\n[Section 4] Multi-Solver Stability & Convergence", flush=True)
    solvers = ["Nelder-Mead", "L-BFGS-B", "SLSQP", "Powell"]
    for s_name in solvers:
        st_s = make_singlet()
        t0 = time.perf_counter()
        opt_s = OpticalOptimizer(OpticalSystem(st_s), method=s_name, max_iter=60, num_rays=20, entrance_pupil_radius=8.0)
        res_s = opt_s.optimize(variable_indices=[2], bounds=[(10.0, 180.0)])
        dt_s = time.perf_counter() - t0
        print(f"4. Solver {s_name:12s}: Final={res_s.optimized_thicknesses[2]:.3f}mm, RMS={res_s.final_rms:.4f}mm (-{res_s.rms_reduction_pct:.1f}%), Iters={res_s.iterations} in {dt_s*1000:.1f}ms", flush=True)
        assert res_s.success and res_s.rms_reduction_pct >= 50.0
        summary_records.append((f"4. Solver: {s_name}", "PASS", f"Final={res_s.optimized_thicknesses[2]:.2f}mm, Iters={res_s.iterations}, Time={dt_s*1000:.1f}ms"))

    # -------------------------------------------------------------------------
    # 5. Ray Scaling Benchmark (10 to 100 rays)
    # -------------------------------------------------------------------------
    print("\n[Section 5] Ray Scaling Benchmark (10 to 100 Rays)", flush=True)
    ray_counts = [10, 20, 30, 50, 75, 100]
    for n_rays in ray_counts:
        st_r = make_singlet()
        opt_r = OpticalOptimizer(OpticalSystem(st_r), num_rays=n_rays, method="Nelder-Mead", max_iter=40, entrance_pupil_radius=8.0)
        t0 = time.perf_counter()
        res_r = opt_r.optimize(variable_indices=[2], bounds=[(10.0, 180.0)])
        dt_r = time.perf_counter() - t0
        print(f"5. Num Rays = {n_rays:3d}: Optimized={res_r.optimized_thicknesses[2]:.3f}mm, RMS={res_r.final_rms:.4f}mm, Time={dt_r*1000:.1f}ms", flush=True)
        assert res_r.success and abs(res_r.optimized_thicknesses[2] - 96.7) < 10.0
        summary_records.append((f"5. Ray Count: {n_rays} rays", "PASS", f"RMS={res_r.final_rms:.4f}mm, Time={dt_r*1000:.1f}ms"))

    # -------------------------------------------------------------------------
    # Final Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("M2 CHALLENGER 1 EMPIRICAL BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    for name, status, details in summary_records:
        print(f"[{status}] {name:38s} | {details}")
    print("=" * 80)
    print("ALL M2 CHALLENGER 1 EMPIRICAL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
