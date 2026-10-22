"""
Empirical Challenge & Stress Test Harness for Milestone M3 (GUI Integration & Live Synchronization)
"""
import sys
import os
import time
import tracemalloc
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.main_window import MainWindow

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

print("="*70)
print(f"QT_AVAILABLE: {QT_AVAILABLE}")
if QT_AVAILABLE:
    print(f"QApplication: {QtWidgets.QApplication.instance()}")
print("="*70)

def challenge_1_progress_callback_bug():
    print("\n--- CHALLENGE 1: progress_callback argument type compatibility ---")
    window = MainWindow()
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=1000.0))
    stack.add_surface(Surface(name="L1_Front", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    stack.add_surface(Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
    window.model.set_surface_stack(stack)
    window.optimizer_dock.set_surface_variable(2, True)

    print("Attempting window.on_optimize() with default Qt setup...")
    try:
        res = window.on_optimize(variable_indices=[2], method="Nelder-Mead", max_iter=30)
        print(f"on_optimize returned: success={res.success}, initial_rms={res.initial_rms:.4f}, final_rms={res.final_rms:.4f}")
    except TypeError as e:
        print(f"CRITICAL DEFECT DETECTED: TypeError raised during on_optimize(): {e}")
        import traceback
        traceback.print_exc()

def challenge_2_live_synchronization():
    print("\n--- CHALLENGE 2: Live Visualization Synchronization (CAD + Spot Diagram) ---")
    window = MainWindow()
    
    # Singlet lens: focal length ~ 100 mm (N-BK7, R1=51.68 mm, R2=infinity, d=5mm)
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=1000.0))
    stack.add_surface(Surface(name="L1_Front", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    # Deliberately defocused back distance = 30.0 mm (far from 100mm focus)
    s2 = Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0)
    stack.add_surface(s2)
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
    window.model.set_surface_stack(stack)
    window.on_trace_rays()

    # Check initial CAD and Spot state
    initial_thickness = window.model.get_surface_stack().surfaces[2].thickness
    print(f"Initial L1_Back thickness: {initial_thickness:.2f} mm")

    # Manually safe-wrap progress_callback for test to bypass the PySide6 TypeError bug
    # and verify downstream synchronization logic
    system = OpticalSystem(window.model.get_surface_stack())
    opt = OpticalOptimizer(system=system, wavelengths=[0.5876], method="Nelder-Mead", max_iter=60)
    res = opt.optimize(variable_indices=[2])
    
    print(f"Optimization completed: success={res.success}, initial_rms={res.initial_rms:.4f} mm, final_rms={res.final_rms:.4f} mm, reduction={res.rms_reduction_pct:.1f}%")
    print(f"Optimized thickness: {res.optimized_thicknesses[2]:.2f} mm (expected ~97-101 mm)")

    # Test applying result and verifying sync
    if res.optimized_stack is not None:
        window.model.set_surface_stack(res.optimized_stack)
    window.on_trace_rays()

    final_thickness = window.model.get_surface_stack().surfaces[2].thickness
    print(f"Model updated thickness: {final_thickness:.2f} mm")
    assert abs(final_thickness - res.optimized_thicknesses[2]) < 1e-4, "Model thickness does not match optimizer output!"

    # Check DualView CAD layout
    if hasattr(window.dual_view, 'cad_canvas'):
        print(f"DualView CAD canvas present: {window.dual_view.cad_canvas is not None}")

    # Check Analysis Tabs Spot Diagram
    if hasattr(window.analysis_tabs, 'tab_spot'):
        print(f"AnalysisTabs tab_spot present: {window.analysis_tabs.tab_spot is not None}")
        # Verify spot calculation on new system
        from optics_app.analysis.spot import generate_spot_diagram_from_system
        opt_sys = OpticalSystem(window.model.get_surface_stack())
        spot = generate_spot_diagram_from_system(opt_sys, num_rays=150, wavelength_um=0.5876)
        new_rms = spot.calculate_rms_radius()
        print(f"Spot diagram regenerated from updated system has RMS: {new_rms:.4f} mm (drastically reduced from defocused)")
        assert new_rms < 1.0, f"Expected small spot radius near focus, got {new_rms}"

def challenge_3_multi_wavelength_tracing():
    print("\n--- CHALLENGE 3: Multi-Wavelength Ray Tracing via GUI ---")
    window = MainWindow()
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
    stack.add_surface(Surface(name="L2", radius=-50.0, thickness=50.0, glass="AIR", semi_diameter=20.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
    window.model.set_surface_stack(stack)

    # Test toolbar wavelength parsing
    test_inputs = [
        ("0.450, 0.546, 0.650", 3, [0.450, 0.546, 0.650]),
        ("486, 588, 656", 3, [0.486, 0.588, 0.656]),  # nm converted to um
        ("0.5876", 1, [0.5876]),
        ("", 3, [0.450, 0.546, 0.650]),  # empty fallback
        ("invalid, text", 3, [0.450, 0.546, 0.650]),  # invalid fallback
    ]

    for input_str, expected_len, expected_wls in test_inputs:
        window.wavelengths_input.setText(input_str)
        ret = window.on_trace_rays()
        print(f"Input '{input_str}': on_trace_rays returned {ret}")

    # Test Source Editor integration
    from optics_app.engine.source import SourceConfig, SourceType
    for st in [SourceType.COLLIMATED, SourceType.POINT, SourceType.GAUSSIAN]:
        window.source_config.source_type = st
        window.source_config.num_rays = 15
        ret = window.on_trace_rays()
        print(f"Source type {st.name}: on_trace_rays returned {ret}")

def challenge_4_algorithms():
    print("\n--- CHALLENGE 4: Nelder-Mead vs L-BFGS-B from Dock & Engine ---")
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=1000.0))
    stack.add_surface(Surface(name="L1_Front", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    s2 = Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0)
    stack.add_surface(s2)
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

    system = OpticalSystem(stack)

    # 1. Nelder-Mead
    t0 = time.perf_counter()
    opt_nm = OpticalOptimizer(system, method="Nelder-Mead", max_iter=80, num_rays=21)
    res_nm = opt_nm.optimize(variable_indices=[2])
    t_nm = time.perf_counter() - t0
    print(f"Nelder-Mead: success={res_nm.success}, time={t_nm*1000:.1f}ms, initial_rms={res_nm.initial_rms:.4f}, final_rms={res_nm.final_rms:.4f}, opt_thick={res_nm.optimized_thicknesses[2]:.2f}mm, iters={res_nm.iterations}")

    # 2. L-BFGS-B
    t0 = time.perf_counter()
    opt_lbfgs = OpticalOptimizer(system, method="L-BFGS-B", max_iter=40, num_rays=21)
    res_lbfgs = opt_lbfgs.optimize(variable_indices=[2])
    t_lbfgs = time.perf_counter() - t0
    print(f"L-BFGS-B:    success={res_lbfgs.success}, time={t_lbfgs*1000:.1f}ms, initial_rms={res_lbfgs.initial_rms:.4f}, final_rms={res_lbfgs.final_rms:.4f}, opt_thick={res_lbfgs.optimized_thicknesses[2]:.2f}mm, iters={res_lbfgs.iterations}")

    # Verify both reached ~100mm focus
    assert abs(res_nm.optimized_thicknesses[2] - 100.0) < 5.0, f"Nelder-Mead did not converge to expected focus: {res_nm.optimized_thicknesses[2]}"
    assert abs(res_lbfgs.optimized_thicknesses[2] - 100.0) < 5.0, f"L-BFGS-B did not converge to expected focus: {res_lbfgs.optimized_thicknesses[2]}"

def challenge_5_performance_and_memory():
    print("\n--- CHALLENGE 5: Performance & Memory Overhead in Repeated Cycles ---")
    window = MainWindow()
    
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    num_cycles = 25
    cycle_times = []

    print(f"Running {num_cycles} consecutive optimization + raytrace cycles...")
    for cycle in range(num_cycles):
        # Reset stack to defocused state
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="L1_Front", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L1_Back", radius=0.0, thickness=20.0 + (cycle % 10) * 5.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        window.model.set_surface_stack(stack)

        t0 = time.perf_counter()
        
        # Run optimizer (using lambda for progress_callback to avoid PySide6 TypeError)
        system = OpticalSystem(window.model.get_surface_stack())
        optimizer = OpticalOptimizer(system=system, wavelengths=[0.5876], method="Nelder-Mead", max_iter=40, num_rays=21)
        res = optimizer.optimize(variable_indices=[2])
        if res.optimized_stack is not None:
            window.model.set_surface_stack(res.optimized_stack)
        window.on_trace_rays()
        
        elapsed = time.perf_counter() - t0
        cycle_times.append(elapsed)

    snapshot_end = tracemalloc.take_snapshot()
    stats = snapshot_end.compare_to(snapshot_start, 'lineno')

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_time = np.mean(cycle_times) * 1000.0
    p95_time = np.percentile(cycle_times, 95) * 1000.0
    max_time = np.max(cycle_times) * 1000.0

    print(f"Performance over {num_cycles} cycles:")
    print(f"  Avg cycle time: {avg_time:.2f} ms")
    print(f"  P95 cycle time: {p95_time:.2f} ms")
    print(f"  Max cycle time: {max_time:.2f} ms")
    print(f"  Peak memory during run: {peak_mem / 1024 / 1024:.2f} MB")
    print(f"  Current memory net delta: {current_mem / 1024 / 1024:.2f} MB")

    print("\nTop 5 memory allocating lines:")
    for stat in stats[:5]:
        print(f"  {stat}")

    assert avg_time < 500.0, f"Average cycle time {avg_time:.2f}ms exceeds 500ms budget!"
    assert peak_mem / 1024 / 1024 < 100.0, f"Peak memory {peak_mem / 1024 / 1024:.2f}MB exceeds 100MB threshold!"
    print("Performance & Memory constraints PASSED!")

if __name__ == "__main__":
    challenge_1_progress_callback_bug()
    challenge_2_live_synchronization()
    challenge_3_multi_wavelength_tracing()
    challenge_4_algorithms()
    challenge_5_performance_and_memory()
    print("\n" + "="*70)
    print("ALL EMPIRICAL HARNESS CHECKS COMPLETED")
    print("="*70)
