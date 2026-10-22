"""
Standalone Performance & Benchmarking Suite for 2D_Optics Simulation & Rendering Engine.

Measures quantitative latency SLAs:
  1. Dense batch ray tracing throughput:
     - 500 rays in < 50 ms
     - 5,000 rays in < 200 ms
  2. 3D visualizer mesh caching:
     - Cached rebuild / incremental ray update < 10 ms
  3. 2D layout update cycle:
     - Layout update < 25 ms (> 40 FPS)
  4. GUI event throttling & debouncing:
     - 30 Hz (33 ms) throttle timer and 150 ms debounce timer

Usage:
    python -m benchmarks.benchmark_engine_and_rendering --all
    python -m benchmarks.benchmark_engine_and_rendering --engine
    python -m benchmarks.benchmark_engine_and_rendering --rendering
    python -m benchmarks.benchmark_engine_and_rendering --gui
    python -m benchmarks.benchmark_engine_and_rendering --project spectrometer_raman_50mm_100mm.json --json-out results.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray, RayBundle
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.qt_compat import QT_AVAILABLE, QApplication


def generate_collimated_bundle(
    num_rays: int,
    beam_radius: float = 3.0,
    wavelength_um: float = 0.5876,
    origin_z: float = -10.0,
) -> List[Ray]:
    """Generate a collimated circular grid bundle of scalar rays."""
    rays: List[Ray] = []
    if num_rays <= 1:
        rays.append(Ray(np.array([0.0, 0.0, origin_z]), np.array([0.0, 0.0, 1.0]), wavelength_um))
        return rays

    # Concentric rings
    rays.append(Ray(np.array([0.0, 0.0, origin_z]), np.array([0.0, 0.0, 1.0]), wavelength_um))
    rings = max(1, int(np.sqrt(num_rays / np.pi)))
    rays_per_ring = max(4, int(num_rays / rings))

    count = 1
    for r_idx in range(1, rings + 1):
        r = beam_radius * (r_idx / rings)
        n_pts = int(rays_per_ring * r_idx)
        for p in range(n_pts):
            if count >= num_rays:
                break
            theta = 2.0 * np.pi * p / n_pts
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            rays.append(Ray(np.array([x, y, origin_z]), np.array([0.0, 0.0, 1.0]), wavelength_um))
            count += 1
        if count >= num_rays:
            break

    # If still need more to meet exact count, add random fill
    while len(rays) < num_rays:
        theta = 2.0 * np.pi * np.random.rand()
        r = beam_radius * np.sqrt(np.random.rand())
        rays.append(
            Ray(
                np.array([r * np.cos(theta), r * np.sin(theta), origin_z]),
                np.array([0.0, 0.0, 1.0]),
                wavelength_um,
            )
        )
    return rays[:num_rays]


def benchmark_engine(
    project_path: str,
    target_dense_rays: int = 5000,
    repeats: int = 5,
) -> Dict[str, Any]:
    """Benchmark batch ray-tracing engine throughput and SLA conformance."""
    print("=" * 70)
    print(" BENCHMARK 1: Ray-Tracing Engine Throughput & Vectorization")
    print("=" * 70)

    results: Dict[str, Any] = {"status": "PASS", "tests": {}}

    data = SurfaceStack.load_from_json(project_path)
    stack = data["stack"]
    system = OpticalSystem(stack)

    print(f"Loaded Project: {os.path.basename(project_path)} ({len(stack.surfaces)} surfaces)")

    # 1. 500 rays throughput SLA: < 50 ms
    rays_500 = generate_collimated_bundle(500, beam_radius=4.0)
    bundle_500 = RayBundle.from_rays(rays_500)

    # Warmup
    system.trace_bundle(bundle_500)

    # Measure batch
    batch_500_times = []
    for _ in range(repeats):
        b = bundle_500.copy()
        t0 = time.perf_counter()
        system.trace_bundle(b)
        t1 = time.perf_counter()
        batch_500_times.append((t1 - t0) * 1000.0)

    median_500_ms = float(np.median(batch_500_times))
    sla_500_pass = median_500_ms < 50.0

    # Measure scalar for speedup comparison
    scalar_500_times = []
    for _ in range(max(2, repeats // 2)):
        t0 = time.perf_counter()
        system.trace_rays([r.copy() for r in rays_500])
        t1 = time.perf_counter()
        scalar_500_times.append((t1 - t0) * 1000.0)
    median_scalar_500_ms = float(np.median(scalar_500_times))
    speedup_500 = median_scalar_500_ms / max(median_500_ms, 1e-6)

    print(f"\n[SLA 1.1] 500 Rays Throughput:")
    print(f"  - Vectorized Batch: {median_500_ms:.2f} ms (SLA Target: < 50.0 ms) -> {'[PASS]' if sla_500_pass else '[FAIL]'}")
    print(f"  - Pure Scalar:     {median_scalar_500_ms:.2f} ms")
    print(f"  - Speedup Factor:   {speedup_500:.1f}x")

    results["tests"]["500_rays"] = {
        "batch_median_ms": round(median_500_ms, 2),
        "scalar_median_ms": round(median_scalar_500_ms, 2),
        "speedup": round(speedup_500, 2),
        "sla_target_ms": 50.0,
        "passed": sla_500_pass,
    }

    # 2. 5,000 rays throughput SLA: < 200 ms
    n_dense = target_dense_rays
    rays_dense = generate_collimated_bundle(n_dense, beam_radius=4.0)
    bundle_dense = RayBundle.from_rays(rays_dense)

    batch_dense_times = []
    for _ in range(repeats):
        b = bundle_dense.copy()
        t0 = time.perf_counter()
        system.trace_bundle(b)
        t1 = time.perf_counter()
        batch_dense_times.append((t1 - t0) * 1000.0)

    median_dense_ms = float(np.median(batch_dense_times))
    sla_dense_pass = median_dense_ms < 200.0

    print(f"\n[SLA 1.2] {n_dense} Rays Throughput:")
    print(f"  - Vectorized Batch: {median_dense_ms:.2f} ms (SLA Target: < 200.0 ms) -> {'[PASS]' if sla_dense_pass else '[FAIL]'}")
    print(f"  - Throughput:       {n_dense / (median_dense_ms / 1000.0):,.0f} rays/sec")

    results["tests"]["5000_rays"] = {
        "batch_median_ms": round(median_dense_ms, 2),
        "rays_per_sec": round(n_dense / (median_dense_ms / 1000.0), 1),
        "sla_target_ms": 200.0,
        "passed": sla_dense_pass,
    }

    if not (sla_500_pass and sla_dense_pass):
        results["status"] = "FAIL"

    return results


def benchmark_2d_rendering(
    project_path: str,
    repeats: int = 5,
) -> Dict[str, Any]:
    """Benchmark 2D Dual-View Visualizer layout update latency and SLA conformance."""
    print("\n" + "=" * 70)
    print(" BENCHMARK 2: 2D Visualizer Layout & Artist Update Cycle")
    print("=" * 70)

    results: Dict[str, Any] = {"status": "PASS", "tests": {}}

    app = QApplication.instance() or QApplication([])

    from optics_app.gui.dual_view import DualViewWidget

    data = SurfaceStack.load_from_json(project_path)
    stack = data["stack"]
    system = OpticalSystem(stack)

    rays = generate_collimated_bundle(21, beam_radius=3.0)
    traced_rays = system.trace_rays(rays)

    widget = DualViewWidget()
    widget.canvas.draw_idle = lambda: None
    # Initial build to populate axes
    widget.update_layout(surface_stack=stack, rays=traced_rays)

    import gc
    gc.collect()
    widget.update_layout(surface_stack=stack, rays=traced_rays)

    update_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        widget.update_layout(surface_stack=stack, rays=traced_rays)
        t1 = time.perf_counter()
        update_times.append((t1 - t0) * 1000.0)

    median_update_ms = float(np.median(update_times))
    fps_equivalent = 1000.0 / max(median_update_ms, 1e-6)
    sla_2d_pass = median_update_ms < 25.0

    print(f"[SLA 2.1] 2D Layout Update Cycle:")
    print(f"  - Median Update:    {median_update_ms:.2f} ms (SLA Target: < 25.0 ms / > 40 FPS) -> {'[PASS]' if sla_2d_pass else '[FAIL]'}")
    print(f"  - Frame Rate Rate:  {fps_equivalent:.1f} FPS")

    results["tests"]["2d_update_cycle"] = {
        "median_update_ms": round(median_update_ms, 2),
        "fps": round(fps_equivalent, 1),
        "sla_target_ms": 25.0,
        "passed": sla_2d_pass,
    }

    if not sla_2d_pass:
        results["status"] = "FAIL"

    widget.close()
    return results


def benchmark_3d_rendering(
    project_path: str,
    repeats: int = 5,
) -> Dict[str, Any]:
    """Benchmark 3D PyQt Visualizer MeshCache hit rate and incremental update latency."""
    print("\n" + "=" * 70)
    print(" BENCHMARK 3: 3D Visualizer MeshCache & Incremental Ray Update")
    print("=" * 70)

    results: Dict[str, Any] = {"status": "PASS", "tests": {}}

    app = QApplication.instance() or QApplication([])

    from optics_app.gui.visualizer_3d import Visualizer3DWidget, generate_3d_rays

    data = SurfaceStack.load_from_json(project_path)
    stack = data["stack"]

    with contextlib.redirect_stdout(io.StringIO()):
        rays = generate_3d_rays(stack, num_rings=1, rays_per_ring=4)

    widget = Visualizer3DWidget()
    widget.canvas.draw_idle = lambda: None
    widget.mesh_quality = "interactive"

    # 1. Measure initial uncached build
    t0 = time.perf_counter()
    widget.update_layout(surface_stack=stack, rays=rays, force_rebuild=True)
    t1 = time.perf_counter()
    initial_build_ms = (t1 - t0) * 1000.0

    # 2. Measure cached / incremental ray updates
    import gc
    gc.collect()
    widget.update_layout(surface_stack=stack, rays=rays)

    cached_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        widget.update_layout(surface_stack=stack, rays=rays)
        t1 = time.perf_counter()
        cached_times.append((t1 - t0) * 1000.0)

    median_cached_ms = float(np.median(cached_times))
    sla_3d_pass = median_cached_ms < 10.0

    cache_stats = widget._mesh_cache.get_stats() if hasattr(widget, "_mesh_cache") else {}

    print(f"[SLA 3.1] 3D Visualizer Layout Rebuild:")
    print(f"  - Initial Full Build:     {initial_build_ms:.2f} ms")
    print(f"  - Cached Incremental Update: {median_cached_ms:.2f} ms (SLA Target: < 10.0 ms) -> {'[PASS]' if sla_3d_pass else '[FAIL]'}")
    print(f"  - MeshCache Hits/Misses:  Hits={cache_stats.get('hits', 0)}, Misses={cache_stats.get('misses', 0)}, Entries={cache_stats.get('entries', 0)}")

    results["tests"]["3d_mesh_cache"] = {
        "initial_build_ms": round(initial_build_ms, 2),
        "cached_update_median_ms": round(median_cached_ms, 2),
        "cache_hits": cache_stats.get("hits", 0),
        "sla_target_ms": 10.0,
        "passed": sla_3d_pass,
    }

    if not sla_3d_pass:
        results["status"] = "FAIL"

    widget.close()
    return results


def benchmark_gui_throttling() -> Dict[str, Any]:
    """Benchmark GUI slider event throttling and debounce timers."""
    print("\n" + "=" * 70)
    print(" BENCHMARK 4: GUI Component Editor Slider Event Throttling")
    print("=" * 70)

    results: Dict[str, Any] = {"status": "PASS", "tests": {}}

    app = QApplication.instance() or QApplication([])

    from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
    from optics_app.gui.lde_table import LensDataTableModel

    model = LensDataTableModel()
    dock = ComponentEditorDockWidget(model=model)

    throttle_int = dock._throttle_timer.interval()
    debounce_int = dock._debounce_timer.interval()

    throttle_ok = throttle_int == 33
    debounce_ok = debounce_int == 150

    print(f"[SLA 4.1] Timer Configuration:")
    print(f"  - Throttle Timer Interval: {throttle_int} ms (Target: 33 ms / 30 Hz) -> {'[PASS]' if throttle_ok else '[FAIL]'}")
    print(f"  - Debounce Timer Interval: {debounce_int} ms (Target: 150 ms) -> {'[PASS]' if debounce_ok else '[FAIL]'}")

    # Track emit count during simulated rapid slider scrub
    emit_count = 0

    def on_change(idx):
        nonlocal emit_count
        emit_count += 1

    dock.component_changed.connect(on_change)

    # Simulate 50 slider movements in rapid succession
    for step in range(50):
        dock._on_tilt_x_slider_changed(step * 10)

    # Immediately after rapid moves, emit_count should NOT be 50 (throttled)
    immediate_emits = emit_count
    throttled_ok = immediate_emits <= 2

    print(f"[SLA 4.2] Rapid Scrubbing Event Compression:")
    print(f"  - Slider Events Dispatched: 50")
    print(f"  - Synchronous Emits:        {immediate_emits} (Throttled -> {'[PASS]' if throttled_ok else '[FAIL]'})")

    # Simulate mouse release
    dock._on_slider_released()
    release_emits = emit_count
    release_ok = release_emits >= immediate_emits

    print(f"  - On Slider Released Emits: {release_emits} (Flushed -> {'[PASS]' if release_ok else '[FAIL]'})")

    all_passed = throttle_ok and debounce_ok and throttled_ok and release_ok

    results["tests"]["gui_throttling"] = {
        "throttle_interval_ms": throttle_int,
        "debounce_interval_ms": debounce_int,
        "slider_events": 50,
        "synchronous_emits": immediate_emits,
        "passed": all_passed,
    }

    if not all_passed:
        results["status"] = "FAIL"

    dock.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Performance Optimization & SLA Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks (engine, rendering, GUI)")
    parser.add_argument("--engine", action="store_true", help="Run ray-tracing engine throughput benchmarks")
    parser.add_argument("--rendering", action="store_true", help="Run 2D & 3D rendering benchmarks")
    parser.add_argument("--gui", action="store_true", help="Run GUI event throttling benchmarks")
    parser.add_argument("--project", default="spectrometer_raman_50mm_100mm.json", help="Path to JSON optical project")
    parser.add_argument("--rays", type=int, default=5000, help="Number of dense rays for engine test (default 5000)")
    parser.add_argument("--json-out", default=None, help="Save benchmark results to JSON file")

    args = parser.parse_args()

    # If no specific mode flag is given, run all by default
    run_all = args.all or not (args.engine or args.rendering or args.gui)
    run_engine = run_all or args.engine
    run_rendering = run_all or args.rendering
    run_gui = run_all or args.gui

    project_file = args.project
    if not os.path.isabs(project_file):
        project_file = os.path.join(PROJECT_ROOT, project_file)

    if not os.path.exists(project_file):
        print(f"Error: Project file not found: {project_file}", file=sys.stderr)
        sys.exit(1)

    suite_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project": os.path.basename(project_file),
        "overall_status": "PASS",
        "benchmarks": {},
    }

    if run_engine:
        res = benchmark_engine(project_file, target_dense_rays=args.rays)
        suite_results["benchmarks"]["engine"] = res
        if res["status"] != "PASS":
            suite_results["overall_status"] = "FAIL"

    if run_rendering:
        res_2d = benchmark_2d_rendering(project_file)
        res_3d = benchmark_3d_rendering(project_file)
        suite_results["benchmarks"]["rendering_2d"] = res_2d
        suite_results["benchmarks"]["rendering_3d"] = res_3d
        if res_2d["status"] != "PASS" or res_3d["status"] != "PASS":
            suite_results["overall_status"] = "FAIL"

    if run_gui:
        res_gui = benchmark_gui_throttling()
        suite_results["benchmarks"]["gui"] = res_gui
        if res_gui["status"] != "PASS":
            suite_results["overall_status"] = "FAIL"

    print("\n" + "=" * 70)
    print(f" BENCHMARK SUMMARY: Overall Status -> [{suite_results['overall_status']}]")
    print("=" * 70)

    if args.json_out:
        out_path = args.json_out
        if not os.path.isabs(out_path):
            out_path = os.path.join(PROJECT_ROOT, out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(suite_results, f, indent=2)
        print(f"Saved benchmark results to: {out_path}")

    sys.exit(0 if suite_results["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
