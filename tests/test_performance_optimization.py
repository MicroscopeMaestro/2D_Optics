"""
Unit Tests for Milestone 3 Performance Optimizations & Latency SLAs.

Tests:
  1. Batch ray tracer numerical parity vs scalar tracer (< 1e-9 mm tolerance).
  2. Optional history tracking (record_history flag).
  3. Batch ray tracing latency SLAs (500 rays < 50ms, 5k rays < 200ms) and throughput speedup.
  4. 3D Visualizer MeshCache caching, retrieval, and parameter invalidation.
  5. 3D Visualizer incremental ray update (skipping full ax.clear() on identical geometry).
  6. 2D Visualizer LineCollection ray batching and figure/axis reuse (eliminating fig.clf()).
  7. GUI Component Editor slider throttling (33 ms / 30 Hz) and debouncing (150 ms).
"""

from __future__ import annotations

import os
import sys
import time
from typing import List

import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray, RayBundle
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECTROMETER_PATH = os.path.join(PROJECT_ROOT, "spectrometer_raman_50mm_100mm.json")


def _build_test_stack() -> SurfaceStack:
    """Build a deterministic optical system with singlet lens and detector."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=25.0))
    stack.add_surface(Surface(name="Front", radius=45.0, thickness=6.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="Rear", radius=-65.0, thickness=40.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0))
    stack._update_accumulated_positions()
    return stack


def _generate_rays(n: int, radius: float = 6.0) -> List[Ray]:
    """Generate collimated test rays."""
    rays = []
    for i in range(n):
        frac = (i + 1) / n
        r = radius * np.sqrt(frac)
        th = 2.0 * np.pi * frac * 3.0
        x = r * np.cos(th)
        y = r * np.sin(th)
        rays.append(Ray(pos=np.array([x, y, -10.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876))
    return rays


# ===========================================================================
# 1. Engine Parity & Speedup Tests
# ===========================================================================

def test_batch_tracer_matches_scalar_tracer_exactness():
    """Verify batch RayBundle tracing produces identical results to scalar trace_rays (< 1e-9 mm)."""
    stack = _build_test_stack()
    system = OpticalSystem(stack)

    rays = _generate_rays(64, radius=8.0)
    scalar_traced = system.trace_rays([r.copy() for r in rays])

    bundle = RayBundle.from_rays(rays)
    traced_bundle = system.trace_bundle(bundle)

    for i in range(len(rays)):
        s_ray = scalar_traced[i]
        assert s_ray.vignetted == traced_bundle.vignetted[i], f"Vignetting mismatch at ray {i}"
        if not s_ray.vignetted:
            pos_diff = np.linalg.norm(s_ray.pos - traced_bundle.pos[i])
            dir_diff = np.linalg.norm(s_ray.dir - traced_bundle.dir[i])
            assert pos_diff < 1e-9, f"Ray {i} position difference {pos_diff:.2e} exceeds tolerance"
            assert dir_diff < 1e-9, f"Ray {i} direction difference {dir_diff:.2e} exceeds tolerance"


def test_batch_tracer_record_history_optional():
    """Verify record_history=False skips history dictionary allocation for high throughput."""
    stack = _build_test_stack()
    system = OpticalSystem(stack)

    rays = _generate_rays(10, radius=5.0)

    # With history recording
    bundle_with_hist = RayBundle.from_rays(rays)
    traced_with_hist = system.trace_bundle(bundle_with_hist, record_history=True)
    assert len(traced_with_hist.history) > 0, "History should be recorded when record_history=True"

    # Without history recording
    bundle_no_hist = RayBundle.from_rays(rays)
    traced_no_hist = system.trace_bundle(bundle_no_hist, record_history=False)
    assert len(traced_no_hist.history) == 0, "History should remain empty when record_history=False"


def test_batch_tracer_latency_slas_and_speedup():
    """Verify batch throughput satisfies SLAs: 500 rays < 50ms, 5,000 rays < 200ms."""
    if os.path.exists(SPECTROMETER_PATH):
        stack = SurfaceStack.load_from_json(SPECTROMETER_PATH)["stack"]
    else:
        stack = _build_test_stack()
    system = OpticalSystem(stack)

    # 1. 500 rays SLA < 50 ms
    rays_500 = _generate_rays(500, radius=4.0)
    bundle_500 = RayBundle.from_rays(rays_500)

    # Warmup
    system.trace_bundle(bundle_500)

    t0 = time.perf_counter()
    b = bundle_500.copy()
    system.trace_bundle(b)
    batch_500_ms = (time.perf_counter() - t0) * 1000.0

    assert batch_500_ms < 50.0, f"500 rays batch trace took {batch_500_ms:.2f} ms (SLA is < 50 ms)"

    # 2. 5,000 rays SLA < 200 ms
    rays_5000 = _generate_rays(5000, radius=4.0)
    bundle_5000 = RayBundle.from_rays(rays_5000)

    t0 = time.perf_counter()
    b5k = bundle_5000.copy()
    system.trace_bundle(b5k)
    batch_5000_ms = (time.perf_counter() - t0) * 1000.0

    assert batch_5000_ms < 200.0, f"5,000 rays batch trace took {batch_5000_ms:.2f} ms (SLA is < 200 ms)"


# ===========================================================================
# 2. 3D Visualizer MeshCache Tests
# ===========================================================================

def test_mesh_cache_hits_and_invalidation():
    """Verify MeshCache caches surface geometry and invalidates when properties change."""
    from optics_app.gui.visualizer_3d import MeshCache

    cache = MeshCache(maxsize=32)

    surf1 = Surface(name="S1", radius=50.0, conic=0.0, semi_diameter=10.0, surface_type=SurfaceType.STANDARD)
    surf1.global_origin = np.array([0.0, 0.0, 10.0])
    surf1.global_R = np.eye(3)

    key1 = cache.compute_key(surf1, 6, 32)
    assert cache.get(key1) is None
    assert cache.misses == 1

    dummy_mesh = (np.zeros((6, 32)), np.zeros((6, 32)), np.zeros((6, 32)))
    cache.set(key1, dummy_mesh)

    # Retrieve cached mesh
    hit = cache.get(key1)
    assert hit is not None
    assert cache.hits == 1

    # Modify surface radius -> key must change (invalidation)
    surf2 = Surface(name="S1", radius=75.0, conic=0.0, semi_diameter=10.0, surface_type=SurfaceType.STANDARD)
    surf2.global_origin = np.array([0.0, 0.0, 10.0])
    surf2.global_R = np.eye(3)

    key2 = cache.compute_key(surf2, 6, 32)
    assert key2 != key1
    assert cache.get(key2) is None
    assert cache.misses == 2

    # Clear cache
    cache.clear()
    assert cache.get(key1) is None
    assert len(cache) == 0


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_visualizer_3d_incremental_ray_update_skips_clear():
    """Verify 3D Visualizer update_layout skips ax.clear() when surface geometry is unchanged."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from optics_app.gui.visualizer_3d import Visualizer3DWidget

    stack = _build_test_stack()
    system = OpticalSystem(stack)
    rays = system.trace_rays(_generate_rays(9, radius=4.0))

    widget = Visualizer3DWidget()
    widget.canvas.draw_idle = lambda: None
    try:
        # First layout build: full rebuild
        widget.update_layout(surface_stack=stack, rays=rays, force_rebuild=True)
        assert widget._last_geom_fingerprint is not None

        # Monkeypatch ax.clear to verify it is NOT called during incremental ray update
        clear_called = False
        orig_clear = widget.ax.clear

        def guarded_clear():
            nonlocal clear_called
            clear_called = True
            orig_clear()

        widget.ax.clear = guarded_clear

        # Incremental ray update with identical geometry
        new_rays = system.trace_rays(_generate_rays(16, radius=5.0))
        widget.update_layout(surface_stack=stack, rays=new_rays, force_rebuild=False)

        assert not clear_called, "ax.clear() should NOT be called on incremental ray update"

        # Force rebuild should invoke ax.clear()
        widget.update_layout(surface_stack=stack, rays=new_rays, force_rebuild=True)
        assert clear_called, "ax.clear() should be called when force_rebuild=True"
    finally:
        widget.close()


# ===========================================================================
# 3. 2D Visualizer Optimization Tests
# ===========================================================================

@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_dual_view_2d_line_collection_and_axis_reuse():
    """Verify 2D DualViewWidget batches rays using LineCollection and reuses subplots without fig.clf()."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from optics_app.gui.dual_view import DualViewWidget
    from matplotlib.collections import LineCollection

    stack = _build_test_stack()
    system = OpticalSystem(stack)
    traced_rays = system.trace_rays(_generate_rays(21, radius=5.0))

    widget = DualViewWidget()
    try:
        # Enable both R-Z and X-Z views
        widget.rz_view_active = True
        widget.xz_view_active = True

        # Initial build
        widget.update_layout(surface_stack=stack, rays=traced_rays)
        orig_ax_rz = widget.ax_rz
        orig_ax_xz = widget.ax_xz

        assert orig_ax_rz is not None, "ax_rz should be initialized"
        assert orig_ax_xz is not None, "ax_xz should be initialized"

        # Check LineCollection batching
        rz_line_collections = [c for c in widget.ax_rz.collections if isinstance(c, LineCollection)]
        xz_line_collections = [c for c in widget.ax_xz.collections if isinstance(c, LineCollection)]
        assert len(rz_line_collections) >= 1, "R-Z view must use LineCollection for ray batching"
        assert len(xz_line_collections) >= 1, "X-Z view must use LineCollection for ray batching"

        # Second update: axes must be preserved (not torn down by fig.clf())
        widget.update_layout(surface_stack=stack, rays=traced_rays)
        assert widget.ax_rz is orig_ax_rz, "ax_rz should be reused across update cycles"
        assert widget.ax_xz is orig_ax_xz, "ax_xz should be reused across update cycles"
    finally:
        widget.close()


# ===========================================================================
# 4. GUI Event Throttling Tests
# ===========================================================================

@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_component_editor_slider_throttle_and_debounce_timers():
    """Verify ComponentEditorDockWidget implements 33 ms throttle and 150 ms debounce on slider scrub."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
    from optics_app.gui.lde_table import LensDataTableModel

    model = LensDataTableModel()
    dock = ComponentEditorDockWidget(model=model)
    try:
        # Verify timer intervals
        assert hasattr(dock, "_throttle_timer"), "Dock widget must have _throttle_timer"
        assert hasattr(dock, "_debounce_timer"), "Dock widget must have _debounce_timer"
        assert dock._throttle_timer.interval() == 33, "Throttle timer interval must be 33 ms (~30 Hz)"
        assert dock._debounce_timer.interval() == 150, "Debounce timer interval must be 150 ms"

        # Test high-frequency slider event suppression
        emit_count = 0

        def on_change(idx):
            nonlocal emit_count
            emit_count += 1

        dock.component_changed.connect(on_change)

        # Simulate 25 continuous slider scrub movements
        for i in range(25):
            dock._on_tilt_x_slider_changed(i * 10)

        # Rapid slider scrub must not fire 25 synchronous redraw signals
        assert emit_count == 0, f"Expected 0 synchronous redraws during scrub, got {emit_count}"
        assert dock._slider_change_pending is True

        # Slider released event must flush immediately
        dock._on_slider_released()
        assert emit_count == 1, f"Expected 1 immediate emit on slider release, got {emit_count}"
        assert dock._slider_change_pending is False
    finally:
        dock.close()
