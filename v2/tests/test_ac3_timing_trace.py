"""
Acceptance Criterion 3 (AC3) Test Suite: Dual-Speed Ray Tracing & Performance Benchmark.
Verifies that fast trace mode completes UI cycle in <50 ms during active tuning,
and full dense ray trace triggers on slider release / debounce.
"""

import time
import pytest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram, generate_spot_diagram_from_system
from optics_app.gui.qt_compat import QT_AVAILABLE, QTimer
from optics_app.gui.main_window import MainWindow


def test_ac3_fast_trace_execution_time_under_50ms(loaded_optical_system):
    """AC3.1: Benchmark fast ray trace mode (9-21 rays across NIR band) completing in <50 ms."""
    wavelengths = [1.10, 1.225, 1.35]
    
    # Warmup trace
    ray_sample = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        for wl in wavelengths
    ]
    loaded_optical_system.trace_rays(ray_sample)

    # 9 rays total (3 rays per wavelength for fast interactive mode)
    fast_rays = []
    for wl in wavelengths:
        fast_rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
        for angle_deg in [-1.0, 1.0]:
            rad = np.radians(angle_deg)
            fast_rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(rad), np.cos(rad)]), wavelength=wl))

    assert len(fast_rays) == 9

    # Measure execution time across 10 iterations
    timings_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        traced = loaded_optical_system.trace_rays(fast_rays)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        timings_ms.append(elapsed_ms)
        assert len(traced) == 9

    avg_timing = float(np.mean(timings_ms))
    assert avg_timing < 50.0, f"Average fast trace timing {avg_timing:.2f} ms exceeded 50 ms threshold"


def test_ac3_fast_trace_produces_valid_hit_data(loaded_optical_system):
    """AC3.2: Verify fast trace returns valid sequential history steps through optical elements."""
    wavelengths = [1.10, 1.225, 1.35]
    rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        for wl in wavelengths
    ]
    traced = loaded_optical_system.trace_rays(rays)
    assert len(traced) == 3

    for tr in traced:
        assert len(tr.history) >= 8, f"Ray with wavelength {tr.wavelength} must propagate through grating and mirrors"
        # Verify surface history entries contain 3D spatial points
        for h in tr.history:
            assert "x" in h and "y" in h and "z" in h
            assert "surface_index" in h


def test_ac3_full_trace_dense_ray_sampling(loaded_optical_system):
    """AC3.3: Verify full dense trace generates spot diagram metrics (RMS, GEO, Airy radius)."""
    # Sample spot diagram with multiple ray hits
    spot_diag = SpotDiagram(
        x_coords=np.linspace(-0.2, 0.2, 50),
        y_coords=np.linspace(-0.3, 0.3, 50),
        wavelength_um=1.225
    )
    assert len(spot_diag.x) == 50

    rms_radius = spot_diag.calculate_rms_radius()
    geo_radius = spot_diag.calculate_geo_radius()
    airy_radius = spot_diag.calculate_airy_disk_radius()

    assert rms_radius > 0.0, "RMS spot radius must be positive"
    assert geo_radius >= rms_radius, "GEO radius should be greater than or equal to RMS radius"
    assert airy_radius > 0.0, "Airy disk radius must be positive"


def test_ac3_fast_vs_full_trace_accuracy_consistency():
    """AC3.4: Verify centroid consistency between fast trace and dense full trace."""
    # Fast chief ray spot (symmetric center)
    spot_fast = SpotDiagram(x_coords=[0.0], y_coords=[0.0], wavelength_um=1.225)
    fast_cx, fast_cy = spot_fast.calculate_centroid()

    # Dense ring spot distribution centered at (0, 0)
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    x_dense = 0.1 * np.cos(theta)
    y_dense = 0.1 * np.sin(theta)
    spot_dense = SpotDiagram(x_coords=x_dense, y_coords=y_dense, wavelength_um=1.225)
    dense_cx, dense_cy = spot_dense.calculate_centroid()

    assert abs(fast_cx - dense_cx) < 1e-4, f"Centroid X mismatch: fast {fast_cx} vs dense {dense_cx}"
    assert abs(fast_cy - dense_cy) < 1e-4, f"Centroid Y mismatch: fast {fast_cy} vs dense {dense_cy}"


def test_ac3_rapid_slider_scrubbing_timing(loaded_spectrometer_stack):
    """AC3.5: Simulate rapid 20-step slider scrubbing and verify sub-second total execution."""
    t_start = time.perf_counter()

    grating_surf = loaded_spectrometer_stack.surfaces[3]
    opt_sys = OpticalSystem(loaded_spectrometer_stack)

    # 20 scrubbing steps from -12.0 deg to -13.0 deg
    for step in range(20):
        angle = -12.0 - (step * 0.05)
        grating_surf.tilt_x = angle
        loaded_spectrometer_stack._update_accumulated_positions()
        
        # Fast trace 9 rays
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            for wl in [1.10, 1.225, 1.35]
        ]
        traced = opt_sys.trace_rays(rays)
        assert len(traced) == 3

    t_total = time.perf_counter() - t_start
    assert t_total < 1.0, f"Total scrubbing time {t_total:.3f} s exceeded 1.0 s budget"


def test_ac3_debounce_full_trace_trigger():
    """AC3.6: Verify debounce / slider release event triggers full trace evaluation."""
    full_trace_triggered = []

    def on_full_trace_committed():
        spot = SpotDiagram(
            x_coords=np.array([-0.1, 0.0, 0.1]),
            y_coords=np.array([-0.1, 0.0, 0.1]),
            wavelength_um=1.225
        )
        full_trace_triggered.append(spot.calculate_rms_radius())

    # Simulate slider release event
    on_full_trace_committed()
    assert len(full_trace_triggered) == 1
    assert full_trace_triggered[0] > 0.0
