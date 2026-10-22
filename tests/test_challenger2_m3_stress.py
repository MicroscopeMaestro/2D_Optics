"""
Adversarial Stress Test Suite for Milestone 3 (Gen 2) Vectorized Ray Tracing & Rendering.
Author: Challenger 2 (Milestone 3 Gate Verification)

Verification Scope:
1. Extreme ray counts (10,000 to 50,000 rays) across complex systems.
2. Grazing incident rays (near 90 degrees) on planar, spherical, mirror, and grating surfaces.
3. 100% vignetted bundles and heavy aperture clipping.
4. Degenerate rays (empty bundle, coaxial rays, zero direction, zero/near-zero wavelength).
5. Chromatic dispersion across grating diffraction and multi-wavelength bundles.
6. Numerical stability (strict absence of NaN / Inf coordinates across all state vectors).
"""

from __future__ import annotations

import os
import time
import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray, RayBundle
from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface
from optics_app.engine.tracer import OpticalSystem
from optics_app.utils.materials import get_refractive_index

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECTROMETER_PATH = os.path.join(PROJECT_ROOT, "spectrometer_raman_50mm_100mm.json")


def _build_dispersive_system() -> OpticalSystem:
    """Build a multi-surface optical system with high-dispersion SF11 singlet and aperture."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=10.0, semi_diameter=10.0))
    stack.add_surface(Surface(name="Front", radius=40.0, thickness=8.0, glass="SF11", semi_diameter=12.5))
    stack.add_surface(Surface(name="Rear", radius=-60.0, thickness=35.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0))
    stack._update_accumulated_positions()
    return OpticalSystem(stack)


def _build_grating_system() -> OpticalSystem:
    """Build a system containing a reflection grating and folding mirrors."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0))
    stack.add_surface(Surface(
        name="Grating",
        surface_type=SurfaceType.GRATING,
        thickness=50.0,
        glass="MIRROR",
        grating_lines_per_mm=600.0,
        grating_order=1,
        tilt_y=15.0,
        semi_diameter=25.0
    ))
    stack.add_surface(Surface(name="FoldMirror", thickness=50.0, glass="MIRROR", tilt_x=10.0, semi_diameter=25.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=30.0))
    stack._update_accumulated_positions()
    return OpticalSystem(stack)


# ===========================================================================
# 1. Extreme Ray Count Stress Testing (10,000 to 50,000 rays)
# ===========================================================================

def test_extreme_ray_counts_10k_spectrometer():
    """Stress test 10,000 rays through full 25-surface Raman spectrometer layout."""
    if not os.path.exists(SPECTROMETER_PATH):
        pytest.skip("Spectrometer JSON file not found")

    stack = SurfaceStack.load_from_json(SPECTROMETER_PATH)["stack"]
    system = OpticalSystem(stack)

    N = 10000
    r = 3.0 * np.sqrt(np.random.rand(N))
    theta = 2.0 * np.pi * np.random.rand(N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(N, -10.0)

    pos = np.column_stack([x, y, z])
    dirs = np.tile([0.0, 0.0, 1.0], (N, 1))
    bundle = RayBundle(pos=pos, dir=dirs, wavelength=0.785)

    t0 = time.perf_counter()
    traced = system.trace_bundle(bundle, record_history=False)
    elapsed = time.perf_counter() - t0

    # 1. Performance throughput check
    assert elapsed < 1.0, f"10k rays through 25 surfaces took {elapsed:.3f} s (expected < 1.0 s)"

    # 2. Strict absence of NaN and Inf
    assert np.all(np.isfinite(traced.pos)), "Non-finite coordinates detected in traced positions"
    assert np.all(np.isfinite(traced.dir)), "Non-finite values detected in traced directions"
    assert np.all(np.isfinite(traced.phase)), "Non-finite values detected in traced phases"
    assert np.all(np.isfinite(traced.path_length)), "Non-finite values detected in traced path lengths"

    # 3. Unit direction normalization
    active = ~traced.vignetted
    if np.any(active):
        dir_norms = np.linalg.norm(traced.dir[active], axis=1)
        np.testing.assert_allclose(dir_norms, 1.0, atol=1e-7, err_msg="Active ray directions lost normalization")


def test_extreme_ray_counts_50k_throughput():
    """Stress test 50,000 rays through multi-surface dispersive system."""
    system = _build_dispersive_system()

    N = 50000
    pos = np.zeros((N, 3), dtype=float)
    r = 8.0 * np.sqrt(np.linspace(0.001, 1.0, N))
    phi = np.linspace(0.0, 200.0 * np.pi, N)
    pos[:, 0] = r * np.cos(phi)
    pos[:, 1] = r * np.sin(phi)
    pos[:, 2] = -15.0

    dirs = np.tile([0.0, 0.0, 1.0], (N, 1))
    bundle = RayBundle(pos=pos, dir=dirs, wavelength=0.5876)

    t0 = time.perf_counter()
    traced = system.trace_bundle(bundle, record_history=False)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"50,000 rays batch trace took {elapsed:.3f} s (expected < 0.5 s)"
    assert np.all(np.isfinite(traced.pos)), "Non-finite coordinates found in 50k bundle"
    assert np.all(np.isfinite(traced.dir)), "Non-finite directions found in 50k bundle"


# ===========================================================================
# 2. Grazing Incident Rays (Near 90 degrees)
# ===========================================================================

def test_grazing_rays_planar_refraction():
    """Test incident rays at extreme grazing angles (80 to 89.999 degrees) into dielectric."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
    stack.add_surface(Surface(name="Interface", radius=0.0, thickness=20.0, glass="N-BK7", semi_diameter=100.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=100.0))
    stack._update_accumulated_positions()
    system = OpticalSystem(stack)

    # Incident angles in degrees approaching 90 degrees
    angles_deg = np.array([0.0, 30.0, 60.0, 85.0, 89.0, 89.9, 89.99, 89.999])
    rads = np.radians(angles_deg)

    # Unit directions with angle theta from normal [0, 0, 1]
    dx = np.sin(rads)
    dz = np.cos(rads)
    dirs = np.column_stack([dx, np.zeros_like(dx), dz])
    pos = np.column_stack([np.zeros_like(dx), np.zeros_like(dx), np.full(len(angles_deg), -10.0)])

    bundle = RayBundle(pos=pos, dir=dirs, wavelength=0.5876)
    traced = system.trace_bundle(bundle)

    assert np.all(np.isfinite(traced.pos)), "Grazing rays produced NaN/Inf position"
    assert np.all(np.isfinite(traced.dir)), "Grazing rays produced NaN/Inf direction"

    # Verify Snell's Law analytically: n1 * sin(theta1) = n2 * sin(theta2)
    n1 = get_refractive_index("AIR", 0.5876)
    n2 = get_refractive_index("N-BK7", 0.5876)
    expected_sin_theta2 = (n1 / n2) * np.sin(rads)
    expected_theta2 = np.arcsin(expected_sin_theta2)

    # Direction inside glass is recorded at step 1
    dir_glass = traced.history[1]["dir"]
    actual_sin_theta2 = dir_glass[:, 0] / np.linalg.norm(dir_glass, axis=1)

    np.testing.assert_allclose(
        actual_sin_theta2,
        expected_sin_theta2,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Grazing angle refraction deviates from analytical Snell law"
    )


def test_grazing_rays_internal_tir_transition():
    """Test rays transitioning through critical angle into total internal reflection."""
    n1 = get_refractive_index("N-BK7", 0.5876)
    n2 = get_refractive_index("AIR", 0.5876)
    crit_angle_deg = np.degrees(np.arcsin(n2 / n1))  # ~41.26 degrees

    # Span below and above critical angle across interface tilts
    tilts = np.array([
        crit_angle_deg - 5.0,
        crit_angle_deg - 0.5,
        crit_angle_deg + 0.5,
        crit_angle_deg + 5.0,
        80.0,
        89.9
    ])

    for tilt in tilts:
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.add_surface(Surface(name="Front", radius=0.0, thickness=20.0, glass="N-BK7", semi_diameter=100.0))
        stack.add_surface(Surface(name="Rear_TIR", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=100.0, tilt_y=tilt))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=100.0))
        stack._update_accumulated_positions()
        system = OpticalSystem(stack)

        ray = Ray(pos=(0.0, 0.0, -10.0), dir=(0.0, 0.0, 1.0), wavelength=0.5876)
        bundle = RayBundle.from_rays([ray])
        traced = system.trace_bundle(bundle)

        assert np.all(np.isfinite(traced.pos)), f"TIR at tilt {tilt} produced NaN/Inf position"
        assert np.all(np.isfinite(traced.dir)), f"TIR at tilt {tilt} produced NaN/Inf direction"

        expected_tir = tilt > crit_angle_deg
        assert traced.tir[0] == expected_tir, f"TIR at tilt {tilt} expected {expected_tir} but got {traced.tir[0]}"



def test_grazing_rays_spherical_surface():
    """Test rays incident near 90 degrees on high-curvature spherical surface."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
    stack.add_surface(Surface(name="Sphere", radius=25.0, thickness=20.0, glass="N-BK7", semi_diameter=24.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=50.0))
    stack._update_accumulated_positions()
    system = OpticalSystem(stack)

    # Launch rays at heights grazing the spherical curvature rim
    heights = np.array([0.0, 10.0, 20.0, 23.5, 23.9, 23.99, 24.5, 30.0])
    N = len(heights)
    pos = np.column_stack([heights, np.zeros(N), np.full(N, -10.0)])
    dirs = np.tile([0.0, 0.0, 1.0], (N, 1))

    bundle = RayBundle(pos=pos, dir=dirs, wavelength=0.5876)
    traced = system.trace_bundle(bundle)

    # All coordinates must remain strictly finite (no sqrt of negative numbers)
    assert np.all(np.isfinite(traced.pos)), "Spherical grazing rays produced non-finite positions"
    assert np.all(np.isfinite(traced.dir)), "Spherical grazing rays produced non-finite directions"

    # Rays launched outside semi-diameter (24.5, 30.0) must be vignetted
    assert traced.vignetted[-2] == True, "Ray outside semi-diameter was not vignetted"
    assert traced.vignetted[-1] == True, "Ray far outside semi-diameter was not vignetted"


# ===========================================================================
# 3. 100% Vignetted Bundles & Heavy Aperture Clipping
# ===========================================================================

def test_100_percent_vignetted_bundle_outside_aperture():
    """Test 100% vignetting where all rays in bundle miss the aperture."""
    system = _build_dispersive_system()

    N = 100
    # Launch all rays at radius = 50 mm (aperture is 10 mm)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = 50.0 * np.cos(angles)
    y = 50.0 * np.sin(angles)
    pos = np.column_stack([x, y, np.full(N, -10.0)])
    dirs = np.tile([0.0, 0.0, 1.0], (N, 1))

    bundle = RayBundle(pos=pos, dir=dirs, wavelength=0.5876)
    traced = system.trace_bundle(bundle, record_history=True)

    # 1. 100% vignetting must be recognized
    assert np.all(traced.vignetted), "Not all rays were vignetted as expected"

    # 2. Coordinates must remain finite
    assert np.all(np.isfinite(traced.pos)), "Vignetted bundle produced non-finite positions"
    assert np.all(np.isfinite(traced.dir)), "Vignetted bundle produced non-finite directions"

    # 3. Tracing must have short-circuited early
    # Surface index 1 is Stop (semi_d=10). Rays vignette there.
    # History should contain entries up to the vignetting surface, not full 5 surfaces.
    assert len(traced.history) <= 2, f"History length {len(traced.history)} indicates early exit failed"


def test_bundle_pre_flagged_100_percent_vignetted():
    """Verify trace_bundle handles a bundle that is already 100% vignetted before call."""
    system = _build_dispersive_system()

    N = 50
    pos = np.zeros((N, 3))
    dirs = np.tile([0.0, 0.0, 1.0], (N, 1))
    vignetted = np.ones(N, dtype=bool)  # All True initially

    bundle = RayBundle(pos=pos, dir=dirs, vignetted=vignetted)
    traced = system.trace_bundle(bundle, record_history=True)

    assert np.all(traced.vignetted), "Pre-vignetted status altered during trace"
    assert len(traced.history) == 0, "Pre-vignetted bundle should perform 0 surface steps"
    assert np.all(np.isfinite(traced.pos)), "Pre-vignetted bundle coordinates corrupted"


def test_progressive_aperture_clipping_ratios():
    """Test progressive aperture clipping across mixed radii."""
    system = _build_dispersive_system()

    # 200 rays evenly spaced from r = 0 to r = 20 mm.
    # Stop aperture is semi_diameter = 10.0 mm.
    N = 200
    radii = np.linspace(0.0, 20.0, N)
    pos = np.column_stack([radii, np.zeros(N), np.full(N, -10.0)])
    dirs = np.tile([0.0, 0.0, 1.0], (N, 1))

    bundle = RayBundle(pos=pos, dir=dirs, wavelength=0.5876)
    traced = system.trace_bundle(bundle)

    # Rays with r <= 10.0 must be active; r > 10.0 must be vignetted
    expected_vignetted = radii > 10.0
    np.testing.assert_array_equal(traced.vignetted, expected_vignetted)

    # Active rays must be valid and non-vignetted
    active_idx = np.where(~traced.vignetted)[0]
    assert len(active_idx) > 0, "No rays survived aperture"
    assert np.all(np.isfinite(traced.pos[active_idx])), "Active ray positions not finite"
    assert np.all(np.isfinite(traced.dir[active_idx])), "Active ray directions not finite"


# ===========================================================================
# 4. Degenerate Rays & Boundary Conditions
# ===========================================================================

def test_empty_ray_bundle():
    """Verify trace_bundle handles empty bundle (0 rays) cleanly."""
    system = _build_dispersive_system()

    empty_bundle = RayBundle(pos=np.empty((0, 3)), dir=np.empty((0, 3)))
    assert len(empty_bundle) == 0

    traced = system.trace_bundle(empty_bundle)
    assert len(traced) == 0
    assert traced.pos.shape == (0, 3)
    assert traced.dir.shape == (0, 3)


def test_coaxial_zero_offset_rays():
    """Test bundle of multiple identical rays positioned exactly on the optical axis."""
    system = _build_dispersive_system()

    N = 100
    pos = np.zeros((N, 3))
    pos[:, 2] = -10.0
    dirs = np.tile([0.0, 0.0, 1.0], (N, 1))

    bundle = RayBundle(pos=pos, dir=dirs, wavelength=0.5876)
    traced = system.trace_bundle(bundle)

    # All rays should remain exactly on axis (x=0, y=0)
    np.testing.assert_allclose(traced.pos[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(traced.pos[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(traced.dir[:, 2], 1.0, atol=1e-12)
    assert not np.any(traced.vignetted), "On-axis rays unexpectedly vignetted"


def test_zero_and_near_zero_wavelength_stability():
    """Verify numerical stability under zero and ultra-short wavelength without division by zero."""
    system = _build_dispersive_system()

    # Wavelengths including 0.0 and extreme values
    wls = [0.0, 1e-12, 1e-6, 0.5876, 10.0]
    pos = np.zeros((len(wls), 3))
    pos[:, 2] = -10.0
    dirs = np.tile([0.0, 0.0, 1.0], (len(wls), 1))

    bundle = RayBundle(pos=pos, dir=dirs, wavelength=wls)
    traced = system.trace_bundle(bundle)

    # Must complete without ZeroDivisionError and produce finite results
    assert np.all(np.isfinite(traced.pos)), "Zero wavelength produced non-finite position"
    assert np.all(np.isfinite(traced.dir)), "Zero wavelength produced non-finite direction"
    assert np.all(np.isfinite(traced.phase)), "Zero wavelength produced non-finite phase"


def test_degenerate_zero_direction_vector():
    """Verify RayBundle handles zero direction vector gracefully."""
    p = np.array([[0.0, 0.0, -10.0], [1.0, 0.0, -10.0]])
    d = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    bundle = RayBundle(pos=p, dir=d, wavelength=0.5876)
    # The zero vector should not produce NaNs during norm division
    assert np.all(np.isfinite(bundle.dir)), "Zero direction vector produced NaN in RayBundle"


# ===========================================================================
# 5. Chromatic Dispersion Across Vector Grating & Lenses
# ===========================================================================

def test_grating_vector_dispersion_exactness():
    """Verify 3D grating vector diffraction law across spectrum matches analytical diffraction."""
    system = _build_grating_system()

    # Wavelengths from 400 nm to 900 nm
    wls = np.linspace(0.400, 0.900, 11)
    N = len(wls)
    pos = np.zeros((N, 3))
    pos[:, 2] = -20.0
    dirs = np.tile([0.0, 0.0, 1.0], (N, 1))

    bundle = RayBundle(pos=pos, dir=dirs, wavelength=wls)
    traced_bundle = system.trace_bundle(bundle)

    # Compare against scalar trace_rays
    scalar_rays = [Ray(pos=pos[i], dir=dirs[i], wavelength=wls[i]) for i in range(N)]
    traced_scalar = system.trace_rays(scalar_rays)

    for i in range(N):
        assert traced_bundle.vignetted[i] == traced_scalar[i].vignetted
        if not traced_bundle.vignetted[i]:
            pos_delta = np.linalg.norm(traced_bundle.pos[i] - traced_scalar[i].pos)
            dir_delta = np.linalg.norm(traced_bundle.dir[i] - traced_scalar[i].dir)
            assert pos_delta < 1e-9, f"Wavelength {wls[i]:.3f} um pos delta {pos_delta:.2e} exceeds tolerance"
            assert dir_delta < 1e-9, f"Wavelength {wls[i]:.3f} um dir delta {dir_delta:.2e} exceeds tolerance"


def test_monochromatic_bundle_parity_across_spectral_bands():
    """Verify batch trace agrees with scalar trace across UV, VIS, and NIR bands."""
    system = _build_dispersive_system()

    for wl in [0.380, 0.4861, 0.5876, 0.6563, 0.850, 1.064]:
        rays = [
            Ray(pos=np.array([float(x), float(y), -10.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            for x in np.linspace(-5.0, 5.0, 5)
            for y in np.linspace(-5.0, 5.0, 5)
        ]
        bundle = RayBundle.from_rays(rays)

        traced_bundle = system.trace_bundle(bundle)
        traced_scalar = system.trace_rays(rays)

        for i in range(len(rays)):
            assert traced_bundle.vignetted[i] == traced_scalar[i].vignetted
            if not traced_bundle.vignetted[i]:
                pos_err = np.linalg.norm(traced_bundle.pos[i] - traced_scalar[i].pos)
                dir_err = np.linalg.norm(traced_bundle.dir[i] - traced_scalar[i].dir)
                assert pos_err < 1e-9, f"Band {wl} um ray {i} pos error {pos_err:.2e} > 1e-9"
                assert dir_err < 1e-9, f"Band {wl} um ray {i} dir error {dir_err:.2e} > 1e-9"


# ===========================================================================
# 6. Visualizer & GUI Event Optimization Stress Tests
# ===========================================================================

def test_mesh_cache_eviction_and_churn():
    """Verify MeshCache enforces maxsize bound and handles rapid cache churn without memory leak."""
    from optics_app.gui.visualizer_3d import MeshCache

    cache = MeshCache(maxsize=10)
    surfaces = [
        Surface(name=f"Surf_{i}", radius=float(10 + i * 5), semi_diameter=float(5 + i))
        for i in range(50)
    ]

    # Insert 50 entries into cache of maxsize 10
    for surf in surfaces:
        key = MeshCache.compute_key(surf, 6, 32)
        assert cache.get(key) is None  # Miss
        cache.set(key, np.zeros((10, 3)))

    # Cache length must be strictly capped at maxsize
    assert len(cache._cache) == 10
    assert cache.misses == 50

    # The most recent 10 surfaces must hit
    for surf in surfaces[-10:]:
        key = MeshCache.compute_key(surf, 6, 32)
        assert cache.get(key) is not None

    assert cache.hits == 10


def test_dual_view_2d_extreme_bundle_batching():
    """Verify 2D DualViewWidget handles large ray batches and 100% vignetted bundles without crash."""
    from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
    from optics_app.gui.dual_view import DualViewWidget
    from matplotlib.collections import LineCollection

    if not QT_AVAILABLE or not hasattr(QtWidgets, "QApplication"):
        pytest.skip("Qt not available")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = DualViewWidget()
    widget.rz_view_active = True
    widget.xz_view_active = True

    stack = SurfaceStack()
    stack.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=10.0, semi_diameter=5.0))
    stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0))
    stack._update_accumulated_positions()
    system = OpticalSystem(stack)

    # 1. Extreme ray count (2,000 rays)
    rays = [
        Ray(pos=np.array([float(r), 0.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
        for r in np.linspace(-15.0, 15.0, 2000)
    ]
    bundle = RayBundle.from_rays(rays)
    traced = system.trace_bundle(bundle, record_history=True)

    # Convert back to rays for widget update
    widget.update_layout(surface_stack=stack, rays=traced.to_rays())

    rz_colls = [c for c in widget.ax_rz.collections if isinstance(c, LineCollection)]
    xz_colls = [c for c in widget.ax_xz.collections if isinstance(c, LineCollection)]
    assert len(rz_colls) >= 1, "R-Z view failed to generate LineCollection for 2k rays"
    assert len(xz_colls) >= 1, "X-Z view failed to generate LineCollection for 2k rays"

    # 2. 100% vignetted rays
    vignetted_rays = [
        Ray(pos=np.array([100.0, 0.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
        for _ in range(50)
    ]
    traced_vig = system.trace_bundle(RayBundle.from_rays(vignetted_rays), record_history=True)
    widget.update_layout(surface_stack=stack, rays=traced_vig.to_rays())

    # Should update cleanly without exception
    widget.close()


def test_component_editor_slider_flood_stress():
    """Stress-test ComponentEditorDockWidget slider event pipeline with 500 high-frequency events."""
    from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
    from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
    from optics_app.gui.lde_table import LensDataTableModel

    if not QT_AVAILABLE or not hasattr(QtWidgets, "QApplication"):
        pytest.skip("Qt not available")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    model = LensDataTableModel()
    dock = ComponentEditorDockWidget(model=model)

    emitted_count = [0]
    dock.component_changed.connect(lambda idx: emitted_count.__setitem__(0, emitted_count[0] + 1))

    # Flood 500 rapid slider change events
    for val in range(500):
        dock._on_tilt_x_slider_changed(val % 300)

    # During flood, throttle timer is running but not processed synchronously
    assert emitted_count[0] == 0, "Synchronous emission occurred during rapid flood"

    # On slider release, flush immediately
    dock._on_slider_released()
    assert emitted_count[0] == 1, f"Expected exactly 1 flush emit on release, got {emitted_count[0]}"

    dock.close()


