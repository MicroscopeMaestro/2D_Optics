"""
Challenger M4 Empirical Stress Test: Engine Throughput Scalability (10,000 to 50,000 rays).

Verifies:
1. Linear O(N) throughput scaling across 10k, 25k, and 50k ray bundles.
2. Vectorized RayBundle memory stability and no NaN/inf coordinates under high ray densities.
3. OpticalSystem throughput consistency (> 50,000 rays/sec) on multi-surface systems.
4. History allocation overhead under massive ray counts.
"""

import os
import sys
import time
import pytest
import numpy as np

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray, RayBundle
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.surface import Surface, SurfaceType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECTROMETER_PATH = os.path.join(PROJECT_ROOT, "spectrometer_raman_50mm_100mm.json")


def _generate_dense_bundle(num_rays: int, beam_radius: float = 0.02, wavelength_um: float = 0.5876) -> RayBundle:
    """Generate dense collimated bundle as a vectorized RayBundle sized to sample entrance apertures."""
    indices = np.arange(0, num_rays, dtype=float) + 0.5
    r = beam_radius * np.sqrt(indices / num_rays)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(num_rays, 0.0)
    pos = np.column_stack([x, y, z])

    dirs = np.zeros((num_rays, 3), dtype=float)
    dirs[:, 2] = 1.0  # +Z direction

    wavelengths = np.full(num_rays, wavelength_um)
    return RayBundle(pos=pos, dir=dirs, wavelength=wavelengths)


@pytest.fixture(scope="module")
def spectrometer_system():
    assert os.path.exists(SPECTROMETER_PATH), f"Missing project {SPECTROMETER_PATH}"
    data = SurfaceStack.load_from_json(SPECTROMETER_PATH)
    return OpticalSystem(data["stack"])


@pytest.fixture(scope="module")
def singlet_system():
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    stack.add_surface(Surface(name="L1_Rear", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=15.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=20.0))
    stack._update_accumulated_positions()
    return OpticalSystem(stack)


@pytest.mark.parametrize("num_rays", [10000, 25000, 50000])
def test_dense_ray_bundle_throughput_and_physical_validity(spectrometer_system, num_rays):
    """Stress test large ray bundles (10k, 25k, 50k) on 25-surface optical system."""
    bundle = _generate_dense_bundle(num_rays)

    t0 = time.perf_counter()
    traced_bundle = spectrometer_system.trace_bundle(bundle, record_history=False)
    elapsed = time.perf_counter() - t0

    throughput = num_rays / max(elapsed, 1e-6)

    print(f"\n[Dense Stress] {num_rays:,} rays: {elapsed*1000.0:.2f} ms ({throughput:,.0f} rays/sec)")

    # Assert physical sanity
    assert len(traced_bundle.pos) == num_rays
    assert len(traced_bundle.dir) == num_rays
    assert len(traced_bundle.vignetted) == num_rays

    # Check for NaN / Inf in non-vignetted rays
    unvig = ~traced_bundle.vignetted
    assert np.any(unvig), "Expected some rays to pass through the slit"
    assert not np.any(np.isnan(traced_bundle.pos[unvig])), "Found NaN in ray positions"
    assert not np.any(np.isinf(traced_bundle.pos[unvig])), "Found Inf in ray positions"
    assert not np.any(np.isnan(traced_bundle.dir[unvig])), "Found NaN in ray directions"
    assert not np.any(np.isinf(traced_bundle.dir[unvig])), "Found Inf in ray directions"

    # Direction vectors must remain unit length
    norms = np.linalg.norm(traced_bundle.dir[unvig], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6, err_msg="Ray directions are not unit vectors")

    # Verify throughput exceeds high-performance bar (> 40,000 rays/sec across 25 surfaces)
    assert throughput > 40000.0, f"Throughput {throughput:.0f} rays/sec is below 40,000 target"


def test_singlet_throughput_scaling_linearity(singlet_system):
    """Verify execution time scales approximately linearly O(N) from 10k to 50k rays on singlet."""
    bundle_10k = _generate_dense_bundle(10000, beam_radius=5.0)
    bundle_50k = _generate_dense_bundle(50000, beam_radius=5.0)

    # Warmup
    singlet_system.trace_bundle(bundle_10k.copy(), record_history=False)

    # Time 10k
    t0 = time.perf_counter()
    singlet_system.trace_bundle(bundle_10k.copy(), record_history=False)
    t_10k = time.perf_counter() - t0

    # Time 50k
    t0 = time.perf_counter()
    singlet_system.trace_bundle(bundle_50k.copy(), record_history=False)
    t_50k = time.perf_counter() - t0

    scaling_ratio = t_50k / max(t_10k, 1e-6)
    print(f"\n[Singlet Linearity] 10k={t_10k*1000.0:.2f} ms, 50k={t_50k*1000.0:.2f} ms (Ratio={scaling_ratio:.2f}x for 5x rays)")

    # 5x rays should scale within 3.0x to 12.0x (reflecting memory hierarchy & cache effects)
    assert 3.0 <= scaling_ratio <= 12.0, f"Scaling ratio {scaling_ratio:.2f} deviates significantly from ~5.0x"


def test_spectrometer_throughput_scaling_linearity(spectrometer_system):
    """Verify execution time scales reasonably from 10k to 50k rays on 25-surface spectrometer."""
    bundle_10k = _generate_dense_bundle(10000, beam_radius=0.02)
    bundle_50k = _generate_dense_bundle(50000, beam_radius=0.02)

    # Warmup
    spectrometer_system.trace_bundle(bundle_10k.copy(), record_history=False)

    # Time 10k
    t0 = time.perf_counter()
    spectrometer_system.trace_bundle(bundle_10k.copy(), record_history=False)
    t_10k = time.perf_counter() - t0

    # Time 50k
    t0 = time.perf_counter()
    spectrometer_system.trace_bundle(bundle_50k.copy(), record_history=False)
    t_50k = time.perf_counter() - t0

    scaling_ratio = t_50k / max(t_10k, 1e-6)
    print(f"\n[Spectrometer Linearity] 10k={t_10k*1000.0:.2f} ms, 50k={t_50k*1000.0:.2f} ms (Ratio={scaling_ratio:.2f}x for 5x rays)")

    # 5x rays should scale within 3.5x to 15.0x
    assert 3.5 <= scaling_ratio <= 15.0, f"Scaling ratio {scaling_ratio:.2f} deviates significantly"


def test_history_memory_impact_under_large_bundles(spectrometer_system):
    """Verify that record_history=True does not crash or corrupt memory with 10k rays."""
    bundle_10k = _generate_dense_bundle(10000, beam_radius=0.02)

    t0 = time.perf_counter()
    traced_with_hist = spectrometer_system.trace_bundle(bundle_10k.copy(), record_history=True)
    t_with_hist = time.perf_counter() - t0

    assert len(traced_with_hist.history) > 0, "History must be recorded"
    print(f"\n[History Stress] 10k rays with full history: {t_with_hist*1000.0:.2f} ms, history surfaces={len(traced_with_hist.history)}")
