"""
Unit Tests for Gaussian Beam Propagation via ABCD Matrix / Complex q-Parameter (Requirement R5 Extension).
"""
import pytest
import numpy as np
from optics_app.engine.gaussian import GaussianBeam, trace_system_gaussian_beam
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem


def test_free_space_gaussian_beam_expansion():
    """Test Gaussian beam free-space expansion formula w(z) = w0 * sqrt(1 + (z/zR)^2)."""
    w0 = 0.1  # 0.1 mm = 100 um
    wl = 0.6328  # 632.8 nm (in um)
    beam = GaussianBeam(w0=w0, wavelength=wl, n=1.0, z=0.0)

    # Convert wavelength to mm in test verification
    wl_mm = wl * 1e-3
    zR_expected = np.pi * (w0**2) / wl_mm
    assert abs(beam.z_R - zR_expected) < 1e-5

    # Check initial waist
    assert abs(beam.spot_radius - w0) < 1e-6
    assert beam.radius_of_curvature == float('inf')

    # Propagate to z = zR
    beam.propagate_distance(zR_expected)
    w_at_zR = beam.spot_radius
    assert abs(w_at_zR - w0 * np.sqrt(2.0)) < 1e-5
    assert abs(beam.radius_of_curvature - 2.0 * zR_expected) < 1e-3


def test_thin_lens_gaussian_beam_transformation():
    """Test thin lens waist focusing for Gaussian beam."""
    w0 = 0.5  # mm
    wl = 0.5876  # um
    beam = GaussianBeam(w0=w0, wavelength=wl, n=1.0, z=0.0)

    # Propagate 100 mm to lens
    beam.propagate_distance(100.0)
    w_lens = beam.spot_radius

    # Pass through thin lens with f = 50 mm
    beam.pass_thin_lens(f=50.0)

    assert beam.spot_radius == w_lens
    assert not np.isinf(beam.radius_of_curvature)


def test_system_gaussian_beam_tracing():
    """Test full system Gaussian beam trace."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Source", thickness=50.0))
    stack.add_surface(Surface(name="Lens1", radius=50.0, glass="BK7", thickness=10.0))
    stack.add_surface(Surface(name="Lens2", radius=-50.0, glass="AIR", thickness=100.0))
    stack.add_surface(Surface(name="Detector", thickness=0.0))

    opt_sys = OpticalSystem(stack)
    res = opt_sys.trace_gaussian_beam(w0=1.0, wavelength=0.6328, source_z=0.0)

    assert len(res.nodes) == 4
    assert res.nodes[0].w == 1.0
    assert res.final_w > 0.0
