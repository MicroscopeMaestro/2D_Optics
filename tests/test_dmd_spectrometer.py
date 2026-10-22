"""
Unit Tests for All-Mirror DMD4500 Compressive Raman Spectrometer.
Validates the complete 7-element optical path:
25 um fiber -> M1 collimator -> Grating -> M2 focusing mirror -> DMD4500 (half-width match) ->
M3/M4 two-mirror relay -> 1.0 mm single-element detector.
"""

import pytest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from scratch.design_dmd_mirror_spectrometer import (
    build_dmd4500_mirror_spectrometer, generate_fiber_rays
)


def test_dmd4500_spectrometer_construction():
    """Verify SurfaceStack contains all 8 required optical surfaces."""
    stack = build_dmd4500_mirror_spectrometer()
    assert len(stack.surfaces) == 8
    
    # 0. Fiber head
    assert stack.surfaces[0].surface_type == SurfaceType.OBJECT
    assert stack.surfaces[0].semi_diameter == 0.0125
    
    # 1. Collimator M1
    assert stack.surfaces[1].radius == -100.0
    assert stack.surfaces[1].glass == "MIRROR"
    
    # 2. Diffraction Grating
    assert stack.surfaces[2].surface_type == SurfaceType.GRATING
    assert stack.surfaces[2].grating_lines_per_mm == 600.0
    
    # 3. Focusing Mirror M2
    assert stack.surfaces[3].radius in (-100.0, -150.0)
    assert stack.surfaces[3].glass == "MIRROR"
    
    # 4. TI DMD4500 Array
    assert "DMD" in stack.surfaces[4].name
    assert abs(stack.surfaces[4].aperture_x_half - 4.9275) < 1e-3 or abs(stack.surfaces[4].aperture_y_half - 4.9275) < 1e-3 # Half of 9.855 mm active width
    assert stack.surfaces[4].tilt_x == 12.0 # +12 deg ON-state micromirror tilt
    
    # 5. Relay Mirror M3
    assert stack.surfaces[5].radius == -300.0 or stack.surfaces[5].radius == -150.0
    
    # 6. Relay Condenser Mirror M4
    assert stack.surfaces[6].radius == -30.0 # Fast condenser f = 15 mm
    
    # 7. Single-Element Detector
    assert stack.surfaces[7].surface_type == SurfaceType.DETECTOR
    assert stack.surfaces[7].semi_diameter == 0.50 # 1.0 mm active area diameter


def test_dmd4500_spectrometer_half_array_dispersion():
    """Verify dispersed Raman band (785-880 nm) fills exactly half of TI DMD4500 array."""
    stack = build_dmd4500_mirror_spectrometer()
    opt_sys = OpticalSystem(stack)
    
    wavelengths = [0.785, 0.880]
    rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        for wl in wavelengths
    ]
    traced = opt_sys.trace_rays(rays)
    
    # Extract positions on DMD surface (index 4)
    hits = {}
    for tr in traced:
        for h in tr.history:
            if h["surface_index"] == 4:
                # Active dispersion coordinate (YZ plane or X plane)
                coord = h["x"] if abs(h["x"]) > 1e-3 else h["y"]
                hits[tr.wavelength] = coord
                
    assert 0.785 in hits and 0.880 in hits
    dmd_spread = abs(hits[0.880] - hits[0.785])
    
    # DMD4500 active width is 8.664 mm. Dispersion span is ~4.95 mm.
    assert 4.5 <= dmd_spread <= 5.2
    dmd_fraction = (dmd_spread / 9.855) * 100.0
    assert 45.0 <= dmd_fraction <= 58.0


def test_dmd4500_spectrometer_full_ray_trace_to_detector():
    """Verify rays from fiber propagate through all 8 surfaces to 1.0 mm detector."""
    stack = build_dmd4500_mirror_spectrometer()
    opt_sys = OpticalSystem(stack)
    
    rays = generate_fiber_rays([0.785, 0.830, 0.880], num_rays_per_ring=8)
    traced = opt_sys.trace_rays(rays)
    
    det_hits = [tr for tr in traced if any(h.get("surface_index") == 7 for h in tr.history)]
    assert len(det_hits) > 0
    # Over 60% throughput through the entire 7-element all-mirror folded system
    throughput_pct = (len(det_hits) / len(rays)) * 100.0
    assert throughput_pct >= 60.0
