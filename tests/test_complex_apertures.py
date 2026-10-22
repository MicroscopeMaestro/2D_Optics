"""
Unit Tests for Complex Geometric Apertures and Vignetting (Requirement R5).
"""
import pytest
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem


def test_circular_aperture():
    surf = Surface(aperture_type="Circular", semi_diameter=20.0, aperture_r_outer=20.0)
    assert surf.check_aperture(0.0, 0.0)
    assert surf.check_aperture(10.0, 10.0)  # r = sqrt(200) ~ 14.14 <= 20
    assert not surf.check_aperture(15.0, 15.0)  # r = sqrt(450) ~ 21.21 > 20


def test_rectangular_aperture():
    surf = Surface(aperture_type="Rectangular", aperture_x_half=10.0, aperture_y_half=20.0)
    assert surf.check_aperture(0.0, 0.0)
    assert surf.check_aperture(9.5, 19.5)
    assert not surf.check_aperture(10.5, 5.0)   # x > 10
    assert not surf.check_aperture(5.0, 20.5)   # y > 20


def test_elliptical_aperture():
    surf = Surface(aperture_type="Elliptical", aperture_a_semi=20.0, aperture_b_semi=10.0)
    assert surf.check_aperture(0.0, 0.0)
    assert surf.check_aperture(15.0, 0.0)   # (15/20)^2 = 0.5625 <= 1
    assert surf.check_aperture(0.0, 8.0)    # (8/10)^2 = 0.64 <= 1
    assert not surf.check_aperture(15.0, 8.0)  # (15/20)^2 + (8/10)^2 = 1.2025 > 1


def test_annular_aperture():
    surf = Surface(aperture_type="Annular", aperture_r_inner=5.0, aperture_r_outer=25.0)
    assert not surf.check_aperture(2.0, 0.0)  # r = 2 < 5 (obscured)
    assert surf.check_aperture(10.0, 0.0)    # 5 <= 10 <= 25 (passed)
    assert not surf.check_aperture(30.0, 0.0)  # r = 30 > 25 (vignetted)


def test_spider_aperture():
    surf = Surface(
        aperture_type="Spider",
        aperture_r_outer=25.0,
        aperture_center_radius=5.0,
        aperture_spider_width=2.0,
        aperture_spider_count=4,
        aperture_spider_angle=0.0
    )
    # Passed region: r between 5 and 25 away from axes
    assert surf.check_aperture(10.0, 10.0)

    # Blocked by central hub (r <= 5)
    assert not surf.check_aperture(3.0, 0.0)

    # Blocked by outer rim (r > 25)
    assert not surf.check_aperture(26.0, 0.0)

    # Blocked by spider leg along X-axis (angle 0 deg, width 2.0 -> |y| <= 1.0 for x in [5, 25])
    assert not surf.check_aperture(15.0, 0.5)
    # Passed just outside leg along X-axis
    assert surf.check_aperture(15.0, 2.0)


def test_tracer_complex_aperture_vignetting():
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="STOP", thickness=10.0, aperture_type="Rectangular", aperture_x_half=5.0, aperture_y_half=5.0))
    stack.add_surface(Surface(name="DET", thickness=0.0))

    opt_sys = OpticalSystem(stack)

    # Ray inside rectangular aperture
    r_pass = Ray(pos=(0, 0, 0), dir=(0.02, 0.02, 1.0))
    res_pass = opt_sys.trace_ray(r_pass)
    assert not res_pass.vignetted

    # Ray outside rectangular aperture (x_hit ~ 0.1 * 100 = 10 > 5)
    r_vignet = Ray(pos=(0, 0, 0), dir=(0.1, 0.0, 1.0))
    res_vignet = opt_sys.trace_ray(r_vignet)
    assert res_vignet.vignetted


def test_spider_aperture_nan_vignetting():
    surf = Surface(
        aperture_type="Spider",
        aperture_r_outer=25.0,
        aperture_center_radius=5.0,
        aperture_spider_width=2.0,
        aperture_spider_count=4
    )
    assert not surf.check_aperture(float('nan'), 0.0)
    assert not surf.check_aperture(0.0, float('nan'))
    assert not surf.check_aperture(float('nan'), float('nan'))
    assert not surf.check_aperture(float('inf'), 0.0)


def test_aperture_string_parsing_safety():
    surf = Surface(aperture_type="Spider")
    surf.set_aperture_params_from_str("LEGS=nan")
    surf.set_aperture_params_from_str("LEGS=inf")

