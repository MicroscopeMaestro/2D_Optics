"""Regression tests for front-to-back lens flipping."""

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface


def _ac254_030_c_stack() -> SurfaceStack:
    stack = SurfaceStack()
    stack.surfaces.extend([
        Surface(name="OBJ", thickness=20.0, glass="AIR"),
        Surface(name="AC254-030-C_S1", radius=21.1, thickness=13.0, glass="N-BAF10"),
        Surface(name="AC254-030-C_S2", radius=-15.2, thickness=1.8, glass="N-SF6HT"),
        Surface(name="AC254-030-C_S3", radius=-71.1, thickness=22.2, glass="AIR"),
    ])
    stack._update_accumulated_positions()
    return stack


def _optical_properties(stack: SurfaceStack):
    return [
        (surface.radius, surface.thickness, surface.glass)
        for surface in stack.surfaces[1:4]
    ]


def test_flip_doublet_reverses_geometry_materials_and_preserves_air_gap():
    stack = _ac254_030_c_stack()

    assert stack.flip_element(1) == [1, 2, 3]
    assert _optical_properties(stack) == [
        (71.1, 1.8, "N-SF6HT"),
        (15.2, 13.0, "N-BAF10"),
        (-21.1, 22.2, "AIR"),
    ]


def test_flip_doublet_twice_restores_original_prescription():
    stack = _ac254_030_c_stack()
    original = _optical_properties(stack)

    stack.flip_element(2)
    stack.flip_element(2)

    assert _optical_properties(stack) == original


def test_flip_planar_non_lens_reports_no_change():
    stack = SurfaceStack()
    stack.surfaces.append(Surface(name="Aperture_Stop", radius=0.0, glass="AIR"))

    assert stack.flip_element(0) == []
