"""
Unit tests for Zemax OpticStudio-style solve status, variables, pickups, and table highlights.
"""
import pytest
import numpy as np

from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.qt_compat import Qt


def test_zemax_variable_display_and_highlight():
    """Verify that variable parameters display 'V' and return Zemax cyan background."""
    s1 = Surface(name="S1", radius=13.212, thickness=4.0, glass="N-BAF10")
    s2 = Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10")
    s3 = Surface(name="S3", radius=23.711, thickness=46.5, glass="AIR")
    s4 = Surface(name="IMA", radius=0.0, thickness=0.0, glass="AIR")

    stack = SurfaceStack()
    stack.surfaces = [s1, s2, s3, s4]

    model = LensDataTableModel(stack)

    # Initially, all fixed
    assert float(model.data(model.index(0, 2), 0)) == 13.212
    assert float(model.data(model.index(0, 3), 0)) == 4.0

    # Set S1 radius variable
    s1.radius_variable = True
    assert model.data(model.index(0, 2), 0) == "13.212 V"
    bg_brush = model.data(model.index(0, 2), 8) # BackgroundRole
    assert bg_brush is not None
    assert bg_brush.color().name().lower() == "#bae6fd"

    # Set S3 thickness variable (back focal distance)
    s3.thickness_variable = True
    assert model.data(model.index(2, 3), 0) == "46.500 V"
    bg_thic = model.data(model.index(2, 3), 8)
    assert bg_thic is not None
    assert bg_thic.color().name().lower() == "#bae6fd"


def test_zemax_pickup_and_solve_display():
    """Verify that pickup (P) and marginal ray solve (S) display correct badges and colors."""
    s1 = Surface(name="S1", radius=13.212, thickness=4.0, glass="N-BAF10")
    s2 = Surface(name="S2", radius=-13.212, thickness=2.0, glass="SF10")
    s2.radius_solve_type = "Pickup"
    s2.radius_pickup_surface = 0
    s2.radius_pickup_scale = -1.0

    s3 = Surface(name="S3", radius=23.711, thickness=46.5, glass="AIR")
    s3.thickness_solve_type = "MarginalRayFocus"

    stack = SurfaceStack()
    stack.surfaces = [s1, s2, s3]
    model = LensDataTableModel(stack)

    # Pickup badge and amber color
    assert "P" in model.data(model.index(1, 2), 0)
    bg_p = model.data(model.index(1, 2), 8)
    assert bg_p is not None
    assert bg_p.color().name().lower() == "#fef08a"

    # Ray solve badge and lavender color
    assert "S" in model.data(model.index(2, 3), 0)
    bg_s = model.data(model.index(2, 3), 8)
    assert bg_s is not None
    assert bg_s.color().name().lower() == "#e9d5ff"


def test_zemax_string_modifiers_in_setdata():
    """Verify that entering 'V' or 'F' via setData toggles variable flags."""
    s1 = Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7")
    stack = SurfaceStack()
    stack.surfaces = [s1]
    model = LensDataTableModel(stack)

    # Enter '55.0 V'
    model.setData(model.index(0, 2), "55.0 V", 2)
    assert s1.radius_variable is True
    assert abs(s1.radius - 55.0) < 1e-4

    # Enter '55.0 F' to lock
    model.setData(model.index(0, 2), "55.0 F", 2)
    assert s1.radius_variable is False

    # Enter '12.0 V' for thickness
    model.setData(model.index(0, 3), "12.0 V", 2)
    assert s1.thickness_variable is True
    assert abs(s1.thickness - 12.0) < 1e-4


def test_zemax_presets():
    """Verify bulk solve presets: zemax defaults, all radii variable, and all fixed."""
    s1 = Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7") # Solid lens
    s2 = Surface(name="S2", radius=-50.0, thickness=40.0, glass="AIR") # Air gap
    stack = SurfaceStack()
    stack.surfaces = [s1, s2]
    model = LensDataTableModel(stack)

    # 1. Zemax defaults: Lenses fixed, Air gaps variable
    model.set_zemax_defaults(var_air_gaps=True, fix_lenses=True)
    assert s1.thickness_variable is False
    assert s1.radius_variable is False
    assert s2.thickness_variable is True

    # 2. Make all radii variable (curvature optimization)
    model.make_all_radii_variable()
    assert s1.radius_variable is True
    assert s2.radius_variable is True

    # 3. Clear all variables (all fixed)
    model.set_all_fixed()
    assert s1.radius_variable is False
    assert s1.thickness_variable is False
    assert s2.radius_variable is False
    assert s2.thickness_variable is False
