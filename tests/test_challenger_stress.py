"""
Adversarial Stress Test Suite for M6 (Complex Apertures, NaN/Inf handling, LDE Row Moves, String Parsing).
Created by challenger_m6_3_2_b.
"""
import math
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.lde_table import LensDataTableModel


def test_aperture_nan_inf_coordinates():
    aperture_types = ["Circular", "Rectangular", "Elliptical", "Annular", "Spider"]
    
    nan_inf_pairs = [
        (float('nan'), float('nan')),
        (float('inf'), float('inf')),
        (-float('inf'), -float('inf')),
        (float('nan'), 0.0),
        (0.0, float('nan')),
        (float('inf'), 0.0),
        (0.0, -float('inf')),
        (np.nan, np.nan),
        (np.inf, np.inf),
    ]

    for atype in aperture_types:
        surf = Surface(
            aperture_type=atype,
            semi_diameter=25.0,
            aperture_x_half=20.0,
            aperture_y_half=15.0,
            aperture_a_semi=20.0,
            aperture_b_semi=15.0,
            aperture_r_inner=5.0,
            aperture_r_outer=25.0,
            aperture_center_radius=5.0,
            aperture_spider_width=1.0,
            aperture_spider_count=4,
            aperture_spider_angle=0.0
        )

        # 1. Test NaN/Inf coordinates return False (vignetted)
        for x, y in nan_inf_pairs:
            assert surf.check_aperture(x, y) is False, f"{atype} failed to vignette x={x}, y={y}"

        # 2. Test valid rays return True
        if atype == "Circular":
            valid_x, valid_y = 5.0, 5.0
        elif atype == "Rectangular":
            valid_x, valid_y = 10.0, 10.0
        elif atype == "Elliptical":
            valid_x, valid_y = 5.0, 5.0
        elif atype == "Annular":
            valid_x, valid_y = 15.0, 0.0
        elif atype == "Spider":
            valid_x, valid_y = 15.0, 15.0

        assert surf.check_aperture(valid_x, valid_y) is True, f"{atype} vignetted valid ray at ({valid_x}, {valid_y})"


def test_lde_row_move_metadata_preservation():
    model = LensDataTableModel()

    # Insert extra rows to have 5 surfaces: OBJ (0), S1 (1), S2 (2), S3 (3), IMA (4)
    model.insertRows(1, 2)
    
    # Configure surface at index 1 with complex metadata
    s1 = model._surfaces[1]
    s1.asphere_coeffs = [1e-4, -2e-6, 3e-8]
    s1.cad_profile = {"mock_cad": True, "points": [(0, 0), (1, 1)]}
    s1.is_active = False
    s1.grating_order = -2
    s1.jones_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    s1.extinction_coeff = 0.05

    target_id = id(s1)

    def verify_surface(surf, desc):
        assert id(surf) == target_id, f"{desc}: Surface identity changed!"
        assert surf.asphere_coeffs == [1e-4, -2e-6, 3e-8], f"{desc}: asphere_coeffs altered"
        assert surf.cad_profile == {"mock_cad": True, "points": [(0, 0), (1, 1)]}, f"{desc}: cad_profile altered"
        assert surf.is_active is False, f"{desc}: is_active altered"
        assert surf.grating_order == -2, f"{desc}: grating_order altered"
        assert surf.jones_matrix is not None and np.array_equal(surf.jones_matrix, np.array([[1, 0], [0, -1]], dtype=complex)), f"{desc}: jones_matrix altered"

    # 1. Move row 1 down to row 2
    model.move_surface_down(1)
    verify_surface(model._surfaces[2], "Move row 1 down to 2")

    # 2. Move row 2 up to row 1
    model.move_surface_up(2)
    verify_surface(model._surfaces[1], "Move row 2 up to 1")

    # 3. Move row 1 to row 3 via move_row
    model.move_row(1, 3)
    verify_surface(model._surfaces[3], "Move row 1 to 3 via move_row")

    # 4. Verify surface stack synchronization
    stack = model.get_surface_stack()
    verify_surface(stack.surfaces[3], "Get surface stack after moves")


def test_string_parsing_safety():
    surf = Surface(
        aperture_type="Spider",
        aperture_center_radius=5.0,
        aperture_spider_count=4,
        aperture_spider_width=1.0,
        aperture_spider_angle=0.0
    )

    # Test LEGS=nan
    surf.set_aperture_params_from_str("LEGS=nan")
    assert isinstance(surf.aperture_spider_count, int) and surf.aperture_spider_count == 4

    # Test LEGS=inf
    surf.set_aperture_params_from_str("LEGS=inf")
    assert isinstance(surf.aperture_spider_count, int) and surf.aperture_spider_count == 4

    # Test RCNT=nan
    surf.set_aperture_params_from_str("RCNT=nan")
    # Verify parsing RCNT=nan does not crash
    assert hasattr(surf, "aperture_center_radius")

    # Test RCNT=inf
    surf.set_aperture_params_from_str("RCNT=inf")
    assert hasattr(surf, "aperture_center_radius")

    # Test malformed strings
    malformed = ["LEGS=abc", "=123", "LEGS=", "foo=bar", ",,,", "X=10, Y=invalid"]
    surf_rect = Surface(aperture_type="Rectangular", aperture_x_half=20.0, aperture_y_half=15.0)
    for s in malformed:
        surf_rect.set_aperture_params_from_str(s)
