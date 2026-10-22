"""
Milestone M1 Adversarial Stress Test Suite (Challenger 2).
Adversarially challenges and stress-tests:
1. Surface dataclass variable thickness extensions (thickness_variable, thickness_min, thickness_max)
2. Edge case bounds: negative, zero, inverted, infinite (inf/-inf), and NaN
3. Dataclass copy, deepcopy, replace, and equality (__eq__) floating point tolerances
4. SurfaceStack JSON serialization, legacy (v1) deserialization, and malformed/corrupt JSON handling
"""

import os
import sys
import copy
import math
import json
import tempfile
import dataclasses
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.lde import SurfaceStack, LensDataEditor
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem


# ===========================================================================
# Group 1: Variable Thickness Attribute Defaults and Edge Case Bounds
# ===========================================================================

def test_surface_variable_thickness_defaults():
    """Verify Surface default values for variable thickness attributes."""
    s = Surface()
    assert s.thickness_variable is False
    assert s.thickness_min == 0.0
    assert s.thickness_max == 1000.0


def test_surface_variable_thickness_custom_values():
    """Verify Surface custom initialization for variable thickness attributes."""
    s = Surface(
        name="CustomSurf",
        thickness=15.0,
        thickness_variable=True,
        thickness_min=2.5,
        thickness_max=75.0
    )
    assert s.thickness_variable is True
    assert s.thickness_min == 2.5
    assert s.thickness_max == 75.0


def test_surface_negative_thickness_bounds():
    """Verify Surface behavior with negative thickness bounds."""
    s = Surface(
        name="NegBounds",
        thickness=5.0,
        thickness_variable=True,
        thickness_min=-10.0,
        thickness_max=-1.0
    )
    assert s.thickness_min == -10.0
    assert s.thickness_max == -1.0
    assert s == copy.copy(s)


def test_surface_zero_bounds():
    """Verify Surface behavior with zero bounds (min=0.0, max=0.0)."""
    s = Surface(
        name="ZeroBounds",
        thickness=0.0,
        thickness_variable=True,
        thickness_min=0.0,
        thickness_max=0.0
    )
    assert s.thickness_min == 0.0
    assert s.thickness_max == 0.0
    assert s == copy.copy(s)


def test_surface_inverted_bounds():
    """Verify Surface behavior when thickness_min > thickness_max."""
    s = Surface(
        name="Inverted",
        thickness=20.0,
        thickness_variable=True,
        thickness_min=100.0,
        thickness_max=10.0
    )
    assert s.thickness_min == 100.0
    assert s.thickness_max == 10.0
    assert s == copy.copy(s)


def test_surface_infinite_bounds_behavior():
    """
    Examine Surface equality behavior when thickness_max or radius is infinity.
    Note: abs(inf - inf) = nan, and nan < 1e-9 is False in Python/IEEE 754.
    """
    s1 = Surface(name="InfSurf", thickness_max=float('inf'))
    s2 = Surface(name="InfSurf", thickness_max=float('inf'))
    
    # Direct identity is True
    assert s1 is s1
    assert s1 == s1
    
    # Comparing two distinct instances with inf evaluates abs(inf - inf) = nan
    # This documents the current implementation finding:
    is_equal = (s1 == s2)
    # Finding documented: s1 == s2 is False when thickness_max = inf due to abs(inf - inf) == nan
    assert is_equal is False, "Documented behavior: abs(inf - inf) yields NaN, so __eq__ evaluates to False"


def test_surface_nan_bounds_behavior():
    """Verify Surface equality behavior with NaN bounds."""
    s = Surface(name="NaNSurf", thickness_min=float('nan'))
    s_copy = copy.copy(s)
    # Per IEEE 754, NaN != NaN
    assert not (s == s_copy)


# ===========================================================================
# Group 2: Dataclass Operations (Copy, Deepcopy, Replace, Equality)
# ===========================================================================

def test_surface_shallow_copy():
    """Verify shallow copy preserves variable thickness and shares mutable lists."""
    s = Surface(
        name="Original",
        thickness=10.0,
        thickness_variable=True,
        thickness_min=1.0,
        thickness_max=50.0,
        asphere_coeffs=[1e-4, 2e-6]
    )
    s_copy = copy.copy(s)
    assert s_copy == s
    assert s_copy is not s
    assert s_copy.thickness_variable is True
    assert s_copy.thickness_min == 1.0
    assert s_copy.thickness_max == 50.0

    # Shallow copy shares mutable list reference
    s_copy.asphere_coeffs.append(3e-8)
    assert len(s.asphere_coeffs) == 3


def test_surface_deepcopy():
    """Verify deepcopy produces completely independent objects."""
    s = Surface(
        name="Original",
        thickness=10.0,
        thickness_variable=True,
        thickness_min=1.0,
        thickness_max=50.0,
        asphere_coeffs=[1e-4, 2e-6]
    )
    s_deep = copy.deepcopy(s)
    assert s_deep == s
    assert s_deep is not s
    assert s_deep.asphere_coeffs is not s.asphere_coeffs

    s_deep.asphere_coeffs.append(5e-8)
    assert len(s.asphere_coeffs) == 2
    assert len(s_deep.asphere_coeffs) == 3


def test_surface_dataclass_replace():
    """Verify dataclasses.replace works seamlessly on variable thickness fields."""
    s = Surface(name="Base", thickness=10.0, thickness_variable=False, thickness_min=0.0, thickness_max=1000.0)
    s_modified = dataclasses.replace(s, thickness_variable=True, thickness_min=5.0, thickness_max=80.0)
    
    assert s_modified.thickness_variable is True
    assert s_modified.thickness_min == 5.0
    assert s_modified.thickness_max == 80.0
    assert s.thickness_variable is False
    assert s.thickness_min == 0.0
    assert s.thickness_max == 1000.0


def test_surface_equality_floating_point_epsilon():
    """Verify __eq__ floating point tolerance (1e-9 threshold) for all variable fields."""
    s_base = Surface(name="Base", thickness=10.0, thickness_variable=True, thickness_min=5.0, thickness_max=100.0)

    # Within 1e-9 tolerance -> Equal
    s_within_tol = Surface(
        name="Base",
        thickness=10.0,
        thickness_variable=True,
        thickness_min=5.0 + 1e-10,
        thickness_max=100.0 - 1e-10
    )
    assert s_base == s_within_tol

    # Outside 1e-9 tolerance on thickness_min -> Not Equal
    s_outside_min = Surface(
        name="Base",
        thickness=10.0,
        thickness_variable=True,
        thickness_min=5.0 + 2e-9,
        thickness_max=100.0
    )
    assert s_base != s_outside_min

    # Outside 1e-9 tolerance on thickness_max -> Not Equal
    s_outside_max = Surface(
        name="Base",
        thickness=10.0,
        thickness_variable=True,
        thickness_min=5.0,
        thickness_max=100.0 + 2e-9
    )
    assert s_base != s_outside_max

    # Different boolean flag -> Not Equal
    s_diff_flag = Surface(
        name="Base",
        thickness=10.0,
        thickness_variable=False,
        thickness_min=5.0,
        thickness_max=100.0
    )
    assert s_base != s_diff_flag


def test_surface_equality_type_safety():
    """Verify __eq__ safely returns False when compared with non-Surface types."""
    s = Surface(name="Base", thickness=10.0)
    assert (s == "Surface") is False
    assert (s == 123) is False
    assert (s == None) is False
    assert (s == [10.0]) is False
    assert (s == {"name": "Base"}) is False


# ===========================================================================
# Group 3: SurfaceStack JSON Serialization & Deserialization
# ===========================================================================

def test_surfacestack_json_roundtrip_v2():
    """Verify full save and load roundtrip with variable thickness attributes."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="LensFront", radius=50.0, thickness=6.0, glass="N-BK7"))
    stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=25.0, glass="AIR", thickness_variable=True, thickness_min=2.0, thickness_max=80.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp_path = tmp.name

    try:
        stack.save_to_json(tmp_path, wavelengths=[0.486, 0.587, 0.656])
        loaded = SurfaceStack.load_from_json(tmp_path)
        loaded_stack = loaded["stack"]

        assert len(loaded_stack.surfaces) == 4
        s_gap = loaded_stack.surfaces[2]
        assert s_gap.thickness_variable is True
        assert s_gap.thickness_min == 2.0
        assert s_gap.thickness_max == 80.0
        assert s_gap == stack.surfaces[2]
        assert loaded["wavelengths"] == [0.486, 0.587, 0.656]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_surfacestack_legacy_json_list_format():
    """Verify backward compatibility loading legacy v1 flat list JSON without variable fields."""
    legacy_list = [
        {"name": "OBJ", "is_active": True, "surface_type": "Standard", "radius": 0.0, "thickness": 100.0, "glass": "AIR"},
        {"name": "L1", "is_active": True, "surface_type": "Standard", "radius": 50.0, "thickness": 5.0, "glass": "N-BK7"},
        {"name": "DET", "is_active": True, "surface_type": "Detector", "radius": 0.0, "thickness": 0.0, "glass": "AIR"}
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(legacy_list, tmp)
        legacy_path = tmp.name

    try:
        loaded = SurfaceStack.load_from_json(legacy_path)
        loaded_stack = loaded["stack"]
        assert len(loaded_stack.surfaces) == 3
        for s in loaded_stack.surfaces:
            assert s.thickness_variable is False
            assert s.thickness_min == 0.0
            assert s.thickness_max == 1000.0
    finally:
        if os.path.exists(legacy_path):
            os.remove(legacy_path)


def test_surfacestack_legacy_json_v1_dict_format():
    """Verify backward compatibility loading legacy v1 dict JSON without variable fields."""
    legacy_dict = {
        "version": 1,
        "surfaces": [
            {"name": "S1", "thickness": 10.0, "radius": 40.0, "glass": "N-BK7"},
            {"name": "S2", "thickness": 50.0, "radius": -40.0, "glass": "AIR"}
        ],
        "wavelengths": [0.5876]
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(legacy_dict, tmp)
        v1_path = tmp.name

    try:
        loaded = SurfaceStack.load_from_json(v1_path)
        loaded_stack = loaded["stack"]
        assert len(loaded_stack.surfaces) == 2
        for s in loaded_stack.surfaces:
            assert s.thickness_variable is False
            assert s.thickness_min == 0.0
            assert s.thickness_max == 1000.0
    finally:
        if os.path.exists(v1_path):
            os.remove(v1_path)


def test_surfacestack_corrupt_json_unknown_keys():
    """
    Test behavior when loading JSON containing unrecognized/foreign keys.
    Currently, Surface(**d) raises TypeError on unexpected keyword arguments.
    """
    corrupt_dict = {
        "version": 2,
        "surfaces": [
            {"name": "S1", "thickness": 10.0, "NON_EXISTENT_KEY_123": True}
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(corrupt_dict, tmp)
        corrupt_path = tmp.name

    try:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            SurfaceStack.load_from_json(corrupt_path)
    finally:
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)
