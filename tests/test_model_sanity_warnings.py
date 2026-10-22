import numpy as np
import pytest

from optics_app.engine.diagnostics import ModelSanityValidator
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.mirror_positioner import MirrorPositioner
from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.tracer import OpticalSystem
from optics_app.ai.tools import inspect_optical_model_sanity


def _lens_stack(front_tilt_y=0.0):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=15.0))
    stack.add_surface(Surface(
        name="Collector_S1", radius=45.0, thickness=4.0, glass="N-BK7",
        semi_diameter=10.0, tilt_y=front_tilt_y, group_name="Collector"
    ))
    stack.add_surface(Surface(
        name="Collector_S2", radius=-45.0, thickness=30.0, glass="AIR",
        semi_diameter=10.0, group_name="Collector"
    ))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0))
    return stack


def test_geometry_sanity_detects_accidental_single_surface_lens_tilt():
    report = ModelSanityValidator.inspect_geometry(_lens_stack(front_tilt_y=45.0))
    codes = {item.code for item in report.items}
    assert "UNCOMMON_REFRACTIVE_SURFACE_TILT" in codes
    assert "LENS_GROUP_TILT_MISMATCH" in codes
    assert any(item.surface_index == 1 for item in report.items)


def test_geometry_sanity_allows_ordinary_aligned_lens():
    report = ModelSanityValidator.inspect_geometry(_lens_stack())
    assert report.items == []


def test_trace_sanity_reports_dominant_blocking_surface():
    stack = _lens_stack()
    rays = []
    for _ in range(3):
        ray = Ray(pos=np.zeros(3), dir=np.array([0.0, 0.0, 1.0]))
        ray.vignetted = True
        ray.history = [{"surface_index": 1, "vignetted": True}]
        rays.append(ray)

    report = ModelSanityValidator.inspect_trace(stack, rays)
    item = next(item for item in report.items if item.code == "NO_RAYS_REACH_DETECTOR")
    assert item.surface_index == 1
    assert "Collector_S1" in item.message


def test_mirror_fold_preset_rejects_refractive_surface():
    stack = _lens_stack()
    with pytest.raises(ValueError, match="only be applied to reflective surfaces"):
        MirrorPositioner.set_mirror_fold_angle(stack, 1, 90.0, "XZ")
    assert stack.surfaces[1].tilt_y == 0.0


def test_mirror_fold_preset_still_accepts_mirror():
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", thickness=10.0))
    stack.add_surface(Surface(name="Fold Mirror", glass="MIRROR"))
    tx, ty = MirrorPositioner.set_mirror_fold_angle(stack, 1, 90.0, "XZ")
    assert tx == 0.0
    assert ty == 45.0


def test_ai_sanity_tool_exposes_geometry_warning():
    output = inspect_optical_model_sanity(system=OpticalSystem(_lens_stack(front_tilt_y=45.0)))
    assert "MODEL SANITY WARNINGS" in output
    assert "UNCOMMON_REFRACTIVE_SURFACE_TILT" in output
    assert "Collector_S1" in output
