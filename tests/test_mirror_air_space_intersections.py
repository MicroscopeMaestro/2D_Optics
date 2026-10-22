"""Folded beams intersect optical faces once; spacing rows never reflect."""

import json
import numpy as np
import pytest

from optics_app.engine.diagnostics import ModelSanityValidator
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.mechanisms import available_states, is_dmd
from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.lde_table import LensDataTableModel


def folded_stack(gap_name="AIR_after_mirror", gap_type=SurfaceType.STANDARD):
    stack = SurfaceStack()
    stack.surfaces = [
        Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0),
        Surface(name="Fold", surface_type=SurfaceType.MIRROR, glass="MIRROR", tilt_x=45.0),
        Surface(name=gap_name, surface_type=gap_type, thickness=30.0),
        Surface(name="Detector", surface_type=SurfaceType.DETECTOR),
    ]
    stack._update_accumulated_positions()
    return stack


def point(hit):
    return np.array([hit["x_global"], hit["y_global"], hit["z_global"]])


@pytest.mark.parametrize("height", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("name,kind", [
    ("AIR_after_mirror", SurfaceType.STANDARD),
    ("AIR_after_DMD", SurfaceType.STANDARD),
    ("Spacing", SurfaceType.AIR_SPACE),
])
def test_mirror_hit_then_air_space_preserves_reflection(height, name, kind):
    stack = folded_stack(name, kind)
    incident = np.array([0.0, 0.0, 1.0])
    ray = OpticalSystem(stack).trace_ray(Ray(pos=np.array([0.0, height, 0.0]), dir=incident))
    assert not ray.vignetted
    assert [h["surface_index"] for h in ray.history] == [0, 1, 2, 3]
    mirror, gap, detector = ray.history[1:]
    normal = stack.surfaces[1].global_R[:, 2]
    assert np.dot(point(mirror) - stack.surfaces[1].global_origin, normal) == pytest.approx(0.0, abs=1e-10)
    assert np.allclose(mirror["dir"], incident - 2 * np.dot(incident, normal) * normal)
    assert np.allclose(point(mirror), [0.0, height, 20.0 + height])
    assert gap["reference_only"] and gap["interaction"] == "air_space"
    assert np.allclose(point(gap), point(mirror))
    assert np.allclose(gap["dir"], mirror["dir"])
    assert np.allclose(gap["jones_vector"], mirror["jones_vector"])
    assert gap["phase"] == mirror["phase"]
    assert gap["intensity"] == mirror["intensity"]
    assert np.allclose(point(detector), [0.0, 30.0, 20.0 + height])
    assert ray.path_length == pytest.approx(50.0)
    for before, after in zip(ray.history, ray.history[1:]):
        assert np.dot(point(after) - point(before), before["dir"]) >= -1e-9


def test_air_alias_is_neither_mirror_nor_dmd():
    for name in ("AIR_after_mirror", "AIR_after_DMD", "AIR_after_grating"):
        gap = Surface(name=name)
        assert gap.is_air_space
        assert not gap.is_reflective()
        assert not gap.is_mirror()
        assert not is_dmd(gap)
        assert available_states(gap) == {}
    assert not Surface(name="Lens rear", radius=-30.0, glass="AIR").is_air_space
    assert not Surface(name="Reference plane", glass="AIR").is_air_space
    assert not Surface(name="AIR_after_DMD", glass="N-BK7").is_air_space


def test_physical_plane_behind_reflected_ray_is_rejected_and_warned():
    stack = folded_stack("Reference plane")
    ray = OpticalSystem(stack).trace_ray(Ray(pos=np.array([0.0, 1.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])))
    assert ray.vignetted
    assert ray.history[-1]["failure_reason"] == "no_forward_intersection"
    assert np.allclose(ray.pos, [0.0, 1.0, 21.0])
    report = ModelSanityValidator.inspect_trace(stack, [ray])
    assert any(item.code == "NO_FORWARD_SURFACE_INTERSECTION" and item.surface_index == 2 for item in report.items)


def test_flat_surface_intersection_is_forward_and_allows_coincident_hits():
    plane = SphericalSurface(radius=0.0)
    assert plane.intersect(Ray(pos=np.array([0.0, 0.0, 1.0]), dir=np.array([0.0, 0.0, 1.0]))) is None
    hit, _ = plane.intersect(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, -1.0])))
    assert np.allclose(hit, [0.0, 0.0, 0.0])


def test_legacy_gap_load_preserves_indices_spacing_and_uids(tmp_path):
    stack = folded_stack("AIR_after_DMD")
    original_uids = [s.component_uid for s in stack.surfaces]
    path = tmp_path / "fold.json"
    stack.save_to_json(str(path), wavelengths=[1.064])
    loaded = SurfaceStack.load_from_json(str(path))["stack"]
    assert len(loaded.surfaces) == 4
    assert loaded.surfaces[2].surface_type == SurfaceType.AIR_SPACE
    assert loaded.surfaces[2].thickness == 30.0
    assert [s.component_uid for s in loaded.surfaces] == original_uids
    assert np.allclose(loaded.surfaces[-1].global_origin, stack.surfaces[-1].global_origin)
    from optics_app.engine.models import SpectrometerSystem
    data = json.loads(path.read_text(encoding="utf-8"))
    data["viewer_settings"] = {"projection_mode": "ortho"}
    api_system = SpectrometerSystem.from_dict(data)
    assert api_system.surfaces[2].surface_type == SurfaceType.AIR_SPACE
    assert api_system.to_dict()["viewer_settings"] == data["viewer_settings"]


def test_lde_edits_keep_spacing_type_and_explicit_mirror_type():
    model = LensDataTableModel()
    stack = folded_stack("Spacing", SurfaceType.AIR_SPACE)
    stack.surfaces[1].glass = "AIR"
    model.set_surface_stack(stack)
    assert model.setData(2, 3, 35.0)
    surfaces = model.get_surface_stack().surfaces
    assert surfaces[2].surface_type == SurfaceType.AIR_SPACE
    assert surfaces[1].surface_type == SurfaceType.MIRROR
    assert surfaces[1].is_reflective()


def test_air_space_is_not_exported_as_a_second_dmd_body():
    from optics_app.cad.exporter import get_exportable_elements
    elements = get_exportable_elements(folded_stack("AIR_after_DMD"))
    assert not any(2 in element["surface_indices"] for element in elements)
    assert any(1 in element["surface_indices"] for element in elements)


def test_desktop_load_renders_lens_edges_after_multiple_air_spaces(tmp_path, monkeypatch):
    from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
    from optics_app.gui.dual_view import HAS_MPL_QT
    from optics_app.gui.main_window import MainWindow, QMessageBox
    from optics_app.engine.system_config import SystemConfig
    if not (QT_AVAILABLE and HAS_MPL_QT):
        pytest.skip("Qt Matplotlib unavailable")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    stack = SurfaceStack()
    stack.surfaces = [
        Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0),
        Surface(name="AIR_after_source"),
        Surface(name="Fold", surface_type=SurfaceType.MIRROR, glass="MIRROR", tilt_x=45.0),
        Surface(name="AIR_after_mirror", thickness=30.0),
        Surface(name="Lens Front", radius=30.0, glass="N-BK7", thickness=3.0, semi_diameter=5.0),
        Surface(name="Lens Rear", radius=-30.0, thickness=20.0, semi_diameter=5.0),
        Surface(name="Detector", surface_type=SurfaceType.DETECTOR),
    ]
    stack.primary_wavelength = 0.800
    stack._update_accumulated_positions()
    config = SystemConfig()
    config.wavelength_list_um = [0.800]
    config.num_rays = 3
    config.beam_type = "Collimated (Parallel Beam)"
    config.aperture_value = 2.0
    path = tmp_path / "spectrometer_manual.json"
    stack.save_to_json(str(path), system_config=config)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    window = MainWindow()
    try:
        # Both projections connect successive physical lens faces, while the
        # prescription keeps every spacing row and its original index.
        window.dual_view.xz_view_active = True
        window.show()
        assert window.on_load_project(str(path))
        assert errors == []
        assert len(window.model.get_surface_stack().surfaces) == 7
        assert window.current_file_path == str(path)
        loaded = window.model.get_surface_stack()
        for axis, color, transverse in ((window.dual_view.ax_rz, "darkblue", 1),
                                        (window.dual_view.ax_xz, "darkgreen", 0)):
            edges = [line for line in axis.lines if line.get_color() == color]
            assert len(edges) == 2
            for sign, edge in zip((-1, 1), edges):
                expected = []
                for surface in loaded.surfaces[4:6]:
                    local = np.zeros(3)
                    local[transverse] = sign * surface.semi_diameter
                    local[2] = surface.sag(surface.semi_diameter)
                    world = surface.global_origin + surface.global_R @ local
                    expected.append([world[2], world[transverse]])
                actual = np.column_stack((edge.get_xdata(), edge.get_ydata()))
                assert np.allclose(actual, expected)
        angles = [label.get_text() for label in window.dual_view.ax_rz.texts
                  if "Fold=" in label.get_text()]
        assert any("Fold=90.0°" in label for label in angles)
    finally:
        window.close()
