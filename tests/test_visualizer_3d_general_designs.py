"""The 3D viewer must represent arbitrary supported surface stacks truthfully."""

import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.system_config import SystemConfig, WavelengthEntry
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.main_window import MainWindow
from optics_app.gui.visualizer_3d import (
    HAS_MPL_QT, Visualizer3DWidget, classify_surface_for_3d,
    diffraction_wavelength_colors, generate_3d_rays,
)


def _stack(*surfaces):
    stack = SurfaceStack()
    stack.surfaces = list(surfaces)
    stack._update_accumulated_positions()
    return stack


def test_visual_roles_do_not_assume_first_object_or_last_detector():
    mirror = Surface(name="Fold A", surface_type=SurfaceType.MIRROR,
                     glass="MIRROR", thickness=20.0, semi_diameter=8.0)
    terminal = Surface(name="Reference plane", surface_type=SurfaceType.STANDARD,
                       glass="AIR", semi_diameter=8.0)
    assert classify_surface_for_3d(mirror, 0, 2)["mirror"]
    assert not classify_surface_for_3d(mirror, 0, 2)["object"]
    assert not classify_surface_for_3d(terminal, 1, 2)["detector"]
    assert classify_surface_for_3d(Surface(name="Photodiode", surface_type=SurfaceType.DETECTOR), 0, 1)["detector"]


def test_standalone_ray_bundle_uses_project_wavelengths():
    stack = _stack(
        Surface(name="Emitter", surface_type=SurfaceType.OBJECT, thickness=20.0, semi_diameter=5.0),
        Surface(name="Sensor", surface_type=SurfaceType.DETECTOR, semi_diameter=5.0),
    )
    stack.primary_wavelength = 1.310
    rays = generate_3d_rays(stack, num_rings=1, rays_per_ring=3)
    assert {ray.wavelength for ray in rays} == {1.310}

    config = SystemConfig()
    config.wavelengths = [WavelengthEntry(wavelength_um=0.780), WavelengthEntry(wavelength_um=0.920)]
    stack.system_config = config.to_dict()
    rays = generate_3d_rays(stack, num_rings=1, rays_per_ring=3)
    assert {ray.wavelength for ray in rays} == {0.780, 0.920}


def test_source_free_design_gets_declared_preview_incident_rays():
    stack = _stack(
        Surface(name="Fold A", surface_type=SurfaceType.MIRROR,
                glass="MIRROR", thickness=20.0, semi_diameter=8.0),
        Surface(name="Reference plane", surface_type=SurfaceType.STANDARD,
                glass="AIR", semi_diameter=8.0),
    )
    rays = generate_3d_rays(stack, wavelengths=[0.850], num_rings=1, rays_per_ring=3)
    assert rays
    assert all(ray.preview_source_assumed for ray in rays)
    assert all(ray.launch_pos[2] < stack.surfaces[0].global_origin[2] for ray in rays)
    assert any(ray.history and ray.history[0]["surface_index"] == 0 for ray in rays)

    one_mirror = _stack(Surface(name="Mirror only", surface_type=SurfaceType.MIRROR,
                                glass="MIRROR", semi_diameter=8.0))
    assert generate_3d_rays(one_mirror, wavelengths=[0.850], num_rings=1, rays_per_ring=3)


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_non_spectrometer_layout_has_generic_labels_and_controls():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    stack = _stack(
        Surface(name="Fold A", surface_type=SurfaceType.MIRROR,
                glass="MIRROR", thickness=20.0, semi_diameter=8.0),
        Surface(name="Reference plane", surface_type=SurfaceType.STANDARD,
                glass="AIR", semi_diameter=8.0),
    )
    widget = Visualizer3DWidget(auto_refresh=False)
    try:
        widget.update_layout(surface_stack=stack, rays=[])
        labels = [item.get_text() for item in widget.ax.texts]
        assert any("Fold A" in label for label in labels)
        assert any("Reference plane" in label for label in labels)
        assert not any("Point Object" in label or "Detector" in label for label in labels)
        assert widget.btn_grating_fan.isHidden()
        assert widget.btn_dmd_state.isHidden()
        assert widget.ax.get_xlabel() == "Global Z (mm)"
        assert widget.ax.collections  # Both mirror solid and terminal reference plane are drawn.
    finally:
        widget.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_custom_dmd_in_3d_uses_declared_name_and_states(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dmd = Surface(
        name="Programmable Mirror", surface_type=SurfaceType.MIRROR, glass="MIRROR",
        mechanism_type="DMD", mechanical_state="A", thickness=15.0,
        aperture_type="Rectangular", aperture_x_half=3.0, aperture_y_half=2.0,
        mechanical_states={
            "A": {"optical_tilt_offset_deg": [0.0, 0.0, 0.0], "route": "camera", "micromirror_tilt_deg": 7.0},
            "B": {"optical_tilt_offset_deg": [0.0, -14.0, 0.0], "route": "trap", "micromirror_tilt_deg": -7.0},
        },
    )
    stack = _stack(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=10.0),
                   dmd, Surface(name="Camera", surface_type=SurfaceType.DETECTOR))
    widget = Visualizer3DWidget(auto_refresh=False)
    try:
        widget.update_layout(surface_stack=stack, rays=[])
        assert not widget.btn_dmd_state.isHidden()
        assert "DMD: A" in widget.btn_dmd_state.text()
        labels = [item.get_text() for item in widget.ax.texts]
        assert any("Programmable Mirror [A]" in label for label in labels)
        assert not any("DLP4500" in label for label in labels)
        monkeypatch.setattr(widget, "trace_3d_rays", lambda: [])
        assert widget.toggle_dmd_state() == "B"
        assert dmd.mechanical_state == "B"
        widget.update_layout(surface_stack=stack, rays=[])
        assert "DMD: B" in widget.btn_dmd_state.text()
    finally:
        widget.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_non_grating_nir_design_still_separates_wavelengths():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    stack = _stack(
        Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0, semi_diameter=5.0),
        Surface(name="Receiver", surface_type=SurfaceType.DETECTOR, semi_diameter=5.0),
    )
    rays = OpticalSystem(stack).trace_rays([
        Ray(pos=(0, 0, -1), dir=(0, 0, 1), wavelength=1.10),
        Ray(pos=(0, 0, -1), dir=(0, 0, 1), wavelength=1.35),
    ])
    widget = Visualizer3DWidget(auto_refresh=False)
    try:
        widget.update_layout(surface_stack=stack, rays=rays)
        colors = diffraction_wavelength_colors([1.10, 1.35])
        assert len(set(colors.values())) == 2
        import matplotlib.colors as mcolors
        drawn_colors = []
        for k, coll in (widget._ray_line_colls.items() if isinstance(getattr(widget, "_ray_line_colls", None), dict) else []):
            drawn_colors.append(k[0])
            c = coll.get_colors()
            if len(c) > 0:
                drawn_colors.append(c[0])
        for coll in widget.ax.collections:
            if hasattr(coll, "get_colors"):
                c = coll.get_colors()
                if len(c) > 0:
                    drawn_colors.append(c[0])
        for line in widget.ax.lines:
            drawn_colors.append(line.get_color())
        assert all(any(np.allclose(mcolors.to_rgba(dc)[:3], mcolors.to_rgba(color)[:3]) for dc in drawn_colors) for color in colors.values())
        assert any("Ray wavelengths (false color)" in text.get_text()
                   for text in widget.ax.texts)
    finally:
        widget.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_main_window_uses_preview_incident_bundle_for_mirror_first_design(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    stack = _stack(
        Surface(name="Fold A", surface_type=SurfaceType.MIRROR,
                glass="MIRROR", thickness=20.0, semi_diameter=8.0),
        Surface(name="Receiver", surface_type=SurfaceType.DETECTOR, semi_diameter=8.0),
    )
    window = MainWindow()
    try:
        monkeypatch.setattr(window.analysis_tabs, "refresh_analysis", lambda **kwargs: True)
        window.model.set_surface_stack(stack)
        assert window.on_trace_rays()
        assert window._last_traced_rays
        assert all(getattr(ray, "preview_source_assumed", False)
                   for ray in window._last_traced_rays)
        assert window.viewer_tabs.tabText(window.viewer_tabs.indexOf(window.visualizer_3d)) == "3D Viewer"
        assert window.visualizer_3d.rays is window._last_traced_rays
    finally:
        window.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_multiple_grating_fans_have_individual_specs_and_cycle_focus():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    stack = _stack(
        Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=10.0),
        Surface(name="Grating A", surface_type=SurfaceType.GRATING, glass="MIRROR",
                grating_lines_per_mm=600.0, thickness=20.0, semi_diameter=10.0),
        Surface(name="Grating B", surface_type=SurfaceType.GRATING, glass="MIRROR",
                grating_lines_per_mm=300.0, thickness=20.0, semi_diameter=10.0),
    )
    ray = Ray(pos=(0, 0, -5), dir=(0, 0, 1), wavelength=0.850)
    ray.launch_pos = ray.pos.copy()
    ray.history = [
        {"surface_index": index, "x_global": float(surface.global_origin[0]),
         "y_global": float(surface.global_origin[1]), "z_global": float(surface.global_origin[2]),
         "dir": np.array([0.0, 0.0, 1.0]), "x_local": 0.0, "y_local": 0.0}
        for index, surface in enumerate(stack.surfaces)
    ]
    widget = Visualizer3DWidget(auto_refresh=False)
    try:
        widget.update_layout(surface_stack=stack, rays=[ray])
        labels = [item.get_text() for item in widget.ax.texts]
        assert any("2 grating(s)" in label for label in labels)
        assert any("600 l/mm" in label for label in labels)
        assert any("300 l/mm" in label for label in labels)
        assert widget.focus_grating_diffraction()
        assert widget._focused_grating_index == 1
        assert widget.focus_grating_diffraction()
        assert widget._focused_grating_index == 2
    finally:
        widget.close()
