"""Unit tests for 3D visualizer ray tracing, volumetric ray generation, and element labeling."""

import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.models import SpectrometerSystem
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.visualizer_3d import (
    HAS_MPL_QT,
    Visualizer3DWidget,
    generate_3d_rays,
    wavelength_to_color,
)
from optics_app.gui.lde_table import LensDataTableModel


if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])


def _build_test_optical_system() -> SurfaceStack:
    stack = SurfaceStack()
    stack.surfaces = [
        Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=25.0, glass="AIR", semi_diameter=10.0),
        Surface(name="Collimator_Front", radius=45.0, thickness=6.0, glass="N-BAF10", semi_diameter=12.7, group_name="Collimator"),
        Surface(name="Collimator_Cement", radius=-35.0, thickness=3.0, glass="N-SF6HT", semi_diameter=12.7, group_name="Collimator"),
        Surface(name="Collimator_Rear", radius=-120.0, thickness=20.0, glass="AIR", semi_diameter=12.7, group_name="Collimator"),
        Surface(name="ApertureStop", surface_type=SurfaceType.STOP, thickness=15.0, glass="AIR", semi_diameter=8.0),
        Surface(name="FoldMirror", surface_type=SurfaceType.MIRROR, thickness=25.0, glass="MIRROR", semi_diameter=12.0),
        Surface(name="Sensor", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=15.0),
    ]
    stack._update_accumulated_positions()
    return stack


def test_wavelength_to_color_mapping():
    assert wavelength_to_color(0.405) == "#3f51b5"
    assert wavelength_to_color(0.486) == "#1e88e5"
    assert wavelength_to_color(0.500) == "#00acc1"
    assert wavelength_to_color(0.546) == "#43a047"
    assert wavelength_to_color(0.5876) == "#f57c00"
    assert wavelength_to_color(0.656) == "#e53935"
    assert wavelength_to_color(0.785) == "#8e24aa"


def test_generate_3d_rays_produces_valid_trajectories():
    stack = _build_test_optical_system()
    rays = generate_3d_rays(stack, wavelengths=[0.486, 0.5876, 0.656], num_rings=1, rays_per_ring=4)

    assert len(rays) > 0
    # Check that rays have 3D coordinates in history
    valid_histories = 0
    for ray in rays:
        assert isinstance(ray, Ray)
        if len(ray.history) >= 2:
            valid_histories += 1
            for hit in ray.history:
                assert "x_global" in hit
                assert "y_global" in hit
                assert "z_global" in hit

    assert valid_histories > 0


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_visualizer_3d_renders_rays_and_element_labels():
    stack = _build_test_optical_system()
    model = LensDataTableModel(stack)
    widget = Visualizer3DWidget(model=model)

    try:
        # Trace rays and update 3D widget
        rays = generate_3d_rays(stack)
        widget.update_layout(surface_stack=stack, rays=rays)

        # Check 3D callout labels exist in widget.ax.texts
        assert len(widget.ax.texts) > 0
        label_texts = [t.get_text().strip() for t in widget.ax.texts]

        # Verify key elements are labeled
        assert any("Source" in txt or "Object" in txt for txt in label_texts)
        assert any("Collimator" in txt for txt in label_texts)
        assert any("Stop" in txt for txt in label_texts)
        assert any("Sensor" in txt or "Detector" in txt for txt in label_texts)

        # Check 3D ray lines exist in widget.ax.lines
        assert len(widget.ax.lines) > 0

        # Test toggling show_labels
        widget.set_show_labels(False)
        assert len(widget.ax.texts) == 0

        widget.set_show_labels(True)
        assert len(widget.ax.texts) > 0

        # Test toggling show_rays
        line_count_with_rays = len(widget.ax.lines)
        widget.set_show_rays(False)
        # Without rays, only label stems (if any) remain
        line_count_without_rays = len(widget.ax.lines)
        assert line_count_without_rays < line_count_with_rays

        widget.set_show_rays(True)
        assert len(widget.ax.lines) == line_count_with_rays

    finally:
        widget.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_visualizer_3d_trace_3d_rays_button_action():
    stack = _build_test_optical_system()
    model = LensDataTableModel(stack)
    widget = Visualizer3DWidget(model=model)

    try:
        widget.update_layout(surface_stack=stack, rays=[])
        # Manually invoke 3D ray trace method
        traced = widget.trace_3d_rays()
        assert len(traced) > 0
        assert len(widget.rays) == len(traced)
        assert len(widget.ax.lines) > 0
    finally:
        widget.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_main_window_propagates_rays_to_visualizer_3d():
    from optics_app.gui.main_window import MainWindow

    stack = _build_test_optical_system()
    window = MainWindow()
    try:
        window.model.set_surface_stack(stack)
        # Trace rays via main window
        success = window.on_trace_rays()
        assert success is True
        assert len(window._last_traced_rays) > 0
        assert len(window.model.get_rays()) == len(window._last_traced_rays)

        if HAS_MPL_QT and window.visualizer_3d is not None:
            assert len(window.visualizer_3d.rays) == len(window._last_traced_rays)
            # Verify 3D lines and labels are rendered
            assert len(window.visualizer_3d.ax.lines) > 0
            assert len(window.visualizer_3d.ax.texts) > 0
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_aperture_stop_dialog_circular_and_rectangular():
    from optics_app.gui.aperture_dialog import ApertureStopDialog

    stack = _build_test_optical_system()
    model = LensDataTableModel(stack)
    surf = stack.surfaces[1] # Collimator front

    dlg = ApertureStopDialog(surface=surf, surface_idx=1, model=model)
    # Set as stop
    dlg.chk_is_stop.setChecked(True)
    # Set as rectangular slit: X=8 mm, Y=4 mm
    dlg.combo_type.setCurrentIndex(1)
    dlg.spin_x_half.setValue(8.0)
    dlg.spin_y_half.setValue(4.0)

    res = dlg.apply_changes()
    assert res["is_stop"] is True
    assert res["aperture_type"] == "Rectangular"
    assert surf.is_stop is True
    assert surf.surface_type == SurfaceType.STOP
    assert surf.aperture_x_half == 8.0
    assert surf.aperture_y_half == 4.0
    assert "(STO)" in surf.name
    # Check model was synchronized
    assert model._data[1][1] == "Stop"
    assert model._data[1][14] == "Rectangular"
    assert "X=8.0" in model._data[1][15]
    dlg.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_aperture_stop_dialog_annular_and_spider():
    from optics_app.gui.aperture_dialog import ApertureStopDialog

    stack = _build_test_optical_system()
    model = LensDataTableModel(stack)
    surf = stack.surfaces[3]

    # 1. Annular
    dlg = ApertureStopDialog(surface=surf, surface_idx=3, model=model)
    dlg.combo_type.setCurrentIndex(2) # Annular
    dlg.spin_r_inner.setValue(3.0)
    dlg.spin_r_outer.setValue(12.0)
    dlg.apply_changes()
    assert surf.aperture_type == "Annular"
    assert surf.aperture_r_inner == 3.0
    assert surf.aperture_r_outer == 12.0
    assert surf.semi_diameter == 12.0

    # 2. Spider
    dlg.combo_type.setCurrentIndex(4) # Spider
    dlg.spin_spider_hub.setValue(4.5)
    dlg.spin_spider_legs.setValue(6)
    dlg.spin_radius.setValue(15.0)
    dlg.apply_changes()
    assert surf.aperture_type == "Spider"
    assert surf.aperture_center_radius == 4.5
    assert surf.aperture_spider_count == 6
    assert surf.semi_diameter == 15.0
    dlg.close()


def test_model_set_stop_and_insert_aperture_stop():
    stack = _build_test_optical_system()
    model = LensDataTableModel(stack)

    # Initially surface 4 is Stop
    assert model._surfaces[4].surface_type == SurfaceType.STOP

    # Set surface 1 as Stop
    res = model.set_stop_surface(1, is_stop=True)
    assert res is True
    assert model._surfaces[1].surface_type == SurfaceType.STOP
    assert model._surfaces[1].is_stop is True
    # Surface 4 should now be Standard (only one stop)
    assert model._surfaces[4].surface_type == SurfaceType.STANDARD
    assert model._surfaces[4].is_stop is False

    # Insert a new Aperture Stop surface after surface 2
    initial_count = len(model._surfaces)
    new_idx = model.insert_aperture_stop_after(2, semi_diameter=6.5, thickness=8.0)
    assert new_idx == 3
    assert len(model._surfaces) == initial_count + 1
    new_surf = model._surfaces[3]
    assert new_surf.surface_type == SurfaceType.STOP
    assert new_surf.is_stop is True
    assert new_surf.semi_diameter == 6.5
    assert new_surf.thickness == 8.0
    # And surface 1 should be converted back to standard
    assert model._surfaces[1].surface_type == SurfaceType.STANDARD


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_component_editor_dock_aperture_controls():
    from optics_app.gui.component_editor_dock import ComponentEditorDockWidget

    stack = _build_test_optical_system()
    model = LensDataTableModel(stack)
    dock = ComponentEditorDockWidget(model=model)

    try:
        # Select surface 1
        dock.set_selected_surface(1)
        assert dock._selected_index == 1

        # Toggle stop via checkbox
        dock.chk_is_stop.setChecked(True)
        assert model._surfaces[1].is_stop is True

        # Change semi-diameter via spinbox
        dock.spin_semi_diam.setValue(18.5)
        assert model._surfaces[1].semi_diameter == 18.5

        # Change aperture shape via combo
        dock.combo_aperture_type.setCurrentIndex(1) # Rectangular
        assert model._surfaces[1].aperture_type == "Rectangular"
    finally:
        dock.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_visualizer_3d_presets_aspect_and_interaction():
    stack = _build_test_optical_system()
    model = LensDataTableModel(stack)
    widget = Visualizer3DWidget(model=model)

    try:
        widget.update_layout(surface_stack=stack, rays=[])

        # 1. Verify camera preset angles
        widget.set_view_isometric()
        assert round(widget.ax.elev) == 20
        assert round(widget.ax.azim) == -60

        widget.set_view_top()
        assert round(widget.ax.elev) == 90
        assert round(widget.ax.azim) == -90

        widget.set_view_side()
        assert round(widget.ax.elev) == 0
        assert round(widget.ax.azim) == -90

        widget.set_view_front()
        assert round(widget.ax.elev) == 0
        assert round(widget.ax.azim) == 0

        widget.reset_view()
        assert round(widget.ax.elev) == 20
        assert round(widget.ax.azim) == -60

        # 2. Verify large grid box sizing & camera distance
        assert hasattr(widget.ax, '_dist')
        assert widget.ax._dist <= 7.5  # Enlarged viewport occupancy compared to mpl default 10.0

        # 3. Verify box aspect ratio modes
        widget.set_aspect_mode("wide")
        assert widget.aspect_mode == "wide"
        aspect_wide = widget.ax.get_box_aspect()
        assert aspect_wide[0] > aspect_wide[1]  # Z axis is elongated horizontally

        widget.set_aspect_mode("cubic")
        assert widget.aspect_mode == "cubic"
        aspect_cubic = widget.ax.get_box_aspect()
        assert np.allclose(aspect_cubic[0], aspect_cubic[1])

        widget.set_aspect_mode("physical")
        assert widget.aspect_mode == "physical"

        # 4. Verify smooth pan and zoom
        xlim_before = widget.ax.get_xlim3d()
        widget.pan_view(dx_pixels=25, dy_pixels=-15)
        xlim_after = widget.ax.get_xlim3d()
        assert xlim_after != xlim_before

        ylim_before = widget.ax.get_ylim3d()
        widget.zoom_view(0.85)  # Zoom in
        ylim_after = widget.ax.get_ylim3d()
        span_before = ylim_before[1] - ylim_before[0]
        span_after = ylim_after[1] - ylim_after[0]
        assert span_after < span_before

        # Fit view restores base limits
        widget.fit_view()
        assert widget.ax.get_xlim3d() == widget._base_xlim

        # 5. Verify projection toggle and grid toggle
        widget.toggle_projection()
        assert widget.projection_mode == "ortho"
        widget.toggle_projection()
        assert widget.projection_mode == "persp"

        widget.set_show_grid(False)
        assert widget.show_grid is False
        widget.set_show_grid(True)
        assert widget.show_grid is True

    finally:
        widget.close()

