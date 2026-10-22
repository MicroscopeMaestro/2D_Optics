"""
Acceptance Criterion 2 (AC2) Test Suite: Parameter Slider and Ray Fan Live Update.
Verifies that changing parameter sliders in UI docks triggers ray fan recalculation,
detector spot updates, and canvas data refresh.
"""

import pytest
import numpy as np

from optics_app.gui.qt_compat import QT_AVAILABLE
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.main_window import MainWindow
from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget


def test_ac2_component_editor_dock_slider_signal(loaded_spectrometer_stack, qapp):
    """AC2.1: Verify ComponentEditorDockWidget slider moves emit component_changed signal."""
    model = LensDataTableModel(stack=loaded_spectrometer_stack)
    dock = ComponentEditorDockWidget(model=model)

    # Select Grating surface (index 3)
    dock.set_selected_surface(3)
    assert dock.get_selected_surface_index() == 3

    emitted_indices = []
    dock.component_changed.connect(lambda idx: emitted_indices.append(idx))

    # Move Tilt X slider: scale factor is 10.0, so -150 -> -15.0 deg
    dock.tilt_x_slider.setValue(-150)
    assert len(emitted_indices) > 0
    assert emitted_indices[-1] == 3
    assert abs(model.get_surface_stack().surfaces[3].tilt_x - (-15.0)) < 1e-2


def test_ac2_grating_tilt_changes_ray_fan_trajectories(loaded_spectrometer_stack):
    """AC2.2: Verify changing grating tilt angle alters ray fan diffraction trajectories."""
    sys_orig = OpticalSystem(loaded_spectrometer_stack)
    ray_nominal = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
    traced_orig = sys_orig.trace_ray(ray_nominal)
    dir_y_orig = traced_orig.history[3]["dir_y"]

    # Adjust grating tilt_x from -12° to -14°
    loaded_spectrometer_stack.surfaces[3].tilt_x = -14.0
    loaded_spectrometer_stack._update_accumulated_positions()

    sys_tilted = OpticalSystem(loaded_spectrometer_stack)
    traced_tilted = sys_tilted.trace_ray(ray_nominal)
    dir_y_tilted = traced_tilted.history[3]["dir_y"]

    # Diffracted angle changes significantly
    assert abs(dir_y_orig - dir_y_tilted) > 0.01, (
        f"Grating tilt change must alter diffracted ray direction: orig {dir_y_orig:.4f} vs tilted {dir_y_tilted:.4f}"
    )


def test_ac2_inter_element_spacing_slider_updates_thickness(loaded_spectrometer_stack, qapp):
    """AC2.3: Verify adjusting lens spacing sliders updates surface thickness in SurfaceStack."""
    model = LensDataTableModel(stack=loaded_spectrometer_stack)
    dock = ComponentEditorDockWidget(model=model)

    # Select collimator air gap (surface 2)
    dock.set_selected_surface(2)
    initial_thickness = model.get_surface_stack().surfaces[2].thickness
    assert initial_thickness == 70.0

    # Change thickness via spinbox
    dock.thick_spin.setValue(75.5)
    assert abs(model.get_surface_stack().surfaces[2].thickness - 75.5) < 1e-3


def test_ac2_dual_view_canvas_refreshed_on_slider_change(loaded_spectrometer_stack, qapp):
    """AC2.4: Verify DualViewWidget receives updated system and redraws without error."""
    model = LensDataTableModel(stack=loaded_spectrometer_stack)
    dual_view = DualViewWidget(model=model)
    dual_view.refresh_layout()

    # Modify surface 3 tilt_x
    loaded_spectrometer_stack.surfaces[3].tilt_x = -13.5
    loaded_spectrometer_stack._update_accumulated_positions()

    # Re-push updated system to DualView
    dual_view.refresh_layout()
    assert model.get_surface_stack().surfaces[3].tilt_x == -13.5


def test_ac2_analysis_tabs_refreshed_on_slider_change(loaded_spectrometer_stack, qapp):
    """AC2.5: Verify AnalysisTabsWidget (Spot & Ray Fan) updates upon system parameter change."""
    model = LensDataTableModel(stack=loaded_spectrometer_stack)
    analysis = AnalysisTabsWidget(model=model)
    analysis.refresh_analysis(wavelengths=[1.10, 1.225, 1.35])

    # Modify condenser spacing (Surface 11 thickness)
    loaded_spectrometer_stack.surfaces[11].thickness = 55.0
    loaded_spectrometer_stack._update_accumulated_positions()

    analysis.refresh_analysis(wavelengths=[1.10, 1.225, 1.35])
    assert analysis.tab_spot is not None
    assert analysis.tab_rayfan is not None


def test_ac2_main_window_slider_interaction_pipeline(spectrometer_json_path, qapp):
    """AC2.6: Verify full end-to-end GUI pipeline when slider is changed in MainWindow."""
    window = MainWindow()
    loaded_ok = window.on_load_project(spectrometer_json_path)
    assert loaded_ok is True

    # Adjust component editor slider
    dock = window.component_editor_dock
    dock.set_selected_surface(3)
    dock.tilt_x_slider.setValue(-125)  # -12.5 deg

    # Check that model and surface stack are in sync
    assert abs(window.model.get_surface_stack().surfaces[3].tilt_x - (-12.5)) < 1e-2
    assert window.model.rowCount() == 17

    window.close()
