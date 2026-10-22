"""General optical projects keep their wavelength, pulse and view context."""

import json

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.system_config import SystemConfig
from optics_app.gui.main_window import MainWindow
from optics_app.gui.qt_compat import QtWidgets
from optics_app.gui.visualizer_3d import Visualizer3DWidget


def test_view_settings_round_trip_on_plain_surface_stack(tmp_path):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fold", surface_type=SurfaceType.MIRROR, glass="MIRROR"))
    stack.viewer_settings = {"projection_mode": "ortho", "color_mode": "field", "show_grid": False}
    path = tmp_path / "generic_design.json"
    stack.save_to_json(str(path), wavelengths=[0.800])
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["viewer_settings"] == stack.viewer_settings
    loaded = SurfaceStack.load_from_json(str(path))
    assert loaded["stack"].viewer_settings == stack.viewer_settings


def test_legacy_project_does_not_require_view_settings(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"version": 3, "surfaces": [], "wavelengths": [1.030]}), encoding="utf-8")
    loaded = SurfaceStack.load_from_json(str(path))
    assert loaded["viewer_settings"] is None


def test_ultrafast_project_starts_clean_and_survives_save_load(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(MainWindow, "on_trace_rays", lambda *args, **kwargs: True)
    window = MainWindow()
    try:
        window.system_config.apply_wavelength_preset("NIR 1064 nm Laser")
        window.source_config.wavelengths = "1.064"
        assert window.on_new_project(ultrafast=True)
        assert window.current_file_path is None
        assert window.system_config.wavelength_list_um == [0.800]
        assert window.system_config.primary_wavelength_um == 0.800
        assert window.source_config.wavelengths == "0.800"
        assert window.analysis_tabs.get_analysis_config()["pulse_width_analysis"]["center_wavelength_nm"] == 800.0

        window.visualizer_3d.set_view_settings({
            "projection_mode": "ortho", "color_mode": "field", "show_grid": False,
            "mesh_quality": "detailed", "aspect_mode": "physical",
            "camera_elevation_deg": 31.0, "camera_azimuth_deg": -24.0,
        })
        path = tmp_path / "ultrafast.json"
        assert window.on_save_project(str(path))
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert "two_photon_config" not in saved
        assert "nonlinear_imaging_config" not in saved
        assert "optical_routes" not in saved
        assert window.on_new_project()
        assert window.system_config.primary_wavelength_um != 0.800
        assert window.on_load_project(str(path))
        assert window.system_config.wavelength_list_um == [0.800]
        assert window.analysis_tabs.get_analysis_config()["pulse_width_analysis"]["center_wavelength_nm"] == 800.0
        settings = window.visualizer_3d.get_view_settings()
        assert settings["projection_mode"] == "ortho"
        assert settings["color_mode"] == "field"
        assert settings["show_grid"] is False
        assert settings["mesh_quality"] == "detailed"
        assert settings["aspect_mode"] == "physical"
        assert settings["camera_elevation_deg"] == 31.0
        assert settings["camera_azimuth_deg"] == -24.0
    finally:
        window.close()


def test_view_settings_reject_unknown_choices():
    viewer = Visualizer3DWidget(auto_refresh=False)
    try:
        viewer.set_view_settings({"aspect_mode": "invalid", "projection_mode": "invalid", "opaque": "yes"})
        settings = viewer.get_view_settings()
        assert settings["aspect_mode"] == "wide"
        assert settings["projection_mode"] == "persp"
        assert settings["opaque"] is False
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


def test_new_project_undo_restores_wavelength_source_and_view(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(MainWindow, "on_trace_rays", lambda *args, **kwargs: True)
    window = MainWindow()
    try:
        window.system_config.apply_wavelength_preset("NIR 1064 nm Laser")
        window.source_config.wavelengths = "1.064"
        window.visualizer_3d.set_view_settings({"projection_mode": "ortho", "show_grid": False})
        assert window.on_new_project(ultrafast=True)
        assert window.on_undo()
        assert window.system_config.primary_wavelength_um == 1.064
        assert window.source_config.wavelengths == "1.064"
        assert window.visualizer_3d.get_view_settings()["projection_mode"] == "ortho"
        assert window.visualizer_3d.get_view_settings()["show_grid"] is False
        assert window.on_redo()
        assert window.system_config.primary_wavelength_um == 0.800
    finally:
        window.close()


def test_ultrafast_design_traces_2d_at_selected_carrier():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        assert window.on_new_project(ultrafast=True)
        window.system_config.num_rays = 3
        stack = window.model.get_surface_stack()
        stack.add_surface(Surface(name="Laser output", surface_type=SurfaceType.OBJECT,
                                  thickness=20.0, semi_diameter=5.0))
        stack.add_surface(Surface(name="Focus reference", surface_type=SurfaceType.STANDARD,
                                  thickness=20.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="Sample plane", surface_type=SurfaceType.DETECTOR,
                                  semi_diameter=10.0))
        window.model.set_surface_stack(stack)
        assert window.on_trace_rays()
        assert window._last_traced_rays
        assert {ray.wavelength for ray in window._last_traced_rays} == {0.800}
        assert window.analysis_tabs.active_tab == "Femtosecond Pulse"
    finally:
        window.close()
