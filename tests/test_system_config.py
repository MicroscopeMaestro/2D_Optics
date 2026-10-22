"""
Unit Tests for Zemax System Configuration & Explorer Panel.
Validates SystemConfig data model, wavelength presets, field presets,
aperture types, and SystemConfigDockWidget integration.
"""

import pytest
import numpy as np

from optics_app.engine.system_config import (
    SystemConfig, ApertureType, ApodizationType, FieldType,
    RayPatternType, WavelengthEntry, FieldPoint
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.system_config_dock import SystemConfigDockWidget

if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])


def test_system_config_defaults():
    cfg = SystemConfig()
    assert cfg.aperture_type == ApertureType.ENTRANCE_PUPIL_DIAMETER
    assert cfg.aperture_value == 25.4
    assert cfg.apodization_type == ApodizationType.UNIFORM
    assert len(cfg.wavelengths) == 3
    assert cfg.primary_wavelength_um == 0.58756
    assert len(cfg.fields) == 1
    assert cfg.fields[0].x == 0.0
    assert cfg.fields[0].y == 0.0


def test_wavelength_presets():
    cfg = SystemConfig()
    
    # 1. NIR 785 nm Raman preset
    cfg.apply_wavelength_preset("NIR 785 nm Raman")
    assert len(cfg.wavelengths) == 3
    assert pytest.approx(cfg.wavelengths[0].wavelength_um) == 0.785
    assert cfg.wavelengths[0].is_primary is True
    assert pytest.approx(cfg.wavelengths[2].wavelength_um) == 0.880

    # 2. Telecom C-Band preset
    cfg.apply_wavelength_preset("Telecom C-Band")
    assert len(cfg.wavelengths) == 3
    assert pytest.approx(cfg.wavelengths[1].wavelength_um) == 1.550
    assert cfg.wavelengths[1].is_primary is True

    # 3. HeNe Laser preset
    cfg.apply_wavelength_preset("HeNe Laser (632.8 nm)")
    assert len(cfg.wavelengths) == 1
    assert pytest.approx(cfg.wavelengths[0].wavelength_um) == 0.6328


def test_field_presets():
    cfg = SystemConfig()
    
    # 1. 3-Field Linear
    cfg.apply_field_preset("3-Field Linear", max_field=10.0)
    assert len(cfg.fields) == 3
    assert cfg.fields[0].y == 0.0
    assert pytest.approx(cfg.fields[1].y, rel=1e-3) == 7.07
    assert pytest.approx(cfg.fields[2].y) == 10.0

    # 2. 5-Field Cross
    cfg.apply_field_preset("5-Field Cross", max_field=5.0)
    assert len(cfg.fields) == 5

    # 3. 9-Field Grid
    cfg.apply_field_preset("9-Field Grid", max_field=6.0)
    assert len(cfg.fields) == 9


def test_config_serialization_roundtrip():
    cfg = SystemConfig()
    cfg.aperture_type = ApertureType.OBJECT_SPACE_NA
    cfg.aperture_value = 0.22
    cfg.apply_wavelength_preset("NIR 785 nm Raman")
    cfg.apply_field_preset("5-Field Cross", max_field=4.0)
    cfg.beam_type = "Collimated (Parallel Beam)"
    cfg.slit_width_um = 42.0
    cfg.apodization_type = ApodizationType.GAUSSIAN
    cfg.apodization_factor = 0.65
    cfg.telecentric_object_space = True
    cfg.fields[2].vignetting_factor = 0.37
    cfg.lens_units = "Centimeters"
    cfg.source_units = "Lumens"
    cfg.temperature_c = 31.5
    cfg.pressure_atm = 0.84
    cfg.ray_pattern = RayPatternType.SQUARE_GRID
    cfg.pupil_rings = 11
    cfg.num_rays = 87

    d = cfg.to_dict()
    assert d["aperture"]["type"] == "Object Space NA"
    assert d["aperture"]["value"] == 0.22
    assert len(d["wavelengths"]) == 3
    assert len(d["fields"]["points"]) == 5

    cfg2 = SystemConfig()
    cfg2.from_dict(d)
    assert cfg2.aperture_type == ApertureType.OBJECT_SPACE_NA
    assert cfg2.aperture_value == 0.22
    assert len(cfg2.wavelengths) == 3
    assert len(cfg2.fields) == 5
    assert cfg2.to_dict() == d


def test_config_serialization_preserves_empty_editors():
    cfg = SystemConfig()
    cfg.wavelengths = []
    cfg.fields = []

    restored = SystemConfig()
    restored.from_dict(cfg.to_dict())

    assert restored.wavelengths == []
    assert restored.fields == []


def test_project_load_rebinds_complete_system_config(tmp_path, monkeypatch):
    from optics_app.gui.main_window import MainWindow

    expected = SystemConfig()
    expected.aperture_type = ApertureType.OBJECT_CONE_ANGLE
    expected.aperture_value = 6.25
    expected.slit_width_um = 75.0
    expected.fields = [FieldPoint(x=1.5, y=-2.0, weight=0.7, vignetting_factor=0.2)]
    expected.lens_units = "Centimeters"
    expected.source_units = "Lumens"
    expected.temperature_c = 28.0
    expected.pressure_atm = 0.72
    expected.ray_pattern = RayPatternType.RANDOM_MONTE_CARLO
    expected.pupil_rings = 9
    expected.num_rays = 55

    project_path = tmp_path / "system-config-roundtrip.json"
    SurfaceStack().save_to_json(str(project_path), system_config=expected)

    window = MainWindow()
    monkeypatch.setattr(window, "on_trace_rays", lambda: None)
    try:
        # A conflicting previous-project value must not survive the load.
        window.system_config.temperature_c = -50.0
        assert window.on_load_project(str(project_path)) is True
        assert window.system_config.to_dict() == expected.to_dict()
        assert window.model.system_config is window.system_config
        assert window.system_config_dock.config is window.system_config
    finally:
        window.close()


def test_system_config_dock_widget_signals():
    cfg = SystemConfig()
    dock = SystemConfigDockWidget(cfg)
    
    signal_fired = []
    dock.system_updated.connect(lambda: signal_fired.append(True))
    
    # Trigger preset
    dock.combo_wl_preset.setCurrentText("NIR 785 nm Raman")
    assert len(signal_fired) > 0
    assert pytest.approx(cfg.wavelengths[0].wavelength_um) == 0.785

    # Trigger add wavelength
    dock._on_add_wavelength()
    assert len(cfg.wavelengths) == 4


def test_system_config_save_load_file_roundtrip(tmp_path):
    cfg = SystemConfig()
    cfg.aperture_type = ApertureType.IMAGE_SPACE_F_NUM
    cfg.aperture_value = 2.8
    cfg.beam_type = "Collimated (Parallel Beam)"
    cfg.slit_width_um = 25.0
    cfg.apply_wavelength_preset("Telecom C-Band")
    cfg.temperature_c = 23.5
    cfg.pressure_atm = 1.05

    cfg_file = tmp_path / "my_system_config.json"
    cfg.save_to_file(str(cfg_file))
    assert cfg_file.exists()

    cfg_loaded = SystemConfig()
    cfg_loaded.load_from_file(str(cfg_file))
    assert cfg_loaded.aperture_type == ApertureType.IMAGE_SPACE_F_NUM
    assert pytest.approx(cfg_loaded.aperture_value) == 2.8
    assert cfg_loaded.beam_type == "Collimated (Parallel Beam)"
    assert pytest.approx(cfg_loaded.slit_width_um) == 25.0
    assert pytest.approx(cfg_loaded.temperature_c) == 23.5
    assert pytest.approx(cfg_loaded.pressure_atm) == 1.05
    assert len(cfg_loaded.wavelengths) == 3


def test_system_config_dock_and_main_window_save_load(tmp_path, monkeypatch):
    from optics_app.gui.main_window import MainWindow

    window = MainWindow()
    monkeypatch.setattr(window, "on_trace_rays", lambda: None)
    try:
        # 1. Modify dock configuration
        window.system_config_dock.combo_beam_type.setCurrentIndex(window.system_config_dock.combo_beam_type.findData("Collimated (Parallel Beam)"))
        window.system_config_dock.spin_ap_val.setValue(12.5)
        window.system_config_dock.spin_temp.setValue(27.0)

        cfg_file = tmp_path / "saved_explorer_cfg.json"
        saved = window.on_save_system_config(str(cfg_file))
        assert saved == str(cfg_file)
        assert cfg_file.exists()

        # 2. Reset values
        window.system_config_dock.spin_ap_val.setValue(50.0)
        window.system_config_dock.spin_temp.setValue(10.0)

        # 3. Load via main window handler
        loaded = window.on_load_system_config(str(cfg_file))
        assert loaded == str(cfg_file)
        assert pytest.approx(window.system_config.aperture_value) == 12.5
        assert pytest.approx(window.system_config.temperature_c) == 27.0
        assert window.system_config.beam_type == "Collimated (Parallel Beam)"
    finally:
        window.close()
