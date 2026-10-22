"""
Acceptance Criterion 1 (AC1) Test Suite: Spectrometer JSON Model Ingestion.
Verifies loading dmd4500_nir_fingerprint_0p8mm_spectrometer.json into V2 components without crashing,
validating 17 surfaces, NIR band, asphere coefficients, grating specs, and 0.8mm detector.
"""

import os
import json
import pytest
import numpy as np

from optics_app.gui.qt_compat import QT_AVAILABLE
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.main_window import MainWindow
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget


def test_ac1_json_file_exists_and_valid_syntax(spectrometer_json_path, spectrometer_raw_data):
    """AC1.1: Verify JSON file exists, is readable, and contains schema version 3."""
    assert os.path.exists(spectrometer_json_path), "Prescription JSON must exist on disk"
    assert isinstance(spectrometer_raw_data, dict), "Root JSON must be an object"
    assert spectrometer_raw_data.get("version") == 3, "Schema version must be 3"
    assert "surfaces" in spectrometer_raw_data, "JSON must contain 'surfaces' list"


def test_ac1_surface_count_is_seventeen(loaded_spectrometer_stack):
    """AC1.2: Verify loaded SurfaceStack contains exactly 17 sequential surfaces."""
    assert len(loaded_spectrometer_stack.surfaces) == 17, (
        f"Expected 17 surfaces, got {len(loaded_spectrometer_stack.surfaces)}"
    )


def test_ac1_nir_wavelength_band_prescription(spectrometer_raw_data, loaded_spectrometer_stack):
    """AC1.3: Verify NIR wavelength band [1.10, 1.225, 1.35] um and primary wavelength 1.225 um."""
    wavelengths = spectrometer_raw_data.get("wavelengths", [])
    assert len(wavelengths) == 3, "Expected 3 NIR wavelengths"
    np.testing.assert_allclose(wavelengths, [1.10, 1.225, 1.35], atol=1e-4)

    primary_wl = spectrometer_raw_data.get("primary_wavelength")
    assert abs(primary_wl - 1.225) < 1e-4, f"Expected primary wavelength 1.225 um, got {primary_wl}"
    assert abs(loaded_spectrometer_stack.primary_wavelength - 1.225) < 1e-4


def test_ac1_grating_surface_parameters(loaded_spectrometer_stack):
    """AC1.4: Verify diffraction grating surface prescription (Surface 3)."""
    grating_surf = loaded_spectrometer_stack.surfaces[3]
    assert "Grating" in grating_surf.name or "Diffraction" in grating_surf.name
    assert grating_surf.surface_type == SurfaceType.GRATING
    assert grating_surf.is_reflective() is True or grating_surf.glass.upper() == "MIRROR"
    assert grating_surf.grating_lines_per_mm == 600.0, "Grating line density must be 600 lpm"
    assert grating_surf.grating_order == 1, "Grating diffraction order must be +1"
    assert abs(grating_surf.tilt_x - (-12.0)) < 1e-3, "Grating tilt_x must be -12.0 deg"
    assert abs(grating_surf.grating_primary_wavelength_um - 1.22) < 1e-2


def test_ac1_aspheric_lens_parameters(loaded_spectrometer_stack):
    """AC1.5: Verify custom aspheric condenser lens (Surface 12, A230-C)."""
    asph_surf = loaded_spectrometer_stack.surfaces[12]
    assert asph_surf.surface_type == SurfaceType.ASPHERIC
    assert asph_surf.name == "A230-C"
    assert "S-NPH1" in asph_surf.glass.upper()
    assert abs(asph_surf.radius - 3.478709) < 1e-4
    assert abs(asph_surf.conic - (-0.1263039)) < 1e-4
    assert len(asph_surf.asphere_coeffs) >= 5
    assert abs(asph_surf.asphere_coeffs[1] - (-0.0012606)) < 1e-6
    assert abs(asph_surf.asphere_coeffs[2] - (-0.000109)) < 1e-6
    assert abs(asph_surf.asphere_coeffs[3] - 3.2255631e-7) < 1e-10
    assert abs(asph_surf.asphere_coeffs[4] - (-7.8343862e-7)) < 1e-10


def test_ac1_single_pixel_0p8mm_detector(loaded_spectrometer_stack):
    """AC1.6: Verify single-pixel detector surface (Surface 16, 0.8mm clear aperture diameter)."""
    det_surf = loaded_spectrometer_stack.surfaces[16]
    assert det_surf.surface_type == SurfaceType.DETECTOR
    assert "Detector" in det_surf.name or "0.8mm" in det_surf.name
    # Semi-diameter 0.4 mm corresponds to full aperture 0.8 mm
    assert abs(det_surf.semi_diameter - 0.4) < 1e-4, (
        f"Detector semi-diameter must be 0.4 mm (0.8 mm full aperture), got {det_surf.semi_diameter}"
    )


def test_ac1_optical_components_material_mapping(loaded_spectrometer_stack):
    """AC1.7: Verify catalog optical materials mapped correctly across all refractive surfaces."""
    expected_materials = {
        0: "AIR",
        1: "AIR",
        2: "MIRROR",
        3: "MIRROR",
        4: "N-LAK22",
        5: "N-SF6HT",
        6: "AIR",
        7: "MIRROR",
        8: "MIRROR",
        9: "N-BAF10",
        10: "N-SF6HT",
        11: "AIR",
        12: "S-NPH1_MOLD",
        13: "AIR",
        14: "BK7",
        15: "AIR",
        16: "AIR",
    }
    for idx, expected_glass in expected_materials.items():
        surf_glass = loaded_spectrometer_stack.surfaces[idx].glass.upper()
        assert expected_glass in surf_glass, (
            f"Surface {idx} ({loaded_spectrometer_stack.surfaces[idx].name}) glass mismatch: "
            f"expected '{expected_glass}', got '{surf_glass}'"
        )


def test_ac1_lde_table_model_loading(loaded_spectrometer_stack, qapp):
    """AC1.8: Verify LensDataTableModel initializes from stack with 17 rows."""
    model = LensDataTableModel(stack=loaded_spectrometer_stack)
    assert model.rowCount() == 17
    stack = model.get_surface_stack()
    assert len(stack.surfaces) == 17
    assert stack.surfaces[3].surface_type == SurfaceType.GRATING
    assert stack.surfaces[12].surface_type == SurfaceType.ASPHERIC
    assert stack.surfaces[16].surface_type == SurfaceType.DETECTOR


def test_ac1_ui_main_window_loading(spectrometer_json_path, qapp):
    """AC1.9: Verify MainWindow loads reference spectrometer JSON without crashing."""
    window = MainWindow()
    assert window is not None
    loaded_ok = window.on_load_project(spectrometer_json_path)
    assert loaded_ok is True
    assert len(window.model.get_surface_stack().surfaces) == 17
    assert window.model.rowCount() == 17
    assert window.component_editor_dock is not None
    assert window.dual_view is not None
    assert window.analysis_tabs is not None
    assert window.optimizer_dock is not None
    window.close()


def test_ac1_dual_view_and_analysis_widget_loading(loaded_spectrometer_stack, qapp):
    """AC1.10: Verify DualViewWidget and AnalysisTabsWidget accept spectrometer stack cleanly."""
    model = LensDataTableModel(stack=loaded_spectrometer_stack)
    dual_view = DualViewWidget(model=model)
    dual_view.refresh_layout()
    assert dual_view.model is not None

    analysis_tabs = AnalysisTabsWidget(model=model)
    analysis_tabs.refresh_analysis(wavelengths=[1.10, 1.225, 1.35])
    assert analysis_tabs.count() >= 3


def test_ac1_roundtrip_json_serialization(loaded_spectrometer_stack, tmp_path):
    """AC1.11: Verify saving SurfaceStack to JSON and reloading preserves all 17 surfaces exactly."""
    save_path = str(tmp_path / "saved_spectrometer.json")
    loaded_spectrometer_stack.save_to_json(save_path, wavelengths=[1.10, 1.225, 1.35])
    assert os.path.exists(save_path)

    reloaded = SurfaceStack.load_from_json(save_path)
    reloaded_stack = reloaded["stack"] if isinstance(reloaded, dict) else reloaded
    assert len(reloaded_stack.surfaces) == 17

    for idx in range(17):
        orig = loaded_spectrometer_stack.surfaces[idx]
        reloaded_s = reloaded_stack.surfaces[idx]
        assert orig.name == reloaded_s.name
        assert orig.surface_type == reloaded_s.surface_type
        assert abs(orig.radius - reloaded_s.radius) < 1e-4
        assert abs(orig.thickness - reloaded_s.thickness) < 1e-4
        assert orig.glass.upper() == reloaded_s.glass.upper()
        assert abs(orig.semi_diameter - reloaded_s.semi_diameter) < 1e-4
        assert abs(orig.tilt_x - reloaded_s.tilt_x) < 1e-4
        assert abs(orig.tilt_y - reloaded_s.tilt_y) < 1e-4
        assert abs(orig.tilt_z - reloaded_s.tilt_z) < 1e-4
