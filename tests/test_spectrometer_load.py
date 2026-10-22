import os
import pytest
from v2.gui.qt_compat import QtWidgets
from v2.gui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_spectrometer_manual_design_load(qapp):
    win = MainWindow()
    project_file = "spectrometer_raman_50mm_100mm_manual.json"
    assert os.path.exists(project_file), f"Project file not found: {project_file}"

    success = win.on_load_project(project_file)
    assert success is True, "Failed to load spectrometer_raman_50mm_100mm_manual.json"

    stack = win.model.get_surface_stack()
    assert len(stack.surfaces) == 26, f"Expected 26 surfaces, got {len(stack.surfaces)}"
    assert len(win._last_traced_rays) > 0, "Expected traced rays to be populated"

    # Verify that wavelengths from system_config are loaded (Raman NIR band)
    wls = win.system_config.wavelength_list_um
    assert len(wls) == 3
    assert abs(wls[0] - 1.087) < 1e-3
    assert abs(wls[1] - 1.225) < 1e-3
    assert abs(wls[2] - 1.350) < 1e-3
