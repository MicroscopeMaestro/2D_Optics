"""
Test Configuration and Fixtures for V2 Spectrometer Tuning & Design E2E Test Suite.
Configures headless Qt application and provides spectrometer model fixtures.
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np

# Ensure headless Qt execution
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

V2_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "v2"))
if V2_ROOT not in sys.path:
    sys.path.insert(0, V2_ROOT)

from optics_app.gui.qt_compat import QT_AVAILABLE, QApplication
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.source import SourceConfig, SourceType


@pytest.fixture(scope="session")
def qapp():
    """Headless QApplication fixture ensuring clean offscreen GUI lifecycle."""
    if not QT_AVAILABLE:
        pytest.skip("Qt bindings (PySide6/PyQt) not available in this environment.")
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture(scope="session")
def spectrometer_json_path():
    """Returns absolute path to the reference DMD4500 NIR spectrometer JSON prescription."""
    path = os.path.join(PROJECT_ROOT, "dmd4500_nir_fingerprint_0p8mm_spectrometer.json")
    if not os.path.exists(path):
        pytest.fail(f"Reference spectrometer JSON file not found at: {path}")
    return path


@pytest.fixture(scope="session")
def spectrometer_raw_data(spectrometer_json_path):
    """Loads and returns raw dictionary from spectrometer JSON."""
    with open(spectrometer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture
def loaded_spectrometer_stack(spectrometer_json_path):
    """Loads and returns fresh SurfaceStack instance from the reference JSON."""
    loaded = SurfaceStack.load_from_json(spectrometer_json_path)
    stack = loaded["stack"] if isinstance(loaded, dict) else loaded
    return stack


@pytest.fixture
def loaded_optical_system(loaded_spectrometer_stack):
    """Returns OpticalSystem initialized with the reference spectrometer SurfaceStack."""
    return OpticalSystem(loaded_spectrometer_stack)


@pytest.fixture
def sample_thorlabs_doublet_zmx(tmp_path):
    """Creates and yields a temporary valid Thorlabs AC254-050-B achromatic doublet .zmx file."""
    zmx_content = """\
VERS 120524 0 0
MODE SEQ
NAME Thorlabs AC254-050-B Achromatic Doublet
UNIT MM
WAVM 1 0.5876 1.0
WAVM 2 0.4861 1.0
WAVM 3 0.6563 1.0
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ 50.0
  DIAM 10.0
SURF 1
  TYPE STANDARD
  CURV 0.029411764705882353
  DISZ 6.0
  GLAS N-BK7 0 0 1.5168 64.17
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV -0.03597122302158273
  DISZ 2.0
  GLAS SF5 0 0 1.6727 32.21
  DIAM 12.7
SURF 3
  TYPE STANDARD
  CURV -0.008695652173913044
  DISZ 45.0
  DIAM 12.7
SURF 4
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
  DIAM 5.0
"""
    file_path = tmp_path / "AC254_050_B_sample.zmx"
    file_path.write_text(zmx_content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def sample_thorlabs_singlet_zmx(tmp_path):
    """Creates and yields a temporary valid Thorlabs LA1951 plano-convex singlet .zmx file."""
    zmx_content = """\
VERS 120524 0 0
MODE SEQ
NAME Thorlabs LA1951 Plano-Convex Lens
UNIT MM
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ 30.0
  DIAM 8.0
SURF 1
  TYPE STANDARD
  CURV 0.07633587786259542
  DISZ 8.8
  GLAS N-BK7 0 0 1.5168 64.17
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 20.0
  DIAM 12.7
SURF 3
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
  DIAM 4.0
"""
    file_path = tmp_path / "LA1951_sample.zmx"
    file_path.write_text(zmx_content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def sample_aspheric_zmx(tmp_path):
    """Creates and yields a temporary valid Zemax .zmx file with even aspheric surface."""
    zmx_content = """\
VERS 120524 0 0
MODE SEQ
NAME Aspheric Condenser Lens
UNIT MM
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ 10.0
  DIAM 5.0
SURF 1
  TYPE EVENASPH
  CURV 0.287463
  DISZ 2.94
  GLAS S-NPH1_MOLD 0 0 1.92 20.0
  CONI -0.1263039
  PARM 1 0.0
  PARM 2 -0.0012606
  PARM 3 -0.000109
  PARM 4 3.2255631e-07
  PARM 5 -7.8343862e-07
  DIAM 3.17
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 10.0
  DIAM 3.17
SURF 3
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
  DIAM 1.0
"""
    file_path = tmp_path / "asphere_sample.zmx"
    file_path.write_text(zmx_content, encoding="utf-8")
    return str(file_path)
