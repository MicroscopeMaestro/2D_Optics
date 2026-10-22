import os
import json
import pytest
from v2.gui.qt_compat import QtWidgets
from v2.gui.main_window import MainWindow

os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])

win = MainWindow()
project_file = "spectrometer_raman_50mm_100mm_manual.json"
success = win.on_load_project(project_file)
print("Load success:", success)

stack = win.model.get_surface_stack()
print("Surfaces:", len(stack.surfaces))
print("Last traced rays:", len(win._last_traced_rays))
print("Wavelengths:", win.system_config.wavelength_list_um)

# Check 2D canvas refresh
try:
    if hasattr(win, "canvas_2d"):
        win.canvas_2d.draw()
        print("2D canvas draw: OK")
except Exception as e:
    print("2D canvas draw error:", e)

# Check 3D canvas refresh
try:
    if hasattr(win, "visualizer_3d"):
        win.visualizer_3d.update()
        print("3D visualizer update: OK")
except Exception as e:
    print("3D visualizer error:", e)

# Check analysis tabs
try:
    if hasattr(win, "analysis_dock"):
        print("Analysis dock: OK")
except Exception as e:
    print("Analysis dock error:", e)
