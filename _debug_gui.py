import sys
import time
import os

def log(msg):
    print(f"[{time.time():.2f}] {msg}", flush=True)

log("Starting debug script...")
t0 = time.time()

log("Importing engine...")
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer

log("Importing GUI...")
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
log(f"QT_AVAILABLE = {QT_AVAILABLE}")

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])
    log(f"QApplication initialized: {_app}")

from optics_app.gui.main_window import MainWindow

log("Creating MainWindow...")
window = MainWindow()
log("MainWindow created.")

stack = SurfaceStack()
stack.add_surface(Surface(name="OBJ", thickness=1000.0))
stack.add_surface(Surface(name="L1_Front", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
s2 = Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0)
stack.add_surface(s2)
stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

log("Setting surface stack...")
window.model.set_surface_stack(stack)
log("Surface stack set.")

log("Calling on_optimize...")
result = window.on_optimize(variable_indices=[2], method="Nelder-Mead", max_iter=60)
log(f"on_optimize finished: success={result.success}, initial_rms={result.initial_rms}, final_rms={result.final_rms}, reduction={result.rms_reduction_pct}%")

log("Testing on_optimize_system...")
window.on_optimize_system()
log("on_optimize_system finished.")

log("All done successfully!")

