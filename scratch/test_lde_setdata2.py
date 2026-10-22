import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    from PyQt5.QtCore import Qt
except ImportError:
    from PySide6.QtCore import Qt

from optics_app.gui.lde_table import LensDataTableModel
from optics_app.engine.surface import Surface, SurfaceType

model = LensDataTableModel()
s = Surface(name="M1", surface_type=SurfaceType.STANDARD, glass="MIRROR", radius=100.0)
s.is_locked = False
model._surfaces = [s]
model._data = [["M1", "Standard", 100.0, 0.0, "MIRROR", 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
model._component_name_mode = False

idx = model.index(0, 2)
print("Before:", model._data[0][2])
res = model.setData(idx, 50.0, Qt.EditRole)
print("SetData returns:", res)
print("After:", model._data[0][2])
