from PyQt5.QtCore import Qt
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.engine.surface import Surface, SurfaceType
model = LensDataTableModel()
s = Surface(name="M1", surface_type=SurfaceType.STANDARD, glass="MIRROR", radius=100.0)
model._surfaces = [s]
model._data = [["M1", "Standard", 100.0, 0.0, "MIRROR", 10.0]]
model._component_name_mode = False

idx = model.index(0, 2)
print("Flags:", int(model.flags(idx)))
print("SetData returns:", model.setData(idx, 50.0, Qt.EditRole))
print("New Data:", model._data[0][2])
