from optics_app.gui.lde_table import LensDataTableModel
from optics_app.engine.surface import Surface, SurfaceType
model = LensDataTableModel()
s = Surface(name="M1", surface_type=SurfaceType.STANDARD, glass="MIRROR", is_locked=True)
model.set_surfaces([s])
print("is_lens_locked:", model.is_lens_locked(0))
