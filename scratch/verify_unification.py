import json
from optics_app.engine.surface import Surface
from optics_app.gui.visualizer_3d import build_rectangular_aperture_geometry, build_aperture_stop_geometry
from optics_app.cad.exporter import create_aperture_stop_solid

# 1. Test decoupling
s = Surface(name="Test", semi_diameter=12.7, aperture_type="Rectangular", aperture_x_half=0.5, aperture_y_half=2.0)
assert s.semi_diameter == 12.7
s.set_aperture_params_from_str("Point")
assert s.semi_diameter == 12.7, "Failed decoupling on Point!"

# 2. Test 3D visualizer returns
faces, irim, orim, verts = build_rectangular_aperture_geometry(s)
assert faces[0][0][0] != faces[0][0][1], "Coordinates should be distinct"

# 3. Test CAD exporter
solid = create_aperture_stop_solid(s)
assert solid is not None

print("All unification checks passed.")
