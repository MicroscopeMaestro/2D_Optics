import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.gui.visualizer_3d import build_aperture_stop_geometry
import numpy as np

surf = Surface(name="Curved_Stop", surface_type=SurfaceType.STOP, radius=100.0, semi_diameter=12.7)
meshes, inner_rim, outer_rim = build_aperture_stop_geometry(surf)
print("Curved stop built successfully! Number of meshes:", len(meshes))
