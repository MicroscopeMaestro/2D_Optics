import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.cad.exporter import _extract_surface_stack, SurfaceStack as ExpSurfaceStack
from optics_app.engine.lde import SurfaceStack as LdeSurfaceStack

print("ExpSurfaceStack id:", id(ExpSurfaceStack), "module:", ExpSurfaceStack.__module__)
print("LdeSurfaceStack id:", id(LdeSurfaceStack), "module:", LdeSurfaceStack.__module__)
print("Same class?", ExpSurfaceStack is LdeSurfaceStack)
