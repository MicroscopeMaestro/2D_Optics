import json
from optics_app.engine.lde import SurfaceType, SurfaceStack
stack = SurfaceStack.load_from_json('fiber_spectrometer_setup.json')['stack']
print(f'Surface 4 type: {stack.surfaces[4].surface_type}')
print(f'Surface 4 type == SurfaceType.GRATING? {stack.surfaces[4].surface_type == SurfaceType.GRATING}')
print(f'is_grating check: {getattr(stack.surfaces[4], "surface_type", None) == 4}')
print(f'SurfaceType.GRATING enum value: {SurfaceType.GRATING.value}')
