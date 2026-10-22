import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.visualizer_3d import Visualizer3DWidget

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
vis = Visualizer3DWidget()
print('Initial show_air_gaps default:', vis.show_air_gaps)
vis.update_layout(surface_stack=stack)
print('Layout updated successfully with air gaps active!')
