import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.gui.qt_compat import QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from optics_app.engine.lde import SurfaceStack
from optics_app.gui.visualizer_3d import Visualizer3DWidget
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
wls = res.get('wavelengths', [1.087, 1.225, 1.35])

sys_opt = OpticalSystem(stack)
rays = [sys_opt.trace_ray(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=wl)) for wl in wls]

vis = Visualizer3DWidget()
vis.surface_stack = stack
vis.rays = rays
vis.update_layout()
print('Visualizer 3D update_layout executed successfully!')
