import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.visualizer_3d import Visualizer3DWidget, compute_surface_groups, classify_surface_for_3d
from optics_app.gui.qt_compat import QtWidgets

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
vis = Visualizer3DWidget()
vis.surface_stack = stack

# Let's inspect what runs in update_layout for solid rendering
surfaces = stack.surfaces
print('Total surfaces:', len(surfaces))

# Check which index is skipped or rendered as air
for index, front in enumerate(surfaces[:-1]):
    rear = surfaces[index + 1]
    role = classify_surface_for_3d(front, index, len(surfaces))
    is_air = (
        vis.show_air_gaps
        and (getattr(front, 'glass', '').upper() in ('AIR', '', 'NONE')
             or role.get('mirror') or role.get('grating') or role.get('dmd'))
        and 0.0 < float(getattr(front, 'thickness', 0.0)) < 1000.0
    )
    print(f"Index {index:02d}: {front.name} -> {rear.name} | is_air={is_air} | t={front.thickness} | glass={front.glass}")
