import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.visualizer_3d import get_effective_pose

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
stack._update_accumulated_positions()

for i in [13, 14, 15, 16, 17, 19, 20, 21, 22]:
    s = stack.surfaces[i]
    c, R = get_effective_pose(s)
    print(f"[{i:02d}] {s.name:<24} | c={c} | t={s.thickness} | tilt=({getattr(s, 'tilt_x', 0)}, {getattr(s, 'tilt_y', 0)})")
