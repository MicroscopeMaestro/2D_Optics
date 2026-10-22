import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.visualizer_3d import classify_surface_for_3d, SurfaceType

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
surfaces = stack.surfaces

for i, s in enumerate(surfaces[:-1]):
    role = classify_surface_for_3d(s, i, len(surfaces))
    is_glass = (
        not any(role.values())
        and s.surface_type in (SurfaceType.STANDARD, SurfaceType.ASPHERIC, SurfaceType.CAD)
        and getattr(s, 'glass', '').upper() not in ('AIR', '', 'NONE', 'MIRROR', 'MIRR', 'REFL')
        and float(getattr(s, 'thickness', 0.0)) > 0.0
    )
    is_air = (
        (getattr(s, 'glass', '').upper() in ('AIR', '', 'NONE')
         or role.get('mirror') or role.get('grating') or role.get('dmd'))
        and 0.0 < float(getattr(s, 'thickness', 0.0)) < 1000.0
    )
    print(f"[{i:02d}] {s.name:<25} | t={s.thickness:<5.1f} | role={role} | is_glass={is_glass} | is_air={is_air}")
