import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.visualizer_3d import classify_surface_for_3d, SurfaceType

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
surfaces = stack.surfaces

solid_face_ids = set()

for index, front in enumerate(surfaces[:-1]):
    if id(front) in solid_face_ids:
        print(f"SKIPPED at index {index:02d}: {front.name} (id in solid_face_ids)")
        continue
    front_role = classify_surface_for_3d(front, index, len(surfaces))
    is_glass_region = (
        not any(front_role.values())
        and front.surface_type in (SurfaceType.STANDARD, SurfaceType.ASPHERIC, SurfaceType.CAD)
        and getattr(front, 'glass', '').upper() not in ('AIR', '', 'NONE', 'MIRROR', 'MIRR', 'REFL')
        and float(getattr(front, 'thickness', 0.0)) > 0.0
    )
    is_air_region = (
        (getattr(front, 'glass', '').upper() in ('AIR', '', 'NONE')
         or front_role.get("mirror") or front_role.get("grating") or front_role.get("dmd"))
        and 0.0 < float(getattr(front, 'thickness', 0.0)) < 1000.0
    )
    if not (is_glass_region or is_air_region):
        continue
    rear = surfaces[index + 1]
    kind = "GLASS" if is_glass_region else "AIR"
    print(f"RENDERED {kind:<5} at index {index:02d}: {front.name} -> {rear.name}")
    solid_face_ids.update((id(front), id(rear)))
