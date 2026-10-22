import re

with open('optics_app/gui/visualizer_3d.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''                is_air_region = (
                    self.show_air_gaps
                    and not any(front_role.values())
                    and front.surface_type in (SurfaceType.STANDARD, SurfaceType.ASPHERIC, SurfaceType.CAD)
                    and getattr(front, 'glass', '').upper() in ('AIR', '', 'NONE')
                    and 0.0 < float(getattr(front, 'thickness', 0.0)) < 1000.0
                )'''

replacement = '''                is_air_region = (
                    self.show_air_gaps
                    and getattr(front, 'glass', '').upper() in ('AIR', '', 'NONE')
                    and 0.0 < float(getattr(front, 'thickness', 0.0)) < 1000.0
                )'''

if target in content:
    content = content.replace(target, replacement)
    with open('optics_app/gui/visualizer_3d.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patched is_air_region successfully.")
else:
    print("Could not find target string.")
