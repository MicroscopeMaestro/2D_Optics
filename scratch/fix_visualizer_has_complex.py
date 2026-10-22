import re

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''                has_complex_aperture = (
                    ap_type in ('rectangular', 'slit', 'elliptical', 'spider')
                    or getattr(surf, 'aperture_r_inner', 0.0) > 0.0
                )'''

new_logic = '''                has_complex_aperture = (
                    ap_type in ('rectangular', 'slit', 'elliptical', 'spider')
                    or getattr(surf, 'aperture_r_inner', 0.0) > 0.0
                    or getattr(surf, 'aperture_r_outer', 0.0) > 0.0
                )'''

content = content.replace(old_logic, new_logic)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
