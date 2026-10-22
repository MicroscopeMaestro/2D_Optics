import re

with open('optics_app/gui/visualizer_3d.py', 'r', encoding='utf-8') as f:
    content = f.read()

target2 = '''                if id(surf) in solid_face_ids and not (is_mirror or is_grating or is_dmd or is_stop):
                    continue'''

replacement2 = '''                if id(surf) in solid_face_ids and not (is_mirror or is_grating or is_dmd or is_stop or has_complex_aperture):
                    continue'''

if target2 in content:
    content = content.replace(target2, replacement2)
    with open('optics_app/gui/visualizer_3d.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patched complex aperture")
else:
    print("Could not find targets.")
