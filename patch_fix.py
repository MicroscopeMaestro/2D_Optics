import re

with open('optics_app/gui/visualizer_3d.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = '''                if is_glass_region:
                    solid_face_ids.update((id(front), id(rear)))'''

replacement1 = '''                solid_face_ids.update((id(front), id(rear)))'''

target2 = '''                if id(surf) in solid_face_ids and not (is_mirror or is_grating or is_dmd):
                    continue'''

replacement2 = '''                if id(surf) in solid_face_ids and not (is_mirror or is_grating or is_dmd or is_stop):
                    continue'''

if target1 in content and target2 in content:
    content = content.replace(target1, replacement1)
    content = content.replace(target2, replacement2)
    with open('optics_app/gui/visualizer_3d.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patched both correctly.")
else:
    print("Could not find targets.")
