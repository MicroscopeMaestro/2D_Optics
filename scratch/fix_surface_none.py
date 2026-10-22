import re

path = 'optics_app/engine/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''        if atype in ("Point", "None"):
            self.semi_diameter = 0.0
            self.aperture_x_half = 0.0
            self.aperture_y_half = 0.0
            self.aperture_r_inner = 0.0
            self.aperture_r_outer = 0.0
            return'''

new_logic = '''        if atype in ("Point", "None"):
            self.aperture_x_half = 0.0
            self.aperture_y_half = 0.0
            self.aperture_r_inner = 0.0
            self.aperture_r_outer = 0.0
            return'''

content = content.replace(old_logic, new_logic)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
