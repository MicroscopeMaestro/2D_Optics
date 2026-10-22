import re

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_rect = '''    hx = max(float(getattr(surf, "aperture_x_half", surf.semi_diameter)), 0.5)
    hy = max(float(getattr(surf, "aperture_y_half", surf.semi_diameter)), 0.5)
    if surrounding_max_sd > max(hx, hy):
        ox = max(hx, surrounding_max_sd)
        oy = max(hy, surrounding_max_sd)
    else:
        ox = hx
        oy = hy'''

new_rect = '''    hx = max(float(getattr(surf, "aperture_x_half", surf.semi_diameter)), 0.001)
    hy = max(float(getattr(surf, "aperture_y_half", surf.semi_diameter)), 0.001)
    # The outer rectangular bounds should exactly match the defined semi_diameter of the mounting plate
    sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
    ox = max(sd, hx + 0.001)
    oy = max(sd, hy + 0.001)'''

content = content.replace(old_rect, new_rect)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
