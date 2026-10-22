import re

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix circular apertures
content = content.replace('r_outer = max(sd * outer_scale, surrounding_max_sd * 1.08, sd + 4.0)', 'r_outer = sd')
content = content.replace('r_outer = max(sd * outer_scale, sd + 4.0)', 'r_outer = sd')

# Fix rectangular apertures
content = content.replace('ox = max(hx * outer_scale, surrounding_max_sd * 1.08, hx + 4.0)', 'ox = max(hx, surrounding_max_sd)')
content = content.replace('oy = max(hy * outer_scale, surrounding_max_sd * 1.08, hy + 4.0)', 'oy = max(hy, surrounding_max_sd)')
content = content.replace('ox = max(hx * outer_scale, hx + 4.0)', 'ox = hx')
content = content.replace('oy = max(hy * outer_scale, hy + 4.0)', 'oy = hy')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
