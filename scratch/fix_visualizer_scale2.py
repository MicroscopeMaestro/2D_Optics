import re

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix circular apertures:
# r_inner is the clear aperture (the hole) -> surf.semi_diameter
# r_outer is the outer mounting plate. If aperture_r_outer is provided, use it exactly!
# If it's not provided, use semi_diameter (so it draws nothing or a tiny rim).

# Let's completely rewrite the top of build_aperture_stop_geometry
old_logic = '''    sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.5)
    r_inner = sd
    custom_r_out = float(getattr(surf, "aperture_r_outer", 0.0) or 0.0)
    if custom_r_out > sd:
        r_outer = custom_r_out
    elif surrounding_max_sd > sd:
        r_outer = sd
    else:
        r_outer = sd'''

new_logic = '''    sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
    r_inner = sd
    custom_r_out = float(getattr(surf, "aperture_r_outer", 0.0) or 0.0)
    if custom_r_out >= sd:
        r_outer = custom_r_out
    else:
        r_outer = sd
        
    # Ensure at least a tiny rim if they are exactly identical so the rendering engine doesn't crash on 0-area polys
    if abs(r_outer - r_inner) < 1e-4:
        r_outer = r_inner + 0.001'''

content = content.replace(old_logic, new_logic)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
