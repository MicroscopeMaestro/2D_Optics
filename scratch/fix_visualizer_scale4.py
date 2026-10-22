import re

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_circ = '''    sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
    r_inner = sd
    custom_r_out = float(getattr(surf, "aperture_r_outer", 0.0) or 0.0)
    if custom_r_out >= sd:
        r_outer = custom_r_out
    else:
        r_outer = sd
        
    # Ensure at least a tiny rim if they are exactly identical so the rendering engine doesn't crash on 0-area polys
    if abs(r_outer - r_inner) < 1e-4:
        r_outer = r_inner + 0.001'''

new_circ = '''    # The outer metal plate size
    sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
    
    # The clear hole size (where light passes through)
    hole_radius = float(getattr(surf, "aperture_r_outer", 0.0))
    if hole_radius <= 0:
        hole_radius = sd
        
    # The inside edge of the drawn wall is the clear hole
    r_inner = hole_radius
    
    # The outside edge of the drawn wall is the semi_diameter (mount size)
    r_outer = max(sd, hole_radius)
        
    # Ensure at least a tiny rim if they are exactly identical so the rendering engine doesn't crash on 0-area polys
    if abs(r_outer - r_inner) < 1e-4:
        r_outer = r_inner + 0.001'''

content = content.replace(old_circ, new_circ)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
