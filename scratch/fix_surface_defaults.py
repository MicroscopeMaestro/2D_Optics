import re

path = 'optics_app/engine/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('semi_diameter: float = 25.0', 'semi_diameter: float = 12.7')
content = content.replace('aperture_x_half: float = 20.0', 'aperture_x_half: float = 0.0')
content = content.replace('aperture_y_half: float = 15.0', 'aperture_y_half: float = 0.0')
content = content.replace('aperture_a_semi: float = 20.0', 'aperture_a_semi: float = 0.0')
content = content.replace('aperture_b_semi: float = 15.0', 'aperture_b_semi: float = 0.0')

# Also fix the SphericalSurface default just in case
content = content.replace('semi_diameter: float = 25.0)', 'semi_diameter: float = 12.7)')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
