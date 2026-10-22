import re

path = 'optics_app/engine/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add is_stop: bool = False to Surface
content = content.replace('surface_type: SurfaceType = SurfaceType.STANDARD', 'surface_type: SurfaceType = SurfaceType.STANDARD\n    is_stop: bool = False')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
