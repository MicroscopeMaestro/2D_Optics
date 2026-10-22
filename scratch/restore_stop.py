import re

path = 'optics_app/engine/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('    ASPHERIC = "Aspheric"\n', '    ASPHERIC = "Aspheric"\n    STOP = "Stop"\n')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
