import re

path = 'optics_app/gui/delegates.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('TYPES = ["Object", "Standard", "Mirror", "Aspheric", "Stop", "Grating", "Detector", "CAD", "Air Space"]', 
                          'TYPES = ["Object", "Standard", "Mirror", "Aspheric", "Grating", "Detector", "CAD", "Air Space"]')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
