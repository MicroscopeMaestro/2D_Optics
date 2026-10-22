import re

path = 'optics_app/gui/component_editor_dock.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('from optics_app.engine.mechanisms import is_dmd\n', 'from optics_app.engine.mechanisms import is_dmd, available_states\n')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
