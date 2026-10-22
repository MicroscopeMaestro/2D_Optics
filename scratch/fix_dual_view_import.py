import re

path = 'optics_app/gui/dual_view.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('from optics_app.engine.tracer import OpticalSystem\n', 'from optics_app.engine.tracer import OpticalSystem\nfrom optics_app.engine.mechanisms import available_states, active_state_name\n')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
