path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('from optics_app.engine.mechanisms import available_states, active_state_name, active_state, available_states',
                          'from optics_app.engine.mechanisms import is_dmd, available_states, active_state_name, active_state')
content = content.replace('from optics_app.engine.mechanisms import MechanicalStateController, active_state_name, available_states',
                          'from optics_app.engine.mechanisms import is_dmd, MechanicalStateController, active_state_name, available_states')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
