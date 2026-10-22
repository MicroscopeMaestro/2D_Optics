import re

path = 'optics_app/gui/component_editor_dock.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace variables
content = content.replace('btn_dmd_on', 'btn_mech_1')
content = content.replace('btn_dmd_off', 'btn_mech_2')
content = content.replace('btn_dmd_flat', 'btn_mech_3')

content = content.replace('DMD ON', 'MECH ON')
content = content.replace('DMD OFF', 'MECH OFF')
content = content.replace('DMD FLAT', 'MECH FLAT')

content = content.replace('dmd = is_dmd(surface)', 'mech = bool(available_states(surface))')
content = content.replace('if dmd:', 'if mech:')

content = content.replace('has_system_dmd', 'has_system_mech')
content = content.replace('if is_dmd(s):', 'if available_states(s):')
content = content.replace('not is_dmd(curr_surf)', 'not available_states(curr_surf)')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
