import re

path = 'optics_app/gui/main_window.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace variables
content = content.replace('act_dmd_toggle', 'act_mech_toggle')
content = content.replace('_toggle_dmd_state', '_toggle_mech_state')
content = content.replace('_sync_dmd_action_ui', '_sync_mech_action_ui')
content = content.replace('toggle_dmd_state', 'toggle_mech_state')
content = content.replace('get_dmd_surface_info', 'get_mech_surface_info')

# Replace exact strings
content = content.replace('" Toggle DMD State (ON / OFF)"', '"⚙ Toggle Mechanism State"')
content = content.replace('"Toggle DMD state between ON (Detector) and OFF (Beam Dump)"', '"Toggle optical mechanism active state"')
content = content.replace('DMD switched to', 'Mechanism switched to')
content = content.replace('DMD: N/A', 'Mechanism: N/A')
content = content.replace('DMD: ON (Detector)', 'Mechanism: ON')
content = content.replace('DMD: OFF (Beam Dump)', 'Mechanism: OFF')
content = content.replace('DMD is ON: light reaches detector. Click to toggle to OFF (Beam Dump).', 'Mechanism is in ON state.')
content = content.replace('DMD is OFF: light deflected 48° to dump. Click to toggle to ON (Detector).', 'Mechanism is in OFF state.')
content = content.replace('DMDs', 'Mechanisms')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
