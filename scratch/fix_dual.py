import re

path = 'optics_app/gui/dual_view.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('is_dmd = \"DMD\" in (surf.name or \"\").upper()', 'is_mech = bool(available_states(surf))')
content = content.replace('elif is_dmd:', 'elif is_mech:')
content = content.replace('dmd_st = active_state_name(surf)', 'mech_st = active_state_name(surf)')
content = content.replace('dmd_st', 'mech_st')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
