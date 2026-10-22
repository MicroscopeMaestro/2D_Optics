import re

def refactor_dual_view():
    path = 'optics_app/gui/dual_view.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Rename variables
    content = content.replace('btn_dmd_toggle', 'btn_mech_toggle')
    content = content.replace('toggle_dmd_state', 'toggle_mech_state')
    content = content.replace('_sync_dmd_button_ui', '_sync_mech_button_ui')
    content = content.replace('_update_dmd_button_ui', '_update_mech_button_ui')
    content = content.replace('get_dmd_surface_info', 'get_mech_surface_info')
    
    # Change text
    content = content.replace('DMD: ON', 'MECH: ON')
    content = content.replace('Toggle DMD micromirrors between', 'Toggle mechanism between')
    content = content.replace('Advance the selected DMD state', 'Advance the selected mechanism state')
    content = content.replace('Cycle the configured DMD mechanical states.', 'Cycle the configured mechanism states.')
    content = content.replace('DMD: {state', 'MECH: {state')
    content = content.replace('DMD [{', 'MECH [{')
    content = content.replace('DMD OFF', 'MECH OFF')
    
    # In get_mech_surface_info:
    content = content.replace('from optics_app.engine.mechanisms import is_dmd, active_state_name',
                              'from optics_app.engine.mechanisms import available_states, active_state_name')
    content = content.replace('if is_dmd(s):', 'if available_states(s):')
    
    content = content.replace('dmd_idx', 'mech_idx')
    content = content.replace('dmd_surf', 'mech_surf')
    content = content.replace('dmd_info', 'mech_info')
    content = content.replace('dmd_surface', 'mech_surface')
    content = content.replace('DMD {mech_info', 'MECH {mech_info')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

refactor_dual_view()
