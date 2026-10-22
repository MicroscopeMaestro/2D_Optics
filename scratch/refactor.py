import re, os

def refactor_main_window():
    path = 'optics_app/gui/main_window.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace act_dmd_toggle with act_mech_toggle
    content = content.replace('act_dmd_toggle', 'act_mech_toggle')
    # Replace _toggle_dmd_state with _toggle_mech_state
    content = content.replace('_toggle_dmd_state', '_toggle_mech_state')
    # Replace _sync_dmd_action_ui with _sync_mech_action_ui
    content = content.replace('_sync_dmd_action_ui', '_sync_mech_action_ui')
    # Replace visualizer_3d._sync_dmd_button_ui with _sync_mech_button_ui
    content = content.replace('_sync_dmd_button_ui', '_sync_mech_button_ui')
    
    # Fix the method contents
    content = re.sub(r'Toggle DMD State', r'Toggle Mechanism State', content)
    content = re.sub(r'Toggle DMD state', r'Toggle mechanism state', content)
    content = re.sub(r'DMD switched to', r'Mechanism switched to', content)
    content = re.sub(r'DMD: N/A', r'Mech: N/A', content)
    content = re.sub(r'DMD: \{state\}', r'Mech: {state}', content)
    content = re.sub(r'DMD_OFF_NO_DETECTOR_SIGNAL_EXPECTED', r'MECH_OFF_NO_DETECTOR_SIGNAL_EXPECTED', content)
    content = re.sub(r'Cycle configured DMD states', r'Cycle configured mechanism states', content)
    
    # In _sync_mech_action_ui:
    # dmd_surface = next((s for s in stack.surfaces if is_dmd(s)), None) if stack else None
    # We should change is_dmd(s) to checking available_states(s)
    content = content.replace('from optics_app.engine.mechanisms import is_dmd, active_state_name, available_states',
                              'from optics_app.engine.mechanisms import MechanicalStateController, active_state_name, available_states')
    
    # We need to replace the logic that finds the surface.
    old_logic = "dmd_surface = next((s for s in stack.surfaces if is_dmd(s)), None) if stack else None"
    new_logic = "mech_surface = next((s for s in stack.surfaces if available_states(s)), None) if stack else None"
    content = content.replace(old_logic, new_logic)
    content = content.replace('dmd_surface', 'mech_surface')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def refactor_visualizer_3d():
    path = 'optics_app/gui/visualizer_3d.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Rename variables
    content = content.replace('btn_dmd_state', 'btn_mech_state')
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
    
    # In get_mech_surface_info:
    content = content.replace('from optics_app.engine.mechanisms import is_dmd, active_state_name',
                              'from optics_app.engine.mechanisms import available_states, active_state_name')
    content = content.replace('if is_dmd(s):', 'if available_states(s):')
    
    content = content.replace('dmd_idx', 'mech_idx')
    content = content.replace('dmd_surf', 'mech_surf')
    content = content.replace('dmd_info', 'mech_info')
    content = content.replace('dmd_surface', 'mech_surface')
    content = content.replace('DMD {state_name}: configured route', 'Mech {state_name}: route')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

refactor_main_window()
refactor_visualizer_3d()
