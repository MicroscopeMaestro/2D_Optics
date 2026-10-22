import re

path = 'optics_app/gui/analysis_tabs.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Make it completely generic, searching for any mechanism that is OFF
old_dmd1 = '''                        from optics_app.engine.mechanisms import is_dmd, active_state_name
                        dmd_off = False
                        if system and getattr(system, 'surface_stack', None):
                            for s in system.surface_stack.surfaces:
                                if is_dmd(s) and active_state_name(s) == "OFF":
                                    dmd_off = True
                                    break
                        if dmd_off:
                            ax_k.text(0.5, 0.5, "DMD OFF\\n(Beam Dump)", horizontalalignment='center', verticalalignment='center',
                                      fontsize=8, fontweight='bold', color='#991b1b')
                        else:'''

new_dmd1 = '''                        from optics_app.engine.mechanisms import available_states, active_state_name
                        mech_off = False
                        if system and getattr(system, 'surface_stack', None):
                            for s in system.surface_stack.surfaces:
                                if available_states(s) and active_state_name(s) == "OFF":
                                    mech_off = True
                                    break
                        if mech_off:
                            ax_k.text(0.5, 0.5, "Mechanism OFF\\n(Beam Dump)", horizontalalignment='center', verticalalignment='center',
                                      fontsize=8, fontweight='bold', color='#991b1b')
                        else:'''

old_dmd2 = '''                    from optics_app.engine.mechanisms import is_dmd, active_state_name
                    dmd_off = False
                    if system and getattr(system, 'surface_stack', None):
                        for s in system.surface_stack.surfaces:
                            if is_dmd(s) and active_state_name(s) == "OFF":
                                dmd_off = True
                                break
                    if dmd_off:
                        ax.text(0.5, 0.55, "⛔ DMD State: OFF (Routed to Beam Dump)\\nDetector Dark (0 hits - Beam Deflected 48°)", 
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=11, fontweight='bold', color='#991b1b',
                                bbox=dict(boxstyle='round,pad=0.6', facecolor='#fee2e2', edgecolor='#fca5a5'))
                        ax.set_title(f"Zemax Spot Diagram (Detector Plane){f_title} - DMD OFF", fontsize=10, fontweight='bold')
                    else:'''

new_dmd2 = '''                    from optics_app.engine.mechanisms import available_states, active_state_name
                    mech_off = False
                    if system and getattr(system, 'surface_stack', None):
                        for s in system.surface_stack.surfaces:
                            if available_states(s) and active_state_name(s) == "OFF":
                                mech_off = True
                                break
                    if mech_off:
                        ax.text(0.5, 0.55, "⛔ Mechanism State: OFF (Routed to Beam Dump)\\nDetector Dark (0 hits)", 
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=11, fontweight='bold', color='#991b1b',
                                bbox=dict(boxstyle='round,pad=0.6', facecolor='#fee2e2', edgecolor='#fca5a5'))
                        ax.set_title(f"Zemax Spot Diagram (Detector Plane){f_title} - Mechanism OFF", fontsize=10, fontweight='bold')
                    else:'''

old_dmd3 = '''            from optics_app.engine.mechanisms import is_dmd, active_state_name
            dmd_off = False
            if system and getattr(system, 'surface_stack', None):
                for s in system.surface_stack.surfaces:
                    if is_dmd(s) and active_state_name(s) == "OFF":
                        dmd_off = True
                        break
            if dmd_off:
                ax_map.text(0.5, 0.5, "⛔ DMD State: OFF\\nDetector Dark (0 hits)",
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=10, fontweight='bold', color='#991b1b',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='#fca5a5'))
            else:'''

new_dmd3 = '''            from optics_app.engine.mechanisms import available_states, active_state_name
            mech_off = False
            if system and getattr(system, 'surface_stack', None):
                for s in system.surface_stack.surfaces:
                    if available_states(s) and active_state_name(s) == "OFF":
                        mech_off = True
                        break
            if mech_off:
                ax_map.text(0.5, 0.5, "⛔ Mechanism State: OFF\\nDetector Dark (0 hits)",
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=10, fontweight='bold', color='#991b1b',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='#fca5a5'))
            else:'''

content = content.replace(old_dmd1, new_dmd1).replace(old_dmd2, new_dmd2).replace(old_dmd3, new_dmd3)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Replaced analysis tabs DMD references.')
