
import re

with open('optics_app/cad/exporter.py', 'r') as f:
    code = f.read()

# We need to replace the entire 'while i < len(stack.surfaces):' loop inside the BUILD123D block.
# Let's find where the loop starts and where it ends.

start_str = '            i = 0\\n            elem_count = 1\\n            while i < len(stack.surfaces):'
start_idx = code.find('            i = 0\\n            elem_count = 1\\n            while i < len(stack.surfaces):')

if start_idx != -1:
    end_str = '            if parts:'
    end_idx = code.find(end_str, start_idx)
    
    new_loop = '''            i = 0
            elem_count = 1
            while i < len(stack.surfaces) - 1:
                surf = stack.surfaces[i]
                
                is_glass = surf.glass.upper() not in ['AIR', '', 'NONE', 'MIRROR', 'MIRR', 'REFL'] and not surf.is_reflective() and surf.thickness > 0
                
                if is_glass:
                    idx_end = i
                    total_thickness = 0
                    while idx_end < len(stack.surfaces) - 1:
                        curr = stack.surfaces[idx_end]
                        if curr.glass.upper() in ['AIR', '', 'NONE', 'MIRROR', 'MIRR', 'REFL'] or curr.is_reflective() or curr.thickness == 0:
                            break
                        total_thickness += curr.thickness
                        idx_end += 1
                        
                    surf_back = stack.surfaces[idx_end]
                    part_3d = create_solid_lens(surf, surf_back) # fallback to old signature
                    
                    # Wait, create_solid_lens doesn't take total_thickness!
                    # Let's redefine create_solid_volume here
                    
'''

