with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Add a global floor and roof shaver right before exporting
shaver_code = '''
    # Guarantee flat floor and roof by shaving any protrusions
    if verbose: print("  -> Shaving global floor and roof...")
    floor_shaver = Part.makeBox(50.0, 300.0, 300.0, FreeCAD.Vector(-50.0 - CLAMSHELL_HALF_THICKNESS, -150.0, -20.0))
    bottom_half = bottom_half.cut(floor_shaver)
    
    roof_shaver = Part.makeBox(50.0, 300.0, 300.0, FreeCAD.Vector(CLAMSHELL_HALF_THICKNESS, -150.0, -20.0))
    top_half = top_half.cut(roof_shaver)
    
    if verbose: print("[STEP 7/7] Validating & Exporting...")
'''

content = content.replace('if verbose: print("[STEP 7/7] Validating & Exporting...")', shaver_code)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Injected global shaver to perfectly flatten the floor and roof!")
