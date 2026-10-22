import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_val = '''    # 7. Validation & Exporting
    if verbose: print("[STEP 7/7] Validating & Exporting...")'''

new_val = '''    # 7. Validation & Exporting
    if verbose: print("[STEP 7/7] Validating & Exporting...")
    def clean_solids(part):
        if part.isNull(): return part
        if len(part.Solids) > 1:
            main_solid = max(part.Solids, key=lambda s: s.Volume)
            return main_solid
        return part

    top_half = clean_solids(top_half)
    bottom_half = clean_solids(bottom_half)
'''

if old_val in content:
    content = content.replace(old_val, new_val)
    print("Replaced successfully!")
else:
    # try another match
    old_val2 = '''    # 7. Validating & Exporting
    if verbose: print("[STEP 7/7] Validating & Exporting...")'''
    if old_val2 in content:
        content = content.replace(old_val2, new_val)
        print("Replaced successfully! (2)")
    else:
        print("Still failed to replace!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
