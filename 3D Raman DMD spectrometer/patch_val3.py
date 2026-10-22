import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_val = '''    # 7. Validate & Export
    if verbose: print(f"[STEP 7/7] Validating & Exporting...\n  Top Vol: {top_half.Volume:.1f}  Bot Vol: {bottom_half.Volume:.1f}")'''

new_val = '''    # 7. Validate & Export
    def clean_solids(part):
        if part.isNull(): return part
        if len(part.Solids) > 1:
            main_solid = max(part.Solids, key=lambda s: s.Volume)
            return main_solid
        return part

    top_half = clean_solids(top_half)
    bottom_half = clean_solids(bottom_half)
    
    if verbose: print(f"[STEP 7/7] Validating & Exporting...\n  Top Vol: {top_half.Volume:.1f}  Bot Vol: {bottom_half.Volume:.1f}")'''

if old_val in content:
    content = content.replace(old_val, new_val)
    print("Replaced successfully!")
else:
    print("Failed!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
