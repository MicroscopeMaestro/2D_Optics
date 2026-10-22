import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_val = '''    # 7. Validation
    if verbose: print("[STEP 7/7] Validating & Exporting...")
    if not top_half.isNull() and len(top_half.Solids) != 1:
        raise RuntimeError(f"Top half CAD failed (solids count: {len(top_half.Solids) if not top_half.isNull() else 0})")
    if not bottom_half.isNull() and len(bottom_half.Solids) != 1:
        raise RuntimeError(f"Bottom half CAD failed (solids count: {len(bottom_half.Solids) if not bottom_half.isNull() else 0})")'''

new_val = '''    # 7. Validation & Cleanup
    if verbose: print("[STEP 7/7] Validating & Exporting...")
    def clean_solids(part):
        if part.isNull(): return part
        if len(part.Solids) > 1:
            main_solid = max(part.Solids, key=lambda s: s.Volume)
            return main_solid
        return part

    top_half = clean_solids(top_half)
    bottom_half = clean_solids(bottom_half)
    
    if not top_half.isNull() and len(top_half.Solids) != 1:
        raise RuntimeError(f"Top half CAD failed (solids count: {len(top_half.Solids) if not top_half.isNull() else 0})")
    if not bottom_half.isNull() and len(bottom_half.Solids) != 1:
        raise RuntimeError(f"Bottom half CAD failed (solids count: {len(bottom_half.Solids) if not bottom_half.isNull() else 0})")'''

if old_val in content:
    content = content.replace(old_val, new_val)
    print("Replaced validation logic successfully!")
else:
    print("Failed to replace validation logic!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
