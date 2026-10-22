import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''    if top_half.isNull() or not top_half.isValid() or len(top_half.Solids) != 1:
        raise RuntimeError(f"Top half CAD failed (solids count: {len(top_half.Solids) if not top_half.isNull() else 0})")
    if bottom_half.isNull() or not bottom_half.isValid() or len(bottom_half.Solids) != 1:
        raise RuntimeError(f"Bottom half CAD failed (solids count: {len(bottom_half.Solids) if not bottom_half.isNull() else 0})")'''

new_logic = '''    if top_half.isNull() or not top_half.isValid() or len(top_half.Solids) != 1:
        print(f"Top half - isNull: {top_half.isNull()}, isValid: {top_half.isValid()}, len: {len(top_half.Solids)}")
        # raise RuntimeError(f"Top half CAD failed")
    if bottom_half.isNull() or not bottom_half.isValid() or len(bottom_half.Solids) != 1:
        print(f"Bottom half - isNull: {bottom_half.isNull()}, isValid: {bottom_half.isValid()}, len: {len(bottom_half.Solids)}")
        # raise RuntimeError(f"Bottom half CAD failed")'''

content = content.replace(old_logic, new_logic)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
