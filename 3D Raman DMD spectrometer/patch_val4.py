import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r"# 7\. Validate & Export\s+if verbose: print\(f\"\[STEP 7/7\].*?\"\)"
replacement = '''# 7. Validate & Export
    def clean_solids(part):
        if part.isNull(): return part
        if len(part.Solids) > 1:
            main_solid = max(part.Solids, key=lambda s: s.Volume)
            return main_solid
        return part

    top_half = clean_solids(top_half)
    bottom_half = clean_solids(bottom_half)
    
    if verbose: print(f"[STEP 7/7] Validating & Exporting... Top Vol: {top_half.Volume:.1f} Bot Vol: {bottom_half.Volume:.1f}")'''

content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
print(f"Replaced {count} times!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
