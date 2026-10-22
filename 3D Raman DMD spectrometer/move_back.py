import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern_4 = r'(\s*# 4\. Optical Subtractions.*?(?=\s*# 1\.8 Massive 4-Corner Block Cuts))'
match = re.search(pattern_4, content, re.DOTALL)
if match:
    step4_code = match.group(1)
    content = content.replace(step4_code, '')
    
    # insert after STEP 3.5
    pattern_insert = r'(# 3\.5 Interlocking Perimeter.*?bottom_half = bottom_half\.fuse\(tongue_solid\)\n\s+top_half = top_half\.cut\(groove_solid\)\n\s+except Exception as e:\n\s+if verbose: print\(f"  \[WARNING\] Perimeter interlock failed: \{e\}"\)\n\s+# ---------------------------------------------------------)'
    
    match_insert = re.search(pattern_insert, content, re.DOTALL)
    if match_insert:
        insert_code = match_insert.group(1)
        content = content.replace(insert_code, insert_code + '\n' + step4_code)
        print("Moved STEP 4 back!")
    else:
        print("Could not find insert point")
else:
    print("Could not find STEP 4")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
