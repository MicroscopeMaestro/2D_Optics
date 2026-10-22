import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find STEP 4/7 Optical subtractions block
pattern_4 = r'(# 4\. Optical Subtractions.*?)(?=# 5\. Fasteners)'
match_4 = re.search(pattern_4, content, re.DOTALL)
if match_4:
    step_4_code = match_4.group(1)
    # Remove it from its current location
    content = content.replace(step_4_code, '')
    
    # Insert it right before 1.8 Massive 4-Corner Block Cuts
    pattern_insert = r'(# 1\.8 Massive 4-Corner Block Cuts)'
    content = re.sub(pattern_insert, step_4_code + r'\n    \1', content)
    
    with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Moved Optical Subtractions to occur before tapers!")
else:
    print("Could not find STEP 4/7")
