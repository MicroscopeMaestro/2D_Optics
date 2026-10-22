import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Use regex to remove everything from STEP 2.95 to STEP 4
pattern = re.compile(r'\[STEP 2\.95/7\].*?\# 4\. Optical Subtractions', re.DOTALL)
if pattern.search(content):
    content = pattern.sub('# Breadboard removed per user request\n    bb_x_min = -CLAMSHELL_HALF_THICKNESS\n\n    # 4. Optical Subtractions', content)
    print("Successfully removed STEP 2.95")
else:
    print("Could not find STEP 2.95")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
