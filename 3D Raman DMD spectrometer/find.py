import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find STEP 4
pattern_4 = r'(# 4\. Optical Subtractions.*?\n\s+top_half, bottom_half = top_half\.cut\(dmd_window_cut\), bottom_half\.cut\(dmd_window_cut\))'
# Wait! In the current file, where is STEP 4?
