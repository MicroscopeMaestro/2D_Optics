with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re
old_screws_pattern = r'MANUAL_SCREW_POSITIONS = \[\s*\( 17\.5, 20\.0\).*?\]'
new_screws = '''MANUAL_SCREW_POSITIONS = [
    ( 17.5, 20.0),   # ENTRANCE RIGHT
    (-17.5, 20.0),   # ENTRANCE LEFT
    (-15.0, 227.0),  # EXIT RIGHT
    (-46.0, 227.0)   # EXIT LEFT
]'''

content = re.sub(old_screws_pattern, new_screws, content, flags=re.DOTALL)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Removed middle screws!")
