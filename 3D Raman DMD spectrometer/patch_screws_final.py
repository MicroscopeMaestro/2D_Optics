import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the screw positions with the rigorously tested safe points
pattern = r'MANUAL_SCREW_POSITIONS = \[.*?\]'
new_screws = '''MANUAL_SCREW_POSITIONS = [
    (14.0, 21.0), (-14.0, 21.0), (18.0, 53.0),
    (-40.0, 110.0), (-25.0, 120.0),
    (-70.0, 150.0), (-30.0, 170.0), (20.0, 170.0),
    (-20.0, 190.0), (-55.0, 220.0)
]'''

content = re.sub(pattern, new_screws, content, flags=re.DOTALL)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated all screw positions!")
