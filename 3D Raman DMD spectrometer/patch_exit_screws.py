import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add the exit screws!
old_screws = '''    (-21.0, 205.0)
]'''
new_screws = '''    (-21.0, 205.0),
    (-15.0, 227.0),  # EXIT RIGHT
    (-50.0, 227.0)   # EXIT LEFT
]'''
content = content.replace(old_screws, new_screws)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added exit clamping screws at Z=227!")
