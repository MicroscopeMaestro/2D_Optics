import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('print(f"# Breadboard removed per user request', 'print("# Breadboard removed per user request")')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
