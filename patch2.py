import re

with open('optics_app/cad/exporter.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('elif etype in ["Singlet", "Air Volume"]:', 'elif etype in ["Singlet", "Air Volume", "Air Gap"]:')

with open('optics_app/cad/exporter.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("done")
