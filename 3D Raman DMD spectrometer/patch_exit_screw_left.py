with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('(-50.0, 227.0)', '(-46.0, 227.0)')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Shifted exit left screw inward to prevent breach!")
