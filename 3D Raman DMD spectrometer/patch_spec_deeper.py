with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('z_max = bb.ZMax + 1.0', 'z_max = bb.ZMax + 8.0')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated spectrometer to fit the deeper bed!")
