with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("preserving the 18mm clear aperture (minimum radius 9.0mm)", "preserving the 19mm clear aperture (minimum radius 9.5mm)")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
