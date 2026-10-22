with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('CLAMSHELL_HALF_THICKNESS    = 17.5', 'CLAMSHELL_HALF_THICKNESS    = 19.5')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Thickened CLAMSHELL_HALF_THICKNESS to 19.5!")
