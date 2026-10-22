with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line.replace('\t', '    '))

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Replaced all tabs with spaces in build_spectrometer.py!")
