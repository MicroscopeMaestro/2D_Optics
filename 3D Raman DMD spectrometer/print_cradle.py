with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'def make_aspheric_cradle' in line:
        for j in range(i, i + 40):
            print(lines[j], end='')
        break
