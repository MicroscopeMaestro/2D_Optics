with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('p_width = 40.4', 'p_width = 30.4')
content = content.replace('p_height = 40.4', 'p_height = 30.4')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated spectrometer script to 30.4x30.4mm pockets!")
