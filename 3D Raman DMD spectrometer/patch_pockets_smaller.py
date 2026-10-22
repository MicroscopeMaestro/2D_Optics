with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('p_width = 30.8', 'p_width = 28.8')
content = content.replace('p_height = 30.8', 'p_height = 28.8')
content = content.replace('z_max = bb.ZMax + 5.0', 'z_max = bb.ZMax + 1.0')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated spectrometer script to 28.8x28.8mm pockets and thinner depth!")
