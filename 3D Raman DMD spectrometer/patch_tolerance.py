with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('p_width = 30.4', 'p_width = 30.8')
content = content.replace('p_height = 30.4', 'p_height = 30.8')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated spectrometer script to 30.8x30.8mm pockets for looser fit!")
