with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('USE_EXTERNAL_ASPHERIC_MOUNTS= False', 'USE_EXTERNAL_ASPHERIC_MOUNTS= True')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Set USE_EXTERNAL_ASPHERIC_MOUNTS to True!")
