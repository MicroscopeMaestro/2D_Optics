with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('if USE_EXTERNAL_ASPHERIC_MOUNTS and "01_collect" in obj.Label.lower():', 
                          'if USE_EXTERNAL_ASPHERIC_MOUNTS and "acl25416u" in obj.Label.lower():')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Forced external pockets for all lenses!")
