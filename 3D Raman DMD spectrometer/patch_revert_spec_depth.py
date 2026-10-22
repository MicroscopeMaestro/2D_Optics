with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Revert the block depth
content = content.replace('z_max = bb.ZMax + 8.0', 'z_max = bb.ZMax + 1.0')

# And fix the sliding tolerance! Since we slide it in the X axis now, we don't need 5.0mm of empty space on the Z axis!
content = content.replace('SLIDING_TOLERANCE           = 5.0', 'SLIDING_TOLERANCE           = 0.4')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Reverted pocket depth and fixed sliding tolerance!")
