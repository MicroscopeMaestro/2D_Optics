with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = "tap_rad = 2.8 if z < 25.0 else SCREW_TAP_RADIUS"
new_logic = "tap_rad = SCREW_TAP_RADIUS"
content = content.replace(old_logic, new_logic)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Removed heat-set insert oversize override for entrance holes!")
