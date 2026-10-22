with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

bad_logic = '''if pos.x > 0: top_half = top_half.cut(cyl)
            else: bottom_half = bottom_half.cut(cyl)'''

good_logic = '''# The holes lie EXACTLY on the X=0 split line!
            top_half = top_half.cut(cyl)
            bottom_half = bottom_half.cut(cyl)'''

content = content.replace(bad_logic, good_logic)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Fixed sensor hole boolean cuts!")
