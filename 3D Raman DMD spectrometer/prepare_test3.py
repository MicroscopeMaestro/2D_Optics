with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '    # 5. Fasteners',
    '    Part.export([top_half], "temp_top_no_fasteners.step")\n    Part.export([bottom_half], "temp_bot_no_fasteners.step")\n    sys.exit(0)\n    # 5. Fasteners'
)

with open('temp_build.py', 'w', encoding='utf-8') as f:
    f.write(content)
