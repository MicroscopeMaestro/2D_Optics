with open('build_external_bed_4_5mm_taper.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Make build_external_bed.py output external_aspheric_bed.step by default
code_default = code.replace('output_step="external_aspheric_bed_4_5mm_taper.step"', 'output_step="external_aspheric_bed.step"')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(code_default)

print("Updated build_external_bed.py!")
