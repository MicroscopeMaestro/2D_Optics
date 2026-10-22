with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "SLIT_SLOT_WIDTH             = 25.4" in line:
        new_lines.append(line)
        new_lines.append("# --- EXPERIMENT PREPARATION OFFSETS ---\n")
        new_lines.append("AXIAL_LENS_OFFSETS          = {}  # e.g., {'01_collect': 2.0, '17_condenser': -1.5}\n")
        new_lines.append("GRATING_ROTATION_OFFSET     = 0.0 # Degrees to rotate the grating hardware\n")
        new_lines.append("SLIT_IS_CIRCULAR_1_INCH     = True # True for 1-inch round disk, False for square plate\n")
        new_lines.append("# --------------------------------------\n")
    else:
        new_lines.append(line)

content = "".join(new_lines)
with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
