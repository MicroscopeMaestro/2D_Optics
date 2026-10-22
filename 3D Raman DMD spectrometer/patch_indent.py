with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith('      if verbose: print("[STEP 2.5/7]'):
        lines[i] = line.replace('      ', '    ', 1)
    if line.startswith('      # We want the '):
        lines[i] = line.replace('      ', '    ', 1)
    if line.startswith('      for x in ['):
        lines[i] = line.replace('      ', '    ', 1)
    if line.startswith('      bot_trim = '):
        lines[i] = line.replace('      ', '    ', 1)
    if line.startswith('      entrance_tunnel = '):
        lines[i] = line.replace('      ', '    ', 1)
    if line.startswith('      # Also, to ensure'):
        lines[i] = line.replace('      ', '    ', 1)
    if line.startswith('      # but wait'):
        lines[i] = line.replace('      ', '    ', 1)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Indentation patched")
