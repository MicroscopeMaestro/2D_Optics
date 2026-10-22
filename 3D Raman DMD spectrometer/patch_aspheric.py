with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_radius = "ASPHERIC_POCKET_RADIUS      = 13.05"
new_radius = "ASPHERIC_POCKET_RADIUS      = 13.25"
content = content.replace(old_radius, new_radius)

old_call = "final_cut = make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, clearance=0.25)"
new_call = "final_cut = make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, clearance=0.50)"
content = content.replace(old_call, new_call)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Increased Aspheric pocket radius to 13.25 and clearance to 0.50!")
