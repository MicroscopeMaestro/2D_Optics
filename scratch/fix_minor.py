path = 'optics_app/gui/measurement_dock.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('and DMDs for', 'and mechanisms for')
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

path = 'optics_app/gui/step_export_dialog.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('\"Mirror\", \"Grating\", \"DMD\"', '\"Mirror\", \"Grating\", \"Mechanism\", \"DMD\"')
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
