with open('optics_app/engine/tracer.py', 'r') as f:
    lines = f.readlines()
with open('optics_app/engine/tracer.py', 'w') as f:
    for line in lines:
        if line.strip() == 'if i == 19:':
            continue
        f.write(line)
