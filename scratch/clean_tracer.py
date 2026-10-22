with open('optics_app/engine/tracer.py', 'r') as f:
    lines = f.readlines()
with open('optics_app/engine/tracer.py', 'w') as f:
    for line in lines:
        if 'print(\"Tracing 19' in line or 'print(\"res from' in line or 'print(\"travel:' in line or 'print(\"Vignetted because' in line:
            continue
        f.write(line)
