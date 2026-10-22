import os, re

paths = [r'optics_app/ai/tools.py', r'v2/ai/tools.py']

for path in paths:
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # We can replace the pattern using regex
    pattern = r'(if system is not None:\s*set_active_optical_system\(system\)\s*)([a-zA-Z0-9_]+),\s*path\s*=\s*_resolve_system_and_path\(json_path\)\s*if system is not None:\s*\2 = system'
    
    replacement = r'''if system is not None:
        set_active_optical_system(system)
        \2 = system
        path = json_path
    else:
        \2, path = _resolve_system_and_path(json_path)'''
        
    text = re.sub(pattern, replacement, text)
    
    # Also handle the cases where they didn't have set_active_optical_system
    pattern2 = r'([a-zA-Z0-9_]+),\s*path\s*=\s*_resolve_system_and_path\(json_path\)\s*if system is not None:\s*\1 = system'
    replacement2 = r'''if system is not None:
        \1 = system
        path = json_path
    else:
        \1, path = _resolve_system_and_path(json_path)'''
    
    text = re.sub(pattern2, replacement2, text)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Regex patched {path}")
