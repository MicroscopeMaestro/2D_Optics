import os

paths = [r'optics_app/ai/tools.py', r'v2/ai/tools.py']

for path in paths:
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Replace flip_optical_element
    target_flip = '''def flip_optical_element(
    element_idx: Union[int, str],
    json_path: Optional[str] = None,
    system: Optional[Any] = None
) -> str:
    if system is not None:
        set_active_optical_system(system)
    sys_obj, path = _resolve_system_and_path(json_path)
    if system is not None:
        sys_obj = system'''
    
    new_flip = '''def flip_optical_element(
    element_idx: Union[int, str],
    json_path: Optional[str] = None,
    system: Optional[Any] = None
) -> str:
    if system is not None:
        set_active_optical_system(system)
        sys_obj = system
        path = json_path
    else:
        sys_obj, path = _resolve_system_and_path(json_path)'''
        
    text = text.replace(target_flip, new_flip)
    
    # Do the same for displace_optical_element
    target_disp = '''def displace_optical_element(
    element_idx: Union[int, str],
    decenter_x: float = 0.0,
    decenter_y: float = 0.0,
    decenter_z: float = 0.0,
    json_path: Optional[str] = None,
    system: Optional[Any] = None
) -> str:
    if system is not None:
        set_active_optical_system(system)
    sys_obj, path = _resolve_system_and_path(json_path)
    if system is not None:
        sys_obj = system'''
        
    new_disp = '''def displace_optical_element(
    element_idx: Union[int, str],
    decenter_x: float = 0.0,
    decenter_y: float = 0.0,
    decenter_z: float = 0.0,
    json_path: Optional[str] = None,
    system: Optional[Any] = None
) -> str:
    if system is not None:
        set_active_optical_system(system)
        sys_obj = system
        path = json_path
    else:
        sys_obj, path = _resolve_system_and_path(json_path)'''
        
    text = text.replace(target_disp, new_disp)
    
    # And tilt
    target_tilt = '''def tilt_optical_element(
    element_idx: Union[int, str],
    angle_deg: float,
    axis: str = "y",
    json_path: Optional[str] = None,
    system: Optional[Any] = None
) -> str:
    if system is not None:
        set_active_optical_system(system)
    sys_obj, path = _resolve_system_and_path(json_path)
    if system is not None:
        sys_obj = system'''
        
    new_tilt = '''def tilt_optical_element(
    element_idx: Union[int, str],
    angle_deg: float,
    axis: str = "y",
    json_path: Optional[str] = None,
    system: Optional[Any] = None
) -> str:
    if system is not None:
        set_active_optical_system(system)
        sys_obj = system
        path = json_path
    else:
        sys_obj, path = _resolve_system_and_path(json_path)'''
        
    text = text.replace(target_tilt, new_tilt)
    
    # And offset
    target_off = '''def offset_optical_element(
    element_idx: Union[int, str],
    reference_surface_idx: Union[int, str],
    offset_bias_mm: float,
    axis: str = "z",
    json_path: Optional[str] = None,
    system: Optional[Any] = None
) -> str:
    if system is not None:
        set_active_optical_system(system)
    sys_obj, path = _resolve_system_and_path(json_path)
    if system is not None:
        sys_obj = system'''
        
    new_off = '''def offset_optical_element(
    element_idx: Union[int, str],
    reference_surface_idx: Union[int, str],
    offset_bias_mm: float,
    axis: str = "z",
    json_path: Optional[str] = None,
    system: Optional[Any] = None
) -> str:
    if system is not None:
        set_active_optical_system(system)
        sys_obj = system
        path = json_path
    else:
        sys_obj, path = _resolve_system_and_path(json_path)'''
        
    text = text.replace(target_off, new_off)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Patched {path}")
