with open('optics_app/cad/exporter.py', 'r') as f:
    content = f.read()

start_sig = '        b3d = _get_build123d()'
if start_sig in content:
    idx = content.find(start_sig)
    
    new_logic = '''        from v2.engine.cad_export import export_system_to_step
        try:
            success = export_system_to_step(stack, file_path, include_air=True, merge_glass=True)
            if success:
                import os
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                return STEPExportResult(file_path, True, len(stack.surfaces), STEPExportEngine.BUILD123D, size)
            else:
                return STEPExportResult(file_path, False, 0, STEPExportEngine.BUILD123D, error_message="Export returned False")
        except Exception as e:
            return STEPExportResult(file_path, False, 0, STEPExportEngine.BUILD123D, error_message=str(e))
'''
    
    end_idx = content.find('    @classmethod\n    def export_assembly_step', idx)
    if end_idx != -1:
        content = content[:idx] + new_logic + '\n' + content[end_idx:]
        
        with open('optics_app/cad/exporter.py', 'w') as f:
            f.write(content)
