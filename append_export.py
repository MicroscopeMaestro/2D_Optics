with open('v2/engine/tracer.py', 'r') as f:
    content = f.read()

if 'def export_step' not in content:
    method_str = '''
    def export_step(self, filename: str, include_air: bool = False, merge_glass: bool = True) -> bool:
        """
        Exports the optical system geometry as a true B-Rep solid STEP file for mechanical CAD software.
        
        Args:
            filename: Output path for the .step file
            include_air: If True, exports the empty air gaps between optics as physical solid bodies.
            merge_glass: If True, seamlessly fuses cemented elements (e.g. Achromats) into single glass bodies.
        """
        from .cad_export import export_system_to_step
        return export_system_to_step(self, filename, include_air=include_air, merge_glass=merge_glass)
'''
    with open('v2/engine/tracer.py', 'a') as f:
        f.write(method_str)
