with open('v2/engine/tracer.py', 'r') as f:
    content = f.read()

# First remove the wrongly appended one
if 'def export_step' in content:
    idx = content.find('    def export_step')
    if idx != -1:
        end_idx = content.find('        return export_system_to_step', idx)
        end_idx = content.find('\n', end_idx)
        content = content[:idx] + content[end_idx+1:]

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
        return export_system_to_step(self.stack, filename, include_air=include_air, merge_glass=merge_glass)
'''

# Insert it before trace_gaussian_beam
target = '    def trace_gaussian_beam('
if target in content:
    idx = content.find(target)
    content = content[:idx] + method_str + '\n' + content[idx:]

with open('v2/engine/tracer.py', 'w') as f:
    f.write(content)
