import sys, os
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
sys_opt = OpticalSystem.from_json('spectrometer_nir_mirror.json')

print('Wavelengths:', sys_opt.config.wavelengths)

for s in sys_opt.surfaces:
    if s.surface_type.name == 'GRATING':
        print('Grating Density (lines/um):', s.lines_per_um)
    if s.surface_type.name == 'MIRROR':
        print('Mirror:', s.name, 'Radius:', s.radius)
