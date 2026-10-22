
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_relay_redesign/spectrometer_nir_900_1700nm.json', 'r') as f: data = json.load(f)
for s in data['surfaces']:
    print(s['name'], s.get('grating_primary_wavelength_um'))

