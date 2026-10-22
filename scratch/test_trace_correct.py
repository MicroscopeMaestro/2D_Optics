import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('c:/Users/Juan/Documents/GitHub/2D_Optics'))

from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.source import SourceConfig, SourceType
from optics_app.analysis.spot import generate_spot_diagram_from_system

res = SurfaceStack.load_from_json('thorlabs_spectrometer.json')
sys_obj = OpticalSystem(res['stack'])
wls = res['wavelengths']

cfg = SourceConfig()
cfg.source_type = SourceType.FIBER
cfg.beam_radius = 0.0125
cfg.divergence_angle = 13.0

for wl in wls:
    sd = generate_spot_diagram_from_system(sys_obj, num_rays=50, wavelength_um=wl, target_surface_index=-1, source_config=cfg)
    print(f'WL: {wl}, Hits on detector: {len(sd.x)}')
