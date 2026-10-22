
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
for s in sys_opt.surfaces:
    if 'CB_After_DMD' in s.name:
        print('CB_After_DMD R:')
        print(s.global_R)
    if 'Relay_S1' in s.name:
        print('Relay_S1 R:')
        print(s.global_R)
        print('Local Z axis of Relay S1:', s.global_R[:, 2])

