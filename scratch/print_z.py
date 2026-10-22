import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.system import OpticalSystem

with open('spectrometer_yz_40mm_100mm_final.json', 'r') as f:
    config = json.load(f)

system = OpticalSystem()
system.load_from_json(config)
system.update_setup()

for i, s in enumerate(system.surfaces):
    print(f"{i}: {s.name} z={s.z_pos:.2f} thick={s.thickness:.2f}")
