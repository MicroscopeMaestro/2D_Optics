import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.cad_export import export_step

system = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_slit.json')

try:
    export_step(system, 'spectrometer_slit.step')
    print("STEP export successful")
except Exception as e:
    import traceback
    traceback.print_exc()
