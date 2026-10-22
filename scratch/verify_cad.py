import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.system import OpticalSystem
from optics_app.engine.cad_exporter import CADExporter

with open('spectrometer_yz_40mm_100mm_slit.json', 'r') as f:
    config = json.load(f)

system = OpticalSystem()
system.load_from_json(config)
system.update_setup()

exporter = CADExporter(system)
exporter.export_step('test_slit.step')
print("STEP export successful")
